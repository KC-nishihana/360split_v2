"""
キーフレーム選択メインモジュール - 360Split用
全評価器を統合したキーフレーム選択パイプライン
2段階パイプライン + マルチスレッド処理 + 最適化NMS実装
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from .video_loader import VideoLoader, VideoMetadata
from .quality_evaluator import QualityEvaluator
from .geometric_evaluator import GeometricEvaluator
from .adaptive_selector import AdaptiveSelector
from .accelerator import get_accelerator
from processing.fisheye_rig import FisheyeRigProcessor
from .exceptions import (
    GeometricDegeneracyError,
    EstimationFailureError,
    InsufficientFeaturesError
)
from config import GRICConfig, Equirect360Config, NormalizationConfig

from utils.logger import get_logger
from utils.image_io import write_image
logger = get_logger(__name__)


@dataclass
class KeyframeInfo:
    """
    キーフレーム情報格納クラス

    Attributes:
    -----------
    frame_index : int
        フレーム番号（0ベース）
    timestamp : float
        時刻（秒）
    quality_scores : dict
        品質スコア（鮮明度、露光等）
    geometric_scores : dict
        幾何学的スコア（GRIC等）
    adaptive_scores : dict
        適応的スコア（SSIM、光学フロー等）
    combined_score : float
        統合スコア（0-1）
    thumbnail : Optional[np.ndarray]
        サムネイル画像（BGR形式）
    is_rescue_mode : bool
        レスキューモードで選択されたフレームか
    is_force_inserted : bool
        露出変化等で強制挿入されたフレームか
    """
    frame_index: int
    timestamp: float
    quality_scores: Dict[str, float] = field(default_factory=dict)
    geometric_scores: Dict[str, float] = field(default_factory=dict)
    adaptive_scores: Dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0
    thumbnail: Optional[np.ndarray] = None
    is_rescue_mode: bool = False
    is_force_inserted: bool = False


class KeyframeSelector:
    """
    キーフレーム選択パイプライン（最適化版）

    複数の品質・幾何学的・適応的評価器を統合して、
    360度ビデオから最適なキーフレームを自動選択する。

    2段階パイプライン：
    - Stage 1: 高速品質フィルタリング（全フレーム）
    - Stage 2: 精密評価（候補フレームのみ）
    """

    def __init__(self, config: Dict = None):
        """
        初期化

        Parameters:
        -----------
        config : dict, optional
            設定パラメータ辞書。以下のキーを含む：
            - WEIGHT_SHARPNESS: 鮮明度重み（推奨: 0.30）
            - WEIGHT_EXPOSURE: 露光重み（推奨: 0.15）
            - WEIGHT_GEOMETRIC: 幾何学的重み（推奨: 0.30）
            - WEIGHT_CONTENT: コンテンツ変化重み（推奨: 0.25）
            - LAPLACIAN_THRESHOLD: 最小ラプラシアン値
            - MIN_KEYFRAME_INTERVAL: 最小キーフレーム間隔
            - MAX_KEYFRAME_INTERVAL: 最大キーフレーム間隔
            - SOFTMAX_BETA: Softmax温度パラメータ
            - GRIC_RATIO_THRESHOLD: GRIC閾値
            - SSIM_CHANGE_THRESHOLD: SSIM変化閾値
            Noneの場合は内部デフォルト値を使用
        """
        # デフォルト設定
        self.config = {
            'WEIGHT_SHARPNESS': 0.30,
            'WEIGHT_EXPOSURE': 0.15,
            'WEIGHT_GEOMETRIC': 0.30,
            'WEIGHT_CONTENT': 0.25,
            'LAPLACIAN_THRESHOLD': 100.0,
            'MIN_KEYFRAME_INTERVAL': 5,
            'MAX_KEYFRAME_INTERVAL': 60,
            'SOFTMAX_BETA': 5.0,
            'GRIC_RATIO_THRESHOLD': 0.8,
            'SSIM_CHANGE_THRESHOLD': 0.85,
            'MOTION_BLUR_THRESHOLD': 0.3,
            'EXPOSURE_THRESHOLD': 0.35,
            'PAIR_MOTION_AGGREGATION': 'max',
            'MIN_FEATURE_MATCHES': 30,
            'THUMBNAIL_SIZE': (192, 108),
            'SAMPLE_INTERVAL': 1,
            'STAGE1_BATCH_SIZE': 32,
            'ENABLE_RIG_STITCHING': True,
            'EQUIRECT_WIDTH': 4096,
            'EQUIRECT_HEIGHT': 2048,
            'RIG_FEATURE_METHOD': 'orb',
            # レスキューモード設定
            'ENABLE_RESCUE_MODE': False,
            'RESCUE_FEATURE_THRESHOLD': 15,  # この値以下で特徴点不足と判定
            'RESCUE_LAPLACIAN_FACTOR': 0.5,  # レスキュー時のLaplacian閾値倍率
            'RESCUE_WINDOW_SIZE': 10,  # レスキューモード判定のウィンドウサイズ
            # 混合環境での強制挿入設定
            'FORCE_KEYFRAME_ON_EXPOSURE_CHANGE': False,
            'EXPOSURE_CHANGE_THRESHOLD': 0.3,  # 露出変化の検知閾値
            'ADAPTIVE_THRESHOLDING': False,  # 動的閾値調整
        }

        # 外部設定でオーバーライド
        if config:
            self.config.update(config)
            # lower_snake_case 設定との互換
            alias_map = {
                'laplacian_threshold': 'LAPLACIAN_THRESHOLD',
                'motion_blur_threshold': 'MOTION_BLUR_THRESHOLD',
                'exposure_threshold': 'EXPOSURE_THRESHOLD',
                'min_keyframe_interval': 'MIN_KEYFRAME_INTERVAL',
                'max_keyframe_interval': 'MAX_KEYFRAME_INTERVAL',
                'softmax_beta': 'SOFTMAX_BETA',
                'ssim_change_threshold': 'SSIM_CHANGE_THRESHOLD',
                'ssim_threshold': 'SSIM_CHANGE_THRESHOLD',
                'weight_sharpness': 'WEIGHT_SHARPNESS',
                'weight_exposure': 'WEIGHT_EXPOSURE',
                'weight_geometric': 'WEIGHT_GEOMETRIC',
                'weight_content': 'WEIGHT_CONTENT',
                'pair_motion_aggregation': 'PAIR_MOTION_AGGREGATION',
                'enable_rig_stitching': 'ENABLE_RIG_STITCHING',
                'equirect_width': 'EQUIRECT_WIDTH',
                'equirect_height': 'EQUIRECT_HEIGHT',
                'rig_feature_method': 'RIG_FEATURE_METHOD',
            }
            for src, dst in alias_map.items():
                if src in config:
                    self.config[dst] = config[src]

        # 正規化設定
        self.normalization = NormalizationConfig(
            SHARPNESS_NORM_FACTOR=self.config.get('SHARPNESS_NORM_FACTOR', 1000.0),
            OPTICAL_FLOW_NORM_FACTOR=self.config.get('OPTICAL_FLOW_NORM_FACTOR', 50.0),
            FEATURE_MATCH_NORM_FACTOR=self.config.get('FEATURE_MATCH_NORM_FACTOR', 200.0),
        )

        # GRIC設定を構築
        gric_config = GRICConfig(
            lambda1=self.config.get('GRIC_LAMBDA1', 2.0),
            lambda2=self.config.get('GRIC_LAMBDA2', 4.0),
            sigma=self.config.get('GRIC_SIGMA', 1.0),
            ransac_threshold=self.config.get('RANSAC_THRESHOLD', 3.0),
            min_inlier_ratio=self.config.get('MIN_INLIER_RATIO', 0.3),
            degeneracy_threshold=self.config.get(
                'GRIC_DEGENERACY_THRESHOLD',
                self.config.get('GRIC_RATIO_THRESHOLD', 0.85)
            ),
            min_matches=self.config.get('MIN_FEATURE_MATCHES', 30),
        )

        # 360°設定を構築
        equirect_config = Equirect360Config(
            mask_polar_ratio=self.config.get('MASK_POLAR_RATIO', 0.10),
            enable_polar_mask=self.config.get('ENABLE_POLAR_MASK', True),
        )

        # 評価器を初期化
        self.quality_evaluator = QualityEvaluator()
        self.geometric_evaluator = GeometricEvaluator(
            gric_config=gric_config,
            equirect_config=equirect_config
        )
        self.adaptive_selector = AdaptiveSelector()
        self.rig_processor = FisheyeRigProcessor(equirect_config=equirect_config)

        # アクセレータから情報を取得
        self.accelerator = get_accelerator()
        logger.info(
            f"アクセレータ: device={self.accelerator.device_name}, "
            f"thread_count={self.accelerator.num_threads}"
        )

        # レスキューモード関連の状態変数
        self.is_rescue_mode = False
        self.feature_count_history = deque(maxlen=self.config['RESCUE_WINDOW_SIZE'])
        self.previous_brightness = None
        self.rescue_mode_keyframes = []  # レスキューモードで選択されたキーフレーム

    def _open_independent_capture(self, video_path) -> cv2.VideoCapture:
        """
        分析用に独立したcv2.VideoCaptureを開く

        VideoLoaderのキャプチャとは別のインスタンスを使用して、
        FFmpegのスレッド安全性問題を回避する。

        Parameters:
        -----------
        video_path : Path
            ビデオファイルパス

        Returns:
        --------
        cv2.VideoCapture
            独立したキャプチャオブジェクト

        Raises:
        -------
        RuntimeError
            ビデオファイルが開けない場合
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"分析用ビデオキャプチャを開けません: {video_path}")
        return cap

    def _check_rescue_mode(self, feature_count: int) -> bool:
        """
        レスキューモード判定

        特徴点マッチング数の履歴から、レスキューモードに入るべきか判定します。

        Parameters:
        -----------
        feature_count : int
            現在のフレームの特徴点マッチング数

        Returns:
        --------
        bool
            レスキューモードに入るべきか
        """
        if not self.config.get('ENABLE_RESCUE_MODE', False):
            return False

        # 履歴に追加
        self.feature_count_history.append(feature_count)

        # ウィンドウが満たされるまではレスキューモードに入らない
        if len(self.feature_count_history) < self.config['RESCUE_WINDOW_SIZE']:
            return False

        # ウィンドウ内の平均特徴点数が閾値以下かチェック
        avg_features = np.mean(list(self.feature_count_history))
        threshold = self.config['RESCUE_FEATURE_THRESHOLD']

        should_rescue = avg_features < threshold

        if should_rescue and not self.is_rescue_mode:
            logger.warning(
                f"レスキューモード発動: 平均特徴点数 {avg_features:.1f} < {threshold}"
            )
        elif not should_rescue and self.is_rescue_mode:
            logger.info(
                f"レスキューモード解除: 平均特徴点数 {avg_features:.1f} >= {threshold}"
            )

        return should_rescue

    def _detect_exposure_change(self, frame: np.ndarray) -> bool:
        """
        露出の急激な変化を検知

        フレームの平均輝度を計算し、前フレームとの変化が閾値を超えるか判定します。
        ドアを抜けた瞬間などの急激な明暗変化を検知します。

        Parameters:
        -----------
        frame : np.ndarray
            現在のフレーム（BGR形式）

        Returns:
        --------
        bool
            急激な露出変化があったか
        """
        if not self.config.get('FORCE_KEYFRAME_ON_EXPOSURE_CHANGE', False):
            return False

        # グレースケール化して平均輝度を計算
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray) / 255.0  # 0-1に正規化

        if self.previous_brightness is None:
            self.previous_brightness = current_brightness
            return False

        # 変化量を計算
        brightness_change = abs(current_brightness - self.previous_brightness)
        threshold = self.config['EXPOSURE_CHANGE_THRESHOLD']

        is_significant_change = brightness_change > threshold

        if is_significant_change:
            logger.info(
                f"急激な露出変化検知: {self.previous_brightness:.3f} → "
                f"{current_brightness:.3f} (変化量: {brightness_change:.3f})"
            )

        self.previous_brightness = current_brightness
        return is_significant_change

    def _get_adjusted_laplacian_threshold(self) -> float:
        """
        レスキューモード時のLaplacian閾値を取得

        Returns:
        --------
        float
            調整されたLaplacian閾値
        """
        base_threshold = self.config['LAPLACIAN_THRESHOLD']

        if self.is_rescue_mode:
            factor = self.config['RESCUE_LAPLACIAN_FACTOR']
            adjusted = base_threshold * factor
            return adjusted

        return base_threshold

    def select_keyframes(self, video_loader,
                        progress_callback: Optional[Callable[[int, int], None]] = None
                        ) -> List[KeyframeInfo]:
        """
        ビデオからキーフレームを自動選択（2段階パイプライン）

        アルゴリズム：
        Stage 1: 高速フィルタリング（全フレーム）
          - 品質スコアのみ計算（~5ms/フレーム）
          - ステレオの場合：L/R両方が基準を満たすかチェック（AND条件）
          - 閾値以下のフレームを即座に除外（60-70%フィルタリング）
          - マルチスレッド処理で高速化

        Stage 2: 精密評価（候補フレームのみ）
          - 幾何学的・適応的評価（~50ms/フレーム）
          - ステレオの場合：Left画像のみで移動判定（コスト削減）
          - Stage 1通過フレームのみ処理
          - NMS適用

        注意：FFmpegのスレッド安全性問題を回避するため、
        分析処理ではVideoLoaderとは独立したcv2.VideoCaptureを使用する。

        Parameters:
        -----------
        video_loader : VideoLoader or DualVideoLoader
            読み込み済みのローダーインスタンス
        progress_callback : callable, optional
            進捗コールバック関数 (current_frame, total_frames)

        Returns:
        --------
        list of KeyframeInfo
            選択されたキーフレーム情報リスト
        """
        metadata = video_loader.get_metadata()
        if metadata is None:
            raise RuntimeError("ビデオが読み込まれていません")

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired
        is_stereo = hasattr(video_loader, 'is_stereo') and video_loader.is_stereo
        rig_type = getattr(video_loader, 'rig_type', getattr(metadata, 'rig_type', 'monocular'))

        if is_paired:
            logger.info(f"ペアレンズモードでキーフレーム選択を実行: rig_type={rig_type}")
        elif is_stereo:
            logger.info("ステレオモード（OSV）でキーフレーム選択を実行")

        total_frames = metadata.frame_count
        logger.info(f"キーフレーム選択開始: {total_frames}フレーム")

        # ===== Stage 1: 高速フィルタリング =====
        logger.info("Stage 1: 高速品質フィルタリング開始")
        stage1_candidates = self._stage1_fast_filter(
            video_loader, metadata, progress_callback
        )
        logger.info(
            f"Stage 1完了: {len(stage1_candidates)}/{total_frames} "
            f"({100*len(stage1_candidates)/max(total_frames, 1):.1f}%)"
        )

        if not stage1_candidates:
            logger.warning("Stage 1でフレームが残りませんでした")
            return []

        # ===== Stage 2: 精密評価 =====
        logger.info(f"Stage 2: 精密評価開始（{len(stage1_candidates)}フレーム）")
        stage2_candidates = self._stage2_precise_evaluation(
            video_loader, metadata, stage1_candidates, progress_callback
        )
        logger.info(f"Stage 2キーフレーム候補: {len(stage2_candidates)}個")

        # 非最大値抑制を適用
        keyframes = self._apply_nms(stage2_candidates)

        # 最大間隔制約を適用
        keyframes = self._enforce_max_interval(keyframes, metadata.fps)

        logger.info(f"最終キーフレーム数: {len(keyframes)}個")

        return keyframes

    def _stage1_fast_filter(self, video_loader, metadata: VideoMetadata,
                           progress_callback: Optional[Callable[[int, int], None]]) -> List[Dict]:
        """
        Stage 1: 高速品質フィルタリング（全フレーム）

        品質スコアのみで閾値以下のフレームを除外。
        独立したcv2.VideoCaptureでシングルスレッド読み込み、
        品質計算のみマルチスレッドで並列化。

        FFmpegのスレッド安全性問題を回避するため、
        VideoLoaderとは独立したキャプチャを使用する。

        Parameters:
        -----------
        video_path : Path
            ビデオファイルパス
        metadata : VideoMetadata
            ビデオメタデータ
        progress_callback : callable, optional
            進捗コールバック

        Returns:
        --------
        list of dict
            通過フレーム情報リスト: [{'frame_idx': int, 'quality_scores': dict}, ...]
        """
        total_frames = metadata.frame_count
        sample_interval = self.config['SAMPLE_INTERVAL']
        batch_size = self.config['STAGE1_BATCH_SIZE']

        # フレームインデックスを収集
        frame_indices = list(range(0, total_frames, sample_interval))

        candidates = []

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired

        # ペアレンズ: レンズ毎品質評価 + AND条件
        if is_paired:
            for idx, frame_idx in enumerate(frame_indices):
                frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                if frame_a is None or frame_b is None:
                    continue

                quality_scores = self._compute_quality_score_pair(frame_a, frame_b)

                if quality_scores['passes_threshold']:
                    candidates.append({
                        'frame_idx': frame_idx,
                        'quality_scores': quality_scores
                    })

                if progress_callback:
                    progress_callback(idx + 1, len(frame_indices))
            return candidates

        # 単眼: 独立キャプチャ + 並列品質計算
        video_path = getattr(video_loader, "_video_path", None)
        if video_path is None:
            raise RuntimeError("単眼モードのビデオパスが取得できません")

        cap = self._open_independent_capture(video_path)
        try:
            num_batches = (len(frame_indices) + batch_size - 1) // batch_size
            last_read_idx = -1

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(frame_indices))
                batch_indices = frame_indices[start_idx:end_idx]

                frames = []
                valid_indices = []
                for frame_idx in batch_indices:
                    if frame_idx != last_read_idx + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                    ret, frame = cap.read()
                    last_read_idx = frame_idx
                    if ret and frame is not None:
                        frames.append(frame)
                        valid_indices.append(frame_idx)

                if not frames:
                    continue

                num_workers = min(self.accelerator.num_threads, len(frames))
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    quality_list = list(executor.map(self._compute_quality_score, frames))

                for frame_idx, quality_scores in zip(valid_indices, quality_list):
                    if quality_scores['sharpness'] >= self.config['LAPLACIAN_THRESHOLD'] and \
                       quality_scores['motion_blur'] <= self.config['MOTION_BLUR_THRESHOLD'] and \
                       quality_scores['exposure'] >= self.config['EXPOSURE_THRESHOLD']:
                        candidates.append({'frame_idx': frame_idx, 'quality_scores': quality_scores})

                if progress_callback:
                    progress_callback(end_idx, len(frame_indices))

        finally:
            cap.release()
            logger.debug("Stage 1: 独立キャプチャを解放")

        return candidates

    def _stage2_precise_evaluation(self, video_loader, metadata: VideoMetadata,
                                   stage1_candidates: List[Dict],
                                   progress_callback: Optional[Callable[[int, int], None]]
                                   ) -> List[KeyframeInfo]:
        """
        Stage 2: 精密評価（候補フレームのみ）

        幾何学的・適応的評価を適用。最小間隔制約も適用。
        独立したcv2.VideoCaptureを使用してFFmpegスレッド安全性を確保。

        Parameters:
        -----------
        video_path : Path
            ビデオファイルパス
        metadata : VideoMetadata
            ビデオメタデータ
        stage1_candidates : list
            Stage 1通過フレーム
        progress_callback : callable, optional
            進捗コールバック

        Returns:
        --------
        list of KeyframeInfo
            精密評価後のキーフレーム候補
        """
        candidates = []
        last_keyframe_idx = -self.config['MIN_KEYFRAME_INTERVAL']
        last_keyframe = None
        last_pair = None

        # フレームウィンドウ（カメラ加速度計算用）
        frame_window = deque(maxlen=5)

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired
        cap = None
        if not is_paired:
            video_path = getattr(video_loader, "_video_path", None)
            if video_path is None:
                raise RuntimeError("単眼モードのビデオパスが取得できません")
            cap = self._open_independent_capture(video_path)

        try:
            for idx, candidate_info in enumerate(stage1_candidates):
                frame_idx = candidate_info['frame_idx']
                quality_scores = candidate_info['quality_scores']

                if progress_callback:
                    progress_callback(idx, len(stage1_candidates))

                if frame_idx - last_keyframe_idx < self.config['MIN_KEYFRAME_INTERVAL']:
                    continue

                current_frame = None
                current_pair = None
                if is_paired:
                    frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                    if frame_a is None or frame_b is None:
                        continue
                    current_pair = (frame_a, frame_b)
                    current_frame = frame_a
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, current_frame = cap.read()
                    if not ret or current_frame is None:
                        continue

                frame_window.append(current_frame)
                exposure_changed = self._detect_exposure_change(current_frame)

                geometric_scores = {}
                adaptive_scores = {}
                force_insert = False

                if last_keyframe is not None:
                    geometric_failed = False

                    geom_ref = last_keyframe
                    geom_cur = current_frame
                    if is_paired and self.config.get('ENABLE_RIG_STITCHING', True):
                        calibration = getattr(metadata, 'rig_calibration', None)
                        out_w = int(self.config.get('EQUIRECT_WIDTH', 4096))
                        out_h = int(self.config.get('EQUIRECT_HEIGHT', 2048))
                        last_pan, last_seam = self.rig_processor.stitch_to_equirect(
                            last_pair[0], last_pair[1], calibration, (out_w, out_h)
                        )
                        cur_pan, cur_seam = self.rig_processor.stitch_to_equirect(
                            current_pair[0], current_pair[1], calibration, (out_w, out_h)
                        )
                        method = self.config.get('RIG_FEATURE_METHOD', 'orb')
                        f_prev = self.rig_processor.extract_360_features(last_pan, last_seam, method=method)
                        f_cur = self.rig_processor.extract_360_features(cur_pan, cur_seam, method=method)
                        quality_scores['seam_features_prev'] = float(f_prev.seam_keypoint_count)
                        quality_scores['seam_features_cur'] = float(f_cur.seam_keypoint_count)
                        geom_ref, geom_cur = last_pan, cur_pan

                    try:
                        geometric_scores = self.geometric_evaluator.evaluate(
                            geom_ref, geom_cur,
                            frame1_idx=last_keyframe_idx,
                            frame2_idx=frame_idx
                        )
                    except GeometricDegeneracyError as e:
                        logger.debug(f"フレーム {frame_idx}: 幾何学的縮退 - {e}")
                        geometric_scores = {
                            'gric': 0.1,
                            'feature_distribution_1': 0.0,
                            'feature_distribution_2': 0.0,
                            'feature_match_count': 0,
                            'ray_dispersion': 0.0
                        }
                        geometric_failed = True
                    except (EstimationFailureError, InsufficientFeaturesError) as e:
                        logger.debug(f"フレーム {frame_idx}: 幾何学的評価失敗 - {e}")
                        self.is_rescue_mode = self._check_rescue_mode(0)
                        if self.is_rescue_mode and frame_idx - last_keyframe_idx >= self.config['MIN_KEYFRAME_INTERVAL']:
                            geometric_scores = {
                                'gric': 0.05,
                                'feature_distribution_1': 0.0,
                                'feature_distribution_2': 0.0,
                                'feature_match_count': 0,
                                'ray_dispersion': 0.0
                            }
                            geometric_failed = True
                            force_insert = True
                        else:
                            continue

                    if not geometric_failed and 'feature_match_count' in geometric_scores:
                        self.is_rescue_mode = self._check_rescue_mode(geometric_scores['feature_match_count'])

                    adaptive_scores = self.adaptive_selector.evaluate(
                        last_keyframe, current_frame, frames_window=list(frame_window)
                    )

                    if is_paired and last_pair is not None and current_pair is not None:
                        flow_a = self.adaptive_selector.compute_optical_flow_magnitude(last_pair[0], current_pair[0])
                        flow_b = self.adaptive_selector.compute_optical_flow_magnitude(last_pair[1], current_pair[1])
                        if self.config.get('PAIR_MOTION_AGGREGATION', 'max') == 'max':
                            adaptive_scores['optical_flow'] = max(flow_a, flow_b)
                        else:
                            adaptive_scores['optical_flow'] = (flow_a + flow_b) * 0.5
                        adaptive_scores['optical_flow_lens_a'] = flow_a
                        adaptive_scores['optical_flow_lens_b'] = flow_b

                    ssim = adaptive_scores.get('ssim', 1.0)
                    if ssim > self.config['SSIM_CHANGE_THRESHOLD'] and not force_insert and not exposure_changed:
                        continue

                    if exposure_changed and frame_idx - last_keyframe_idx >= self.config['MIN_KEYFRAME_INTERVAL']:
                        force_insert = True
                else:
                    geometric_failed = False

                combined_score = self._compute_combined_score(
                    quality_scores, geometric_scores, adaptive_scores
                )

                candidate = KeyframeInfo(
                    frame_index=frame_idx,
                    timestamp=frame_idx / metadata.fps,
                    quality_scores=quality_scores,
                    geometric_scores=geometric_scores,
                    adaptive_scores=adaptive_scores,
                    combined_score=combined_score,
                    thumbnail=None,
                    is_rescue_mode=self.is_rescue_mode,
                    is_force_inserted=force_insert or exposure_changed
                )
                candidates.append(candidate)

                if self.is_rescue_mode:
                    self.rescue_mode_keyframes.append(frame_idx)

                last_keyframe_idx = frame_idx
                last_keyframe = current_frame
                last_pair = current_pair

        finally:
            if cap is not None:
                cap.release()
                logger.debug("Stage 2: 独立キャプチャを解放")

        return candidates

    def _compute_quality_score(self, frame: np.ndarray) -> Dict[str, float]:
        """
        品質スコアを計算（Stage 1用）

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム

        Returns:
        --------
        dict
            品質スコア
        """
        return self.quality_evaluator.evaluate(
            frame,
            beta=self.config['SOFTMAX_BETA']
        )

    def _compute_quality_score_pair(self, frame_a: np.ndarray,
                                    frame_b: np.ndarray) -> Dict[str, float]:
        """
        ペアレンズフレームの品質スコアを計算（Conservative AND条件）。

        どちらか片方でも閾値未達なら除外する。
        """
        score_a = self.quality_evaluator.evaluate(frame_a, beta=self.config['SOFTMAX_BETA'])
        score_b = self.quality_evaluator.evaluate(frame_b, beta=self.config['SOFTMAX_BETA'])

        sharpness_ok = (
            score_a.get('sharpness', 0.0) >= self.config['LAPLACIAN_THRESHOLD'] and
            score_b.get('sharpness', 0.0) >= self.config['LAPLACIAN_THRESHOLD']
        )
        motion_ok = (
            score_a.get('motion_blur', 1.0) <= self.config['MOTION_BLUR_THRESHOLD'] and
            score_b.get('motion_blur', 1.0) <= self.config['MOTION_BLUR_THRESHOLD']
        )
        exposure_ok = (
            score_a.get('exposure', 0.0) >= self.config['EXPOSURE_THRESHOLD'] and
            score_b.get('exposure', 0.0) >= self.config['EXPOSURE_THRESHOLD']
        )

        combined_score = {
            'sharpness': min(score_a.get('sharpness', 0.0), score_b.get('sharpness', 0.0)),
            'exposure': min(score_a.get('exposure', 0.0), score_b.get('exposure', 0.0)),
            'motion_blur': max(score_a.get('motion_blur', 1.0), score_b.get('motion_blur', 1.0)),
            'softmax_depth': min(score_a.get('softmax_depth', 0.0), score_b.get('softmax_depth', 0.0)),
            'lens_a': score_a,
            'lens_b': score_b,
            'passes_threshold': bool(sharpness_ok and motion_ok and exposure_ok),
        }
        return combined_score

    def _compute_quality_score_stereo(self, frame_l: np.ndarray,
                                      frame_r: np.ndarray) -> Dict[str, float]:
        """後方互換ラッパー。"""
        return self._compute_quality_score_pair(frame_l, frame_r)

    def _compute_combined_score(self, quality_scores: Dict[str, float],
                               geometric_scores: Dict[str, float],
                               adaptive_scores: Dict[str, float]) -> float:
        """
        複数のスコアを統合して総合スコアを計算

        正規化にはNormalizationConfigの係数を使用。

        Parameters:
        -----------
        quality_scores : dict
            品質スコア
        geometric_scores : dict
            幾何学的スコア
        adaptive_scores : dict
            適応的スコア

        Returns:
        --------
        float
            総合スコア（0-1）
        """
        norm = self.normalization

        # 品質スコアを正規化
        sharpness = min(
            quality_scores.get('sharpness', 0) / norm.SHARPNESS_NORM_FACTOR,
            1.0
        )
        exposure = quality_scores.get('exposure', 0.5)

        # 幾何学的スコアを正規化
        if geometric_scores:
            # v2: gricスコアは既に0-1（高いほど視差あり）
            gric = geometric_scores.get('gric', 0.5)
            dist1 = geometric_scores.get('feature_distribution_1', 0.5)
            dist2 = geometric_scores.get('feature_distribution_2', 0.5)
            ray_disp = geometric_scores.get('ray_dispersion', 0.5)

            geometric_score = (gric + dist1 + dist2 + ray_disp) / 4.0
        else:
            geometric_score = 0.5

        # 適応的スコアを正規化
        if adaptive_scores:
            ssim_change = 1.0 - adaptive_scores.get('ssim', 1.0)
            optical_flow = min(
                adaptive_scores.get('optical_flow', 0) / norm.OPTICAL_FLOW_NORM_FACTOR,
                1.0
            )
            content_score = (ssim_change + optical_flow) / 2.0
        else:
            content_score = 0.5

        # 重み付け統合
        combined = (
            self.config['WEIGHT_SHARPNESS'] * sharpness +
            self.config['WEIGHT_EXPOSURE'] * exposure +
            self.config['WEIGHT_GEOMETRIC'] * geometric_score +
            self.config['WEIGHT_CONTENT'] * content_score
        )

        return float(np.clip(combined, 0.0, 1.0))

    def _apply_nms(self, candidates: List[KeyframeInfo],
                  time_window: float = 1.0) -> List[KeyframeInfo]:
        """
        非最大値抑制（NMS）を適用（最適化版）

        スコアが高い候補をスコア降順で処理し、
        時間ウィンドウ内の低スコア候補を除外。
        O(N*M) ネストループを避ける。

        Parameters:
        -----------
        candidates : list of KeyframeInfo
            候補キーフレームリスト
        time_window : float
            時間ウィンドウ（秒）

        Returns:
        --------
        list of KeyframeInfo
            NMS適用後のキーフレームリスト
        """
        if len(candidates) == 0:
            return []

        # スコアでソート（降順）
        sorted_candidates = sorted(candidates, key=lambda x: x.combined_score, reverse=True)

        selected = []

        for candidate in sorted_candidates:
            # 既選択キーフレームとの時間距離をチェック
            is_within_window = False

            for selected_kf in selected:
                time_diff = abs(candidate.timestamp - selected_kf.timestamp)
                if time_diff < time_window:
                    is_within_window = True
                    break

            if not is_within_window:
                selected.append(candidate)

        # フレーム番号でソート
        selected.sort(key=lambda x: x.frame_index)

        return selected

    def _enforce_max_interval(self, keyframes: List[KeyframeInfo],
                             fps: float) -> List[KeyframeInfo]:
        """
        最大キーフレーム間隔制約を適用

        キーフレーム間の時間が最大間隔を超える場合は
        その間から品質スコアが最高のフレームを追加

        Parameters:
        -----------
        keyframes : list of KeyframeInfo
            キーフレームリスト
        fps : float
            フレームレート

        Returns:
        --------
        list of KeyframeInfo
            制約適用後のキーフレームリスト
        """
        if len(keyframes) < 2:
            return keyframes

        max_interval = self.config['MAX_KEYFRAME_INTERVAL'] / fps

        enforced_keyframes = [keyframes[0]]

        for i in range(1, len(keyframes)):
            current_kf = keyframes[i]
            last_kf = enforced_keyframes[-1]

            time_diff = current_kf.timestamp - last_kf.timestamp

            if time_diff <= max_interval:
                enforced_keyframes.append(current_kf)
            else:
                num_missing = int(np.ceil(time_diff / max_interval)) - 1

                for j in range(1, num_missing + 1):
                    enforced_keyframes.append(current_kf)

                enforced_keyframes.append(current_kf)

        return enforced_keyframes

    def export_keyframes(self, keyframes: List[KeyframeInfo],
                        video_loader: VideoLoader,
                        output_dir: str,
                        format: str = 'png') -> Dict[int, Path]:
        """
        キーフレーム画像をファイルにエクスポート

        Parameters:
        -----------
        keyframes : list of KeyframeInfo
            エクスポートするキーフレームリスト
        video_loader : VideoLoader
            ビデオローダー
        output_dir : str
            出力ディレクトリ
        format : str
            出力形式（'png' または 'jpg'）

        Returns:
        --------
        dict
            {フレーム番号: ファイルパス} の辞書
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported = {}

        for kf in keyframes:
            frame = video_loader.get_frame(kf.frame_index)
            if frame is None:
                logger.warning(f"フレーム {kf.frame_index} の読み込み失敗")
                continue

            # サムネイルを遅延生成
            if kf.thumbnail is None:
                kf.thumbnail = self._create_thumbnail(frame)

            # ファイル名を生成
            timestamp_str = f"{kf.timestamp:.2f}".replace('.', '-')
            filename = f"keyframe_{kf.frame_index:06d}_{timestamp_str}.{format}"
            filepath = output_path / filename

            # 保存
            if format.lower() in ['jpg', 'jpeg']:
                saved = write_image(
                    filepath,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            else:  # png
                saved = write_image(filepath, frame)

            if not saved:
                logger.warning(f"キーフレーム保存失敗: {filepath}")
                continue

            exported[kf.frame_index] = filepath
            logger.info(f"キーフレーム保存: {filepath}")

        logger.info(f"合計 {len(exported)}個のキーフレームをエクスポート")

        return exported

    def _create_thumbnail(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームからサムネイルを生成

        Parameters:
        -----------
        frame : np.ndarray
            元フレーム

        Returns:
        --------
        np.ndarray
            サムネイル画像
        """
        thumbnail_size = self.config['THUMBNAIL_SIZE']
        thumbnail = cv2.resize(frame, thumbnail_size)
        return thumbnail
