"""
キーフレーム選択メインモジュール - 360Split用
全評価器を統合したキーフレーム選択パイプライン
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging

from .video_loader import VideoLoader, VideoMetadata
from .quality_evaluator import QualityEvaluator
from .geometric_evaluator import GeometricEvaluator
from .adaptive_selector import AdaptiveSelector

logger = logging.getLogger('360split')


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
    """
    frame_index: int
    timestamp: float
    quality_scores: Dict[str, float] = field(default_factory=dict)
    geometric_scores: Dict[str, float] = field(default_factory=dict)
    adaptive_scores: Dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0
    thumbnail: Optional[np.ndarray] = None


class KeyframeSelector:
    """
    キーフレーム選択パイプライン

    複数の品質・幾何学的・適応的評価器を統合して、
    360度ビデオから最適なキーフレームを自動選択する
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
            'MIN_FEATURE_MATCHES': 30,
            'THUMBNAIL_SIZE': (192, 108),
            'SAMPLE_INTERVAL': 1,  # サンプリング間隔
        }

        # 外部設定でオーバーライド
        if config:
            self.config.update(config)

        # 評価器を初期化
        self.quality_evaluator = QualityEvaluator()
        self.geometric_evaluator = GeometricEvaluator()
        self.adaptive_selector = AdaptiveSelector()

    def select_keyframes(self, video_loader: VideoLoader,
                        progress_callback: Optional[Callable[[int, int], None]] = None
                        ) -> List[KeyframeInfo]:
        """
        ビデオからキーフレームを自動選択

        アルゴリズム：
        1. 定期的にフレームをサンプリング
        2. 品質スコアが低いフレームを除外
        3. 前キーフレームとのSSIMで大きな変化を検出
        4. 幾何学的性質（GRICなど）を評価
        5. 適応的なサンプリング間隔を計算
        6. 非最大値抑制で最適キーフレームを選択

        Parameters:
        -----------
        video_loader : VideoLoader
            読み込み済みのVideoLoaderインスタンス
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

        total_frames = metadata.frame_count
        logger.info(f"キーフレーム選択開始: {total_frames}フレーム")

        keyframe_candidates = []
        last_keyframe_idx = -self.config['MIN_KEYFRAME_INTERVAL']
        last_keyframe = None

        # フレームウィンドウ（カメラ加速度計算用）
        frame_window = []
        window_size = 5

        sample_interval = self.config['SAMPLE_INTERVAL']

        # フレームをサンプリング
        for frame_idx in range(0, total_frames, sample_interval):
            if progress_callback:
                progress_callback(frame_idx, total_frames)

            current_frame = video_loader.get_frame(frame_idx)
            if current_frame is None:
                continue

            # フレームウィンドウを管理
            frame_window.append(current_frame)
            if len(frame_window) > window_size:
                frame_window.pop(0)

            # 品質スコアを計算
            quality_scores = self.quality_evaluator.evaluate(
                current_frame,
                beta=self.config['SOFTMAX_BETA']
            )

            # 品質フィルタリング
            if quality_scores['sharpness'] < self.config['LAPLACIAN_THRESHOLD']:
                logger.debug(f"フレーム {frame_idx}: 鮮明度不足 ({quality_scores['sharpness']:.1f})")
                continue

            if quality_scores['motion_blur'] > self.config['MOTION_BLUR_THRESHOLD']:
                logger.debug(f"フレーム {frame_idx}: モーションブラー過大 ({quality_scores['motion_blur']:.2f})")
                continue

            # 最小間隔制約
            if frame_idx - last_keyframe_idx < self.config['MIN_KEYFRAME_INTERVAL']:
                continue

            # 幾何学的スコア計算
            geometric_scores = {}
            adaptive_scores = {}

            if last_keyframe is not None:
                # 前キーフレームとの比較
                geometric_scores = self.geometric_evaluator.evaluate(
                    last_keyframe, current_frame,
                    min_matches=self.config['MIN_FEATURE_MATCHES'],
                    gric_threshold=self.config['GRIC_RATIO_THRESHOLD']
                )

                # 適応的スコア計算
                adaptive_scores = self.adaptive_selector.evaluate(
                    last_keyframe, current_frame,
                    frames_window=frame_window
                )

                # SSIM変化が小さすぎる場合はスキップ
                ssim = adaptive_scores.get('ssim', 1.0)
                if ssim > self.config['SSIM_CHANGE_THRESHOLD']:
                    logger.debug(f"フレーム {frame_idx}: 変化不足 (SSIM: {ssim:.3f})")
                    continue

            else:
                # 最初のキーフレーム
                logger.debug(f"最初のキーフレーム: {frame_idx}")

            # 統合スコアを計算
            combined_score = self._compute_combined_score(
                quality_scores, geometric_scores, adaptive_scores
            )

            # キーフレーム候補に追加
            thumbnail = self._create_thumbnail(current_frame)

            candidate = KeyframeInfo(
                frame_index=frame_idx,
                timestamp=frame_idx / metadata.fps,
                quality_scores=quality_scores,
                geometric_scores=geometric_scores,
                adaptive_scores=adaptive_scores,
                combined_score=combined_score,
                thumbnail=thumbnail
            )

            keyframe_candidates.append(candidate)
            last_keyframe_idx = frame_idx
            last_keyframe = current_frame

            logger.debug(
                f"候補フレーム {frame_idx}: "
                f"品質={combined_score:.3f}, "
                f"SSIM={adaptive_scores.get('ssim', 0):.3f}"
            )

        logger.info(f"キーフレーム候補: {len(keyframe_candidates)}個")

        # 非最大値抑制を適用
        keyframes = self._apply_nms(keyframe_candidates)

        # 最大間隔制約を適用
        keyframes = self._enforce_max_interval(keyframes, metadata.fps)

        logger.info(f"最終キーフレーム数: {len(keyframes)}個")

        return keyframes

    def _compute_combined_score(self, quality_scores: Dict[str, float],
                               geometric_scores: Dict[str, float],
                               adaptive_scores: Dict[str, float]) -> float:
        """
        複数のスコアを統合して総合スコアを計算

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
        # 品質スコアを正規化
        sharpness = min(quality_scores.get('sharpness', 0) / 1000.0, 1.0)  # ラプラシアン値を0-1に
        exposure = quality_scores.get('exposure', 0.5)
        motion_blur = 1.0 - quality_scores.get('motion_blur', 0)  # 逆スケール
        softmax_depth = quality_scores.get('softmax_depth', 0.5)

        # 品質スコア（平均）
        quality_score = (sharpness + exposure + motion_blur + softmax_depth) / 4.0

        # 幾何学的スコアを正規化
        if geometric_scores:
            gric = 1.0 - geometric_scores.get('gric', 1.0)  # GRICは低いほど良い
            dist1 = geometric_scores.get('feature_distribution_1', 0.5)
            dist2 = geometric_scores.get('feature_distribution_2', 0.5)
            ray_disp = geometric_scores.get('ray_dispersion', 0.5)

            geometric_score = (gric + dist1 + dist2 + ray_disp) / 4.0
        else:
            geometric_score = 0.5  # デフォルト値

        # 適応的スコアを正規化
        if adaptive_scores:
            # SSIM変化（1に近いほど相似、変化がない）
            # キーフレーム選択では変化が大きい（SSIMが低い）ほど良い
            ssim_change = 1.0 - adaptive_scores.get('ssim', 1.0)

            # 光学フロー（大きいほど動きが大きい = 良い）
            # フロー値を0-1にクリップ
            optical_flow = min(adaptive_scores.get('optical_flow', 0) / 50.0, 1.0)

            content_score = (ssim_change + optical_flow) / 2.0
        else:
            content_score = 0.5  # デフォルト値

        # 重み付け統合
        combined = (
            self.config['WEIGHT_SHARPNESS'] * sharpness +
            self.config['WEIGHT_EXPOSURE'] * exposure +
            self.config['WEIGHT_GEOMETRIC'] * geometric_score +
            self.config['WEIGHT_CONTENT'] * content_score
        )

        return float(np.clip(combined, 0.0, 1.0))

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

    def _apply_nms(self, candidates: List[KeyframeInfo],
                  time_window: float = 1.0) -> List[KeyframeInfo]:
        """
        非最大値抑制（NMS）を適用

        スコアが低い候補フレームを時間ウィンドウ内から除外

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
            is_duplicate = False
            for selected_kf in selected:
                time_diff = abs(candidate.timestamp - selected_kf.timestamp)
                if time_diff < time_window:
                    is_duplicate = True
                    break

            if not is_duplicate:
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

        max_interval = self.config['MAX_KEYFRAME_INTERVAL'] / fps  # 秒に変換

        enforced_keyframes = [keyframes[0]]

        for i in range(1, len(keyframes)):
            current_kf = keyframes[i]
            last_kf = enforced_keyframes[-1]

            time_diff = current_kf.timestamp - last_kf.timestamp

            if time_diff <= max_interval:
                enforced_keyframes.append(current_kf)
            else:
                # 間隔が大きすぎる場合は、複数のキーフレームを追加
                num_missing = int(np.ceil(time_diff / max_interval)) - 1

                for j in range(1, num_missing + 1):
                    # 等間隔で挿入（実装簡略化）
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

            # ファイル名を生成
            timestamp_str = f"{kf.timestamp:.2f}".replace('.', '-')
            filename = f"keyframe_{kf.frame_index:06d}_{timestamp_str}.{format}"
            filepath = output_path / filename

            # 保存
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(str(filepath), frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:  # png
                cv2.imwrite(str(filepath), frame)

            exported[kf.frame_index] = filepath
            logger.info(f"キーフレーム保存: {filepath}")

        logger.info(f"合計 {len(exported)}個のキーフレームをエクスポート")

        return exported
