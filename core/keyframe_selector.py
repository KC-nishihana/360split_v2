"""
キーフレーム選択メインモジュール - 360Split用
全評価器を統合したキーフレーム選択パイプライン
2段階パイプライン + マルチスレッド処理 + 最適化NMS実装
"""

import cv2
import numpy as np
import uuid
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import Counter, deque
from time import perf_counter

from .video_loader import VideoLoader, VideoMetadata, create_video_capture
from .quality_evaluator import QualityEvaluator
from .quality_score import (
    apply_abs_guard,
    compose_quality,
    compose_legacy_quality_proxy,
    compute_raw_metrics,
    normalize_batch_p10_p90,
    parse_roi_spec,
)
from .geometric_evaluator import GeometricEvaluator
from .adaptive_selector import AdaptiveSelector
from .accelerator import get_accelerator
from processing.fisheye_rig import FisheyeRigProcessor
from processing.mask_processor import MaskProcessor
from .visual_odometry import KLTVisualOdometry, calibration_from_dict, integrate_relative_trajectory
from .pipeline import run_stage1_filter, run_stage2_evaluator, run_stage3_refiner
from .stage_temp_store import StageTempStore
from .exceptions import (
    GeometricDegeneracyError,
    EstimationFailureError,
    InsufficientFeaturesError
)
from config import (
    GRICConfig,
    Equirect360Config,
    NormalizationConfig,
    KeyframeConfig,
    SELECTOR_ALIAS_MAP,
    normalize_config_dict,
)

from utils.logger import get_logger
from utils.image_io import write_image
logger = get_logger(__name__)

DYNAMIC_MASK_DEFAULT_CLASSES = ["人物", "人", "自転車", "バイク", "車両", "動物"]
_LEGACY_CONFIG_WARNING_EMITTED = False


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
    dynamic_mask: Optional[np.ndarray] = None
    is_stationary: bool = False
    stationary_penalty_applied: bool = False
    stage3_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class Stage2PerfStats:
    enabled: bool = True
    total_candidates: int = 0
    processed_candidates: int = 0
    selected_candidates: int = 0
    frame_read_s: float = 0.0
    dynamic_mask_s: float = 0.0
    geometric_eval_s: float = 0.0
    adaptive_eval_s: float = 0.0
    rerun_log_s: float = 0.0
    dynamic_mask_calls: int = 0
    dynamic_mask_mode_both_calls: int = 0
    dynamic_mask_mode_yolo_only_calls: int = 0
    dynamic_mask_mode_motion_only_calls: int = 0
    total_s: float = 0.0


@dataclass
class Stage2FrameRecord:
    frame_index: int
    frame: Optional[np.ndarray]
    quality_scores: Dict[str, float]
    geometric_scores: Dict[str, float]
    adaptive_scores: Dict[str, float]
    metrics: Dict[str, Any]
    is_candidate: bool = False
    is_keyframe: bool = False
    drop_reason: str = ""
    t_xyz: Optional[List[float]] = None
    q_wxyz: Optional[List[float]] = None
    points_world: Optional[np.ndarray] = None


@dataclass
class Stage1ScanArtifact:
    candidates: List[Dict[str, Any]]
    records: List[Dict[str, Any]]
    sampled_frames: int
    total_frames: int


class KeyframeSelector:
    """
    キーフレーム選択パイプライン（最適化版）

    複数の品質・幾何学的・適応的評価器を統合して、
    360度ビデオから最適なキーフレームを自動選択する。

    2段階パイプライン：
    - Stage 1: 高速品質フィルタリング（全フレーム）
    - Stage 2: 精密評価（候補フレームのみ）
    """

    def __init__(self, config: Optional[Dict[str, Any] | KeyframeConfig] = None):
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
            'NMS_TIME_WINDOW': 1.0,
            'GRIC_RATIO_THRESHOLD': 0.8,
            'SSIM_CHANGE_THRESHOLD': 0.85,
            'MOTION_BLUR_THRESHOLD': 0.3,
            'EXPOSURE_THRESHOLD': 0.35,
            'QUALITY_FILTER_ENABLED': True,
            'QUALITY_THRESHOLD': 0.50,
            'QUALITY_ROI_MODE': 'circle',
            'QUALITY_ROI_RATIO': 0.40,
            'QUALITY_ABS_LAPLACIAN_MIN': 35.0,
            'QUALITY_USE_ORB': True,
            'QUALITY_WEIGHT_SHARPNESS': 0.40,
            'QUALITY_WEIGHT_TENENGRAD': 0.30,
            'QUALITY_WEIGHT_EXPOSURE': 0.15,
            'QUALITY_WEIGHT_KEYPOINTS': 0.15,
            'QUALITY_NORM_P_LOW': 10.0,
            'QUALITY_NORM_P_HIGH': 90.0,
            'QUALITY_DEBUG': False,
            'QUALITY_TENENGRAD_SCALE': 1.0,
            'PAIR_MOTION_AGGREGATION': 'max',
            'MIN_FEATURE_MATCHES': 30,
            'THUMBNAIL_SIZE': (192, 108),
            'SAMPLE_INTERVAL': 1,
            'STAGE1_BATCH_SIZE': 32,
            'STAGE1_GRAB_THRESHOLD': 30,
            'STAGE1_EVAL_SCALE': 0.5,
            'OPENCV_THREAD_COUNT': 0,
            'STAGE1_PROCESS_WORKERS': 0,
            'STAGE1_PREFETCH_SIZE': 32,
            'STAGE1_METRICS_BATCH_SIZE': 64,
            'STAGE1_GPU_BATCH_ENABLED': True,
            'DARWIN_CAPTURE_BACKEND': 'auto',
            'MPS_MIN_PIXELS': 256 * 256,
            'ENABLE_RIG_STITCHING': True,
            'EQUIRECT_WIDTH': 4096,
            'EQUIRECT_HEIGHT': 2048,
            'RIG_FEATURE_METHOD': 'orb',
            'ENABLE_DYNAMIC_MASK_REMOVAL': False,
            'ENABLE_FISHEYE_BORDER_MASK': True,
            'FISHEYE_MASK_RADIUS_RATIO': 0.94,
            'FISHEYE_MASK_CENTER_OFFSET_X': 0,
            'FISHEYE_MASK_CENTER_OFFSET_Y': 0,
            'DYNAMIC_MASK_USE_YOLO_SAM': True,
            'DYNAMIC_MASK_USE_MOTION_DIFF': True,
            'DYNAMIC_MASK_MOTION_FRAMES': 3,
            'DYNAMIC_MASK_MOTION_THRESHOLD': 30,
            'DYNAMIC_MASK_DILATION_SIZE': 5,
            'DYNAMIC_MASK_TARGET_CLASSES': list(DYNAMIC_MASK_DEFAULT_CLASSES),
            'DYNAMIC_MASK_INPAINT_ENABLED': False,
            'DYNAMIC_MASK_INPAINT_MODULE': '',
            'YOLO_MODEL_PATH': 'yolo26n-seg.pt',
            'SAM_MODEL_PATH': 'sam3_t.pt',
            'CONFIDENCE_THRESHOLD': 0.25,
            'DETECTION_DEVICE': 'auto',
            'ENABLE_PROFILE': False,
            'STAGE2_PERF_PROFILE': True,
            'STAGE2_MASK_CACHE_TTL_FRAMES': 0,
            # レスキューモード設定
            'ENABLE_RESCUE_MODE': False,
            'RESCUE_FEATURE_THRESHOLD': 15,  # この値以下で特徴点不足と判定
            'RESCUE_LAPLACIAN_FACTOR': 0.5,  # レスキュー時のLaplacian閾値倍率
            'RESCUE_WINDOW_SIZE': 10,  # レスキューモード判定のウィンドウサイズ
            # 混合環境での強制挿入設定
            'FORCE_KEYFRAME_ON_EXPOSURE_CHANGE': False,
            'EXPOSURE_CHANGE_THRESHOLD': 0.3,  # 露出変化の検知閾値
            'ADAPTIVE_THRESHOLDING': False,  # 動的閾値調整
            # 停止区間検出・抑制
            'STATIONARY_ENABLE': True,
            'STATIONARY_MIN_DURATION_SEC': 0.7,
            'STATIONARY_USE_QUANTILE_THRESHOLD': True,
            'STATIONARY_QUANTILE': 0.10,
            'STATIONARY_TRANSLATION_THRESHOLD': None,
            'STATIONARY_ROTATION_THRESHOLD': None,
            'STATIONARY_FLOW_THRESHOLD': None,
            'STATIONARY_MIN_MATCH_COUNT_FOR_VO': 30,
            'STATIONARY_FALLBACK_WHEN_VO_UNRELIABLE': 'not_stationary',
            'STATIONARY_SOFT_PENALTY': True,
            'STATIONARY_PENALTY': 0.7,
            'STATIONARY_ALLOW_BOUNDARY_FRAMES': True,
            'STATIONARY_BOUNDARY_GRACE_FRAMES': 2,
            'STATIONARY_HYSTERESIS_EXIT_SCALE': 1.25,
            # Stage0/Stage3
            'ENABLE_STAGE0_SCAN': True,
            'STAGE0_STRIDE': 5,
            'ENABLE_STAGE3_REFINEMENT': True,
            'STAGE3_WEIGHT_BASE': 0.70,
            'STAGE3_WEIGHT_TRAJECTORY': 0.25,
            'STAGE3_WEIGHT_STAGE0_RISK': 0.05,
            'STAGE3_DISABLE_TRAJ_WHEN_VO_UNRELIABLE': True,
            'STAGE3_VO_VALID_RATIO_THRESHOLD': 0.50,
            'VO_ENABLED': True,
            'VO_CENTER_ROI_RATIO': 0.6,
            'VO_MAX_FEATURES': 600,
            'VO_QUALITY_LEVEL': 0.01,
            'VO_MIN_DISTANCE': 8,
            'VO_MIN_TRACK_POINTS': 24,
            'VO_RANSAC_THRESHOLD': 1.0,
            'VO_DOWNSCALE_LONG_EDGE': 1000,
            'VO_MATCH_NORM_FACTOR': 120.0,
            'VO_T_SIGN': 1.0,
            'VO_FRAME_SUBSAMPLE': 1,
            'VO_ADAPTIVE_ROI_ENABLE': True,
            'VO_ADAPTIVE_ROI_MIN': 0.45,
            'VO_ADAPTIVE_ROI_MAX': 0.70,
            'VO_FAST_FAIL_INLIER_RATIO': 0.12,
            'VO_STEP_PROXY_CLIP_PX': 80.0,
            'VO_ESSENTIAL_METHOD': 'auto',
            'VO_SUBPIXEL_REFINE': True,
            'VO_ADAPTIVE_SUBSAMPLE': False,
            'VO_SUBSAMPLE_MIN': 1,
            'VO_CONFIDENCE_LOW_THRESHOLD': 0.35,
            'VO_CONFIDENCE_MID_THRESHOLD': 0.55,
            'CALIB_XML': '',
            'FRONT_CALIB_XML': '',
            'REAR_CALIB_XML': '',
            'CALIB_MODEL': 'auto',
            'POSE_BACKEND': 'vo',
            'COLMAP_KEYFRAME_POLICY': '',
            'COLMAP_KEYFRAME_TARGET_MODE': 'auto',
            'COLMAP_KEYFRAME_TARGET_MIN': 120,
            'COLMAP_KEYFRAME_TARGET_MAX': 240,
            'COLMAP_NMS_WINDOW_SEC': 0.35,
            'COLMAP_ENABLE_STAGE0': True,
            'COLMAP_MOTION_AWARE_SELECTION': True,
            'COLMAP_NMS_MOTION_WINDOW_RATIO': 0.5,
            'COLMAP_STAGE1_ADAPTIVE_THRESHOLD': True,
            'COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN': 3,
            'COLMAP_STAGE1_MAX_CANDIDATES': 360,
            'COLMAP_RIG_POLICY': 'lr_opk',
            'COLMAP_RIG_SEED_OPK_DEG': [0.0, 0.0, 180.0],
            'COLMAP_WORKSPACE_SCOPE': 'run_scoped',
            'COLMAP_REUSE_DB': False,
            'COLMAP_ANALYSIS_MASK_PROFILE': 'colmap_safe',
            'COLMAP_ANALYSIS_TARGET_CLASSES': list(DYNAMIC_MASK_DEFAULT_CLASSES),
            'MOTION_BLUR_METHOD': 'legacy',
            'ENABLE_STAGE2_PIPELINE_PARALLEL': False,
            'FLOW_DOWNSCALE': 1.0,
            'RESUME_ENABLED': False,
            'KEEP_TEMP_ON_SUCCESS': True,
        }

        # 外部設定でオーバーライド
        if config:
            normalized_from_dataclass: Dict[str, Any]
            incoming: Dict[str, Any]
            if isinstance(config, KeyframeConfig):
                normalized_from_dataclass = config.to_selector_dict()
                incoming = {}
            else:
                incoming = dict(config)
                normalized_from_dataclass = KeyframeConfig.from_dict(incoming).to_selector_dict()
                global _LEGACY_CONFIG_WARNING_EMITTED
                if not _LEGACY_CONFIG_WARNING_EMITTED:
                    lowered = normalize_config_dict(incoming)
                    used_legacy = [k for k in SELECTOR_ALIAS_MAP if k in lowered and k not in incoming and SELECTOR_ALIAS_MAP[k] in incoming]
                    if any(k in incoming for k in SELECTOR_ALIAS_MAP) or used_legacy:
                        logger.warning("Legacy lower_snake_case config keys are accepted for now; migrate to KeyframeConfig/upper keys.")
                        _LEGACY_CONFIG_WARNING_EMITTED = True
            # Preserve unknown keys while overriding canonical keys from KeyframeConfig.
            if incoming:
                self.config.update(incoming)
            self.config.update(normalized_from_dataclass)

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
        stage1_eval_scale = float(np.clip(self.config.get('STAGE1_EVAL_SCALE', 0.5), 0.1, 1.0))
        self.quality_evaluator = QualityEvaluator(
            eval_scale=stage1_eval_scale,
            motion_blur_method=str(self.config.get("MOTION_BLUR_METHOD", "legacy")),
        )
        self.geometric_evaluator = GeometricEvaluator(
            gric_config=gric_config,
            equirect_config=equirect_config
        )
        self.adaptive_selector = AdaptiveSelector(
            flow_downscale=float(np.clip(self.config.get('FLOW_DOWNSCALE', 1.0), 0.1, 1.0))
        )
        self.rig_processor = FisheyeRigProcessor(equirect_config=equirect_config)

        # アクセレータから情報を取得
        self.accelerator = get_accelerator()
        self.accelerator.configure_runtime(self.config)
        logger.info(
            f"アクセレータ: device={self.accelerator.device_name}, "
            f"thread_count={self.accelerator.num_threads}"
        )

        # レスキューモード関連の状態変数
        self.is_rescue_mode = False
        self.feature_count_history = deque(maxlen=self.config['RESCUE_WINDOW_SIZE'])
        self.previous_brightness = None
        self.rescue_mode_keyframes = []  # レスキューモードで選択されたキーフレーム
        self.target_mask_generator = self._build_target_mask_generator()
        self.mask_processor = MaskProcessor()
        self.vo_estimator = KLTVisualOdometry(
            max_features=int(self.config.get('VO_MAX_FEATURES', 600)),
            quality_level=float(self.config.get('VO_QUALITY_LEVEL', 0.01)),
            min_distance=float(self.config.get('VO_MIN_DISTANCE', 8.0)),
            min_track_points=int(self.config.get('VO_MIN_TRACK_POINTS', 24)),
            ransac_threshold=float(self.config.get('VO_RANSAC_THRESHOLD', 1.0)),
            center_roi_ratio=float(self.config.get('VO_CENTER_ROI_RATIO', 0.6)),
            downscale_long_edge=int(self.config.get('VO_DOWNSCALE_LONG_EDGE', 1000)),
            fast_fail_inlier_ratio=float(self.config.get('VO_FAST_FAIL_INLIER_RATIO', 0.12)),
            step_proxy_clip_px=float(self.config.get('VO_STEP_PROXY_CLIP_PX', 80.0)),
            essential_method=str(self.config.get('VO_ESSENTIAL_METHOD', 'auto')),
            subpixel_refine=bool(self.config.get('VO_SUBPIXEL_REFINE', True)),
        )
        self.stage1_quality_records: List[Dict[str, Any]] = []
        self.last_selection_runtime: Dict[str, Any] = {}

    def _load_inpaint_hook(self):
        module_name = str(self.config.get('DYNAMIC_MASK_INPAINT_MODULE', '') or '').strip()
        if not module_name:
            return None
        try:
            mod = __import__(module_name, fromlist=['inpaint_frame'])
            hook = getattr(mod, 'inpaint_frame', None)
            if callable(hook):
                return hook
            logger.warning(f"インペイントモジュールに inpaint_frame がありません: {module_name}")
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.warning(f"インペイントモジュール読み込み失敗: {module_name}, err={e}")
        return None

    def _build_target_mask_generator(self):
        if not self.config.get('ENABLE_DYNAMIC_MASK_REMOVAL', False):
            return None

        use_yolo_sam = bool(self.config.get('DYNAMIC_MASK_USE_YOLO_SAM', True))
        use_motion = bool(self.config.get('DYNAMIC_MASK_USE_MOTION_DIFF', True))
        if not use_yolo_sam and not use_motion:
            logger.info("動体除去は有効だが、YOLO/SAM と MotionDiff が両方無効のためスキップ")
            return None

        try:
            from processing.target_mask_generator import TargetMaskGenerator

            inpaint_hook = self._load_inpaint_hook() if self.config.get('DYNAMIC_MASK_INPAINT_ENABLED', False) else None
            return TargetMaskGenerator(
                yolo_model_path=str(self.config.get('YOLO_MODEL_PATH', 'yolo26n-seg.pt')),
                sam_model_path=str(self.config.get('SAM_MODEL_PATH', 'sam3_t.pt')),
                confidence_threshold=float(self.config.get('CONFIDENCE_THRESHOLD', 0.25)),
                device=str(self.config.get('DETECTION_DEVICE', 'auto')),
                enable_motion_detection=use_motion,
                motion_history_frames=int(self.config.get('DYNAMIC_MASK_MOTION_FRAMES', 3)),
                motion_threshold=int(self.config.get('DYNAMIC_MASK_MOTION_THRESHOLD', 30)),
                motion_mask_dilation_size=int(self.config.get('DYNAMIC_MASK_DILATION_SIZE', 5)),
                enable_mask_inpaint=bool(self.config.get('DYNAMIC_MASK_INPAINT_ENABLED', False)),
                inpaint_hook=inpaint_hook,
            )
        except (ImportError, RuntimeError, OSError, ValueError, TypeError, AttributeError) as e:
            logger.warning(f"動体マスク生成器の初期化に失敗したため無効化: {e}")
            return None

    def _build_dynamic_masks(
        self,
        frame_prev: np.ndarray,
        frame_cur: np.ndarray,
        context_frames: List[np.ndarray],
        frame_prev_idx: Optional[int] = None,
        frame_cur_idx: Optional[int] = None,
        mask_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.target_mask_generator is None:
            return None, None

        target_classes = self.config.get(
            'COLMAP_ANALYSIS_TARGET_CLASSES',
            self.config.get('DYNAMIC_MASK_TARGET_CLASSES', DYNAMIC_MASK_DEFAULT_CLASSES),
        )
        if not isinstance(target_classes, list):
            target_classes = list(target_classes) if target_classes else list(DYNAMIC_MASK_DEFAULT_CLASSES)

        use_yolo_sam = bool(self.config.get('DYNAMIC_MASK_USE_YOLO_SAM', True))
        classes_for_detection = target_classes if use_yolo_sam else []

        motion_window = int(self.config.get('DYNAMIC_MASK_MOTION_FRAMES', 3))
        ctx = [f for f in context_frames if f is not None]
        ctx.append(frame_prev)
        ctx.append(frame_cur)
        if len(ctx) > motion_window:
            ctx = ctx[-motion_window:]

        ttl = max(0, int(self.config.get('STAGE2_MASK_CACHE_TTL_FRAMES', 0)))

        def _select_cached_mask(frame_idx: Optional[int]) -> Optional[np.ndarray]:
            if mask_cache is None or frame_idx is None or ttl <= 0:
                return None
            if frame_idx in mask_cache:
                return mask_cache[frame_idx]
            nearest_idx = None
            nearest_gap = ttl + 1
            for cached_idx in mask_cache.keys():
                gap = abs(cached_idx - frame_idx)
                if gap <= ttl and gap < nearest_gap:
                    nearest_idx = cached_idx
                    nearest_gap = gap
            return mask_cache.get(nearest_idx) if nearest_idx is not None else None

        def _generate_mask(frame: np.ndarray, frame_idx: Optional[int]) -> np.ndarray:
            cached = _select_cached_mask(frame_idx)
            if cached is not None:
                return cached
            generated = self.target_mask_generator.generate_mask(
                frame,
                classes_for_detection,
                motion_frames=ctx,
            )
            if mask_cache is not None and frame_idx is not None and ttl > 0:
                mask_cache[frame_idx] = generated
            return generated

        try:
            mask_prev = _generate_mask(frame_prev, frame_prev_idx)
            mask_cur = _generate_mask(frame_cur, frame_cur_idx)
            return mask_prev, mask_cur
        except (cv2.error, RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"動体マスク生成失敗（無効化して継続）: {e}")
            return None, None

    def _build_single_frame_dynamic_mask(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.target_mask_generator is None:
            return None
        target_classes = self.config.get(
            'COLMAP_ANALYSIS_TARGET_CLASSES',
            self.config.get('DYNAMIC_MASK_TARGET_CLASSES', DYNAMIC_MASK_DEFAULT_CLASSES),
        )
        if not isinstance(target_classes, list):
            target_classes = list(target_classes) if target_classes else list(DYNAMIC_MASK_DEFAULT_CLASSES)
        classes_for_detection = target_classes if bool(self.config.get('DYNAMIC_MASK_USE_YOLO_SAM', True)) else []
        try:
            return self.target_mask_generator.generate_mask(
                frame,
                classes_for_detection,
                motion_frames=[frame],
            )
        except (cv2.error, RuntimeError, ValueError, TypeError):
            return None

    @staticmethod
    def _emit_frame_log(
        frame_log_callback: Optional[Callable[[Dict[str, Any]], None]],
        payload: Dict[str, Any],
        perf_stats: Optional[Stage2PerfStats] = None,
    ) -> None:
        if frame_log_callback is None:
            return
        started = perf_counter() if perf_stats is not None and perf_stats.enabled else 0.0
        try:
            frame_log_callback(payload)
        except (RuntimeError, ValueError, TypeError) as e:
            frame_idx = payload.get("frame_index", -1)
            logger.debug(f"frame_log_callback失敗: frame={frame_idx}, err={e}")
        finally:
            if perf_stats is not None and perf_stats.enabled:
                perf_stats.rerun_log_s += max(0.0, perf_counter() - started)

    @staticmethod
    def _log_stage2_perf_summary(perf_stats: Stage2PerfStats) -> None:
        if not perf_stats.enabled:
            return
        candidates = max(1, perf_stats.total_candidates)
        processed = max(1, perf_stats.processed_candidates)
        total_ms = perf_stats.total_s * 1000.0
        read_ms = perf_stats.frame_read_s * 1000.0
        mask_ms = perf_stats.dynamic_mask_s * 1000.0
        geom_ms = perf_stats.geometric_eval_s * 1000.0
        adapt_ms = perf_stats.adaptive_eval_s * 1000.0
        rerun_ms = perf_stats.rerun_log_s * 1000.0
        logger.info(
            "stage2_perf,"
            f" total_candidates={perf_stats.total_candidates},"
            f" processed_candidates={perf_stats.processed_candidates},"
            f" selected_candidates={perf_stats.selected_candidates},"
            f" total_ms={total_ms:.3f},"
            f" per_candidate_ms={total_ms/candidates:.3f},"
            f" per_processed_ms={total_ms/processed:.3f},"
            f" frame_read_ms={read_ms:.3f},"
            f" dynamic_mask_ms={mask_ms:.3f},"
            f" geometric_eval_ms={geom_ms:.3f},"
            f" adaptive_eval_ms={adapt_ms:.3f},"
            f" rerun_log_ms={rerun_ms:.3f},"
            f" dynamic_mask_calls={perf_stats.dynamic_mask_calls},"
            f" dynamic_mask_mode_both_calls={perf_stats.dynamic_mask_mode_both_calls},"
            f" dynamic_mask_mode_yolo_only_calls={perf_stats.dynamic_mask_mode_yolo_only_calls},"
            f" dynamic_mask_mode_motion_only_calls={perf_stats.dynamic_mask_mode_motion_only_calls}"
        )

    def _is_fisheye_border_mask_enabled(self, is_paired: bool) -> bool:
        return bool(is_paired and self.config.get('ENABLE_FISHEYE_BORDER_MASK', True))

    def _create_fisheye_valid_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return self.mask_processor.create_fisheye_valid_mask(
            width=w,
            height=h,
            radius_ratio=float(self.config.get('FISHEYE_MASK_RADIUS_RATIO', 0.94)),
            offset_x=int(self.config.get('FISHEYE_MASK_CENTER_OFFSET_X', 0)),
            offset_y=int(self.config.get('FISHEYE_MASK_CENTER_OFFSET_Y', 0)),
        )

    @staticmethod
    def _combine_feature_masks(
        dynamic_mask: Optional[np.ndarray],
        fisheye_valid_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if dynamic_mask is None:
            return fisheye_valid_mask
        if fisheye_valid_mask is None:
            return dynamic_mask
        return cv2.bitwise_and(dynamic_mask.astype(np.uint8), fisheye_valid_mask.astype(np.uint8))

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
        backend_pref = str(
            self.config.get(
                "DARWIN_CAPTURE_BACKEND",
                self.config.get("darwin_capture_backend", "auto"),
            )
            or "auto"
        )
        cap = create_video_capture(str(video_path), backend_preference=backend_pref)
        if not cap.isOpened():
            raise RuntimeError(f"分析用ビデオキャプチャを開けません: {video_path}")
        return cap

    def _open_independent_pair_captures(self, video_loader):
        """
        ペア入力（stereo_lr / front_rear）向けに独立キャプチャを開く。

        解析中にGUI再生側の VideoCapture と競合しないよう、
        video_loader が保持するパスから別インスタンスを作成する。
        パス情報が取得できない場合は (None, None) を返し、
        呼び出し側で get_frame_pair() フォールバックを使う。
        """
        path_a = None
        path_b = None

        if hasattr(video_loader, "left_path") and hasattr(video_loader, "right_path"):
            path_a = getattr(video_loader, "left_path", None)
            path_b = getattr(video_loader, "right_path", None)
        elif hasattr(video_loader, "front_path") and hasattr(video_loader, "rear_path"):
            path_a = getattr(video_loader, "front_path", None)
            path_b = getattr(video_loader, "rear_path", None)

        if not path_a or not path_b:
            return None, None

        cap_a = self._open_independent_capture(path_a)
        cap_b = self._open_independent_capture(path_b)
        return cap_a, cap_b

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

    @staticmethod
    def _extract_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        return runs

    @staticmethod
    def _resolve_stationary_threshold(
        values: np.ndarray,
        fixed_threshold: Optional[float],
        use_quantile: bool,
        quantile: float,
    ) -> float:
        finite = values[np.isfinite(values)]
        if fixed_threshold is not None and not use_quantile:
            return float(fixed_threshold)
        if finite.size == 0:
            return float(fixed_threshold if fixed_threshold is not None else 0.0)
        if fixed_threshold is None or use_quantile:
            return float(np.quantile(finite, np.clip(quantile, 0.0, 1.0)))
        return float(fixed_threshold)

    def _compute_stationary_flags(
        self,
        records: List[Stage2FrameRecord],
        fps: float,
    ) -> Dict[str, Any]:
        n = len(records)
        if n == 0 or not bool(self.config.get('STATIONARY_ENABLE', True)):
            for record in records:
                record.metrics["stationary_vo_flag"] = 0.0
                record.metrics["stationary_flow_flag"] = 0.0
                record.metrics["is_stationary"] = 0.0
                record.metrics["stationary_confidence"] = 0.0
                record.metrics["stationary_penalty_applied"] = 0.0
            return {"mask": np.zeros(0, dtype=bool), "boundary_mask": np.zeros(0, dtype=bool)}

        trans = np.asarray(
            [r.metrics.get("vo_step_proxy_norm", r.metrics.get("translation_delta", np.nan)) for r in records],
            dtype=np.float64,
        )
        rot = np.asarray([r.metrics.get("rotation_delta", np.nan) for r in records], dtype=np.float64)
        flow = np.asarray([r.metrics.get("flow_mag", np.nan) for r in records], dtype=np.float64)
        match_count = np.asarray([r.metrics.get("match_count", np.nan) for r in records], dtype=np.float64)

        use_quantile = bool(self.config.get('STATIONARY_USE_QUANTILE_THRESHOLD', True))
        quantile = float(self.config.get('STATIONARY_QUANTILE', 0.10))
        t_th = self._resolve_stationary_threshold(
            trans,
            self.config.get('STATIONARY_TRANSLATION_THRESHOLD'),
            use_quantile,
            quantile,
        )
        r_th = self._resolve_stationary_threshold(
            rot,
            self.config.get('STATIONARY_ROTATION_THRESHOLD'),
            use_quantile,
            quantile,
        )
        f_th = self._resolve_stationary_threshold(
            flow,
            self.config.get('STATIONARY_FLOW_THRESHOLD'),
            use_quantile,
            quantile,
        )
        min_match = int(self.config.get('STATIONARY_MIN_MATCH_COUNT_FOR_VO', self.config.get('MIN_FEATURE_MATCHES', 30)))
        fallback = str(self.config.get('STATIONARY_FALLBACK_WHEN_VO_UNRELIABLE', 'not_stationary')).strip().lower()

        vo_reliable = np.isfinite(match_count) & (match_count >= float(min_match))
        vo_enter = np.isfinite(trans) & np.isfinite(rot) & (trans <= t_th) & (rot <= r_th)
        flow_enter = np.isfinite(flow) & (flow <= f_th)

        if fallback == "flow_only":
            vo_enter = np.where(vo_reliable, vo_enter, flow_enter)
        else:
            vo_enter = np.where(vo_reliable, vo_enter, False)

        enter_mask = vo_enter & flow_enter

        exit_scale = max(1.0, float(self.config.get('STATIONARY_HYSTERESIS_EXIT_SCALE', 1.25)))
        vo_exit = np.isfinite(trans) & np.isfinite(rot) & (trans <= t_th * exit_scale) & (rot <= r_th * exit_scale)
        flow_exit = np.isfinite(flow) & (flow <= f_th * exit_scale)
        if fallback == "flow_only":
            vo_exit = np.where(vo_reliable, vo_exit, flow_exit)
        else:
            vo_exit = np.where(vo_reliable, vo_exit, False)

        stable_mask = np.zeros(n, dtype=bool)
        in_stationary = False
        for i in range(n):
            if not in_stationary:
                if enter_mask[i]:
                    in_stationary = True
                    stable_mask[i] = True
            else:
                if vo_exit[i] & flow_exit[i]:
                    stable_mask[i] = True
                else:
                    in_stationary = False

        min_len = max(1, int(round(float(self.config.get('STATIONARY_MIN_DURATION_SEC', 0.7)) * max(fps, 1e-6))))
        for s, e in self._extract_true_runs(stable_mask):
            if (e - s) < min_len:
                stable_mask[s:e] = False

        boundary_mask = np.zeros(n, dtype=bool)
        if bool(self.config.get('STATIONARY_ALLOW_BOUNDARY_FRAMES', True)):
            grace = max(0, int(self.config.get('STATIONARY_BOUNDARY_GRACE_FRAMES', 2)))
            for s, e in self._extract_true_runs(stable_mask):
                bs = max(0, s - grace)
                be = min(n, s + grace + 1)
                boundary_mask[bs:be] = True
                es = max(0, e - 1 - grace)
                ee = min(n, e + grace)
                boundary_mask[es:ee] = True

        for i, record in enumerate(records):
            record.metrics["flow_mag"] = float(flow[i]) if np.isfinite(flow[i]) else 0.0
            record.metrics["stationary_vo_flag"] = 1.0 if bool(vo_enter[i]) else 0.0
            record.metrics["stationary_flow_flag"] = 1.0 if bool(flow_enter[i]) else 0.0
            record.metrics["is_stationary"] = 1.0 if bool(stable_mask[i]) else 0.0
            record.metrics["stationary_confidence"] = 1.0 if bool(vo_reliable[i]) else 0.0
            record.metrics["stationary_penalty_applied"] = 0.0
            record.metrics["stationary_translation_threshold"] = float(t_th)
            record.metrics["stationary_rotation_threshold"] = float(r_th)
            record.metrics["stationary_flow_threshold"] = float(f_th)

        return {"mask": stable_mask, "boundary_mask": boundary_mask}

    def _apply_stationary_penalty(
        self,
        candidates: List[KeyframeInfo],
        records: List[Stage2FrameRecord],
        fps: float,
    ) -> List[KeyframeInfo]:
        stationary = self._compute_stationary_flags(records, fps=fps)
        if not candidates:
            return candidates

        mask = stationary["mask"]
        boundary_mask = stationary["boundary_mask"]

        frame_to_idx = {record.frame_index: i for i, record in enumerate(records)}
        soft = bool(self.config.get('STATIONARY_SOFT_PENALTY', True))
        penalty = float(np.clip(self.config.get('STATIONARY_PENALTY', 0.7), 0.0, 1.0))

        for candidate in candidates:
            idx = frame_to_idx.get(candidate.frame_index)
            if idx is None:
                continue
            is_stationary = bool(mask[idx])
            candidate.is_stationary = is_stationary
            if candidate.is_force_inserted:
                continue
            if not is_stationary:
                continue
            if bool(boundary_mask[idx]):
                continue
            if soft:
                candidate.combined_score *= (1.0 - penalty)
            else:
                candidate.combined_score = 0.0
            candidate.stationary_penalty_applied = True
            records[idx].metrics["stationary_penalty_applied"] = 1.0

        return candidates

    @staticmethod
    def _lookup_stage0_metrics(
        frame_idx: int,
        stage0_metrics: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not stage0_metrics:
            return {"flow_mag_light": 0.0, "ssim_light": 1.0, "motion_risk": 0.0}
        if frame_idx in stage0_metrics:
            return stage0_metrics[frame_idx]
        keys = sorted(stage0_metrics.keys())
        pos = int(np.searchsorted(keys, frame_idx))
        nearest = keys[max(0, min(len(keys) - 1, pos))]
        if pos > 0 and pos < len(keys):
            left = keys[pos - 1]
            right = keys[pos]
            nearest = left if abs(frame_idx - left) <= abs(frame_idx - right) else right
        return stage0_metrics.get(nearest, {"flow_mag_light": 0.0, "ssim_light": 1.0, "motion_risk": 0.0})

    def _get_runtime_vo_calibration(self, is_paired: bool):
        runtime = self.config.get("calibration_runtime", {})
        if not isinstance(runtime, dict):
            return None
        if is_paired:
            front = calibration_from_dict(runtime.get("front"))
            if front is not None:
                return front
            return calibration_from_dict(runtime.get("mono"))
        return calibration_from_dict(runtime.get("mono"))

    def _resolve_vo_runtime(self, is_paired: bool, stage_label: str) -> Tuple[bool, Optional[object], str]:
        if not bool(self.config.get('VO_ENABLED', True)):
            return False, None, "vo_disabled_by_config"

        vo_calib = self._get_runtime_vo_calibration(is_paired=is_paired)
        if vo_calib is None:
            logger.warning(f"{stage_label} VO disabled: calibration not available")
            return False, None, "calibration_unavailable"

        projection_mode = str(
            self.config.get('PROJECTION_MODE', self.config.get('projection_mode', ''))
        ).strip().lower()
        if (not is_paired) and projection_mode in ("equirectangular", "cubemap"):
            logger.warning(f"{stage_label} VO skipped for panorama projection mode (future support)")
            return False, vo_calib, "projection_mode_unsupported"

        return True, vo_calib, "enabled"

    def get_vo_runtime_status(self, is_paired: bool) -> Dict[str, Any]:
        enabled, calib, reason = self._resolve_vo_runtime(is_paired=is_paired, stage_label="Runtime")
        return {
            "enabled": bool(enabled),
            "reason": str(reason),
            "calibration_loaded": bool(calib is not None),
        }

    def _inject_stage0_vo_metrics_into_stage2_records(
        self,
        records: List[Stage2FrameRecord],
        stage0_metrics: Dict[int, Dict[str, Any]],
    ) -> None:
        if not records or not stage0_metrics:
            return
        for record in records:
            m = self._lookup_stage0_metrics(int(record.frame_index), stage0_metrics)
            if "translation_delta" in m:
                record.metrics["translation_delta"] = float(m.get("translation_delta", record.metrics.get("translation_delta", 0.0)))
            if "rotation_delta" in m:
                record.metrics["rotation_delta"] = float(m.get("rotation_delta", record.metrics.get("rotation_delta", 0.0)))
            if "match_count" in m:
                record.metrics["match_count"] = float(m.get("match_count", record.metrics.get("match_count", 0.0)))
            if "vo_step_proxy" in m:
                record.metrics["vo_step_proxy"] = float(m.get("vo_step_proxy", 0.0))
            if "vo_step_proxy_norm" in m:
                record.metrics["vo_step_proxy_norm"] = float(m.get("vo_step_proxy_norm", 0.0))
            if "vo_inlier_ratio" in m:
                record.metrics["vo_inlier_ratio"] = float(m.get("vo_inlier_ratio", 0.0))
            if "vo_rot_deg" in m:
                record.metrics["vo_rot_deg"] = float(m.get("vo_rot_deg", 0.0))
            if "vo_confidence" in m:
                record.metrics["vo_confidence"] = float(m.get("vo_confidence", 0.0))
            if "vo_feature_uniformity" in m:
                record.metrics["vo_feature_uniformity"] = float(m.get("vo_feature_uniformity", 0.0))
            if "vo_track_sufficiency" in m:
                record.metrics["vo_track_sufficiency"] = float(m.get("vo_track_sufficiency", 0.0))
            if "vo_pose_plausibility" in m:
                record.metrics["vo_pose_plausibility"] = float(m.get("vo_pose_plausibility", 0.0))
            if "vo_tracked_count" in m:
                record.metrics["vo_tracked_count"] = float(m.get("vo_tracked_count", 0.0))
            if "vo_essential_method" in m:
                record.metrics["vo_essential_method"] = str(m.get("vo_essential_method", "none"))
            if "vo_status_reason" in m:
                record.metrics["vo_status_reason"] = str(m.get("vo_status_reason", "unknown"))

    def _stage0_lightweight_motion_scan(
        self,
        video_loader,
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[int, int], None]],
        frame_log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        total_frames = max(0, int(metadata.frame_count))
        stride = max(1, int(self.config.get('STAGE0_STRIDE', 5)))
        frame_indices = list(range(0, total_frames, stride))
        if not frame_indices:
            return {}

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired
        use_rig = bool(is_paired and self.config.get('ENABLE_RIG_STITCHING', True))
        out_w = int(self.config.get('EQUIRECT_WIDTH', 4096))
        out_h = int(self.config.get('EQUIRECT_HEIGHT', 2048))
        calibration = getattr(metadata, 'rig_calibration', None)
        vo_enabled, vo_calib, vo_status_reason = self._resolve_vo_runtime(
            is_paired=is_paired,
            stage_label="Stage0",
        )
        vo_subsample = max(1, int(self.config.get('VO_FRAME_SUBSAMPLE', 1)))
        vo_adaptive_subsample = bool(self.config.get('VO_ADAPTIVE_SUBSAMPLE', False))
        vo_subsample_min = int(max(1, min(vo_subsample, int(self.config.get('VO_SUBSAMPLE_MIN', 1)))))
        adaptive_roi_enable = bool(self.config.get('VO_ADAPTIVE_ROI_ENABLE', True))
        roi_min = float(np.clip(self.config.get('VO_ADAPTIVE_ROI_MIN', 0.45), 0.2, 1.0))
        roi_max = float(np.clip(self.config.get('VO_ADAPTIVE_ROI_MAX', 0.70), roi_min, 1.0))
        step_clip = float(max(0.0, self.config.get('VO_STEP_PROXY_CLIP_PX', 80.0)))
        if use_rig and vo_enabled:
            # TODO: panorama/equirect/cubemap VO will be supported in a future milestone.
            logger.warning("Stage0 VO uses representative lens only (stitch/equirect VO is skipped by design)")
        lap_th = max(1.0, float(self.config.get('LAPLACIAN_THRESHOLD', 100.0)))
        flow_norm_factor = max(1e-6, float(self.normalization.OPTICAL_FLOW_NORM_FACTOR))

        metrics: Dict[int, Dict[str, Any]] = {}
        prev_frame = None
        prev_vo_frame = None
        trajectory_samples: List[Dict[str, Any]] = []

        for idx, frame_idx in enumerate(frame_indices):
            if is_paired:
                frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                if frame_a is None or frame_b is None:
                    continue
                frame_vo_cur = frame_a
                if use_rig:
                    frame_cur, _ = self.rig_processor.stitch_to_equirect(
                        frame_a, frame_b, calibration, (out_w, out_h)
                    )
                else:
                    frame_cur = frame_a
            else:
                frame_cur = video_loader.get_frame(frame_idx)
                if frame_cur is None:
                    continue
                frame_vo_cur = frame_cur

            if prev_frame is None:
                flow_mag = 0.0
                ssim_light = 1.0
            else:
                flow_mag = float(self.adaptive_selector.compute_optical_flow_magnitude(prev_frame, frame_cur))
                ssim_light = float(self.adaptive_selector.compute_ssim(prev_frame, frame_cur))

            sharpness = float(self.quality_evaluator.evaluate(
                frame_cur,
                beta=float(self.config.get('SOFTMAX_BETA', 5.0))
            ).get("sharpness", 0.0))
            texture_risk = float(np.clip(1.0 - min(sharpness / lap_th, 1.0), 0.0, 1.0))
            flow_risk = float(np.clip(flow_mag / flow_norm_factor, 0.0, 1.0))
            ssim_change = float(np.clip(1.0 - ssim_light, 0.0, 1.0))
            motion_risk = float(np.clip(0.4 * texture_risk + 0.4 * flow_risk + 0.2 * ssim_change, 0.0, 1.0))
            roi_ratio = float(self.config.get('VO_CENTER_ROI_RATIO', 0.6))
            if adaptive_roi_enable:
                flow_norm = float(np.clip(flow_mag / flow_norm_factor, 0.0, 1.0))
                roi_ratio = float(np.clip(roi_min + (roi_max - roi_min) * flow_norm, roi_min, roi_max))
            flow_norm_for_subsample = float(np.clip(flow_mag / flow_norm_factor, 0.0, 1.0))
            if vo_adaptive_subsample and vo_subsample > vo_subsample_min:
                effective_subsample = int(round(vo_subsample_min + (1.0 - flow_norm_for_subsample) * (vo_subsample - vo_subsample_min)))
            else:
                effective_subsample = int(vo_subsample)
            effective_subsample = max(1, effective_subsample)
            vo_should_run = bool(vo_enabled and prev_vo_frame is not None and (idx % effective_subsample == 0))
            vo = self.vo_estimator.estimate(
                prev_vo_frame,
                frame_vo_cur,
                calibration=vo_calib,
                center_roi_ratio=roi_ratio,
            ) if vo_should_run else None
            vo_valid = bool(vo.vo_valid) if vo is not None else False
            step_proxy = float(vo.step_proxy if vo else 0.0)
            if step_clip > 0.0:
                step_proxy = min(step_proxy, step_clip)
            t_dir = [float(x) for x in getattr(vo, "t_dir", [0.0, 0.0, 0.0])] if vo_valid else [0.0, 0.0, 0.0]
            r_rel_q = [float(x) for x in getattr(vo, "r_rel_q_wxyz", [1.0, 0.0, 0.0, 0.0])] if vo_valid else [1.0, 0.0, 0.0, 0.0]
            vo_confidence = float(vo.vo_confidence if vo else 0.0)
            vo_feature_uniformity = float(vo.feature_uniformity if vo else 0.0)
            vo_track_sufficiency = float(vo.track_sufficiency if vo else 0.0)
            vo_pose_plausibility = float(vo.pose_plausibility if vo else 0.0)
            vo_tracked_count = float(vo.tracked_count if vo else 0.0)
            vo_essential_method = str(vo.essential_method_used if vo else "none")
            trajectory_samples.append(
                {
                    "frame_idx": int(frame_idx),
                    "vo_valid": bool(vo_valid),
                    "t_dir": t_dir,
                    "step_proxy": float(step_proxy),
                    "r_rel_q_wxyz": r_rel_q,
                }
            )

            metrics[int(frame_idx)] = {
                "flow_mag_light": flow_mag,
                "ssim_light": ssim_light,
                "motion_risk": motion_risk,
                "rotation_delta": float(vo.rotation_delta_deg if vo else 0.0),
                "translation_delta": float(vo.translation_delta_rel if vo else 0.0),
                "vo_step_proxy": step_proxy,
                "vo_step_proxy_norm": 0.0,
                "vo_rot_deg": float(vo.rotation_delta_deg if vo else 0.0),
                "match_count": float(vo.match_count if vo else 0.0),
                "vo_inlier_ratio": float(vo.inlier_ratio if vo else 0.0),
                "vo_confidence": vo_confidence,
                "vo_feature_uniformity": vo_feature_uniformity,
                "vo_track_sufficiency": vo_track_sufficiency,
                "vo_pose_plausibility": vo_pose_plausibility,
                "vo_tracked_count": vo_tracked_count,
                "vo_essential_method": vo_essential_method,
                "vo_valid": 1.0 if vo_valid else 0.0,
                "vo_attempted": 1.0 if vo_should_run else 0.0,
                "vo_pose_valid": 0.0,
                "vo_t_dir": t_dir,
                "vo_r_rel_q_wxyz": r_rel_q,
                "vo_effective_subsample": float(effective_subsample),
                "vo_status_reason": (
                    "frame_subsample_skip" if (vo_enabled and prev_vo_frame is not None and not vo_should_run)
                    else ("enabled" if vo_valid else ("estimate_failed_or_low_inlier" if vo_should_run else vo_status_reason))
                ),
            }
            prev_frame = frame_cur
            prev_vo_frame = frame_vo_cur

            if progress_callback:
                progress_callback(idx + 1, len(frame_indices))

        step_values = [
            float(v.get("vo_step_proxy", 0.0))
            for v in metrics.values()
            if float(v.get("vo_valid", 0.0)) > 0.5 and float(v.get("vo_step_proxy", 0.0)) > 0.0
        ]
        step_median = float(np.median(np.asarray(step_values, dtype=np.float64))) if step_values else 1.0
        if step_median <= 1e-12:
            step_median = 1.0
        for m in metrics.values():
            step = float(m.get("vo_step_proxy", 0.0))
            m["vo_step_proxy_norm"] = float(step / step_median) if step > 0.0 else 0.0

        trajectory_map = integrate_relative_trajectory(
            trajectory_samples,
            t_sign=float(self.config.get("VO_T_SIGN", 1.0)),
        )
        for frame_idx, m in metrics.items():
            pose = trajectory_map.get(int(frame_idx))
            if pose is None:
                m["t_xyz"] = None
                m["q_wxyz"] = None
                m["vo_pose_valid"] = 0.0
                continue
            m["t_xyz"] = [float(x) for x in pose.get("t_xyz", [0.0, 0.0, 0.0])]
            m["q_wxyz"] = [float(x) for x in pose.get("q_wxyz", [1.0, 0.0, 0.0, 0.0])]
            m["vo_pose_valid"] = 1.0

        if frame_log_callback:
            for frame_idx in sorted(metrics.keys()):
                m = metrics[int(frame_idx)]
                self._emit_frame_log(
                    frame_log_callback,
                    {
                        "frame_index": int(frame_idx),
                        "frame": None,
                        "is_keyframe": False,
                        "metrics": dict(m),
                        "quality_scores": {},
                        "geometric_scores": {},
                        "adaptive_scores": {},
                        "t_xyz": m.get("t_xyz"),
                        "q_wxyz": m.get("q_wxyz"),
                        "points_world": None,
                        "stage_label": "Stage0",
                    },
                )

        return metrics

    def _stage3_refine_with_trajectory(
        self,
        metadata: VideoMetadata,
        stage2_candidates: List[KeyframeInfo],
        stage2_final: List[KeyframeInfo],
        stage2_records: List[Stage2FrameRecord],
        stage0_metrics: Dict[int, Dict[str, Any]],
        video_loader=None,
    ) -> List[KeyframeInfo]:
        if not stage2_candidates:
            return stage2_candidates

        if video_loader is None:
            logger.warning("Stage3をスキップ: video_loaderが未指定")
            return stage2_candidates

        tracked_set = {int(kf.frame_index) for kf in stage2_candidates}
        tracked_set.update(int(kf.frame_index) for kf in stage2_final)
        tracked_indices = sorted(tracked_set)
        if len(tracked_indices) < 2:
            for kf in stage2_candidates:
                base = float(kf.combined_score)
                risk = float(np.clip(self._lookup_stage0_metrics(kf.frame_index, stage0_metrics).get("motion_risk", 0.0), 0.0, 1.0))
                vo_conf = float(np.clip(self._lookup_stage0_metrics(kf.frame_index, stage0_metrics).get("vo_confidence", 0.0), 0.0, 1.0))
                kf.stage3_scores = {
                    "trajectory_consistency": 0.5,
                    "trajectory_consistency_effective": float(0.5 * (0.5 + 0.5 * vo_conf)),
                    "vo_confidence": vo_conf,
                    "stage0_motion_risk": risk,
                    "combined_stage2": base,
                    "combined_stage3": base,
                }
            return stage2_candidates

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired
        vo_enabled, vo_calib, vo_status_reason = self._resolve_vo_runtime(
            is_paired=is_paired,
            stage_label="Stage3",
        )
        vo_subsample = max(1, int(self.config.get('VO_FRAME_SUBSAMPLE', 1)))
        vo_adaptive_subsample = bool(self.config.get('VO_ADAPTIVE_SUBSAMPLE', False))
        vo_subsample_min = int(max(1, min(vo_subsample, int(self.config.get('VO_SUBSAMPLE_MIN', 1)))))
        step_clip = float(max(0.0, self.config.get('VO_STEP_PROXY_CLIP_PX', 80.0)))
        flow_norm_factor = max(1e-6, float(self.normalization.OPTICAL_FLOW_NORM_FACTOR))

        frame_cache: Dict[int, Optional[np.ndarray]] = {}

        def _get_eval_frame(frame_idx: int) -> Optional[np.ndarray]:
            if frame_idx in frame_cache:
                return frame_cache[frame_idx]
            if is_paired:
                frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                if frame_a is None or frame_b is None:
                    frame_cache[frame_idx] = None
                    return None
                frame_cache[frame_idx] = frame_a
                return frame_cache[frame_idx]
            frame = video_loader.get_frame(frame_idx)
            frame_cache[frame_idx] = frame
            return frame

        trajectory_consistency: Dict[int, float] = {tracked_indices[0]: 0.5}
        vo_step_proxy_norm: Dict[int, float] = {tracked_indices[0]: 0.0}
        vo_dir_cos_prev: Dict[int, float] = {tracked_indices[0]: 0.5}
        vo_rot_deg: Dict[int, float] = {tracked_indices[0]: 0.0}
        trajectory_samples: List[Dict[str, Any]] = [
            {
                "frame_idx": int(tracked_indices[0]),
                "vo_valid": False,
                "t_dir": [0.0, 0.0, 0.0],
                "step_proxy": 0.0,
                "r_rel_q_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        ]
        per_frame_vo: Dict[int, Dict[str, Any]] = {
            int(tracked_indices[0]): {
                "vo_valid": 0.0, "step_proxy": 0.0, "inlier_ratio": 0.0, "rot_deg": 0.0,
                "vo_status_reason": "init",
                "vo_attempted": 0.0,
                "confidence": float(np.clip(self._lookup_stage0_metrics(int(tracked_indices[0]), stage0_metrics).get("vo_confidence", 0.0), 0.0, 1.0)),
                "feature_uniformity": 0.0,
                "track_sufficiency": 0.0,
                "pose_plausibility": 0.0,
                "tracked_count": 0.0,
                "essential_method": "none",
            }
        }
        step_proxy_raw: Dict[int, float] = {tracked_indices[0]: 0.0}
        valid_count = 0
        prev_rot = None
        prev_step = None
        prev_dir = None
        prev_valid = False
        for i in range(1, len(tracked_indices)):
            if not vo_enabled:
                cur_idx = tracked_indices[i]
                trajectory_consistency[cur_idx] = 0.5
                vo_step_proxy_norm[cur_idx] = 0.0
                vo_dir_cos_prev[cur_idx] = 0.5
                vo_rot_deg[cur_idx] = 0.0
                trajectory_samples.append(
                    {
                        "frame_idx": int(cur_idx),
                        "vo_valid": False,
                        "t_dir": [0.0, 0.0, 0.0],
                        "step_proxy": 0.0,
                        "r_rel_q_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                )
                per_frame_vo[int(cur_idx)] = {"vo_valid": 0.0, "step_proxy": 0.0, "inlier_ratio": 0.0, "rot_deg": 0.0}
                per_frame_vo[int(cur_idx)]["vo_status_reason"] = vo_status_reason
                per_frame_vo[int(cur_idx)]["vo_attempted"] = 0.0
                per_frame_vo[int(cur_idx)]["confidence"] = float(np.clip(self._lookup_stage0_metrics(int(cur_idx), stage0_metrics).get("vo_confidence", 0.0), 0.0, 1.0))
                per_frame_vo[int(cur_idx)]["feature_uniformity"] = 0.0
                per_frame_vo[int(cur_idx)]["track_sufficiency"] = 0.0
                per_frame_vo[int(cur_idx)]["pose_plausibility"] = 0.0
                per_frame_vo[int(cur_idx)]["tracked_count"] = 0.0
                per_frame_vo[int(cur_idx)]["essential_method"] = "none"
                step_proxy_raw[int(cur_idx)] = 0.0
                continue
            prev_idx = tracked_indices[i - 1]
            cur_idx = tracked_indices[i]
            prev_frame = _get_eval_frame(prev_idx)
            cur_frame = _get_eval_frame(cur_idx)
            if prev_frame is None or cur_frame is None:
                trajectory_consistency[cur_idx] = 0.0
                vo_step_proxy_norm[cur_idx] = 0.0
                vo_dir_cos_prev[cur_idx] = 0.5
                vo_rot_deg[cur_idx] = 0.0
                trajectory_samples.append(
                    {
                        "frame_idx": int(cur_idx),
                        "vo_valid": False,
                        "t_dir": [0.0, 0.0, 0.0],
                        "step_proxy": 0.0,
                        "r_rel_q_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                )
                per_frame_vo[int(cur_idx)] = {"vo_valid": 0.0, "step_proxy": 0.0, "inlier_ratio": 0.0, "rot_deg": 0.0}
                per_frame_vo[int(cur_idx)]["vo_status_reason"] = "frame_unavailable"
                per_frame_vo[int(cur_idx)]["vo_attempted"] = 0.0
                per_frame_vo[int(cur_idx)]["confidence"] = float(np.clip(self._lookup_stage0_metrics(int(cur_idx), stage0_metrics).get("vo_confidence", 0.0), 0.0, 1.0))
                per_frame_vo[int(cur_idx)]["feature_uniformity"] = 0.0
                per_frame_vo[int(cur_idx)]["track_sufficiency"] = 0.0
                per_frame_vo[int(cur_idx)]["pose_plausibility"] = 0.0
                per_frame_vo[int(cur_idx)]["tracked_count"] = 0.0
                per_frame_vo[int(cur_idx)]["essential_method"] = "none"
                step_proxy_raw[int(cur_idx)] = 0.0
                prev_valid = False
                continue

            stage0_cur = self._lookup_stage0_metrics(int(cur_idx), stage0_metrics)
            flow_light = float(stage0_cur.get("flow_mag_light", np.nan))
            if np.isfinite(flow_light):
                motion_norm = float(np.clip(flow_light / flow_norm_factor, 0.0, 1.0))
            else:
                motion_norm = float(np.clip(stage0_cur.get("motion_risk", 0.0), 0.0, 1.0))
            if vo_adaptive_subsample and vo_subsample > vo_subsample_min:
                effective_subsample = int(round(vo_subsample_min + (1.0 - motion_norm) * (vo_subsample - vo_subsample_min)))
            else:
                effective_subsample = int(vo_subsample)
            effective_subsample = max(1, effective_subsample)
            vo_should_run = bool(vo_enabled and (i % effective_subsample == 0))
            vo = self.vo_estimator.estimate(
                prev_frame,
                cur_frame,
                calibration=vo_calib,
                center_roi_ratio=float(self.config.get('VO_CENTER_ROI_RATIO', 0.6)),
            ) if vo_should_run else None
            if not vo_should_run:
                trajectory_consistency[cur_idx] = 0.0
                vo_step_proxy_norm[cur_idx] = 0.0
                vo_dir_cos_prev[cur_idx] = 0.5
                vo_rot_deg[cur_idx] = 0.0
                trajectory_samples.append(
                    {
                        "frame_idx": int(cur_idx),
                        "vo_valid": False,
                        "t_dir": [0.0, 0.0, 0.0],
                        "step_proxy": 0.0,
                        "r_rel_q_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                )
                per_frame_vo[int(cur_idx)] = {
                    "vo_valid": 0.0, "step_proxy": 0.0, "inlier_ratio": 0.0, "rot_deg": 0.0,
                    "vo_status_reason": "frame_subsample_skip",
                    "vo_attempted": 0.0,
                    "confidence": float(np.clip(stage0_cur.get("vo_confidence", 0.0), 0.0, 1.0)),
                    "feature_uniformity": 0.0,
                    "track_sufficiency": 0.0,
                    "pose_plausibility": 0.0,
                    "tracked_count": 0.0,
                    "essential_method": "none",
                }
                step_proxy_raw[int(cur_idx)] = 0.0
                prev_valid = False
                continue
            if vo is None or not vo.vo_valid:
                trajectory_consistency[cur_idx] = 0.0
                vo_step_proxy_norm[cur_idx] = 0.0
                vo_dir_cos_prev[cur_idx] = 0.5
                vo_rot_deg[cur_idx] = 0.0
                trajectory_samples.append(
                    {
                        "frame_idx": int(cur_idx),
                        "vo_valid": False,
                        "t_dir": [0.0, 0.0, 0.0],
                        "step_proxy": 0.0,
                        "r_rel_q_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                )
                per_frame_vo[int(cur_idx)] = {
                    "vo_valid": 0.0, "step_proxy": 0.0, "inlier_ratio": 0.0, "rot_deg": 0.0,
                    "vo_status_reason": "estimate_failed_or_low_inlier",
                    "vo_attempted": 1.0,
                    "confidence": float(np.clip(stage0_cur.get("vo_confidence", 0.0), 0.0, 1.0)),
                    "feature_uniformity": 0.0,
                    "track_sufficiency": 0.0,
                    "pose_plausibility": 0.0,
                    "tracked_count": 0.0,
                    "essential_method": "none",
                }
                step_proxy_raw[int(cur_idx)] = 0.0
                prev_valid = False
                continue

            step_val = float(max(0.0, getattr(vo, "step_proxy", 0.0)))
            if step_clip > 0.0:
                step_val = min(step_val, step_clip)
            dir_vec = np.asarray(getattr(vo, "t_dir", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            dir_norm = float(np.linalg.norm(dir_vec))
            if dir_norm > 1e-12:
                dir_vec = dir_vec / dir_norm
            else:
                dir_vec = np.zeros(3, dtype=np.float64)
            valid_count += 1
            valid_ratio = float(np.clip(valid_count / float(i), 0.0, 1.0))
            if not prev_valid or prev_rot is None or prev_step is None or prev_dir is None:
                dir_term = 0.5
                rot_term = 0.5
                step_term = 0.5
            else:
                dot_dir = float(np.clip(float(np.dot(dir_vec, prev_dir)), -1.0, 1.0))
                dir_term = float(np.clip((dot_dir + 1.0) * 0.5, 0.0, 1.0))
                rot_term = float(np.clip(1.0 - abs(float(vo.rotation_delta_deg) - float(prev_rot)) / 15.0, 0.0, 1.0))
                if step_val > 1e-12 and prev_step > 1e-12:
                    step_term = float(
                        np.clip(1.0 - abs(np.log(step_val) - np.log(float(prev_step))) / 1.0, 0.0, 1.0)
                    )
                else:
                    step_term = 0.5

            traj = float(np.clip(0.40 * valid_ratio + 0.20 * dir_term + 0.20 * rot_term + 0.20 * step_term, 0.0, 1.0))
            trajectory_consistency[cur_idx] = traj
            vo_step_proxy_norm[cur_idx] = 0.0
            vo_dir_cos_prev[cur_idx] = float(dir_term)
            vo_rot_deg[cur_idx] = float(vo.rotation_delta_deg)
            trajectory_samples.append(
                {
                    "frame_idx": int(cur_idx),
                    "vo_valid": True,
                    "t_dir": [float(x) for x in dir_vec.tolist()],
                    "step_proxy": float(step_val),
                    "r_rel_q_wxyz": [float(x) for x in getattr(vo, "r_rel_q_wxyz", [1.0, 0.0, 0.0, 0.0])],
                }
            )
            per_frame_vo[int(cur_idx)] = {
                "vo_valid": 1.0,
                "step_proxy": float(step_val),
                "inlier_ratio": float(getattr(vo, "inlier_ratio", 0.0)),
                "rot_deg": float(vo.rotation_delta_deg),
                "vo_status_reason": "enabled",
                "vo_attempted": 1.0,
                "confidence": float(getattr(vo, "vo_confidence", 0.0)),
                "feature_uniformity": float(getattr(vo, "feature_uniformity", 0.0)),
                "track_sufficiency": float(getattr(vo, "track_sufficiency", 0.0)),
                "pose_plausibility": float(getattr(vo, "pose_plausibility", 0.0)),
                "tracked_count": float(getattr(vo, "tracked_count", 0.0)),
                "essential_method": str(getattr(vo, "essential_method_used", "none")),
            }
            step_proxy_raw[int(cur_idx)] = float(step_val)
            prev_rot = float(vo.rotation_delta_deg)
            prev_step = float(step_val)
            prev_dir = dir_vec.copy()
            prev_valid = True

        step_values = [v for v in step_proxy_raw.values() if v > 0.0]
        step_median = float(np.median(np.asarray(step_values, dtype=np.float64))) if step_values else 1.0
        if step_median <= 1e-12:
            step_median = 1.0
        for idx in tracked_indices:
            raw = float(step_proxy_raw.get(int(idx), 0.0))
            vo_step_proxy_norm[int(idx)] = float(raw / step_median) if raw > 0.0 else 0.0

        trajectory_map = integrate_relative_trajectory(
            trajectory_samples,
            t_sign=float(self.config.get("VO_T_SIGN", 1.0)),
        )

        w_base = float(self.config.get('STAGE3_WEIGHT_BASE', 0.70))
        w_traj = float(self.config.get('STAGE3_WEIGHT_TRAJECTORY', 0.25))
        w_risk = float(self.config.get('STAGE3_WEIGHT_STAGE0_RISK', 0.05))
        disable_when_unreliable = bool(self.config.get("STAGE3_DISABLE_TRAJ_WHEN_VO_UNRELIABLE", True))
        min_vo_valid_ratio = float(np.clip(self.config.get("STAGE3_VO_VALID_RATIO_THRESHOLD", 0.50), 0.0, 1.0))
        vo_attempted = int(sum(1 for v in per_frame_vo.values() if float(v.get("vo_attempted", 0.0)) > 0.5))
        vo_valid = int(sum(1 for v in per_frame_vo.values() if float(v.get("vo_valid", 0.0)) > 0.5))
        vo_valid_ratio = float(vo_valid / max(vo_attempted, 1)) if vo_attempted > 0 else 0.0
        if disable_when_unreliable and vo_attempted > 0 and vo_valid_ratio < min_vo_valid_ratio and w_traj > 0.0:
            logger.warning(
                "Stage3: VO有効率が低いため trajectory 重みを無効化 "
                f"(valid_ratio={vo_valid_ratio:.3f}, threshold={min_vo_valid_ratio:.3f})"
            )
            w_base += w_traj
            w_traj = 0.0

        record_map = {int(r.frame_index): r for r in stage2_records}
        for kf in stage2_candidates:
            base_score = float(kf.combined_score)
            traj = float(np.clip(trajectory_consistency.get(int(kf.frame_index), 0.5), 0.0, 1.0))
            vo_info = per_frame_vo.get(int(kf.frame_index), {})
            vo_conf = float(
                np.clip(
                    vo_info.get(
                        "confidence",
                        self._lookup_stage0_metrics(int(kf.frame_index), stage0_metrics).get("vo_confidence", 0.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            traj_effective = float(np.clip(traj * (0.5 + 0.5 * vo_conf), 0.0, 1.0))
            risk = float(np.clip(
                self._lookup_stage0_metrics(int(kf.frame_index), stage0_metrics).get("motion_risk", 0.0),
                0.0, 1.0
            ))
            combined_stage3 = float(np.clip(w_base * base_score + w_traj * traj_effective - w_risk * risk, 0.0, 1.0))
            kf.stage3_scores = {
                "trajectory_consistency": traj,
                "trajectory_consistency_effective": traj_effective,
                "vo_confidence": vo_conf,
                "stage0_motion_risk": risk,
                "combined_stage2": base_score,
                "combined_stage3": combined_stage3,
                "stage3_vo_valid_ratio": vo_valid_ratio,
                "stage3_w_base": w_base,
                "stage3_w_traj": w_traj,
            }
            kf.combined_score = combined_stage3

            record = record_map.get(int(kf.frame_index))
            if record is not None:
                record.metrics["trajectory_consistency"] = traj
                record.metrics["trajectory_consistency_effective"] = traj_effective
                record.metrics["vo_confidence"] = vo_conf
                record.metrics["combined_stage2"] = base_score
                record.metrics["combined_stage3"] = combined_stage3
                record.metrics["stage0_motion_risk"] = risk
                record.metrics["stage3_vo_valid_ratio"] = vo_valid_ratio
                record.metrics["stage3_w_base"] = w_base
                record.metrics["stage3_w_traj"] = w_traj

        pose_keys = sorted(int(k) for k in trajectory_map.keys())
        for idx, record in record_map.items():
            pose = trajectory_map.get(int(idx))
            if pose is None and pose_keys:
                insert = int(np.searchsorted(pose_keys, int(idx)))
                lookup_idx = pose_keys[max(0, min(len(pose_keys) - 1, insert))]
                if insert > 0 and insert < len(pose_keys):
                    left = pose_keys[insert - 1]
                    right = pose_keys[insert]
                    lookup_idx = left if abs(int(idx) - left) <= abs(int(idx) - right) else right
                pose = trajectory_map.get(int(lookup_idx))
            if pose is not None:
                record.t_xyz = [float(x) for x in pose.get("t_xyz", [0.0, 0.0, 0.0])]
                record.q_wxyz = [float(x) for x in pose.get("q_wxyz", [1.0, 0.0, 0.0, 0.0])]
            pose_direct = bool(int(idx) in trajectory_map)
            record.metrics["vo_pose_valid"] = 1.0 if pose_direct else 0.0
            vo_info = per_frame_vo.get(
                int(idx),
                {
                    "vo_valid": 0.0,
                    "step_proxy": 0.0,
                    "inlier_ratio": 0.0,
                    "rot_deg": 0.0,
                    "confidence": 0.0,
                    "feature_uniformity": 0.0,
                    "track_sufficiency": 0.0,
                    "pose_plausibility": 0.0,
                    "tracked_count": 0.0,
                    "essential_method": "none",
                },
            )
            record.metrics["vo_valid"] = float(vo_info.get("vo_valid", record.metrics.get("vo_valid", 0.0)))
            record.metrics["vo_inlier_ratio"] = float(vo_info.get("inlier_ratio", record.metrics.get("vo_inlier_ratio", 0.0)))
            record.metrics["vo_step_proxy"] = float(vo_info.get("step_proxy", 0.0))
            record.metrics["vo_step_proxy_norm"] = float(vo_step_proxy_norm.get(int(idx), 0.0))
            record.metrics["vo_dir_cos_prev"] = float(vo_dir_cos_prev.get(int(idx), 0.5))
            record.metrics["vo_rot_deg"] = float(vo_rot_deg.get(int(idx), vo_info.get("rot_deg", 0.0)))
            record.metrics["vo_confidence"] = float(np.clip(vo_info.get("confidence", record.metrics.get("vo_confidence", 0.0)), 0.0, 1.0))
            record.metrics["vo_feature_uniformity"] = float(vo_info.get("feature_uniformity", 0.0))
            record.metrics["vo_track_sufficiency"] = float(vo_info.get("track_sufficiency", 0.0))
            record.metrics["vo_pose_plausibility"] = float(vo_info.get("pose_plausibility", 0.0))
            record.metrics["vo_tracked_count"] = float(vo_info.get("tracked_count", 0.0))
            record.metrics["vo_essential_method"] = str(vo_info.get("essential_method", "none"))
            record.metrics["vo_status_reason"] = str(vo_info.get("vo_status_reason", vo_status_reason))
            record.metrics["vo_attempted"] = float(vo_info.get("vo_attempted", 0.0))

        return stage2_candidates

    @staticmethod
    def _serialize_keyframe_info(candidate: KeyframeInfo) -> Dict[str, Any]:
        return {
            "frame_index": int(candidate.frame_index),
            "timestamp": float(candidate.timestamp),
            "quality_scores": dict(candidate.quality_scores or {}),
            "geometric_scores": dict(candidate.geometric_scores or {}),
            "adaptive_scores": dict(candidate.adaptive_scores or {}),
            "combined_score": float(candidate.combined_score),
            "is_rescue_mode": bool(candidate.is_rescue_mode),
            "is_force_inserted": bool(candidate.is_force_inserted),
            "is_stationary": bool(candidate.is_stationary),
            "stationary_penalty_applied": bool(candidate.stationary_penalty_applied),
            "stage3_scores": dict(candidate.stage3_scores or {}),
        }

    @staticmethod
    def _deserialize_keyframe_info(row: Dict[str, Any]) -> KeyframeInfo:
        return KeyframeInfo(
            frame_index=int(row.get("frame_index", 0)),
            timestamp=float(row.get("timestamp", 0.0)),
            quality_scores=dict(row.get("quality_scores", {}) or {}),
            geometric_scores=dict(row.get("geometric_scores", {}) or {}),
            adaptive_scores=dict(row.get("adaptive_scores", {}) or {}),
            combined_score=float(row.get("combined_score", 0.0)),
            thumbnail=None,
            is_rescue_mode=bool(row.get("is_rescue_mode", False)),
            is_force_inserted=bool(row.get("is_force_inserted", False)),
            dynamic_mask=None,
            is_stationary=bool(row.get("is_stationary", False)),
            stationary_penalty_applied=bool(row.get("stationary_penalty_applied", False)),
            stage3_scores=dict(row.get("stage3_scores", {}) or {}),
        )

    @staticmethod
    def _serialize_stage2_record(record: Stage2FrameRecord) -> Dict[str, Any]:
        return {
            "frame_index": int(record.frame_index),
            "quality_scores": dict(record.quality_scores or {}),
            "geometric_scores": dict(record.geometric_scores or {}),
            "adaptive_scores": dict(record.adaptive_scores or {}),
            "metrics": dict(record.metrics or {}),
            "is_candidate": bool(record.is_candidate),
            "is_keyframe": bool(record.is_keyframe),
            "drop_reason": str(record.drop_reason or ""),
            "t_xyz": list(record.t_xyz) if record.t_xyz is not None else None,
            "q_wxyz": list(record.q_wxyz) if record.q_wxyz is not None else None,
        }

    @staticmethod
    def _deserialize_stage2_record(row: Dict[str, Any]) -> Stage2FrameRecord:
        t_xyz_raw = row.get("t_xyz")
        q_wxyz_raw = row.get("q_wxyz")
        t_xyz = [float(v) for v in t_xyz_raw] if isinstance(t_xyz_raw, (list, tuple)) and len(t_xyz_raw) == 3 else None
        q_wxyz = [float(v) for v in q_wxyz_raw] if isinstance(q_wxyz_raw, (list, tuple)) and len(q_wxyz_raw) == 4 else None
        return Stage2FrameRecord(
            frame_index=int(row.get("frame_index", 0)),
            frame=None,
            quality_scores=dict(row.get("quality_scores", {}) or {}),
            geometric_scores=dict(row.get("geometric_scores", {}) or {}),
            adaptive_scores=dict(row.get("adaptive_scores", {}) or {}),
            metrics=dict(row.get("metrics", {}) or {}),
            is_candidate=bool(row.get("is_candidate", False)),
            is_keyframe=bool(row.get("is_keyframe", False)),
            drop_reason=str(row.get("drop_reason", "")),
            t_xyz=t_xyz,
            q_wxyz=q_wxyz,
            points_world=None,
        )

    def _resolve_colmap_keyframe_runtime(self) -> Dict[str, Any]:
        pose_backend = str(
            self.config.get("POSE_BACKEND", self.config.get("pose_backend", "vo")) or "vo"
        ).strip().lower()
        if pose_backend not in {"vo", "colmap"}:
            pose_backend = "vo"

        raw_pipeline_mode = str(
            self.config.get("COLMAP_PIPELINE_MODE", self.config.get("colmap_pipeline_mode", "")) or ""
        ).strip().lower()
        if raw_pipeline_mode not in {"", "legacy", "minimal_v1"}:
            raw_pipeline_mode = ""
        pipeline_mode = raw_pipeline_mode if raw_pipeline_mode else ("minimal_v1" if pose_backend == "colmap" else "legacy")
        minimal_mode = bool(pose_backend == "colmap" and pipeline_mode == "minimal_v1")

        raw_policy = str(
            self.config.get("COLMAP_KEYFRAME_POLICY", self.config.get("colmap_keyframe_policy", "")) or ""
        ).strip().lower()
        if raw_policy not in {"", "legacy", "stage2_relaxed", "stage1_only"}:
            raw_policy = ""
        policy = raw_policy if raw_policy else ("stage2_relaxed" if pose_backend == "colmap" else "legacy")

        raw_profile = str(
            self.config.get("COLMAP_SELECTION_PROFILE", self.config.get("colmap_selection_profile", "")) or ""
        ).strip().lower()
        if raw_profile not in {"", "legacy", "no_vo_coverage"}:
            raw_profile = ""
        selection_profile = raw_profile if raw_profile else ("no_vo_coverage" if pose_backend == "colmap" else "legacy")

        raw_target_mode = str(
            self.config.get("COLMAP_KEYFRAME_TARGET_MODE", self.config.get("colmap_keyframe_target_mode", "")) or ""
        ).strip().lower()
        if raw_target_mode not in {"", "fixed", "auto"}:
            raw_target_mode = ""
        target_mode = raw_target_mode if raw_target_mode else ("auto" if pose_backend == "colmap" else "fixed")

        target_min = int(max(1, self.config.get("COLMAP_KEYFRAME_TARGET_MIN", self.config.get("colmap_keyframe_target_min", 120))))
        target_max = int(
            max(
                target_min,
                self.config.get("COLMAP_KEYFRAME_TARGET_MAX", self.config.get("colmap_keyframe_target_max", 240)),
            )
        )
        colmap_nms_window = float(
            max(0.01, self.config.get("COLMAP_NMS_WINDOW_SEC", self.config.get("colmap_nms_window_sec", 0.35)))
        )
        colmap_enable_stage0 = bool(
            self.config.get("COLMAP_ENABLE_STAGE0", self.config.get("colmap_enable_stage0", True))
        )
        colmap_motion_aware_selection = bool(
            self.config.get(
                "COLMAP_MOTION_AWARE_SELECTION",
                self.config.get("colmap_motion_aware_selection", True),
            )
        )
        if selection_profile == "no_vo_coverage" or minimal_mode:
            colmap_motion_aware_selection = False
        colmap_nms_motion_window_ratio = float(
            max(
                0.0,
                self.config.get(
                    "COLMAP_NMS_MOTION_WINDOW_RATIO",
                    self.config.get("colmap_nms_motion_window_ratio", 0.5),
                ),
            )
        )
        colmap_stage1_adaptive_threshold = bool(
            self.config.get(
                "COLMAP_STAGE1_ADAPTIVE_THRESHOLD",
                self.config.get("colmap_stage1_adaptive_threshold", True),
            )
        )
        colmap_stage1_min_candidates_per_bin = int(
            max(
                0,
                self.config.get(
                    "COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN",
                    self.config.get("colmap_stage1_min_candidates_per_bin", 3),
                ),
            )
        )
        colmap_stage1_max_candidates = int(
            max(
                1,
                self.config.get(
                    "COLMAP_STAGE1_MAX_CANDIDATES",
                    self.config.get("colmap_stage1_max_candidates", 360),
                ),
            )
        )
        colmap_stage2_entry_budget = int(
            max(
                1,
                self.config.get(
                    "COLMAP_STAGE2_ENTRY_BUDGET",
                    self.config.get("colmap_stage2_entry_budget", 180),
                ),
            )
        )
        colmap_stage2_entry_min_gap = int(
            max(
                0,
                self.config.get(
                    "COLMAP_STAGE2_ENTRY_MIN_GAP",
                    self.config.get("colmap_stage2_entry_min_gap", 3),
                ),
            )
        )
        colmap_diversity_ssim_threshold = float(
            np.clip(
                self.config.get(
                    "COLMAP_DIVERSITY_SSIM_THRESHOLD",
                    self.config.get("colmap_diversity_ssim_threshold", 0.93),
                ),
                0.0,
                1.0,
            )
        )
        colmap_diversity_phash_hamming = int(
            max(
                0,
                self.config.get(
                    "COLMAP_DIVERSITY_PHASH_HAMMING",
                    self.config.get("colmap_diversity_phash_hamming", 10),
                ),
            )
        )
        colmap_final_target_policy = str(
            self.config.get(
                "COLMAP_FINAL_TARGET_POLICY",
                self.config.get("colmap_final_target_policy", "soft_auto"),
            ) or "soft_auto"
        ).strip().lower()
        if colmap_final_target_policy not in {"soft_auto", "fixed"}:
            colmap_final_target_policy = "soft_auto"
        colmap_final_soft_min = int(
            max(
                1,
                self.config.get(
                    "COLMAP_FINAL_SOFT_MIN",
                    self.config.get("colmap_final_soft_min", 80),
                ),
            )
        )
        colmap_final_soft_max = int(
            max(
                colmap_final_soft_min,
                self.config.get(
                    "COLMAP_FINAL_SOFT_MAX",
                    self.config.get("colmap_final_soft_max", 220),
                ),
            )
        )
        colmap_no_supplement_on_low_quality = bool(
            self.config.get(
                "COLMAP_NO_SUPPLEMENT_ON_LOW_QUALITY",
                self.config.get("colmap_no_supplement_on_low_quality", True),
            )
        )

        rig_policy = str(
            self.config.get("COLMAP_RIG_POLICY", self.config.get("colmap_rig_policy", ""))
            or ""
        ).strip().lower()
        if rig_policy not in {"off", "lr_opk"}:
            rig_policy = "lr_opk" if pose_backend == "colmap" else "off"
        rig_seed_raw = self.config.get(
            "COLMAP_RIG_SEED_OPK_DEG",
            self.config.get("colmap_rig_seed_opk_deg", [0.0, 0.0, 180.0]),
        )
        if isinstance(rig_seed_raw, str):
            rig_seed_raw = [v.strip() for v in rig_seed_raw.split(",") if v.strip()]
        if not isinstance(rig_seed_raw, (list, tuple)) or len(rig_seed_raw) != 3:
            rig_seed_raw = [0.0, 0.0, 180.0]
        try:
            rig_seed_opk = [float(rig_seed_raw[0]), float(rig_seed_raw[1]), float(rig_seed_raw[2])]
        except (TypeError, ValueError):
            rig_seed_opk = [0.0, 0.0, 180.0]
        workspace_scope = str(
            self.config.get("COLMAP_WORKSPACE_SCOPE", self.config.get("colmap_workspace_scope", ""))
            or ""
        ).strip().lower()
        if workspace_scope not in {"shared", "run_scoped"}:
            workspace_scope = "run_scoped"
        reuse_db = bool(self.config.get("COLMAP_REUSE_DB", self.config.get("colmap_reuse_db", False)))
        analysis_mask_profile = str(
            self.config.get("COLMAP_ANALYSIS_MASK_PROFILE", self.config.get("colmap_analysis_mask_profile", ""))
            or ""
        ).strip().lower()
        if analysis_mask_profile not in {"legacy", "colmap_safe"}:
            analysis_mask_profile = "colmap_safe" if pose_backend == "colmap" else "legacy"
        default_nms_window = float(
            max(0.01, self.config.get("NMS_TIME_WINDOW", self.config.get("nms_time_window", 1.0)))
        )

        colmap_shortcut = bool(pose_backend == "colmap" and policy != "legacy" and (not minimal_mode))
        stage1_only = bool(pose_backend == "colmap" and policy == "stage1_only")
        relax_stage2 = bool(pose_backend == "colmap" and policy == "stage2_relaxed")
        effective_nms = colmap_nms_window if colmap_shortcut else default_nms_window
        stage0_on = bool(self.config.get("ENABLE_STAGE0_SCAN", True))
        stage3_on = bool(self.config.get("ENABLE_STAGE3_REFINEMENT", True))
        disabled_components = [
            "stage0",
            "stage1_5",
            "stage3",
            "dynamic_mask",
            "retarget",
            "vo_dependent_selection",
        ] if minimal_mode else []
        if colmap_shortcut:
            stage0_on = bool(colmap_enable_stage0) and selection_profile != "no_vo_coverage"
            stage3_on = False
        if minimal_mode:
            stage0_on = False
            stage3_on = False
        effective_stage_plan = (
            "Stage1->Stage2(minimal_v1)"
            if minimal_mode
            else "Stage1 only"
            if stage1_only
            else (
                (
                    "Stage1->Stage1.5->Stage2->Stage2.5(no_vo_coverage)"
                    if (selection_profile == "no_vo_coverage" and colmap_shortcut)
                    else f"{'Stage1->Stage0->Stage2(relaxed)' if stage0_on else 'Stage1->Stage2(relaxed)'}"
                )
                if relax_stage2
                else f"Stage1->{'0->' if stage0_on else ''}2{'->3' if stage3_on else ''}"
            )
        )

        # Persist normalized runtime config for downstream consumers.
        self.config["POSE_BACKEND"] = pose_backend
        self.config["COLMAP_PIPELINE_MODE"] = pipeline_mode
        self.config["colmap_pipeline_mode"] = pipeline_mode
        self.config["COLMAP_KEYFRAME_POLICY"] = policy
        self.config["COLMAP_SELECTION_PROFILE"] = selection_profile
        self.config["COLMAP_KEYFRAME_TARGET_MODE"] = target_mode
        self.config["COLMAP_KEYFRAME_TARGET_MIN"] = target_min
        self.config["COLMAP_KEYFRAME_TARGET_MAX"] = target_max
        self.config["COLMAP_NMS_WINDOW_SEC"] = colmap_nms_window
        self.config["COLMAP_ENABLE_STAGE0"] = bool(colmap_enable_stage0)
        self.config["COLMAP_MOTION_AWARE_SELECTION"] = bool(colmap_motion_aware_selection)
        self.config["COLMAP_NMS_MOTION_WINDOW_RATIO"] = float(colmap_nms_motion_window_ratio)
        self.config["COLMAP_STAGE1_ADAPTIVE_THRESHOLD"] = bool(colmap_stage1_adaptive_threshold)
        self.config["COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN"] = int(colmap_stage1_min_candidates_per_bin)
        self.config["COLMAP_STAGE1_MAX_CANDIDATES"] = int(colmap_stage1_max_candidates)
        self.config["COLMAP_STAGE2_ENTRY_BUDGET"] = int(colmap_stage2_entry_budget)
        self.config["COLMAP_STAGE2_ENTRY_MIN_GAP"] = int(colmap_stage2_entry_min_gap)
        self.config["COLMAP_DIVERSITY_SSIM_THRESHOLD"] = float(colmap_diversity_ssim_threshold)
        self.config["COLMAP_DIVERSITY_PHASH_HAMMING"] = int(colmap_diversity_phash_hamming)
        self.config["COLMAP_FINAL_TARGET_POLICY"] = str(colmap_final_target_policy)
        self.config["COLMAP_FINAL_SOFT_MIN"] = int(colmap_final_soft_min)
        self.config["COLMAP_FINAL_SOFT_MAX"] = int(colmap_final_soft_max)
        self.config["COLMAP_NO_SUPPLEMENT_ON_LOW_QUALITY"] = bool(colmap_no_supplement_on_low_quality)
        self.config["COLMAP_RIG_POLICY"] = rig_policy
        self.config["COLMAP_RIG_SEED_OPK_DEG"] = list(rig_seed_opk)
        self.config["COLMAP_WORKSPACE_SCOPE"] = workspace_scope
        self.config["COLMAP_REUSE_DB"] = reuse_db
        self.config["COLMAP_ANALYSIS_MASK_PROFILE"] = analysis_mask_profile
        self.config["COLMAP_MINIMAL_MODE"] = bool(minimal_mode)
        self.config["colmap_minimal_mode"] = bool(minimal_mode)
        if minimal_mode:
            logger.warning(
                "COLMAP minimal_v1 mode enabled: "
                "legacy knobs (policy/target/stage0/stage1.5/stage3/retarget/dynamic-mask) are ignored."
            )
        if pose_backend == "colmap" and selection_profile == "no_vo_coverage":
            self.config["ENABLE_DYNAMIC_MASK_REMOVAL"] = True
            self.config["enable_dynamic_mask_removal"] = True
        if minimal_mode:
            self.config["ENABLE_DYNAMIC_MASK_REMOVAL"] = False
            self.config["enable_dynamic_mask_removal"] = False
        if pose_backend == "colmap" and analysis_mask_profile == "colmap_safe":
            classes = self.config.get("DYNAMIC_MASK_TARGET_CLASSES", DYNAMIC_MASK_DEFAULT_CLASSES)
            if not isinstance(classes, list):
                classes = list(classes) if classes else list(DYNAMIC_MASK_DEFAULT_CLASSES)
            self.config["COLMAP_ANALYSIS_TARGET_CLASSES"] = [c for c in classes if str(c) != "空"]
        else:
            classes = self.config.get("DYNAMIC_MASK_TARGET_CLASSES", DYNAMIC_MASK_DEFAULT_CLASSES)
            if not isinstance(classes, list):
                classes = list(classes) if classes else list(DYNAMIC_MASK_DEFAULT_CLASSES)
            self.config["COLMAP_ANALYSIS_TARGET_CLASSES"] = list(classes)

        return {
            "pose_backend": pose_backend,
            "pipeline_mode": pipeline_mode,
            "minimal_mode": bool(minimal_mode),
            "policy": policy,
            "selection_profile": selection_profile,
            "target_mode": target_mode,
            "target_min": target_min,
            "target_max": target_max,
            "colmap_shortcut": colmap_shortcut,
            "stage1_only": stage1_only,
            "relax_stage2": relax_stage2,
            "effective_nms_window": effective_nms,
            "colmap_enable_stage0": bool(colmap_enable_stage0),
            "colmap_motion_aware_selection": bool(colmap_motion_aware_selection),
            "colmap_nms_motion_window_ratio": float(colmap_nms_motion_window_ratio),
            "colmap_stage1_adaptive_threshold": bool(colmap_stage1_adaptive_threshold),
            "colmap_stage1_min_candidates_per_bin": int(colmap_stage1_min_candidates_per_bin),
            "colmap_stage1_max_candidates": int(colmap_stage1_max_candidates),
            "colmap_stage2_entry_budget": int(colmap_stage2_entry_budget),
            "colmap_stage2_entry_min_gap": int(colmap_stage2_entry_min_gap),
            "colmap_diversity_ssim_threshold": float(colmap_diversity_ssim_threshold),
            "colmap_diversity_phash_hamming": int(colmap_diversity_phash_hamming),
            "colmap_final_target_policy": str(colmap_final_target_policy),
            "colmap_final_soft_min": int(colmap_final_soft_min),
            "colmap_final_soft_max": int(colmap_final_soft_max),
            "colmap_no_supplement_on_low_quality": bool(colmap_no_supplement_on_low_quality),
            "force_stage0_off": bool((colmap_shortcut and (not stage0_on)) or minimal_mode),
            "force_stage3_off": bool(colmap_shortcut or minimal_mode),
            "rig_policy": rig_policy,
            "rig_seed_opk_deg": list(rig_seed_opk),
            "workspace_scope": workspace_scope,
            "reuse_db": reuse_db,
            "analysis_mask_profile": analysis_mask_profile,
            "disabled_components": list(disabled_components),
            "effective_stage_plan": effective_stage_plan,
        }

    @staticmethod
    def _score_from_stage1_candidate(candidate: Dict[str, Any]) -> float:
        qs = candidate.get("quality_scores", {}) if isinstance(candidate, dict) else {}
        q = float(qs.get("quality", 0.0)) if isinstance(qs, dict) else 0.0
        return float(np.clip(q, 0.0, 1.0))

    def _build_stage1_keyframes(
        self,
        stage1_candidates: List[Dict[str, Any]],
        *,
        fps: float,
    ) -> List[KeyframeInfo]:
        fps_safe = max(float(fps), 1e-6)
        keyframes: List[KeyframeInfo] = []
        for cand in sorted(stage1_candidates, key=lambda x: int(x.get("frame_idx", 0))):
            frame_idx = int(cand.get("frame_idx", 0))
            quality_scores = dict(cand.get("quality_scores", {}) or {})
            keyframes.append(
                KeyframeInfo(
                    frame_index=frame_idx,
                    timestamp=frame_idx / fps_safe,
                    quality_scores=quality_scores,
                    geometric_scores={},
                    adaptive_scores={},
                    combined_score=self._score_from_stage1_candidate(cand),
                )
            )
        return keyframes

    @staticmethod
    def _stage1_candidate_from_quality_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(record, dict):
            return None
        try:
            frame_idx = int(record.get("frame_index", -1))
        except (TypeError, ValueError):
            return None
        if frame_idx < 0:
            return None
        quality = float(record.get("quality", 0.0) or 0.0)
        raw = dict(record.get("raw_metrics", {}) or {})
        legacy = dict(record.get("legacy_quality_scores", {}) or {})
        sharpness = float(raw.get("laplacian_var", legacy.get("sharpness", 0.0)) or 0.0)
        exposure = float(raw.get("exposure", legacy.get("exposure", 0.0)) or 0.0)
        return {
            "frame_idx": frame_idx,
            "quality_scores": {
                "quality": float(np.clip(quality, 0.0, 1.0)),
                "sharpness": sharpness,
                "exposure": exposure,
                "passes_threshold": bool(record.get("is_pass", False)),
            },
        }

    def _limit_stage1_candidates_by_quality_time(
        self,
        candidates: List[Dict[str, Any]],
        *,
        total_frames: int,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        if target_count <= 0:
            return []
        if len(candidates) <= target_count:
            return sorted(candidates, key=lambda x: int(x.get("frame_idx", 0)))

        by_idx: Dict[int, Dict[str, Any]] = {}
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            idx = int(cand.get("frame_idx", -1))
            if idx < 0:
                continue
            by_idx[idx] = dict(cand)
        if len(by_idx) <= target_count:
            return sorted(by_idx.values(), key=lambda x: int(x.get("frame_idx", 0)))

        idxs_sorted = sorted(by_idx.keys())
        selected_idxs = {idxs_sorted[0], idxs_sorted[-1]}
        bins = int(max(1, target_count))
        bin_width = float(max(1, total_frames)) / float(max(1, bins))
        for b in range(bins):
            if len(selected_idxs) >= target_count:
                break
            start = int(np.floor(b * bin_width))
            end = int(np.floor((b + 1) * bin_width)) if b < bins - 1 else int(max(total_frames, idxs_sorted[-1] + 1))
            pool = [idx for idx in idxs_sorted if start <= idx < end and idx not in selected_idxs]
            if not pool:
                continue
            best_idx = max(
                pool,
                key=lambda idx: (
                    self._score_from_stage1_candidate(by_idx[idx]),
                    -abs(idx - int((start + end) * 0.5)),
                ),
            )
            selected_idxs.add(best_idx)

        if len(selected_idxs) < target_count:
            leftovers = [idx for idx in idxs_sorted if idx not in selected_idxs]
            leftovers_sorted = sorted(
                leftovers,
                key=lambda idx: (self._score_from_stage1_candidate(by_idx[idx]), -idx),
                reverse=True,
            )
            for idx in leftovers_sorted:
                if len(selected_idxs) >= target_count:
                    break
                selected_idxs.add(idx)

        selected = [by_idx[idx] for idx in sorted(selected_idxs)]
        if len(selected) > target_count:
            sampled = self._time_distributed_downsample(
                self._build_stage1_keyframes(selected, fps=1.0),
                target_count,
            )
            selected_map = {int(k.frame_index): by_idx[int(k.frame_index)] for k in sampled if int(k.frame_index) in by_idx}
            if idxs_sorted[0] in by_idx:
                selected_map[idxs_sorted[0]] = by_idx[idxs_sorted[0]]
            if idxs_sorted[-1] in by_idx:
                selected_map[idxs_sorted[-1]] = by_idx[idxs_sorted[-1]]
            selected = [selected_map[idx] for idx in sorted(selected_map.keys())]
            if len(selected) > target_count:
                selected = selected[:target_count]
        return selected

    def _apply_stage1_adaptive_threshold_and_bin_floor(
        self,
        stage1_candidates: List[Dict[str, Any]],
        *,
        total_frames: int,
        target_hint: int,
        min_candidates_per_bin: int,
        max_candidates: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        selected_map: Dict[int, Dict[str, Any]] = {}
        for cand in stage1_candidates:
            if not isinstance(cand, dict):
                continue
            idx = int(cand.get("frame_idx", -1))
            if idx < 0:
                continue
            selected_map[idx] = dict(cand)
        raw_count = len(selected_map)

        pool_rows: List[Dict[str, Any]] = []
        for rec in list(self.stage1_quality_records or []):
            cand = self._stage1_candidate_from_quality_record(rec)
            if cand is None:
                continue
            pool_rows.append(cand)
        if not pool_rows:
            return list(stage1_candidates), {
                "raw_count": int(raw_count),
                "effective_count": int(raw_count),
                "base_threshold": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                "effective_threshold": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                "q_pass_target": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                "threshold_added_count": 0,
                "bin_floor_added_count": 0,
                "max_cap_trimmed_count": 0,
                "bins": 0,
            }

        quality_vals = np.asarray(
            [self._score_from_stage1_candidate(c) for c in pool_rows],
            dtype=np.float64,
        )
        base_threshold = float(np.clip(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50)), 0.0, 1.0))
        q_pass_target = float(np.quantile(quality_vals, 0.85)) if quality_vals.size > 0 else base_threshold
        effective_threshold = float(min(base_threshold, q_pass_target))

        threshold_added = 0
        if effective_threshold < base_threshold:
            for cand in pool_rows:
                idx = int(cand.get("frame_idx", -1))
                if idx < 0 or idx in selected_map:
                    continue
                if self._score_from_stage1_candidate(cand) >= effective_threshold:
                    selected_map[idx] = dict(cand)
                    threshold_added += 1

        bins = int(np.clip(max(6, target_hint // 12), 6, 48))
        min_per_bin = int(max(0, min_candidates_per_bin))
        bin_floor_added = 0
        if min_per_bin > 0 and bins > 0:
            bin_width = float(max(1, total_frames)) / float(max(1, bins))
            for b in range(bins):
                start = int(np.floor(b * bin_width))
                end = int(np.floor((b + 1) * bin_width)) if b < bins - 1 else int(max(total_frames, 1))
                in_bin_selected = [idx for idx in selected_map.keys() if start <= idx < end]
                needed = max(0, min_per_bin - len(in_bin_selected))
                if needed <= 0:
                    continue
                pool = [c for c in pool_rows if start <= int(c["frame_idx"]) < end and int(c["frame_idx"]) not in selected_map]
                if not pool:
                    continue
                pool = sorted(
                    pool,
                    key=lambda c: (
                        self._score_from_stage1_candidate(c),
                        -abs(int(c["frame_idx"]) - int((start + end) * 0.5)),
                    ),
                    reverse=True,
                )
                for cand in pool[:needed]:
                    idx = int(cand["frame_idx"])
                    if idx in selected_map:
                        continue
                    selected_map[idx] = dict(cand)
                    bin_floor_added += 1

        trimmed = 0
        if len(selected_map) > int(max_candidates):
            before = len(selected_map)
            limited = self._limit_stage1_candidates_by_quality_time(
                list(selected_map.values()),
                total_frames=int(max(1, total_frames)),
                target_count=int(max_candidates),
            )
            selected_map = {int(c["frame_idx"]): dict(c) for c in limited}
            trimmed = max(0, before - len(selected_map))

        out = [selected_map[idx] for idx in sorted(selected_map.keys())]
        return out, {
            "raw_count": int(raw_count),
            "effective_count": int(len(out)),
            "base_threshold": float(base_threshold),
            "effective_threshold": float(effective_threshold),
            "q_pass_target": float(q_pass_target),
            "threshold_added_count": int(threshold_added),
            "bin_floor_added_count": int(bin_floor_added),
            "max_cap_trimmed_count": int(trimmed),
            "bins": int(bins),
        }

    @staticmethod
    def _build_cumulative_motion_map(
        stage0_metrics: Dict[int, Dict[str, Any]],
        total_frames: int,
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        if total_frames <= 0 or not stage0_metrics:
            return {}, {
                "sample_count": 0,
                "motion_median_step": 1.0,
                "motion_min": 0.0,
                "motion_max": 0.0,
            }

        sample_indices = sorted(int(idx) for idx in stage0_metrics.keys() if int(idx) >= 0)
        if not sample_indices:
            return {}, {
                "sample_count": 0,
                "motion_median_step": 1.0,
                "motion_min": 0.0,
                "motion_max": 0.0,
            }

        sample_motion = []
        running = 0.0
        positive_steps: List[float] = []
        for idx in sample_indices:
            raw_flow = float((stage0_metrics.get(idx, {}) or {}).get("flow_mag_light", 0.0) or 0.0)
            flow = max(0.0, raw_flow)
            running += flow
            sample_motion.append(running)
            if flow > 0.0:
                positive_steps.append(flow)

        x = np.asarray(sample_indices, dtype=np.float64)
        y = np.asarray(sample_motion, dtype=np.float64)
        full_x = np.arange(0, int(max(1, total_frames)), dtype=np.float64)
        full_y = np.interp(full_x, x, y, left=float(y[0]), right=float(y[-1]))
        full_map = {int(i): float(v) for i, v in enumerate(full_y.tolist())}
        median_step = float(np.median(np.asarray(positive_steps, dtype=np.float64))) if positive_steps else 1.0
        if median_step <= 1e-12:
            median_step = 1.0
        return full_map, {
            "sample_count": int(len(sample_indices)),
            "motion_median_step": float(median_step),
            "motion_min": float(full_y[0]) if full_y.size > 0 else 0.0,
            "motion_max": float(full_y[-1]) if full_y.size > 0 else 0.0,
        }

    @staticmethod
    def _count_motion_bins_occupied(
        frame_indices: List[int],
        cumulative_motion_map: Dict[int, float],
        *,
        bins: int = 12,
    ) -> int:
        if not frame_indices or not cumulative_motion_map:
            return 0
        bins = int(max(1, bins))
        motions = [float(cumulative_motion_map.get(int(idx), 0.0)) for idx in frame_indices]
        m_min = float(min(motions))
        m_max = float(max(motions))
        if m_max - m_min <= 1e-9:
            return 1
        step = (m_max - m_min) / float(bins)
        occupied = set()
        for m in motions:
            b = int(np.floor((m - m_min) / max(step, 1e-12)))
            b = max(0, min(bins - 1, b))
            occupied.add(b)
        return int(len(occupied))

    @staticmethod
    def _count_time_bins_occupied(
        frame_indices: List[int],
        *,
        total_frames: int,
        bins: int = 24,
    ) -> int:
        if total_frames <= 0 or not frame_indices:
            return 0
        bins = int(max(1, bins))
        width = float(max(1, total_frames)) / float(max(1, bins))
        occupied = set()
        for idx in frame_indices:
            b = int(np.floor(float(max(0, idx)) / max(width, 1e-12)))
            b = max(0, min(bins - 1, b))
            occupied.add(b)
        return int(len(occupied))

    @staticmethod
    def _estimate_sky_ratio(frame: Optional[np.ndarray], valid_mask: Optional[np.ndarray] = None) -> float:
        if frame is None:
            return 0.0
        try:
            from processing.target_mask_generator import TargetMaskGenerator

            sky_mask = TargetMaskGenerator._detect_sky_mask(frame)
            if sky_mask is None or sky_mask.size == 0:
                return 0.0

            if valid_mask is not None and valid_mask.shape[:2] == sky_mask.shape[:2]:
                valid = valid_mask > 0
            else:
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                valid = gray > 8

            denom = int(np.count_nonzero(valid))
            if denom <= 0:
                return 0.0
            numer = int(np.count_nonzero((sky_mask > 0) & valid))
            return float(numer / max(denom, 1))
        except Exception:
            return 0.0

    @staticmethod
    def _summarize_stage1_lr_statistics(
        stage1_records: List[Dict[str, Any]],
        *,
        quality_threshold: float,
        merge_mode: str,
    ) -> Dict[str, Any]:
        matrix = {
            "both_pass": 0,
            "a_only_pass": 0,
            "b_only_pass": 0,
            "neither": 0,
        }
        merge_counts = {
            "merge_mode_configured": str(merge_mode),
            "strict_min_applied": 0,
            "asymmetric_applied": 0,
            "asym_eligible": 0,
            "weak_floor_reject": 0,
            "asym_abs_guard_reject": 0,
            "auto_relaxed_records": 0,
        }
        sky_diffs: List[float] = []
        sky_values: List[float] = []
        effective_sky_thrs: List[float] = []
        effective_weak_floors: List[float] = []
        for rec in stage1_records or []:
            qa = rec.get("quality_lens_a")
            qb = rec.get("quality_lens_b")
            if isinstance(qa, (int, float)) and isinstance(qb, (int, float)):
                pass_a = float(qa) >= float(quality_threshold)
                pass_b = float(qb) >= float(quality_threshold)
                if pass_a and pass_b:
                    matrix["both_pass"] += 1
                elif pass_a and not pass_b:
                    matrix["a_only_pass"] += 1
                elif pass_b and not pass_a:
                    matrix["b_only_pass"] += 1
                else:
                    matrix["neither"] += 1
            if bool(rec.get("lr_asym_eligible", False)):
                merge_counts["asym_eligible"] += 1
            mode_applied = str(rec.get("lr_merge_mode_applied", "") or "")
            if mode_applied == "asymmetric_sky_v1":
                merge_counts["asymmetric_applied"] += 1
            elif mode_applied == "strict_min":
                merge_counts["strict_min_applied"] += 1
            if str(rec.get("drop_reason", "")) == "lr_weak_floor":
                merge_counts["weak_floor_reject"] += 1
            if mode_applied == "asymmetric_sky_v1" and str(rec.get("drop_reason", "")) == "abs_laplacian_guard":
                merge_counts["asym_abs_guard_reject"] += 1
            if bool(rec.get("lr_auto_relaxed", False)):
                merge_counts["auto_relaxed_records"] += 1

            sky_a = rec.get("sky_ratio_lens_a")
            sky_b = rec.get("sky_ratio_lens_b")
            if isinstance(sky_a, (int, float)) and isinstance(sky_b, (int, float)):
                a = float(np.clip(float(sky_a), 0.0, 1.0))
                b = float(np.clip(float(sky_b), 0.0, 1.0))
                sky_values.extend([a, b])
                sky_diffs.append(abs(a - b))
            if isinstance(rec.get("lr_sky_ratio_threshold"), (int, float)):
                effective_sky_thrs.append(float(np.clip(float(rec.get("lr_sky_ratio_threshold")), 0.0, 1.0)))
            if isinstance(rec.get("lr_weak_floor"), (int, float)):
                effective_weak_floors.append(float(np.clip(float(rec.get("lr_weak_floor")), 0.0, 1.0)))

        if effective_sky_thrs:
            sky_thr_arr = np.asarray(effective_sky_thrs, dtype=np.float64)
            merge_counts["effective_sky_ratio_threshold_median"] = float(np.median(sky_thr_arr))
        if effective_weak_floors:
            weak_arr = np.asarray(effective_weak_floors, dtype=np.float64)
            merge_counts["effective_weak_floor_median"] = float(np.median(weak_arr))

        if sky_diffs:
            sky_arr = np.asarray(sky_diffs, dtype=np.float64)
            sky_val_arr = np.asarray(sky_values, dtype=np.float64) if sky_values else np.zeros(0, dtype=np.float64)
            sky_stats = {
                "count": int(sky_arr.size),
                "diff_mean": float(np.mean(sky_arr)),
                "diff_median": float(np.median(sky_arr)),
                "diff_p90": float(np.percentile(sky_arr, 90)),
                "diff_max": float(np.max(sky_arr)),
                "sky_mean": float(np.mean(sky_val_arr)) if sky_val_arr.size > 0 else 0.0,
            }
        else:
            sky_stats = {
                "count": 0,
                "diff_mean": 0.0,
                "diff_median": 0.0,
                "diff_p90": 0.0,
                "diff_max": 0.0,
                "sky_mean": 0.0,
            }

        return {
            "stage1_lr_merge_mode": str(merge_mode),
            "stage1_lr_merge_counts": dict(merge_counts),
            "stage1_lens_pass_matrix": dict(matrix),
            "stage1_sky_asymmetry_stats": dict(sky_stats),
        }

    @staticmethod
    def _gray_preview(frame: Optional[np.ndarray], size: int = 96) -> Optional[np.ndarray]:
        if frame is None:
            return None
        try:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

    @staticmethod
    def _compute_phash_signature(gray: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if gray is None:
            return None
        try:
            resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            dct = cv2.dct(resized)
            low = dct[:8, :8]
            med = float(np.median(low[1:, :])) if low.size > 1 else float(np.median(low))
            bits = (low.flatten() > med).astype(np.uint8)
            return bits
        except Exception:
            return None

    @staticmethod
    def _hamming_distance_bits(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> int:
        if a is None or b is None:
            return 64
        if a.shape != b.shape:
            return 64
        return int(np.count_nonzero(a != b))

    def _stage15_build_visual_signatures(
        self,
        video_loader,
        frame_indices: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        is_paired = bool(getattr(video_loader, "is_paired", False))
        out: Dict[int, Dict[str, Any]] = {}
        for idx in frame_indices:
            if is_paired and hasattr(video_loader, "get_frame_pair"):
                frame_l, frame_r = video_loader.get_frame_pair(int(idx))
                gray_l = self._gray_preview(frame_l)
                gray_r = self._gray_preview(frame_r)
                out[int(idx)] = {
                    "paired": True,
                    "gray_l": gray_l,
                    "gray_r": gray_r,
                    "phash_l": self._compute_phash_signature(gray_l),
                    "phash_r": self._compute_phash_signature(gray_r),
                }
            elif hasattr(video_loader, "get_frame"):
                frame = video_loader.get_frame(int(idx))
                gray = self._gray_preview(frame)
                out[int(idx)] = {
                    "paired": False,
                    "gray": gray,
                    "phash": self._compute_phash_signature(gray),
                }
            else:
                out[int(idx)] = {"paired": False}
        return out

    def _stage15_similarity(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> Tuple[float, int]:
        if bool(a.get("paired", False)) and bool(b.get("paired", False)):
            gray_al = a.get("gray_l")
            gray_ar = a.get("gray_r")
            gray_bl = b.get("gray_l")
            gray_br = b.get("gray_r")
            ssim_l = float(self.adaptive_selector.compute_ssim(gray_al, gray_bl)) if gray_al is not None and gray_bl is not None else 0.0
            ssim_r = float(self.adaptive_selector.compute_ssim(gray_ar, gray_br)) if gray_ar is not None and gray_br is not None else 0.0
            ham_l = self._hamming_distance_bits(a.get("phash_l"), b.get("phash_l"))
            ham_r = self._hamming_distance_bits(a.get("phash_r"), b.get("phash_r"))
            return float(min(ssim_l, ssim_r)), int(max(ham_l, ham_r))
        gray_a = a.get("gray")
        gray_b = b.get("gray")
        ssim = float(self.adaptive_selector.compute_ssim(gray_a, gray_b)) if gray_a is not None and gray_b is not None else 0.0
        ham = self._hamming_distance_bits(a.get("phash"), b.get("phash"))
        return ssim, ham

    def _run_stage15_entry_budget(
        self,
        stage1_candidates: List[Dict[str, Any]],
        *,
        total_frames: int,
        video_loader,
        entry_budget: int,
        min_gap: int,
        diversity_ssim_threshold: float,
        diversity_phash_hamming: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        if not stage1_candidates:
            return [], {
                "entry_count": 0,
                "drop_reason_counts": {},
                "coverage_bins_before": 0,
                "coverage_bins_after": 0,
            }, []

        by_idx: Dict[int, Dict[str, Any]] = {}
        for cand in stage1_candidates:
            if not isinstance(cand, dict):
                continue
            idx = int(cand.get("frame_idx", -1))
            if idx < 0:
                continue
            by_idx[idx] = dict(cand)
        sorted_idxs = sorted(by_idx.keys())
        if not sorted_idxs:
            return [], {
                "entry_count": 0,
                "drop_reason_counts": {},
                "coverage_bins_before": 0,
                "coverage_bins_after": 0,
            }, []

        if len(sorted_idxs) <= int(max(1, entry_budget)):
            trace = []
            for idx in sorted_idxs:
                trace.append(
                    {
                        "frame": int(idx),
                        "stage": "stage15",
                        "kept": 1,
                        "reason": "pass_through",
                        "score_quality": float(self._score_from_stage1_candidate(by_idx[idx])),
                        "score_novelty_ssim": 0.0,
                        "score_novelty_phash": 0.0,
                        "score_combined": float(self._score_from_stage1_candidate(by_idx[idx])),
                    }
                )
            coverage_bins = self._count_time_bins_occupied(sorted_idxs, total_frames=int(total_frames), bins=24)
            return [by_idx[idx] for idx in sorted_idxs], {
                "entry_count": int(len(sorted_idxs)),
                "drop_reason_counts": {},
                "coverage_bins_before": int(coverage_bins),
                "coverage_bins_after": int(coverage_bins),
            }, trace

        signatures = self._stage15_build_visual_signatures(video_loader, sorted_idxs)
        bins = 24
        bin_width = float(max(1, total_frames)) / float(max(1, bins))

        trace_map: Dict[int, Dict[str, Any]] = {}
        drop_counts: Counter = Counter()
        quality_map = {idx: float(self._score_from_stage1_candidate(by_idx[idx])) for idx in sorted_idxs}
        novelty_map: Dict[int, Tuple[float, float, float]] = {}
        for pos, idx in enumerate(sorted_idxs):
            if pos <= 0:
                novelty_map[idx] = (1.0, 1.0, quality_map[idx])
                continue
            prev_idx = sorted_idxs[pos - 1]
            ssim_prev, ham_prev = self._stage15_similarity(signatures.get(idx, {}), signatures.get(prev_idx, {}))
            novelty_ssim = float(np.clip(1.0 - ssim_prev, 0.0, 1.0))
            novelty_phash = float(np.clip(float(ham_prev) / 64.0, 0.0, 1.0))
            combined = float(0.6 * quality_map[idx] + 0.25 * novelty_ssim + 0.15 * novelty_phash)
            novelty_map[idx] = (novelty_ssim, novelty_phash, combined)

        selected_idxs: List[int] = []
        selected_set = set()
        # Coverage-first: each time bin keeps best candidate.
        for b in range(bins):
            start = int(np.floor(b * bin_width))
            end = int(np.floor((b + 1) * bin_width)) if b < bins - 1 else int(max(total_frames, 1))
            pool = [idx for idx in sorted_idxs if start <= idx < end]
            if not pool:
                continue
            best_idx = max(
                pool,
                key=lambda idx: (
                    novelty_map[idx][2],
                    quality_map[idx],
                    -abs(idx - int((start + end) * 0.5)),
                ),
            )
            if best_idx in selected_set:
                continue
            selected_set.add(best_idx)
            selected_idxs.append(best_idx)
            trace_map[best_idx] = {
                "frame": int(best_idx),
                "stage": "stage15",
                "kept": 1,
                "reason": "coverage_seed",
                "score_quality": float(quality_map[best_idx]),
                "score_novelty_ssim": float(novelty_map[best_idx][0]),
                "score_novelty_phash": float(novelty_map[best_idx][1]),
                "score_combined": float(novelty_map[best_idx][2]),
            }

        ranked = sorted(
            [idx for idx in sorted_idxs if idx not in selected_set],
            key=lambda idx: (novelty_map[idx][2], quality_map[idx], -idx),
            reverse=True,
        )
        for idx in ranked:
            base_trace = {
                "frame": int(idx),
                "stage": "stage15",
                "score_quality": float(quality_map[idx]),
                "score_novelty_ssim": float(novelty_map[idx][0]),
                "score_novelty_phash": float(novelty_map[idx][1]),
                "score_combined": float(novelty_map[idx][2]),
            }
            if len(selected_set) >= int(max(1, entry_budget)):
                drop_counts["entry_budget_trim"] += 1
                trace_map[idx] = dict(base_trace, kept=0, reason="entry_budget_trim")
                continue
            if any(abs(int(idx) - int(sidx)) < int(max(0, min_gap)) for sidx in selected_idxs):
                drop_counts["min_gap_trim"] += 1
                trace_map[idx] = dict(base_trace, kept=0, reason="min_gap_trim")
                continue
            duplicate_reason = ""
            for sidx in selected_idxs:
                ssim_pair, ham_pair = self._stage15_similarity(signatures.get(idx, {}), signatures.get(sidx, {}))
                if ssim_pair >= float(diversity_ssim_threshold):
                    duplicate_reason = "near_duplicate_ssim"
                    break
                if ham_pair <= int(max(0, diversity_phash_hamming)):
                    duplicate_reason = "near_duplicate_phash"
                    break
            if duplicate_reason:
                drop_counts[duplicate_reason] += 1
                trace_map[idx] = dict(base_trace, kept=0, reason=duplicate_reason)
                continue
            selected_set.add(idx)
            selected_idxs.append(idx)
            trace_map[idx] = dict(base_trace, kept=1, reason="selected")

        selected_idxs = sorted(selected_set)
        out = [by_idx[idx] for idx in selected_idxs]
        trace_rows = [trace_map[idx] for idx in sorted(trace_map.keys())]
        coverage_before = self._count_time_bins_occupied(sorted_idxs, total_frames=int(total_frames), bins=24)
        coverage_after = self._count_time_bins_occupied(selected_idxs, total_frames=int(total_frames), bins=24)
        return out, {
            "entry_count": int(len(out)),
            "drop_reason_counts": dict(drop_counts),
            "coverage_bins_before": int(coverage_before),
            "coverage_bins_after": int(coverage_after),
        }, trace_rows

    @staticmethod
    def _save_selection_trace_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        fieldnames = [
            "frame",
            "stage",
            "kept",
            "reason",
            "score_quality",
            "score_novelty_ssim",
            "score_novelty_phash",
            "score_combined",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _coverage_backfill_stage1_candidates(
        self,
        stage1_candidates: List[Dict[str, Any]],
        *,
        total_frames: int,
        target_hint: int,
        cumulative_motion_map: Optional[Dict[int, float]] = None,
        motion_aware_enabled: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if total_frames <= 0:
            return list(stage1_candidates), {"added": 0, "bins": 0}

        selected_map: Dict[int, Dict[str, Any]] = {}
        for cand in stage1_candidates:
            if not isinstance(cand, dict):
                continue
            idx = int(cand.get("frame_idx", -1))
            if idx < 0:
                continue
            selected_map[idx] = dict(cand)

        bins = int(np.clip(max(6, target_hint // 12), 6, 48))
        bin_width = float(max(1, total_frames)) / float(max(1, bins))

        pool_rows = []
        for rec in list(self.stage1_quality_records or []):
            cand = self._stage1_candidate_from_quality_record(rec)
            if cand is None:
                continue
            pool_rows.append(cand)
        if not pool_rows:
            return list(stage1_candidates), {"added": 0, "bins": bins}

        added_time = 0
        for b in range(bins):
            start = int(np.floor(b * bin_width))
            end = int(np.floor((b + 1) * bin_width)) if b < bins - 1 else total_frames
            in_bin_selected = [idx for idx in selected_map.keys() if start <= idx < end]
            if in_bin_selected:
                continue
            pool = [c for c in pool_rows if start <= int(c["frame_idx"]) < end]
            if not pool:
                continue
            best = max(
                pool,
                key=lambda c: (
                    self._score_from_stage1_candidate(c),
                    -abs(int(c["frame_idx"]) - int((start + end) * 0.5)),
                ),
            )
            idx = int(best["frame_idx"])
            if idx in selected_map:
                continue
            selected_map[idx] = dict(best)
            added_time += 1

        added_motion = 0
        motion_bins = 0
        if motion_aware_enabled and cumulative_motion_map:
            motion_bins = int(np.clip(max(6, target_hint // 12), 6, 48))
            motion_values = [float(cumulative_motion_map.get(i, 0.0)) for i in range(total_frames)]
            motion_min = float(min(motion_values)) if motion_values else 0.0
            motion_max = float(max(motion_values)) if motion_values else 0.0
            if motion_max - motion_min > 1e-9:
                motion_step = (motion_max - motion_min) / float(max(1, motion_bins))
                for b in range(motion_bins):
                    start_m = motion_min + motion_step * b
                    end_m = motion_min + motion_step * (b + 1) if b < motion_bins - 1 else motion_max + 1e-9
                    in_bin_selected = [
                        idx for idx in selected_map.keys()
                        if start_m <= float(cumulative_motion_map.get(idx, motion_min)) < end_m
                    ]
                    if in_bin_selected:
                        continue
                    pool = [
                        c for c in pool_rows
                        if start_m <= float(cumulative_motion_map.get(int(c["frame_idx"]), motion_min)) < end_m
                    ]
                    if not pool:
                        continue
                    best = max(
                        pool,
                        key=lambda c: (
                            self._score_from_stage1_candidate(c),
                            -int(c["frame_idx"]),
                        ),
                    )
                    idx = int(best["frame_idx"])
                    if idx in selected_map:
                        continue
                    selected_map[idx] = dict(best)
                    added_motion += 1

        out = sorted(selected_map.values(), key=lambda x: int(x.get("frame_idx", 0)))
        return out, {
            "added": int(added_time + added_motion),
            "added_time_bins": int(added_time),
            "added_motion_bins": int(added_motion),
            "bins": int(bins),
            "motion_bins": int(motion_bins),
            "after": len(out),
        }

    @staticmethod
    def _compute_temporal_coverage(
        keyframes: List[KeyframeInfo],
        *,
        total_frames: int,
    ) -> Dict[str, Any]:
        if total_frames <= 0:
            return {"count": len(keyframes), "coverage_ratio": 0.0, "max_gap_frames": 0, "max_gap_ratio": 0.0}
        if not keyframes:
            return {"count": 0, "coverage_ratio": 0.0, "max_gap_frames": total_frames, "max_gap_ratio": 1.0}
        idxs = sorted(int(k.frame_index) for k in keyframes)
        unique_idxs = sorted(set(idxs))
        if len(unique_idxs) == 1:
            max_gap = total_frames - 1
        else:
            gaps = [unique_idxs[i] - unique_idxs[i - 1] for i in range(1, len(unique_idxs))]
            max_gap = int(max(gaps) if gaps else 0)
        span = int(unique_idxs[-1] - unique_idxs[0]) if len(unique_idxs) > 1 else 0
        coverage_ratio = float(np.clip((span + 1) / max(1, total_frames), 0.0, 1.0))
        return {
            "count": int(len(unique_idxs)),
            "coverage_ratio": coverage_ratio,
            "max_gap_frames": int(max_gap),
            "max_gap_ratio": float(max_gap / max(1, total_frames)),
            "first_frame": int(unique_idxs[0]),
            "last_frame": int(unique_idxs[-1]),
        }

    def _estimate_auto_target_bounds(
        self,
        *,
        total_frames: int,
        fps: float,
        stage1_candidates: List[Dict[str, Any]],
        stage2_records: Optional[List[Stage2FrameRecord]],
        min_bound: int,
        max_bound: int,
    ) -> Tuple[int, int, Dict[str, Any]]:
        fps_safe = max(float(fps), 1e-6)
        duration_sec = float(total_frames / fps_safe)
        base_count = float(duration_sec / 1.0)

        motion_values: List[float] = []
        feature_match_values: List[float] = []
        if stage2_records:
            for rec in stage2_records:
                if not isinstance(rec, Stage2FrameRecord):
                    continue
                motion_values.append(float((rec.adaptive_scores or {}).get("optical_flow", 0.0) or 0.0))
                feature_match_values.append(float((rec.geometric_scores or {}).get("feature_match_count", 0.0) or 0.0))

        if motion_values:
            motion_med = float(np.median(np.asarray(motion_values, dtype=np.float64)))
            motion_factor = float(np.clip(0.70 + motion_med / 20.0, 0.70, 1.80))
        else:
            motion_med = 0.0
            motion_factor = 1.0

        if feature_match_values:
            feats = np.asarray(feature_match_values, dtype=np.float64)
            effective_feature_ratio = float(np.mean(feats >= 30.0))
            feature_factor = float(np.clip(0.80 + effective_feature_ratio, 0.70, 1.40))
        else:
            effective_feature_ratio = float(
                np.clip(len(stage1_candidates) / max(1, total_frames), 0.0, 1.0)
            )
            feature_factor = float(np.clip(0.80 + effective_feature_ratio, 0.70, 1.40))

        target_center = int(round(base_count * motion_factor * feature_factor))
        target_center = int(np.clip(target_center, min_bound, max_bound))
        half_span = int(max(8, round(target_center * 0.15)))
        auto_min = int(max(min_bound, target_center - half_span))
        auto_max = int(min(max_bound, target_center + half_span))
        auto_max = max(auto_max, auto_min)
        return auto_min, auto_max, {
            "base_count": float(base_count),
            "motion_median": float(motion_med),
            "motion_factor": float(motion_factor),
            "effective_feature_ratio": float(effective_feature_ratio),
            "feature_factor": float(feature_factor),
            "target_center": int(target_center),
            "auto_min": int(auto_min),
            "auto_max": int(auto_max),
        }

    @staticmethod
    def _time_distributed_downsample(
        keyframes: List[KeyframeInfo],
        target_count: int,
    ) -> List[KeyframeInfo]:
        if target_count <= 0:
            return []
        if len(keyframes) <= target_count:
            return list(keyframes)
        if target_count == 1:
            return [keyframes[0]]

        positions = np.linspace(0, len(keyframes) - 1, num=target_count, dtype=np.float64)
        sampled_indices = [int(round(p)) for p in positions.tolist()]
        # Keep order and de-duplicate.
        dedup: List[int] = []
        seen = set()
        for idx in sampled_indices:
            idx = int(np.clip(idx, 0, len(keyframes) - 1))
            if idx in seen:
                continue
            seen.add(idx)
            dedup.append(idx)
        if 0 not in seen:
            dedup.insert(0, 0)
            seen.add(0)
        last = len(keyframes) - 1
        if last not in seen:
            dedup.append(last)
            seen.add(last)
        dedup = sorted(dedup)
        if len(dedup) > target_count:
            pick_pos = np.linspace(0, len(dedup) - 1, num=target_count, dtype=np.float64)
            dedup = [dedup[int(round(p))] for p in pick_pos.tolist()]
            dedup = sorted(set(dedup))
            if 0 not in dedup:
                dedup.insert(0, 0)
            if last not in dedup:
                dedup.append(last)
            dedup = sorted(dedup)[:target_count]
        return [keyframes[i] for i in dedup]

    def _motion_distributed_downsample(
        self,
        keyframes: List[KeyframeInfo],
        target_count: int,
        cumulative_motion_map: Dict[int, float],
    ) -> List[KeyframeInfo]:
        if target_count <= 0:
            return []
        if len(keyframes) <= target_count:
            return list(keyframes)
        if target_count == 1:
            return [keyframes[0]]
        if not cumulative_motion_map:
            return self._time_distributed_downsample(keyframes, target_count)

        ordered = sorted(keyframes, key=lambda k: int(k.frame_index))
        motions = np.asarray(
            [float(cumulative_motion_map.get(int(k.frame_index), 0.0)) for k in ordered],
            dtype=np.float64,
        )
        m_min = float(np.min(motions))
        m_max = float(np.max(motions))
        if m_max - m_min <= 1e-9:
            return self._time_distributed_downsample(ordered, target_count)

        target_motion = np.linspace(m_min, m_max, num=target_count, dtype=np.float64)
        chosen: List[int] = []
        used = set()
        for tm in target_motion.tolist():
            ranking = np.argsort(np.abs(motions - float(tm))).tolist()
            pick = None
            for ridx in ranking:
                if int(ridx) in used:
                    continue
                pick = int(ridx)
                break
            if pick is None:
                continue
            used.add(pick)
            chosen.append(pick)

        if 0 not in used:
            chosen.append(0)
            used.add(0)
        last_idx = len(ordered) - 1
        if last_idx not in used:
            chosen.append(last_idx)
            used.add(last_idx)
        chosen = sorted(set(chosen))
        if len(chosen) > target_count:
            pick_pos = np.linspace(0, len(chosen) - 1, num=target_count, dtype=np.float64)
            chosen = sorted(set(chosen[int(round(p))] for p in pick_pos.tolist()))
            if 0 not in chosen:
                chosen.insert(0, 0)
            if last_idx not in chosen:
                chosen.append(last_idx)
            chosen = sorted(chosen)[:target_count]
        return [ordered[i] for i in chosen]

    def _retarget_keyframes_for_colmap(
        self,
        keyframes: List[KeyframeInfo],
        stage1_candidates: List[Dict[str, Any]],
        *,
        total_frames: int,
        fps: float,
        target_mode: str,
        target_min: int,
        target_max: int,
        stage2_records: Optional[List[Stage2FrameRecord]] = None,
        cumulative_motion_map: Optional[Dict[int, float]] = None,
        final_target_policy: str = "fixed",
        final_soft_min: int = 80,
        final_soft_max: int = 220,
        no_supplement_on_low_quality: bool = True,
    ) -> Tuple[List[KeyframeInfo], Dict[str, Any]]:
        mode = str(target_mode or "fixed").strip().lower()
        if mode not in {"fixed", "auto"}:
            mode = "fixed"
        effective_min = int(max(1, target_min))
        effective_max = int(max(effective_min, target_max))
        auto_details: Dict[str, Any] = {}
        if mode == "auto":
            effective_min, effective_max, auto_details = self._estimate_auto_target_bounds(
                total_frames=int(max(1, total_frames)),
                fps=float(fps),
                stage1_candidates=stage1_candidates,
                stage2_records=stage2_records,
                min_bound=effective_min,
                max_bound=effective_max,
            )
        policy = str(final_target_policy or "fixed").strip().lower()
        if policy not in {"soft_auto", "fixed"}:
            policy = "fixed"
        if policy == "soft_auto":
            effective_min = int(max(1, final_soft_min))
            effective_max = int(max(effective_min, final_soft_max))

        selected = sorted(keyframes, key=lambda k: int(k.frame_index))
        pre_count = len(selected)
        reason_parts: List[str] = []
        final_reject_reason = ""
        pre_motion_bins = self._count_motion_bins_occupied(
            [int(k.frame_index) for k in selected],
            cumulative_motion_map or {},
        )
        if pre_count > effective_max:
            if cumulative_motion_map:
                selected = self._motion_distributed_downsample(selected, effective_max, cumulative_motion_map)
                reason_parts.append("motion_downsample_to_max")
            else:
                selected = self._time_distributed_downsample(selected, effective_max)
                reason_parts.append("downsample_to_max")

        selected_by_idx: Dict[int, KeyframeInfo] = {int(k.frame_index): k for k in selected}
        selected_idxs = sorted(selected_by_idx.keys())

        if len(selected_idxs) < effective_min:
            fps_safe = max(float(fps), 1e-6)
            stage1_map: Dict[int, Dict[str, Any]] = {}
            for cand in stage1_candidates:
                if not isinstance(cand, dict):
                    continue
                idx = int(cand.get("frame_idx", -1))
                if idx < 0:
                    continue
                stage1_map[idx] = cand

            quality_guard = float(np.clip(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50)), 0.0, 1.0))
            if no_supplement_on_low_quality:
                pool = [
                    idx for idx in sorted(stage1_map.keys())
                    if idx not in selected_by_idx and self._score_from_stage1_candidate(stage1_map[idx]) >= quality_guard
                ]
            else:
                pool = [idx for idx in sorted(stage1_map.keys()) if idx not in selected_by_idx]
            pre_supplement_count = len(selected_by_idx)
            while len(selected_by_idx) < effective_min and pool:
                if selected_idxs and cumulative_motion_map:
                    best_idx = max(
                        pool,
                        key=lambda idx: (
                            min(
                                abs(
                                    float(cumulative_motion_map.get(idx, 0.0)) -
                                    float(cumulative_motion_map.get(sidx, 0.0))
                                ) for sidx in selected_idxs
                            ),
                            self._score_from_stage1_candidate(stage1_map[idx]),
                            -idx,
                        ),
                    )
                elif selected_idxs:
                    best_idx = max(
                        pool,
                        key=lambda idx: (
                            min(abs(idx - sidx) for sidx in selected_idxs),
                            self._score_from_stage1_candidate(stage1_map[idx]),
                            -idx,
                        ),
                    )
                else:
                    best_idx = pool[len(pool) // 2]
                cand = stage1_map[best_idx]
                quality_scores = dict(cand.get("quality_scores", {}) or {})
                selected_by_idx[best_idx] = KeyframeInfo(
                    frame_index=best_idx,
                    timestamp=best_idx / fps_safe,
                    quality_scores=quality_scores,
                    geometric_scores={},
                    adaptive_scores={},
                    combined_score=self._score_from_stage1_candidate(cand),
                )
                pool.remove(best_idx)
                selected_idxs = sorted(selected_by_idx.keys())
            if len(selected_by_idx) > pre_supplement_count:
                reason_parts.append("supplement_to_min" if policy == "fixed" else "supplement_soft_min")
            if len(selected_by_idx) < effective_min:
                if no_supplement_on_low_quality:
                    final_reject_reason = "under_target_quality_guard"
                    reason_parts.append("under_target_quality_guard")
                else:
                    reason_parts.append("insufficient_stage1_candidates")

        out = [selected_by_idx[idx] for idx in sorted(selected_by_idx.keys())]
        post_count = len(out)
        post_motion_bins = self._count_motion_bins_occupied(
            [int(k.frame_index) for k in out],
            cumulative_motion_map or {},
        )
        reason = "+".join(reason_parts) if reason_parts else "within_target"
        return out, {
            "target_mode": mode,
            "final_target_policy": policy,
            "effective_target_min": int(effective_min),
            "effective_target_max": int(effective_max),
            "auto_target": dict(auto_details),
            "pre_retarget_count": int(pre_count),
            "post_retarget_count": int(post_count),
            "retarget_reason": reason,
            "final_reject_reason": str(final_reject_reason),
            "motion_bins_occupied_before": int(pre_motion_bins),
            "motion_bins_occupied_after": int(post_motion_bins),
        }

    def run_stage1_scan(
        self,
        video_loader,
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Stage1ScanArtifact:
        candidates = run_stage1_filter(self, video_loader, metadata, progress_callback)
        records = list(self.stage1_quality_records or [])
        total_frames = int(getattr(metadata, "frame_count", 0) or 0)
        sample_interval = max(1, int(self.config.get("SAMPLE_INTERVAL", 1)))
        sampled_frames = len(range(0, total_frames, sample_interval))
        return Stage1ScanArtifact(
            candidates=list(candidates),
            records=records,
            sampled_frames=int(sampled_frames),
            total_frames=int(total_frames),
        )

    def select_keyframes(
        self,
        video_loader,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        frame_log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stage_temp_store: Optional[StageTempStore] = None,
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
        run_id = str(self.config.get("ANALYSIS_RUN_ID", self.config.get("analysis_run_id", "n/a")))
        analysis_mode = str(
            self.config.get("ANALYSIS_MODE", self.config.get("analysis_mode", "full"))
        ).strip().lower()
        logger.info(f"解析モード: {analysis_mode}")
        colmap_runtime = self._resolve_colmap_keyframe_runtime()
        logger.info(
            "keyframe_policy, "
            f"pose_backend={colmap_runtime['pose_backend']}, "
            f"pipeline={colmap_runtime.get('pipeline_mode', 'legacy')}, "
            f"minimal_mode={bool(colmap_runtime.get('minimal_mode', False))}, "
            f"policy={colmap_runtime['policy']}, "
            f"profile={colmap_runtime.get('selection_profile', 'legacy')}, "
            f"target_mode={colmap_runtime['target_mode']}, "
            f"target={colmap_runtime['target_min']}-{colmap_runtime['target_max']}, "
            f"nms={colmap_runtime['effective_nms_window']:.2f}, "
            f"rig={colmap_runtime['rig_policy']}@{colmap_runtime['rig_seed_opk_deg']}, "
            f"workspace_scope={colmap_runtime['workspace_scope']}, "
            f"reuse_db={colmap_runtime['reuse_db']}, "
            f"mask_profile={colmap_runtime['analysis_mask_profile']}, "
            f"plan={colmap_runtime['effective_stage_plan']}"
        )
        self.last_selection_runtime = {
            "pose_backend": colmap_runtime["pose_backend"],
            "pipeline_mode": str(colmap_runtime.get("pipeline_mode", "legacy")),
            "minimal_mode": bool(colmap_runtime.get("minimal_mode", False)),
            "disabled_components": list(colmap_runtime.get("disabled_components", [])),
            "policy": colmap_runtime["policy"],
            "selection_profile": colmap_runtime.get("selection_profile", "legacy"),
            "target_mode": colmap_runtime["target_mode"],
            "effective_stage_plan": colmap_runtime["effective_stage_plan"],
            "stage1_candidates_raw": 0,
            "stage1_candidates_effective": 0,
            "stage15_entry_count": 0,
            "stage15_drop_reason_counts": {},
            "stage0_executed_count": 0,
            "stage3_executed_count": 0,
            "stage2_read_success_count": 0,
            "stage1_adaptive_threshold_base": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
            "stage1_adaptive_threshold_effective": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
            "stage1_bin_floor_added_count": 0,
            "motion_median_step": 0.0,
            "effective_motion_window": 0.0,
            "motion_bins_occupied_before": 0,
            "motion_bins_occupied_after": 0,
            "stage2_drop_reason_counts": {},
            "pre_retarget_count": 0,
            "post_retarget_count": 0,
            "retarget_reason": "n/a",
            "final_target_policy": str(colmap_runtime.get("colmap_final_target_policy", "soft_auto")),
            "final_target_soft_range": [
                int(colmap_runtime.get("colmap_final_soft_min", 80)),
                int(colmap_runtime.get("colmap_final_soft_max", 220)),
            ],
            "final_reject_reason": "",
            "effective_target_min": int(colmap_runtime["target_min"]),
            "effective_target_max": int(colmap_runtime["target_max"]),
            "auto_target": {},
            "coverage_before": {},
            "coverage_after": {},
            "coverage_bins_before_stage15": 0,
            "coverage_bins_after_stage15": 0,
            "coverage_bins_before_final": 0,
            "coverage_bins_after_final": 0,
        }

        def _stage_start(stage: str, **kwargs):
            extra = ",".join([f" {k}={v}" for k, v in kwargs.items()])
            logger.info(f"stage_start, analysis_run_id={run_id}, stage={stage},{extra}".rstrip(","))

        def _stage_summary(stage: str, **kwargs):
            extra = ",".join([f" {k}={v}" for k, v in kwargs.items()])
            logger.info(f"stage_summary, analysis_run_id={run_id}, stage={stage},{extra}".rstrip(","))

        mode_enable_stage0 = bool(self.config.get('ENABLE_STAGE0_SCAN', True))
        mode_enable_stage3 = bool(self.config.get('ENABLE_STAGE3_REFINEMENT', True))
        if colmap_runtime["force_stage0_off"]:
            mode_enable_stage0 = False
        if colmap_runtime["force_stage3_off"]:
            mode_enable_stage3 = False
        profile_enabled = bool(self.config.get('ENABLE_PROFILE', False))
        resume_enabled = bool(self.config.get('RESUME_ENABLED', self.config.get('resume_enabled', False)))
        keep_temp_on_success = bool(
            self.config.get('KEEP_TEMP_ON_SUCCESS', self.config.get('keep_temp_on_success', False))
        )
        current_stage = "init"
        success = False

        if run_id in {"", "n/a", "none", "null"}:
            run_id = str(uuid.uuid4())
            self.config["analysis_run_id"] = run_id
            self.config["ANALYSIS_RUN_ID"] = run_id

        store = stage_temp_store if stage_temp_store is not None else StageTempStore(run_id)
        store.record_resume_state(enabled=resume_enabled)

        def _profile_log(stage: str, elapsed_s: float, frames: int) -> None:
            if not profile_enabled:
                return
            total_ms = max(0.0, elapsed_s) * 1000.0
            if frames > 0 and elapsed_s > 0.0:
                ms_per_frame = total_ms / float(frames)
                fps = float(frames) / float(elapsed_s)
            else:
                ms_per_frame = 0.0
                fps = 0.0
            logger.info(
                f"[PROFILE] Stage{stage}: {total_ms:.0f}ms total, "
                f"{ms_per_frame:.1f}ms/frame, {fps:.1f}frames/s"
            )

        if analysis_mode == "stage0":
            mode_enable_stage0 = True
            mode_enable_stage3 = False
        elif analysis_mode == "stage2":
            mode_enable_stage3 = False
        elif analysis_mode == "stage3":
            mode_enable_stage0 = True
            mode_enable_stage3 = True

        try:
            selection_trace_rows: List[Dict[str, Any]] = []
            # ===== Stage 0 only mode =====
            if analysis_mode == "stage0":
                current_stage = "0"
                stage0_metrics: Dict[int, Dict[str, Any]] = {}
                if mode_enable_stage0:
                    t_stage0 = perf_counter() if profile_enabled else 0.0
                    _stage_start("0", total_frames=total_frames, stride=int(self.config.get("STAGE0_STRIDE", 5)))
                    if resume_enabled and store.has_stage0():
                        logger.info("Stage 0: テンポラリ結果を再利用")
                        stage0_metrics = store.load_stage0()
                    else:
                        logger.info("Stage 0: 軽量運動量走査開始")
                        stage0_metrics = self._stage0_lightweight_motion_scan(
                            video_loader, metadata, progress_callback, frame_log_callback
                        )
                        stage0_path = store.save_stage0(stage0_metrics)
                        stage0_metrics = store.load_stage0()
                        store.mark_stage_done(
                            "0",
                            files={"metrics": stage0_path},
                            counts={"samples": len(stage0_metrics)},
                        )
                    logger.info(f"Stage 0完了: {len(stage0_metrics)}サンプル")
                    _stage_summary("0", samples=len(stage0_metrics), temp_file=str(store.run_dir / store.STAGE0_METRICS_FILE))
                    if profile_enabled:
                        _profile_log("0", perf_counter() - t_stage0, len(stage0_metrics))
                logger.info("Stage 0モードのため Stage1/2/3 をスキップ")
                logger.info(
                    f"analysis_result, analysis_run_id={run_id}, mode={analysis_mode},"
                    " keyframes=0, note=stage0_only"
                )
                self.last_selection_runtime.update(
                    {
                        "pre_retarget_count": 0,
                        "post_retarget_count": 0,
                        "retarget_reason": "stage0_only",
                    }
                )
                success = True
                return []

            # ===== Stage 1: 高速フィルタリング =====
            current_stage = "1"
            t_stage1 = perf_counter() if profile_enabled else 0.0
            _stage_start("1", total_frames=total_frames)
            if resume_enabled and store.has_stage1():
                logger.info("Stage 1: テンポラリ結果を再利用")
            else:
                logger.info("Stage 1: 高速品質フィルタリング開始")
                stage1_artifact = self.run_stage1_scan(video_loader, metadata, progress_callback)
                stage1_files = store.save_stage1(stage1_artifact.candidates, stage1_artifact.records)
                store.mark_stage_done(
                    "1",
                    files=stage1_files,
                    counts={
                        "candidates": len(stage1_artifact.candidates),
                        "records": len(stage1_artifact.records),
                        "sampled_frames": int(stage1_artifact.sampled_frames),
                    },
                )
            stage1_candidates, stage1_records = store.load_stage1()
            self.stage1_quality_records = list(stage1_records)
            stage1_lr_merge_mode = str(
                self.config.get("STAGE1_LR_MERGE_MODE", self.config.get("stage1_lr_merge_mode", "asymmetric_sky_v1"))
                or "asymmetric_sky_v1"
            ).strip().lower()
            if stage1_lr_merge_mode not in {"asymmetric_sky_v1", "strict_min"}:
                stage1_lr_merge_mode = "asymmetric_sky_v1"
            stage1_lr_summary = self._summarize_stage1_lr_statistics(
                list(stage1_records),
                quality_threshold=float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                merge_mode=stage1_lr_merge_mode,
            )
            stage1_candidates_raw = sorted(
                [dict(c) for c in stage1_candidates if isinstance(c, dict)],
                key=lambda x: int(x.get("frame_idx", 0)),
            )
            stage1_candidates = list(stage1_candidates_raw)
            stage1_adaptive_info: Dict[str, Any] = {
                "raw_count": int(len(stage1_candidates_raw)),
                "effective_count": int(len(stage1_candidates_raw)),
                "base_threshold": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                "effective_threshold": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                "q_pass_target": float(self.config.get("QUALITY_THRESHOLD", self.config.get("quality_threshold", 0.50))),
                "threshold_added_count": 0,
                "bin_floor_added_count": 0,
                "max_cap_trimmed_count": 0,
                "bins": 0,
            }
            stage1_backfill_info: Dict[str, Any] = {
                "added": 0,
                "added_time_bins": 0,
                "added_motion_bins": 0,
                "bins": 0,
                "motion_bins": 0,
                "after": int(len(stage1_candidates)),
            }
            stage1_effective_rows = []
            if colmap_runtime["colmap_shortcut"] and resume_enabled:
                stage1_effective_rows = [
                    dict(c) for c in store.load_stage1_effective() if isinstance(c, dict)
                ]
            if stage1_effective_rows:
                stage1_candidates = sorted(stage1_effective_rows, key=lambda x: int(x.get("frame_idx", 0)))
            elif colmap_runtime["colmap_shortcut"]:
                if bool(colmap_runtime.get("colmap_stage1_adaptive_threshold", True)):
                    stage1_candidates, stage1_adaptive_info = self._apply_stage1_adaptive_threshold_and_bin_floor(
                        stage1_candidates,
                        total_frames=int(total_frames),
                        target_hint=int(colmap_runtime["target_min"]),
                        min_candidates_per_bin=int(colmap_runtime.get("colmap_stage1_min_candidates_per_bin", 3)),
                        max_candidates=int(colmap_runtime.get("colmap_stage1_max_candidates", 360)),
                    )
                before_stage1_cov = len(stage1_candidates)
                stage1_candidates, stage1_backfill_info = self._coverage_backfill_stage1_candidates(
                    stage1_candidates,
                    total_frames=int(total_frames),
                    target_hint=int(colmap_runtime["target_min"]),
                )
                if int(stage1_backfill_info.get("added", 0)) > 0:
                    logger.info(
                        "stage1_coverage_backfill, "
                        f"before={before_stage1_cov}, after={len(stage1_candidates)}, "
                        f"added={int(stage1_backfill_info.get('added', 0))}, "
                        f"time_added={int(stage1_backfill_info.get('added_time_bins', 0))}, "
                        f"bins={int(stage1_backfill_info.get('bins', 0))}"
                    )

            stage1_effective_path = store.save_stage1_effective(stage1_candidates)
            store.mark_stage_done(
                "1",
                files={
                    "candidates": str(store.run_dir / store.STAGE1_CANDIDATES_FILE),
                    "records": str(store.run_dir / store.STAGE1_RECORDS_FILE),
                    "candidates_effective": str(stage1_effective_path),
                },
                counts={
                    "candidates_raw": int(len(stage1_candidates_raw)),
                    "candidates_effective": int(len(stage1_candidates)),
                    "records": int(len(stage1_records)),
                },
            )
            logger.info(
                f"Stage 1完了: {len(stage1_candidates)}/{total_frames} "
                f"({100*len(stage1_candidates)/max(total_frames, 1):.1f}%)"
            )
            _stage_summary(
                "1",
                passed=len(stage1_candidates),
                raw_candidates=len(stage1_candidates_raw),
                effective_candidates=len(stage1_candidates),
                total=total_frames,
                records=len(stage1_records),
                temp_file=str(stage1_effective_path),
                pass_ratio=f"{100*len(stage1_candidates)/max(total_frames, 1):.2f}",
            )
            if profile_enabled:
                sample_interval = max(1, int(self.config.get('SAMPLE_INTERVAL', 1)))
                stage1_samples = len(range(0, total_frames, sample_interval))
                _profile_log("1", perf_counter() - t_stage1, stage1_samples)
            self.last_selection_runtime.update(
                {
                    "stage1_candidates_raw": int(len(stage1_candidates_raw)),
                    "stage1_candidates_effective": int(len(stage1_candidates)),
                    "stage1_adaptive_threshold_base": float(stage1_adaptive_info.get("base_threshold", 0.0)),
                    "stage1_adaptive_threshold_effective": float(stage1_adaptive_info.get("effective_threshold", 0.0)),
                    "stage1_q_pass_target": float(stage1_adaptive_info.get("q_pass_target", 0.0)),
                    "stage1_threshold_added_count": int(stage1_adaptive_info.get("threshold_added_count", 0)),
                    "stage1_bin_floor_added_count": int(stage1_adaptive_info.get("bin_floor_added_count", 0)),
                    "stage1_backfill_added_count": int(stage1_backfill_info.get("added", 0)),
                    "stage1_lr_merge_mode": str(stage1_lr_summary.get("stage1_lr_merge_mode", "")),
                    "stage1_lr_merge_counts": dict(stage1_lr_summary.get("stage1_lr_merge_counts", {})),
                    "stage1_lens_pass_matrix": dict(stage1_lr_summary.get("stage1_lens_pass_matrix", {})),
                    "stage1_sky_asymmetry_stats": dict(stage1_lr_summary.get("stage1_sky_asymmetry_stats", {})),
                }
            )

            # ===== Stage 1.5: COLMAP no-VO coverage entry budgeting =====
            stage15_info: Dict[str, Any] = {
                "entry_count": 0 if colmap_runtime["minimal_mode"] else int(len(stage1_candidates)),
                "drop_reason_counts": {},
                "coverage_bins_before": 0 if colmap_runtime["minimal_mode"] else self._count_time_bins_occupied(
                    [int(c.get("frame_idx", -1)) for c in stage1_candidates if int(c.get("frame_idx", -1)) >= 0],
                    total_frames=int(total_frames),
                    bins=24,
                ),
                "coverage_bins_after": 0 if colmap_runtime["minimal_mode"] else self._count_time_bins_occupied(
                    [int(c.get("frame_idx", -1)) for c in stage1_candidates if int(c.get("frame_idx", -1)) >= 0],
                    total_frames=int(total_frames),
                    bins=24,
                ),
            }
            if (
                (not colmap_runtime["minimal_mode"])
                and colmap_runtime["colmap_shortcut"]
                and str(colmap_runtime.get("selection_profile", "legacy")) == "no_vo_coverage"
            ):
                stage15_file = store.run_dir / store.STAGE15_CANDIDATES_FILE
                if resume_enabled and stage15_file.exists():
                    stage15_rows = [dict(c) for c in store.load_stage15() if isinstance(c, dict)]
                    if stage15_rows:
                        stage1_candidates = sorted(stage15_rows, key=lambda x: int(x.get("frame_idx", 0)))
                        stage15_info["entry_count"] = int(len(stage1_candidates))
                        stage15_info["coverage_bins_after"] = self._count_time_bins_occupied(
                            [int(c.get("frame_idx", -1)) for c in stage1_candidates if int(c.get("frame_idx", -1)) >= 0],
                            total_frames=int(total_frames),
                            bins=24,
                        )
                else:
                    stage1_candidates, stage15_info, stage15_trace_rows = self._run_stage15_entry_budget(
                        stage1_candidates,
                        total_frames=int(total_frames),
                        video_loader=video_loader,
                        entry_budget=int(colmap_runtime.get("colmap_stage2_entry_budget", 180)),
                        min_gap=int(colmap_runtime.get("colmap_stage2_entry_min_gap", 3)),
                        diversity_ssim_threshold=float(colmap_runtime.get("colmap_diversity_ssim_threshold", 0.93)),
                        diversity_phash_hamming=int(colmap_runtime.get("colmap_diversity_phash_hamming", 10)),
                    )
                    selection_trace_rows.extend(stage15_trace_rows)
                    stage15_path = store.save_stage15(stage1_candidates)
                    store.mark_stage_done(
                        "1.5",
                        files={"candidates": stage15_path},
                        counts={"entry_count": int(len(stage1_candidates))},
                    )
                logger.info(
                    "stage1_5_summary, "
                    f"entry_count={int(stage15_info.get('entry_count', len(stage1_candidates)))}, "
                    f"coverage_bins={int(stage15_info.get('coverage_bins_before', 0))}->{int(stage15_info.get('coverage_bins_after', 0))}, "
                    f"drop_reason_counts={dict(stage15_info.get('drop_reason_counts', {}))}"
                )
                self.last_selection_runtime.update(
                    {
                        "stage15_entry_count": int(stage15_info.get("entry_count", len(stage1_candidates))),
                        "stage15_drop_reason_counts": dict(stage15_info.get("drop_reason_counts", {})),
                        "coverage_bins_before_stage15": int(stage15_info.get("coverage_bins_before", 0)),
                        "coverage_bins_after_stage15": int(stage15_info.get("coverage_bins_after", 0)),
                        "stage1_candidates_effective": int(len(stage1_candidates)),
                    }
                )
            elif colmap_runtime["minimal_mode"]:
                self.last_selection_runtime.update(
                    {
                        "stage15_entry_count": 0,
                        "stage15_drop_reason_counts": {},
                        "coverage_bins_before_stage15": 0,
                        "coverage_bins_after_stage15": 0,
                    }
                )

            if not stage1_candidates:
                logger.warning("Stage 1でフレームが残りませんでした")
                logger.info(
                    f"analysis_result, analysis_run_id={run_id}, mode={analysis_mode},"
                    " keyframes=0, note=stage1_empty"
                )
                self.last_selection_runtime.update(
                    {
                        "pre_retarget_count": 0,
                        "post_retarget_count": 0,
                        "retarget_reason": "stage1_empty",
                    }
                )
                success = True
                return []

            # ===== Stage 0: 軽量走査 =====
            current_stage = "0"
            stage0_metrics = {}
            cumulative_motion_map: Dict[int, float] = {}
            motion_map_info: Dict[str, Any] = {
                "sample_count": 0,
                "motion_median_step": 0.0,
                "motion_min": 0.0,
                "motion_max": 0.0,
            }
            effective_motion_window = 0.0
            if mode_enable_stage0:
                self.last_selection_runtime["stage0_executed_count"] = 1
                t_stage0 = perf_counter() if profile_enabled else 0.0
                _stage_start("0", total_frames=total_frames, stride=int(self.config.get("STAGE0_STRIDE", 5)))
                if resume_enabled and store.has_stage0():
                    logger.info("Stage 0: テンポラリ結果を再利用")
                    stage0_metrics = store.load_stage0()
                else:
                    logger.info("Stage 0: 軽量運動量走査開始")
                    stage0_metrics = self._stage0_lightweight_motion_scan(
                        video_loader, metadata, progress_callback, frame_log_callback
                    )
                    stage0_path = store.save_stage0(stage0_metrics)
                    stage0_metrics = store.load_stage0()
                    store.mark_stage_done(
                        "0",
                        files={"metrics": stage0_path},
                        counts={"samples": len(stage0_metrics)},
                    )
                logger.info(f"Stage 0完了: {len(stage0_metrics)}サンプル")
                _stage_summary("0", samples=len(stage0_metrics), temp_file=str(store.run_dir / store.STAGE0_METRICS_FILE))
                if profile_enabled:
                    _profile_log("0", perf_counter() - t_stage0, len(stage0_metrics))
            else:
                self.last_selection_runtime["stage0_executed_count"] = 0
                _stage_summary("0", samples=0, skipped="disabled")
            if colmap_runtime["colmap_shortcut"] and bool(colmap_runtime.get("colmap_motion_aware_selection", True)):
                cumulative_motion_map, motion_map_info = self._build_cumulative_motion_map(
                    stage0_metrics,
                    int(total_frames),
                )
                motion_ratio = float(colmap_runtime.get("colmap_nms_motion_window_ratio", 0.5))
                effective_motion_window = float(
                    max(0.0, motion_ratio * float(motion_map_info.get("motion_median_step", 1.0)))
                )
                self.last_selection_runtime.update(
                    {
                        "motion_median_step": float(motion_map_info.get("motion_median_step", 0.0)),
                        "effective_motion_window": float(effective_motion_window),
                    }
                )
                if cumulative_motion_map:
                    before_motion_backfill = len(stage1_candidates)
                    stage1_candidates, motion_backfill_info = self._coverage_backfill_stage1_candidates(
                        stage1_candidates,
                        total_frames=int(total_frames),
                        target_hint=int(colmap_runtime["target_min"]),
                        cumulative_motion_map=cumulative_motion_map,
                        motion_aware_enabled=True,
                    )
                    if int(motion_backfill_info.get("added", 0)) > 0:
                        logger.info(
                            "stage1_motion_backfill, "
                            f"before={before_motion_backfill}, after={len(stage1_candidates)}, "
                            f"added={int(motion_backfill_info.get('added', 0))}, "
                            f"time_added={int(motion_backfill_info.get('added_time_bins', 0))}, "
                            f"motion_added={int(motion_backfill_info.get('added_motion_bins', 0))}, "
                            f"motion_bins={int(motion_backfill_info.get('motion_bins', 0))}"
                        )
                        stage1_backfill_info["added"] = int(stage1_backfill_info.get("added", 0)) + int(motion_backfill_info.get("added", 0))
                        stage1_backfill_info["added_time_bins"] = int(stage1_backfill_info.get("added_time_bins", 0)) + int(motion_backfill_info.get("added_time_bins", 0))
                        stage1_backfill_info["added_motion_bins"] = int(stage1_backfill_info.get("added_motion_bins", 0)) + int(motion_backfill_info.get("added_motion_bins", 0))
                        stage1_backfill_info["motion_bins"] = int(motion_backfill_info.get("motion_bins", 0))
                        stage1_backfill_info["after"] = int(motion_backfill_info.get("after", len(stage1_candidates)))
                    stage1_effective_path = store.save_stage1_effective(stage1_candidates)
                    store.mark_stage_done(
                        "1",
                        files={
                            "candidates": str(store.run_dir / store.STAGE1_CANDIDATES_FILE),
                            "records": str(store.run_dir / store.STAGE1_RECORDS_FILE),
                            "candidates_effective": str(stage1_effective_path),
                        },
                        counts={
                            "candidates_raw": int(len(stage1_candidates_raw)),
                            "candidates_effective": int(len(stage1_candidates)),
                            "records": int(len(stage1_records)),
                        },
                    )
                    self.last_selection_runtime["stage1_candidates_effective"] = int(len(stage1_candidates))
                    self.last_selection_runtime["stage1_backfill_added_count"] = int(stage1_backfill_info.get("added", 0))

            # ===== Stage 2: 精密評価 =====
            current_stage = "2"
            t_stage2 = perf_counter() if profile_enabled else 0.0
            stage2_records: List[Stage2FrameRecord] = []
            if colmap_runtime["minimal_mode"]:
                _stage_start("2", candidates=len(stage1_candidates), mode="minimal_v1")
                logger.info(
                    "Stage 2(minimal_v1): read_successフレームを全採用（optical_flowは記録のみ）"
                )
                if resume_enabled and store.has_stage2():
                    logger.info("Stage 2(minimal_v1): テンポラリ結果を再利用")
                    stage2_candidate_rows, stage2_record_rows = store.load_stage2()
                    stage2_candidates = [self._deserialize_keyframe_info(row) for row in stage2_candidate_rows]
                    stage2_records = [self._deserialize_stage2_record(row) for row in stage2_record_rows]
                else:
                    stage2_candidates, stage2_records = self._stage2_minimal_motion_only_evaluation(
                        video_loader,
                        metadata,
                        stage1_candidates,
                        progress_callback,
                    )
                    stage2_candidate_rows = [self._serialize_keyframe_info(c) for c in stage2_candidates]
                    stage2_record_rows = [self._serialize_stage2_record(r) for r in stage2_records]
                    stage2_files = store.save_stage2(stage2_candidate_rows, stage2_record_rows)
                    store.mark_stage_done(
                        "2",
                        files=stage2_files,
                        counts={"candidates": len(stage2_candidate_rows), "records": len(stage2_record_rows)},
                    )
                    stage2_candidate_rows, stage2_record_rows = store.load_stage2()
                    stage2_candidates = [self._deserialize_keyframe_info(row) for row in stage2_candidate_rows]
                    stage2_records = [self._deserialize_stage2_record(row) for row in stage2_record_rows]
            elif colmap_runtime["stage1_only"]:
                _stage_start("2", candidates=len(stage1_candidates), mode="stage1_only")
                logger.info("Stage 2: stage1_only policy により Stage1候補を直接利用します")
                stage2_candidates = self._build_stage1_keyframes(stage1_candidates, fps=metadata.fps)
            else:
                _stage_start("2", candidates=len(stage1_candidates))
                logger.info(f"Stage 2: 精密評価開始（{len(stage1_candidates)}フレーム）")
                ssim_original = float(self.config.get("SSIM_CHANGE_THRESHOLD", 0.85))
                ssim_effective = ssim_original
                if colmap_runtime["relax_stage2"]:
                    ssim_effective = float(max(ssim_original, 0.95))
                    self.config["SSIM_CHANGE_THRESHOLD"] = ssim_effective
                    logger.info(
                        f"Stage 2緩和: SSIM skip閾値を引き上げます "
                        f"({ssim_original:.3f} -> {ssim_effective:.3f})"
                    )
                try:
                    if resume_enabled and store.has_stage2():
                        logger.info("Stage 2: テンポラリ結果を再利用")
                        stage2_candidate_rows, stage2_record_rows = store.load_stage2()
                        stage2_candidates = [self._deserialize_keyframe_info(row) for row in stage2_candidate_rows]
                        stage2_records = [self._deserialize_stage2_record(row) for row in stage2_record_rows]
                    else:
                        stage2_candidates, stage2_records = run_stage2_evaluator(
                            self,
                            video_loader,
                            metadata,
                            stage1_candidates,
                            progress_callback,
                            frame_log_callback,
                            stage0_metrics,
                        )

                        self._inject_stage0_vo_metrics_into_stage2_records(stage2_records, stage0_metrics)
                        stage2_candidates = self._apply_stationary_penalty(stage2_candidates, stage2_records, fps=metadata.fps)

                        stage2_candidate_rows = [self._serialize_keyframe_info(c) for c in stage2_candidates]
                        stage2_record_rows = [self._serialize_stage2_record(r) for r in stage2_records]
                        stage2_files = store.save_stage2(stage2_candidate_rows, stage2_record_rows)
                        store.mark_stage_done(
                            "2",
                            files=stage2_files,
                            counts={"candidates": len(stage2_candidate_rows), "records": len(stage2_record_rows)},
                        )
                        stage2_candidate_rows, stage2_record_rows = store.load_stage2()
                        stage2_candidates = [self._deserialize_keyframe_info(row) for row in stage2_candidate_rows]
                        stage2_records = [self._deserialize_stage2_record(row) for row in stage2_record_rows]
                finally:
                    self.config["SSIM_CHANGE_THRESHOLD"] = ssim_original

            logger.info(f"Stage 2キーフレーム候補: {len(stage2_candidates)}個")
            _stage_summary(
                "2",
                candidates=len(stage2_candidates),
                records=len(stage2_records),
                temp_file=str(store.run_dir / store.STAGE2_CANDIDATES_FILE),
            )
            if profile_enabled:
                _profile_log("2", perf_counter() - t_stage2, len(stage1_candidates))
            stage2_drop_reason_counts = dict(
                Counter(str(getattr(r, "drop_reason", "") or "unknown") for r in stage2_records)
            )
            stage2_read_success_count = int(
                sum(
                    1
                    for r in stage2_records
                    if str(getattr(r, "drop_reason", "") or "") != "read_fail"
                )
            )
            if not stage2_records and not colmap_runtime["minimal_mode"]:
                stage2_read_success_count = int(len(stage2_candidates))
            self.last_selection_runtime["stage2_drop_reason_counts"] = dict(stage2_drop_reason_counts)
            self.last_selection_runtime["stage2_read_success_count"] = int(stage2_read_success_count)
            if stage2_drop_reason_counts:
                logger.info(f"stage2_drop_reason_counts, {stage2_drop_reason_counts}")

            if colmap_runtime["minimal_mode"]:
                stage2_keyframes_pre_stage3 = list(stage2_candidates)
            else:
                stage2_keyframes_pre_stage3 = self._enforce_max_interval(
                    self._apply_nms(
                        stage2_candidates,
                        time_window=colmap_runtime["effective_nms_window"],
                        cumulative_motion_map=cumulative_motion_map,
                        motion_window=effective_motion_window,
                        motion_aware_selection=bool(colmap_runtime.get("colmap_motion_aware_selection", True)),
                    ),
                    metadata.fps,
                    source_candidates=stage2_candidates,
                )

            # ===== Stage 3: 軌跡再評価 =====
            current_stage = "3"
            keyframes = stage2_keyframes_pre_stage3
            stage3_executed_count = 0
            if mode_enable_stage3 and resume_enabled and store.has_stage3():
                stage3_executed_count = 1
                logger.info("Stage 3: テンポラリ結果を再利用")
                keyframes = [self._deserialize_keyframe_info(row) for row in store.load_stage3()]
            else:
                if mode_enable_stage3:
                    stage3_executed_count = 1
                    t_stage3 = perf_counter() if profile_enabled else 0.0
                    _stage_start("3", input_candidates=len(stage2_candidates))
                    logger.info("Stage 3: 軌跡再評価開始")
                    stage2_candidates = run_stage3_refiner(
                        self,
                        metadata=metadata,
                        stage2_candidates=stage2_candidates,
                        stage2_final=stage2_keyframes_pre_stage3,
                        stage2_records=stage2_records,
                        stage0_metrics=stage0_metrics,
                        video_loader=video_loader,
                    )
                    keyframes = self._enforce_max_interval(
                        self._apply_nms(
                            stage2_candidates,
                            time_window=colmap_runtime["effective_nms_window"],
                            cumulative_motion_map=cumulative_motion_map,
                            motion_window=effective_motion_window,
                            motion_aware_selection=bool(colmap_runtime.get("colmap_motion_aware_selection", True)),
                        ),
                        metadata.fps,
                        source_candidates=stage2_candidates,
                    )
                    if profile_enabled:
                        _profile_log("3", perf_counter() - t_stage3, len(stage2_candidates))
                stage3_rows = [self._serialize_keyframe_info(kf) for kf in keyframes]
                stage3_path = store.save_stage3(stage3_rows)
                stage3_counts = (
                    {"keyframes": 0, "compat_keyframes": len(stage3_rows)}
                    if colmap_runtime["minimal_mode"]
                    else {"keyframes": len(stage3_rows)}
                )
                store.mark_stage_done("3", files={"keyframes": stage3_path}, counts=stage3_counts)
                keyframes = [self._deserialize_keyframe_info(row) for row in store.load_stage3()]

            if colmap_runtime["minimal_mode"]:
                stage3_executed_count = 0
            self.last_selection_runtime["stage3_executed_count"] = int(stage3_executed_count)
            logger.info(f"Stage 3完了: {len(keyframes)}個")
            _stage_summary(
                "3",
                keyframes=(0 if colmap_runtime["minimal_mode"] else len(keyframes)),
                compat_keyframes=len(keyframes),
                temp_file=str(store.run_dir / store.STAGE3_KEYFRAMES_FILE),
                skipped=("minimal_v1" if colmap_runtime["minimal_mode"] else ""),
            )

            pre_retarget_count = len(keyframes)
            post_retarget_count = pre_retarget_count
            retarget_reason = "within_target"
            retarget: Dict[str, Any] = {}
            coverage_before = self._compute_temporal_coverage(keyframes, total_frames=int(total_frames))
            coverage_after = dict(coverage_before)
            coverage_bins_before_final = self._count_time_bins_occupied(
                [int(k.frame_index) for k in keyframes],
                total_frames=int(total_frames),
                bins=24,
            )
            coverage_bins_after_final = int(coverage_bins_before_final)
            if colmap_runtime["minimal_mode"]:
                retarget_reason = "disabled_minimal_mode"
                retarget = {
                    "retarget_reason": "disabled_minimal_mode",
                    "target_mode": colmap_runtime["target_mode"],
                    "effective_target_min": int(colmap_runtime["target_min"]),
                    "effective_target_max": int(colmap_runtime["target_max"]),
                    "post_retarget_count": int(len(keyframes)),
                }
            elif colmap_runtime["colmap_shortcut"]:
                keyframes, retarget = self._retarget_keyframes_for_colmap(
                    keyframes,
                    stage1_candidates,
                    total_frames=int(total_frames),
                    fps=metadata.fps,
                    target_mode=colmap_runtime["target_mode"],
                    target_min=colmap_runtime["target_min"],
                    target_max=colmap_runtime["target_max"],
                    stage2_records=stage2_records,
                    cumulative_motion_map=(
                        cumulative_motion_map
                        if bool(colmap_runtime.get("colmap_motion_aware_selection", True))
                        else None
                    ),
                    final_target_policy=str(colmap_runtime.get("colmap_final_target_policy", "soft_auto")),
                    final_soft_min=int(colmap_runtime.get("colmap_final_soft_min", 80)),
                    final_soft_max=int(colmap_runtime.get("colmap_final_soft_max", 220)),
                    no_supplement_on_low_quality=bool(colmap_runtime.get("colmap_no_supplement_on_low_quality", True)),
                )
                post_retarget_count = int(retarget.get("post_retarget_count", len(keyframes)))
                retarget_reason = str(retarget.get("retarget_reason", "n/a"))
                coverage_after = self._compute_temporal_coverage(keyframes, total_frames=int(total_frames))
                coverage_bins_after_final = self._count_time_bins_occupied(
                    [int(k.frame_index) for k in keyframes],
                    total_frames=int(total_frames),
                    bins=24,
                )
                logger.info(
                    "keyframe_retarget, "
                    f"policy={colmap_runtime['policy']}, "
                    f"final_policy={str(retarget.get('final_target_policy', colmap_runtime.get('colmap_final_target_policy', 'soft_auto')))}, "
                    f"target_mode={str(retarget.get('target_mode', colmap_runtime['target_mode']))}, "
                    f"target={int(retarget.get('effective_target_min', colmap_runtime['target_min']))}"
                    f"-{int(retarget.get('effective_target_max', colmap_runtime['target_max']))}, "
                    f"before={pre_retarget_count}, after={post_retarget_count}, "
                    f"coverage={coverage_before.get('coverage_ratio', 0.0):.3f}->{coverage_after.get('coverage_ratio', 0.0):.3f}, "
                    f"max_gap={coverage_before.get('max_gap_frames', 0)}->{coverage_after.get('max_gap_frames', 0)}, "
                    f"reason={retarget_reason}, reject={retarget.get('final_reject_reason', '')}, auto={retarget.get('auto_target', {})}"
                )
            for kf in keyframes:
                selection_trace_rows.append(
                    {
                        "frame": int(kf.frame_index),
                        "stage": "final",
                        "kept": 1,
                        "reason": "selected",
                        "score_quality": float(kf.quality_scores.get("quality", 0.0) if isinstance(kf.quality_scores, dict) else 0.0),
                        "score_novelty_ssim": "",
                        "score_novelty_phash": "",
                        "score_combined": float(kf.combined_score),
                    }
                )
            self.last_selection_runtime.update(
                {
                    "pre_retarget_count": int(pre_retarget_count),
                    "post_retarget_count": int(post_retarget_count),
                    "retarget_reason": retarget_reason,
                    "final_target_policy": str(retarget.get("final_target_policy", colmap_runtime.get("colmap_final_target_policy", "soft_auto"))),
                    "final_target_soft_range": [
                        int(colmap_runtime.get("colmap_final_soft_min", 80)),
                        int(colmap_runtime.get("colmap_final_soft_max", 220)),
                    ],
                    "final_reject_reason": str(retarget.get("final_reject_reason", "")),
                    "effective_target_min": int(
                        retarget.get("effective_target_min", colmap_runtime["target_min"])
                    ) if colmap_runtime["colmap_shortcut"] else int(colmap_runtime["target_min"]),
                    "effective_target_max": int(
                        retarget.get("effective_target_max", colmap_runtime["target_max"])
                    ) if colmap_runtime["colmap_shortcut"] else int(colmap_runtime["target_max"]),
                    "auto_target": dict(retarget.get("auto_target", {})) if colmap_runtime["colmap_shortcut"] else {},
                    "coverage_before": dict(coverage_before),
                    "coverage_after": dict(coverage_after),
                    "motion_bins_occupied_before": int(retarget.get("motion_bins_occupied_before", 0)),
                    "motion_bins_occupied_after": int(retarget.get("motion_bins_occupied_after", 0)),
                    "coverage_bins_before_final": int(coverage_bins_before_final),
                    "coverage_bins_after_final": int(coverage_bins_after_final),
                }
            )
            selection_trace_path = None
            if selection_trace_rows:
                try:
                    selection_trace_path = store.run_dir / "selection_trace.csv"
                    self._save_selection_trace_csv(selection_trace_path, selection_trace_rows)
                    logger.info(f"selection_trace_saved, path={selection_trace_path}")
                except Exception:
                    logger.debug("selection trace save failed")
            analysis_summary_payload = {
                "analysis_run_id": str(run_id),
                "analysis_mode": str(analysis_mode),
                "minimal_mode": bool(colmap_runtime.get("minimal_mode", False)),
                "disabled_components": list(colmap_runtime.get("disabled_components", [])),
                "total_frames": int(total_frames),
                "fps": float(metadata.fps),
                "stage_plan": str(colmap_runtime.get("effective_stage_plan", "Stage1->Stage2->Stage3")),
                "colmap_runtime": dict(colmap_runtime),
                "stage_counts": {
                    "stage1_candidates_raw": int(len(stage1_candidates_raw)),
                    "stage1_candidates_effective": int(len(stage1_candidates)),
                    "stage1_candidates": int(len(stage1_candidates)),
                    "stage15_entry_count": int(
                        0 if colmap_runtime["minimal_mode"] else stage15_info.get("entry_count", len(stage1_candidates))
                    ),
                    "stage1_records": int(len(stage1_records)),
                    "stage0_samples": int(len(stage0_metrics)),
                    "stage2_candidates": int(len(stage2_candidates)),
                    "stage2_records": int(len(stage2_records)),
                    "stage2_read_success_count": int(stage2_read_success_count),
                    "stage3_keyframes_pre_retarget": int(pre_retarget_count),
                    "final_keyframes": int(
                        stage2_read_success_count if colmap_runtime["minimal_mode"] else len(keyframes)
                    ),
                    "stage0_executed_count": int(self.last_selection_runtime.get("stage0_executed_count", 0)),
                    "stage3_executed_count": int(self.last_selection_runtime.get("stage3_executed_count", 0)),
                },
                "stage1_candidates_raw": int(len(stage1_candidates_raw)),
                "stage1_candidates_effective": int(len(stage1_candidates)),
                "stage15_entry_count": int(
                    0 if colmap_runtime["minimal_mode"] else stage15_info.get("entry_count", len(stage1_candidates))
                ),
                "stage15_drop_reason_counts": dict(stage15_info.get("drop_reason_counts", {})),
                "stage2_read_success_count": int(stage2_read_success_count),
                "stage1_adaptive_threshold_base": float(stage1_adaptive_info.get("base_threshold", 0.0)),
                "stage1_adaptive_threshold_effective": float(stage1_adaptive_info.get("effective_threshold", 0.0)),
                "stage1_q_pass_target": float(stage1_adaptive_info.get("q_pass_target", 0.0)),
                "stage1_bin_floor_added_count": int(stage1_adaptive_info.get("bin_floor_added_count", 0)),
                "stage1_lr_merge_mode": str(stage1_lr_summary.get("stage1_lr_merge_mode", "")),
                "stage1_lr_merge_counts": dict(stage1_lr_summary.get("stage1_lr_merge_counts", {})),
                "stage1_lens_pass_matrix": dict(stage1_lr_summary.get("stage1_lens_pass_matrix", {})),
                "stage1_sky_asymmetry_stats": dict(stage1_lr_summary.get("stage1_sky_asymmetry_stats", {})),
                "motion_median_step": float(motion_map_info.get("motion_median_step", 0.0)),
                "effective_motion_window": float(effective_motion_window),
                "motion_bins_occupied_before": int(retarget.get("motion_bins_occupied_before", 0)),
                "motion_bins_occupied_after_retarget": int(retarget.get("motion_bins_occupied_after", 0)),
                "coverage_bins_before_stage15": int(stage15_info.get("coverage_bins_before", 0)),
                "coverage_bins_after_stage15": int(stage15_info.get("coverage_bins_after", 0)),
                "coverage_bins_before_final": int(coverage_bins_before_final),
                "coverage_bins_after_final": int(coverage_bins_after_final),
                "stage2_drop_reason_counts": dict(stage2_drop_reason_counts),
                "final_target_policy": str(retarget.get("final_target_policy", colmap_runtime.get("colmap_final_target_policy", "soft_auto"))),
                "final_target_soft_range": [
                    int(colmap_runtime.get("colmap_final_soft_min", 80)),
                    int(colmap_runtime.get("colmap_final_soft_max", 220)),
                ],
                "final_reject_reason": str(retarget.get("final_reject_reason", "")),
                "retarget": {
                    "reason": str(retarget_reason),
                    "details": dict(retarget),
                    "coverage_before": dict(coverage_before),
                    "coverage_after": dict(coverage_after),
                },
                "selection_trace_csv": str(selection_trace_path) if selection_trace_path is not None else "",
                "selection_runtime": dict(self.last_selection_runtime),
            }
            summary_path = None
            try:
                summary_path = store.save_analysis_summary(analysis_summary_payload)
                self.last_selection_runtime["analysis_summary_path"] = str(summary_path)
                logger.info(f"analysis_summary_saved, path={summary_path}")
            except Exception:
                logger.debug("analysis summary save failed")

            if frame_log_callback:
                selected_idx = {kf.frame_index for kf in keyframes}
                for record in stage2_records:
                    record.is_keyframe = record.frame_index in selected_idx
                    record.metrics["keyframe_flag"] = 1.0 if record.is_keyframe else 0.0
                    record.metrics["stage3_selected_flag"] = 1.0 if record.is_keyframe else 0.0
                    self._emit_frame_log(
                        frame_log_callback,
                        {
                            "frame_index": record.frame_index,
                            "frame": record.frame,
                            "is_keyframe": record.is_keyframe,
                            "metrics": record.metrics,
                            "quality_scores": record.quality_scores,
                            "geometric_scores": record.geometric_scores,
                            "adaptive_scores": record.adaptive_scores,
                            "t_xyz": record.t_xyz,
                            "q_wxyz": record.q_wxyz,
                            "points_world": record.points_world,
                        },
                    )

            logger.info(f"最終キーフレーム数: {len(keyframes)}個")
            mean_score = float(np.mean([kf.combined_score for kf in keyframes])) if keyframes else 0.0
            logger.info(
                f"analysis_result, analysis_run_id={run_id}, mode={analysis_mode},"
                f" keyframes={len(keyframes)}, mean_score={mean_score:.4f}"
            )
            success = True
            return keyframes
        except Exception as e:
            try:
                store.mark_failed(current_stage, str(e))
            except Exception:
                pass
            raise
        finally:
            if success:
                if keep_temp_on_success:
                    try:
                        store.mark_retained_on_success()
                    except Exception:
                        logger.debug("stage temp retain manifest update failed")
                    logger.info(f"stage tempを保持します（KEEP_TEMP_ON_SUCCESS=true, path={store.run_dir}）")
                else:
                    try:
                        store.cleanup_on_success()
                    except Exception:
                        logger.debug("stage temp cleanup failed")

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
        grab_threshold = int(self.config.get('STAGE1_GRAB_THRESHOLD', 30))
        use_grab = (sample_interval > 1 and sample_interval <= grab_threshold)
        frame_indices = list(range(0, total_frames, sample_interval))
        fps = float(getattr(metadata, "fps", 0.0) or 30.0)

        candidates: List[Dict[str, Any]] = []
        self.stage1_quality_records = []

        quality_filter_enabled = bool(self.config.get('QUALITY_FILTER_ENABLED', True))
        quality_threshold = float(np.clip(self.config.get('QUALITY_THRESHOLD', 0.50), 0.0, 1.0))
        quality_abs_laplacian_min = float(max(0.0, self.config.get('QUALITY_ABS_LAPLACIAN_MIN', 35.0)))
        quality_use_orb = bool(self.config.get('QUALITY_USE_ORB', True))
        quality_norm_p_low = float(np.clip(self.config.get('QUALITY_NORM_P_LOW', 10.0), 0.0, 100.0))
        quality_norm_p_high = float(
            np.clip(self.config.get('QUALITY_NORM_P_HIGH', 90.0), quality_norm_p_low, 100.0)
        )
        quality_debug = bool(self.config.get('QUALITY_DEBUG', False))
        quality_tenengrad_scale = float(np.clip(self.config.get('QUALITY_TENENGRAD_SCALE', 1.0), 0.1, 1.0))
        roi_mode = str(self.config.get('QUALITY_ROI_MODE', 'circle')).strip().lower()
        roi_ratio = float(np.clip(self.config.get('QUALITY_ROI_RATIO', 0.40), 0.05, 1.0))
        roi_spec = parse_roi_spec(f"{roi_mode}:{roi_ratio}")
        quality_weights = {
            "quality_weight_sharpness": float(max(0.0, self.config.get('QUALITY_WEIGHT_SHARPNESS', 0.40))),
            "quality_weight_tenengrad": float(max(0.0, self.config.get('QUALITY_WEIGHT_TENENGRAD', 0.30))),
            "quality_weight_exposure": float(max(0.0, self.config.get('QUALITY_WEIGHT_EXPOSURE', 0.15))),
            "quality_weight_keypoints": float(max(0.0, self.config.get('QUALITY_WEIGHT_KEYPOINTS', 0.15))),
        }
        quality_fields = ("laplacian_var", "tenengrad", "exposure", "orb_keypoints")
        stage1_lr_merge_mode = str(
            self.config.get("STAGE1_LR_MERGE_MODE", self.config.get("stage1_lr_merge_mode", "asymmetric_sky_v1")) or "asymmetric_sky_v1"
        ).strip().lower()
        if stage1_lr_merge_mode not in {"asymmetric_sky_v1", "strict_min"}:
            stage1_lr_merge_mode = "asymmetric_sky_v1"
        stage1_lr_asym_weak_floor = float(
            np.clip(
                self.config.get("STAGE1_LR_ASYM_WEAK_FLOOR", self.config.get("stage1_lr_asym_weak_floor", 0.35)),
                0.0,
                1.0,
            )
        )
        stage1_lr_sky_ratio_threshold = float(
            np.clip(
                self.config.get("STAGE1_LR_SKY_RATIO_THRESHOLD", self.config.get("stage1_lr_sky_ratio_threshold", 0.55)),
                0.0,
                1.0,
            )
        )
        stage1_lr_sky_ratio_diff_threshold = float(
            np.clip(
                self.config.get(
                    "STAGE1_LR_SKY_RATIO_DIFF_THRESHOLD",
                    self.config.get("stage1_lr_sky_ratio_diff_threshold", 0.20),
                ),
                0.0,
                1.0,
            )
        )
        stage1_lr_quality_gap_threshold = float(
            np.clip(
                self.config.get(
                    "STAGE1_LR_QUALITY_GAP_THRESHOLD",
                    self.config.get("stage1_lr_quality_gap_threshold", 0.15),
                ),
                0.0,
                1.0,
            )
        )
        stage1_lr_semantic_sky_enabled = bool(
            self.config.get(
                "STAGE1_LR_SEMANTIC_SKY_ENABLED",
                self.config.get("stage1_lr_semantic_sky_enabled", True),
            )
        )

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired
        use_fisheye_border_mask = self._is_fisheye_border_mask_enabled(is_paired)

        if is_paired:
            pair_entries: List[Dict[str, Any]] = []
            cap_a, cap_b = self._open_independent_pair_captures(video_loader)
            try:
                last_read_idx = -1
                current_pos = -1
                if use_grab and frame_indices and cap_a is not None and cap_b is not None:
                    first_idx = frame_indices[0]
                    cap_a.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
                    cap_b.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
                    current_pos = first_idx

                cached_mask_a = None
                cached_mask_b = None
                cached_shape_a = None
                cached_shape_b = None

                for idx, frame_idx in enumerate(frame_indices):
                    if cap_a is not None and cap_b is not None:
                        if use_grab:
                            while current_pos < frame_idx:
                                ok_a = cap_a.grab()
                                ok_b = cap_b.grab()
                                if not ok_a or not ok_b:
                                    break
                                current_pos += 1
                            if current_pos != frame_idx:
                                cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                current_pos = frame_idx
                        else:
                            if frame_idx != last_read_idx + 1:
                                cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                        ret_a, frame_a = cap_a.read()
                        ret_b, frame_b = cap_b.read()
                        if use_grab:
                            current_pos += 1
                        last_read_idx = frame_idx
                        if not ret_a or not ret_b or frame_a is None or frame_b is None:
                            continue
                    else:
                        frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                        if frame_a is None or frame_b is None:
                            continue

                    if use_fisheye_border_mask:
                        shape_a = frame_a.shape[:2]
                        shape_b = frame_b.shape[:2]
                        if cached_mask_a is None or cached_shape_a != shape_a:
                            cached_mask_a = self._create_fisheye_valid_mask(frame_a)
                            cached_shape_a = shape_a
                        if cached_mask_b is None or cached_shape_b != shape_b:
                            cached_mask_b = self._create_fisheye_valid_mask(frame_b)
                            cached_shape_b = shape_b
                        frame_a = self.mask_processor.apply_valid_region_mask(frame_a, cached_mask_a, fill_value=0)
                        frame_b = self.mask_processor.apply_valid_region_mask(frame_b, cached_mask_b, fill_value=0)

                    timestamp = float(frame_idx / max(fps, 1e-6))
                    if quality_filter_enabled:
                        raw_a = compute_raw_metrics(
                            frame_a,
                            roi_spec=roi_spec,
                            use_orb=quality_use_orb,
                            tenengrad_scale=quality_tenengrad_scale,
                        )
                        raw_b = compute_raw_metrics(
                            frame_b,
                            roi_spec=roi_spec,
                            use_orb=quality_use_orb,
                            tenengrad_scale=quality_tenengrad_scale,
                        )
                        pair_entries.append(
                            {
                                "frame_idx": int(frame_idx),
                                "timestamp": timestamp,
                                "lens_a_raw": raw_a,
                                "lens_b_raw": raw_b,
                                "sky_ratio_lens_a": (
                                    self._estimate_sky_ratio(frame_a, cached_mask_a if use_fisheye_border_mask else None)
                                    if stage1_lr_semantic_sky_enabled else 0.0
                                ),
                                "sky_ratio_lens_b": (
                                    self._estimate_sky_ratio(frame_b, cached_mask_b if use_fisheye_border_mask else None)
                                    if stage1_lr_semantic_sky_enabled else 0.0
                                ),
                            }
                        )
                    else:
                        quality_scores = self._compute_quality_score_pair(frame_a, frame_b)
                        if bool(quality_scores.get('passes_threshold', False)):
                            candidates.append({'frame_idx': frame_idx, 'quality_scores': quality_scores})
                        self.stage1_quality_records.append(
                            {
                                "frame_index": int(frame_idx),
                                "timestamp": timestamp,
                                "quality": float(quality_scores.get("quality", 0.0)),
                                "is_pass": bool(quality_scores.get('passes_threshold', False)),
                                "drop_reason": "pass" if bool(quality_scores.get('passes_threshold', False)) else "legacy_threshold",
                                "legacy_quality_scores": dict(quality_scores),
                                "raw_metrics": {},
                                "norm_metrics": {},
                            }
                        )

                    if progress_callback:
                        progress_callback(idx + 1, len(frame_indices))
            finally:
                if cap_a is not None:
                    cap_a.release()
                if cap_b is not None:
                    cap_b.release()

            if not quality_filter_enabled:
                return candidates

            raw_a_list = [entry["lens_a_raw"] for entry in pair_entries]
            raw_b_list = [entry["lens_b_raw"] for entry in pair_entries]
            norm_a_list, stats_a = normalize_batch_p10_p90(
                raw_a_list,
                p_low=quality_norm_p_low,
                p_high=quality_norm_p_high,
                fields=quality_fields,
            )
            norm_b_list, stats_b = normalize_batch_p10_p90(
                raw_b_list,
                p_low=quality_norm_p_low,
                p_high=quality_norm_p_high,
                fields=quality_fields,
            )

            effective_sky_ratio_threshold = float(stage1_lr_sky_ratio_threshold)
            effective_weak_floor = float(stage1_lr_asym_weak_floor)
            stage1_lr_auto_relaxed = False
            if stage1_lr_merge_mode == "asymmetric_sky_v1" and stage1_lr_semantic_sky_enabled and pair_entries:
                sky_max_values = [
                    float(
                        max(
                            float(entry.get("sky_ratio_lens_a", 0.0)),
                            float(entry.get("sky_ratio_lens_b", 0.0)),
                        )
                    )
                    for entry in pair_entries
                ]
                sky_max_arr = np.asarray(sky_max_values, dtype=np.float64)
                observed_sky_max = float(np.max(sky_max_arr)) if sky_max_arr.size > 0 else 0.0
                if observed_sky_max + 1e-9 < effective_sky_ratio_threshold:
                    observed_sky_p90 = float(np.percentile(sky_max_arr, 90)) if sky_max_arr.size > 0 else observed_sky_max
                    effective_sky_ratio_threshold = float(
                        np.clip(
                            max(0.20, min(0.35, observed_sky_p90)),
                            0.0,
                            1.0,
                        )
                    )
                    # Keep weak-side quality guard, but relax one step to avoid dead zone.
                    effective_weak_floor = float(min(effective_weak_floor, 0.30))
                    stage1_lr_auto_relaxed = True
                    logger.warning(
                        "stage1_lr_auto_relax_applied,"
                        f" configured_sky_threshold={stage1_lr_sky_ratio_threshold:.3f},"
                        f" effective_sky_threshold={effective_sky_ratio_threshold:.3f},"
                        f" configured_weak_floor={stage1_lr_asym_weak_floor:.3f},"
                        f" effective_weak_floor={effective_weak_floor:.3f},"
                        f" observed_sky_max={observed_sky_max:.3f},"
                        f" observed_sky_p90={observed_sky_p90:.3f}"
                    )

            for entry, norm_a, norm_b in zip(pair_entries, norm_a_list, norm_b_list):
                frame_idx = int(entry["frame_idx"])
                raw_a = entry["lens_a_raw"]
                raw_b = entry["lens_b_raw"]
                sky_ratio_a = float(np.clip(entry.get("sky_ratio_lens_a", 0.0), 0.0, 1.0))
                sky_ratio_b = float(np.clip(entry.get("sky_ratio_lens_b", 0.0), 0.0, 1.0))
                quality_a = compose_quality(norm_a, quality_weights)
                quality_b = compose_quality(norm_b, quality_weights)
                abs_ok_a = apply_abs_guard(raw_a.get("laplacian_var", 0.0), quality_abs_laplacian_min)
                abs_ok_b = apply_abs_guard(raw_b.get("laplacian_var", 0.0), quality_abs_laplacian_min)

                quality_min = float(min(quality_a, quality_b))
                quality_max = float(max(quality_a, quality_b))
                quality_gap = float(quality_max - quality_min)
                sky_max = float(max(sky_ratio_a, sky_ratio_b))
                sky_diff = float(abs(sky_ratio_a - sky_ratio_b))
                dominant_lens = "a" if quality_a >= quality_b else "b"
                weak_lens = "b" if dominant_lens == "a" else "a"
                asym_eligible = bool(
                    stage1_lr_merge_mode == "asymmetric_sky_v1"
                    and quality_gap >= stage1_lr_quality_gap_threshold
                    and sky_max >= effective_sky_ratio_threshold
                    and sky_diff >= stage1_lr_sky_ratio_diff_threshold
                )

                if asym_eligible:
                    quality = quality_max
                    abs_ok = bool(abs_ok_a if dominant_lens == "a" else abs_ok_b)
                    weak_quality_ok = bool(quality_min >= effective_weak_floor)
                    quality_merge_strategy = "asymmetric_max_with_weak_floor"
                    merge_mode_applied = "asymmetric_sky_v1"
                else:
                    quality = quality_min
                    abs_ok = bool(abs_ok_a and abs_ok_b)
                    weak_quality_ok = True
                    quality_merge_strategy = "strict_min"
                    merge_mode_applied = "strict_min"

                quality_ok = bool(quality >= quality_threshold)
                passes = bool(quality_ok and abs_ok and weak_quality_ok)
                drop_reason = "pass"
                if not abs_ok:
                    drop_reason = "abs_laplacian_guard"
                elif not weak_quality_ok:
                    drop_reason = "lr_weak_floor"
                elif not quality_ok:
                    drop_reason = "quality_below_threshold"

                quality_scores = {
                    "sharpness": float(
                        raw_a.get("laplacian_var", 0.0) if asym_eligible and dominant_lens == "a"
                        else raw_b.get("laplacian_var", 0.0) if asym_eligible and dominant_lens == "b"
                        else min(raw_a.get("laplacian_var", 0.0), raw_b.get("laplacian_var", 0.0))
                    ),
                    "exposure": float(
                        raw_a.get("exposure", 0.0) if asym_eligible and dominant_lens == "a"
                        else raw_b.get("exposure", 0.0) if asym_eligible and dominant_lens == "b"
                        else min(raw_a.get("exposure", 0.0), raw_b.get("exposure", 0.0))
                    ),
                    "motion_blur": 0.0,
                    "softmax_depth": 0.0,
                    "quality": float(quality),
                    "quality_lens_a": float(quality_a),
                    "quality_lens_b": float(quality_b),
                    "sky_ratio_lens_a": float(sky_ratio_a),
                    "sky_ratio_lens_b": float(sky_ratio_b),
                    "lr_asym_eligible": bool(asym_eligible),
                    "lr_dominant_lens": str(dominant_lens),
                    "lr_weak_lens": str(weak_lens),
                    "lr_weak_floor": float(effective_weak_floor),
                    "lr_sky_ratio_threshold": float(effective_sky_ratio_threshold),
                    "lr_auto_relaxed": bool(stage1_lr_auto_relaxed),
                    "quality_merge_strategy": str(quality_merge_strategy),
                    "lr_merge_mode_applied": str(merge_mode_applied),
                    "lens_a": {
                        "raw": dict(raw_a),
                        "norm": {k: float(norm_a.get(f"norm_{k}", 0.0)) for k in quality_fields},
                    },
                    "lens_b": {
                        "raw": dict(raw_b),
                        "norm": {k: float(norm_b.get(f"norm_{k}", 0.0)) for k in quality_fields},
                    },
                    "passes_threshold": passes,
                }
                if passes:
                    candidates.append({'frame_idx': frame_idx, 'quality_scores': quality_scores})

                self.stage1_quality_records.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp": float(entry["timestamp"]),
                        "quality": float(quality),
                        "quality_lens_a": float(quality_a),
                        "quality_lens_b": float(quality_b),
                        "is_pass": passes,
                        "drop_reason": drop_reason,
                        "sky_ratio_lens_a": float(sky_ratio_a),
                        "sky_ratio_lens_b": float(sky_ratio_b),
                        "lr_asym_eligible": bool(asym_eligible),
                        "lr_dominant_lens": str(dominant_lens),
                        "lr_weak_lens": str(weak_lens),
                        "lr_weak_floor": float(effective_weak_floor),
                        "lr_sky_ratio_threshold": float(effective_sky_ratio_threshold),
                        "lr_auto_relaxed": bool(stage1_lr_auto_relaxed),
                        "quality_merge_strategy": str(quality_merge_strategy),
                        "lr_merge_mode_applied": str(merge_mode_applied),
                        "lens_a_raw": dict(raw_a),
                        "lens_a_norm": {k: float(norm_a.get(f"norm_{k}", 0.0)) for k in quality_fields},
                        "lens_b_raw": dict(raw_b),
                        "lens_b_norm": {k: float(norm_b.get(f"norm_{k}", 0.0)) for k in quality_fields},
                    }
                )

            if quality_debug:
                passed_count = sum(1 for rec in self.stage1_quality_records if bool(rec.get("is_pass", False)))
                qualities = np.asarray(
                    [float(rec.get("quality", 0.0)) for rec in self.stage1_quality_records],
                    dtype=np.float64,
                )
                q50 = float(np.median(qualities)) if qualities.size > 0 else 0.0
                logger.info(
                    "stage1_quality_debug,"
                    f" mode=paired, records={len(self.stage1_quality_records)},"
                    f" passed={passed_count}, pass_ratio={100.0*passed_count/max(len(self.stage1_quality_records), 1):.2f},"
                    f" threshold={quality_threshold:.3f}, abs_laplacian_min={quality_abs_laplacian_min:.3f},"
                    f" lr_mode={stage1_lr_merge_mode}, weak_floor={effective_weak_floor:.3f},"
                    f" sky_thr={effective_sky_ratio_threshold:.3f}, sky_diff_thr={stage1_lr_sky_ratio_diff_threshold:.3f},"
                    f" lr_auto_relaxed={stage1_lr_auto_relaxed},"
                    f" q_gap_thr={stage1_lr_quality_gap_threshold:.3f},"
                    f" q50={q50:.4f}, roi={roi_spec.mode}:{roi_spec.ratio:.2f},"
                    f" p_low={quality_norm_p_low:.1f}, p_high={quality_norm_p_high:.1f},"
                    f" stats_a={stats_a}, stats_b={stats_b}"
                )
            return candidates

        video_path = getattr(video_loader, "_video_path", None)
        if video_path is None:
            raise RuntimeError("単眼モードのビデオパスが取得できません")

        from .stage1_engine import run_stage1_mono_scan

        result = run_stage1_mono_scan(
            video_path=str(video_path),
            config=self.config,
            sample_interval=sample_interval,
            is_running_cb=lambda: True,
            on_progress_cb=progress_callback,
        )
        self.stage1_quality_records = list(result.get("records", []))
        return list(result.get("candidates", []))

    def _stage2_precise_evaluation(
        self,
        video_loader,
        metadata: VideoMetadata,
        stage1_candidates: List[Dict],
        progress_callback: Optional[Callable[[int, int], None]],
        frame_log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stage0_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[List[KeyframeInfo], List[Stage2FrameRecord]]:
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
        tuple
            (精密評価後のキーフレーム候補, 可視化用フレーム記録)
        """
        candidates = []
        records: List[Stage2FrameRecord] = []
        last_keyframe_idx = -self.config['MIN_KEYFRAME_INTERVAL']
        last_keyframe = None
        last_pair = None

        # フレームウィンドウ（カメラ加速度計算用）
        frame_window = deque(maxlen=5)
        geom_frame_window = deque(maxlen=max(2, int(self.config.get('DYNAMIC_MASK_MOTION_FRAMES', 3))))

        is_paired = hasattr(video_loader, 'is_paired') and video_loader.is_paired
        use_fisheye_border_mask = self._is_fisheye_border_mask_enabled(is_paired)
        cap = None
        cap_a = None
        cap_b = None
        if not is_paired:
            video_path = getattr(video_loader, "_video_path", None)
            if video_path is None:
                raise RuntimeError("単眼モードのビデオパスが取得できません")
            cap = self._open_independent_capture(video_path)
        else:
            cap_a, cap_b = self._open_independent_pair_captures(video_loader)
        perf_stats = Stage2PerfStats(
            enabled=bool(self.config.get('STAGE2_PERF_PROFILE', True)),
            total_candidates=len(stage1_candidates),
        )
        stage2_started = perf_counter()
        dynamic_mask_cache: Dict[int, np.ndarray] = {}

        def _build_metrics(
            q_scores: Dict[str, float],
            g_scores: Dict[str, float],
            a_scores: Dict[str, float],
            combined: float,
            is_keyframe_flag: float = 0.0,
            stage0_motion_risk: float = 0.0,
            vo_status_reason: str = "not_evaluated",
        ) -> Dict[str, Any]:
            ssim = float(a_scores.get("ssim_pair", a_scores.get("ssim", 1.0)))
            flow_mag = float(a_scores.get("optical_flow", 0.0))
            return {
                "translation_delta": flow_mag,
                "rotation_delta": float(max(0.0, 1.0 - ssim) * 180.0),
                "flow_mag": flow_mag,
                "vo_step_proxy": 0.0,
                "vo_step_proxy_norm": 0.0,
                "vo_inlier_ratio": 0.0,
                "vo_rot_deg": float(max(0.0, 1.0 - ssim) * 180.0),
                "vo_dir_cos_prev": 0.5,
                "vo_confidence": 0.0,
                "vo_feature_uniformity": 0.0,
                "vo_track_sufficiency": 0.0,
                "vo_pose_plausibility": 0.0,
                "vo_tracked_count": 0.0,
                "vo_essential_method": "none",
                "laplacian_var": float(q_scores.get("sharpness", 0.0)),
                "match_count": float(g_scores.get("feature_match_count", q_scores.get("seam_features_cur", 0.0))),
                "overlap_ratio": float(g_scores.get("gric", 0.0)),
                "exposure_ratio": float(q_scores.get("exposure", 0.0)),
                "keyframe_flag": float(is_keyframe_flag),
                "combined_score": float(combined),
                "stationary_vo_flag": 0.0,
                "stationary_flow_flag": 0.0,
                "is_stationary": 0.0,
                "stationary_confidence": 0.0,
                "stationary_penalty_applied": 0.0,
                "stage0_motion_risk": float(np.clip(stage0_motion_risk, 0.0, 1.0)),
                "trajectory_consistency": 0.5,
                "trajectory_consistency_effective": 0.25,
                "combined_stage2": float(combined),
                "combined_stage3": float(combined),
                "stage3_selected_flag": 0.0,
                "vo_status_reason": str(vo_status_reason),
                "vo_pose_valid": 0.0,
                "vo_attempted": 0.0,
            }

        def _append_record(
            frame_idx: int,
            frame_img: Optional[np.ndarray],
            q_scores: Dict[str, float],
            g_scores: Dict[str, float],
            a_scores: Dict[str, float],
            metrics: Dict[str, float],
            is_candidate: bool,
            drop_reason: str,
        ) -> None:
            keep_frame = bool(self.config.get('STAGE2_DEFERRED_FRAME_LOG_IMAGES', False))
            records.append(
                Stage2FrameRecord(
                    frame_index=int(frame_idx),
                    frame=frame_img.copy() if (keep_frame and frame_img is not None) else None,
                    quality_scores=dict(q_scores),
                    geometric_scores=dict(g_scores),
                    adaptive_scores=dict(a_scores),
                    metrics=dict(metrics),
                    is_candidate=is_candidate,
                    drop_reason=str(drop_reason),
                )
            )

        try:
            last_read_idx = -1
            last_read_idx_pair = -1
            last_pair_feature_mask = None
            for idx, candidate_info in enumerate(stage1_candidates):
                frame_idx = candidate_info['frame_idx']
                quality_scores = candidate_info['quality_scores']
                stage0_entry = self._lookup_stage0_metrics(frame_idx, stage0_metrics or {})
                stage0_motion_risk = float(np.clip(stage0_entry.get("motion_risk", 0.0), 0.0, 1.0))
                vo_status_reason = str(stage0_entry.get("vo_status_reason", "not_evaluated"))

                if progress_callback:
                    progress_callback(idx, len(stage1_candidates))

                current_frame = None
                current_pair = None
                current_pair_feature_mask = None
                frame_read_started = perf_counter() if perf_stats.enabled else 0.0
                if is_paired:
                    if cap_a is not None and cap_b is not None:
                        if frame_idx != last_read_idx_pair + 1:
                            cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret_a, frame_a = cap_a.read()
                        ret_b, frame_b = cap_b.read()
                        last_read_idx_pair = frame_idx
                        if not ret_a or not ret_b or frame_a is None or frame_b is None:
                            combined = float(self._compute_combined_score(quality_scores, {}, {}))
                            metrics = _build_metrics(
                                quality_scores, {}, {}, combined,
                                is_keyframe_flag=0.0, stage0_motion_risk=stage0_motion_risk, vo_status_reason=vo_status_reason
                            )
                            _append_record(
                                frame_idx=frame_idx,
                                frame_img=None,
                                q_scores=quality_scores,
                                g_scores={},
                                a_scores={},
                                metrics=metrics,
                                is_candidate=False,
                                drop_reason="read_fail",
                            )
                            continue
                    else:
                        # テストダミーローダー等のフォールバック
                        frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                        if frame_a is None or frame_b is None:
                            combined = float(self._compute_combined_score(quality_scores, {}, {}))
                            metrics = _build_metrics(
                                quality_scores, {}, {}, combined,
                                is_keyframe_flag=0.0, stage0_motion_risk=stage0_motion_risk, vo_status_reason=vo_status_reason
                            )
                            _append_record(
                                frame_idx=frame_idx,
                                frame_img=None,
                                q_scores=quality_scores,
                                g_scores={},
                                a_scores={},
                                metrics=metrics,
                                is_candidate=False,
                                drop_reason="read_fail",
                            )
                            continue
                    if use_fisheye_border_mask:
                        valid_mask_a = self._create_fisheye_valid_mask(frame_a)
                        valid_mask_b = self._create_fisheye_valid_mask(frame_b)
                        frame_a = self.mask_processor.apply_valid_region_mask(frame_a, valid_mask_a, fill_value=0)
                        frame_b = self.mask_processor.apply_valid_region_mask(frame_b, valid_mask_b, fill_value=0)
                        current_pair_feature_mask = valid_mask_a
                    current_pair = (frame_a, frame_b)
                    current_frame = frame_a
                else:
                    if frame_idx != last_read_idx + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, current_frame = cap.read()
                    last_read_idx = frame_idx
                    if not ret or current_frame is None:
                        combined = float(self._compute_combined_score(quality_scores, {}, {}))
                        metrics = _build_metrics(
                            quality_scores, {}, {}, combined,
                            is_keyframe_flag=0.0, stage0_motion_risk=stage0_motion_risk, vo_status_reason=vo_status_reason
                        )
                        _append_record(
                            frame_idx=frame_idx,
                            frame_img=None,
                            q_scores=quality_scores,
                            g_scores={},
                            a_scores={},
                            metrics=metrics,
                            is_candidate=False,
                            drop_reason="read_fail",
                        )
                        continue
                if perf_stats.enabled:
                    perf_stats.frame_read_s += max(0.0, perf_counter() - frame_read_started)
                perf_stats.processed_candidates += 1

                geometric_scores = {}
                adaptive_scores = {}
                force_insert = False
                exposure_changed = False
                is_selected = False
                dynamic_mask_cur = None

                if frame_idx - last_keyframe_idx < self.config['MIN_KEYFRAME_INTERVAL']:
                    combined = float(self._compute_combined_score(quality_scores, {}, {}))
                    metrics = _build_metrics(
                        quality_scores, geometric_scores, adaptive_scores, combined,
                        is_keyframe_flag=0.0, stage0_motion_risk=stage0_motion_risk, vo_status_reason=vo_status_reason
                    )
                    _append_record(
                        frame_idx=frame_idx,
                        frame_img=current_frame,
                        q_scores=quality_scores,
                        g_scores=geometric_scores,
                        a_scores=adaptive_scores,
                        metrics=metrics,
                        is_candidate=False,
                        drop_reason="min_interval",
                    )
                    continue

                frame_window.append(current_frame)
                exposure_changed = self._detect_exposure_change(current_frame)

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

                    dynamic_mask_ref = None
                    dynamic_mask_cur = None
                    if self.config.get('ENABLE_DYNAMIC_MASK_REMOVAL', False):
                        use_yolo = bool(self.config.get('DYNAMIC_MASK_USE_YOLO_SAM', True))
                        use_motion = bool(self.config.get('DYNAMIC_MASK_USE_MOTION_DIFF', True))
                        perf_stats.dynamic_mask_calls += 1
                        if use_yolo and use_motion:
                            perf_stats.dynamic_mask_mode_both_calls += 1
                        elif use_yolo:
                            perf_stats.dynamic_mask_mode_yolo_only_calls += 1
                        elif use_motion:
                            perf_stats.dynamic_mask_mode_motion_only_calls += 1
                        mask_started = perf_counter() if perf_stats.enabled else 0.0
                        dynamic_mask_ref, dynamic_mask_cur = self._build_dynamic_masks(
                            geom_ref, geom_cur, list(geom_frame_window),
                            frame_prev_idx=last_keyframe_idx,
                            frame_cur_idx=frame_idx,
                            mask_cache=dynamic_mask_cache,
                        )
                        if perf_stats.enabled:
                            perf_stats.dynamic_mask_s += max(0.0, perf_counter() - mask_started)
                    if is_paired and use_fisheye_border_mask and not self.config.get('ENABLE_RIG_STITCHING', True):
                        dynamic_mask_ref = self._combine_feature_masks(dynamic_mask_ref, last_pair_feature_mask)
                        dynamic_mask_cur = self._combine_feature_masks(dynamic_mask_cur, current_pair_feature_mask)

                    try:
                        geom_started = perf_counter() if perf_stats.enabled else 0.0
                        geometric_scores = self.geometric_evaluator.evaluate(
                            geom_ref, geom_cur,
                            frame1_idx=last_keyframe_idx,
                            frame2_idx=frame_idx,
                            frame1_mask=dynamic_mask_ref,
                            frame2_mask=dynamic_mask_cur,
                        )
                        if perf_stats.enabled:
                            perf_stats.geometric_eval_s += max(0.0, perf_counter() - geom_started)
                    except GeometricDegeneracyError as e:
                        if perf_stats.enabled:
                            perf_stats.geometric_eval_s += max(0.0, perf_counter() - geom_started)
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
                        if perf_stats.enabled:
                            perf_stats.geometric_eval_s += max(0.0, perf_counter() - geom_started)
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
                            combined = float(self._compute_combined_score(quality_scores, {}, {}))
                            metrics = _build_metrics(
                                quality_scores, geometric_scores, adaptive_scores, combined,
                                is_keyframe_flag=0.0, stage0_motion_risk=stage0_motion_risk, vo_status_reason=vo_status_reason
                            )
                            _append_record(
                                frame_idx=frame_idx,
                                frame_img=current_frame,
                                q_scores=quality_scores,
                                g_scores=geometric_scores,
                                a_scores=adaptive_scores,
                                metrics=metrics,
                                is_candidate=False,
                                drop_reason="geometric_fail",
                            )
                            continue

                    geom_frame_window.append(geom_cur)

                    if not geometric_failed and 'feature_match_count' in geometric_scores:
                        self.is_rescue_mode = self._check_rescue_mode(geometric_scores['feature_match_count'])

                    adaptive_started = perf_counter() if perf_stats.enabled else 0.0
                    adaptive_scores = self.adaptive_selector.evaluate(
                        last_keyframe, current_frame, frames_window=list(frame_window)
                    )

                    if is_paired and last_pair is not None and current_pair is not None:
                        flow_a = self.adaptive_selector.compute_optical_flow_magnitude(last_pair[0], current_pair[0])
                        flow_b = self.adaptive_selector.compute_optical_flow_magnitude(last_pair[1], current_pair[1])
                        ssim_a = float(adaptive_scores.get('ssim', 1.0))
                        ssim_b = float(self.adaptive_selector.compute_ssim(last_pair[1], current_pair[1]))
                        adaptive_scores['ssim_lens_a'] = ssim_a
                        adaptive_scores['ssim_lens_b'] = ssim_b
                        adaptive_scores['ssim_pair'] = float(min(ssim_a, ssim_b))
                        if self.config.get('PAIR_MOTION_AGGREGATION', 'max') == 'max':
                            adaptive_scores['optical_flow'] = max(flow_a, flow_b)
                        else:
                            adaptive_scores['optical_flow'] = (flow_a + flow_b) * 0.5
                        adaptive_scores['optical_flow_lens_a'] = flow_a
                        adaptive_scores['optical_flow_lens_b'] = flow_b
                    if perf_stats.enabled:
                        perf_stats.adaptive_eval_s += max(0.0, perf_counter() - adaptive_started)

                    ssim = adaptive_scores.get('ssim_pair', adaptive_scores.get('ssim', 1.0))
                    if ssim > self.config['SSIM_CHANGE_THRESHOLD'] and not force_insert and not exposure_changed:
                        combined = float(self._compute_combined_score(quality_scores, geometric_scores, adaptive_scores))
                        metrics = _build_metrics(
                            quality_scores, geometric_scores, adaptive_scores, combined,
                            is_keyframe_flag=0.0, stage0_motion_risk=stage0_motion_risk, vo_status_reason=vo_status_reason
                        )
                        _append_record(
                            frame_idx=frame_idx,
                            frame_img=current_frame,
                            q_scores=quality_scores,
                            g_scores=geometric_scores,
                            a_scores=adaptive_scores,
                            metrics=metrics,
                            is_candidate=False,
                            drop_reason="ssim_skip",
                        )
                        continue

                    if exposure_changed and frame_idx - last_keyframe_idx >= self.config['MIN_KEYFRAME_INTERVAL']:
                        force_insert = True
                else:
                    geometric_failed = False

                combined_score = self._compute_combined_score(
                    quality_scores, geometric_scores, adaptive_scores
                )
                if self.config.get('ENABLE_DYNAMIC_MASK_REMOVAL', False) and dynamic_mask_cur is None:
                    dynamic_mask_cur = self._build_single_frame_dynamic_mask(current_frame)

                candidate = KeyframeInfo(
                    frame_index=frame_idx,
                    timestamp=frame_idx / metadata.fps,
                    quality_scores=quality_scores,
                    geometric_scores=geometric_scores,
                    adaptive_scores=adaptive_scores,
                    combined_score=combined_score,
                    thumbnail=None,
                    is_rescue_mode=self.is_rescue_mode,
                    is_force_inserted=force_insert or exposure_changed,
                    dynamic_mask=dynamic_mask_cur.copy() if dynamic_mask_cur is not None else None,
                )
                candidates.append(candidate)
                perf_stats.selected_candidates += 1
                is_selected = True

                if self.is_rescue_mode:
                    self.rescue_mode_keyframes.append(frame_idx)

                last_keyframe_idx = frame_idx
                last_keyframe = current_frame
                last_pair = current_pair
                last_pair_feature_mask = current_pair_feature_mask

                metrics = _build_metrics(
                    quality_scores,
                    geometric_scores,
                    adaptive_scores,
                    float(combined_score),
                    is_keyframe_flag=1.0 if is_selected else 0.0,
                    stage0_motion_risk=stage0_motion_risk,
                    vo_status_reason=vo_status_reason,
                )
                _append_record(
                    frame_idx=frame_idx,
                    frame_img=current_frame,
                    q_scores=quality_scores,
                    g_scores=geometric_scores,
                    a_scores=adaptive_scores,
                    metrics=metrics,
                    is_candidate=True,
                    drop_reason="selected",
                )

        finally:
            if perf_stats.enabled:
                perf_stats.total_s = max(0.0, perf_counter() - stage2_started)
            self._log_stage2_perf_summary(perf_stats)
            if cap is not None:
                cap.release()
                logger.debug("Stage 2: 独立キャプチャを解放")
            if cap_a is not None:
                cap_a.release()
            if cap_b is not None:
                cap_b.release()

        return candidates, records

    def _stage2_minimal_motion_only_evaluation(
        self,
        video_loader,
        metadata: VideoMetadata,
        stage1_candidates: List[Dict],
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> Tuple[List[KeyframeInfo], List[Stage2FrameRecord]]:
        """
        COLMAP minimal_v1 向け Stage2。

        Stage1候補を順次読み込み、read_fail 以外をすべて採用する。
        optical_flow は記録のみで、採否判定には利用しない。
        """
        candidates: List[KeyframeInfo] = []
        records: List[Stage2FrameRecord] = []
        is_paired = hasattr(video_loader, "is_paired") and video_loader.is_paired
        use_fisheye_border_mask = self._is_fisheye_border_mask_enabled(is_paired)
        cap = None
        cap_a = None
        cap_b = None
        if not is_paired:
            video_path = getattr(video_loader, "_video_path", None)
            if video_path is None:
                raise RuntimeError("単眼モードのビデオパスが取得できません")
            cap = self._open_independent_capture(video_path)
        else:
            cap_a, cap_b = self._open_independent_pair_captures(video_loader)

        perf_stats = Stage2PerfStats(
            enabled=bool(self.config.get("STAGE2_PERF_PROFILE", True)),
            total_candidates=len(stage1_candidates),
        )
        stage2_started = perf_counter()

        def _build_metrics(
            q_scores: Dict[str, float],
            a_scores: Dict[str, float],
            combined: float,
            *,
            is_keyframe_flag: float,
            vo_status_reason: str,
        ) -> Dict[str, Any]:
            ssim = float(a_scores.get("ssim_pair", a_scores.get("ssim", 1.0)))
            flow_mag = float(a_scores.get("optical_flow", 0.0))
            return {
                "translation_delta": flow_mag,
                "rotation_delta": float(max(0.0, 1.0 - ssim) * 180.0),
                "flow_mag": flow_mag,
                "vo_step_proxy": 0.0,
                "vo_step_proxy_norm": 0.0,
                "vo_inlier_ratio": 0.0,
                "vo_rot_deg": float(max(0.0, 1.0 - ssim) * 180.0),
                "vo_dir_cos_prev": 0.5,
                "vo_confidence": 0.0,
                "vo_feature_uniformity": 0.0,
                "vo_track_sufficiency": 0.0,
                "vo_pose_plausibility": 0.0,
                "vo_tracked_count": 0.0,
                "vo_essential_method": "none",
                "laplacian_var": float(q_scores.get("sharpness", 0.0)),
                "match_count": 0.0,
                "overlap_ratio": 0.0,
                "exposure_ratio": float(q_scores.get("exposure", 0.0)),
                "keyframe_flag": float(is_keyframe_flag),
                "combined_score": float(combined),
                "stationary_vo_flag": 0.0,
                "stationary_flow_flag": 0.0,
                "is_stationary": 0.0,
                "stationary_confidence": 0.0,
                "stationary_penalty_applied": 0.0,
                "stage0_motion_risk": 0.0,
                "trajectory_consistency": 0.5,
                "trajectory_consistency_effective": 0.25,
                "combined_stage2": float(combined),
                "combined_stage3": float(combined),
                "stage3_selected_flag": 0.0,
                "vo_status_reason": str(vo_status_reason),
                "vo_pose_valid": 0.0,
                "vo_attempted": 0.0,
            }

        def _append_record(
            frame_idx: int,
            frame_img: Optional[np.ndarray],
            q_scores: Dict[str, float],
            a_scores: Dict[str, float],
            metrics: Dict[str, Any],
            *,
            is_candidate: bool,
            drop_reason: str,
        ) -> None:
            keep_frame = bool(self.config.get("STAGE2_DEFERRED_FRAME_LOG_IMAGES", False))
            records.append(
                Stage2FrameRecord(
                    frame_index=int(frame_idx),
                    frame=frame_img.copy() if (keep_frame and frame_img is not None) else None,
                    quality_scores=dict(q_scores),
                    geometric_scores={},
                    adaptive_scores=dict(a_scores),
                    metrics=dict(metrics),
                    is_candidate=is_candidate,
                    drop_reason=str(drop_reason),
                )
            )

        prev_frame = None
        prev_pair = None
        last_read_idx = -1
        last_read_idx_pair = -1
        try:
            for idx, candidate_info in enumerate(stage1_candidates):
                frame_idx = int(candidate_info.get("frame_idx", 0))
                quality_scores = dict(candidate_info.get("quality_scores", {}) or {})
                if progress_callback:
                    progress_callback(idx, len(stage1_candidates))

                current_frame = None
                current_pair = None
                frame_read_started = perf_counter() if perf_stats.enabled else 0.0
                if is_paired:
                    if cap_a is not None and cap_b is not None:
                        if frame_idx != last_read_idx_pair + 1:
                            cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret_a, frame_a = cap_a.read()
                        ret_b, frame_b = cap_b.read()
                        last_read_idx_pair = frame_idx
                        if not ret_a or not ret_b or frame_a is None or frame_b is None:
                            combined = float(self._compute_combined_score(quality_scores, {}, {}))
                            metrics = _build_metrics(
                                quality_scores,
                                {},
                                combined,
                                is_keyframe_flag=0.0,
                                vo_status_reason="minimal_mode",
                            )
                            _append_record(
                                frame_idx=frame_idx,
                                frame_img=None,
                                q_scores=quality_scores,
                                a_scores={},
                                metrics=metrics,
                                is_candidate=False,
                                drop_reason="read_fail",
                            )
                            continue
                    else:
                        frame_a, frame_b = video_loader.get_frame_pair(frame_idx)
                        if frame_a is None or frame_b is None:
                            combined = float(self._compute_combined_score(quality_scores, {}, {}))
                            metrics = _build_metrics(
                                quality_scores,
                                {},
                                combined,
                                is_keyframe_flag=0.0,
                                vo_status_reason="minimal_mode",
                            )
                            _append_record(
                                frame_idx=frame_idx,
                                frame_img=None,
                                q_scores=quality_scores,
                                a_scores={},
                                metrics=metrics,
                                is_candidate=False,
                                drop_reason="read_fail",
                            )
                            continue
                    if use_fisheye_border_mask:
                        valid_mask_a = self._create_fisheye_valid_mask(frame_a)
                        valid_mask_b = self._create_fisheye_valid_mask(frame_b)
                        frame_a = self.mask_processor.apply_valid_region_mask(frame_a, valid_mask_a, fill_value=0)
                        frame_b = self.mask_processor.apply_valid_region_mask(frame_b, valid_mask_b, fill_value=0)
                    current_pair = (frame_a, frame_b)
                    current_frame = frame_a
                else:
                    if frame_idx != last_read_idx + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, current_frame = cap.read()
                    last_read_idx = frame_idx
                    if not ret or current_frame is None:
                        combined = float(self._compute_combined_score(quality_scores, {}, {}))
                        metrics = _build_metrics(
                            quality_scores,
                            {},
                            combined,
                            is_keyframe_flag=0.0,
                            vo_status_reason="minimal_mode",
                        )
                        _append_record(
                            frame_idx=frame_idx,
                            frame_img=None,
                            q_scores=quality_scores,
                            a_scores={},
                            metrics=metrics,
                            is_candidate=False,
                            drop_reason="read_fail",
                        )
                        continue
                if perf_stats.enabled:
                    perf_stats.frame_read_s += max(0.0, perf_counter() - frame_read_started)
                perf_stats.processed_candidates += 1

                adaptive_scores: Dict[str, float] = {}
                adaptive_started = perf_counter() if perf_stats.enabled else 0.0
                if is_paired and prev_pair is not None and current_pair is not None:
                    flow_a = float(self.adaptive_selector.compute_optical_flow_magnitude(prev_pair[0], current_pair[0]))
                    flow_b = float(self.adaptive_selector.compute_optical_flow_magnitude(prev_pair[1], current_pair[1]))
                    ssim_a = float(self.adaptive_selector.compute_ssim(prev_pair[0], current_pair[0]))
                    ssim_b = float(self.adaptive_selector.compute_ssim(prev_pair[1], current_pair[1]))
                    adaptive_scores["ssim_lens_a"] = ssim_a
                    adaptive_scores["ssim_lens_b"] = ssim_b
                    adaptive_scores["ssim_pair"] = float(min(ssim_a, ssim_b))
                    if self.config.get("PAIR_MOTION_AGGREGATION", "max") == "max":
                        adaptive_scores["optical_flow"] = float(max(flow_a, flow_b))
                    else:
                        adaptive_scores["optical_flow"] = float((flow_a + flow_b) * 0.5)
                    adaptive_scores["optical_flow_lens_a"] = flow_a
                    adaptive_scores["optical_flow_lens_b"] = flow_b
                elif prev_frame is not None and current_frame is not None:
                    flow_mag = float(self.adaptive_selector.compute_optical_flow_magnitude(prev_frame, current_frame))
                    ssim = float(self.adaptive_selector.compute_ssim(prev_frame, current_frame))
                    adaptive_scores["optical_flow"] = flow_mag
                    adaptive_scores["ssim"] = ssim
                    adaptive_scores["ssim_pair"] = ssim
                else:
                    adaptive_scores["optical_flow"] = 0.0
                    adaptive_scores["ssim"] = 1.0
                    adaptive_scores["ssim_pair"] = 1.0
                if perf_stats.enabled:
                    perf_stats.adaptive_eval_s += max(0.0, perf_counter() - adaptive_started)

                combined_score = float(self._compute_combined_score(quality_scores, {}, adaptive_scores))
                candidate = KeyframeInfo(
                    frame_index=frame_idx,
                    timestamp=frame_idx / max(float(metadata.fps), 1e-6),
                    quality_scores=quality_scores,
                    geometric_scores={},
                    adaptive_scores=adaptive_scores,
                    combined_score=combined_score,
                    thumbnail=None,
                    is_rescue_mode=False,
                    is_force_inserted=False,
                    dynamic_mask=None,
                )
                candidates.append(candidate)
                perf_stats.selected_candidates += 1

                metrics = _build_metrics(
                    quality_scores,
                    adaptive_scores,
                    combined_score,
                    is_keyframe_flag=1.0,
                    vo_status_reason="minimal_mode",
                )
                _append_record(
                    frame_idx=frame_idx,
                    frame_img=current_frame,
                    q_scores=quality_scores,
                    a_scores=adaptive_scores,
                    metrics=metrics,
                    is_candidate=True,
                    drop_reason="selected",
                )
                prev_frame = current_frame
                prev_pair = current_pair
        finally:
            if perf_stats.enabled:
                perf_stats.total_s = max(0.0, perf_counter() - stage2_started)
            self._log_stage2_perf_summary(perf_stats)
            if cap is not None:
                cap.release()
                logger.debug("Stage 2(minimal_v1): 独立キャプチャを解放")
            if cap_a is not None:
                cap_a.release()
            if cap_b is not None:
                cap_b.release()

        return candidates, records

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
        return self.quality_evaluator.evaluate_stage1_fast(frame)

    def _compute_quality_score_pair(self, frame_a: np.ndarray,
                                    frame_b: np.ndarray) -> Dict[str, float]:
        """
        ペアレンズフレームの品質スコアを計算（Conservative AND条件）。

        どちらか片方でも閾値未達なら除外する。
        """
        score_a = self.quality_evaluator.evaluate_stage1_fast(frame_a)
        score_b = self.quality_evaluator.evaluate_stage1_fast(frame_b)
        quality_a = compose_legacy_quality_proxy(
            score_a,
            laplacian_threshold=float(self.config['LAPLACIAN_THRESHOLD']),
            motion_blur_threshold=float(self.config['MOTION_BLUR_THRESHOLD']),
            exposure_threshold=float(self.config['EXPOSURE_THRESHOLD']),
        )
        quality_b = compose_legacy_quality_proxy(
            score_b,
            laplacian_threshold=float(self.config['LAPLACIAN_THRESHOLD']),
            motion_blur_threshold=float(self.config['MOTION_BLUR_THRESHOLD']),
            exposure_threshold=float(self.config['EXPOSURE_THRESHOLD']),
        )

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
            'quality': float(min(quality_a, quality_b)),
            'quality_lens_a': float(quality_a),
            'quality_lens_b': float(quality_b),
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

    def _apply_nms(
        self,
        candidates: List[KeyframeInfo],
        time_window: Optional[float] = None,
        stage0_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        cumulative_motion_map: Optional[Dict[int, float]] = None,
        motion_window: Optional[float] = None,
        motion_aware_selection: bool = False,
        **_kwargs,
    ) -> List[KeyframeInfo]:
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
        if time_window is None:
            time_window = float(
                self.config.get("NMS_TIME_WINDOW", self.config.get("nms_time_window", 1.0))
            )
        time_window = float(max(0.01, time_window))

        if cumulative_motion_map is None and stage0_metrics:
            cumulative_motion_map, _ = self._build_cumulative_motion_map(
                stage0_metrics,
                max(int(max((c.frame_index for c in candidates), default=0)) + 1, 1),
            )
        motion_enabled = bool(motion_aware_selection and cumulative_motion_map)
        if motion_window is None:
            ratio = float(
                max(
                    0.0,
                    self.config.get(
                        "COLMAP_NMS_MOTION_WINDOW_RATIO",
                        self.config.get("colmap_nms_motion_window_ratio", 0.5),
                    ),
                )
            )
            if motion_enabled and stage0_metrics:
                _, motion_info = self._build_cumulative_motion_map(
                    stage0_metrics,
                    max(int(max((c.frame_index for c in candidates), default=0)) + 1, 1),
                )
                motion_window = float(max(0.0, ratio * float(motion_info.get("motion_median_step", 1.0))))
            else:
                motion_window = 0.0
        motion_window = float(max(0.0, motion_window or 0.0))

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
                if motion_enabled and motion_window > 0.0:
                    c_motion = float(cumulative_motion_map.get(int(candidate.frame_index), 0.0))
                    s_motion = float(cumulative_motion_map.get(int(selected_kf.frame_index), 0.0))
                    if abs(c_motion - s_motion) < motion_window:
                        is_within_window = True
                        break

            if not is_within_window:
                selected.append(candidate)

        # フレーム番号でソート
        selected.sort(key=lambda x: x.frame_index)

        return selected

    def _enforce_max_interval(
        self,
        keyframes: List[KeyframeInfo],
        fps: float,
        source_candidates: Optional[List[KeyframeInfo]] = None,
    ) -> List[KeyframeInfo]:
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

        fps_safe = max(float(fps), 1e-6)
        max_interval = self.config['MAX_KEYFRAME_INTERVAL'] / fps_safe

        selected_sorted = sorted(keyframes, key=lambda x: x.frame_index)
        source_sorted = sorted(source_candidates or keyframes, key=lambda x: x.frame_index)
        used_frames = {int(kf.frame_index) for kf in selected_sorted[:1]}
        enforced_keyframes = [selected_sorted[0]]

        for i in range(1, len(selected_sorted)):
            current_kf = selected_sorted[i]

            while True:
                last_kf = enforced_keyframes[-1]
                time_diff = current_kf.timestamp - last_kf.timestamp
                if time_diff <= max_interval:
                    break

                logger.debug(
                    f"最大間隔超過: frame {last_kf.frame_index} → {current_kf.frame_index} "
                    f"(gap={time_diff:.2f}s > max={max_interval:.2f}s)"
                )

                between = [
                    c for c in source_sorted
                    if last_kf.frame_index < c.frame_index < current_kf.frame_index
                    and int(c.frame_index) not in used_frames
                ]
                if not between:
                    logger.warning(
                        "最大間隔超過を補完できません: "
                        f"frame {last_kf.frame_index} → {current_kf.frame_index}"
                    )
                    break

                # 目標時刻（last + max_interval）に近い候補を優先し、同距離なら高スコアを採用
                target_time = last_kf.timestamp + max_interval
                insert_kf = min(
                    between,
                    key=lambda c: (abs(c.timestamp - target_time), -float(c.combined_score)),
                )
                enforced_keyframes.append(insert_kf)
                used_frames.add(int(insert_kf.frame_index))
                logger.debug(
                    f"最大間隔補完フレーム追加: frame={insert_kf.frame_index}, "
                    f"score={insert_kf.combined_score:.3f}"
                )

            if int(current_kf.frame_index) not in used_frames:
                enforced_keyframes.append(current_kf)
                used_frames.add(int(current_kf.frame_index))

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
