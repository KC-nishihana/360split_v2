"""
360Split - Configuration
360度動画ベース3D再構成GUIソフトウェア設定

dataclassベースの構造化された設定とレガシー定数を併存。
KeyframeSelectorは KeyframeConfig を使用し、
GUI/CLIではdict形式でオーバーライドできる。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


# =============================================================================
# データクラスベース設定（v2）
# =============================================================================

@dataclass
class NormalizationConfig:
    """
    スコア正規化用の係数

    各評価スコアを0-1に正規化する際に使用する分母・上限値。
    """
    SHARPNESS_NORM_FACTOR: float = 1000.0    # ラプラシアン分散の正規化分母
    OPTICAL_FLOW_NORM_FACTOR: float = 50.0   # 光学フロー大きさの正規化分母
    FEATURE_MATCH_NORM_FACTOR: float = 200.0  # マッチ数の正規化分母


@dataclass
class WeightsConfig:
    """
    最終スコア算出時の重み

    alpha + beta + gamma + delta = 1.0 を推奨
    """
    alpha: float = 0.30    # 鮮明度スコア重み (sharpness)
    beta: float = 0.30     # 幾何学的スコア重み (geometric/GRIC)
    gamma: float = 0.25    # コンテンツ変化スコア重み (content/SSIM)
    delta: float = 0.15    # 露光スコア重み (exposure)


@dataclass
class GRICConfig:
    """
    GRIC (Geometric Robust Information Criterion) 計算用パラメータ

    GRIC = sum(rho(r_i^2, sigma)) + lambda1 * d * n + lambda2 * k

    rho: ロバスト誤差関数（Huber相当）
    r_i: 各点の残差
    sigma: 残差の標準偏差推定値
    d: データ次元（2D点 = 2）
    n: 点数
    k: モデルパラメータ数（H=8, F=7）
    lambda1, lambda2: 正則化係数
    """
    lambda1: float = 2.0     # データ適合ペナルティ係数
    lambda2: float = 4.0     # モデル複雑さペナルティ係数
    sigma: float = 1.0       # 残差標準偏差（ピクセル）
    ransac_threshold: float = 3.0    # RANSAC再投影誤差閾値
    min_inlier_ratio: float = 0.3    # 最小インライア率（未満なら推定失敗）
    degeneracy_threshold: float = 0.85  # H行列インライア率がこの値以上→縮退（回転のみ）
    min_matches: int = 15    # GRIC計算に必要な最小マッチ数


@dataclass
class SelectionConfig:
    """
    キーフレーム選択パラメータ
    """
    min_keyframe_interval: int = 5      # 最小キーフレーム間隔（フレーム数）
    max_keyframe_interval: int = 60     # 最大キーフレーム間隔（フレーム数）
    laplacian_threshold: float = 100.0  # Stage 1 鮮明度閾値
    motion_blur_threshold: float = 0.3  # モーションブラー許容閾値
    exposure_threshold: float = 0.35    # 露光スコア最小閾値
    quality_filter_enabled: bool = True
    quality_threshold: float = 0.50
    quality_roi_mode: str = "circle"
    quality_roi_ratio: float = 0.40
    quality_abs_laplacian_min: float = 35.0
    quality_use_orb: bool = True
    quality_weight_sharpness: float = 0.40
    quality_weight_tenengrad: float = 0.30
    quality_weight_exposure: float = 0.15
    quality_weight_keypoints: float = 0.15
    quality_norm_p_low: float = 10.0
    quality_norm_p_high: float = 90.0
    quality_debug: bool = False
    quality_tenengrad_scale: float = 1.0
    ssim_change_threshold: float = 0.85  # SSIM変化検知閾値
    softmax_beta: float = 5.0           # Softmax温度パラメータ
    nms_time_window: float = 1.0        # NMS時間ウィンドウ（秒）
    stationary_enable: bool = True
    stationary_min_duration_sec: float = 0.7
    stationary_use_quantile_threshold: bool = True
    stationary_quantile: float = 0.10
    stationary_translation_threshold: Optional[float] = None
    stationary_rotation_threshold: Optional[float] = None
    stationary_flow_threshold: Optional[float] = None
    stationary_min_match_count_for_vo: int = 15
    stationary_fallback_when_vo_unreliable: str = "not_stationary"
    stationary_soft_penalty: bool = True
    stationary_penalty: float = 0.7
    stationary_allow_boundary_frames: bool = True
    stationary_boundary_grace_frames: int = 2
    stationary_hysteresis_exit_scale: float = 1.25


@dataclass
class Equirect360Config:
    """
    360度特有の処理設定
    """
    mask_polar_ratio: float = 0.10  # 天頂/天底マスク比率（上下10%をマスク）
    enable_polar_mask: bool = True  # 特徴点抽出時のポーラーマスク有効化
    enable_fisheye_border_mask: bool = True
    fisheye_mask_radius_ratio: float = 0.94
    fisheye_mask_center_offset_x: int = 0
    fisheye_mask_center_offset_y: int = 0
    enable_dynamic_mask_removal: bool = False
    dynamic_mask_use_yolo_sam: bool = True
    dynamic_mask_use_motion_diff: bool = True
    dynamic_mask_motion_frames: int = 3
    dynamic_mask_motion_threshold: int = 30
    dynamic_mask_dilation_size: int = 5
    dynamic_mask_target_classes: Tuple[str, ...] = ("人物", "人", "自転車", "バイク", "車両", "動物")
    dynamic_mask_inpaint_enabled: bool = False
    dynamic_mask_inpaint_module: str = ""
    yolo_model_path: str = "yolo26n-seg.pt"
    sam_model_path: str = "sam3_t.pt"
    confidence_threshold: float = 0.25
    detection_device: str = "auto"


@dataclass
class KeyframeConfig:
    """
    キーフレーム選択の統合設定

    全サブ設定を束ね、dict変換メソッドとファクトリメソッドを提供。
    """
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    gric: GRICConfig = field(default_factory=GRICConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    equirect360: Equirect360Config = field(default_factory=Equirect360Config)

    # パイプライン設定
    sample_interval: int = 1           # フレームサンプリング間隔
    stage1_batch_size: int = 32        # Stage 1 バッチサイズ
    stage1_grab_threshold: int = 30    # Stage1でgrab方式を使う最大サンプル間隔
    stage1_eval_scale: float = 0.5     # Stage1品質評価の縮小スケール
    opencv_thread_count: int = 0       # 0=auto
    stage1_process_workers: int = 0    # 0=auto
    stage1_prefetch_size: int = 32
    stage1_metrics_batch_size: int = 64
    stage1_gpu_batch_enabled: bool = True
    darwin_capture_backend: str = "auto"  # auto|avfoundation|ffmpeg
    mps_min_pixels: int = 256 * 256
    thumbnail_size: Tuple[int, int] = (192, 108)
    enable_rerun_logging: bool = False  # GUI実行時のRerunログ有効化
    enable_stage0_scan: bool = True
    stage0_stride: int = 5
    enable_stage3_refinement: bool = True
    stage3_weight_base: float = 0.70
    stage3_weight_trajectory: float = 0.25
    stage3_weight_stage0_risk: float = 0.05
    stage3_disable_traj_when_vo_unreliable: bool = True
    stage3_vo_valid_ratio_threshold: float = 0.50
    flow_downscale: float = 1.0
    resume_enabled: bool = False
    keep_temp_on_success: bool = True
    vo_enabled: bool = True
    vo_center_roi_ratio: float = 0.6
    vo_downscale_long_edge: int = 1000
    vo_max_features: int = 600
    vo_t_sign: float = 1.0
    vo_frame_subsample: int = 1
    vo_adaptive_roi_enable: bool = True
    vo_adaptive_roi_min: float = 0.45
    vo_adaptive_roi_max: float = 0.70
    vo_fast_fail_inlier_ratio: float = 0.12
    vo_step_proxy_clip_px: float = 80.0
    vo_essential_method: str = "auto"
    vo_subpixel_refine: bool = True
    vo_adaptive_subsample: bool = False
    vo_subsample_min: int = 1
    vo_confidence_low_threshold: float = 0.35
    vo_confidence_mid_threshold: float = 0.55
    calib_xml: str = ""
    front_calib_xml: str = ""
    rear_calib_xml: str = ""
    calib_model: str = "auto"
    pose_backend: str = "vo"
    colmap_path: str = "colmap"
    colmap_workspace: str = ""
    colmap_db_path: str = ""
    colmap_keyframe_policy: str = ""
    colmap_keyframe_target_mode: str = "auto"
    colmap_keyframe_target_min: int = 120
    colmap_keyframe_target_max: int = 240
    colmap_nms_window_sec: float = 0.35
    colmap_enable_stage0: bool = True
    colmap_motion_aware_selection: bool = True
    colmap_nms_motion_window_ratio: float = 0.5
    colmap_stage1_adaptive_threshold: bool = True
    colmap_stage1_min_candidates_per_bin: int = 3
    colmap_stage1_max_candidates: int = 360
    colmap_rig_policy: str = "lr_opk"
    colmap_rig_seed_opk_deg: Tuple[float, float, float] = (0.0, 0.0, 180.0)
    colmap_workspace_scope: str = "run_scoped"
    colmap_reuse_db: bool = False
    colmap_analysis_mask_profile: str = "colmap_safe"
    pose_export_format: str = "internal"
    pose_select_translation_threshold: float = 1.2
    pose_select_rotation_threshold_deg: float = 5.0
    pose_select_min_observations: int = 30
    pose_select_enable_translation: bool = True
    pose_select_enable_rotation: bool = True
    pose_select_enable_observations: bool = False
    motion_blur_method: str = "legacy"
    enable_stage2_pipeline_parallel: bool = False

    def to_selector_dict(self) -> dict:
        """
        KeyframeSelector互換のdict形式に変換

        Returns:
        --------
        dict
            KeyframeSelectorの内部config辞書と互換のキー名で返す
        """
        return {
            'WEIGHT_SHARPNESS': self.weights.alpha,
            'WEIGHT_EXPOSURE': self.weights.delta,
            'WEIGHT_GEOMETRIC': self.weights.beta,
            'WEIGHT_CONTENT': self.weights.gamma,
            'LAPLACIAN_THRESHOLD': self.selection.laplacian_threshold,
            'MIN_KEYFRAME_INTERVAL': self.selection.min_keyframe_interval,
            'MAX_KEYFRAME_INTERVAL': self.selection.max_keyframe_interval,
            'SOFTMAX_BETA': self.selection.softmax_beta,
            'NMS_TIME_WINDOW': self.selection.nms_time_window,
            # 互換性のため旧キーと新キーの両方を出力
            'GRIC_RATIO_THRESHOLD': self.gric.degeneracy_threshold,
            'GRIC_DEGENERACY_THRESHOLD': self.gric.degeneracy_threshold,
            'GRIC_LAMBDA1': self.gric.lambda1,
            'GRIC_LAMBDA2': self.gric.lambda2,
            'GRIC_SIGMA': self.gric.sigma,
            'RANSAC_THRESHOLD': self.gric.ransac_threshold,
            'SSIM_CHANGE_THRESHOLD': self.selection.ssim_change_threshold,
            'MOTION_BLUR_THRESHOLD': self.selection.motion_blur_threshold,
            'EXPOSURE_THRESHOLD': self.selection.exposure_threshold,
            'QUALITY_FILTER_ENABLED': self.selection.quality_filter_enabled,
            'QUALITY_THRESHOLD': self.selection.quality_threshold,
            'QUALITY_ROI_MODE': self.selection.quality_roi_mode,
            'QUALITY_ROI_RATIO': self.selection.quality_roi_ratio,
            'QUALITY_ABS_LAPLACIAN_MIN': self.selection.quality_abs_laplacian_min,
            'QUALITY_USE_ORB': self.selection.quality_use_orb,
            'QUALITY_WEIGHT_SHARPNESS': self.selection.quality_weight_sharpness,
            'QUALITY_WEIGHT_TENENGRAD': self.selection.quality_weight_tenengrad,
            'QUALITY_WEIGHT_EXPOSURE': self.selection.quality_weight_exposure,
            'QUALITY_WEIGHT_KEYPOINTS': self.selection.quality_weight_keypoints,
            'QUALITY_NORM_P_LOW': self.selection.quality_norm_p_low,
            'QUALITY_NORM_P_HIGH': self.selection.quality_norm_p_high,
            'QUALITY_DEBUG': self.selection.quality_debug,
            'QUALITY_TENENGRAD_SCALE': self.selection.quality_tenengrad_scale,
            'MIN_FEATURE_MATCHES': self.gric.min_matches,
            'STATIONARY_ENABLE': self.selection.stationary_enable,
            'STATIONARY_MIN_DURATION_SEC': self.selection.stationary_min_duration_sec,
            'STATIONARY_USE_QUANTILE_THRESHOLD': self.selection.stationary_use_quantile_threshold,
            'STATIONARY_QUANTILE': self.selection.stationary_quantile,
            'STATIONARY_TRANSLATION_THRESHOLD': self.selection.stationary_translation_threshold,
            'STATIONARY_ROTATION_THRESHOLD': self.selection.stationary_rotation_threshold,
            'STATIONARY_FLOW_THRESHOLD': self.selection.stationary_flow_threshold,
            'STATIONARY_MIN_MATCH_COUNT_FOR_VO': int(
                self.selection.stationary_min_match_count_for_vo or self.gric.min_matches
            ),
            'STATIONARY_FALLBACK_WHEN_VO_UNRELIABLE': self.selection.stationary_fallback_when_vo_unreliable,
            'STATIONARY_SOFT_PENALTY': self.selection.stationary_soft_penalty,
            'STATIONARY_PENALTY': self.selection.stationary_penalty,
            'STATIONARY_ALLOW_BOUNDARY_FRAMES': self.selection.stationary_allow_boundary_frames,
            'STATIONARY_BOUNDARY_GRACE_FRAMES': self.selection.stationary_boundary_grace_frames,
            'STATIONARY_HYSTERESIS_EXIT_SCALE': self.selection.stationary_hysteresis_exit_scale,
            'ENABLE_POLAR_MASK': self.equirect360.enable_polar_mask,
            'MASK_POLAR_RATIO': self.equirect360.mask_polar_ratio,
            'ENABLE_FISHEYE_BORDER_MASK': self.equirect360.enable_fisheye_border_mask,
            'FISHEYE_MASK_RADIUS_RATIO': self.equirect360.fisheye_mask_radius_ratio,
            'FISHEYE_MASK_CENTER_OFFSET_X': self.equirect360.fisheye_mask_center_offset_x,
            'FISHEYE_MASK_CENTER_OFFSET_Y': self.equirect360.fisheye_mask_center_offset_y,
            'ENABLE_DYNAMIC_MASK_REMOVAL': self.equirect360.enable_dynamic_mask_removal,
            'DYNAMIC_MASK_USE_YOLO_SAM': self.equirect360.dynamic_mask_use_yolo_sam,
            'DYNAMIC_MASK_USE_MOTION_DIFF': self.equirect360.dynamic_mask_use_motion_diff,
            'DYNAMIC_MASK_MOTION_FRAMES': self.equirect360.dynamic_mask_motion_frames,
            'DYNAMIC_MASK_MOTION_THRESHOLD': self.equirect360.dynamic_mask_motion_threshold,
            'DYNAMIC_MASK_DILATION_SIZE': self.equirect360.dynamic_mask_dilation_size,
            'DYNAMIC_MASK_TARGET_CLASSES': list(self.equirect360.dynamic_mask_target_classes),
            'DYNAMIC_MASK_INPAINT_ENABLED': self.equirect360.dynamic_mask_inpaint_enabled,
            'DYNAMIC_MASK_INPAINT_MODULE': self.equirect360.dynamic_mask_inpaint_module,
            'YOLO_MODEL_PATH': self.equirect360.yolo_model_path,
            'SAM_MODEL_PATH': self.equirect360.sam_model_path,
            'CONFIDENCE_THRESHOLD': self.equirect360.confidence_threshold,
            'DETECTION_DEVICE': self.equirect360.detection_device,
            'THUMBNAIL_SIZE': self.thumbnail_size,
            'SAMPLE_INTERVAL': self.sample_interval,
            'STAGE1_BATCH_SIZE': self.stage1_batch_size,
            'STAGE1_GRAB_THRESHOLD': self.stage1_grab_threshold,
            'STAGE1_EVAL_SCALE': self.stage1_eval_scale,
            'OPENCV_THREAD_COUNT': self.opencv_thread_count,
            'STAGE1_PROCESS_WORKERS': self.stage1_process_workers,
            'STAGE1_PREFETCH_SIZE': self.stage1_prefetch_size,
            'STAGE1_METRICS_BATCH_SIZE': self.stage1_metrics_batch_size,
            'STAGE1_GPU_BATCH_ENABLED': self.stage1_gpu_batch_enabled,
            'DARWIN_CAPTURE_BACKEND': self.darwin_capture_backend,
            'MPS_MIN_PIXELS': self.mps_min_pixels,
            'enable_rerun_logging': self.enable_rerun_logging,
            'ENABLE_STAGE0_SCAN': self.enable_stage0_scan,
            'STAGE0_STRIDE': self.stage0_stride,
            'ENABLE_STAGE3_REFINEMENT': self.enable_stage3_refinement,
            'STAGE3_WEIGHT_BASE': self.stage3_weight_base,
            'STAGE3_WEIGHT_TRAJECTORY': self.stage3_weight_trajectory,
            'STAGE3_WEIGHT_STAGE0_RISK': self.stage3_weight_stage0_risk,
            'STAGE3_DISABLE_TRAJ_WHEN_VO_UNRELIABLE': self.stage3_disable_traj_when_vo_unreliable,
            'STAGE3_VO_VALID_RATIO_THRESHOLD': self.stage3_vo_valid_ratio_threshold,
            'FLOW_DOWNSCALE': self.flow_downscale,
            'RESUME_ENABLED': self.resume_enabled,
            'KEEP_TEMP_ON_SUCCESS': self.keep_temp_on_success,
            'VO_ENABLED': self.vo_enabled,
            'VO_CENTER_ROI_RATIO': self.vo_center_roi_ratio,
            'VO_DOWNSCALE_LONG_EDGE': self.vo_downscale_long_edge,
            'VO_MAX_FEATURES': self.vo_max_features,
            'VO_T_SIGN': self.vo_t_sign,
            'VO_FRAME_SUBSAMPLE': self.vo_frame_subsample,
            'VO_ADAPTIVE_ROI_ENABLE': self.vo_adaptive_roi_enable,
            'VO_ADAPTIVE_ROI_MIN': self.vo_adaptive_roi_min,
            'VO_ADAPTIVE_ROI_MAX': self.vo_adaptive_roi_max,
            'VO_FAST_FAIL_INLIER_RATIO': self.vo_fast_fail_inlier_ratio,
            'VO_STEP_PROXY_CLIP_PX': self.vo_step_proxy_clip_px,
            'VO_ESSENTIAL_METHOD': self.vo_essential_method,
            'VO_SUBPIXEL_REFINE': self.vo_subpixel_refine,
            'VO_ADAPTIVE_SUBSAMPLE': self.vo_adaptive_subsample,
            'VO_SUBSAMPLE_MIN': self.vo_subsample_min,
            'VO_CONFIDENCE_LOW_THRESHOLD': self.vo_confidence_low_threshold,
            'VO_CONFIDENCE_MID_THRESHOLD': self.vo_confidence_mid_threshold,
            'CALIB_XML': self.calib_xml,
            'FRONT_CALIB_XML': self.front_calib_xml,
            'REAR_CALIB_XML': self.rear_calib_xml,
            'CALIB_MODEL': self.calib_model,
            'POSE_BACKEND': self.pose_backend,
            'COLMAP_PATH': self.colmap_path,
            'COLMAP_WORKSPACE': self.colmap_workspace,
            'COLMAP_DB_PATH': self.colmap_db_path,
            'COLMAP_KEYFRAME_POLICY': self.colmap_keyframe_policy,
            'COLMAP_KEYFRAME_TARGET_MODE': self.colmap_keyframe_target_mode,
            'COLMAP_KEYFRAME_TARGET_MIN': self.colmap_keyframe_target_min,
            'COLMAP_KEYFRAME_TARGET_MAX': self.colmap_keyframe_target_max,
            'COLMAP_NMS_WINDOW_SEC': self.colmap_nms_window_sec,
            'COLMAP_ENABLE_STAGE0': self.colmap_enable_stage0,
            'COLMAP_MOTION_AWARE_SELECTION': self.colmap_motion_aware_selection,
            'COLMAP_NMS_MOTION_WINDOW_RATIO': self.colmap_nms_motion_window_ratio,
            'COLMAP_STAGE1_ADAPTIVE_THRESHOLD': self.colmap_stage1_adaptive_threshold,
            'COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN': self.colmap_stage1_min_candidates_per_bin,
            'COLMAP_STAGE1_MAX_CANDIDATES': self.colmap_stage1_max_candidates,
            'COLMAP_RIG_POLICY': self.colmap_rig_policy,
            'COLMAP_RIG_SEED_OPK_DEG': list(self.colmap_rig_seed_opk_deg),
            'COLMAP_WORKSPACE_SCOPE': self.colmap_workspace_scope,
            'COLMAP_REUSE_DB': self.colmap_reuse_db,
            'COLMAP_ANALYSIS_MASK_PROFILE': self.colmap_analysis_mask_profile,
            'POSE_EXPORT_FORMAT': self.pose_export_format,
            'POSE_SELECT_TRANSLATION_THRESHOLD': self.pose_select_translation_threshold,
            'POSE_SELECT_ROTATION_THRESHOLD_DEG': self.pose_select_rotation_threshold_deg,
            'POSE_SELECT_MIN_OBSERVATIONS': self.pose_select_min_observations,
            'POSE_SELECT_ENABLE_TRANSLATION': self.pose_select_enable_translation,
            'POSE_SELECT_ENABLE_ROTATION': self.pose_select_enable_rotation,
            'POSE_SELECT_ENABLE_OBSERVATIONS': self.pose_select_enable_observations,
            'MOTION_BLUR_METHOD': self.motion_blur_method,
            'ENABLE_STAGE2_PIPELINE_PARALLEL': self.enable_stage2_pipeline_parallel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'KeyframeConfig':
        """
        GUI設定dictからKeyframeConfigを生成

        Parameters:
        -----------
        d : dict
            GUI設定の小文字キー辞書

        Returns:
        --------
        KeyframeConfig
        """
        normalized = normalize_config_dict(d or {})
        config = cls()
        # Weights
        config.weights.alpha = float(normalized.get('weight_sharpness', config.weights.alpha))
        config.weights.delta = float(normalized.get('weight_exposure', config.weights.delta))
        config.weights.beta = float(normalized.get('weight_geometric', config.weights.beta))
        config.weights.gamma = float(normalized.get('weight_content', config.weights.gamma))
        # Selection
        config.selection.ssim_change_threshold = float(normalized.get('ssim_threshold', config.selection.ssim_change_threshold))
        config.sample_interval = int(max(1, normalized.get('sample_interval', config.sample_interval)))
        config.stage1_batch_size = int(max(1, normalized.get('stage1_batch_size', config.stage1_batch_size)))
        config.stage1_grab_threshold = int(max(1, normalized.get('stage1_grab_threshold', config.stage1_grab_threshold)))
        config.stage1_eval_scale = float(np.clip(normalized.get('stage1_eval_scale', config.stage1_eval_scale), 0.1, 1.0))
        config.opencv_thread_count = int(max(0, normalized.get('opencv_thread_count', config.opencv_thread_count)))
        config.stage1_process_workers = int(max(0, normalized.get('stage1_process_workers', config.stage1_process_workers)))
        config.stage1_prefetch_size = int(max(1, normalized.get('stage1_prefetch_size', config.stage1_prefetch_size)))
        config.stage1_metrics_batch_size = int(max(1, normalized.get('stage1_metrics_batch_size', config.stage1_metrics_batch_size)))
        config.stage1_gpu_batch_enabled = bool(normalized.get('stage1_gpu_batch_enabled', config.stage1_gpu_batch_enabled))
        config.darwin_capture_backend = str(
            normalized.get('darwin_capture_backend', config.darwin_capture_backend) or "auto"
        ).strip().lower()
        if config.darwin_capture_backend not in {"auto", "avfoundation", "ffmpeg"}:
            config.darwin_capture_backend = "auto"
        config.mps_min_pixels = int(max(1, normalized.get('mps_min_pixels', config.mps_min_pixels)))
        config.selection.min_keyframe_interval = int(normalized.get('min_keyframe_interval', config.selection.min_keyframe_interval))
        config.selection.max_keyframe_interval = int(normalized.get('max_keyframe_interval', config.selection.max_keyframe_interval))
        config.selection.softmax_beta = float(normalized.get('softmax_beta', config.selection.softmax_beta))
        config.selection.nms_time_window = float(max(0.01, normalized.get('nms_time_window', config.selection.nms_time_window)))
        config.selection.laplacian_threshold = float(normalized.get('laplacian_threshold', config.selection.laplacian_threshold))
        config.selection.motion_blur_threshold = float(normalized.get('motion_blur_threshold', config.selection.motion_blur_threshold))
        config.selection.exposure_threshold = float(normalized.get('exposure_threshold', config.selection.exposure_threshold))
        config.selection.quality_filter_enabled = bool(normalized.get('quality_filter_enabled', config.selection.quality_filter_enabled))
        config.selection.quality_threshold = float(np.clip(normalized.get('quality_threshold', config.selection.quality_threshold), 0.0, 1.0))
        config.selection.quality_roi_mode = str(normalized.get('quality_roi_mode', config.selection.quality_roi_mode) or "circle").strip().lower()
        if config.selection.quality_roi_mode not in {"circle", "rect"}:
            config.selection.quality_roi_mode = "circle"
        config.selection.quality_roi_ratio = float(np.clip(normalized.get('quality_roi_ratio', config.selection.quality_roi_ratio), 0.05, 1.0))
        config.selection.quality_abs_laplacian_min = float(max(0.0, normalized.get('quality_abs_laplacian_min', config.selection.quality_abs_laplacian_min)))
        config.selection.quality_use_orb = bool(normalized.get('quality_use_orb', config.selection.quality_use_orb))
        config.selection.quality_weight_sharpness = float(max(0.0, normalized.get('quality_weight_sharpness', config.selection.quality_weight_sharpness)))
        config.selection.quality_weight_tenengrad = float(max(0.0, normalized.get('quality_weight_tenengrad', config.selection.quality_weight_tenengrad)))
        config.selection.quality_weight_exposure = float(max(0.0, normalized.get('quality_weight_exposure', config.selection.quality_weight_exposure)))
        config.selection.quality_weight_keypoints = float(max(0.0, normalized.get('quality_weight_keypoints', config.selection.quality_weight_keypoints)))
        config.selection.quality_norm_p_low = float(np.clip(normalized.get('quality_norm_p_low', config.selection.quality_norm_p_low), 0.0, 100.0))
        config.selection.quality_norm_p_high = float(np.clip(normalized.get('quality_norm_p_high', config.selection.quality_norm_p_high), config.selection.quality_norm_p_low, 100.0))
        config.selection.quality_debug = bool(normalized.get('quality_debug', config.selection.quality_debug))
        config.selection.quality_tenengrad_scale = float(
            np.clip(
                normalized.get('quality_tenengrad_scale', config.selection.quality_tenengrad_scale),
                0.1,
                1.0,
            )
        )
        config.selection.stationary_enable = bool(normalized.get('stationary_enable', config.selection.stationary_enable))
        config.selection.stationary_min_duration_sec = float(normalized.get(
            'stationary_min_duration_sec', config.selection.stationary_min_duration_sec
        ))
        config.selection.stationary_use_quantile_threshold = bool(normalized.get(
            'stationary_use_quantile_threshold', config.selection.stationary_use_quantile_threshold
        ))
        config.selection.stationary_quantile = float(normalized.get(
            'stationary_quantile', config.selection.stationary_quantile
        ))
        config.selection.stationary_translation_threshold = normalized.get(
            'stationary_translation_threshold', config.selection.stationary_translation_threshold
        )
        config.selection.stationary_rotation_threshold = normalized.get(
            'stationary_rotation_threshold', config.selection.stationary_rotation_threshold
        )
        config.selection.stationary_flow_threshold = normalized.get(
            'stationary_flow_threshold', config.selection.stationary_flow_threshold
        )
        config.selection.stationary_min_match_count_for_vo = int(normalized.get(
            'stationary_min_match_count_for_vo', config.selection.stationary_min_match_count_for_vo
        ))
        config.selection.stationary_fallback_when_vo_unreliable = str(normalized.get(
            'stationary_fallback_when_vo_unreliable',
            config.selection.stationary_fallback_when_vo_unreliable,
        ))
        config.selection.stationary_soft_penalty = bool(normalized.get(
            'stationary_soft_penalty', config.selection.stationary_soft_penalty
        ))
        config.selection.stationary_penalty = float(normalized.get(
            'stationary_penalty', config.selection.stationary_penalty
        ))
        config.selection.stationary_allow_boundary_frames = bool(normalized.get(
            'stationary_allow_boundary_frames', config.selection.stationary_allow_boundary_frames
        ))
        config.selection.stationary_boundary_grace_frames = int(normalized.get(
            'stationary_boundary_grace_frames', config.selection.stationary_boundary_grace_frames
        ))
        config.selection.stationary_hysteresis_exit_scale = float(normalized.get(
            'stationary_hysteresis_exit_scale', config.selection.stationary_hysteresis_exit_scale
        ))
        # GRIC
        config.gric.ransac_threshold = float(normalized.get('ransac_threshold', config.gric.ransac_threshold))
        config.gric.lambda1 = float(normalized.get('gric_lambda1', config.gric.lambda1))
        config.gric.lambda2 = float(normalized.get('gric_lambda2', config.gric.lambda2))
        config.gric.sigma = float(normalized.get('gric_sigma', config.gric.sigma))
        config.gric.degeneracy_threshold = float(normalized.get('gric_degeneracy_threshold', config.gric.degeneracy_threshold))
        # 360
        config.equirect360.enable_polar_mask = bool(normalized.get('enable_polar_mask', config.equirect360.enable_polar_mask))
        config.equirect360.mask_polar_ratio = float(normalized.get('mask_polar_ratio', config.equirect360.mask_polar_ratio))
        config.equirect360.enable_fisheye_border_mask = bool(normalized.get(
            'enable_fisheye_border_mask',
            config.equirect360.enable_fisheye_border_mask,
        ))
        config.equirect360.fisheye_mask_radius_ratio = float(normalized.get(
            'fisheye_mask_radius_ratio',
            config.equirect360.fisheye_mask_radius_ratio,
        ))
        config.equirect360.fisheye_mask_center_offset_x = int(normalized.get(
            'fisheye_mask_center_offset_x',
            config.equirect360.fisheye_mask_center_offset_x,
        ))
        config.equirect360.fisheye_mask_center_offset_y = int(normalized.get(
            'fisheye_mask_center_offset_y',
            config.equirect360.fisheye_mask_center_offset_y,
        ))
        config.equirect360.enable_dynamic_mask_removal = bool(normalized.get(
            'enable_dynamic_mask_removal',
            config.equirect360.enable_dynamic_mask_removal,
        ))
        config.equirect360.dynamic_mask_use_yolo_sam = bool(normalized.get(
            'dynamic_mask_use_yolo_sam',
            config.equirect360.dynamic_mask_use_yolo_sam,
        ))
        config.equirect360.dynamic_mask_use_motion_diff = bool(normalized.get(
            'dynamic_mask_use_motion_diff',
            config.equirect360.dynamic_mask_use_motion_diff,
        ))
        config.equirect360.dynamic_mask_motion_frames = int(normalized.get(
            'dynamic_mask_motion_frames',
            config.equirect360.dynamic_mask_motion_frames,
        ))
        config.equirect360.dynamic_mask_motion_threshold = int(normalized.get(
            'dynamic_mask_motion_threshold',
            config.equirect360.dynamic_mask_motion_threshold,
        ))
        config.equirect360.dynamic_mask_dilation_size = int(normalized.get(
            'dynamic_mask_dilation_size',
            config.equirect360.dynamic_mask_dilation_size,
        ))
        target_classes = normalized.get(
            'dynamic_mask_target_classes',
            normalized.get('target_classes', config.equirect360.dynamic_mask_target_classes),
        )
        if isinstance(target_classes, list):
            config.equirect360.dynamic_mask_target_classes = tuple(target_classes)
        elif isinstance(target_classes, tuple):
            config.equirect360.dynamic_mask_target_classes = target_classes
        config.equirect360.dynamic_mask_inpaint_enabled = bool(normalized.get(
            'dynamic_mask_inpaint_enabled',
            config.equirect360.dynamic_mask_inpaint_enabled,
        ))
        config.equirect360.dynamic_mask_inpaint_module = normalized.get(
            'dynamic_mask_inpaint_module',
            config.equirect360.dynamic_mask_inpaint_module,
        )
        config.equirect360.yolo_model_path = str(normalized.get('yolo_model_path', config.equirect360.yolo_model_path))
        config.equirect360.sam_model_path = str(normalized.get('sam_model_path', config.equirect360.sam_model_path))
        config.equirect360.confidence_threshold = float(normalized.get(
            'confidence_threshold', config.equirect360.confidence_threshold
        ))
        config.equirect360.detection_device = str(normalized.get('detection_device', config.equirect360.detection_device))
        # Rerun
        config.enable_rerun_logging = bool(normalized.get('enable_rerun_logging', config.enable_rerun_logging))
        config.enable_stage0_scan = bool(normalized.get('enable_stage0_scan', config.enable_stage0_scan))
        config.stage0_stride = int(max(1, normalized.get('stage0_stride', config.stage0_stride)))
        config.enable_stage3_refinement = bool(normalized.get('enable_stage3_refinement', config.enable_stage3_refinement))
        config.stage3_weight_base = float(normalized.get('stage3_weight_base', config.stage3_weight_base))
        config.stage3_weight_trajectory = float(normalized.get('stage3_weight_trajectory', config.stage3_weight_trajectory))
        config.stage3_weight_stage0_risk = float(normalized.get('stage3_weight_stage0_risk', config.stage3_weight_stage0_risk))
        config.stage3_disable_traj_when_vo_unreliable = bool(
            normalized.get(
                'stage3_disable_traj_when_vo_unreliable',
                config.stage3_disable_traj_when_vo_unreliable,
            )
        )
        config.stage3_vo_valid_ratio_threshold = float(
            np.clip(
                normalized.get(
                    'stage3_vo_valid_ratio_threshold',
                    config.stage3_vo_valid_ratio_threshold,
                ),
                0.0,
                1.0,
            )
        )
        config.flow_downscale = float(np.clip(normalized.get('flow_downscale', config.flow_downscale), 0.1, 1.0))
        config.resume_enabled = bool(normalized.get('resume_enabled', config.resume_enabled))
        config.keep_temp_on_success = bool(normalized.get('keep_temp_on_success', config.keep_temp_on_success))
        config.vo_enabled = bool(normalized.get('vo_enabled', config.vo_enabled))
        config.vo_center_roi_ratio = float(np.clip(normalized.get('vo_center_roi_ratio', config.vo_center_roi_ratio), 0.2, 1.0))
        config.vo_downscale_long_edge = int(max(256, normalized.get('vo_downscale_long_edge', config.vo_downscale_long_edge)))
        config.vo_max_features = int(max(64, normalized.get('vo_max_features', config.vo_max_features)))
        config.vo_t_sign = float(normalized.get('vo_t_sign', config.vo_t_sign))
        config.vo_frame_subsample = int(max(1, normalized.get('vo_frame_subsample', config.vo_frame_subsample)))
        config.vo_adaptive_roi_enable = bool(normalized.get('vo_adaptive_roi_enable', config.vo_adaptive_roi_enable))
        config.vo_adaptive_roi_min = float(np.clip(normalized.get('vo_adaptive_roi_min', config.vo_adaptive_roi_min), 0.2, 1.0))
        config.vo_adaptive_roi_max = float(np.clip(normalized.get('vo_adaptive_roi_max', config.vo_adaptive_roi_max), config.vo_adaptive_roi_min, 1.0))
        config.vo_fast_fail_inlier_ratio = float(np.clip(normalized.get('vo_fast_fail_inlier_ratio', config.vo_fast_fail_inlier_ratio), 0.0, 1.0))
        config.vo_step_proxy_clip_px = float(max(0.0, normalized.get('vo_step_proxy_clip_px', config.vo_step_proxy_clip_px)))
        config.vo_essential_method = str(normalized.get('vo_essential_method', config.vo_essential_method) or "auto").strip().lower()
        if config.vo_essential_method not in {"auto", "ransac", "magsac"}:
            config.vo_essential_method = "auto"
        config.vo_subpixel_refine = bool(normalized.get('vo_subpixel_refine', config.vo_subpixel_refine))
        config.vo_adaptive_subsample = bool(normalized.get('vo_adaptive_subsample', config.vo_adaptive_subsample))
        config.vo_subsample_min = int(max(1, normalized.get('vo_subsample_min', config.vo_subsample_min)))
        config.vo_confidence_low_threshold = float(
            np.clip(normalized.get('vo_confidence_low_threshold', config.vo_confidence_low_threshold), 0.0, 1.0)
        )
        config.vo_confidence_mid_threshold = float(
            np.clip(
                normalized.get('vo_confidence_mid_threshold', config.vo_confidence_mid_threshold),
                config.vo_confidence_low_threshold,
                1.0,
            )
        )
        config.calib_xml = str(normalized.get('calib_xml', config.calib_xml) or "")
        config.front_calib_xml = str(normalized.get('front_calib_xml', config.front_calib_xml) or "")
        config.rear_calib_xml = str(normalized.get('rear_calib_xml', config.rear_calib_xml) or "")
        config.calib_model = str(normalized.get('calib_model', config.calib_model) or "auto")
        config.pose_backend = str(normalized.get('pose_backend', config.pose_backend) or "vo").strip().lower()
        if config.pose_backend not in {"vo", "colmap"}:
            config.pose_backend = "vo"
        config.colmap_path = str(normalized.get('colmap_path', config.colmap_path) or "colmap").strip() or "colmap"
        config.colmap_workspace = str(normalized.get('colmap_workspace', config.colmap_workspace) or "").strip()
        config.colmap_db_path = str(normalized.get('colmap_db_path', config.colmap_db_path) or "").strip()
        config.colmap_keyframe_policy = str(
            normalized.get('colmap_keyframe_policy', config.colmap_keyframe_policy) or ""
        ).strip().lower()
        if config.colmap_keyframe_policy not in {"", "legacy", "stage2_relaxed", "stage1_only"}:
            config.colmap_keyframe_policy = ""
        config.colmap_keyframe_target_mode = str(
            normalized.get('colmap_keyframe_target_mode', config.colmap_keyframe_target_mode) or "auto"
        ).strip().lower()
        if config.colmap_keyframe_target_mode not in {"fixed", "auto"}:
            config.colmap_keyframe_target_mode = "auto"
        config.colmap_keyframe_target_min = int(
            max(1, normalized.get('colmap_keyframe_target_min', config.colmap_keyframe_target_min))
        )
        config.colmap_keyframe_target_max = int(
            max(
                config.colmap_keyframe_target_min,
                normalized.get('colmap_keyframe_target_max', config.colmap_keyframe_target_max),
            )
        )
        config.colmap_nms_window_sec = float(
            max(0.01, normalized.get('colmap_nms_window_sec', config.colmap_nms_window_sec))
        )
        config.colmap_enable_stage0 = bool(
            normalized.get('colmap_enable_stage0', config.colmap_enable_stage0)
        )
        config.colmap_motion_aware_selection = bool(
            normalized.get('colmap_motion_aware_selection', config.colmap_motion_aware_selection)
        )
        config.colmap_nms_motion_window_ratio = float(
            max(0.0, normalized.get('colmap_nms_motion_window_ratio', config.colmap_nms_motion_window_ratio))
        )
        config.colmap_stage1_adaptive_threshold = bool(
            normalized.get('colmap_stage1_adaptive_threshold', config.colmap_stage1_adaptive_threshold)
        )
        config.colmap_stage1_min_candidates_per_bin = int(
            max(0, normalized.get('colmap_stage1_min_candidates_per_bin', config.colmap_stage1_min_candidates_per_bin))
        )
        config.colmap_stage1_max_candidates = int(
            max(1, normalized.get('colmap_stage1_max_candidates', config.colmap_stage1_max_candidates))
        )
        config.colmap_rig_policy = str(
            normalized.get('colmap_rig_policy', config.colmap_rig_policy) or "lr_opk"
        ).strip().lower()
        if config.colmap_rig_policy not in {"off", "lr_opk"}:
            config.colmap_rig_policy = "off"
        seed = normalized.get('colmap_rig_seed_opk_deg', config.colmap_rig_seed_opk_deg)
        if isinstance(seed, str):
            parts = [p.strip() for p in seed.split(",") if p.strip()]
            if len(parts) == 3:
                try:
                    seed = [float(parts[0]), float(parts[1]), float(parts[2])]
                except (TypeError, ValueError):
                    seed = config.colmap_rig_seed_opk_deg
        if not isinstance(seed, (list, tuple)) or len(seed) != 3:
            seed = config.colmap_rig_seed_opk_deg
        try:
            config.colmap_rig_seed_opk_deg = (float(seed[0]), float(seed[1]), float(seed[2]))
        except (TypeError, ValueError, IndexError):
            config.colmap_rig_seed_opk_deg = (0.0, 0.0, 180.0)
        config.colmap_workspace_scope = str(
            normalized.get('colmap_workspace_scope', config.colmap_workspace_scope) or "run_scoped"
        ).strip().lower()
        if config.colmap_workspace_scope not in {"shared", "run_scoped"}:
            config.colmap_workspace_scope = "run_scoped"
        config.colmap_reuse_db = bool(normalized.get('colmap_reuse_db', config.colmap_reuse_db))
        config.colmap_analysis_mask_profile = str(
            normalized.get('colmap_analysis_mask_profile', config.colmap_analysis_mask_profile) or "colmap_safe"
        ).strip().lower()
        if config.colmap_analysis_mask_profile not in {"legacy", "colmap_safe"}:
            config.colmap_analysis_mask_profile = "colmap_safe"
        config.pose_export_format = str(normalized.get('pose_export_format', config.pose_export_format) or "internal").strip().lower()
        if config.pose_export_format not in {"internal", "metashape"}:
            config.pose_export_format = "internal"
        config.pose_select_translation_threshold = float(
            max(0.0, normalized.get('pose_select_translation_threshold', config.pose_select_translation_threshold))
        )
        config.pose_select_rotation_threshold_deg = float(
            max(0.0, normalized.get('pose_select_rotation_threshold_deg', config.pose_select_rotation_threshold_deg))
        )
        config.pose_select_min_observations = int(
            max(0, normalized.get('pose_select_min_observations', config.pose_select_min_observations))
        )
        config.pose_select_enable_translation = bool(
            normalized.get('pose_select_enable_translation', config.pose_select_enable_translation)
        )
        config.pose_select_enable_rotation = bool(
            normalized.get('pose_select_enable_rotation', config.pose_select_enable_rotation)
        )
        config.pose_select_enable_observations = bool(
            normalized.get('pose_select_enable_observations', config.pose_select_enable_observations)
        )
        config.motion_blur_method = str(normalized.get("motion_blur_method", config.motion_blur_method) or "legacy").strip().lower()
        if config.motion_blur_method not in {"legacy", "angle_hist", "fft_hybrid"}:
            config.motion_blur_method = "legacy"
        config.enable_stage2_pipeline_parallel = bool(
            normalized.get("enable_stage2_pipeline_parallel", config.enable_stage2_pipeline_parallel)
        )
        config.validate()
        return config

    def validate(self) -> None:
        wsum = float(self.weights.alpha + self.weights.beta + self.weights.gamma + self.weights.delta)
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(f"weight sum must be 1.0, got {wsum:.6f}")


SELECTOR_ALIAS_MAP: Dict[str, str] = {
    'laplacian_threshold': 'LAPLACIAN_THRESHOLD',
    'motion_blur_threshold': 'MOTION_BLUR_THRESHOLD',
    'exposure_threshold': 'EXPOSURE_THRESHOLD',
    'quality_filter_enabled': 'QUALITY_FILTER_ENABLED',
    'quality_threshold': 'QUALITY_THRESHOLD',
    'quality_roi_mode': 'QUALITY_ROI_MODE',
    'quality_roi_ratio': 'QUALITY_ROI_RATIO',
    'quality_abs_laplacian_min': 'QUALITY_ABS_LAPLACIAN_MIN',
    'quality_use_orb': 'QUALITY_USE_ORB',
    'quality_weight_sharpness': 'QUALITY_WEIGHT_SHARPNESS',
    'quality_weight_tenengrad': 'QUALITY_WEIGHT_TENENGRAD',
    'quality_weight_exposure': 'QUALITY_WEIGHT_EXPOSURE',
    'quality_weight_keypoints': 'QUALITY_WEIGHT_KEYPOINTS',
    'quality_norm_p_low': 'QUALITY_NORM_P_LOW',
    'quality_norm_p_high': 'QUALITY_NORM_P_HIGH',
    'quality_debug': 'QUALITY_DEBUG',
    'quality_tenengrad_scale': 'QUALITY_TENENGRAD_SCALE',
    'min_keyframe_interval': 'MIN_KEYFRAME_INTERVAL',
    'max_keyframe_interval': 'MAX_KEYFRAME_INTERVAL',
    'softmax_beta': 'SOFTMAX_BETA',
    'nms_time_window': 'NMS_TIME_WINDOW',
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
    'gric_ratio_threshold': 'GRIC_RATIO_THRESHOLD',
    'gric_degeneracy_threshold': 'GRIC_DEGENERACY_THRESHOLD',
    'min_feature_matches': 'MIN_FEATURE_MATCHES',
    'enable_dynamic_mask_removal': 'ENABLE_DYNAMIC_MASK_REMOVAL',
    'enable_fisheye_border_mask': 'ENABLE_FISHEYE_BORDER_MASK',
    'fisheye_mask_radius_ratio': 'FISHEYE_MASK_RADIUS_RATIO',
    'fisheye_mask_center_offset_x': 'FISHEYE_MASK_CENTER_OFFSET_X',
    'fisheye_mask_center_offset_y': 'FISHEYE_MASK_CENTER_OFFSET_Y',
    'dynamic_mask_use_yolo_sam': 'DYNAMIC_MASK_USE_YOLO_SAM',
    'dynamic_mask_use_motion_diff': 'DYNAMIC_MASK_USE_MOTION_DIFF',
    'dynamic_mask_motion_frames': 'DYNAMIC_MASK_MOTION_FRAMES',
    'dynamic_mask_motion_threshold': 'DYNAMIC_MASK_MOTION_THRESHOLD',
    'dynamic_mask_dilation_size': 'DYNAMIC_MASK_DILATION_SIZE',
    'dynamic_mask_target_classes': 'DYNAMIC_MASK_TARGET_CLASSES',
    'dynamic_mask_inpaint_enabled': 'DYNAMIC_MASK_INPAINT_ENABLED',
    'dynamic_mask_inpaint_module': 'DYNAMIC_MASK_INPAINT_MODULE',
    'yolo_model_path': 'YOLO_MODEL_PATH',
    'sam_model_path': 'SAM_MODEL_PATH',
    'confidence_threshold': 'CONFIDENCE_THRESHOLD',
    'detection_device': 'DETECTION_DEVICE',
    'stage1_grab_threshold': 'STAGE1_GRAB_THRESHOLD',
    'stage1_eval_scale': 'STAGE1_EVAL_SCALE',
    'opencv_thread_count': 'OPENCV_THREAD_COUNT',
    'stage1_process_workers': 'STAGE1_PROCESS_WORKERS',
    'stage1_prefetch_size': 'STAGE1_PREFETCH_SIZE',
    'stage1_metrics_batch_size': 'STAGE1_METRICS_BATCH_SIZE',
    'stage1_gpu_batch_enabled': 'STAGE1_GPU_BATCH_ENABLED',
    'darwin_capture_backend': 'DARWIN_CAPTURE_BACKEND',
    'mps_min_pixels': 'MPS_MIN_PIXELS',
    'enable_profile': 'ENABLE_PROFILE',
    'stage2_perf_profile': 'STAGE2_PERF_PROFILE',
    'stage2_mask_cache_ttl_frames': 'STAGE2_MASK_CACHE_TTL_FRAMES',
    'enable_rescue_mode': 'ENABLE_RESCUE_MODE',
    'rescue_feature_threshold': 'RESCUE_FEATURE_THRESHOLD',
    'rescue_laplacian_factor': 'RESCUE_LAPLACIAN_FACTOR',
    'force_keyframe_on_exposure_change': 'FORCE_KEYFRAME_ON_EXPOSURE_CHANGE',
    'exposure_change_threshold': 'EXPOSURE_CHANGE_THRESHOLD',
    'adaptive_thresholding': 'ADAPTIVE_THRESHOLDING',
    'stationary_enable': 'STATIONARY_ENABLE',
    'stationary_min_duration_sec': 'STATIONARY_MIN_DURATION_SEC',
    'stationary_use_quantile_threshold': 'STATIONARY_USE_QUANTILE_THRESHOLD',
    'stationary_quantile': 'STATIONARY_QUANTILE',
    'stationary_translation_threshold': 'STATIONARY_TRANSLATION_THRESHOLD',
    'stationary_rotation_threshold': 'STATIONARY_ROTATION_THRESHOLD',
    'stationary_flow_threshold': 'STATIONARY_FLOW_THRESHOLD',
    'stationary_min_match_count_for_vo': 'STATIONARY_MIN_MATCH_COUNT_FOR_VO',
    'stationary_fallback_when_vo_unreliable': 'STATIONARY_FALLBACK_WHEN_VO_UNRELIABLE',
    'stationary_soft_penalty': 'STATIONARY_SOFT_PENALTY',
    'stationary_penalty': 'STATIONARY_PENALTY',
    'stationary_allow_boundary_frames': 'STATIONARY_ALLOW_BOUNDARY_FRAMES',
    'stationary_boundary_grace_frames': 'STATIONARY_BOUNDARY_GRACE_FRAMES',
    'stationary_hysteresis_exit_scale': 'STATIONARY_HYSTERESIS_EXIT_SCALE',
    'enable_stage0_scan': 'ENABLE_STAGE0_SCAN',
    'stage0_stride': 'STAGE0_STRIDE',
    'enable_stage3_refinement': 'ENABLE_STAGE3_REFINEMENT',
    'stage3_weight_base': 'STAGE3_WEIGHT_BASE',
    'stage3_weight_trajectory': 'STAGE3_WEIGHT_TRAJECTORY',
    'stage3_weight_stage0_risk': 'STAGE3_WEIGHT_STAGE0_RISK',
    'stage3_disable_traj_when_vo_unreliable': 'STAGE3_DISABLE_TRAJ_WHEN_VO_UNRELIABLE',
    'stage3_vo_valid_ratio_threshold': 'STAGE3_VO_VALID_RATIO_THRESHOLD',
    'flow_downscale': 'FLOW_DOWNSCALE',
    'resume_enabled': 'RESUME_ENABLED',
    'keep_temp_on_success': 'KEEP_TEMP_ON_SUCCESS',
    'projection_mode': 'PROJECTION_MODE',
    'vo_enabled': 'VO_ENABLED',
    'vo_center_roi_ratio': 'VO_CENTER_ROI_RATIO',
    'vo_max_features': 'VO_MAX_FEATURES',
    'vo_quality_level': 'VO_QUALITY_LEVEL',
    'vo_min_distance': 'VO_MIN_DISTANCE',
    'vo_min_track_points': 'VO_MIN_TRACK_POINTS',
    'vo_ransac_threshold': 'VO_RANSAC_THRESHOLD',
    'vo_downscale_long_edge': 'VO_DOWNSCALE_LONG_EDGE',
    'vo_match_norm_factor': 'VO_MATCH_NORM_FACTOR',
    'vo_t_sign': 'VO_T_SIGN',
    'vo_frame_subsample': 'VO_FRAME_SUBSAMPLE',
    'vo_adaptive_roi_enable': 'VO_ADAPTIVE_ROI_ENABLE',
    'vo_adaptive_roi_min': 'VO_ADAPTIVE_ROI_MIN',
    'vo_adaptive_roi_max': 'VO_ADAPTIVE_ROI_MAX',
    'vo_fast_fail_inlier_ratio': 'VO_FAST_FAIL_INLIER_RATIO',
    'vo_step_proxy_clip_px': 'VO_STEP_PROXY_CLIP_PX',
    'vo_essential_method': 'VO_ESSENTIAL_METHOD',
    'vo_subpixel_refine': 'VO_SUBPIXEL_REFINE',
    'vo_adaptive_subsample': 'VO_ADAPTIVE_SUBSAMPLE',
    'vo_subsample_min': 'VO_SUBSAMPLE_MIN',
    'vo_confidence_low_threshold': 'VO_CONFIDENCE_LOW_THRESHOLD',
    'vo_confidence_mid_threshold': 'VO_CONFIDENCE_MID_THRESHOLD',
    'calib_xml': 'CALIB_XML',
    'front_calib_xml': 'FRONT_CALIB_XML',
    'rear_calib_xml': 'REAR_CALIB_XML',
    'calib_model': 'CALIB_MODEL',
    'pose_backend': 'POSE_BACKEND',
    'colmap_path': 'COLMAP_PATH',
    'colmap_workspace': 'COLMAP_WORKSPACE',
    'colmap_db_path': 'COLMAP_DB_PATH',
    'colmap_keyframe_policy': 'COLMAP_KEYFRAME_POLICY',
    'colmap_keyframe_target_mode': 'COLMAP_KEYFRAME_TARGET_MODE',
    'colmap_keyframe_target_min': 'COLMAP_KEYFRAME_TARGET_MIN',
    'colmap_keyframe_target_max': 'COLMAP_KEYFRAME_TARGET_MAX',
    'colmap_nms_window_sec': 'COLMAP_NMS_WINDOW_SEC',
    'colmap_enable_stage0': 'COLMAP_ENABLE_STAGE0',
    'colmap_motion_aware_selection': 'COLMAP_MOTION_AWARE_SELECTION',
    'colmap_nms_motion_window_ratio': 'COLMAP_NMS_MOTION_WINDOW_RATIO',
    'colmap_stage1_adaptive_threshold': 'COLMAP_STAGE1_ADAPTIVE_THRESHOLD',
    'colmap_stage1_min_candidates_per_bin': 'COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN',
    'colmap_stage1_max_candidates': 'COLMAP_STAGE1_MAX_CANDIDATES',
    'colmap_rig_policy': 'COLMAP_RIG_POLICY',
    'colmap_rig_seed_opk_deg': 'COLMAP_RIG_SEED_OPK_DEG',
    'colmap_workspace_scope': 'COLMAP_WORKSPACE_SCOPE',
    'colmap_reuse_db': 'COLMAP_REUSE_DB',
    'colmap_analysis_mask_profile': 'COLMAP_ANALYSIS_MASK_PROFILE',
    'pose_export_format': 'POSE_EXPORT_FORMAT',
    'pose_select_translation_threshold': 'POSE_SELECT_TRANSLATION_THRESHOLD',
    'pose_select_rotation_threshold_deg': 'POSE_SELECT_ROTATION_THRESHOLD_DEG',
    'pose_select_min_observations': 'POSE_SELECT_MIN_OBSERVATIONS',
    'pose_select_enable_translation': 'POSE_SELECT_ENABLE_TRANSLATION',
    'pose_select_enable_rotation': 'POSE_SELECT_ENABLE_ROTATION',
    'pose_select_enable_observations': 'POSE_SELECT_ENABLE_OBSERVATIONS',
    'analysis_mode': 'ANALYSIS_MODE',
    'motion_blur_method': 'MOTION_BLUR_METHOD',
    'enable_stage2_pipeline_parallel': 'ENABLE_STAGE2_PIPELINE_PARALLEL',
    'sample_interval': 'SAMPLE_INTERVAL',
    'stage1_batch_size': 'STAGE1_BATCH_SIZE',
    'ransac_threshold': 'RANSAC_THRESHOLD',
    'gric_lambda1': 'GRIC_LAMBDA1',
    'gric_lambda2': 'GRIC_LAMBDA2',
    'gric_sigma': 'GRIC_SIGMA',
    'enable_polar_mask': 'ENABLE_POLAR_MASK',
    'mask_polar_ratio': 'MASK_POLAR_RATIO',
    'enable_rerun_logging': 'enable_rerun_logging',
}

_UPPER_TO_LOWER_MAP: Dict[str, str] = {v: k for k, v in SELECTOR_ALIAS_MAP.items()}


def normalize_config_dict(d: dict) -> dict:
    normalized = dict(d)
    for upper_key, lower_key in _UPPER_TO_LOWER_MAP.items():
        if upper_key in normalized and lower_key not in normalized:
            normalized[lower_key] = normalized[upper_key]
    return normalized


# =============================================================================
# レガシー定数（v1互換、既存コードからの参照用）
# =============================================================================

# === キーフレーム選択パラメータ ===
LAPLACIAN_THRESHOLD = 100.0
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220
MOTION_BLUR_THRESHOLD = 0.3
SOFTMAX_BETA = 5.0

# Geometric Selection
GRIC_RATIO_THRESHOLD = 0.8
MIN_FEATURE_MATCHES = 30
FEATURE_DISTRIBUTION_THRESHOLD = 0.4
RANSAC_REPROJ_THRESHOLD = 3.0

# Adaptive Selection
SSIM_CHANGE_THRESHOLD = 0.85
MIN_KEYFRAME_INTERVAL = 5
MAX_KEYFRAME_INTERVAL = 60
MOMENTUM_BOOST_FACTOR = 2.0

# === 360度画像処理パラメータ ===
EQUIRECT_WIDTH = 4096
EQUIRECT_HEIGHT = 2048
CUBEMAP_FACE_SIZE = 1024
PERSPECTIVE_FOV = 90.0

# === マスク処理パラメータ ===
MASK_DILATION_KERNEL = 15
MASK_CONFIDENCE_THRESHOLD = 0.5

# === スコアリング重み ===
WEIGHT_SHARPNESS = 0.30
WEIGHT_EXPOSURE = 0.15
WEIGHT_GEOMETRIC = 0.30
WEIGHT_CONTENT = 0.25

# === 高速化パラメータ ===
EVAL_SCALE = 0.5
SSIM_SCALE = 0.5
FRAME_CACHE_SIZE = 100
PREFETCH_AHEAD = 10
BATCH_SIZE = 32
FEATURE_CACHE_SIZE = 50
STAGE1_QUALITY_THRESHOLD = 0.3
USE_SPARSE_FLOW = True

# === GUI設定 ===
THUMBNAIL_SIZE = (192, 108)
TIMELINE_HEIGHT = 80
MAX_PREVIEW_SIZE = (1920, 1080)

# === 出力設定 ===
OUTPUT_IMAGE_FORMAT = "png"
OUTPUT_JPEG_QUALITY = 95
PLY_EXPORT_ENABLED = False
