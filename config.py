"""
360Split - Configuration
360度動画ベース3D再構成GUIソフトウェア設定

dataclassベースの構造化された設定とレガシー定数を併存。
KeyframeSelectorは KeyframeConfig を使用し、
GUI/CLIではdict形式でオーバーライドできる。
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple


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
    ssim_change_threshold: float = 0.85  # SSIM変化検知閾値
    softmax_beta: float = 5.0           # Softmax温度パラメータ
    nms_time_window: float = 1.0        # NMS時間ウィンドウ（秒）


@dataclass
class Equirect360Config:
    """
    360度特有の処理設定
    """
    mask_polar_ratio: float = 0.10  # 天頂/天底マスク比率（上下10%をマスク）
    enable_polar_mask: bool = True  # 特徴点抽出時のポーラーマスク有効化
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
    thumbnail_size: Tuple[int, int] = (192, 108)
    enable_rerun_logging: bool = False  # GUI実行時のRerunログ有効化

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
            'MIN_FEATURE_MATCHES': self.gric.min_matches,
            'ENABLE_POLAR_MASK': self.equirect360.enable_polar_mask,
            'MASK_POLAR_RATIO': self.equirect360.mask_polar_ratio,
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
            'enable_rerun_logging': self.enable_rerun_logging,
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
        config = cls()
        # Weights
        config.weights.alpha = d.get('weight_sharpness', config.weights.alpha)
        config.weights.delta = d.get('weight_exposure', config.weights.delta)
        config.weights.beta = d.get('weight_geometric', config.weights.beta)
        config.weights.gamma = d.get('weight_content', config.weights.gamma)
        # Selection
        config.selection.ssim_change_threshold = d.get('ssim_threshold', config.selection.ssim_change_threshold)
        config.selection.min_keyframe_interval = d.get('min_keyframe_interval', config.selection.min_keyframe_interval)
        config.selection.max_keyframe_interval = d.get('max_keyframe_interval', config.selection.max_keyframe_interval)
        config.selection.softmax_beta = d.get('softmax_beta', config.selection.softmax_beta)
        config.selection.laplacian_threshold = d.get('laplacian_threshold', config.selection.laplacian_threshold)
        config.selection.motion_blur_threshold = d.get('motion_blur_threshold', config.selection.motion_blur_threshold)
        config.selection.exposure_threshold = d.get('exposure_threshold', config.selection.exposure_threshold)
        # GRIC
        config.gric.ransac_threshold = d.get('ransac_threshold', config.gric.ransac_threshold)
        config.gric.lambda1 = d.get('gric_lambda1', config.gric.lambda1)
        config.gric.lambda2 = d.get('gric_lambda2', config.gric.lambda2)
        config.gric.sigma = d.get('gric_sigma', config.gric.sigma)
        config.gric.degeneracy_threshold = d.get('gric_degeneracy_threshold', config.gric.degeneracy_threshold)
        # 360
        config.equirect360.enable_polar_mask = d.get('enable_polar_mask', config.equirect360.enable_polar_mask)
        config.equirect360.mask_polar_ratio = d.get('mask_polar_ratio', config.equirect360.mask_polar_ratio)
        config.equirect360.enable_dynamic_mask_removal = d.get(
            'enable_dynamic_mask_removal',
            config.equirect360.enable_dynamic_mask_removal,
        )
        config.equirect360.dynamic_mask_use_yolo_sam = d.get(
            'dynamic_mask_use_yolo_sam',
            config.equirect360.dynamic_mask_use_yolo_sam,
        )
        config.equirect360.dynamic_mask_use_motion_diff = d.get(
            'dynamic_mask_use_motion_diff',
            config.equirect360.dynamic_mask_use_motion_diff,
        )
        config.equirect360.dynamic_mask_motion_frames = d.get(
            'dynamic_mask_motion_frames',
            config.equirect360.dynamic_mask_motion_frames,
        )
        config.equirect360.dynamic_mask_motion_threshold = d.get(
            'dynamic_mask_motion_threshold',
            config.equirect360.dynamic_mask_motion_threshold,
        )
        config.equirect360.dynamic_mask_dilation_size = d.get(
            'dynamic_mask_dilation_size',
            config.equirect360.dynamic_mask_dilation_size,
        )
        target_classes = d.get(
            'dynamic_mask_target_classes',
            d.get('target_classes', config.equirect360.dynamic_mask_target_classes),
        )
        if isinstance(target_classes, list):
            config.equirect360.dynamic_mask_target_classes = tuple(target_classes)
        elif isinstance(target_classes, tuple):
            config.equirect360.dynamic_mask_target_classes = target_classes
        config.equirect360.dynamic_mask_inpaint_enabled = d.get(
            'dynamic_mask_inpaint_enabled',
            config.equirect360.dynamic_mask_inpaint_enabled,
        )
        config.equirect360.dynamic_mask_inpaint_module = d.get(
            'dynamic_mask_inpaint_module',
            config.equirect360.dynamic_mask_inpaint_module,
        )
        config.equirect360.yolo_model_path = d.get('yolo_model_path', config.equirect360.yolo_model_path)
        config.equirect360.sam_model_path = d.get('sam_model_path', config.equirect360.sam_model_path)
        config.equirect360.confidence_threshold = d.get(
            'confidence_threshold', config.equirect360.confidence_threshold
        )
        config.equirect360.detection_device = d.get('detection_device', config.equirect360.detection_device)
        # Rerun
        config.enable_rerun_logging = bool(d.get('enable_rerun_logging', config.enable_rerun_logging))
        return config


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
