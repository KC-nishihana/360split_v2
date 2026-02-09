"""
360Split - Configuration Constants
360度動画ベース3D再構成GUIソフトウェア設定
"""

# === キーフレーム選択パラメータ ===
# Quality-based Selection
LAPLACIAN_THRESHOLD = 100.0          # ラプラシアン鮮明度閾値
BRIGHTNESS_MIN = 40                   # 最小輝度（暗すぎるフレーム排除）
BRIGHTNESS_MAX = 220                  # 最大輝度（白飛びフレーム排除）
MOTION_BLUR_THRESHOLD = 0.3          # モーションブラー検出閾値
SOFTMAX_BETA = 5.0                    # Softmax-scaling深度の温度パラメータβ

# Geometric Selection
GRIC_RATIO_THRESHOLD = 0.8           # GRICスコア閾値（ホモグラフィ vs 基礎行列）
MIN_FEATURE_MATCHES = 30             # 最小特徴点マッチ数
FEATURE_DISTRIBUTION_THRESHOLD = 0.4  # 特徴点分散バランス閾値
RANSAC_REPROJ_THRESHOLD = 3.0        # RANSACの再投影誤差閾値

# Adaptive Selection
SSIM_CHANGE_THRESHOLD = 0.85         # SSIM変化検知閾値（低いほど変化大）
MIN_KEYFRAME_INTERVAL = 5            # 最小キーフレーム間隔（フレーム数）
MAX_KEYFRAME_INTERVAL = 60           # 最大キーフレーム間隔（フレーム数）
MOMENTUM_BOOST_FACTOR = 2.0          # 加速度変化時のサンプリング密度倍率

# === 360度画像処理パラメータ ===
EQUIRECT_WIDTH = 4096                 # Equirectangular出力幅
EQUIRECT_HEIGHT = 2048                # Equirectangular出力高さ
CUBEMAP_FACE_SIZE = 1024              # Cubemap面サイズ
PERSPECTIVE_FOV = 90.0                # Perspective投影FOV（度）

# === マスク処理パラメータ ===
MASK_DILATION_KERNEL = 15             # マスク膨張カーネルサイズ
MASK_CONFIDENCE_THRESHOLD = 0.5       # セグメンテーション信頼度閾値

# === スコアリング重み ===
WEIGHT_SHARPNESS = 0.30               # 鮮明度スコア重み
WEIGHT_EXPOSURE = 0.15                # 露光スコア重み
WEIGHT_GEOMETRIC = 0.30               # 幾何学的スコア重み
WEIGHT_CONTENT = 0.25                 # コンテンツ変化スコア重み

# === 高速化パラメータ ===
EVAL_SCALE = 0.5                      # 品質評価時のダウンスケール率（0.25-1.0）
SSIM_SCALE = 0.5                      # SSIM計算時のダウンスケール率
FRAME_CACHE_SIZE = 100                # フレームキャッシュ容量（LRU）
PREFETCH_AHEAD = 10                   # フレームプリフェッチ先読み数
BATCH_SIZE = 32                       # バッチ処理サイズ（並列処理用）
FEATURE_CACHE_SIZE = 50               # 特徴点キャッシュ容量
STAGE1_QUALITY_THRESHOLD = 0.3        # Stage 1 品質フィルタ閾値（低品質除外）
USE_SPARSE_FLOW = True                # スパースオプティカルフロー使用（True推奨）

# === GUI設定 ===
THUMBNAIL_SIZE = (192, 108)           # サムネイルサイズ
TIMELINE_HEIGHT = 80                  # タイムラインウィジェット高さ
MAX_PREVIEW_SIZE = (1920, 1080)       # プレビュー最大サイズ

# === 出力設定 ===
OUTPUT_IMAGE_FORMAT = "png"           # 出力画像フォーマット
OUTPUT_JPEG_QUALITY = 95              # JPEG品質
PLY_EXPORT_ENABLED = False            # PLYエクスポート（3DGS連携時有効化）
