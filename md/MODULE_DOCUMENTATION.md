# 360Split コアモジュール ドキュメント

## 概要

360Splitは、360度ビデオからキーフレームを自動抽出するPythonライブラリです。品質評価、幾何学的評価、適応的フレーム選択を統合して、フォトグラメトリーおよび3DGS（Gaussian Splatting）に最適なキーフレームを自動選択します。

---

## モジュール構成

### 1. `utils/logger.py`
ロギング設定ユーティリティ

**クラス:**
- `ColoredFormatter`: コンソール出力用のカラーフォーマッター

**関数:**
- `setup_logger(name, level, log_file, use_color)`: ロガーセットアップ
- `get_logger(name)`: ロガー取得

**使用例:**
```python
from utils.logger import setup_logger
logger = setup_logger('myapp', log_file='app.log')
logger.info("処理開始")
```

---

### 2. `core/video_loader.py`
ビデオファイル読み込みとフレーム抽出

**クラス:**
- `VideoMetadata`: ビデオメタデータ（fps, frame_count, 解像度, duration）
- `VideoLoader`: ビデオ操作クラス

**VideoLoader主要メソッド:**
- `load(path)` → VideoMetadata: ビデオファイルを開く
- `get_frame(index)` → np.ndarray: 指定フレームを取得
- `get_frame_at_time(seconds)` → np.ndarray: 指定時刻のフレームを取得
- `extract_frames(start, end, step)` → List[np.ndarray]: フレーム範囲を抽出
- `get_metadata()` → VideoMetadata: メタデータを取得

**プロパティ:**
- `fps`, `frame_count`, `width`, `height`, `duration`

**使用例:**
```python
from core.video_loader import VideoLoader

with VideoLoader() as loader:
    metadata = loader.load('video.mp4')
    print(f"FPS: {loader.fps}, フレーム数: {loader.frame_count}")
    
    frame = loader.get_frame(100)
    frames = loader.extract_frames(0, 500, step=5)
```

---

### 3. `core/quality_evaluator.py`
画像品質の多面的評価

**クラス:**
- `QualityEvaluator`: 品質スコア計算

**静的メソッド:**
- `compute_sharpness(frame)` → float: ラプラシアン分散スコア（鮮明度）
- `compute_motion_blur(frame)` → float: モーションブラースコア（0-1）
- `compute_exposure_score(frame)` → float: 露光スコア（0-1、1が最適）
- `compute_softmax_depth_score(frame, beta)` → float: Softmax-scaling深度スコア
- `evaluate(frame, beta)` → dict: 全スコアを計算

**返却スコア:**
- `sharpness`: ラプラシアン値（大きいほど鮮明）
- `motion_blur`: ブラー度（0-1、低いほど良い）
- `exposure`: 露光最適性（0-1、1が最適）
- `softmax_depth`: 深度情報品質（0-1）

**使用例:**
```python
from core.quality_evaluator import QualityEvaluator

frame = cv2.imread('frame.jpg')
scores = QualityEvaluator.evaluate(frame)
print(f"鮮明度: {scores['sharpness']:.1f}")
print(f"露光: {scores['exposure']:.3f}")
```

---

### 4. `core/geometric_evaluator.py`
特徴点ベースの幾何学的評価

**クラス:**
- `GeometricEvaluator`: 特徴点マッチング・幾何評価

**主要メソッド:**
- `compute_gric_score(frame1, frame2)` → float: ホモグラフィ vs 基礎行列比較
  - スコアが低い ⇒ 視差あり（良い）
  - スコアが高い ⇒ 視差なし（悪い）
  
- `compute_feature_distribution(frame)` → float: 特徴点分散スコア（0-1）
  - 画像を4×4グリッドに分割
  - 均等分布 ⇒ スコア高
  
- `compute_feature_match_count(frame1, frame2)` → int: マッチ数
  
- `compute_ray_dispersion(frame, is_equirectangular)` → float: 光線分散スコア
  - エクイレクタングラ対応
  - 多様な方向からの観察 ⇒ スコア高
  
- `evaluate(frame1, frame2)` → dict: 総合幾何学的評価

**使用例:**
```python
from core.geometric_evaluator import GeometricEvaluator

evaluator = GeometricEvaluator()
scores = evaluator.evaluate(frame1, frame2)
print(f"GRIC: {scores['gric']:.3f}")
print(f"マッチ数: {scores['feature_match_count']}")
```

---

### 5. `core/adaptive_selector.py`
適応的フレーム選択エンジン

**クラス:**
- `AdaptiveSelector`: SSIM, 光学フロー, カメラ動き推定

**静的メソッド:**
- `compute_ssim(frame1, frame2, window_size, sigma)` → float: SSIM値（-1～1）
  - スキップイムリキー依存なし（numpy/cv2のみ）
  
- `compute_optical_flow_magnitude(frame1, frame2, method)` → float: 光学フロー大きさ
  - 方式: 'farneback' （密フロー）または 'lucas_kanade' （疎フロー）
  
- `compute_camera_momentum(frames_window, method)` → float: カメラ加速度
  - ウィンドウ内の光学フロー変化率
  
- `get_adaptive_interval(momentum, base_interval, boost_factor, min/max_interval)` → int: 適応間隔
  - 加速度大 ⇒ 間隔短
  
- `evaluate(frame1, frame2, frames_window)` → dict: 適応的スコア

**使用例:**
```python
from core.adaptive_selector import AdaptiveSelector

# SSIM計算
ssim = AdaptiveSelector.compute_ssim(frame1, frame2)
print(f"SSIM: {ssim:.3f}")

# カメラ加速度から適応間隔を計算
momentum = AdaptiveSelector.compute_camera_momentum([f1, f2, f3])
interval = AdaptiveSelector.get_adaptive_interval(momentum, base_interval=10)
print(f"適応間隔: {interval}フレーム")
```

---

### 6. `core/keyframe_selector.py`
キーフレーム選択メインパイプライン

**データクラス:**
- `KeyframeInfo`: 選択されたキーフレーム情報
  - `frame_index`: フレーム番号
  - `timestamp`: 時刻（秒）
  - `quality_scores`: 品質スコア辞書
  - `geometric_scores`: 幾何学的スコア辞書
  - `adaptive_scores`: 適応的スコア辞書
  - `combined_score`: 統合スコア（0-1）
  - `thumbnail`: サムネイル画像（BGR）

**クラス:**
- `KeyframeSelector`: キーフレーム選択エンジン

**主要メソッド:**
- `select_keyframes(video_loader, progress_callback)` → List[KeyframeInfo]
  - アルゴリズム:
    1. 定期的にフレームをサンプリング
    2. 品質フィルタリング（鮮明度, モーションブラー）
    3. SSIMで変化を検出
    4. 幾何学的性質を評価
    5. 非最大値抑制（NMS）で最適キーフレーム選択
    6. 最大間隔制約を適用
  
- `export_keyframes(keyframes, video_loader, output_dir, format)` → Dict[int, Path]
  - キーフレーム画像をファイルにエクスポート

**設定パラメータ:**
```python
config = {
    # 重み設定（合計が1.0になるように）
    'WEIGHT_SHARPNESS': 0.30,
    'WEIGHT_EXPOSURE': 0.15,
    'WEIGHT_GEOMETRIC': 0.30,
    'WEIGHT_CONTENT': 0.25,
    
    # 品質閾値
    'LAPLACIAN_THRESHOLD': 100.0,
    'MOTION_BLUR_THRESHOLD': 0.3,
    
    # キーフレーム間隔（フレーム数）
    'MIN_KEYFRAME_INTERVAL': 5,
    'MAX_KEYFRAME_INTERVAL': 60,
    
    # その他パラメータ
    'SOFTMAX_BETA': 5.0,
    'GRIC_RATIO_THRESHOLD': 0.8,
    'SSIM_CHANGE_THRESHOLD': 0.85,
    'MIN_FEATURE_MATCHES': 30,
}
```

**使用例:**
```python
from core.video_loader import VideoLoader
from core.keyframe_selector import KeyframeSelector

# ビデオを読み込む
loader = VideoLoader()
loader.load('360video.mp4')

# キーフレームを選択
selector = KeyframeSelector()
keyframes = selector.select_keyframes(loader)

print(f"選択されたキーフレーム: {len(keyframes)}個")
for kf in keyframes:
    print(f"  フレーム {kf.frame_index}: {kf.timestamp:.2f}秒, スコア: {kf.combined_score:.3f}")

# エクスポート
exported = selector.export_keyframes(keyframes, loader, 'output/', format='png')
print(f"エクスポート完了: {len(exported)}個")
```

---

## 統合使用例

```python
import cv2
from core.video_loader import VideoLoader
from core.keyframe_selector import KeyframeSelector
from utils.logger import setup_logger

# ロガー設定
logger = setup_logger('360split', log_file='360split.log')

# カスタム設定
config = {
    'WEIGHT_SHARPNESS': 0.35,
    'WEIGHT_GEOMETRIC': 0.35,
    'WEIGHT_CONTENT': 0.20,
    'WEIGHT_EXPOSURE': 0.10,
    'MIN_KEYFRAME_INTERVAL': 3,
    'MAX_KEYFRAME_INTERVAL': 50,
}

# パイプライン実行
with VideoLoader() as loader:
    metadata = loader.load('360_video.mp4')
    logger.info(f"ビデオ読み込み完了: {metadata.duration:.1f}秒")
    
    selector = KeyframeSelector(config)
    
    # 進捗コールバック
    def progress(current, total):
        progress = current / total * 100
        print(f"進捗: {progress:.1f}%", end='\r')
    
    keyframes = selector.select_keyframes(loader, progress_callback=progress)
    
    logger.info(f"キーフレーム選択完了: {len(keyframes)}個")
    
    # エクスポート
    exported = selector.export_keyframes(
        keyframes, loader, 'keyframes_output/', format='png'
    )
```

---

## パフォーマンス最適化

### メモリ効率
- `VideoLoader` は一度に1フレームのみメモリに保持
- 大規模ビデオでも低メモリフットプリント

### 処理速度
- 特徴点検出: ORB（SIFT可選）
- 光学フロー: Farnebäck（密フロー）
- サンプリング間隔調整で処理時間を制御可能

### 依存関係
- 必須: `numpy`, `opencv-python`
- オプション: `scipy`, `scikit-image` (スキップ可能)

---

## エラーハンドリング

```python
from core.video_loader import VideoLoader

loader = VideoLoader()

try:
    loader.load('nonexistent.mp4')
except FileNotFoundError:
    print("ファイルが見つかりません")
    
try:
    loader.load('corrupted.mp4')
except RuntimeError:
    print("ビデオファイルが破損しているか形式がサポートされていません")
```

---

## 拡張方法

### カスタム評価器の追加
```python
class CustomEvaluator:
    @staticmethod
    def compute_custom_score(frame):
        # カスタムロジック
        return score
```

### 重み設定のカスタマイズ
```python
custom_weights = {
    'WEIGHT_SHARPNESS': 0.40,
    'WEIGHT_GEOMETRIC': 0.40,
    'WEIGHT_CONTENT': 0.15,
    'WEIGHT_EXPOSURE': 0.05,
}
selector = KeyframeSelector(custom_weights)
```

---

## トラブルシューティング

### 問題: キーフレームが選択されない
**原因:** `LAPLACIAN_THRESHOLD` が高すぎる
**解決:** 値を下げて試す

### 問題: キーフレーム数が多すぎる
**原因:** `MIN_KEYFRAME_INTERVAL` が小さい
**解決:** 値を増やす

### 問題: 光学フロー計算エラー
**原因:** OpenCVバージョンの不互換
**解決:** `adaptive_selector.py` で実装を確認

---

## ライセンス・参考文献

360Split は研究・教育目的で開発されました。

**実装の根拠:**
- Quality-based Selection: ラプラシアン分散、露光評価
- Geometric Selection: GRIC スコア、特徴点分析
- Adaptive Selection: SSIM、光学フロー、カメラ動き推定

