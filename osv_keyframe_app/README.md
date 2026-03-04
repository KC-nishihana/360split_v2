# OSV Keyframe App

OSV（前後2魚眼カメラ）動画からキーフレームを抽出し、SfM/3DGS用に最適な画像セットを生成するツール。

## 概要

1. OSV動画をfront/backストリームに分離（ffmpeg）
2. 各ストリームのfisheye画像を4方向（front/left/right/back）のpinhole画像に投影
3. 各画像の品質指標（sharpness, exposure, ORB, SSIM）を計算
4. 2系統の閾値で選別：SfM用（厳格）/ 3DGS用（緩和）
5. 選別結果をCOLMAPに投入（任意）

## 前提条件

- Python 3.12+
- ffmpeg（OSVストリーム分離に必要）
- COLMAP（オプション、SfM/image_registratorに必要）

### インストール

```bash
pip install opencv-python numpy pyyaml PySide6 pyqtgraph scikit-image
```

## 使い方

### CLI

```bash
python -m osv_keyframe_app --config config.yaml --osv input.osv
```

#### オプション

| フラグ | 説明 |
|--------|------|
| `--config` | YAML設定ファイル（必須） |
| `--osv` | OSVファイルパス（config内のosv_pathを上書き） |
| `--output`, `-o` | 出力ディレクトリ（デフォルト: config.output_dir） |
| `--no-colmap` | COLMAP処理をスキップ |
| `--gui` | GUIモードで起動 |
| `--verbose`, `-v` | 詳細ログ出力 |

### GUI

```bash
python -m osv_keyframe_app --config config.yaml --gui
```

#### GUIタブ

- **Metrics**: sharpness/exposure/ORB/SSIMの時系列グラフ。stream/directionでフィルタ可能
- **Selection**: SfM/3DGSの閾値スライダ。スライダ操作で採用枚数が即座に更新。方向別内訳テーブル付き
- **COLMAP**: SfM(mapper)/Registration(image_registrator)/Triangulatorの実行ボタン。ログ表示と登録結果テーブル

## 設定ファイル例（config.yaml）

```yaml
camera_front:
  K: [[1200, 0, 960], [0, 1200, 960], [0, 0, 1]]
  D: [0.01, -0.02, 0.005, -0.001]
  image_size: [1920, 1920]
  model: fisheye

camera_back:
  K: [[1200, 0, 960], [0, 1200, 960], [0, 0, 1]]
  D: [0.01, -0.02, 0.005, -0.001]
  image_size: [1920, 1920]
  model: fisheye

projection:
  directions:
    - {name: front, yaw_deg: 0}
    - {name: left, yaw_deg: -90}
    - {name: right, yaw_deg: 90}
    - {name: back, yaw_deg: 180}
  hfov_deg: 90
  vfov_deg: 90
  output_size: [1600, 1600]

extraction:
  fps: 2.0
  start_sec: 0.0
  end_sec: null
  max_frames: null

selection:
  sfm:
    sharpness_min: 150.0
    exposure_min: 0.4
    orb_min: 100
    ssim_max: 0.92
    per_direction_min: 20
    max_total: null
  gs:
    sharpness_min: 80.0
    exposure_min: 0.3
    orb_min: 50
    ssim_max: 0.96
    per_direction_min: 10
    max_total: null

colmap:
  enabled: false
  binary_path: colmap
  workspace: colmap_workspace
  camera_model: PINHOLE
  use_gpu: false

output_dir: out
project_name: my_project
```

## 出力ファイル

| ファイル | 説明 |
|----------|------|
| `metrics_all.csv` | 全フレーム全方向の品質指標 |
| `manifest.csv` | 選別結果（採否理由、スコア、tier） |
| `out/sfm/images/` | SfM用に選択された画像 |
| `out/gs/images/` | 3DGS用に選択された画像（SfMの上位集合） |
| `config_used.json` | 再現性のための使用設定 |

### metrics_all.csv カラム

| カラム | 説明 |
|--------|------|
| `frame_idx` | フレーム番号 |
| `timestamp` | タイムスタンプ（秒） |
| `stream` | front / back |
| `direction` | front / left / right / back |
| `laplacian_var` | Laplacian分散（シャープネス指標） |
| `mean_intensity` | 平均輝度 |
| `clipped_high_ratio` | 白飛び比率（>245） |
| `clipped_low_ratio` | 黒潰れ比率（<16） |
| `exposure_score` | 露出スコア [0,1] |
| `orb_keypoints` | ORBキーポイント数 |
| `ssim_prev` | 前フレームとのSSIM（同stream/direction） |

## COLMAP二段構え

### Phase 1: SfM（mapper）

SfM用画像で基本的なSparse Reconstructionを構築：

```
feature_extractor → sequential_matcher → mapper
```

### Phase 2: 3DGS追加登録（image_registrator）

3DGS用の追加画像を既存モデルに登録（BA/三角測量なし）：

```
feature_extractor(追加分) → matcher → image_registrator
```

必要に応じて`point_triangulator`で三角測量を実行。

## よくある失敗と対処

### 向き・反転の問題

- **症状**: 投影画像が裏返し、または意図しない方向
- **対処**: `projection.directions`のyaw_degを調整。OSVのstream 0/1がfront/backに正しくマッピングされているか確認

### 露出の問題

- **症状**: 白飛び/黒潰れが多く、選別でほぼ全部落ちる
- **対処**: `selection.sfm.exposure_min`を下げる（0.2程度）。clipped_high_ratio / clipped_low_ratioをmetrics_all.csvで確認

### COLMAP登録失敗

- **症状**: image_registratorで登録率が低い
- **対処**:
  - SfM画像数が少なすぎないか確認（per_direction_min を増やす）
  - matching_method を `exhaustive` に変更
  - 3DGS画像のsharpness_minを上げて品質を改善

### ffmpegが見つからない

```
macOS:   brew install ffmpeg
Ubuntu:  sudo apt install ffmpeg
Windows: https://ffmpeg.org/download.html
```

### COLMAPが見つからない

```
macOS:   brew install colmap
Ubuntu:  https://colmap.github.io/install.html
Windows: https://github.com/colmap/colmap/releases
```

## テスト

```bash
pytest tests/test_osv_*.py tests/test_fisheye_projector.py -v
```
