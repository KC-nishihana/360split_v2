# 360Split

360度動画からフォトグラメトリ・3D Gaussian Splatting (3DGS) 用の最適フレームを自動抽出するデスクトップツールです。

品質評価・幾何学的評価・適応的選択の3軸で動画フレームをスコアリングし、3D再構成に最適なキーフレームを自動選別します。GUIモードとCLIモードの両方に対応しています。

> **最新アップデート（2026年2月）**
> - Visual Odometry（カメラ軌跡推定）モジュールの実験的実装を追加
> - エクスポートダイアログUIの追加
> - 画像I/Oユーティリティの拡張
> - 設定デフォルトの定義を `ConfigManager.default_config()` に一本化
> - ステレオエクスポート時のステッチングモード（Fast/HQ/Depth-aware）を実装
> - 評価済みキーフレーム向けの対象マスク生成（YOLO + SAM）を追加
> - 設定ダイアログに「対象マスク」タブを追加（キーフレーム設定から分離）


## 主な特徴

- **2段階キーフレーム選択パイプライン** — Stage 1で品質ベースの高速フィルタリング（60〜70%除外）、Stage 2で幾何学的・適応的精密評価を行う効率的なアーキテクチャ
- **360度映像ネイティブ対応** — Equirectangular / Cubemap / Perspective投影変換をサポート
- **対象マスク生成** — 選択済みキーフレームに対して人物・車両など指定対象の二値マスク（対象=黒/背景=白）を生成
- **GPU高速化** — Apple Silicon (MPS) / NVIDIA CUDA の自動検出と活用
- **GUIモード** — PySide6ベースの直感的なインタフェースで動画プレビュー、タイムライン操作、リアルタイム分析が可能
- **CLIモード** — スクリプト連携やバッチ処理に対応したコマンドラインインタフェース
- **クロスプラットフォーム** — macOS (Apple Silicon) / Windows (CUDA) / Linux で動作


## スクリーンショット

（準備中）


## 動作環境

- Python 3.10以上（3.12推奨）
- OS: macOS (Apple Silicon) / Windows 10+ / Linux
- メモリ: 16GB以上推奨（8K映像処理時）
- GPU（オプション）: Apple Silicon (Metal/MPS) または NVIDIA GPU (CUDA 12.1+)


## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/YOUR_USERNAME/360split.git
cd 360split
```

（注：`YOUR_USERNAME`を実際のGitHubユーザー名に置き換えてください）

### 2. Python仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. GPU高速化（オプション）

`core/accelerator.py` がハードウェア（MPS/CUDA/CPU）を自動検出します。GPUを使用する場合は以下をインストールしてください。

**Apple Silicon (MPS) の場合:**
```bash
pip install torch torchvision
```

**Windows/Linux (CUDA) の場合:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA対応OpenCV（さらに高速化、オプション）:**
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### 5. 対象マスク生成（オプション）

対象マスク生成（YOLO + SAM）を利用する場合は `ultralytics` を追加インストールしてください。

```bash
pip install ultralytics
```


## 使い方

### GUIモード

```bash
python main.py
```

GUIが起動したら、メニューまたはツールバーから動画ファイルを読み込み、「分析実行」でキーフレーム抽出を開始します。

### CLIモード

```bash
# 基本的な使い方
python main.py --cli input_video.mp4

# 出力先とフォーマットを指定
python main.py --cli input.mp4 -o ./output --format png

# 360度動画としてCubemapも出力
python main.py --cli input.mp4 --equirectangular --cubemap

# 天底マスク処理を有効化（三脚・撮影者除去）
python main.py --cli input.mp4 --apply-mask

# 設定ファイルを指定
python main.py --cli input.mp4 --config settings.json

# 環境プリセットを使用
python main.py --cli input.mp4 --preset outdoor  # 屋外・高品質
python main.py --cli input.mp4 --preset indoor   # 屋内・追跡重視
python main.py --cli input.mp4 --preset mixed    # 混合・適応型

# 詳細ログ出力
python main.py --cli input.mp4 -v
```

#### CLIオプション一覧

| オプション | 説明 | デフォルト |
|---|---|---|
| `--cli VIDEO` | CLIモードで動画を解析 | — |
| `-o, --output DIR` | 出力ディレクトリ | `./keyframes` |
| `--format {png,jpg,tiff}` | 出力画像フォーマット | `png` |
| `--preset {outdoor,indoor,mixed}` | 環境プリセット（下記参照） | — |
| `--max-keyframes N` | 最大キーフレーム数 | 自動決定 |
| `--min-interval N` | 最小キーフレーム間隔（フレーム数） | `5` |
| `--ssim-threshold F` | SSIM変化検知閾値 (0.0-1.0) | `0.85` |
| `--equirectangular` | 360度Equirectangular動画として処理 | `false` |
| `--apply-mask` | 天底マスク処理を適用 | `false` |
| `--cubemap` | Cubemap形式でも出力 | `false` |
| `--config FILE` | 設定ファイル（JSON） | — |
| `-v, --verbose` | 詳細ログ出力 | `false` |
| `--rerun-stream` | 抽出中にRerunへストリーミング | `false` |
| `--rerun-spawn` | Rerun Viewerを自動起動（`--rerun-stream`時） | `false` |
| `--rerun-save PATH` | Rerunログを`.rrd`で保存 | — |

### Rerunログ（キーフレーム検証）

抽出結果をRerunで可視化し、`metrics/*` の時系列と軌跡を同時に確認できます。

GUIでは右側の`⚙ 設定`パネル（または設定ダイアログ）で
`解析時にRerunログを有効化（GUI）` をONにすると、Stage 2 / フル解析時にRerunへ送信されます。

```bash
# オンラインストリーミング（Viewer起動）
python main.py --cli input.mp4 --rerun-stream --rerun-spawn

# ストリーミング + .rrd保存
python main.py --cli input.mp4 --rerun-stream --rerun-spawn --rerun-save ./logs/keyframe_check.rrd
```

ログされる主なエンティティ:
- `cam/image`
- `world/cam`
- `world/trajectory`
- `world/keyframes`
- `metrics/translation_delta`
- `metrics/rotation_delta`
- `metrics/laplacian_var`
- `metrics/match_count`
- `metrics/overlap_ratio`
- `metrics/exposure_ratio`
- `metrics/keyframe_flag`

### オフライン再生（CSV/JSON -> .rrd）

抽出後データ（CSVまたはJSON）を順番に再生して`.rrd`を生成できます。

```bash
python scripts/rerun_offline_replay.py \
  --input ./logs/frame_metrics.json \
  --rrd ./logs/keyframe_offline.rrd \
  --spawn
```

入力レコードの主なキー:
- `frame_index` または `frame_idx`
- `image_path`（任意）
- `t_xyz` / `q_wxyz`（任意、未指定時は既定値）
- `is_keyframe` または `keyframe_flag`
- `metrics`（辞書）または `translation_delta` などの列
- `points_world` または `points_path`（`.npy`、任意）


## 環境プリセット機能

撮影環境（屋外・屋内・混合）に応じて最適なキーフレーム抽出パラメータを即座に適用できるプリセット管理システムを搭載しています。

### プリセット一覧

#### 🌞 Outdoor (屋外・高品質)
**戦略**: 品質重視。3D Gaussian Splattingの学習データとして最高品質を目指します。

```bash
python main.py --cli video.mp4 --preset outdoor
```

**特徴**:
- **laplacian_threshold: 300.0** — 非常に高い鮮明度を要求
- **motion_blur_threshold: 0.15** — わずかなブレも許容しない
- **min_keyframe_interval: 10** — 遠景が多いため間隔を広げてデータ量を削減
- **weight_sharpness: 0.40** — 鮮明度を最重視

**適用シーン**: 晴天の屋外撮影、建築物の外観、風景の3Dモデル作成

#### 🏠 Indoor (屋内・追跡重視)
**戦略**: 接続性重視。特徴点が少なく暗い環境でもSfMの追跡が切れないことを最優先します。

```bash
python main.py --cli video.mp4 --preset indoor
```

**特徴**:
- **laplacian_threshold: 50.0** — 高感度ノイズやソフトフォーカスを許容
- **motion_blur_threshold: 0.4** — 多少のブレよりもフレーム数を確保
- **weight_geometric: 0.60** — 幾何学的な繋がりを最重視
- **min_feature_matches: 20** — 特徴点が少ない壁面などに対応
- **enable_rescue_mode: true** — レスキューモード有効化

**適用シーン**: 屋内撮影、暗所、テクスチャの少ない環境（オフィス、廊下など）

#### 🌗 Mixed (混合・適応型)
**戦略**: 適応性重視。明暗差（ダイナミックレンジ）の激しい変化に対応します。

```bash
python main.py --cli video.mp4 --preset mixed
```

**特徴**:
- **weight_exposure: 0.40** — 露出の急激な変化を検知
- **ssim_threshold: 0.90** — シーンの変化に敏感に反応
- **adaptive_thresholding: true** — スコア履歴に基づく動的しきい値
- **force_keyframe_on_exposure_change: true** — ドアを抜けた瞬間などの急激な画変わりで強制挿入

**適用シーン**: 屋外⇔屋内の移動、トンネル出入口、明暗差の激しい環境

### GUIでのプリセット使用

GUIモードでは、設定ダイアログの「キーフレーム選択」タブ上部にプリセット選択UIがあります。

1. プリセットを選択すると、全パラメータが即座に更新されます
2. プリセット適用後も、個別のパラメータを微調整可能です
3. 「Custom (手動設定)」を選択すると、プリセット非適用状態になります

設定ダイアログのタブ構成:
- `キーフレーム選択`: 評価・選択パラメータ
- `360度処理`: 投影・解像度・ステッチング設定
- `マスク処理`: ナディア/装備マスク設定
- `出力設定`: 出力形式・品質・ディレクトリ
- `対象マスク`: 対象クラス、YOLO/SAMモデル、信頼度閾値、マスク命名規則

### レスキューモード

**Indoor**および**Mixed**プリセットでは、特徴点不足時の「レスキューモード」が有効化されています。

**機能**:
- 特徴点マッチング数が閾値（デフォルト15点）以下の状態が続くと自動的に発動
- Laplacian閾値を一時的に緩和（デフォルト: 50%倍率）
- 画質が低くても、追跡維持のためにキーフレームを強制採用
- GUIタイムライン上で黄色/オレンジ色で表示され、後で確認・削除可能

**目的**: 屋内・暗所でSfMのトラッキングロストを防止


## アーキテクチャ

### プロジェクト構成

```
360split/
├── main.py                  # エントリポイント（GUI/CLI）
├── config.py                # 設定定数・データクラス定義
├── requirements.txt         # 依存パッケージ
├── presets/                 # 環境別プリセット設定
│   ├── outdoor_high_quality.json     # 屋外・高品質プリセット
│   ├── indoor_robust_tracking.json   # 屋内・追跡重視プリセット
│   └── mixed_adaptive.json           # 混合・適応型プリセット
├── core/                    # コアアルゴリズム
│   ├── config_loader.py     # プリセット管理システム
│   ├── accelerator.py       # ハードウェア抽象化レイヤ（MPS/CUDA/CPU自動検出）
│   ├── video_loader.py      # ビデオ読み込み（HWデコード、LRUキャッシュ、プリフェッチ）
│   ├── quality_evaluator.py # 品質評価（ラプラシアン鮮明度、露光、モーションブラー）
│   ├── geometric_evaluator.py # 幾何学的評価（GRIC、特徴点マッチング、光線分散）
│   ├── adaptive_selector.py # 適応的選択（SSIM、オプティカルフロー、カメラ運動量）
│   ├── keyframe_selector.py # 2段階パイプライン統合（メインエンジン）
│   └── exceptions.py        # カスタム例外定義
├── processing/              # 360度画像処理 / 対象マスク処理
│   ├── equirectangular.py   # Equirectangular ↔ Cubemap / Perspective変換
│   ├── mask_processor.py    # 天底/天頂マスク、撮影機材マスク生成
│   ├── stitching.py         # 画像スティッチング（Fast/HQS/DMS 3モード）
│   ├── object_detector.py   # YOLOベース物体検出
│   ├── instance_segmentor.py # SAMベースインスタンスセグメンテーション
│   └── target_mask_generator.py # 対象マスク生成（OR合成、二値化、保存パス生成）
├── gui/                     # PySide6 GUI
│   ├── main_window.py       # メインウィンドウ、UI統合
│   ├── video_player.py      # 動画プレビュー、フレームナビゲーション
│   ├── timeline_widget.py   # タイムラインUI、pyqtgraphスコアグラフ
│   ├── keyframe_panel.py    # キーフレーム一覧・詳細表示パネル（MainWindowで使用）
│   ├── keyframe_list.py     # 旧キーフレーム一覧ウィジェット（互換用）
│   ├── settings_dialog.py   # 設定ダイアログ（5タブ構成）
│   ├── settings_panel.py    # 設定パネルコンポーネント
│   ├── export_dialog.py     # エクスポートダイアログ（出力設定、フォーマット選択）
│   └── workers.py           # バックグラウンド処理ワーカー（Stage1/2/Export）
├── utils/                   # ユーティリティ
│   ├── logger.py            # ロギングユーティリティ
│   └── image_io.py          # 画像入出力（フォーマット変換、メタデータ保存）
└── test/                    # 開発用テストスクリプト、サンプルデータ
```

### 2段階キーフレーム選択パイプライン

```
入力動画 (360° / 通常)
         │
    ┌────▼─────────────────────────────────────────┐
    │  Stage 1: 高速品質フィルタリング               　│
    │  ─────────────────────────────────────────   │
    │  • ラプラシアン鮮明度評価                        │
    │  • モーションブラー検出                         │
    │  • 露光・輝度バランス                           │
    │  • Softmax深度スコアリング                      │
    │  → 全フレームの60〜70%を高速除外                 │
    └────┬─────────────────────────────────────────┘
         │ 候補フレーム (30〜40%)
    ┌────▼─────────────────────────────────────────┐
    │  Stage 2: 精密幾何学・適応評価                  │
    │  ─────────────────────────────────────────    │
    │  • GRIC: ホモグラフィ vs 基礎行列比較          │
    │  • 特徴点マッチングと空間分布評価                │
    │  • SSIM変化量による冗長フレーム除外              │
    │  • オプティカルフロー（カメラ運動量）             │
    │  → NMS（非最大値抑制）で最終選別                 │
    └────┬─────────────────────────────────────────┘
         │
    最適キーフレーム群 → エクスポート
```


## 設定ファイル

JSON形式の設定ファイルで各パラメータをカスタマイズできます。デフォルト値は `core/config_loader.py` の `ConfigManager.default_config()` で一元管理され、JSON形式でオーバーライド可能です。

```json
{
  "laplacian_threshold": 100.0,
  "motion_blur_threshold": 0.3,
  "softmax_beta": 5.0,
  "gric_degeneracy_threshold": 0.85,
  "min_feature_matches": 15,
  "ssim_threshold": 0.85,
  "min_keyframe_interval": 5,
  "max_keyframe_interval": 60,
  "weight_sharpness": 0.30,
  "weight_exposure": 0.15,
  "weight_geometric": 0.30,
  "weight_content": 0.25,
  "enable_polar_mask": true,
  "mask_polar_ratio": 0.10,
  "enable_stereo_stitch": true,
  "stitching_mode": "Fast",
  "output_image_format": "png",
  "output_jpeg_quality": 95,
  "enable_target_mask_generation": false,
  "target_classes": ["人物", "人", "自転車", "バイク", "車両", "動物"],
  "yolo_model_path": "yolo26n-seg.pt",
  "sam_model_path": "sam3_t.pt",
  "confidence_threshold": 0.25,
  "detection_device": "auto",
  "mask_output_dirname": "masks",
  "mask_add_suffix": true,
  "mask_suffix": "_mask",
  "mask_output_format": "same"
}
```

### 主要パラメータ

| パラメータ | 説明 | デフォルト値 |
|---|---|---|
| `laplacian_threshold` | ラプラシアン鮮明度の最小閾値（Stage 1）。低いほどブレたフレームも許容 | 100.0 |
| `motion_blur_threshold` | モーションブラー許容閾値。高いほど動きのあるフレームも許容 | 0.3 |
| `ssim_threshold` | SSIM変化検知閾値。低いほど大きな変化のみキーフレームとして採用 | 0.85 |
| `min_keyframe_interval` | キーフレーム間の最小フレーム数 | 5 |
| `max_keyframe_interval` | キーフレーム間の最大フレーム数（超過時は強制挿入） | 60 |
| `weight_sharpness` | 鮮明度スコアの重み（α） | 0.30 |
| `weight_geometric` | 幾何学的スコアの重み（β、GRIC） | 0.30 |
| `weight_content` | コンテンツ変化スコアの重み（γ、SSIM） | 0.25 |
| `weight_exposure` | 露光スコアの重み（δ） | 0.15 |
| `gric_degeneracy_threshold` | GRIC縮退判定閾値。高いほど回転のみシーンを除外しやすい | 0.85 |
| `enable_polar_mask` | 360度動画の天頂/天底マスク有効化 | true |
| `mask_polar_ratio` | 天頂/天底マスク比率（上下の何%をマスクするか） | 0.10 |
| `enable_target_mask_generation` | キーフレーム出力後に対象マスクを生成 | false |
| `target_classes` | 検出対象ラベル（複数選択可） | `["人物","人","自転車","バイク","車両","動物"]` |
| `yolo_model_path` | YOLOモデル名/パス | `yolo26n-seg.pt` |
| `sam_model_path` | SAMモデル名/パス | `sam3_t.pt` |
| `confidence_threshold` | 検出信頼度閾値 | 0.25 |
| `detection_device` | 推論デバイス（`auto/cpu/mps/cuda/0`） | `auto` |
| `mask_output_dirname` | マスク出力ディレクトリ名 | `masks` |
| `mask_add_suffix` | マスクファイル名に接尾辞を付与 | true |
| `mask_suffix` | マスクファイル接尾辞 | `_mask` |
| `mask_output_format` | マスク拡張子（`same/png/jpg/tiff`） | `same` |


## 出力

### キーフレーム画像

指定フォーマット（PNG/JPEG/TIFF）で出力されます。ファイル名にはフレーム番号が含まれます。

```
output/
├── keyframe_000000.png
├── keyframe_000150.png
├── keyframe_000312.png
├── ...
├── keyframe_metadata.json
└── cubemap/                    # --cubemap指定時
    ├── frame_000000/
    │   ├── front.png
    │   ├── back.png
    │   ├── left.png
    │   ├── right.png
    │   ├── top.png
    │   └── bottom.png
    └── ...
```

### 対象マスク画像（有効時）

対象検出を有効化した場合、キーフレームと同じ相対構造で `masks/` 配下へ保存されます。

```
output/
├── L/
│   ├── keyframe_000150_L.png
│   └── ...
├── R/
│   ├── keyframe_000150_R.png
│   └── ...
└── masks/
    ├── L/
    │   ├── keyframe_000150_L_mask.png
    │   └── ...
    └── R/
        ├── keyframe_000150_R_mask.png
        └── ...
```

- 画素値: 対象領域 `0`（黒）、背景 `255`（白）
- 対象未検出フレーム: 全白マスク
- 命名規則: `mask_add_suffix` / `mask_suffix` / `mask_output_format` で変更可能

### メタデータ (keyframe_metadata.json)

各キーフレームのスコア詳細を含むJSONファイルが自動生成されます。フォトグラメトリソフトウェアとの連携や後処理スクリプトの入力として利用できます。


## 実験的機能（開発中）

### Visual Odometry（カメラ軌跡推定）

`test/`ディレクトリに、360度動画からカメラの3D軌跡を推定するVisual Odometry（VO）モジュールが開発されています。

**主な機能:**
- モノキュラーVisual Odometry（単眼カメラによる自己位置推定）
- 既存の`GeometricEvaluator`を活用した特徴点検出・マッチング
- Essential Matrix推定によるカメラ姿勢計算
- IMU融合による絶対スケール推定（開発中）
- 3D軌跡の可視化

**使用方法:**

```bash
cd test/

# VO単独実行
python3 vo_only_test.py

# VO+IMU融合実行（IMUデータ準備後）
python3 vo_imu_fusion.py
```

**詳細:**
- `test/VO開発完了レポート.md` - 技術詳細とアーキテクチャ
- `test/3D軌跡解析レポート.md` - 解析結果レポート
- `test/OSVファイル_メタデータ調査レポート.md` - センサーデータ調査

**注意:** この機能は実験的なものであり、メインシステムへの統合は今後予定されています。


## ライセンス

（準備中）
