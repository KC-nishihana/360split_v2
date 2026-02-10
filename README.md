# 360Split

360度動画からフォトグラメトリ・3D Gaussian Splatting (3DGS) 用の最適フレームを自動抽出するデスクトップツールです。

品質評価・幾何学的評価・適応的選択の3軸で動画フレームをスコアリングし、3D再構成に最適なキーフレームを自動選別します。GUIモードとCLIモードの両方に対応しています。


## 主な特徴

- **2段階キーフレーム選択パイプライン** — Stage 1で品質ベースの高速フィルタリング（60〜70%除外）、Stage 2で幾何学的・適応的精密評価を行う効率的なアーキテクチャ
- **360度映像ネイティブ対応** — Equirectangular / Cubemap / Perspective投影変換をサポート
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
git clone https://github.com/yourname/360split.git
cd 360split
```

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

Apple Silicon (MPS) の場合:
```bash
pip install torch torchvision
```

Windows (CUDA) の場合:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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

# 詳細ログ出力
python main.py --cli input.mp4 -v
```

#### CLIオプション一覧

| オプション | 説明 | デフォルト |
|---|---|---|
| `--cli VIDEO` | CLIモードで動画を解析 | — |
| `-o, --output DIR` | 出力ディレクトリ | `./keyframes` |
| `--format {png,jpg,tiff}` | 出力画像フォーマット | `png` |
| `--max-keyframes N` | 最大キーフレーム数 | 自動決定 |
| `--min-interval N` | 最小キーフレーム間隔（フレーム数） | `5` |
| `--ssim-threshold F` | SSIM変化検知閾値 (0.0-1.0) | `0.85` |
| `--equirectangular` | 360度Equirectangular動画として処理 | `false` |
| `--apply-mask` | 天底マスク処理を適用 | `false` |
| `--cubemap` | Cubemap形式でも出力 | `false` |
| `--config FILE` | 設定ファイル（JSON） | — |
| `-v, --verbose` | 詳細ログ出力 | `false` |


## アーキテクチャ

### プロジェクト構成

```
360split/
├── main.py                  # エントリポイント（GUI/CLI）
├── config.py                # 設定定数
├── requirements.txt         # 依存パッケージ
├── core/                    # コアアルゴリズム
│   ├── accelerator.py       # ハードウェア抽象化レイヤ（MPS/CUDA/CPU自動検出）
│   ├── video_loader.py      # ビデオ読み込み（HWデコード、LRUキャッシュ、プリフェッチ）
│   ├── quality_evaluator.py # 品質評価（ラプラシアン鮮明度、露光、モーションブラー）
│   ├── geometric_evaluator.py # 幾何学的評価（GRIC、特徴点マッチング、光線分散）
│   ├── adaptive_selector.py # 適応的選択（SSIM、オプティカルフロー、カメラ運動量）
│   └── keyframe_selector.py # 2段階パイプライン統合（メインエンジン）
├── processing/              # 360度画像処理
│   ├── equirectangular.py   # Equirectangular ↔ Cubemap / Perspective変換
│   ├── mask_processor.py    # 天底/天頂マスク、撮影機材マスク生成
│   └── stitching.py         # 画像スティッチング（Fast/HQS/DMS 3モード）
├── gui/                     # PySide6 GUI
│   ├── main_window.py       # メインウィンドウ、分析ワーカースレッド
│   ├── video_player.py      # 動画プレビュー、フレームナビゲーション
│   ├── timeline_widget.py   # タイムラインUI、品質スコア可視化
│   ├── keyframe_panel.py    # キーフレーム一覧、サムネイル表示
│   └── settings_dialog.py   # 設定ダイアログ（4タブ構成）
└── utils/
    └── logger.py            # ロギングユーティリティ
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

JSON形式の設定ファイルで各パラメータをカスタマイズできます。

```json
{
  "laplacian_threshold": 100.0,
  "motion_blur_threshold": 0.3,
  "softmax_beta": 5.0,
  "gric_ratio_threshold": 0.8,
  "min_feature_matches": 30,
  "ssim_change_threshold": 0.85,
  "min_keyframe_interval": 5,
  "max_keyframe_interval": 60,
  "weight_sharpness": 0.30,
  "weight_exposure": 0.15,
  "weight_geometric": 0.30,
  "weight_content": 0.25,
  "output_image_format": "png",
  "output_jpeg_quality": 95
}
```

### 主要パラメータ

| パラメータ | 説明 | 推奨値 |
|---|---|---|
| `laplacian_threshold` | ラプラシアン鮮明度の最小閾値。低いほどブレたフレームも許容 | 100.0 |
| `ssim_change_threshold` | SSIMの変化閾値。低いほど大きな変化のみキーフレームとして採用 | 0.85 |
| `min_keyframe_interval` | キーフレーム間の最小フレーム数 | 5 |
| `max_keyframe_interval` | キーフレーム間の最大フレーム数（超過時は強制挿入） | 60 |
| `weight_sharpness` | 鮮明度スコアの重み | 0.30 |
| `weight_geometric` | 幾何学的スコアの重み | 0.30 |
| `weight_content` | コンテンツ変化スコアの重み | 0.25 |
| `weight_exposure` | 露光スコアの重み | 0.15 |


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

### メタデータ (keyframe_metadata.json)

各キーフレームのスコア詳細を含むJSONファイルが自動生成されます。フォトグラメトリソフトウェアとの連携や後処理スクリプトの入力として利用できます。


## ライセンス

（準備中）
