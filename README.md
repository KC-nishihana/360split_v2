# 360Split

360度動画から、SfM/フォトグラメトリ/3D Gaussian Splatting (3DGS) 向けフレームを抽出する GUI/CLI ツールです。

`Stage1 -> Stage0 -> Stage2 -> Stage3` の統合パイプラインで、
品質・幾何・運動の観点からキーフレームを選別します。

## 2026-02 主要アップデート

- Stage1に **A案品質フィルタ** を導入
  - ROI中心評価（魚眼向け）
  - 分位点正規化（p10/p90）
  - 合成品質スコア `quality in [0,1]`
  - 絶対ガード（Laplacian下限）
- 品質診断ファイルを追加
  - `quality_metrics.json`
  - `quality_metrics.csv`
- CLIに品質フィルタ制御オプションを追加
  - `--quality-filter/--no-quality-filter`
  - `--quality-threshold`
  - `--quality-roi`
  - `--quality-abs-laplacian-min`
  - `--quality-debug`
- GUI設定に品質フィルタのON/OFF・閾値を追加

## 主な機能

- 4段階解析パイプライン（Stage0/1/2/3）
- 単眼動画 / `.OSV`（L/R） / 前後魚眼2動画（Front/Rear）に対応
- 360処理（Equirectangular/Cubemap）
- 魚眼外周マスク、cross5分割出力（OSV）
- VOベースの軌跡再評価（Stage0/Stage3）
- GUI解析、CLIバッチ処理、Rerun可視化

## 動作環境

- Python 3.10+
- macOS / Windows / Linux
- 推奨メモリ: 16GB以上
- GPU（任意）
  - Apple Silicon (MPS)
  - NVIDIA CUDA

## インストール

```bash
git clone https://github.com/KC-nishihana/360split_v2.git
cd 360split
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

オプション:

```bash
# YOLO/SAM利用時
pip install ultralytics

# Apple Silicon (MPS)
pip install torch torchvision

# CUDA環境
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## クイックスタート

### GUI

```bash
python main.py
```

### CLI（基本）

```bash
python main.py --cli input.mp4
python main.py --cli input.osv -o output_dir
python main.py --front-video front.mp4 --rear-video rear.mp4 -o output_dir
```

## A案 品質フィルタ（Stage1）

デフォルトで有効です。

- `quality_filter_enabled = true`
- `quality_threshold = 0.50`
- `quality_roi_mode = circle`
- `quality_roi_ratio = 0.40`
- `quality_abs_laplacian_min = 35.0`
- `quality_use_orb = true`

合成重み（デフォルト）:

- `sharpness = 0.40`
- `tenengrad = 0.30`
- `exposure = 0.15`
- `keypoints = 0.15`

正規化:

- `p_low = 10`, `p_high = 90`
- バッチ内分位点で各指標を `[0,1]` 化

判定:

- `quality >= quality_threshold`
- かつ `laplacian_var >= quality_abs_laplacian_min`

ペア入力（OSV/Front-Rear）:

- 各レンズ別に正規化・合成
- 最終 `quality = min(quality_lens_a, quality_lens_b)`
- 絶対ガードは両レンズ適用

## CLI 使用例

```bash
# 品質フィルタしきい値を調整
python main.py --cli input.mp4 --quality-threshold 0.60

# ROIを矩形に変更
python main.py --cli input.mp4 --quality-roi rect:0.60

# 絶対Laplacianガードを変更
python main.py --cli input.mp4 --quality-abs-laplacian-min 50

# 品質フィルタを無効化（旧判定へフォールバック）
python main.py --cli input.mp4 --no-quality-filter

# 品質デバッグ統計ログを有効化
python main.py --cli input.mp4 --quality-debug

# Stage0/Stage3を制御
python main.py --cli input.mp4 --disable-stage0-scan
python main.py --cli input.mp4 --disable-stage3-refinement

# 動体除去を有効化
python main.py --cli input.mp4 --remove-dynamic-objects

# VO用キャリブレーション指定
python main.py --cli input.mp4 --calib-xml calib/cam1.xml --calib-model auto
```

## CLI オプション一覧

### 入出力・基本

| オプション | 説明 |
|---|---|
| `--cli VIDEO` | CLI入力動画 |
| `--front-video PATH` | 前後魚眼入力（front） |
| `--rear-video PATH` | 前後魚眼入力（rear） |
| `-o, --output DIR` | 出力先ディレクトリ |
| `--config FILE` | 設定JSON |
| `--analysis-run-id ID` | 解析実行ID |
| `--resume` | 既存runの中間成果で再開 |
| `--keep-temp` | 正常終了時も中間成果を保持 |
| `--colmap-format` | `colmap/` 互換出力を生成 |
| `--preset {outdoor,indoor,mixed}` | 環境プリセット |
| `--format {png,jpg,tiff}` | 出力画像形式 |
| `--max-keyframes N` | 出力上限数 |
| `--min-interval N` | 最小キーフレーム間隔 |
| `--ssim-threshold F` | SSIMしきい値 |
| `-v, --verbose` | 詳細ログ |
| `--profile` | ステージ性能ログ |

### 360 / マスク / 動体

| オプション | 説明 |
|---|---|
| `--equirectangular` | 入力をEquirectangularとして扱う |
| `--apply-mask` | ナディアマスク適用 |
| `--cubemap` | Cubemap出力 |
| `--remove-dynamic-objects` | Stage2動体除去有効 |
| `--dynamic-mask-frames N` | モーション差分フレーム数 |
| `--dynamic-mask-threshold N` | モーション差分しきい値 |
| `--dynamic-mask-dilation N` | 動体マスク膨張サイズ |
| `--dynamic-mask-inpaint` | インペイント有効 |
| `--dynamic-mask-inpaint-module MOD` | インペイントモジュール |

### Stage0/1/3・品質フィルタ

| オプション | 説明 |
|---|---|
| `--disable-stage0-scan` | Stage0無効化 |
| `--stage0-stride N` | Stage0サンプリング間隔 |
| `--stage1-grab-threshold N` | Stage1 grab切替閾値 |
| `--stage1-eval-scale F` | Stage1評価縮小率 |
| `--opencv-threads N` | OpenCVスレッド数（0=auto） |
| `--stage1-process-workers N` | Stage1品質計算プロセス数（0=auto） |
| `--stage1-prefetch-size N` | Stage1先読みキューサイズ |
| `--stage1-metrics-batch-size N` | Stage1品質計算バッチサイズ |
| `--stage1-gpu-batch / --no-stage1-gpu-batch` | Stage1 GPUバッチ品質計算 ON/OFF |
| `--darwin-capture-backend {auto,avfoundation,ffmpeg}` | macOS VideoCaptureバックエンド指定 |
| `--mps-min-pixels N` | MPS経路を使う最小画素数 |
| `--quality-filter / --no-quality-filter` | A案品質フィルタ ON/OFF |
| `--quality-threshold F` | 品質しきい値 |
| `--quality-roi SPEC` | `circle:0.40` / `rect:0.60` |
| `--quality-abs-laplacian-min F` | 絶対Laplacian下限 |
| `--quality-debug / --no-quality-debug` | 品質デバッグログ |
| `--disable-stage3-refinement` | Stage3無効化 |
| `--stage3-weight-base F` | Stage3 base重み |
| `--stage3-weight-trajectory F` | Stage3 trajectory重み |
| `--stage3-weight-stage0-risk F` | Stage3 stage0 risk重み |

### 魚眼外周 / cross5 分割

| オプション | 説明 |
|---|---|
| `--disable-fisheye-border-mask` | 魚眼外周マスク無効 |
| `--fisheye-mask-radius-ratio F` | 魚眼有効領域半径比 |
| `--fisheye-mask-center-offset-x N` | 魚眼中心Xオフセット |
| `--fisheye-mask-center-offset-y N` | 魚眼中心Yオフセット |
| `--split-views / --no-split-views` | cross5分割出力 ON/OFF |
| `--split-view-size N` | cross5出力サイズ |
| `--split-view-hfov F` | cross5 HFOV |
| `--split-view-vfov F` | cross5 VFOV |
| `--split-cross-yaw-deg F` | cross基準 yaw |
| `--split-cross-pitch-deg F` | cross基準 pitch |
| `--split-cross-inward-deg F` | cross内向き補正 |
| `--split-inward-up-deg F` | up内向き角 |
| `--split-inward-down-deg F` | down内向き角 |
| `--split-inward-left-deg F` | left内向き角 |
| `--split-inward-right-deg F` | right内向き角 |

### キャリブレーション / VO

| オプション | 説明 |
|---|---|
| `--calib-xml PATH` | 単眼/代表レンズXML |
| `--calib-model {auto,opencv,fisheye}` | キャリブモデル |
| `--front-calib-xml PATH` | front XML |
| `--rear-calib-xml PATH` | rear XML |
| `--calib-check` | キャリブ検証モード |
| `--calib-check-frame N` | 検証フレーム指定 |
| `--calib-check-out DIR` | 検証出力先 |
| `--disable-vo` | VO無効化 |
| `--vo-center-roi-ratio F` | VO中心ROI比率 |
| `--vo-max-features N` | VO最大特徴点 |
| `--vo-downscale-long-edge N` | VO入力縮小長辺 |
| `--vo-frame-subsample N` | VO計算間引き |
| `--vo-adaptive-roi / --no-vo-adaptive-roi` | VO ROI動的調整 |
| `--vo-fast-fail-inlier-ratio F` | VO早期失敗比率 |
| `--vo-step-proxy-clip-px F` | VO step_proxy上限 |
| `--vo-essential-method {auto,ransac,magsac}` | Essential推定法 |
| `--vo-subpixel-refine / --no-vo-subpixel-refine` | サブピクセル補正ON/OFF |
| `--vo-adaptive-subsample / --no-vo-adaptive-subsample` | VO動的サブサンプリングON/OFF |
| `--vo-subsample-min N` | 動的サブサンプル時の最小間引き |

### Rerun

| オプション | 説明 |
|---|---|
| `--rerun-stream` | Rerunストリーミング |
| `--rerun-spawn` | Viewer自動起動 |
| `--rerun-save PATH` | `.rrd`保存 |

## 出力ファイル

### 画像

- 単眼: `output/keyframe_XXXXXX.{ext}`
- ステレオ（OSV/Front-Rear）: `output/images/L|R/` or `F|R/`
- cross5有効時: `output/images/L_front/...` なども生成

### メタデータ

- `keyframe_metadata.json`
  - 入力情報、設定、キャリブ、キーフレーム一覧
  - `quality_filter` セクション
  - `quality_summary` セクション

### COLMAP互換（任意）

- `--colmap-format` 有効時:
  - `colmap/images/`
  - `colmap/cameras.txt`
  - `colmap/image_list.txt`

### 診断

- `frame_metrics.json`（Stage0/2/3の指標）
- `vo_diagnostics.json`
- `vo_trajectory.csv`
- `quality_metrics.json`（全Stage1サンプル）
- `quality_metrics.csv`（全Stage1サンプル）

`vo_trajectory.csv` は `vo_confidence` 列を含みます。  
`vo_diagnostics.json` は `vo_confidence_mean/p10/p50/p90` を含みます。

`quality_metrics.*` の各レコードには少なくとも次を含みます。

- `frame_index`, `timestamp`
- `quality`, `is_pass`, `drop_reason`
- 単眼: `raw_metrics`, `norm_metrics`
- ペア: `quality_lens_a`, `quality_lens_b`, `lens_a_*`, `lens_b_*`
- 品質フィルタ無効時: `legacy_quality_scores`

解析中間結果は `~/.360split/tmp_runs/<analysis_run_id>/` に `*.jsonl` で段階保存されます。
正常終了時は自動削除、失敗時は保持されます。
`--keep-temp` 指定時は正常終了時も保持し、`--resume --analysis-run-id <id>` で再利用できます。

## パイプライン概要

```text
Stage1: 品質フィルタ（A案）
  -> Stage0: 軽量走査（flow/ssim + VO補助）
      -> Stage2: 幾何+適応評価
          -> Stage3: 軌跡再評価・再スコア
              -> NMS/間隔補完 -> 出力
```

## 設定ファイル（JSON）

設定は `ConfigManager.default_config()` をベースに、JSONで上書きします。

品質フィルタ関連キー（追加）:

```json
{
  "quality_filter_enabled": true,
  "quality_threshold": 0.5,
  "quality_roi_mode": "circle",
  "quality_roi_ratio": 0.4,
  "quality_abs_laplacian_min": 35.0,
  "quality_use_orb": true,
  "quality_weight_sharpness": 0.4,
  "quality_weight_tenengrad": 0.3,
  "quality_weight_exposure": 0.15,
  "quality_weight_keypoints": 0.15,
  "quality_norm_p_low": 10.0,
  "quality_norm_p_high": 90.0,
  "quality_debug": false,
  "quality_tenengrad_scale": 1.0,
  "flow_downscale": 1.0,
  "resume_enabled": false,
  "keep_temp_on_success": false,
  "stage3_disable_traj_when_vo_unreliable": true,
  "stage3_vo_valid_ratio_threshold": 0.5,
  "vo_essential_method": "auto",
  "vo_subpixel_refine": true,
  "vo_adaptive_subsample": false,
  "vo_subsample_min": 1,
  "vo_confidence_low_threshold": 0.35,
  "vo_confidence_mid_threshold": 0.55
}
```

## プリセット

- `outdoor`（屋外高品質）
- `indoor`（屋内ロバスト追跡）
- `mixed`（混合環境適応）

```bash
python main.py --cli input.mp4 --preset outdoor
python main.py --cli input.mp4 --preset indoor
python main.py --cli input.mp4 --preset mixed
```

## Rerun可視化

```bash
# ライブストリーム
python main.py --cli input.mp4 --rerun-stream --rerun-spawn

# 保存
python main.py --cli input.mp4 --rerun-stream --rerun-save logs/run.rrd
```

オフライン再生:

```bash
python scripts/rerun_offline_replay.py \
  --input output/frame_metrics.json \
  --rrd logs/offline.rrd \
  --spawn
```

## プロジェクト構成（主要）

```text
360split/
├── main.py
├── config.py
├── core/
│   ├── keyframe_selector.py
│   ├── quality_evaluator.py
│   ├── quality_score.py
│   ├── geometric_evaluator.py
│   ├── adaptive_selector.py
│   ├── video_loader.py
│   └── config_loader.py
├── processing/
├── gui/
├── presets/
├── tests/
└── utils/
```

## ライセンス

本リポジトリ本体は `MIT` ライセンスです。詳細は [LICENSE](LICENSE) を参照してください。

利用ライブラリにはそれぞれ個別ライセンスが適用されます（例: PySide6, OpenCV, NumPy, PyTorch, ultralytics など）。
