# 360split\_v2 × COLMAP / VO 統合設計書

## 1. 目的

本設計の目的は以下の通り。

- 魚眼カメラのキャリブレーション情報を活用し、 高精度な自己位置推定（オフライン）を実現する
- COLMAP（SfM）を姿勢推定バックエンドとして統合する
- 従来のVO（Visual Odometry）とCOLMAPをCLIで切替可能にする
- 推定結果から「必要な画像のみ」を抽出できる仕組みを構築する

リアルタイム性は要求しない。 精度優先・バッチ処理前提とする。

---

# 2. 全体アーキテクチャ

```
           ┌────────────────────┐
           │ 360split_v2        │
           │  フレーム抽出     │
           │  Stage0/3品質評価 │
           └────────┬─────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Pose Backend Selector    │
        │                          │
        │ 1) VO (OpenCV)           │
        │ 2) COLMAP (SfM)          │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ Pose Trajectory          │
        │ (統一フォーマット)        │
        └────────┬─────────────────┘
                 │
                 ▼
        必要画像抽出ロジック
```

---

# 3. CLI／GUI 設計

## 3.1 CLI（既存拡張）

```
--pose-backend {vo,colmap}

--colmap-path <path>
--colmap-db-path <path>
--colmap-workspace <path>

--calib-xml <path>

--pose-export-format {internal,metashape}
```

デフォルト:

```
--pose-backend vo
```

## 3.2 GUI（改良要件）

### 3.2.1 目的

- CLI同等の機能（VO/COLMAP切替、キャリブ指定、出力設定）をGUIから操作可能にする
- 実行ログ／進捗／失敗理由をGUIで可視化し、再実行しやすくする
- 解析結果（軌跡・選別画像・統計）をGUIで確認できるようにする

### 3.2.2 UI追加項目（画面／パネル）

#### A. 入力パネル（Input）

- 動画／画像ディレクトリ選択（既存に合わせる）
- キャリブXMLの選択
  - 単一（`--calib-xml`）
  - 前後分離（`--front-calib-xml` / `--rear-calib-xml` 相当）※将来拡張
- 画像サイズ／縮小率（VO／COLMAPで共通に扱える設定のみ）

#### B. Pose Backend パネル（Pose）

- バックエンド選択：`VO` / `COLMAP`

- 共通オプション

  - 出力フォーマット：Internal / Metashape
  - 必要画像抽出：ON/OFF

- VO専用オプション（既存UI/CLIに合わせて露出）

  - 特徴点数、RANSAC/MAGSAC等、confidence閾値など

- COLMAP専用オプション

  - COLMAP実行方式：CLI / PyCOLMAP（将来）
  - colmap実行パス（自動検出＋手動指定）
  - workspace（作業フォルダ）
  - database.db の再利用／クリア
  - SIFT/Mapperプリセット（後述）

#### C. 実行・進捗パネル（Run）

- 実行ボタン（Pose推定→画像抽出→エクスポートをパイプライン実行）
- ステータス表示
  - キュー／実行中／完了／失敗
- 進捗
  - feature\_extractor / matcher / mapper のステップ進捗
- ログビュー（stdout/stderr）
  - COLMAP stderr をそのまま表示
  - 失敗時に「最後のエラー行」をハイライト

#### D. 結果パネル（Results）

- 軌跡プレビュー（簡易3D：matplotlib/pyqtgraph いずれか）
  - XYZの折れ線
  - confidenceの色分け（任意）
- 統計
  - 画像数（入力／採用／除外）
  - 平均/中央値Δt、Δθ
  - COLMAP再構成の観測点数（可能なら）
- 出力フォルダを開くボタン
  - `vo_trajectory.csv` / `metashape_import.csv`
  - 抽出済み画像フォルダ

### 3.2.3 GUI内部設計（実装方針）

- GUIは「パイプライン」を直接実装せず、CLIと同じコア処理（Python関数）を呼ぶ
  - 例：`run_pipeline(config)`
- 進捗とログは Workerスレッドで受ける
  - COLMAPは `subprocess.Popen` で逐次出力を読み取りGUIに流す
- 設定は `config.json` として保存／読み込み
  - 再現性とバグ報告に有効

### 3.2.4 GUIの失敗ハンドリング

- COLMAP未インストール／パス不正：
  - GUIで即時検出し「インストール手順へリンク」表示
- database/db破損やworkspace混在：
  - 「DBクリア」「workspace初期化」ボタンを用意
- 解析に失敗した場合：
  - 直近ログ＋推奨アクション（例：画像枚数減らす、閾値変更）を提示

---

# 4. PoseEstimator 抽象クラス

```python
class PoseEstimator:
    def estimate(self, image_dir: str) -> list:
        """
        Returns:
            List[Pose]
        """
        raise NotImplementedError


class Pose:
    filename: str
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    confidence: float
```

---

# 5. VO Backend

既存のVO実装をラップする。

```
class VOPoseEstimator(PoseEstimator):
    def estimate(self, image_dir):
        # 既存VO処理を呼び出す
        # vo_trajectory.csv を読み込む
        # Poseリストへ変換
```

---

# 6. COLMAP Backend

## 6.1 前提

- cam1.xml を OPENCV モデルとして使用
- カメラ固定（intrinsics固定）
- バンドル調整のみ実施

## 6.2 実行手順（CLI方式）

1. feature\_extractor
2. exhaustive\_matcher
3. mapper

## 6.3 Python実装例

```python
import subprocess

class COLMAPPoseEstimator(PoseEstimator):
    def estimate(self, image_dir):
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", "database.db",
            "--image_path", image_dir
        ])

        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", "database.db"
        ])

        subprocess.run([
            "colmap", "mapper",
            "--database_path", "database.db",
            "--image_path", image_dir,
            "--output_path", "sparse"
        ])

        return self._parse_images_txt("sparse/0/images.txt")
```

---

# 7. images.txt パース

```
IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
```

```python
import numpy as np


def parse_images_txt(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]

            poses.append(Pose(name, qw, qx, qy, qz, tx, ty, tz, 1.0))

    return poses
```

---

# 8. 必要画像抽出ロジック

### 8.1 並進ベース間引き

```
Δt > threshold
```

### 8.2 回転ベース間引き

```
Δθ > threshold
```

### 8.3 観測点数ベース

3D点観測数が少ない画像は削除

---

# 9. Metashape用エクスポート

```
filename,x,y,z,yaw,pitch,roll
```

クォータニオン → オイラー角変換実装を追加する。

---

# 10. 将来改良案

- fisheye(OPENCV\_FISHEYE)正式対応
- hloc連携によるローカライゼーション
- GCP/RTKによるスケール固定
- GPU SIFT利用

---

# 11. 生成AI用 開発プロンプト

以下をAIに渡す：

""" 360split\_v2 に PoseEstimator 抽象クラスを導入し、 --pose-backend {vo,colmap} で切替可能にしてください。

COLMAP backend は CLI 実行方式とし、 images.txt を解析して Pose 配列へ変換してください。

Pose は filename, qw,qx,qy,qz, tx,ty,tz, confidence を保持する。

既存VOコードは変更せずラップしてください。

必要画像抽出ロジックを

- 並進
- 回転
- 観測点数 で選択可能にしてください。

Metashape用CSV出力も追加してください。 """

