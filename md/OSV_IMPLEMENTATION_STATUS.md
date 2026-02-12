# OSV（Omnidirectional Stereo Video）対応 実装状況レポート

## 概要

360Split に OSV ファイル（左右ステレオストリーム）対応を実装しました。
本レポートは実装済み部分と残りのタスクを整理します。

---

## ✅ 実装完了部分

### 1. DualVideoLoader クラス（core/video_loader.py）

**機能**:
- ffmpeg で `.osv` ファイルを `left_eye.mp4` / `right_eye.mp4` に分離
- 左右フレームの同期読み込み
- VideoLoader 互換インターフェース

**主要メソッド**:
```python
class DualVideoLoader:
    def load(osv_path: str) -> VideoMetadata
        # OSV ファイルを読み込み、左右ストリームに分離

    def get_frame_pair(index: int) -> Tuple[np.ndarray, np.ndarray]
        # 左右のフレームペアを同期して取得

    def get_frame(index: int) -> np.ndarray
        # 左目フレームのみ取得（VideoLoader 互換）

    @property
    def is_stereo() -> bool
        # ステレオ判定フラグ
```

**キャッシュ機能**:
- L/R 個別の LRU フレームバッファ（各100フレーム）
- 分離済みストリームの再利用（2回目以降は高速）

**使用例**:
```python
from core.video_loader import DualVideoLoader

loader = DualVideoLoader(temp_dir="temp_streams")
metadata = loader.load("video.osv")  # 自動分離

# 左右ペアを取得
frame_l, frame_r = loader.get_frame_pair(100)

# 左のみ取得（VideoLoader互換）
frame = loader.get_frame(100)
```

---

### 2. KeyframeSelector ステレオ対応（core/keyframe_selector.py）

**実装内容**:
- `_compute_quality_score_stereo()` メソッド追加
- `select_keyframes()` にステレオ検出機能追加

**品質評価ロジック**:
```python
def _compute_quality_score_stereo(frame_l, frame_r) -> Dict[str, float]:
    """
    Conservative: L/R 両方が基準を満たす場合のみ採用（AND条件）

    - sharpness: min(L, R)
    - exposure: min(L, R)
    - motion_blur: max(L, R)  # ブラーは大きい方が悪い
    """
```

**判定方針**:
- **品質チェック**: L/R 両方が基準を満たすかチェック
- **移動判定**: Left 画像のみで計算（コスト削減）
  - カメラリグは剛体なので、Lが動けばRも動く

**使用例**:
```python
from core.keyframe_selector import KeyframeSelector

selector = KeyframeSelector()
keyframes = selector.select_keyframes(loader)  # 自動的にステレオ検出
```

---

## 🔄 部分実装（要完成）

### 3. ExportWorker ペア出力（gui/workers.py）

**現状**:
- 基本構造のみ実装済み
- 完全なペア出力ロジックは未実装

**必要な実装**:
```python
class ExportWorker(QThread):
    def __init__(self, ...,
                 is_stereo=False,
                 stereo_left_path=None,
                 stereo_right_path=None):
        # ステレオフラグとパスを保存

    def run(self):
        if self.is_stereo:
            # 左右両方のキャプチャを開く
            cap_l = cv2.VideoCapture(self.stereo_left_path)
            cap_r = cv2.VideoCapture(self.stereo_right_path)

            for frame_idx in self.frame_indices:
                # 左右を同期読み込み
                cap_l.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_l, frame_l = cap_l.read()
                ret_r, frame_r = cap_r.read()

                # 処理を適用（360度処理、マスク処理）
                processed_l = self._apply_processing(frame_l)
                processed_r = self._apply_processing(frame_r)

                # ペアで保存
                filename_l = f"{self.prefix}_{frame_idx:06d}_L.{ext}"
                filename_r = f"{self.prefix}_{frame_idx:06d}_R.{ext}"
                cv2.imwrite(output_path / filename_l, processed_l)
                cv2.imwrite(output_path / filename_r, processed_r)
        else:
            # 既存の単眼ロジック
```

**出力形式**:
```
output/
├── keyframe_000001_L.jpg
├── keyframe_000001_R.jpg
├── keyframe_000050_L.jpg
├── keyframe_000050_R.jpg
└── ...
```

---

## ⏳ 未実装部分

### 4. GUI .osv 対応（gui/main_window.py）

**必要な変更**:

#### 4.1 ファイル選択ダイアログ
```python
def open_video(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self,
        "ビデオファイルを開く",
        "",
        "Video Files (*.mp4 *.mov *.avi *.mkv *.osv);;All Files (*)"
        #                                         ^^^^^^ 追加
    )

    if file_path:
        self._load_video(file_path)

def _load_video(self, file_path):
    # OSV 判定
    if file_path.lower().endswith('.osv'):
        from core.video_loader import DualVideoLoader
        self.video_loader = DualVideoLoader()
        self.is_stereo = True
    else:
        from core.video_loader import VideoLoader
        self.video_loader = VideoLoader()
        self.is_stereo = False

    metadata = self.video_loader.load(file_path)
    # ... 既存の処理
```

#### 4.2 エクスポート処理
```python
def export_keyframes(self):
    # ... 既存の設定読み込み

    if hasattr(self.video_loader, 'is_stereo') and self.video_loader.is_stereo:
        # ステレオの場合
        self._export_worker = ExportWorker(
            self.video_path, selected, export_dir,
            is_stereo=True,
            stereo_left_path=self.video_loader.left_path,
            stereo_right_path=self.video_loader.right_path,
            # ... その他のパラメータ
        )
    else:
        # 通常の処理
```

---

### 5. CLI .osv 対応（main.py）

**必要な変更**:

```python
def run_cli(args):
    from core.video_loader import VideoLoader, DualVideoLoader

    video_path = args.cli

    # OSV 判定
    if video_path.lower().endswith('.osv'):
        loader = DualVideoLoader()
        is_stereo = True
        logger.info("OSV モードで実行")
    else:
        loader = VideoLoader()
        is_stereo = False

    metadata = loader.load(video_path)

    # キーフレーム選択
    selector = KeyframeSelector(config)
    keyframes = selector.select_keyframes(loader)  # 自動的にステレオ検出

    # エクスポート
    if is_stereo:
        # ペア出力
        for kf in keyframes:
            frame_l, frame_r = loader.get_frame_pair(kf.frame_index)

            # 処理を適用
            processed_l = apply_processing(frame_l, config)
            processed_r = apply_processing(frame_r, config)

            # 保存
            filename_l = f"keyframe_{kf.frame_index:06d}_L.{fmt}"
            filename_r = f"keyframe_{kf.frame_index:06d}_R.{fmt}"
            cv2.imwrite(str(output_dir / filename_l), processed_l)
            cv2.imwrite(str(output_dir / filename_r), processed_r)
    else:
        # 既存の単眼ロジック
```

---

## 📋 実装チェックリスト

### ✅ 完了
- [x] DualVideoLoader クラス実装
- [x] ffmpeg ストリーム分離機能
- [x] 左右フレーム同期読み込み
- [x] LRU キャッシュ（L/R個別）
- [x] KeyframeSelector ステレオ品質評価
- [x] select_keyframes() ステレオ検出
- [x] ExportWorker ペア出力ロジック完成
  - [x] 基本構造
  - [x] 左右キャプチャ同期
  - [x] ペアファイル命名（_L, _R）
  - [x] 処理適用（360度、マスク）
  - [x] キャプチャクリーンアップ修正
- [x] GUI .osv ファイル対応
  - [x] ファイル選択ダイアログに .osv 追加
  - [x] ドラッグ＆ドロップで .osv 対応
  - [x] _load_video() に DualVideoLoader 統合
  - [x] ステレオ状態管理（is_stereo, stereo_left_path, stereo_right_path）
  - [x] export_keyframes() ステレオ対応
- [x] CLI .osv 対応
  - [x] main.py に DualVideoLoader 統合
  - [x] run_cli() ステレオ判定
  - [x] ペア出力ロジック（_L, _R サフィックス）

### オプション（未実装）
- [ ] ビデオプレーヤーでステレオ表示（左右並列表示）
- [ ] リアルタイムステレオプレビュー

---

## 🔧 動作確認手順（実装完了後）

### 1. CLI での動作確認

```bash
# OSV ファイルでキーフレーム抽出
python main.py --cli video.osv --output output/

# 出力確認
ls output/
# → keyframe_000001_L.jpg
# → keyframe_000001_R.jpg
# → keyframe_000050_L.jpg
# → keyframe_000050_R.jpg
```

### 2. GUI での動作確認

1. アプリ起動
2. ファイル → 開く → `video.osv` を選択
3. キーフレーム抽出実行
4. エクスポート
5. 出力フォルダ確認

### 3. ffmpeg インストール確認

```bash
# ffmpeg が必要
which ffmpeg
# または
ffmpeg -version

# インストール（必要に応じて）
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

---

## 📝 実装優先度

### 高優先度（必須）
1. **ExportWorker ペア出力完成** - ステレオエクスポートの核心
2. **CLI .osv 対応** - 基本的な動作確認に必要

### 中優先度（推奨）
3. **GUI .osv 対応** - ユーザビリティ向上

### 低優先度（オプション）
4. ビデオプレーヤーでのステレオ表示
5. リアルタイムステレオプレビュー

---

## 🚀 次のステップ

### 即座に実装可能
- ExportWorker の `run()` メソッドにステレオペア出力ロジック追加
- main.py に `.osv` 判定とペア出力追加

### 中期的に実装
- GUI の完全統合
- ステレオプレビュー機能

---

## 📚 参考資料

### OSV ファイル形式
- OSV は通常、2つのビデオストリームを含むコンテナ
- ストリーム0: Left Eye
- ストリーム1: Right Eye
- ffmpeg で簡単に分離可能

### 3DGS/COLMAP との連携
```
output/
├── images/
│   ├── keyframe_000001_L.jpg
│   ├── keyframe_000001_R.jpg
│   ├── keyframe_000050_L.jpg
│   └── keyframe_000050_R.jpg
└── sparse/
    └── cameras.txt  # ステレオカメラパラメータ
```

ファイル名に `_L`, `_R` を付与することで、後段のツールがステレオペアとして認識可能。

---

**実装日**: 2026-02-12
**実装者**: Claude Sonnet 4.5
**ステータス**: ✅ **実装完了**（コア機能、GUI、CLI すべて統合完了）

---

## ✅ 実装完了サマリ

### 実装された機能

1. **DualVideoLoader（core/video_loader.py）**
   - ffmpeg によるステレオストリーム分離
   - 左右フレームの同期読み込み
   - VideoLoader 互換インターフェース
   - LRU キャッシュ（L/R 個別）

2. **KeyframeSelector ステレオ対応（core/keyframe_selector.py）**
   - Conservative 品質評価（L/R 両方が基準を満たす必要がある）
   - Left-only 移動判定（コスト削減）
   - 自動ステレオ検出

3. **ExportWorker ステレオペア出力（gui/workers.py）**
   - ステレオキャプチャの同期読み込み
   - _L / _R サフィックス付きファイル出力
   - 360度処理とマスク処理対応
   - 適切なキャプチャクリーンアップ

4. **GUI .osv 対応（gui/main_window.py）**
   - ファイル選択ダイアログで .osv 対応
   - ドラッグ＆ドロップで .osv 対応
   - DualVideoLoader 自動切り替え
   - ステレオ状態管理
   - エクスポート時のステレオパス受け渡し

5. **CLI .osv 対応（main.py）**
   - .osv ファイル自動検出
   - DualVideoLoader 使用
   - ステレオペア出力（_L, _R）
   - 進捗表示対応

### 使用方法

#### GUI モード
```bash
python main.py
# ファイル → 開く → video.osv を選択
# または video.osv をドラッグ＆ドロップ
# キーフレーム抽出 → エクスポート
```

#### CLI モード
```bash
python main.py --cli video.osv --output output/
# → output/keyframe_000001_L.png
# → output/keyframe_000001_R.png
# → output/keyframe_000050_L.png
# → output/keyframe_000050_R.png
```

### 出力形式
```
output/
├── keyframe_000001_L.jpg  # 左目
├── keyframe_000001_R.jpg  # 右目
├── keyframe_000050_L.jpg
├── keyframe_000050_R.jpg
└── ...
```

この命名規則により、COLMAP や 3DGS などの後段ツールがステレオペアとして認識可能です。
