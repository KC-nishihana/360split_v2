# OSV（Omnidirectional Stereo Video）機能テストガイド

## 概要

このガイドでは、360Split の OSV（ステレオビデオ）対応機能の動作確認手順を説明します。

---

## 前提条件

### 1. ffmpeg のインストール確認

OSV ファイルの処理には ffmpeg が必要です：

```bash
# ffmpeg がインストールされているか確認
which ffmpeg
# または
ffmpeg -version
```

**未インストールの場合：**

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロードしてパスを通す
```

### 2. OSV テストファイルの準備

OSV ファイルがない場合、2つの通常動画から擬似的に作成できます：

```bash
# 左右の動画を2ストリームのコンテナに結合
ffmpeg -i left.mp4 -i right.mp4 \
  -map 0:v -map 1:v -c copy test.osv
```

---

## テスト手順

### テスト 1: CLI モードでの基本動作確認

#### 1.1 OSV ファイルの読み込みと分離

```bash
python main.py --cli video.osv --output test_output/
```

**期待される出力：**
```
INFO - OSV（ステレオ）ファイルを検出しました
INFO - ステレオストリームを分離しました: L=temp_streams/video_left.mp4, R=temp_streams/video_right.mp4
INFO - ステレオモード（OSV）: 有効（L/Rペア出力）
```

#### 1.2 出力ファイルの確認

```bash
ls test_output/
```

**期待される結果：**
```
keyframe_000001_L.png
keyframe_000001_R.png
keyframe_000050_L.png
keyframe_000050_R.png
keyframe_000100_L.png
keyframe_000100_R.png
...
keyframe_metadata.json
```

#### 1.3 メタデータの確認

```bash
cat test_output/keyframe_metadata.json | grep -A 5 "keyframes"
```

**確認ポイント：**
- キーフレーム数が正しいか
- frame_index, timestamp, combined_score が記録されているか

---

### テスト 2: プリセット適用テスト

#### 2.1 屋外プリセット

```bash
python main.py --cli video.osv --output output_outdoor/ --preset outdoor
```

**期待される動作：**
- 高品質フィルタリング（厳格な閾値）
- 少ないキーフレーム数
- ログに「プリセット 'outdoor' を適用しました」と表示

#### 2.2 屋内プリセット

```bash
python main.py --cli video.osv --output output_indoor/ --preset indoor
```

**期待される動作：**
- 寛容なフィルタリング（緩い閾値）
- より多くのキーフレーム数
- 暗所・低特徴環境に対応

---

### テスト 3: 360度処理との組み合わせ

#### 3.1 Equirectangular 変換 + ステレオ

```bash
python main.py --cli video.osv \
  --output output_equirect/ \
  --equirectangular \
  --apply-mask
```

**期待される動作：**
- 左右両方のフレームに360度処理が適用される
- ナディアマスク（天底マスク）が適用される
- 出力: `keyframe_NNNNNN_L.png`, `keyframe_NNNNNN_R.png`

#### 3.2 Cubemap 出力 + ステレオ

```bash
python main.py --cli video.osv \
  --output output_cubemap/ \
  --equirectangular \
  --cubemap
```

**期待される出力：**
```
output_cubemap/
├── keyframe_000001_L.png
├── keyframe_000001_R.png
├── cubemap/
│   ├── frame_000001_L/
│   │   ├── front.png
│   │   ├── back.png
│   │   ├── left.png
│   │   ├── right.png
│   │   ├── top.png
│   │   └── bottom.png
│   └── frame_000001_R/
│       ├── front.png
│       ├── back.png
│       └── ...
```

---

### テスト 4: GUI モードでの動作確認

#### 4.1 ファイル選択ダイアログ

```bash
python main.py
```

1. メニューバー → ファイル → 開く
2. ファイルダイアログで `.osv` ファイルが選択可能か確認
3. OSV ファイルを選択

**期待される動作：**
- ステータスバーに「読み込み完了（OSV - ステレオ）」と表示
- ビデオプレーヤーに左目ストリームが表示される
- タイムラインが正しく初期化される

#### 4.2 ドラッグ＆ドロップ

1. OSV ファイルを GUI ウィンドウにドラッグ＆ドロップ

**期待される動作：**
- ファイルが正しく読み込まれる
- ステータスバーに「ステレオ」と表示

#### 4.3 キーフレーム解析

1. メニューバー → 解析 → フル解析 (Ctrl+R)
2. 解析完了まで待機

**期待される動作：**
- プログレスバーが正常に動作
- タイムラインにキーフレームマーカーが表示
- キーフレーム一覧に検出結果が表示

#### 4.4 エクスポート

1. メニューバー → ファイル → キーフレームをエクスポート
2. エクスポート設定ダイアログで設定
3. エクスポート実行

**期待される動作：**
- ステータスバーに「エクスポート: ステレオペア出力モード（L/R）」と表示
- 出力フォルダに `*_L.png` と `*_R.png` が生成される

---

### テスト 5: エラーハンドリング

#### 5.1 不正な OSV ファイル

```bash
# 通常の動画ファイルを .osv にリネーム
cp normal_video.mp4 fake.osv
python main.py --cli fake.osv
```

**期待される動作：**
- エラーメッセージが表示される
- アプリケーションがクラッシュしない

#### 5.2 ffmpeg が利用不可

```bash
# 一時的に PATH から ffmpeg を削除してテスト
PATH=/usr/bin python main.py --cli video.osv
```

**期待される動作：**
- 「ffmpeg が見つかりません」というエラーメッセージ
- 適切なエラーハンドリング

---

## パフォーマンステスト

### ストリーム分離時間の測定

```bash
time python main.py --cli video.osv --output perf_test/
```

**ベンチマーク目安：**
- 4K 30fps 1分動画: 初回分離 10-30秒、2回目以降 < 1秒（キャッシュ利用）
- 8K 30fps 1分動画: 初回分離 30-60秒

### メモリ使用量の確認

```bash
# Linux/macOS
/usr/bin/time -v python main.py --cli video.osv --output mem_test/

# Python スクリプトでメモリプロファイリング
python -m memory_profiler main.py --cli video.osv
```

---

## トラブルシューティング

### 問題 1: "ffmpeg が見つかりません"

**解決策：**
```bash
# ffmpeg をインストール
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu

# パスを確認
which ffmpeg
```

### 問題 2: "ステレオストリームを開けません"

**原因：**
- OSV ファイルが破損している
- ストリームが2つ未満

**確認方法：**
```bash
# ストリーム情報を確認
ffprobe video.osv
```

### 問題 3: 左右のフレーム数が異なる

**原因：**
- OSV ファイルの左右ストリームが同期していない

**解決策：**
```bash
# 手動で同期確認
ffprobe -select_streams v:0 -count_frames video.osv
ffprobe -select_streams v:1 -count_frames video.osv
```

### 問題 4: メモリ不足エラー

**解決策：**
```bash
# サンプリング間隔を広げる
python main.py --cli video.osv --min-interval 10
```

---

## 3DGS/COLMAP との連携確認

### COLMAP での読み込みテスト

```bash
# 出力画像を COLMAP に渡す
colmap feature_extractor \
  --database_path database.db \
  --image_path test_output/

# ステレオペアが正しく認識されるか確認
colmap database_view \
  --database_path database.db
```

**期待される動作：**
- `*_L.png` と `*_R.png` が別々の画像として認識される
- ファイル名の `_L` / `_R` サフィックスが保持される

---

## チェックリスト

- [ ] ffmpeg がインストールされている
- [ ] CLI モードで OSV ファイルが読み込める
- [ ] ステレオペア（_L, _R）が正しく出力される
- [ ] GUI モードで .osv ファイルが選択できる
- [ ] GUI モードでドラッグ＆ドロップが機能する
- [ ] GUI モードでエクスポートが正常動作する
- [ ] プリセット（outdoor/indoor/mixed）が適用される
- [ ] 360度処理（Equirectangular, Cubemap）がステレオで動作する
- [ ] エラーハンドリングが適切に機能する
- [ ] メタデータが正しく出力される
- [ ] COLMAP で画像が読み込める

---

## サポート

問題が発生した場合は、以下の情報を含めて報告してください：

1. 動作環境（OS, Python バージョン）
2. ffmpeg バージョン（`ffmpeg -version`）
3. OSV ファイルの情報（`ffprobe video.osv`）
4. エラーログ（`logs/360split.log`）
5. 実行コマンド

---

**作成日**: 2026-02-12
**バージョン**: 1.0.0
