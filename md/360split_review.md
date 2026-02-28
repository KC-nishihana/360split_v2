# 360Split レビュー＆改良提案書

**作成日**: 2026-02-28  
**対象**: 360Split v1.1.0+ (Unreleased含む)  
**レビュー範囲**: アーキテクチャ、パフォーマンス、堅牢性、コード品質、機能拡張

---

## 1. 総合評価

360Splitは、360度動画から3DGS/フォトグラメトリ向けキーフレームを抽出する高度なツールとして、十分に機能する成熟したシステムです。4段階パイプライン（Stage1→0→2→3）、クロスプラットフォームGPU対応、VO統合、GUI/CLI両対応など、包括的な設計が行われています。

**強み**:
- 明確に分離されたパイプラインステージ設計
- Apple Silicon (MPS) / CUDA 双方のGPU自動検出とフォールバック
- 品質フィルタのA案（ROI + 分位点正規化）が実戦的
- StageTempStore による中間結果保存で障害復旧が可能
- 34テスト全パス、包括的な監査レポートが存在

**改善余地のある領域**: 以下に詳述します。

---

## 2. 潜在バグ・リスク（安全性修正 — 最優先）

### 2.1 Stage2パイプライン並列モードの無効なプロデューサー・コンシューマー

`core/pipeline/stage2_evaluator.py` の `ENABLE_STAGE2_PIPELINE_PARALLEL` 有効時、プロデューサーが `ThreadPoolExecutor` でサブミットされますが、コンシューマーがメインスレッドで `queue.get()` してから **同じリストを再代入** するだけで、実質的に並列化されていません。

```python
# 現在のコード（問題箇所）
queue: Queue = Queue(maxsize=32)
def _producer() -> None:
    for item in stage1_candidates:
        queue.put(item)
    queue.put(None)

ordered_candidates: List[Dict] = []
with ThreadPoolExecutor(max_workers=1) as ex:
    ex.submit(_producer)
    while True:
        item = queue.get()
        if item is None:
            break
        ordered_candidates.append(item)
stage1_candidates = ordered_candidates  # 元のリストと同一内容
```

**問題**: デッドロックリスクはないが、不要なオーバーヘッド（Queue + Thread生成）が毎回発生します。`maxsize=32` でプロデューサーがブロックする可能性もあります。

**提案**: この並列スケルトンは将来のフレームプリフェッチ用の足場と推察しますが、現時点では `stage1_candidates` をそのまま渡すべきです。もし将来フレームデコードの先読みを行うなら、ビデオローダー層でプリフェッチする方が適切です。

### 2.2 `_enforce_max_interval` のギャップ補完における例外安全性

監査レポートでは「ギャップ補完挿入を実装済み」とありますが、source候補が存在しない区間で `VideoLoader.get_frame()` を呼んだ場合の `None` チェックが重要です。VO無効時にStage3がスキップされると、この関数に渡される候補リストが空になるケースがあり得ます。

### 2.3 品質フィルタ無効時のレコード構造不整合

`quality_filter_enabled=False` 時、`quality_metrics.*` には `legacy_quality_scores` フィールドが含まれますが、Stage2の `quality_scores` との統合時にキー名の不一致が起きる可能性があります。特に `quality` キーが存在しないため、NMS/Stage3で `KeyframeInfo.quality_scores.get("quality", 0.0)` が常に0.0を返すリスクがあります。

---

## 3. パフォーマンス改良提案

### 3.1 Sobel+ヒストグラム計算のダウンスケール（推定4-8倍高速化）

**現在**: `quality_score.py` の `compute_raw_metrics` 内でSobel/Tenengrad計算がフル解像度（HD: 1920x960等）で実行されています。Stage1で `eval_scale=0.5` のダウンスケールが適用されていますが、Sobel計算自体は縮小後画像でも31ms/frame程度かかっています。

**提案**: Sobel/Tenengrad計算をさらに0.25xにダウンスケールする専用パスを追加。

```python
def compute_raw_metrics(frame, roi_spec, ..., tenengrad_scale=0.25):
    # ROIマスク適用後、tenengrad計算のみ追加縮小
    if tenengrad_scale < 1.0:
        small = cv2.resize(gray, None, fx=tenengrad_scale, fy=tenengrad_scale)
        sobel_x = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
        ...
```

**期待効果**: 31ms → 4-8ms/frame。Stage1全体で396フレーム処理時に約9秒短縮。

### 3.2 Lucas-Kanade疎フローのダウンスケール（推定3-5倍高速化）

**現在**: Stage0の `compute_optical_flow_magnitude` が90ms/frame（HD解像度）。

**提案**: フロー計算入力を0.5xまたは0.25xにダウンスケール。特徴点座標を元のスケールに逆変換する必要はなく、平均フロー大きさ（magnitude）はスケール補正で対応可能です。

```python
def compute_optical_flow_magnitude(self, frame1, frame2, downscale=0.5):
    scale = downscale
    small1 = cv2.resize(gray1, None, fx=scale, fy=scale)
    small2 = cv2.resize(gray2, None, fx=scale, fy=scale)
    pts = cv2.goodFeaturesToTrack(small1, ...)
    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(small1, small2, pts, ...)
    # magnitude をスケール補正
    return mean_magnitude / scale
```

**期待効果**: 90ms → 20-30ms/frame。Stage0全体（396フレーム）で約25秒短縮。

### 3.3 Stage1バッチ処理の粒度最適化

**現在**: `stage1_metrics_batch_size` でバッチ品質計算が可能ですが、ORB検出がバッチ化されていません。

**提案**: ORB検出をバッチから除外するオプション `quality_use_orb_stage1=False` を追加。Stage1は「明らかに悪いフレームを落とす」目的なので、ORBの0.15ウェイトは省略しても選別精度への影響は小さいです。Stage2で再計算する際に正確なORBスコアを使用すれば十分です。

**期待効果**: Stage1処理時間の約15%削減。

### 3.4 Stage0/Stage2間のフレーム再デコード回避

**現在**: Stage0でデコード済みのフレームがStage2で再デコードされています。

**提案**: Stage0で評価した最新フレームを軽量キャッシュ（LRU 2-4枚）で保持し、Stage2の初回参照時にキャッシュヒットさせる。ビデオローダー層の `prefetch_size` 拡張で実現可能です。

---

## 4. 堅牢性・エラーハンドリング改善

### 4.1 VO推定失敗時のグレースフルデグラデーション強化

**現在**: VO推定失敗（`vo_valid=False`）時、Stage3の軌跡一貫性スコアが0.0になりますが、再スコア式 `w_base*stage2 + w_traj*0.0 - w_risk*risk` で `w_traj` の重みが無駄になります。

**提案**: VO失敗率が50%を超える場合、自動的に `w_traj=0.0, w_base=1.0` に切り替えるフォールバック。

```python
vo_success_rate = sum(1 for m in stage0_metrics.values() if m.get("vo_valid")) / len(stage0_metrics)
if vo_success_rate < 0.5:
    logger.warning(f"VO成功率が低い({vo_success_rate:.0%})。軌跡重みを無効化します。")
    w_traj = 0.0
    w_base = w_base + original_w_traj
```

### 4.2 メモリ使用量の監視と制限

**現在**: `frame_window`（Stage2のSSIM/フロー計算用ウィンドウ）のサイズ制限がconfig依存で、大量フレームの動画では予想外のメモリ消費が発生する可能性があります。

**提案**: `psutil` でメモリ使用率を定期監視し、80%超過時にウィンドウサイズを動的に縮小。

### 4.3 動画デコードエラーのリトライ機構

**現在**: `get_frame()` が `None` を返した場合、そのフレームはスキップされますが、一時的なデコードエラー（I/O負荷等）と永続的な破損を区別していません。

**提案**: `None` 返却時に1回リトライし、それでも失敗した場合のみスキップ。リトライはsleepなしの即時再試行で十分です。

---

## 5. コード品質・設計改善

### 5.1 `keyframe_selector.py` の責務分離

**現在**: `KeyframeSelector` クラスが約2000行以上あり、以下の全責務を担っています：
- Stage0/1/2/3の実行オーケストレーション
- フレームデコード・ペア管理
- 品質/幾何/適応評価の呼び出し
- NMS・間隔制約
- エクスポート
- 動体マスク生成
- VO連携

**提案**: 段階的にリファクタリング：

1. **Stage0Executor, Stage2Executor** クラスを作成し、各ステージのロジックを移動（Stage1/3は既に `pipeline/` に薄いラッパーがある）
2. **ExportManager** クラスを分離（エクスポートロジックは評価と無関係）
3. **DynamicMaskManager** クラスを分離

これにより単体テストが容易になり、各ステージの独立した最適化が可能になります。

### 5.2 設定キーの命名規則統一

**現在**: 設定キーに `snake_case`（`quality_filter_enabled`）と `SCREAMING_CASE`（`LAPLACIAN_THRESHOLD`）が混在しています。`normalize_config_dict()` で変換していますが、新しいキー追加時にエイリアスの追加漏れが起きやすいです。

**提案**: 新規キーは全て `snake_case` に統一し、レガシーキーのエイリアスマップ（`SELECTOR_ALIAS_MAP`）を非推奨警告付きで維持。

### 5.3 型ヒントの強化

**現在**: 多くの関数で `Dict[str, Any]` が使われており、dict内のキー・値型が不明確です。

**提案**: `TypedDict` または `dataclass` を使用：

```python
from typing import TypedDict

class Stage0Metrics(TypedDict):
    motion_risk: float
    flow_mag_light: float
    ssim_light: float
    vo_valid: bool
    vo_status_reason: str
    vo_confidence: float
```

---

## 6. 機能拡張提案

### 6.1 【高優先】自動キャリブレーション推定

**現状の課題**: VO機能はキャリブレーションファイル必須で、ファイルがない場合はVO無効化されます。多くのユーザーは内部パラメータを持っていません。

**提案**: 動画メタデータ（EXIF, 解像度）と一般的なカメラモデルのデータベースから、大まかなキャリブレーションを自動推定。

```python
def estimate_calibration_from_video(metadata: VideoMetadata) -> Optional[CalibrationParams]:
    # 1. EXIF focal length があれば使用
    # 2. 360カメラの既知モデル (Insta360, Ricoh Theta, etc.) マッチング
    # 3. フォールバック: 画角推定（FOV=180度と仮定して焦点距離計算）
    focal_estimate = max(metadata.width, metadata.height) / (2 * math.tan(math.radians(90)))
    ...
```

### 6.2 【高優先】バッチ処理の進捗永続化

**現状の課題**: 長時間処理（4K動画のStage1 = 33分）が中断された場合、最初からやり直しになります。

**提案**: `StageTempStore` を活用して、完了済みステージの結果を自動保存・再利用する機能を`--resume` CLIオプションとして公開。

```bash
# 最初の実行（途中で中断）
python main.py --cli input.mp4 -o output --run-id my_run

# 再開（Stage1完了済みならStage0から再開）
python main.py --cli input.mp4 -o output --resume my_run
```

現在 `StageTempStore` は `select_keyframes` 内部で使われていますが、CLIからの明示的なresume制御がありません。

### 6.3 【中優先】品質スコアの可視化HTMLレポート

**現状の課題**: `quality_metrics.json` は詳細ですが、可読性が低いです。

**提案**: 解析完了後に自動生成されるHTML/SVGレポート。Plotlyやmatplotlibの静的画像を含み、以下を可視化：
- フレームごとの品質スコア推移グラフ
- 選択されたキーフレームのサムネイル一覧
- Stage0のmotion_risk heatmap
- VO軌跡のtop-down view

### 6.4 【中優先】COLMAP互換出力形式

**現状の課題**: 出力はキーフレーム画像 + `keyframe_metadata.json` ですが、COLMAPへの直接入力にはユーザー側の変換作業が必要。

**提案**: `--colmap-format` オプションで、COLMAP準拠のディレクトリ構造を出力。

```
output/
├── images/
│   ├── frame_000001.jpg
│   └── frame_000050.jpg
├── cameras.txt (intrinsics from calibration)
└── image_list.txt
```

### 6.5 【低優先】A/Bテスト用の比較モード

**提案**: 2つの設定で同じ動画を処理し、キーフレーム選択結果を比較するユーティリティ。

```bash
python scripts/compare.py --video input.mp4 --config-a outdoor.json --config-b indoor.json
```

---

## 7. テスト強化提案

### 7.1 エッジケーステスト追加

現在34テストはパスしていますが、以下のエッジケースのテストが不足しています：

1. **極短動画（10フレーム未満）**: Stage0/Stage2で十分な候補が生成されないケース
2. **全フレーム低品質**: Stage1で全候補が落ちた場合のレスキューモード動作
3. **VO推定が全失敗**: キャリブレーション不正時のStage3フォールバック
4. **巨大解像度（8K）**: メモリ制限下での動作
5. **FPS不整合**: metadata.fpsが実際のフレームレートと異なる場合

### 7.2 回帰テストの自動化

**提案**: ベンチマーク用の小さなテスト動画（5秒程度）をリポジトリに含め、CI上でStage1-3の出力フレーム数・スコア分布が期待範囲内であることを検証。

---

## 8. 優先度マトリクス

| # | 改善項目 | 種別 | 難易度 | 効果 | 優先度 |
|---|---------|------|--------|------|--------|
| 1 | Stage2並列モードの修正 (2.1) | バグ修正 | 低 | 中 | ★★★★★ |
| 2 | 品質フィルタ無効時のレコード不整合 (2.3) | バグ修正 | 低 | 中 | ★★★★★ |
| 3 | Sobel/Tenengradダウンスケール (3.1) | 性能 | 低 | 高 | ★★★★★ |
| 4 | LKフローダウンスケール (3.2) | 性能 | 低 | 高 | ★★★★★ |
| 5 | VO失敗時の重み自動調整 (4.1) | 堅牢性 | 低 | 中 | ★★★★ |
| 6 | バッチ処理のresume対応 (6.2) | 機能 | 中 | 高 | ★★★★ |
| 7 | エッジケーステスト追加 (7.1) | テスト | 中 | 中 | ★★★★ |
| 8 | Stage1 ORBスキップオプション (3.3) | 性能 | 低 | 低〜中 | ★★★ |
| 9 | KeyframeSelector責務分離 (5.1) | 設計 | 高 | 高 | ★★★ |
| 10 | 自動キャリブレーション推定 (6.1) | 機能 | 中 | 高 | ★★★ |
| 11 | COLMAP互換出力 (6.4) | 機能 | 低 | 中 | ★★★ |
| 12 | 品質スコアHTMLレポート (6.3) | 機能 | 中 | 中 | ★★☆ |
| 13 | TypedDict導入 (5.3) | 設計 | 中 | 中 | ★★☆ |
| 14 | A/B比較モード (6.5) | 機能 | 中 | 低 | ★☆☆ |

---

## 9. 推奨実装ロードマップ

### Phase A: 安全性＋即効性能改善（1週間）
1. Stage2並列モードの修正 or 無効化
2. 品質フィルタ無効時のレコード整合性修正
3. Sobel/Tenengradダウンスケール実装
4. LKフローダウンスケール実装
5. 上記の単体テスト追加

### Phase B: 堅牢性＋UX（2週間）
1. VO失敗時の重み自動調整
2. `--resume` CLIオプション（StageTempStore活用）
3. エッジケーステスト追加
4. COLMAP互換出力

### Phase C: アーキテクチャ改善（1ヶ月）
1. KeyframeSelector責務分離（段階的リファクタリング）
2. TypedDict/dataclass導入
3. 品質スコアHTMLレポート
4. 自動キャリブレーション推定

---

## 10. 既存VO改良提案との整合性

既存の `vo_improvement_proposals.md` で計画されているPhase 1（D1信頼度、A2 MAGSAC++、A3サブピクセル、C3適応サブサンプリング）は全て実装済みです。本レビューの提案はこれらと競合せず、以下の補完関係にあります：

- **本レビュー 3.2（LKフローダウンスケール）**: C1 GPUフローの前段として、CPU環境での即効改善
- **本レビュー 4.1（VO失敗時重み調整）**: D1信頼度スコアの活用拡張
- **本レビュー 6.2（resume対応）**: StageTempStoreの機能完成

VO提案のPhase 2（E系GUI/Rerun強化）は本レビューのスコープ外ですが、Phase Bと並行実施可能です。
