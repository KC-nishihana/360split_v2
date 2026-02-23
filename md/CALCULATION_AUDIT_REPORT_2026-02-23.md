# 360Split 全計算処理監査レポート

作成日: 2026-02-23  
対象リビジョン: 作業ツリー現状（`main.py`, `core/`, `processing/`, `gui/workers.py`）

## 1. 目的とスコープ
- 目的: 搭載されている計算処理の内容と処理間連携を、実装ベースで棚卸しする。
- 対象:
  - `core/` の評価・選択・VO・補助演算
  - `processing/` の投影変換・ステッチング・マスク生成・検出系
  - `gui/workers.py` のオーケストレーション連携
  - `main.py` の CLI/GUI 実行フロー
- 非対象:
  - UI見た目の詳細
  - 学習モデル内部の推論アルゴリズム詳細（YOLO/SAM自体の内部）

## 2. パイプライン連携図（CLI/GUI共通）

```text
main.py
  -> 入力解決(resolve_cli_input/create_loader)
  -> 設定マージ(load_config + apply_cli_overrides)
  -> KeyframeSelector.select_keyframes()
       Stage0: 軽量走査
       Stage1: 品質フィルタ
       Stage2: 幾何 + 適応評価
       Stage3: VO軌跡再評価
  -> 出力(export + metadata/frame_metrics)
  -> (必要時) Equirect/Cubemap/Perspective + TargetMask
```

### 入力モード差分

| モード | ローダー | 評価で代表フレーム | エクスポート |
|---|---|---|---|
| 単眼 | `VideoLoader` | 単眼フレーム | 単一画像 + 任意で Cubemap/Perspective |
| OSV | `DualVideoLoader` | Left を代表（必要時 Pair） | `images/L`, `images/R` ペア出力 |
| Front/Rear | `FrontRearVideoLoader` | Front を代表（必要時 Pair） | `images/F`, `images/R` ペア出力 |

## 3. Stage別計算処理

## Stage0: 軽量運動量走査
- 実装: `core/keyframe_selector.py::_stage0_lightweight_motion_scan`
- 主計算:
  - `flow_mag_light = optical_flow(prev, cur)`
  - `ssim_light = SSIM(prev, cur)`
  - `sharpness = QualityEvaluator.evaluate(...).sharpness`
  - `texture_risk = clip(1 - min(sharpness / lap_th, 1), 0, 1)`
  - `flow_risk = clip(flow_mag / flow_norm_factor, 0, 1)`
  - `ssim_change = clip(1 - ssim_light, 0, 1)`
  - `motion_risk = clip(0.4*texture_risk + 0.4*flow_risk + 0.2*ssim_change, 0, 1)`
  - キャリブレーション有効時は VO も計算し `vo_step_proxy_norm` を中央値正規化
- 出力:
  - `stage0_metrics[frame_idx] = {motion_risk, flow_mag_light, ssim_light, VO群}`

## Stage1: 高速品質フィルタ
- 実装: `core/keyframe_selector.py::_stage1_fast_filter`
- 判定条件（単眼）:
  - `sharpness >= LAPLACIAN_THRESHOLD`
  - `motion_blur <= MOTION_BLUR_THRESHOLD`
  - `exposure >= EXPOSURE_THRESHOLD`
- 判定条件（ペア）:
  - 両レンズが閾値を満たす AND 条件
  - `sharpness = min(lens_a, lens_b)`, `motion_blur = max(lens_a, lens_b)` の保守統合
- 出力:
  - Stage2入力候補 `[{frame_idx, quality_scores}, ...]`

## Stage2: 精密評価
- 実装: `core/keyframe_selector.py::_stage2_precise_evaluation`
- 主計算:
  - 幾何評価 `GeometricEvaluator.evaluate(last_keyframe, current_frame, ...)`
  - 適応評価 `AdaptiveSelector.evaluate(last_keyframe, current_frame, frames_window)`
  - 低変化スキップ:
    - `ssim > SSIM_CHANGE_THRESHOLD` かつ強制挿入条件なしなら除外
  - 統合スコア:
    - `combined = w_sharpness*sharpness_norm + w_exposure*exposure + w_geometric*geom + w_content*content`
    - `sharpness_norm = min(sharpness / SHARPNESS_NORM_FACTOR, 1)`
    - `content = (1-ssim + optical_flow_norm)/2`
  - 停止区間ペナルティ:
    - VO/flow閾値から stationary 区間抽出
    - `combined *= (1 - STATIONARY_PENALTY)`（soft時）
- 出力:
  - `Stage2 candidates`
  - `Stage2 records`（指標・姿勢・キー判定ログ）

## Stage3: 軌跡再評価
- 実装: `core/keyframe_selector.py::_stage3_refine_with_trajectory`
- 主計算:
  - 候補フレーム列で VO 逐次推定
  - 軌跡整合スコア `trajectory_consistency` を `valid_ratio, dir_term, rot_term, step_term` で合成
  - 連続相対姿勢を `integrate_relative_trajectory(...)` で world座標に積分
  - 再スコア:
    - `combined_stage3 = clip(w_base*combined_stage2 + w_traj*trajectory_consistency - w_risk*stage0_motion_risk, 0, 1)`
- 出力:
  - 最終キーフレーム集合（NMS + 最大間隔制約後）
  - `t_xyz`, `q_wxyz` を records へ反映

## 4. 実装単位インベントリ（core）

| モジュール | 関数 | 入力 | 算出値 | 主要式・判定 | 出力先 |
|---|---|---|---|---|---|
| `core/quality_evaluator.py` | `evaluate` | frame | sharpness, motion_blur, exposure, softmax_depth | Sobel/Laplacian/ガウス露光/softmax深度 | Stage0/1/2 |
| `core/quality_evaluator.py` | `_compute_sharpness` | gray | sharpness | `var(Laplacian(gray))` | 品質スコア |
| `core/quality_evaluator.py` | `_compute_motion_blur` | sobel_x/y | motion_blur | 勾配方向バランス偏り | 品質スコア |
| `core/quality_evaluator.py` | `_compute_exposure_score` | gray | exposure | 輝度128中心のガウス評価 | 品質スコア |
| `core/quality_evaluator.py` | `_compute_softmax_depth_score` | gradients | softmax_depth | edge confidence + log-sum-exp | 品質スコア |
| `core/geometric_evaluator.py` | `evaluate` | frame1/2 + masks | gric, feature_dist_1/2, match_count, ray_dispersion | GRIC + 分布 + 光線分散 | Stage2 |
| `core/geometric_evaluator.py` | `_build_match_context` | frame1/2 | kp/desc/matches/pts | ORB/SIFT + ratio test | GRIC計算前段 |
| `core/geometric_evaluator.py` | `_compute_gric_score_from_context` | pts1/2 | gric score | H/F推定・誤差・GRIC比較・縮退判定 | Stage2幾何 |
| `core/geometric_evaluator.py` | `_compute_gric` | residuals | GRIC値 | Torr 1998式 | GRIC評価 |
| `core/geometric_evaluator.py` | `_compute_feature_distribution_from_keypoints` | kp | 分布スコア | 4x4ヒストグラムのエントロピー正規化 | 幾何補助 |
| `core/geometric_evaluator.py` | `_compute_ray_dispersion_from_keypoints` | kp | 光線分散スコア | ray共分散の固有値比 | 幾何補助 |
| `core/adaptive_selector.py` | `evaluate` | frame1/2/window | ssim, optical_flow, momentum | SSIM + LK flow + flow差分加速度 | Stage0/2 |
| `core/adaptive_selector.py` | `compute_ssim` | frame1/2 | ssim | 標準SSIM式 | 適応評価 |
| `core/adaptive_selector.py` | `compute_optical_flow_magnitude` | frame1/2 | flow magnitude | LK疎フロー or Farneback | 適応評価 |
| `core/adaptive_selector.py` | `compute_camera_momentum` | frame window | momentum | `mean(abs(diff(flow_magnitudes)))` | 適応評価 |
| `core/keyframe_selector.py` | `select_keyframes` | loader, callbacks | keyframes | Stage0->1->2->3統合 | CLI/GUI 主出力 |
| `core/keyframe_selector.py` | `_compute_combined_score` | quality/geometric/adaptive | combined | 重み付き線形統合 | Stage2/3 |
| `core/keyframe_selector.py` | `_compute_stationary_flags` | Stage2 records | stationary mask | quantile閾値 + ヒステリシス + min duration | Stage2抑制 |
| `core/keyframe_selector.py` | `_apply_nms` | candidates | selected | 時間窓NMS | 最終候補 |
| `core/keyframe_selector.py` | `_enforce_max_interval` | keyframes | keyframes | 最大間隔チェック（ログ） | 最終候補 |
| `core/visual_odometry/vo_klt.py` | `estimate` | prev/cur frame, calib | VO metrics | goodFeatures + LK + Essential + recoverPose | Stage0/3 |
| `core/visual_odometry/trajectory_integrator.py` | `integrate_relative_trajectory` | relative samples | world poses | 相対回転積 + 進行方向積分 | Stage3 pose |
| `core/visual_odometry/calibration_check.py` | `run_calibration_check` | video/calib | 比較画像 + 指標 | undistort前後の線分長比 | 検証モード出力 |
| `core/accelerator.py` | `compute_laplacian_var` | gray | var | CUDA/MPS/CPU分岐 | 品質評価 |
| `core/accelerator.py` | `compute_optical_flow_sparse` | prev/curr gray | mean flow | LK疎フロー | 適応評価 |
| `core/accelerator.py` | `gpu_remap` | src + map_x/y | remap image | GPU remap fallback CPU | 360投影 |
| `core/accelerator.py` | `batch_ssim` | frames + ref | ssim list | Torch batch / CPU fallback | 適応評価補助 |

## 5. 実装単位インベントリ（processing）

| モジュール | 関数 | 入力 | 算出値 | 主要式・判定 | 出力先 |
|---|---|---|---|---|---|
| `processing/fisheye_rig.py` | `calibrate_from_checkerboard` | front/rear checkerboard列 | `RigCalibration` | fisheye calibrate + stereoCalibrate | front/rearリグ校正 |
| `processing/fisheye_rig.py` | `stitch_to_equirect` | front/rear frame | stitched, seam_mask | 前後半球合成 + seam blend | Stage2/Export |
| `processing/fisheye_rig.py` | `extract_360_features` | equirect + seam | keypoints, desc, seam_count | ORB/SIFT + polar mask | Stage2 quality補助 |
| `processing/fisheye_rig.py` | `_fisheye_to_equirect` | fisheye frame | equirect projection | 球面->画像 remap | Stage2/Export |
| `processing/equirectangular.py` | `to_cubemap` | equirect | 6 faces | UV計算 + remap + cache | Export |
| `processing/equirectangular.py` | `from_cubemap` | 6 faces | equirect | 逆投影合成 + 重み正規化 | Export/再構成 |
| `processing/equirectangular.py` | `to_perspective` | equirect + yaw/pitch/fov | perspective image | UV計算 + remap | Export |
| `processing/equirectangular.py` | `compute_coverage_map` | keyframe poses | coverage map | 視線方向集計 | 360解析補助 |
| `processing/equirectangular.py` | `compute_overlap` | pose1/pose2 | overlap | 視野角ベース重なり | 解析補助 |
| `processing/stitching.py` | `stitch_fast` | image list | stitched image | overlapフェザリング | Export |
| `processing/stitching.py` | `stitch_high_quality` | image list | stitched image | ORB + H推定 + warp + blend | Export |
| `processing/stitching.py` | `stitch_depth_aware` | image list (+depth) | stitched image | 深度重みブレンド | Export |
| `processing/stitching.py` | `_estimate_depth_maps` | images | depth maps | Sobel edge inverse | depth-aware補助 |
| `processing/mask_processor.py` | `create_nadir_mask` | w,h | nadir mask | 底部円マスク | CLI/Export |
| `processing/mask_processor.py` | `create_zenith_mask` | w,h | zenith mask | 上部円マスク | 360補助 |
| `processing/mask_processor.py` | `create_fisheye_valid_mask` | w,h,ratio,offset | valid mask | レンズ有効円 | Stage1/2/Export |
| `processing/mask_processor.py` | `detect_moving_objects` | frames | motion mask | 背景差分 + morph | TargetMask |
| `processing/mask_processor.py` | `dilate_mask` | mask | dilated mask | morphology dilate | TargetMask |
| `processing/target_mask_generator.py` | `generate_mask` | frame, target_classes, motion_frames | binary mask(0/255) | YOLO+SAM OR 空 OR motion | Stage2/Export |
| `processing/target_mask_generator.py` | `_detect_motion_mask` | frame sequence | motion mask | `detect_moving_objects` 呼び出し | TargetMask |
| `processing/target_mask_generator.py` | `_detect_sky_mask` | frame | sky mask | HSV+テクスチャ+上端連結成分 | TargetMask |
| `processing/object_detector.py` | `detect` | frame + class filters | detections | YOLO推論 + class map filtering | TargetMask |
| `processing/instance_segmentor.py` | `segment` | frame + boxes | instance masks | SAM推論 / box fallback | TargetMask |

## 6. GUI連携（計算オーケストレーション）

| モジュール | クラス | 役割 |
|---|---|---|
| `gui/workers.py` | `Stage1Worker` | 品質計算のみ先行し、バッチでGUIへ送信 |
| `gui/workers.py` | `Stage2Worker` | `KeyframeSelector.select_keyframes` 実行、指標をStage1結果へマージ |
| `gui/workers.py` | `UnifiedAnalysisWorker` | Stage1+Stage2/3 を1回実行で統合 |
| `gui/workers.py` | `ExportWorker` | 画像出力、ステッチング、投影変換、対象マスク生成を統合 |
| `gui/workers.py` | `GenerateMasksWorker` | 既存画像に後処理マスク生成 |

注記: GUIは計算本体を再実装せず、`core`/`processing` の API 呼び出しで連携している。

## 7. 360/ステレオ/マスク連携整理

## 魚眼有効領域マスク
- Stage1/2で `ENABLE_FISHEYE_BORDER_MASK` 有効時に円形有効領域外を無効化。
- Pair評価では feature mask と dynamic mask を `bitwise_and` 合成して幾何評価へ入力。

## 動体マスク
- Stage2: `ENABLE_DYNAMIC_MASK_REMOVAL` 時に `TargetMaskGenerator.generate_mask` を呼び、幾何評価の特徴点抽出マスクとして使用。
- Export: 解析時 precomputed mask がある場合は再利用可能。欠損分は再解析。

## シーム特徴
- Front/Rear + `ENABLE_RIG_STITCHING` 時に `stitch_to_equirect` と `extract_360_features` を使い、シーム近傍特徴点数を品質メタに追加。

## ステッチング方式差分（Export）
- Fast: overlap線形フェザリング
- HQ: ORB特徴 + homography + warp/blend
- Depth-aware: 深度重みでoverlap合成

## 再投影
- Cubemap: 6面 `front/back/left/right/up/down`
- Perspective: `(yaw, pitch, fov)` 組ごとに投影

## 8. 主I/F一覧（明示対象）
- `KeyframeSelector.select_keyframes(...)`
- `QualityEvaluator.evaluate(...)`
- `GeometricEvaluator.evaluate(...)`
- `AdaptiveSelector.evaluate(...)`
- `KLTVisualOdometry.estimate(...)`
- `integrate_relative_trajectory(...)`
- `EquirectangularProcessor.to_cubemap(...)`, `to_perspective(...)`, `from_cubemap(...)`
- `StitchingProcessor.stitch_fast(...)`, `stitch_high_quality(...)`, `stitch_depth_aware(...)`
- `TargetMaskGenerator.generate_mask(...)`

## 9. シナリオ別チェック観点

| シナリオ | チェック観点 |
|---|---|
| 1. 単眼CLI（Stage0-3 + export） | Stage0/3 VO有効条件、metadata出力、frame_metrics整合 |
| 2. OSVステレオ | `DualVideoLoader` 分離・同期、L/R保存、pair品質AND |
| 3. front/rear | `FrontRearVideoLoader` 同期、`FisheyeRigProcessor` 経路、魚眼外周マスク |
| 4. 動体除去ON | YOLO/SAM + motion差分、幾何評価へのmask伝播 |
| 5. VO有効/無効・calib有無 | calib未指定時VO無効化と警告、指定時姿勢推定値反映 |
| 6. GUI UnifiedAnalysisWorker | Stage1逐次表示 + Stage2/3後の指標マージとkeyframe同期 |

## 10. 妥当性チェック結果

実行コマンド:

```bash
pytest -q
```

結果:
- import path 未設定のため `ModuleNotFoundError` で収集失敗

実行コマンド:

```bash
PYTHONPATH=. pytest -q
```

結果:
- `34 passed in 0.64s`

## 11. 要注意点（監査所見）

追補（2026-02-23 実装修正反映）:
- `core/accelerator.py` の `batch_ssim` CPUフォールバック引数不一致は修正済み。
- `_enforce_max_interval` はギャップ補完挿入を実装済み（source candidates から補完）。
- `--equirectangular` の説明は `main.py` / `README.md` で「入力モード指定」である旨を明記済み。

1. `core/accelerator.py` の `batch_ssim` CPUフォールバック呼び出し
- 旧コードでは `selector.compute_ssim(ref=reference, frame=f)` となっており引数不一致リスクがあった。
- 追補時点では `selector.compute_ssim(reference, f)` に修正済み。

2. `main.py` の `--equirectangular` オプション
- 仕様上、`config["projection_mode"]` を設定する入力モード指定であり、CLI出力時の再投影強制とは別。
- 誤解回避のため、`main.py` 引数help/`README.md` でこの挙動を明記済み。

3. `_enforce_max_interval` の挙動
- 旧実装はログのみだったが、追補時点でギャップ区間補完（source候補から挿入）を実装済み。
- 候補が存在しない区間は警告ログを出して継続する。

## 12. 前提
- 本レポートは静的読解 + テスト結果（`PYTHONPATH=. pytest -q`）を根拠とする。
- 「全計算処理」は `core` と `processing` を主対象、`gui/workers.py` は連携層として扱う。
