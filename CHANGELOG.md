# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Stage別ベンチマークCLI `scripts/benchmark.py` を追加（JSON/CSV出力、比較表示）。
- Stage1共通エンジン `core/stage1_engine.py` を追加（GUI/Selector共通のmono走査）。
- Stage結果テンポラリ保存 `core/stage_temp_store.py` を追加。
- 性能設定キーを追加:
  - `opencv_thread_count`
  - `stage1_process_workers`
  - `stage1_prefetch_size`
  - `stage1_metrics_batch_size`
  - `stage1_gpu_batch_enabled`
  - `darwin_capture_backend`
  - `mps_min_pixels`

### Changed
- 解析順序を `Stage1 -> Stage0 -> Stage2 -> Stage3` に統一。
- GUI/CLIでStage中間結果を `~/.360split/tmp_runs/<analysis_run_id>/` にJSONL保存する方式へ変更（成功時削除、失敗時保持）。
- `core/accelerator.py`:
  - macOS arm64で `sysctl hw.perflevel0.logicalcpu` に基づくOpenCVスレッド最適化を実装。
  - runtime再設定API `configure_runtime()` を追加。
  - MPS向け `gpu_cvtColor/gpu_filter2D/gpu_resize/gpu_remap` 経路を追加（失敗時CPUフォールバック）。
  - `batch_laplacian_var()` を追加。
- `core/video_loader.py`:
  - `create_video_capture()` を追加し、macOSバックエンド優先順を統一。
  - prefetch既定値を 10 -> 32 に変更。
- `core/quality_score.py`:
  - `compute_raw_metrics_batch()` を追加し、Stage1バッチ品質計算に対応。
- `main.py`:
  - 上記性能キーに対応するCLIオプションを追加。
- `gui/settings_dialog.py`:
  - 上記性能キーを設定ダイアログから編集・保存可能に変更。

## [1.1.0] - 2026-02-21
### Added
- **Visual Odometry Module** (`test/vo_only_test.py`) - モノキュラーVisual Odometryによるカメラ軌跡推定機能（実験的）
- **IMU融合システム** (`test/vo_imu_fusion.py`) - VOとIMUセンサーデータの融合（開発中）
- **エクスポートダイアログ** (`gui/export_dialog.py`) - 出力設定とフォーマット選択UI
- **画像I/Oユーティリティ** (`utils/image_io.py`) - 画像の入出力とメタデータ管理の拡張
- OSVファイルフォーマット対応（センサーデータ抽出）
- 3D軌跡可視化機能
- 前後魚眼2動画入力モード（`--front-video` / `--rear-video`）
- Stage2動体除去のCLIオプション群（`--remove-dynamic-objects` ほか）
- 魚眼外周マスク調整オプション（有効/無効、半径比、中心オフセット）

### Changed
- GeometricEvaluatorモジュールの再利用性向上（VO統合のため）
- プロジェクト構造ドキュメントの更新
- `ConfigManager.default_config()` を唯一のデフォルト設定ソースとして統一
- READMEのCLIオプション、設定例、出力構造を現行実装に同期

### Documentation
- `test/VO開発完了レポート.md` - Visual Odometry開発の技術詳細
- `test/3D軌跡解析レポート.md` - 軌跡解析結果レポート
- `test/OSVファイル_メタデータ調査レポート.md` - センサーデータ調査レポート
- `README.md` の更新（実験的機能セクション、CLI入力モード、動体除去設定）

## [1.0.0] - YYYY-MM-DD

### Added
- 2段階キーフレーム選択パイプライン（Stage 1: 品質フィルタリング、Stage 2: 幾何学的評価）
- GUIモード（PySide6ベース）
  - メインウィンドウ
  - 動画プレビュー
  - タイムラインウィジェット
  - キーフレーム詳細パネル
  - 設定ダイアログ
- CLIモード（バッチ処理対応）
- 環境プリセットシステム（Outdoor/Indoor/Mixed）
- 360度映像対応
  - Equirectangular → Cubemap変換
  - 天頂/天底マスク処理
- GPU高速化（Apple Silicon MPS / NVIDIA CUDA対応）
- コアアルゴリズム
  - 品質評価（鮮明度、露光、モーションブラー）
  - 幾何学的評価（GRIC、特徴点マッチング）
  - 適応的選択（SSIM、オプティカルフロー）
- 画像処理機能
  - Equirectangularプロジェクション変換
  - マスク処理
  - スティッチング（Fast/HQS/DMS）
- ハードウェア抽象化レイヤ（accelerator.py）
- 設定管理システム（JSON/プリセット）
- ロギングシステム

### Features
- レスキューモード（特徴点不足時の自動対応）
- NMS（Non-Maximum Suppression）による最終選別
- Softmax深度スコアリング
- LRUキャッシュによる特徴点キャッシング
- フレームプリフェッチ

---

## バージョン番号について

- **Major (X.0.0)**: 互換性のない大きな変更
- **Minor (0.X.0)**: 後方互換性のある機能追加
- **Patch (0.0.X)**: 後方互換性のあるバグ修正

---

**注記**: [Unreleased]セクションは開発中の機能を示します。正式リリース時にバージョン番号が付与されます。
