# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Visual Odometry Module** (`test/vo_only_test.py`) - モノキュラーVisual Odometryによるカメラ軌跡推定機能（実験的）
- **IMU融合システム** (`test/vo_imu_fusion.py`) - VOとIMUセンサーデータの融合（開発中）
- **エクスポートダイアログ** (`gui/export_dialog.py`) - 出力設定とフォーマット選択UI
- **画像I/Oユーティリティ** (`utils/image_io.py`) - 画像の入出力とメタデータ管理の拡張
- OSVファイルフォーマット対応（センサーデータ抽出）
- 3D軌跡可視化機能

### Changed
- GeometricEvaluatorモジュールの再利用性向上（VO統合のため）
- プロジェクト構造ドキュメントの更新

### Documentation
- `test/VO開発完了レポート.md` - Visual Odometry開発の技術詳細
- `test/3D軌跡解析レポート.md` - 軌跡解析結果レポート
- `test/OSVファイル_メタデータ調査レポート.md` - センサーデータ調査レポート
- README.mdの更新（実験的機能セクション追加）

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
