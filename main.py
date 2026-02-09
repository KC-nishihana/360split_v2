#!/usr/bin/env python3
"""
360Split - 360度動画ベース3D再構成GUIソフトウェア
メインエントリポイント

360度動画からフォトグラメトリや3DGS用画像の最適抽出を行うツール。
キーフレーム選択機能により、幾何学的整合性を担保したフレームを自動選別する。

Usage:
    python main.py                  # GUIモード起動
    python main.py --cli video.mp4  # CLIモード（GUI不要）
    python main.py --help           # ヘルプ表示
"""

import sys
import os
import argparse
import json
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_arguments():
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        prog="360Split",
        description="360度動画ベース3D再構成GUIソフトウェア - キーフレーム最適抽出ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                           GUIモードで起動
  python main.py --cli input.mp4           CLIモードで解析
  python main.py --cli input.mp4 -o out/   出力先指定
  python main.py --cli input.mp4 --config settings.json  設定ファイル指定
        """
    )

    parser.add_argument(
        "--cli",
        metavar="VIDEO",
        type=str,
        default=None,
        help="CLIモード: 指定した動画ファイルを解析（GUIなし）"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: 動画と同じディレクトリ内の 'keyframes' フォルダ）"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="設定ファイルパス（JSON形式）"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "tiff"],
        default="png",
        help="出力画像フォーマット（デフォルト: png）"
    )

    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=None,
        help="最大キーフレーム数（指定しない場合は自動決定）"
    )

    parser.add_argument(
        "--min-interval",
        type=int,
        default=None,
        help="最小キーフレーム間隔（フレーム数）"
    )

    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=None,
        help="SSIM変化検知閾値（0.0-1.0）"
    )

    parser.add_argument(
        "--equirectangular",
        action="store_true",
        default=False,
        help="入力を360度Equirectangular動画として処理"
    )

    parser.add_argument(
        "--apply-mask",
        action="store_true",
        default=False,
        help="天底マスク処理を適用（撮影者/三脚の除去）"
    )

    parser.add_argument(
        "--cubemap",
        action="store_true",
        default=False,
        help="キーフレームをCubemap形式でも出力"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="詳細ログ出力"
    )

    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """設定をロードする。設定ファイルが指定されていればそれを読み込み、なければデフォルト値を使用。"""
    import config as default_config

    settings = {
        "laplacian_threshold": default_config.LAPLACIAN_THRESHOLD,
        "brightness_min": default_config.BRIGHTNESS_MIN,
        "brightness_max": default_config.BRIGHTNESS_MAX,
        "motion_blur_threshold": default_config.MOTION_BLUR_THRESHOLD,
        "softmax_beta": default_config.SOFTMAX_BETA,
        "gric_ratio_threshold": default_config.GRIC_RATIO_THRESHOLD,
        "min_feature_matches": default_config.MIN_FEATURE_MATCHES,
        "ssim_change_threshold": default_config.SSIM_CHANGE_THRESHOLD,
        "min_keyframe_interval": default_config.MIN_KEYFRAME_INTERVAL,
        "max_keyframe_interval": default_config.MAX_KEYFRAME_INTERVAL,
        "momentum_boost_factor": default_config.MOMENTUM_BOOST_FACTOR,
        "weight_sharpness": default_config.WEIGHT_SHARPNESS,
        "weight_exposure": default_config.WEIGHT_EXPOSURE,
        "weight_geometric": default_config.WEIGHT_GEOMETRIC,
        "weight_content": default_config.WEIGHT_CONTENT,
        "equirect_width": default_config.EQUIRECT_WIDTH,
        "equirect_height": default_config.EQUIRECT_HEIGHT,
        "cubemap_face_size": default_config.CUBEMAP_FACE_SIZE,
        "perspective_fov": default_config.PERSPECTIVE_FOV,
        "mask_dilation_kernel": default_config.MASK_DILATION_KERNEL,
        "output_image_format": default_config.OUTPUT_IMAGE_FORMAT,
        "output_jpeg_quality": default_config.OUTPUT_JPEG_QUALITY,
    }

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            settings.update(user_config)
            logger.info(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗: {e} （デフォルト設定を使用）")

    return settings


def run_cli(args):
    """CLIモードでキーフレーム抽出を実行する。"""
    from core.video_loader import VideoLoader
    from core.keyframe_selector import KeyframeSelector
    from processing.equirectangular import EquirectangularProcessor
    from processing.mask_processor import MaskProcessor

    video_path = args.cli
    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    # 設定ロード
    config = load_config(args.config)

    # コマンドライン引数で設定をオーバーライド
    if args.min_interval is not None:
        config["min_keyframe_interval"] = args.min_interval
    if args.ssim_threshold is not None:
        config["ssim_change_threshold"] = args.ssim_threshold
    if args.format:
        config["output_image_format"] = args.format

    # 出力ディレクトリ
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(video_path).parent / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("360Split - CLIモード")
    logger.info("=" * 60)
    logger.info(f"入力動画: {video_path}")
    logger.info(f"出力先:   {output_dir}")
    logger.info(f"フォーマット: {config['output_image_format']}")
    if args.equirectangular:
        logger.info("360度 Equirectangular モード: 有効")
    if args.apply_mask:
        logger.info("マスク処理: 有効")
    logger.info("-" * 60)

    # 動画読み込み
    loader = VideoLoader()
    if not loader.load(video_path):
        logger.error("動画の読み込みに失敗しました。")
        sys.exit(1)

    meta = loader.get_metadata()
    logger.info(f"動画情報: {meta['width']}x{meta['height']}, "
                f"{meta['fps']:.1f}fps, {meta['frame_count']}フレーム, "
                f"{meta['duration']:.1f}秒")

    # キーフレーム選択
    selector = KeyframeSelector(config)

    def progress_callback(current, total, message=""):
        pct = int(current / total * 100) if total > 0 else 0
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}% {message}", end="", flush=True)

    logger.info("キーフレーム解析を開始...")
    keyframes = selector.select_keyframes(
        loader,
        progress_callback=progress_callback
    )
    print()  # 改行

    if not keyframes:
        logger.warning("キーフレームが検出されませんでした。閾値の調整を検討してください。")
        loader.release()
        sys.exit(0)

    logger.info(f"検出キーフレーム数: {len(keyframes)}")

    # 最大キーフレーム数で制限
    if args.max_keyframes and len(keyframes) > args.max_keyframes:
        keyframes.sort(key=lambda kf: kf.combined_score, reverse=True)
        keyframes = keyframes[:args.max_keyframes]
        keyframes.sort(key=lambda kf: kf.frame_index)
        logger.info(f"上位 {args.max_keyframes} フレームに制限")

    # マスク処理
    mask_processor = MaskProcessor() if args.apply_mask else None
    equirect_processor = EquirectangularProcessor() if args.equirectangular else None

    # キーフレーム出力
    logger.info("キーフレームを出力中...")
    fmt = config["output_image_format"]

    for i, kf in enumerate(keyframes):
        frame = loader.get_frame(kf.frame_index)
        if frame is None:
            continue

        # マスク処理
        if mask_processor and args.apply_mask:
            h, w = frame.shape[:2]
            nadir_mask = mask_processor.create_nadir_mask(w, h)
            frame = mask_processor.apply_mask(frame, nadir_mask)

        # 通常画像の出力
        filename = f"keyframe_{kf.frame_index:06d}.{fmt}"
        filepath = output_dir / filename

        import cv2
        if fmt == "jpg":
            cv2.imwrite(str(filepath), frame,
                       [cv2.IMWRITE_JPEG_QUALITY, config["output_jpeg_quality"]])
        elif fmt == "tiff":
            cv2.imwrite(str(filepath), frame)
        else:
            cv2.imwrite(str(filepath), frame)

        # Cubemap出力
        if args.cubemap and equirect_processor:
            cubemap_dir = output_dir / "cubemap" / f"frame_{kf.frame_index:06d}"
            cubemap_dir.mkdir(parents=True, exist_ok=True)
            faces = equirect_processor.to_cubemap(frame, config["cubemap_face_size"])
            for face_name, face_img in faces.items():
                face_path = cubemap_dir / f"{face_name}.{fmt}"
                cv2.imwrite(str(face_path), face_img)

        # スコア情報の表示
        logger.info(f"  [{i+1}/{len(keyframes)}] Frame {kf.frame_index:6d} | "
                    f"Time {kf.timestamp:7.2f}s | Score {kf.combined_score:.3f}")

    # メタデータ出力
    metadata = {
        "video_path": str(Path(video_path).resolve()),
        "total_frames": meta["frame_count"],
        "fps": meta["fps"],
        "duration": meta["duration"],
        "resolution": f"{meta['width']}x{meta['height']}",
        "keyframe_count": len(keyframes),
        "settings": config,
        "keyframes": [
            {
                "frame_index": kf.frame_index,
                "timestamp": round(kf.timestamp, 3),
                "combined_score": round(kf.combined_score, 4),
                "quality_scores": {k: round(v, 4) for k, v in kf.quality_scores.items()} if kf.quality_scores else {},
                "geometric_scores": {k: round(v, 4) for k, v in kf.geometric_scores.items()} if kf.geometric_scores else {},
                "adaptive_scores": {k: round(v, 4) for k, v in kf.adaptive_scores.items()} if kf.adaptive_scores else {},
            }
            for kf in keyframes
        ]
    }

    metadata_path = output_dir / "keyframe_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("-" * 60)
    logger.info(f"完了: {len(keyframes)} キーフレームを出力しました")
    logger.info(f"出力先: {output_dir}")
    logger.info(f"メタデータ: {metadata_path}")

    loader.release()


def run_gui():
    """GUIモードでアプリケーションを起動する。"""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
    except ImportError:
        logger.error(
            "PySide6がインストールされていません。\n"
            "  pip install PySide6\n"
            "または CLIモードを使用してください:\n"
            "  python main.py --cli video.mp4"
        )
        sys.exit(1)

    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("360Split")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("360Split")

    # アプリケーション全体のフォント設定
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    # Note: Qt6/PySide6 handles HiDPI automatically, Qt.AA_UseHighDpiPixmaps is not available

    # メインウィンドウ
    window = MainWindow()
    window.show()

    logger.info("360Split GUI を起動しました")

    sys.exit(app.exec())


def main():
    """メインエントリポイント。"""
    args = parse_arguments()

    # ログレベル設定
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
