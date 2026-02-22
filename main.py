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
import argparse
import json
from pathlib import Path
from typing import Tuple
import cv2

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger, get_logger, set_log_level
from utils.image_io import write_image

# アプリケーション起動時にルートロガーを初期化
setup_logger()
logger = get_logger(__name__)


def parse_arguments():
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        prog="360Split",
        description="360度動画ベース3D再構成GUIソフトウェア - キーフレーム最適抽出ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                                  GUIモードで起動
  python main.py --cli input.mp4                  CLIモードで解析
  python main.py --cli input.mp4 -o out/          出力先指定
  python main.py --cli input.mp4 --preset indoor  屋内プリセット使用
  python main.py --cli input.mp4 --preset outdoor 屋外高品質プリセット
  python main.py --cli input.mp4 --config settings.json  設定ファイル指定

プリセット:
  outdoor : 屋外・晴天用（高品質重視、厳格な品質基準）
  indoor  : 屋内・暗所用（追跡維持重視、特徴点不足に対応）
  mixed   : 混合環境用（適応型、露出変化に対応）
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
        "--front-video",
        type=str,
        default=None,
        help="前後魚眼モード: 前方レンズ動画パス"
    )
    parser.add_argument(
        "--rear-video",
        type=str,
        default=None,
        help="前後魚眼モード: 後方レンズ動画パス"
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
        "--preset",
        type=str,
        choices=["outdoor", "indoor", "mixed"],
        default=None,
        help="環境プリセット: outdoor（屋外・高品質）、indoor（屋内・追跡重視）、mixed（混合・適応型）"
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
        "--remove-dynamic-objects",
        action="store_true",
        default=False,
        help="Stage2で動体領域を除外して幾何評価を実行する"
    )
    parser.add_argument(
        "--dynamic-mask-frames",
        type=int,
        default=None,
        help="動体差分に使うフレーム数（2以上）"
    )
    parser.add_argument(
        "--dynamic-mask-threshold",
        type=int,
        default=None,
        help="動体差分しきい値（1-255）"
    )
    parser.add_argument(
        "--dynamic-mask-dilation",
        type=int,
        default=None,
        help="動体マスク膨張サイズ（0で無効）"
    )
    parser.add_argument(
        "--dynamic-mask-inpaint",
        action="store_true",
        default=False,
        help="動体マスクのインペイントフックを有効化する"
    )
    parser.add_argument(
        "--dynamic-mask-inpaint-module",
        type=str,
        default=None,
        help="インペイントフックモジュール（inpaint_frame(frame, mask) を実装）"
    )
    parser.add_argument(
        "--disable-stage0-scan",
        action="store_true",
        default=False,
        help="Stage0軽量走査を無効化する"
    )
    parser.add_argument(
        "--stage0-stride",
        type=int,
        default=None,
        help="Stage0固定サンプリング間隔（フレーム数）"
    )
    parser.add_argument(
        "--disable-stage3-refinement",
        action="store_true",
        default=False,
        help="Stage3軌跡再評価を無効化する"
    )
    parser.add_argument(
        "--stage3-weight-base",
        type=float,
        default=None,
        help="Stage3再スコア式のbase重み"
    )
    parser.add_argument(
        "--stage3-weight-trajectory",
        type=float,
        default=None,
        help="Stage3再スコア式のtrajectory重み"
    )
    parser.add_argument(
        "--stage3-weight-stage0-risk",
        type=float,
        default=None,
        help="Stage3再スコア式のstage0リスク重み"
    )
    parser.add_argument(
        "--disable-fisheye-border-mask",
        action="store_true",
        default=False,
        help="魚眼外周マスクを無効化（OSV/前後魚眼時の既定ONを上書き）"
    )
    parser.add_argument(
        "--fisheye-mask-radius-ratio",
        type=float,
        default=None,
        help="魚眼有効領域半径比（0.0-1.0, default=0.94）"
    )
    parser.add_argument(
        "--fisheye-mask-center-offset-x",
        type=int,
        default=None,
        help="魚眼有効領域中心Xオフセット（px）"
    )
    parser.add_argument(
        "--fisheye-mask-center-offset-y",
        type=int,
        default=None,
        help="魚眼有効領域中心Yオフセット（px）"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="詳細ログ出力"
    )

    parser.add_argument(
        "--rerun-stream",
        action="store_true",
        default=False,
        help="Rerun Viewerへフレーム指標をストリーミングする"
    )
    parser.add_argument(
        "--rerun-spawn",
        action="store_true",
        default=False,
        help="Rerun Viewerを自動起動する（--rerun-streamと併用）"
    )
    parser.add_argument(
        "--rerun-save",
        type=str,
        default=None,
        help="Rerunログを .rrd に保存するパス"
    )

    return parser.parse_args()


def load_config(config_path: str = None, preset_id: str = None) -> dict:
    """
    設定をロードする。

    プリセット → 設定ファイル → デフォルト の優先順位でマージします。

    Parameters:
    -----------
    config_path : str, optional
        設定ファイルパス（JSON形式）
    preset_id : str, optional
        プリセットID（'outdoor', 'indoor', 'mixed'）

    Returns:
    --------
    dict
        マージされた設定辞書
    """
    from core.config_loader import ConfigManager

    config_manager = ConfigManager()
    settings = config_manager.default_config()

    # プリセット適用
    if preset_id:
        try:
            preset_config = config_manager.load_preset(preset_id, settings)
            settings = preset_config
            logger.info(f"プリセット '{preset_id}' を適用しました")
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info("デフォルト設定を使用します")
        except Exception as e:
            logger.warning(f"プリセット読み込みエラー: {e} （デフォルト設定を使用）")

    # 設定ファイル適用（プリセットの上から更に上書き）
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            settings.update(user_config)
            logger.info(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗: {e}")

    return settings


def resolve_cli_input(args) -> Tuple[str, bool, bool]:
    """CLI入力パスとモード種別を解決する。"""
    video_path = args.cli
    is_front_rear = bool(args.front_video and args.rear_video)

    if is_front_rear:
        if not Path(args.front_video).exists() or not Path(args.rear_video).exists():
            logger.error(f"前後動画が見つかりません: front={args.front_video}, rear={args.rear_video}")
            sys.exit(1)
        video_path = args.front_video
    elif not video_path or not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    is_osv = (not is_front_rear) and video_path.lower().endswith(".osv")
    return video_path, is_front_rear, is_osv


def apply_cli_overrides(config: dict, args) -> None:
    """CLI引数で設定を上書きする。"""
    if args.min_interval is not None:
        config["min_keyframe_interval"] = args.min_interval
    if args.ssim_threshold is not None:
        config["ssim_change_threshold"] = args.ssim_threshold
        config["ssim_threshold"] = args.ssim_threshold
    if args.format:
        config["output_image_format"] = args.format
    if args.remove_dynamic_objects:
        config["enable_dynamic_mask_removal"] = True
    if args.dynamic_mask_frames is not None:
        config["dynamic_mask_motion_frames"] = max(2, args.dynamic_mask_frames)
    if args.dynamic_mask_threshold is not None:
        config["dynamic_mask_motion_threshold"] = max(1, min(255, args.dynamic_mask_threshold))
    if args.dynamic_mask_dilation is not None:
        config["dynamic_mask_dilation_size"] = max(0, args.dynamic_mask_dilation)
    if args.dynamic_mask_inpaint:
        config["dynamic_mask_inpaint_enabled"] = True
    if args.dynamic_mask_inpaint_module:
        config["dynamic_mask_inpaint_module"] = args.dynamic_mask_inpaint_module.strip()
    if args.disable_stage0_scan:
        config["enable_stage0_scan"] = False
    if args.stage0_stride is not None:
        config["stage0_stride"] = max(1, int(args.stage0_stride))
    if args.disable_stage3_refinement:
        config["enable_stage3_refinement"] = False
    if args.stage3_weight_base is not None:
        config["stage3_weight_base"] = float(args.stage3_weight_base)
    if args.stage3_weight_trajectory is not None:
        config["stage3_weight_trajectory"] = float(args.stage3_weight_trajectory)
    if args.stage3_weight_stage0_risk is not None:
        config["stage3_weight_stage0_risk"] = float(args.stage3_weight_stage0_risk)
    if args.disable_fisheye_border_mask:
        config["enable_fisheye_border_mask"] = False
    if args.fisheye_mask_radius_ratio is not None:
        config["fisheye_mask_radius_ratio"] = float(max(0.0, min(1.0, args.fisheye_mask_radius_ratio)))
    if args.fisheye_mask_center_offset_x is not None:
        config["fisheye_mask_center_offset_x"] = int(args.fisheye_mask_center_offset_x)
    if args.fisheye_mask_center_offset_y is not None:
        config["fisheye_mask_center_offset_y"] = int(args.fisheye_mask_center_offset_y)


def resolve_output_dir(video_path: str, output_arg: str = None) -> Path:
    """出力ディレクトリを決定して作成する。"""
    output_dir = Path(output_arg) if output_arg else Path(video_path).parent / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_loader(video_path: str, args, is_front_rear: bool, is_osv: bool):
    """入力モードに応じてローダーを初期化する。"""
    from core.video_loader import VideoLoader, DualVideoLoader, FrontRearVideoLoader

    if is_front_rear:
        loader = FrontRearVideoLoader()
        try:
            loader.load(args.front_video, args.rear_video)
            logger.info(f"前後ストリームを読み込みました: F={args.front_video}, R={args.rear_video}")
            return loader
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"前後魚眼ファイルの読み込みに失敗しました: {e}")
            sys.exit(1)

    if is_osv:
        loader = DualVideoLoader()
        try:
            loader.load(video_path)
            logger.info(f"ステレオストリームを分離しました: L={loader.left_path}, R={loader.right_path}")
            return loader
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"OSVファイルの読み込みに失敗しました: {e}")
            sys.exit(1)

    loader = VideoLoader()
    try:
        loader.load(video_path)
        return loader
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"動画の読み込みに失敗しました: {e}")
        sys.exit(1)


def limit_keyframes(keyframes, max_keyframes: int):
    """スコア上位のキーフレームに制限する。"""
    if not max_keyframes or len(keyframes) <= max_keyframes:
        return keyframes
    keyframes.sort(key=lambda kf: kf.combined_score, reverse=True)
    limited = keyframes[:max_keyframes]
    limited.sort(key=lambda kf: kf.frame_index)
    logger.info(f"上位 {max_keyframes} フレームに制限")
    return limited


def save_frame_image(frame, filepath: Path, fmt: str, jpeg_quality: int) -> bool:
    """画像をフォーマットに応じて保存する。"""
    if fmt == "jpg":
        return write_image(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return write_image(filepath, frame)


def serialize_rig_metadata(meta_obj):
    """メタデータ用のリグ情報を辞書へ変換する。"""
    rig = {"rig_type": getattr(meta_obj, "rig_type", "monocular")}

    calib = getattr(meta_obj, "rig_calibration", None)
    if calib is not None:
        rig["calibration"] = {
            "lens_a": {
                "camera_matrix": calib.lens_a.camera_matrix.tolist(),
                "distortion_coeffs": calib.lens_a.distortion_coeffs.tolist(),
                "image_width": calib.lens_a.image_width,
                "image_height": calib.lens_a.image_height,
                "model": calib.lens_a.model,
            },
            "lens_b": {
                "camera_matrix": calib.lens_b.camera_matrix.tolist(),
                "distortion_coeffs": calib.lens_b.distortion_coeffs.tolist(),
                "image_width": calib.lens_b.image_width,
                "image_height": calib.lens_b.image_height,
                "model": calib.lens_b.model,
            },
            "rotation_ab": calib.rotation_ab.tolist(),
            "translation_ab": calib.translation_ab.tolist(),
            "reprojection_error": float(calib.reprojection_error),
        }

    transforms = getattr(meta_obj, "rig_transforms", None)
    if transforms is not None and getattr(transforms, "matrices", None):
        rig["transforms"] = {k: v.tolist() for k, v in transforms.matrices.items()}

    return rig


def round_json_friendly(value):
    """JSON化前に数値を丸めつつ、辞書/配列は再帰的に処理する。"""
    if isinstance(value, dict):
        return {k: round_json_friendly(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_json_friendly(v) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def run_cli(args):
    """CLIモードでキーフレーム抽出を実行する。"""
    from core.keyframe_selector import KeyframeSelector
    from processing.equirectangular import EquirectangularProcessor
    from processing.mask_processor import MaskProcessor
    from utils.rerun_logger import RerunKeyframeLogger

    video_path, is_front_rear, is_osv = resolve_cli_input(args)
    is_stereo_mode = is_osv or is_front_rear

    if is_osv:
        logger.info("OSV（ステレオ）ファイルを検出しました")
    if is_front_rear:
        logger.info("前後魚眼モードを検出しました")

    # 設定ロード（プリセット → 設定ファイル → CLI引数の順で優先）
    config = load_config(args.config, args.preset)
    apply_cli_overrides(config, args)
    output_dir = resolve_output_dir(video_path, args.output)

    logger.info("=" * 60)
    logger.info("360Split - CLIモード")
    logger.info("=" * 60)
    logger.info(f"入力動画: {video_path}")
    logger.info(f"出力先:   {output_dir}")
    logger.info(f"フォーマット: {config['output_image_format']}")
    if is_osv:
        logger.info("ステレオモード（OSV）: 有効（L/Rペア出力）")
    if is_front_rear:
        logger.info("ステレオモード（Front/Rear）: 有効（F/Rペア出力）")
    if args.equirectangular:
        logger.info("360度 Equirectangular モード: 有効")
    if args.apply_mask:
        logger.info("マスク処理: 有効")
    fisheye_border_mask_enabled = bool(config.get("enable_fisheye_border_mask", True)) and is_stereo_mode
    if fisheye_border_mask_enabled:
        logger.info(
            "魚眼外周マスク: 有効 "
            f"(ratio={config.get('fisheye_mask_radius_ratio', 0.94):.2f}, "
            f"dx={int(config.get('fisheye_mask_center_offset_x', 0))}, "
            f"dy={int(config.get('fisheye_mask_center_offset_y', 0))})"
        )
    if config.get("enable_dynamic_mask_removal", False):
        logger.info(
            "Stage2動体除去: 有効 "
            f"(frames={config.get('dynamic_mask_motion_frames', 3)}, "
            f"th={config.get('dynamic_mask_motion_threshold', 30)}, "
            f"dilate={config.get('dynamic_mask_dilation_size', 5)})"
        )
    logger.info(
        "Stage0/Stage3: "
        f"stage0={'ON' if config.get('enable_stage0_scan', True) else 'OFF'} "
        f"(stride={config.get('stage0_stride', 5)}), "
        f"stage3={'ON' if config.get('enable_stage3_refinement', True) else 'OFF'}"
    )
    logger.info("-" * 60)

    loader = create_loader(video_path, args, is_front_rear, is_osv)

    meta = loader.get_metadata()
    logger.info(f"動画情報: {meta.width}x{meta.height}, "
                f"{meta.fps:.1f}fps, {meta.frame_count}フレーム, "
                f"{meta.duration:.1f}秒")

    # キーフレーム選択
    selector = KeyframeSelector(config)
    rerun_enabled = bool(args.rerun_stream or args.rerun_save)
    rerun_logger = None
    if rerun_enabled:
        rerun_logger = RerunKeyframeLogger(
            app_id="keyframe_check",
            spawn=bool(args.rerun_spawn),
            save_path=args.rerun_save,
            timeline_name="frame",
        )

    _last_logged_pct = -1

    def progress_callback(current, total, message=""):
        nonlocal _last_logged_pct
        pct = int(current / total * 100) if total > 0 else 0
        # 10% 刻みでログ出力（大量出力を防止）
        if pct >= _last_logged_pct + 10 or pct == 100:
            _last_logged_pct = pct
            logger.info(f"進捗: {pct}% {message}")

    logger.info("キーフレーム解析を開始...")
    def frame_log_callback(payload: dict):
        if rerun_logger is None or not rerun_logger.enabled:
            return
        frame_idx = int(payload.get("frame_index", 0))
        frame = payload.get("frame")
        metrics = payload.get("metrics", {})
        rerun_logger.log_frame(
            frame_idx=frame_idx,
            img=frame,
            t_xyz=payload.get("t_xyz"),
            q_wxyz=payload.get("q_wxyz"),
            is_keyframe=bool(payload.get("is_keyframe", False)),
            metrics=metrics,
            points_world=payload.get("points_world"),
        )

    keyframes = selector.select_keyframes(
        loader,
        progress_callback=progress_callback,
        frame_log_callback=frame_log_callback if rerun_enabled else None,
    )

    if not keyframes:
        logger.warning("キーフレームが検出されませんでした。閾値の調整を検討してください。")
        loader.close()
        sys.exit(0)

    logger.info(f"検出キーフレーム数: {len(keyframes)}")
    keyframes = limit_keyframes(keyframes, args.max_keyframes)

    # マスク処理
    mask_processor = MaskProcessor() if args.apply_mask else None
    equirect_processor = EquirectangularProcessor() if args.equirectangular else None

    # キーフレーム出力
    logger.info("キーフレームを出力中...")
    fmt = config["output_image_format"]
    stereo_suffixes = ("_F", "_R") if is_front_rear else ("_L", "_R")

    for i, kf in enumerate(keyframes):
        # ステレオ判定
        if is_stereo_mode:
            # ステレオペア取得
            frame_l, frame_r = loader.get_frame_pair(kf.frame_index)
            if frame_l is None or frame_r is None:
                logger.warning(f"ペアフレーム読み込み失敗: {kf.frame_index}")
                continue
            frames_to_process = [(frame_l, stereo_suffixes[0]), (frame_r, stereo_suffixes[1])]
        else:
            # 単眼フレーム取得
            frame = loader.get_frame(kf.frame_index)
            if frame is None:
                continue
            frames_to_process = [(frame, '')]

        # 各フレーム（L/R または単眼）を処理
        for frame, suffix in frames_to_process:
            # ステレオの場合はスティッチ未処理のため分割処理をスキップ
            if is_stereo_mode:
                # パノラマ画像のみを L/ または R/ フォルダに保存
                output_subdir = output_dir / suffix.strip('_')  # 'L' or 'R'
                output_subdir.mkdir(parents=True, exist_ok=True)

                # ファイル名にサフィックスあり (_L or _R)
                filename = f"keyframe_{kf.frame_index:06d}{suffix}.{fmt}"
                filepath = output_subdir / filename

                saved = save_frame_image(frame, filepath, fmt, config["output_jpeg_quality"])
                if not saved:
                    logger.warning(f"保存失敗（フレーム {kf.frame_index}{suffix}）: {filepath}")
                continue  # ステレオの場合はここで次のフレームへ

            # マスク処理
            if mask_processor and args.apply_mask:
                h, w = frame.shape[:2]
                nadir_mask = mask_processor.create_nadir_mask(w, h)
                frame = mask_processor.apply_mask(frame, nadir_mask)

            # 通常画像の出力
            filename = f"keyframe_{kf.frame_index:06d}.{fmt}"
            filepath = output_dir / filename

            saved = save_frame_image(frame, filepath, fmt, config["output_jpeg_quality"])
            if not saved:
                logger.warning(f"保存失敗（フレーム {kf.frame_index}）: {filepath}")
                continue

            # Cubemap出力
            if args.cubemap and equirect_processor:
                faces = equirect_processor.to_cubemap(frame, config["cubemap_face_size"])
                for face_name, face_img in faces.items():
                    # 方向ごとのフォルダ: cubemap/front/, cubemap/back/ など
                    face_dir = output_dir / "cubemap" / face_name
                    face_dir.mkdir(parents=True, exist_ok=True)

                    # ファイル名: keyframe_NNNNNN_front.jpg（サフィックスあり）
                    face_path = face_dir / f"keyframe_{kf.frame_index:06d}_{face_name}.{fmt}"
                    if not write_image(face_path, face_img):
                        logger.warning(f"Cubemap保存失敗: {face_path}")

        # スコア情報の表示
        suffix_str = " (F/R)" if is_front_rear else (" (L/R)" if is_osv else "")
        logger.info(f"  [{i+1}/{len(keyframes)}] Frame {kf.frame_index:6d}{suffix_str} | "
                    f"Time {kf.timestamp:7.2f}s | Score {kf.combined_score:.3f}")

    # メタデータ出力
    metadata = {
        "video_path": str(Path(video_path).resolve()),
        "total_frames": meta.frame_count,
        "fps": meta.fps,
        "duration": meta.duration,
        "resolution": f"{meta.width}x{meta.height}",
        "rig": serialize_rig_metadata(meta),
        "keyframe_count": len(keyframes),
        "settings": config,
        "keyframes": [
            {
                "frame_index": kf.frame_index,
                "timestamp": round(kf.timestamp, 3),
                "combined_score": round(kf.combined_score, 4),
                "quality_scores": round_json_friendly(kf.quality_scores) if kf.quality_scores else {},
                "geometric_scores": round_json_friendly(kf.geometric_scores) if kf.geometric_scores else {},
                "adaptive_scores": round_json_friendly(kf.adaptive_scores) if kf.adaptive_scores else {},
                "stage3_scores": round_json_friendly(kf.stage3_scores) if kf.stage3_scores else {},
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

    loader.close()


def run_gui():
    """GUIモードでアプリケーションを起動する。"""
    try:
        from PySide6.QtWidgets import QApplication
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
        set_log_level(logging.DEBUG)

    if args.cli or (args.front_video and args.rear_video):
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
