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
import csv
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger, get_logger, set_log_level
from utils.image_io import write_image
from core.visual_odometry.calibration import (
    calibration_from_dict,
    calibration_to_dict,
    load_calibration_xml,
    log_calibration_summary,
    summarize_calibration,
)
from core.visual_odometry.calibration_check import run_calibration_check

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
        help="入力を360度Equirectangular動画として扱う（入力モード指定。出力変換を強制しない）"
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
        "--stage1-grab-threshold",
        type=int,
        default=None,
        help="Stage1でgrab方式を使う最大サンプリング間隔（1以上）"
    )
    parser.add_argument(
        "--stage1-eval-scale",
        type=float,
        default=None,
        help="Stage1品質評価の縮小スケール（0.1-1.0）"
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
        "--split-views",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="OSV出力時にcross5分割画像を出力する（--no-split-viewsで無効化）"
    )
    parser.add_argument("--split-view-size", type=int, default=None, help="cross5分割画像サイズ(px)")
    parser.add_argument("--split-view-hfov", type=float, default=None, help="cross5分割HFOV(deg)")
    parser.add_argument("--split-view-vfov", type=float, default=None, help="cross5分割VFOV(deg)")
    parser.add_argument("--split-cross-yaw-deg", type=float, default=None, help="cross5横方向基準角(deg)")
    parser.add_argument("--split-cross-pitch-deg", type=float, default=None, help="cross5縦方向基準角(deg)")
    parser.add_argument("--split-cross-inward-deg", type=float, default=None, help="cross5内向き調整角(deg)")
    parser.add_argument("--split-inward-up-deg", type=float, default=None, help="cross5 up内向き角(deg)")
    parser.add_argument("--split-inward-down-deg", type=float, default=None, help="cross5 down内向き角(deg)")
    parser.add_argument("--split-inward-left-deg", type=float, default=None, help="cross5 left内向き角(deg)")
    parser.add_argument("--split-inward-right-deg", type=float, default=None, help="cross5 right内向き角(deg)")
    parser.add_argument(
        "--calib-xml",
        type=str,
        default=None,
        help="単眼または代表レンズ用キャリブレーションXML"
    )
    parser.add_argument(
        "--calib-model",
        type=str,
        default=None,
        choices=["auto", "opencv", "fisheye"],
        help="キャリブレーションモデル種別（auto/opencv/fisheye）"
    )
    parser.add_argument(
        "--front-calib-xml",
        type=str,
        default=None,
        help="front/rear入力時のfrontキャリブレーションXML"
    )
    parser.add_argument(
        "--rear-calib-xml",
        type=str,
        default=None,
        help="front/rear入力時のrearキャリブレーションXML"
    )
    parser.add_argument(
        "--calib-check",
        action="store_true",
        default=False,
        help="キャリブレーション検証モード（キーフレーム抽出は実行しない）"
    )
    parser.add_argument(
        "--calib-check-frame",
        type=int,
        default=None,
        help="キャリブレーション検証に使うフレーム番号"
    )
    parser.add_argument(
        "--calib-check-out",
        type=str,
        default=None,
        help="キャリブレーション検証出力ディレクトリ"
    )
    parser.add_argument(
        "--disable-vo",
        action="store_true",
        default=False,
        help="VOを無効化する"
    )
    parser.add_argument(
        "--vo-center-roi-ratio",
        type=float,
        default=None,
        help="VO中心ROI比率（0.2-1.0）"
    )
    parser.add_argument(
        "--vo-max-features",
        type=int,
        default=None,
        help="VOの最大追跡特徴点数"
    )
    parser.add_argument(
        "--vo-downscale-long-edge",
        type=int,
        default=None,
        help="VO入力の長辺縮小ピクセル（速度優先）"
    )
    parser.add_argument(
        "--vo-frame-subsample",
        type=int,
        default=None,
        help="VO計算をnフレームごとに実行（1で従来同等）"
    )
    parser.add_argument(
        "--vo-adaptive-roi",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="VO中心ROIをフロー量に応じて動的調整する（--no-vo-adaptive-roi で無効化）"
    )
    parser.add_argument(
        "--vo-fast-fail-inlier-ratio",
        type=float,
        default=None,
        help="VO早期失敗判定の最小inlier比率（0.0-1.0）"
    )
    parser.add_argument(
        "--vo-step-proxy-clip-px",
        type=float,
        default=None,
        help="VO step_proxy の上限クリップ値（px）"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="詳細ログ出力"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="各ステージ処理時間と ms/frame をログ出力する（性能計測）"
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
    if args.stage1_grab_threshold is not None:
        config["stage1_grab_threshold"] = max(1, int(args.stage1_grab_threshold))
    if args.stage1_eval_scale is not None:
        config["stage1_eval_scale"] = float(max(0.1, min(1.0, args.stage1_eval_scale)))
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
    if args.split_views is not None:
        config["enable_split_views"] = bool(args.split_views)
    if args.split_view_size is not None:
        config["split_view_size"] = int(max(128, args.split_view_size))
    if args.split_view_hfov is not None:
        config["split_view_hfov"] = float(max(1.0, min(179.0, args.split_view_hfov)))
    if args.split_view_vfov is not None:
        config["split_view_vfov"] = float(max(1.0, min(179.0, args.split_view_vfov)))
    if args.split_cross_yaw_deg is not None:
        config["split_cross_yaw_deg"] = float(max(0.0, args.split_cross_yaw_deg))
    if args.split_cross_pitch_deg is not None:
        config["split_cross_pitch_deg"] = float(max(0.0, args.split_cross_pitch_deg))
    if args.split_cross_inward_deg is not None:
        config["split_cross_inward_deg"] = float(max(0.0, args.split_cross_inward_deg))
    if args.split_inward_up_deg is not None:
        config["split_inward_up_deg"] = float(max(0.0, args.split_inward_up_deg))
    if args.split_inward_down_deg is not None:
        config["split_inward_down_deg"] = float(max(0.0, args.split_inward_down_deg))
    if args.split_inward_left_deg is not None:
        config["split_inward_left_deg"] = float(max(0.0, args.split_inward_left_deg))
    if args.split_inward_right_deg is not None:
        config["split_inward_right_deg"] = float(max(0.0, args.split_inward_right_deg))
    if args.calib_xml:
        config["calib_xml"] = str(args.calib_xml)
    if args.calib_model:
        config["calib_model"] = str(args.calib_model).strip().lower()
    if args.front_calib_xml:
        config["front_calib_xml"] = str(args.front_calib_xml)
    if args.rear_calib_xml:
        config["rear_calib_xml"] = str(args.rear_calib_xml)
    if args.disable_vo:
        config["vo_enabled"] = False
    if args.vo_center_roi_ratio is not None:
        config["vo_center_roi_ratio"] = float(max(0.2, min(1.0, args.vo_center_roi_ratio)))
    if args.vo_max_features is not None:
        config["vo_max_features"] = int(max(50, args.vo_max_features))
    if args.vo_downscale_long_edge is not None:
        config["vo_downscale_long_edge"] = int(max(0, args.vo_downscale_long_edge))
    if args.vo_frame_subsample is not None:
        config["vo_frame_subsample"] = int(max(1, args.vo_frame_subsample))
    if args.vo_adaptive_roi is not None:
        config["vo_adaptive_roi_enable"] = bool(args.vo_adaptive_roi)
    if args.vo_fast_fail_inlier_ratio is not None:
        config["vo_fast_fail_inlier_ratio"] = float(max(0.0, min(1.0, args.vo_fast_fail_inlier_ratio)))
    if args.vo_step_proxy_clip_px is not None:
        config["vo_step_proxy_clip_px"] = float(max(0.0, args.vo_step_proxy_clip_px))
    if args.profile:
        config["enable_profile"] = True
        config["stage2_perf_profile"] = True
    if args.equirectangular:
        config["projection_mode"] = "Equirectangular"
    if args.cubemap:
        config["projection_mode"] = "Cubemap"


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


def summarize_vo_diagnostics(frame_metrics_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    reason_counter: Counter = Counter()
    vo_attempted = 0
    vo_valid = 0
    vo_pose_valid = 0
    with_pose = 0
    for rec in frame_metrics_records:
        metrics = rec.get("metrics", {}) if isinstance(rec, dict) else {}
        reason = str(metrics.get("vo_status_reason", "unknown"))
        reason_counter[reason] += 1
        vo_attempted += 1 if float(metrics.get("vo_attempted", 0.0)) > 0.5 else 0
        vo_valid += 1 if float(metrics.get("vo_valid", 0.0)) > 0.5 else 0
        vo_pose_valid += 1 if float(metrics.get("vo_pose_valid", 0.0)) > 0.5 else 0
        with_pose += 1 if isinstance(rec.get("t_xyz"), list) and len(rec.get("t_xyz")) == 3 else 0

    dominant_reason = reason_counter.most_common(1)[0][0] if reason_counter else "unknown"
    valid_ratio = float(vo_valid / vo_attempted) if vo_attempted > 0 else 0.0
    return {
        "total_records": int(len(frame_metrics_records)),
        "vo_attempted_frames": int(vo_attempted),
        "vo_valid_frames": int(vo_valid),
        "vo_valid_ratio": float(valid_ratio),
        "vo_pose_valid_frames": int(vo_pose_valid),
        "trajectory_points": int(with_pose),
        "vo_status_reason_counts": dict(reason_counter),
        "dominant_vo_status_reason": str(dominant_reason),
    }


def write_vo_diagnostics(output_dir: Path, frame_metrics_records: List[Dict[str, Any]]) -> Tuple[Path, Path, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = summarize_vo_diagnostics(frame_metrics_records)
    diagnostics_path = output_dir / "vo_diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    trajectory_path = output_dir / "vo_trajectory.csv"
    with open(trajectory_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "t_x",
                "t_y",
                "t_z",
                "q_w",
                "q_x",
                "q_y",
                "q_z",
                "vo_valid",
                "vo_inlier_ratio",
                "vo_step_proxy",
            ],
        )
        writer.writeheader()
        for rec in sorted(frame_metrics_records, key=lambda x: int(x.get("frame_index", 0))):
            metrics = rec.get("metrics", {}) if isinstance(rec, dict) else {}
            t_xyz = rec.get("t_xyz") if isinstance(rec.get("t_xyz"), list) else [None, None, None]
            q_wxyz = rec.get("q_wxyz") if isinstance(rec.get("q_wxyz"), list) else [None, None, None, None]
            writer.writerow(
                {
                    "frame_idx": int(rec.get("frame_index", 0)),
                    "t_x": t_xyz[0] if len(t_xyz) == 3 else None,
                    "t_y": t_xyz[1] if len(t_xyz) == 3 else None,
                    "t_z": t_xyz[2] if len(t_xyz) == 3 else None,
                    "q_w": q_wxyz[0] if len(q_wxyz) == 4 else None,
                    "q_x": q_wxyz[1] if len(q_wxyz) == 4 else None,
                    "q_y": q_wxyz[2] if len(q_wxyz) == 4 else None,
                    "q_z": q_wxyz[3] if len(q_wxyz) == 4 else None,
                    "vo_valid": float(metrics.get("vo_valid", 0.0)),
                    "vo_inlier_ratio": float(metrics.get("vo_inlier_ratio", 0.0)),
                    "vo_step_proxy": float(metrics.get("vo_step_proxy", 0.0)),
                }
            )
    return diagnostics_path, trajectory_path, diagnostics


def _normalize_path_value(raw_path: object) -> str:
    s = str(raw_path or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    if not s:
        return ""
    if os.name != "nt" and "\\" in s:
        s = s.replace("\\", "/")
    return os.path.expandvars(os.path.expanduser(s))


def _resolve_calibration_path(raw_path: object, search_roots: List[Path]) -> Optional[str]:
    norm = _normalize_path_value(raw_path)
    if not norm:
        return None
    p = Path(norm)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([root / p for root in search_roots])
        candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(p)


def _load_calibrations_for_runtime(
    config: dict,
    meta,
    is_front_rear: bool,
    is_stereo_mode: bool,
) -> dict:
    model_hint = str(config.get("calib_model", "auto") or "auto").strip().lower()
    fallback_size = (int(meta.width), int(meta.height))
    search_roots = [Path.cwd(), PROJECT_ROOT]
    if hasattr(meta, "video_path") and getattr(meta, "video_path", None):
        try:
            search_roots.append(Path(str(meta.video_path)).expanduser().resolve().parent)
        except Exception:
            pass

    mono_xml = _resolve_calibration_path(config.get("calib_xml", ""), search_roots)
    front_xml = _resolve_calibration_path(config.get("front_calib_xml", ""), search_roots) or mono_xml
    rear_xml = _resolve_calibration_path(config.get("rear_calib_xml", ""), search_roots) or mono_xml

    mono_calib = None
    front_calib = None
    rear_calib = None

    if is_front_rear:
        if front_xml:
            front_calib = load_calibration_xml(front_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
            log_calibration_summary(logger, "front calibration", front_calib)
        if rear_xml:
            rear_calib = load_calibration_xml(rear_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
            log_calibration_summary(logger, "rear calibration", rear_calib)
    else:
        if mono_xml:
            mono_calib = load_calibration_xml(mono_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
            log_calibration_summary(logger, "mono calibration", mono_calib)
        if is_stereo_mode:
            if front_xml:
                front_calib = load_calibration_xml(front_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
                log_calibration_summary(logger, "stereo-front calibration", front_calib)
            if rear_xml:
                rear_calib = load_calibration_xml(rear_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
                log_calibration_summary(logger, "stereo-rear calibration", rear_calib)
            if front_calib is None and mono_calib is not None:
                front_calib = mono_calib
            if rear_calib is None and mono_calib is not None:
                rear_calib = mono_calib
        if is_stereo_mode and front_calib is None and rear_calib is None and mono_calib is None:
            logger.warning("stereo input detected without calibration XML: VO will be disabled")

    runtime = {
        "model_hint": model_hint,
        "mono": calibration_to_dict(mono_calib),
        "front": calibration_to_dict(front_calib),
        "rear": calibration_to_dict(rear_calib),
    }
    return runtime


def run_cli(args):
    """CLIモードでキーフレーム抽出を実行する。"""
    from core.keyframe_selector import KeyframeSelector
    from processing.equirectangular import EquirectangularProcessor
    from processing.fisheye_splitter import Cross5FisheyeSplitter, Cross5SplitConfig
    from processing.mask_processor import MaskProcessor
    from processing.target_mask_generator import TargetMaskGenerator
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
    if is_osv:
        logger.info(
            "cross5分割: "
            f"{'ON' if bool(config.get('enable_split_views', True)) else 'OFF'} "
            f"(size={int(config.get('split_view_size', 1600))}, "
            f"hfov={float(config.get('split_view_hfov', 80.0)):.1f}, "
            f"vfov={float(config.get('split_view_vfov', 80.0)):.1f}, "
            f"inward={float(config.get('split_cross_inward_deg', 10.0)):.1f}, "
            f"u={float(config.get('split_inward_up_deg', 25.0)):.1f}, "
            f"d={float(config.get('split_inward_down_deg', 25.0)):.1f}, "
            f"l={float(config.get('split_inward_left_deg', 25.0)):.1f}, "
            f"r={float(config.get('split_inward_right_deg', 25.0)):.1f})"
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
    try:
        setattr(meta, "video_path", str(video_path))
    except Exception:
        pass
    logger.info(f"動画情報: {meta.width}x{meta.height}, "
                f"{meta.fps:.1f}fps, {meta.frame_count}フレーム, "
                f"{meta.duration:.1f}秒")

    calibration_runtime = _load_calibrations_for_runtime(
        config=config,
        meta=meta,
        is_front_rear=is_front_rear,
        is_stereo_mode=is_stereo_mode,
    )
    config["calibration_runtime"] = calibration_runtime
    if not is_stereo_mode:
        meta.monocular_calibration = calibration_runtime.get("mono")

    if args.calib_check:
        check_out_dir = Path(args.calib_check_out) if args.calib_check_out else output_dir / "calib_check"
        check_out_dir.mkdir(parents=True, exist_ok=True)
        report = run_calibration_check(
            out_dir=str(check_out_dir),
            video_path=None if is_front_rear else video_path,
            calib=None if is_front_rear else calibration_from_dict(calibration_runtime.get("mono")),
            front_video_path=args.front_video if is_front_rear else None,
            rear_video_path=args.rear_video if is_front_rear else None,
            front_calib=calibration_from_dict(calibration_runtime.get("front")) if is_front_rear else None,
            rear_calib=calibration_from_dict(calibration_runtime.get("rear")) if is_front_rear else None,
            frame_index=args.calib_check_frame,
            roi_ratio=float(config.get("vo_center_roi_ratio", 0.6)),
            logger=logger,
        )
        logger.info(f"キャリブレーション検証完了: {report.get('report_path')}")
        loader.close()
        return

    # キーフレーム選択
    selector = KeyframeSelector(config)
    vo_runtime = selector.get_vo_runtime_status(is_paired=bool(getattr(loader, "is_paired", False)))
    if vo_runtime["enabled"]:
        logger.info(
            f"VO runtime: enabled (reason={vo_runtime['reason']}, calibration_loaded={vo_runtime['calibration_loaded']})"
        )
    else:
        logger.warning(
            f"VO runtime: disabled (reason={vo_runtime['reason']}, calibration_loaded={vo_runtime['calibration_loaded']})"
        )
    rerun_enabled = bool(args.rerun_stream or args.rerun_save)
    rerun_logger = None
    frame_metrics_records = []
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
        frame_idx = int(payload.get("frame_index", 0))
        metrics = payload.get("metrics", {})
        frame_metrics_records.append(
            {
                "frame_index": frame_idx,
                "is_keyframe": bool(payload.get("is_keyframe", False)),
                "t_xyz": round_json_friendly(payload.get("t_xyz")) if payload.get("t_xyz") is not None else None,
                "q_wxyz": round_json_friendly(payload.get("q_wxyz")) if payload.get("q_wxyz") is not None else None,
                "metrics": round_json_friendly(metrics) if isinstance(metrics, dict) else {},
            }
        )
        if rerun_logger is None or not rerun_logger.enabled:
            return
        frame = payload.get("frame")
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
        frame_log_callback=frame_log_callback,
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
    mask_output_dirname = str(config.get("mask_output_dirname", "masks") or "masks")
    mask_add_suffix = bool(config.get("mask_add_suffix", True))
    mask_suffix = str(config.get("mask_suffix", "_mask") or "_mask")
    mask_output_format = str(config.get("mask_output_format", "same") or "same").lower()
    masks_root = output_dir / mask_output_dirname
    fisheye_mask_export_enabled = bool(config.get("enable_dynamic_mask_removal", False)) and is_stereo_mode

    target_mask_generator = None
    if fisheye_mask_export_enabled:
        try:
            inpaint_hook = None
            if bool(config.get("dynamic_mask_inpaint_enabled", False)) and str(config.get("dynamic_mask_inpaint_module", "")).strip():
                try:
                    mod = __import__(str(config.get("dynamic_mask_inpaint_module")).strip(), fromlist=["inpaint_frame"])
                    hook = getattr(mod, "inpaint_frame", None)
                    if callable(hook):
                        inpaint_hook = hook
                except Exception as e:
                    logger.warning(f"CLIマスク出力: インペイントモジュール読み込み失敗: {e}")
            target_mask_generator = TargetMaskGenerator(
                yolo_model_path=str(config.get("yolo_model_path", "yolo26n-seg.pt")),
                sam_model_path=str(config.get("sam_model_path", "sam3_t.pt")),
                confidence_threshold=float(config.get("confidence_threshold", 0.25)),
                device=str(config.get("detection_device", "auto")),
                enable_motion_detection=bool(config.get("dynamic_mask_use_motion_diff", True)),
                motion_history_frames=max(2, int(config.get("dynamic_mask_motion_frames", 3))),
                motion_threshold=int(config.get("dynamic_mask_motion_threshold", 30)),
                motion_mask_dilation_size=max(0, int(config.get("dynamic_mask_dilation_size", 5))),
                enable_mask_inpaint=bool(config.get("dynamic_mask_inpaint_enabled", False)),
                inpaint_hook=inpaint_hook,
            )
            logger.info("CLIマスク出力: 魚眼マスク生成を有効化しました")
        except Exception as e:
            logger.warning(f"CLIマスク出力: 初期化失敗のため無効化します: {e}")
            target_mask_generator = None
            fisheye_mask_export_enabled = False

    splitters = {}
    split_views_enabled = bool(config.get("enable_split_views", True))
    split_cfg = Cross5SplitConfig(
        size=int(config.get("split_view_size", 1600)),
        hfov=float(config.get("split_view_hfov", 80.0)),
        vfov=float(config.get("split_view_vfov", 80.0)),
        cross_yaw_deg=float(config.get("split_cross_yaw_deg", 50.5)),
        cross_pitch_deg=float(config.get("split_cross_pitch_deg", 50.5)),
        cross_inward_deg=float(config.get("split_cross_inward_deg", 10.0)),
        inward_up_deg=float(config.get("split_inward_up_deg", 25.0)),
        inward_down_deg=float(config.get("split_inward_down_deg", 25.0)),
        inward_left_deg=float(config.get("split_inward_left_deg", 25.0)),
        inward_right_deg=float(config.get("split_inward_right_deg", 25.0)),
    )
    if is_osv and split_views_enabled:
        calib_l = calibration_runtime.get("front") or calibration_runtime.get("mono")
        calib_r = calibration_runtime.get("rear") or calibration_runtime.get("mono")
        try:
            if calib_l is not None:
                splitters["_L"] = Cross5FisheyeSplitter(calib_l, cfg=split_cfg)
            if calib_r is not None:
                splitters["_R"] = Cross5FisheyeSplitter(calib_r, cfg=split_cfg)
        except Exception as e:
            splitters = {}
            logger.warning(f"cross5分割: 初期化に失敗したためスキップします: {e}")
        if not splitters:
            logger.warning("cross5分割: 利用可能なキャリブレーションがないためスキップします")
    elif is_osv and not split_views_enabled:
        logger.info("cross5分割出力: 無効")

    # キーフレーム出力
    logger.info("キーフレームを出力中...")
    fmt = config["output_image_format"]
    stereo_suffixes = ("_F", "_R") if is_front_rear else ("_L", "_R")
    stereo_images_root = output_dir / "images" if is_stereo_mode else output_dir
    mask_histories = {stereo_suffixes[0]: deque(maxlen=max(2, int(config.get("dynamic_mask_motion_frames", 3)))),
                      stereo_suffixes[1]: deque(maxlen=max(2, int(config.get("dynamic_mask_motion_frames", 3))))}
    classes_for_detection = (
        list(config.get("dynamic_mask_target_classes", ["人物", "人", "自転車", "バイク", "車両", "動物"]))
        if bool(config.get("dynamic_mask_use_yolo_sam", True))
        else []
    )

    def build_split_mask_path(split_image_path: Path) -> Path:
        stem = f"{split_image_path.stem}{mask_suffix}" if mask_add_suffix else split_image_path.stem
        if mask_output_format == "same":
            ext = split_image_path.suffix
        else:
            ext = f".{mask_output_format.lstrip('.')}"
        return masks_root / f"{stem}{ext}"

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
                output_subdir = stereo_images_root / suffix.strip('_')  # 'L' or 'R'
                output_subdir.mkdir(parents=True, exist_ok=True)

                # ファイル名にサフィックスあり (_L or _R)
                filename = f"keyframe_{kf.frame_index:06d}{suffix}.{fmt}"
                filepath = output_subdir / filename

                saved = save_frame_image(frame, filepath, fmt, config["output_jpeg_quality"])
                if not saved:
                    logger.warning(f"保存失敗（フレーム {kf.frame_index}{suffix}）: {filepath}")
                    continue

                fisheye_binary_mask = None
                if fisheye_mask_export_enabled and target_mask_generator is not None:
                    try:
                        history = mask_histories.setdefault(
                            suffix, deque(maxlen=max(2, int(config.get("dynamic_mask_motion_frames", 3))))
                        )
                        history.append(frame.copy())
                        fisheye_binary_mask = target_mask_generator.generate_mask(
                            frame,
                            classes_for_detection,
                            motion_frames=list(history),
                        ).astype(np.uint8)
                        if bool(config.get("enable_fisheye_border_mask", True)):
                            h_m, w_m = fisheye_binary_mask.shape[:2]
                            valid_mask = np.zeros((h_m, w_m), dtype=np.uint8)
                            radius = int(min(w_m, h_m) * 0.5 * float(np.clip(config.get("fisheye_mask_radius_ratio", 0.94), 0.0, 1.0)))
                            cx = int(np.clip((w_m // 2) + int(config.get("fisheye_mask_center_offset_x", 0)), 0, max(w_m - 1, 0)))
                            cy = int(np.clip((h_m // 2) + int(config.get("fisheye_mask_center_offset_y", 0)), 0, max(h_m - 1, 0)))
                            if radius > 0:
                                cv2.circle(valid_mask, (cx, cy), radius, 255, -1)
                                fisheye_binary_mask = cv2.bitwise_and(fisheye_binary_mask, valid_mask)
                        fisheye_mask_path = TargetMaskGenerator.build_mask_path(
                            image_path=filepath,
                            images_root=stereo_images_root,
                            masks_root=masks_root,
                            add_suffix=mask_add_suffix,
                            suffix=mask_suffix,
                            mask_ext=mask_output_format,
                            flatten_stereo_lr=True,
                        )
                        fisheye_mask_path.parent.mkdir(parents=True, exist_ok=True)
                        if not write_image(fisheye_mask_path, fisheye_binary_mask):
                            logger.warning(f"魚眼マスク保存失敗: {fisheye_mask_path}")
                    except Exception as e:
                        logger.warning(f"魚眼マスク生成失敗（フレーム {kf.frame_index}{suffix}）: {e}")
                        fisheye_binary_mask = None

                splitter = splitters.get(suffix) if is_osv else None
                if splitter is not None:
                    try:
                        split_views = splitter.split_image_with_valid_mask(frame)
                        projected_masks = splitter.project_mask(fisheye_binary_mask) if fisheye_binary_mask is not None else {}
                        for view_name, (view_img, _valid_mask) in split_views.items():
                            split_dir = stereo_images_root / f"{suffix.strip('_')}_{view_name}"
                            split_dir.mkdir(parents=True, exist_ok=True)
                            split_image_path = split_dir / f"keyframe_{kf.frame_index:06d}{suffix}_{view_name}.{fmt}"
                            if not save_frame_image(view_img, split_image_path, fmt, config["output_jpeg_quality"]):
                                logger.warning(f"分割画像保存失敗: {split_image_path}")
                                continue
                            if fisheye_binary_mask is None:
                                continue
                            split_mask = projected_masks.get(view_name)
                            if split_mask is None:
                                continue
                            split_mask_path = build_split_mask_path(split_image_path)
                            split_mask_path.parent.mkdir(parents=True, exist_ok=True)
                            if not write_image(split_mask_path, split_mask):
                                logger.warning(f"分割マスク保存失敗: {split_mask_path}")
                    except Exception as e:
                        logger.warning(f"cross5分割出力失敗（フレーム {kf.frame_index}{suffix}）: {e}")
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
        "calibration": {
            "mono": summarize_calibration(calibration_from_dict(calibration_runtime.get("mono"))),
            "front": summarize_calibration(calibration_from_dict(calibration_runtime.get("front"))),
            "rear": summarize_calibration(calibration_from_dict(calibration_runtime.get("rear"))),
        },
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
    if frame_metrics_records:
        frame_metrics_path = output_dir / "frame_metrics.json"
        with open(frame_metrics_path, "w", encoding="utf-8") as f:
            json.dump({"records": frame_metrics_records}, f, ensure_ascii=False, indent=2)
        logger.info(f"フレームメトリクス: {frame_metrics_path}")
        vo_diag_path, vo_traj_path, vo_diag = write_vo_diagnostics(output_dir, frame_metrics_records)
        logger.info(f"VO diagnostics: {vo_diag_path}")
        logger.info(f"VO trajectory: {vo_traj_path}")
        logger.info(
            "VO summary: "
            f"attempted={vo_diag['vo_attempted_frames']}, "
            f"valid={vo_diag['vo_valid_frames']}, "
            f"valid_ratio={vo_diag['vo_valid_ratio']:.3f}, "
            f"pose_valid={vo_diag['vo_pose_valid_frames']}, "
            f"reason={vo_diag['dominant_vo_status_reason']}"
        )

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
