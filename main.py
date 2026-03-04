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
import shutil
import subprocess
import uuid
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
from core.pose import run_pose_pipeline
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


def _parse_opk_seed_text(value: Any) -> List[float]:
    parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("OPK seed must be 'OMEGA,PHI,KAPPA' (3 values)")
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid OPK seed: {e}") from e


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
        "--analysis-run-id",
        type=str,
        default=None,
        help="解析実行ID（resumeや再現実行用）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="既存 analysis-run-id の中間成果を再利用して再開する"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="正常終了時も stage temp artifacts を保持する"
    )
    parser.add_argument(
        "--colmap-format",
        action="store_true",
        default=False,
        help="COLMAP互換ディレクトリ（colmap/）を追加出力する"
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
        "--opencv-threads",
        type=int,
        default=None,
        help="OpenCVスレッド数（0=auto）"
    )
    parser.add_argument(
        "--stage1-process-workers",
        type=int,
        default=None,
        help="Stage1品質計算プロセス数（0=auto）"
    )
    parser.add_argument(
        "--stage1-prefetch-size",
        type=int,
        default=None,
        help="Stage1先読みキューサイズ（1以上）"
    )
    parser.add_argument(
        "--stage1-metrics-batch-size",
        type=int,
        default=None,
        help="Stage1品質計算バッチサイズ（1以上）"
    )
    parser.add_argument(
        "--stage1-gpu-batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stage1品質計算でGPUバッチ処理を有効化/無効化"
    )
    parser.add_argument(
        "--darwin-capture-backend",
        type=str,
        choices=["auto", "avfoundation", "ffmpeg"],
        default=None,
        help="macOSのVideoCaptureバックエンド"
    )
    parser.add_argument(
        "--mps-min-pixels",
        type=int,
        default=None,
        help="MPS経路を使う最小画素数（1以上）"
    )
    parser.add_argument(
        "--quality-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stage1品質フィルタ（ROI+分位点正規化）を有効化/無効化"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="品質スコアしきい値（0.0-1.0）"
    )
    parser.add_argument(
        "--quality-roi",
        type=str,
        default=None,
        help="品質評価ROI（例: circle:0.40 / rect:0.60）"
    )
    parser.add_argument(
        "--quality-abs-laplacian-min",
        type=float,
        default=None,
        help="品質フィルタの絶対ラプラシアン下限"
    )
    parser.add_argument(
        "--quality-debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="品質フィルタのデバッグ統計ログを有効化/無効化"
    )
    parser.add_argument(
        "--stage1-lr-merge-mode",
        type=str,
        choices=["asymmetric_sky_v1", "strict_min"],
        default=None,
        help="Stage1 LR統合方式（asymmetric_sky_v1 / strict_min）"
    )
    parser.add_argument(
        "--stage1-lr-asym-weak-floor",
        type=float,
        default=None,
        help="Stage1 LR非対称統合時の弱レンズ品質下限（0.0-1.0）"
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
        "--vo-essential-method",
        type=str,
        choices=["auto", "ransac", "magsac"],
        default=None,
        help="VO Essential Matrix 推定法"
    )
    parser.add_argument(
        "--vo-subpixel-refine",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="VO追跡点をサブピクセル補正する（--no-vo-subpixel-refine で無効化）"
    )
    parser.add_argument(
        "--vo-adaptive-subsample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="VOサブサンプル間隔を動き量に応じて動的調整する"
    )
    parser.add_argument(
        "--vo-subsample-min",
        type=int,
        default=None,
        help="VO動的サブサンプル時の最小間引き間隔"
    )
    parser.add_argument(
        "--pose-backend",
        type=str,
        choices=["vo", "colmap"],
        default=None,
        help="姿勢推定バックエンド（vo / colmap）"
    )
    parser.add_argument(
        "--colmap-path",
        type=str,
        default=None,
        help="COLMAP実行ファイルパス（デフォルト: colmap）"
    )
    parser.add_argument(
        "--colmap-workspace",
        type=str,
        default=None,
        help="COLMAP作業ディレクトリ"
    )
    parser.add_argument(
        "--colmap-db-path",
        type=str,
        default=None,
        help="COLMAP database.db パス"
    )
    parser.add_argument(
        "--colmap-pipeline-mode",
        type=str,
        choices=["minimal_v1", "legacy"],
        default=None,
        help="COLMAPキーフレーム抽出パイプライン（minimal_v1 / legacy）"
    )
    parser.add_argument(
        "--colmap-keyframe-policy",
        type=str,
        choices=["legacy", "stage2_relaxed", "stage1_only"],
        default=None,
        help="COLMAP向けキーフレームポリシー"
    )
    parser.add_argument(
        "--colmap-selection-profile",
        type=str,
        choices=["legacy", "no_vo_coverage"],
        default=None,
        help="COLMAP向け選抜プロファイル（legacy / no_vo_coverage）"
    )
    parser.add_argument(
        "--colmap-keyframe-target-mode",
        type=str,
        choices=["fixed", "auto"],
        default=None,
        help="COLMAP投入枚数の決定方式（fixed / auto）"
    )
    parser.add_argument(
        "--colmap-keyframe-target-min",
        type=int,
        default=None,
        help="COLMAP投入キーフレーム下限（不足時は補完）"
    )
    parser.add_argument(
        "--colmap-keyframe-target-max",
        type=int,
        default=None,
        help="COLMAP投入キーフレーム上限（超過時は均等間引き）"
    )
    parser.add_argument(
        "--colmap-nms-window-sec",
        type=float,
        default=None,
        help="COLMAP向けNMS窓（秒）"
    )
    parser.add_argument(
        "--colmap-enable-stage0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="COLMAPショートカット時もStage0軽量走査を有効化/無効化"
    )
    parser.add_argument(
        "--colmap-motion-aware-selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="COLMAP向けにtime+motion選択を有効化/無効化"
    )
    parser.add_argument(
        "--colmap-nms-motion-window-ratio",
        type=float,
        default=None,
        help="COLMAP motion-aware NMSの移動量窓倍率（median step比）"
    )
    parser.add_argument(
        "--colmap-stage1-adaptive-threshold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="COLMAP向けStage1適応しきい値を有効化/無効化"
    )
    parser.add_argument(
        "--colmap-stage1-min-candidates-per-bin",
        type=int,
        default=None,
        help="COLMAP向けStage1時間ビンあたり最小候補数"
    )
    parser.add_argument(
        "--colmap-stage1-max-candidates",
        type=int,
        default=None,
        help="COLMAP向けStage1候補数上限"
    )
    parser.add_argument(
        "--colmap-stage2-entry-budget",
        type=int,
        default=None,
        help="COLMAP向けStage2投入前の入口予算（Stage1.5）"
    )
    parser.add_argument(
        "--colmap-stage2-entry-min-gap",
        type=int,
        default=None,
        help="COLMAP向けStage2投入前の最小フレーム間隔（Stage1.5）"
    )
    parser.add_argument(
        "--colmap-diversity-ssim-threshold",
        type=float,
        default=None,
        help="COLMAP向け多様性判定SSIMしきい値（高いほど重複を除外）"
    )
    parser.add_argument(
        "--colmap-diversity-phash-hamming",
        type=int,
        default=None,
        help="COLMAP向け多様性判定pHashハミング距離しきい値"
    )
    parser.add_argument(
        "--colmap-final-target-policy",
        type=str,
        choices=["soft_auto", "fixed"],
        default=None,
        help="COLMAP最終ターゲット方針（soft_auto / fixed）"
    )
    parser.add_argument(
        "--colmap-final-soft-min",
        type=int,
        default=None,
        help="COLMAP soft_auto時の下限目安"
    )
    parser.add_argument(
        "--colmap-final-soft-max",
        type=int,
        default=None,
        help="COLMAP soft_auto時の上限目安"
    )
    parser.add_argument(
        "--colmap-no-supplement-on-low-quality",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="COLMAP最終補充時に低品質フレーム補充を禁止（--no-colmap-no-supplement-on-low-quality で解除）"
    )
    parser.add_argument(
        "--colmap-rig-policy",
        type=str,
        choices=["off", "lr_opk"],
        default=None,
        help="COLMAP rig 方針（off / lr_opk）"
    )
    parser.add_argument(
        "--colmap-rig-seed-opk",
        type=_parse_opk_seed_text,
        default=None,
        help="COLMAP rig 初期姿勢 OPK seed（例: 0,0,180）"
    )
    parser.add_argument(
        "--colmap-workspace-scope",
        type=str,
        choices=["shared", "run_scoped"],
        default=None,
        help="COLMAP workspace のスコープ"
    )
    parser.add_argument(
        "--colmap-reuse-db",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="COLMAP database.db を再利用する"
    )
    parser.add_argument(
        "--colmap-analysis-mask-profile",
        type=str,
        choices=["legacy", "colmap_safe"],
        default=None,
        help="COLMAP向け解析時マスクプロファイル"
    )
    parser.add_argument(
        "--colmap-sparse-model-pick-policy",
        type=str,
        choices=["registered_then_coverage", "coverage_then_registered", "latest_legacy"],
        default=None,
        help="COLMAP sparse モデルの採用方針"
    )
    parser.add_argument(
        "--colmap-input-subset-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="COLMAP入力に幾何縮退ゲート付きサブセットを使う"
    )
    parser.add_argument(
        "--colmap-input-gate-method",
        type=str,
        choices=["homography_degeneracy_v1", "off"],
        default=None,
        help="COLMAP入力サブセットのゲート方式"
    )
    parser.add_argument(
        "--colmap-input-gate-strength",
        type=str,
        choices=["weak", "medium", "strong"],
        default=None,
        help="COLMAP入力サブセットのゲート強度"
    )
    parser.add_argument(
        "--pose-export-format",
        type=str,
        choices=["internal", "metashape"],
        default=None,
        help="姿勢CSVエクスポート形式"
    )
    parser.add_argument(
        "--pose-select-translation-threshold",
        type=float,
        default=None,
        help="必要画像抽出の並進しきい値（正規化ステップ）"
    )
    parser.add_argument(
        "--pose-select-rotation-threshold-deg",
        type=float,
        default=None,
        help="必要画像抽出の回転しきい値（deg）"
    )
    parser.add_argument(
        "--pose-select-min-observations",
        type=int,
        default=None,
        help="必要画像抽出の最小観測点数"
    )
    parser.add_argument(
        "--pose-select-enable-translation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="必要画像抽出で並進条件を有効化/無効化"
    )
    parser.add_argument(
        "--pose-select-enable-rotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="必要画像抽出で回転条件を有効化/無効化"
    )
    parser.add_argument(
        "--pose-select-enable-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="必要画像抽出で観測点数条件を有効化/無効化"
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
    if args.analysis_run_id:
        config["analysis_run_id"] = str(args.analysis_run_id).strip()
    if args.resume:
        config["resume_enabled"] = True
    if args.keep_temp:
        config["keep_temp_on_success"] = True
    if args.colmap_format:
        config["colmap_format"] = True
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
    if args.opencv_threads is not None:
        config["opencv_thread_count"] = int(max(0, args.opencv_threads))
    if args.stage1_process_workers is not None:
        config["stage1_process_workers"] = int(max(0, args.stage1_process_workers))
    if args.stage1_prefetch_size is not None:
        config["stage1_prefetch_size"] = int(max(1, args.stage1_prefetch_size))
    if args.stage1_metrics_batch_size is not None:
        config["stage1_metrics_batch_size"] = int(max(1, args.stage1_metrics_batch_size))
    if args.stage1_gpu_batch is not None:
        config["stage1_gpu_batch_enabled"] = bool(args.stage1_gpu_batch)
    if args.darwin_capture_backend is not None:
        config["darwin_capture_backend"] = str(args.darwin_capture_backend).strip().lower()
    if args.mps_min_pixels is not None:
        config["mps_min_pixels"] = int(max(1, args.mps_min_pixels))
    if args.quality_filter is not None:
        config["quality_filter_enabled"] = bool(args.quality_filter)
    if args.quality_threshold is not None:
        config["quality_threshold"] = float(max(0.0, min(1.0, args.quality_threshold)))
    if args.quality_roi:
        roi_text = str(args.quality_roi).strip().lower()
        if ":" in roi_text:
            mode, ratio_text = roi_text.split(":", 1)
            try:
                ratio = float(ratio_text.strip())
            except (TypeError, ValueError):
                ratio = float(config.get("quality_roi_ratio", 0.40))
            mode = mode.strip()
        else:
            mode = roi_text
            ratio = float(config.get("quality_roi_ratio", 0.40))
        if mode not in {"circle", "rect"}:
            mode = "circle"
        config["quality_roi_mode"] = mode
        config["quality_roi_ratio"] = float(max(0.05, min(1.0, ratio)))
    if args.quality_abs_laplacian_min is not None:
        config["quality_abs_laplacian_min"] = float(max(0.0, args.quality_abs_laplacian_min))
    if args.quality_debug is not None:
        config["quality_debug"] = bool(args.quality_debug)
    if args.stage1_lr_merge_mode is not None:
        config["stage1_lr_merge_mode"] = str(args.stage1_lr_merge_mode).strip().lower()
    if args.stage1_lr_asym_weak_floor is not None:
        config["stage1_lr_asym_weak_floor"] = float(max(0.0, min(1.0, args.stage1_lr_asym_weak_floor)))
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
    if args.vo_essential_method is not None:
        config["vo_essential_method"] = str(args.vo_essential_method).strip().lower()
    if args.vo_subpixel_refine is not None:
        config["vo_subpixel_refine"] = bool(args.vo_subpixel_refine)
    if args.vo_adaptive_subsample is not None:
        config["vo_adaptive_subsample"] = bool(args.vo_adaptive_subsample)
    if args.vo_subsample_min is not None:
        config["vo_subsample_min"] = int(max(1, args.vo_subsample_min))
    if args.pose_backend is not None:
        config["pose_backend"] = str(args.pose_backend).strip().lower()
    if args.colmap_path is not None:
        config["colmap_path"] = str(args.colmap_path).strip()
    if args.colmap_workspace is not None:
        config["colmap_workspace"] = str(args.colmap_workspace).strip()
    if args.colmap_db_path is not None:
        config["colmap_db_path"] = str(args.colmap_db_path).strip()
    if args.colmap_pipeline_mode is not None:
        config["colmap_pipeline_mode"] = str(args.colmap_pipeline_mode).strip().lower()
    if args.colmap_keyframe_policy is not None:
        config["colmap_keyframe_policy"] = str(args.colmap_keyframe_policy).strip().lower()
    if args.colmap_selection_profile is not None:
        config["colmap_selection_profile"] = str(args.colmap_selection_profile).strip().lower()
    if args.colmap_keyframe_target_mode is not None:
        config["colmap_keyframe_target_mode"] = str(args.colmap_keyframe_target_mode).strip().lower()
    if args.colmap_keyframe_target_min is not None:
        config["colmap_keyframe_target_min"] = int(max(1, args.colmap_keyframe_target_min))
    if args.colmap_keyframe_target_max is not None:
        config["colmap_keyframe_target_max"] = int(max(1, args.colmap_keyframe_target_max))
    if args.colmap_nms_window_sec is not None:
        config["colmap_nms_window_sec"] = float(max(0.01, args.colmap_nms_window_sec))
    if args.colmap_enable_stage0 is not None:
        config["colmap_enable_stage0"] = bool(args.colmap_enable_stage0)
    if args.colmap_motion_aware_selection is not None:
        config["colmap_motion_aware_selection"] = bool(args.colmap_motion_aware_selection)
    if args.colmap_nms_motion_window_ratio is not None:
        config["colmap_nms_motion_window_ratio"] = float(max(0.0, args.colmap_nms_motion_window_ratio))
    if args.colmap_stage1_adaptive_threshold is not None:
        config["colmap_stage1_adaptive_threshold"] = bool(args.colmap_stage1_adaptive_threshold)
    if args.colmap_stage1_min_candidates_per_bin is not None:
        config["colmap_stage1_min_candidates_per_bin"] = int(max(0, args.colmap_stage1_min_candidates_per_bin))
    if args.colmap_stage1_max_candidates is not None:
        config["colmap_stage1_max_candidates"] = int(max(1, args.colmap_stage1_max_candidates))
    if args.colmap_stage2_entry_budget is not None:
        config["colmap_stage2_entry_budget"] = int(max(1, args.colmap_stage2_entry_budget))
    if args.colmap_stage2_entry_min_gap is not None:
        config["colmap_stage2_entry_min_gap"] = int(max(0, args.colmap_stage2_entry_min_gap))
    if args.colmap_diversity_ssim_threshold is not None:
        config["colmap_diversity_ssim_threshold"] = float(max(0.0, min(1.0, args.colmap_diversity_ssim_threshold)))
    if args.colmap_diversity_phash_hamming is not None:
        config["colmap_diversity_phash_hamming"] = int(max(0, args.colmap_diversity_phash_hamming))
    if args.colmap_final_target_policy is not None:
        config["colmap_final_target_policy"] = str(args.colmap_final_target_policy).strip().lower()
    if args.colmap_final_soft_min is not None:
        config["colmap_final_soft_min"] = int(max(1, args.colmap_final_soft_min))
    if args.colmap_final_soft_max is not None:
        config["colmap_final_soft_max"] = int(max(1, args.colmap_final_soft_max))
    if args.colmap_no_supplement_on_low_quality is not None:
        config["colmap_no_supplement_on_low_quality"] = bool(args.colmap_no_supplement_on_low_quality)
    if args.colmap_rig_policy is not None:
        config["colmap_rig_policy"] = str(args.colmap_rig_policy).strip().lower()
    if args.colmap_rig_seed_opk is not None:
        config["colmap_rig_seed_opk_deg"] = [float(args.colmap_rig_seed_opk[0]), float(args.colmap_rig_seed_opk[1]), float(args.colmap_rig_seed_opk[2])]
    if args.colmap_workspace_scope is not None:
        config["colmap_workspace_scope"] = str(args.colmap_workspace_scope).strip().lower()
    if args.colmap_reuse_db is not None:
        config["colmap_reuse_db"] = bool(args.colmap_reuse_db)
    if args.colmap_analysis_mask_profile is not None:
        config["colmap_analysis_mask_profile"] = str(args.colmap_analysis_mask_profile).strip().lower()
    if args.colmap_sparse_model_pick_policy is not None:
        config["colmap_sparse_model_pick_policy"] = str(args.colmap_sparse_model_pick_policy).strip().lower()
    if args.colmap_input_subset_enabled is not None:
        config["colmap_input_subset_enabled"] = bool(args.colmap_input_subset_enabled)
    if args.colmap_input_gate_method is not None:
        config["colmap_input_gate_method"] = str(args.colmap_input_gate_method).strip().lower()
    if args.colmap_input_gate_strength is not None:
        config["colmap_input_gate_strength"] = str(args.colmap_input_gate_strength).strip().lower()
    if args.pose_export_format is not None:
        config["pose_export_format"] = str(args.pose_export_format).strip().lower()
    if args.pose_select_translation_threshold is not None:
        config["pose_select_translation_threshold"] = float(max(0.0, args.pose_select_translation_threshold))
    if args.pose_select_rotation_threshold_deg is not None:
        config["pose_select_rotation_threshold_deg"] = float(max(0.0, args.pose_select_rotation_threshold_deg))
    if args.pose_select_min_observations is not None:
        config["pose_select_min_observations"] = int(max(0, args.pose_select_min_observations))
    if args.pose_select_enable_translation is not None:
        config["pose_select_enable_translation"] = bool(args.pose_select_enable_translation)
    if args.pose_select_enable_rotation is not None:
        config["pose_select_enable_rotation"] = bool(args.pose_select_enable_rotation)
    if args.pose_select_enable_observations is not None:
        config["pose_select_enable_observations"] = bool(args.pose_select_enable_observations)
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


def create_loader(video_path: str, args, is_front_rear: bool, is_osv: bool, config: Optional[dict] = None):
    """入力モードに応じてローダーを初期化する。"""
    from core.video_loader import VideoLoader, DualVideoLoader, FrontRearVideoLoader

    cfg = dict(config or {})
    backend_pref = str(cfg.get("darwin_capture_backend", cfg.get("DARWIN_CAPTURE_BACKEND", "auto")) or "auto")

    if is_front_rear:
        loader = FrontRearVideoLoader(backend_preference=backend_pref)
        try:
            loader.load(args.front_video, args.rear_video)
            logger.info(f"前後ストリームを読み込みました: F={args.front_video}, R={args.rear_video}")
            return loader
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"前後魚眼ファイルの読み込みに失敗しました: {e}")
            sys.exit(1)

    if is_osv:
        loader = DualVideoLoader(backend_preference=backend_pref)
        try:
            loader.load(video_path)
            logger.info(f"ステレオストリームを分離しました: L={loader.left_path}, R={loader.right_path}")
            return loader
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"OSVファイルの読み込みに失敗しました: {e}")
            sys.exit(1)

    loader = VideoLoader(config=cfg)
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


def _extract_intrinsics(
    calib_dict: Optional[Dict[str, Any]],
    *,
    fallback_width: int,
    fallback_height: int,
) -> Tuple[float, float, float, float]:
    if calib_dict and isinstance(calib_dict, dict):
        try:
            k = np.asarray(calib_dict.get("camera_matrix", []), dtype=np.float64).reshape(3, 3)
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
            if fx > 0.0 and fy > 0.0:
                return fx, fy, cx, cy
        except Exception:
            pass
    w = max(1, int(fallback_width))
    h = max(1, int(fallback_height))
    f = float(max(w, h) * 0.5)
    return f, f, float(w * 0.5), float(h * 0.5)


def _write_colmap_bundle(
    output_dir: Path,
    image_paths: List[Path],
    *,
    width: int,
    height: int,
    calibration_runtime: Dict[str, Any],
) -> Optional[Path]:
    if not image_paths:
        return None

    colmap_root = output_dir / "colmap"
    colmap_images = colmap_root / "images"
    colmap_root.mkdir(parents=True, exist_ok=True)
    colmap_images.mkdir(parents=True, exist_ok=True)

    camera_entries: Dict[int, Tuple[float, float, float, float]] = {}
    image_rows: List[Tuple[str, int]] = []
    mono_intr = _extract_intrinsics(calibration_runtime.get("mono"), fallback_width=width, fallback_height=height)
    left_intr = _extract_intrinsics(calibration_runtime.get("front"), fallback_width=width, fallback_height=height)
    right_intr = _extract_intrinsics(calibration_runtime.get("rear"), fallback_width=width, fallback_height=height)

    for src_path in image_paths:
        parent = src_path.parent.name
        camera_id = 1
        if parent in {"L", "F"}:
            camera_id = 2
        elif parent in {"R"}:
            camera_id = 3
        if camera_id == 1:
            camera_entries[camera_id] = mono_intr
        elif camera_id == 2:
            camera_entries[camera_id] = left_intr
        else:
            camera_entries[camera_id] = right_intr

        dst_name = f"{parent}__{src_path.name}" if parent in {"L", "R", "F"} else src_path.name
        dst_path = colmap_images / dst_name
        shutil.copy2(src_path, dst_path)
        image_rows.append((dst_name, camera_id))

    cameras_txt = colmap_root / "cameras.txt"
    with cameras_txt.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id in sorted(camera_entries.keys()):
            fx, fy, cx, cy = camera_entries[camera_id]
            f.write(
                f"{camera_id} PINHOLE {int(width)} {int(height)} "
                f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
            )

    image_list_txt = colmap_root / "image_list.txt"
    with image_list_txt.open("w", encoding="utf-8") as f:
        for image_name, camera_id in image_rows:
            f.write(f"{image_name} {camera_id}\n")

    return colmap_root


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
    confidences: List[float] = []
    for rec in frame_metrics_records:
        metrics = rec.get("metrics", {}) if isinstance(rec, dict) else {}
        reason = str(metrics.get("vo_status_reason", "unknown"))
        reason_counter[reason] += 1
        vo_attempted += 1 if float(metrics.get("vo_attempted", 0.0)) > 0.5 else 0
        vo_valid += 1 if float(metrics.get("vo_valid", 0.0)) > 0.5 else 0
        vo_pose_valid += 1 if float(metrics.get("vo_pose_valid", 0.0)) > 0.5 else 0
        with_pose += 1 if isinstance(rec.get("t_xyz"), list) and len(rec.get("t_xyz")) == 3 else 0
        conf = metrics.get("vo_confidence")
        if isinstance(conf, (float, int)):
            confidences.append(float(conf))

    dominant_reason = reason_counter.most_common(1)[0][0] if reason_counter else "unknown"
    valid_ratio = float(vo_valid / vo_attempted) if vo_attempted > 0 else 0.0
    conf_arr = np.asarray(confidences, dtype=np.float64) if confidences else np.zeros(0, dtype=np.float64)
    return {
        "total_records": int(len(frame_metrics_records)),
        "vo_attempted_frames": int(vo_attempted),
        "vo_valid_frames": int(vo_valid),
        "vo_valid_ratio": float(valid_ratio),
        "vo_pose_valid_frames": int(vo_pose_valid),
        "trajectory_points": int(with_pose),
        "vo_status_reason_counts": dict(reason_counter),
        "dominant_vo_status_reason": str(dominant_reason),
        "vo_confidence_mean": float(np.mean(conf_arr)) if conf_arr.size > 0 else 0.0,
        "vo_confidence_p10": float(np.percentile(conf_arr, 10.0)) if conf_arr.size > 0 else 0.0,
        "vo_confidence_p50": float(np.percentile(conf_arr, 50.0)) if conf_arr.size > 0 else 0.0,
        "vo_confidence_p90": float(np.percentile(conf_arr, 90.0)) if conf_arr.size > 0 else 0.0,
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
                "vo_confidence",
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
                    "vo_confidence": float(metrics.get("vo_confidence", 0.0)),
                }
            )
    return diagnostics_path, trajectory_path, diagnostics


def _parse_frame_index_from_filename(name: str) -> int:
    stem = Path(str(name or "")).stem
    for part in stem.split("_"):
        if part.isdigit():
            try:
                return int(part)
            except Exception:
                return -1
    return -1


def _build_exported_entries(image_root: Path, exported_image_paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in exported_image_paths:
        try:
            rel = p.resolve().relative_to(image_root.resolve())
        except Exception:
            rel = Path(p.name)
        rel_name = str(rel).replace("\\", "/")
        frame_idx = _parse_frame_index_from_filename(rel_name)
        rows.append(
            {
                "filename": rel_name,
                "frame_index": int(frame_idx),
                "abs_path": str(p.resolve()),
            }
        )
    return rows


def _build_frame_metrics_map(frame_metrics_records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rec in frame_metrics_records:
        try:
            idx = int(rec.get("frame_index", -1))
        except Exception:
            continue
        if idx < 0:
            continue
        out[idx] = rec
    return out


def summarize_quality_records(quality_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = int(len(quality_records))
    passed = 0
    reasons: Counter = Counter()
    quality_vals: List[float] = []
    for rec in quality_records:
        is_pass = bool(rec.get("is_pass", False))
        passed += 1 if is_pass else 0
        reason = str(rec.get("drop_reason", "unknown"))
        reasons[reason] += 1
        q = rec.get("quality")
        if isinstance(q, (int, float)):
            quality_vals.append(float(q))
    quality_arr = np.asarray(quality_vals, dtype=np.float64) if quality_vals else np.zeros(0, dtype=np.float64)
    summary = {
        "total_records": total,
        "passed_records": int(passed),
        "pass_ratio": float(passed / max(total, 1)),
        "drop_reason_counts": dict(reasons),
        "quality_min": float(np.min(quality_arr)) if quality_arr.size > 0 else None,
        "quality_max": float(np.max(quality_arr)) if quality_arr.size > 0 else None,
        "quality_median": float(np.median(quality_arr)) if quality_arr.size > 0 else None,
    }
    return summary


def write_quality_metrics(output_dir: Path, quality_records: List[Dict[str, Any]]) -> Tuple[Path, Path, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_quality_records(quality_records)

    json_path = output_dir / "quality_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": quality_records}, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "quality_metrics.csv"
    fieldnames = [
        "frame_index",
        "timestamp",
        "quality",
        "quality_lens_a",
        "quality_lens_b",
        "is_pass",
        "drop_reason",
        "raw_metrics",
        "norm_metrics",
        "lens_a_raw",
        "lens_a_norm",
        "lens_b_raw",
        "lens_b_norm",
        "legacy_quality_scores",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in quality_records:
            writer.writerow(
                {
                    "frame_index": int(rec.get("frame_index", 0)),
                    "timestamp": rec.get("timestamp"),
                    "quality": rec.get("quality"),
                    "quality_lens_a": rec.get("quality_lens_a"),
                    "quality_lens_b": rec.get("quality_lens_b"),
                    "is_pass": bool(rec.get("is_pass", False)),
                    "drop_reason": str(rec.get("drop_reason", "")),
                    "raw_metrics": json.dumps(rec.get("raw_metrics", {}), ensure_ascii=False, sort_keys=True),
                    "norm_metrics": json.dumps(rec.get("norm_metrics", {}), ensure_ascii=False, sort_keys=True),
                    "lens_a_raw": json.dumps(rec.get("lens_a_raw", {}), ensure_ascii=False, sort_keys=True),
                    "lens_a_norm": json.dumps(rec.get("lens_a_norm", {}), ensure_ascii=False, sort_keys=True),
                    "lens_b_raw": json.dumps(rec.get("lens_b_raw", {}), ensure_ascii=False, sort_keys=True),
                    "lens_b_norm": json.dumps(rec.get("lens_b_norm", {}), ensure_ascii=False, sort_keys=True),
                    "legacy_quality_scores": json.dumps(rec.get("legacy_quality_scores", {}), ensure_ascii=False, sort_keys=True),
                }
            )

    return json_path, csv_path, summary


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


def _validate_colmap_executable(colmap_path: str) -> Tuple[bool, str]:
    candidate = str(colmap_path or "colmap").strip() or "colmap"
    path_obj = Path(candidate)

    candidates: List[str] = []
    if path_obj.is_absolute() or os.sep in candidate:
        candidates.append(str(path_obj))
    else:
        found = shutil.which(candidate)
        if found:
            candidates.append(found)
        # Prefer Homebrew path if PATH is shadowed by legacy /usr/local binary.
        candidates.extend(
            [
                f"/opt/homebrew/bin/{candidate}",
                f"/usr/local/bin/{candidate}",
            ]
        )

    seen = set()
    final_candidates: List[str] = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        final_candidates.append(c)

    if not final_candidates:
        return False, f"COLMAP not found in PATH: {candidate}"

    errors: List[str] = []
    for resolved in final_candidates:
        p = Path(resolved)
        if not p.exists():
            errors.append(f"{resolved}: not found")
            continue
        try:
            proc = subprocess.run(
                [resolved, "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            if int(proc.returncode) == 0:
                return True, resolved
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = detail[-1] if detail else f"returncode={proc.returncode}"
            errors.append(f"{resolved}: {tail}")
        except Exception as e:
            errors.append(f"{resolved}: {e}")
            continue

    return False, "COLMAP executable check failed: " + " | ".join(errors)


def _resolve_colmap_keyframe_policy(config: Dict[str, Any]) -> str:
    pose_backend = str(config.get("pose_backend", "vo") or "vo").strip().lower()
    raw_policy = str(config.get("colmap_keyframe_policy", "") or "").strip().lower()
    if raw_policy not in {"legacy", "stage2_relaxed", "stage1_only"}:
        raw_policy = ""
    if raw_policy:
        return raw_policy
    return "stage2_relaxed" if pose_backend == "colmap" else "legacy"


def _resolve_colmap_pipeline_mode(config: Dict[str, Any]) -> str:
    pose_backend = str(config.get("pose_backend", "vo") or "vo").strip().lower()
    raw_mode = str(config.get("colmap_pipeline_mode", "") or "").strip().lower()
    if raw_mode not in {"legacy", "minimal_v1"}:
        raw_mode = ""
    if raw_mode:
        return raw_mode
    return "minimal_v1" if pose_backend == "colmap" else "legacy"


def _resolve_colmap_selection_profile(config: Dict[str, Any]) -> str:
    pose_backend = str(config.get("pose_backend", "vo") or "vo").strip().lower()
    raw_profile = str(config.get("colmap_selection_profile", "") or "").strip().lower()
    if raw_profile not in {"legacy", "no_vo_coverage"}:
        raw_profile = ""
    if raw_profile:
        return raw_profile
    return "no_vo_coverage" if pose_backend == "colmap" else "legacy"


def _normalize_colmap_rig_seed(seed_raw: Any) -> List[float]:
    if isinstance(seed_raw, str):
        try:
            return _parse_opk_seed_text(seed_raw)
        except argparse.ArgumentTypeError:
            return [0.0, 0.0, 180.0]
    if isinstance(seed_raw, (list, tuple)) and len(seed_raw) == 3:
        try:
            return [float(seed_raw[0]), float(seed_raw[1]), float(seed_raw[2])]
        except (TypeError, ValueError):
            return [0.0, 0.0, 180.0]
    return [0.0, 0.0, 180.0]


def _apply_colmap_keyframe_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    pose_backend = str(config.get("pose_backend", "vo") or "vo").strip().lower()
    pipeline_mode = _resolve_colmap_pipeline_mode(config)
    policy = _resolve_colmap_keyframe_policy(config)
    selection_profile = _resolve_colmap_selection_profile(config)
    minimal_mode = bool(pose_backend == "colmap" and pipeline_mode == "minimal_v1")
    target_mode = str(config.get("colmap_keyframe_target_mode", "") or "").strip().lower()
    if target_mode not in {"fixed", "auto"}:
        target_mode = "auto" if pose_backend == "colmap" else "fixed"
    target_min = int(max(1, config.get("colmap_keyframe_target_min", 120)))
    target_max = int(max(target_min, config.get("colmap_keyframe_target_max", 240)))
    nms_window = float(max(0.01, config.get("colmap_nms_window_sec", 0.35)))
    colmap_enable_stage0 = bool(config.get("colmap_enable_stage0", True))
    colmap_motion_aware_selection = bool(config.get("colmap_motion_aware_selection", True))
    if selection_profile == "no_vo_coverage" or minimal_mode:
        colmap_motion_aware_selection = False
    colmap_nms_motion_window_ratio = float(max(0.0, config.get("colmap_nms_motion_window_ratio", 0.5)))
    colmap_stage1_adaptive_threshold = bool(config.get("colmap_stage1_adaptive_threshold", True))
    colmap_stage1_min_candidates_per_bin = int(max(0, config.get("colmap_stage1_min_candidates_per_bin", 3)))
    colmap_stage1_max_candidates = int(max(1, config.get("colmap_stage1_max_candidates", 360)))
    colmap_stage2_entry_budget = int(max(1, config.get("colmap_stage2_entry_budget", 180)))
    colmap_stage2_entry_min_gap = int(max(0, config.get("colmap_stage2_entry_min_gap", 3)))
    colmap_diversity_ssim_threshold = float(
        max(0.0, min(1.0, config.get("colmap_diversity_ssim_threshold", 0.93)))
    )
    colmap_diversity_phash_hamming = int(max(0, config.get("colmap_diversity_phash_hamming", 10)))
    colmap_final_target_policy = str(config.get("colmap_final_target_policy", "") or "").strip().lower()
    if colmap_final_target_policy not in {"soft_auto", "fixed"}:
        colmap_final_target_policy = "soft_auto"
    colmap_final_soft_min = int(max(1, config.get("colmap_final_soft_min", 80)))
    colmap_final_soft_max = int(max(colmap_final_soft_min, config.get("colmap_final_soft_max", 220)))
    colmap_no_supplement_on_low_quality = bool(config.get("colmap_no_supplement_on_low_quality", True))
    rig_policy = str(config.get("colmap_rig_policy", "") or "").strip().lower()
    if rig_policy not in {"off", "lr_opk"}:
        rig_policy = "lr_opk" if pose_backend == "colmap" else "off"
    rig_seed_opk = _normalize_colmap_rig_seed(config.get("colmap_rig_seed_opk_deg", [0.0, 0.0, 180.0]))
    workspace_scope = str(config.get("colmap_workspace_scope", "") or "").strip().lower()
    if workspace_scope not in {"shared", "run_scoped"}:
        workspace_scope = "run_scoped"
    reuse_db = bool(config.get("colmap_reuse_db", False))
    analysis_mask_profile = str(config.get("colmap_analysis_mask_profile", "") or "").strip().lower()
    if analysis_mask_profile not in {"legacy", "colmap_safe"}:
        analysis_mask_profile = "colmap_safe" if pose_backend == "colmap" else "legacy"
    sparse_model_pick_policy = str(config.get("colmap_sparse_model_pick_policy", "") or "").strip().lower()
    if sparse_model_pick_policy not in {"registered_then_coverage", "coverage_then_registered", "latest_legacy"}:
        sparse_model_pick_policy = "registered_then_coverage"
    colmap_input_subset_enabled = bool(config.get("colmap_input_subset_enabled", pose_backend == "colmap"))
    colmap_input_gate_method = str(config.get("colmap_input_gate_method", "") or "").strip().lower()
    if colmap_input_gate_method not in {"homography_degeneracy_v1", "off"}:
        colmap_input_gate_method = "homography_degeneracy_v1"
    colmap_input_gate_strength = str(config.get("colmap_input_gate_strength", "") or "").strip().lower()
    if colmap_input_gate_strength not in {"weak", "medium", "strong"}:
        colmap_input_gate_strength = "medium"
    colmap_input_min_keep_ratio = float(
        max(0.0, min(1.0, config.get("colmap_input_min_keep_ratio", 0.20)))
    )
    colmap_input_max_gap_rescue_frames = int(max(1, config.get("colmap_input_max_gap_rescue_frames", 150)))

    if minimal_mode:
        logger.warning(
            "COLMAP minimal_v1 mode enabled: "
            "legacy keyframe knobs (policy/target/stage0/stage1.5/stage3/retarget/dynamic-mask) are ignored."
        )

    config["colmap_pipeline_mode"] = pipeline_mode
    config["COLMAP_PIPELINE_MODE"] = pipeline_mode
    config["colmap_keyframe_policy"] = policy
    config["COLMAP_KEYFRAME_POLICY"] = policy
    config["colmap_selection_profile"] = selection_profile
    config["COLMAP_SELECTION_PROFILE"] = selection_profile
    config["colmap_keyframe_target_mode"] = target_mode
    config["COLMAP_KEYFRAME_TARGET_MODE"] = target_mode
    config["colmap_keyframe_target_min"] = target_min
    config["COLMAP_KEYFRAME_TARGET_MIN"] = target_min
    config["colmap_keyframe_target_max"] = target_max
    config["COLMAP_KEYFRAME_TARGET_MAX"] = target_max
    config["colmap_nms_window_sec"] = nms_window
    config["COLMAP_NMS_WINDOW_SEC"] = nms_window
    config["colmap_enable_stage0"] = colmap_enable_stage0
    config["COLMAP_ENABLE_STAGE0"] = colmap_enable_stage0
    config["colmap_motion_aware_selection"] = colmap_motion_aware_selection
    config["COLMAP_MOTION_AWARE_SELECTION"] = colmap_motion_aware_selection
    config["colmap_nms_motion_window_ratio"] = colmap_nms_motion_window_ratio
    config["COLMAP_NMS_MOTION_WINDOW_RATIO"] = colmap_nms_motion_window_ratio
    config["colmap_stage1_adaptive_threshold"] = colmap_stage1_adaptive_threshold
    config["COLMAP_STAGE1_ADAPTIVE_THRESHOLD"] = colmap_stage1_adaptive_threshold
    config["colmap_stage1_min_candidates_per_bin"] = colmap_stage1_min_candidates_per_bin
    config["COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN"] = colmap_stage1_min_candidates_per_bin
    config["colmap_stage1_max_candidates"] = colmap_stage1_max_candidates
    config["COLMAP_STAGE1_MAX_CANDIDATES"] = colmap_stage1_max_candidates
    config["colmap_stage2_entry_budget"] = colmap_stage2_entry_budget
    config["COLMAP_STAGE2_ENTRY_BUDGET"] = colmap_stage2_entry_budget
    config["colmap_stage2_entry_min_gap"] = colmap_stage2_entry_min_gap
    config["COLMAP_STAGE2_ENTRY_MIN_GAP"] = colmap_stage2_entry_min_gap
    config["colmap_diversity_ssim_threshold"] = colmap_diversity_ssim_threshold
    config["COLMAP_DIVERSITY_SSIM_THRESHOLD"] = colmap_diversity_ssim_threshold
    config["colmap_diversity_phash_hamming"] = colmap_diversity_phash_hamming
    config["COLMAP_DIVERSITY_PHASH_HAMMING"] = colmap_diversity_phash_hamming
    config["colmap_final_target_policy"] = colmap_final_target_policy
    config["COLMAP_FINAL_TARGET_POLICY"] = colmap_final_target_policy
    config["colmap_final_soft_min"] = colmap_final_soft_min
    config["COLMAP_FINAL_SOFT_MIN"] = colmap_final_soft_min
    config["colmap_final_soft_max"] = colmap_final_soft_max
    config["COLMAP_FINAL_SOFT_MAX"] = colmap_final_soft_max
    config["colmap_no_supplement_on_low_quality"] = colmap_no_supplement_on_low_quality
    config["COLMAP_NO_SUPPLEMENT_ON_LOW_QUALITY"] = colmap_no_supplement_on_low_quality
    config["colmap_rig_policy"] = rig_policy
    config["COLMAP_RIG_POLICY"] = rig_policy
    config["colmap_rig_seed_opk_deg"] = list(rig_seed_opk)
    config["COLMAP_RIG_SEED_OPK_DEG"] = list(rig_seed_opk)
    config["colmap_workspace_scope"] = workspace_scope
    config["COLMAP_WORKSPACE_SCOPE"] = workspace_scope
    config["colmap_reuse_db"] = reuse_db
    config["COLMAP_REUSE_DB"] = reuse_db
    config["colmap_analysis_mask_profile"] = analysis_mask_profile
    config["COLMAP_ANALYSIS_MASK_PROFILE"] = analysis_mask_profile
    config["colmap_sparse_model_pick_policy"] = sparse_model_pick_policy
    config["COLMAP_SPARSE_MODEL_PICK_POLICY"] = sparse_model_pick_policy
    config["colmap_input_subset_enabled"] = colmap_input_subset_enabled
    config["COLMAP_INPUT_SUBSET_ENABLED"] = colmap_input_subset_enabled
    config["colmap_input_gate_method"] = colmap_input_gate_method
    config["COLMAP_INPUT_GATE_METHOD"] = colmap_input_gate_method
    config["colmap_input_gate_strength"] = colmap_input_gate_strength
    config["COLMAP_INPUT_GATE_STRENGTH"] = colmap_input_gate_strength
    config["colmap_input_min_keep_ratio"] = colmap_input_min_keep_ratio
    config["COLMAP_INPUT_MIN_KEEP_RATIO"] = colmap_input_min_keep_ratio
    config["colmap_input_max_gap_rescue_frames"] = colmap_input_max_gap_rescue_frames
    config["COLMAP_INPUT_MAX_GAP_RESCUE_FRAMES"] = colmap_input_max_gap_rescue_frames

    if pose_backend == "colmap" and (policy != "legacy" or minimal_mode):
        stage0_enabled = bool(colmap_enable_stage0) and selection_profile != "no_vo_coverage" and (not minimal_mode)
        config["enable_stage0_scan"] = stage0_enabled
        config["ENABLE_STAGE0_SCAN"] = stage0_enabled
        config["enable_stage3_refinement"] = False
        config["ENABLE_STAGE3_REFINEMENT"] = False
    if pose_backend == "colmap" and minimal_mode:
        config["enable_stage3_refinement"] = False
        config["ENABLE_STAGE3_REFINEMENT"] = False
        config["enable_dynamic_mask_removal"] = False
        config["ENABLE_DYNAMIC_MASK_REMOVAL"] = False
    elif pose_backend == "colmap" and selection_profile == "no_vo_coverage":
        config["enable_dynamic_mask_removal"] = True
        config["ENABLE_DYNAMIC_MASK_REMOVAL"] = True

    if pose_backend == "colmap" and analysis_mask_profile == "colmap_safe":
        dm_classes = config.get("dynamic_mask_target_classes", config.get("DYNAMIC_MASK_TARGET_CLASSES", []))
        if not isinstance(dm_classes, list):
            dm_classes = list(dm_classes) if dm_classes else []
        config["colmap_analysis_target_classes"] = [c for c in dm_classes if str(c) != "空"]
    else:
        config["colmap_analysis_target_classes"] = list(
            config.get("dynamic_mask_target_classes", config.get("DYNAMIC_MASK_TARGET_CLASSES", [])) or []
        )
    config["COLMAP_ANALYSIS_TARGET_CLASSES"] = list(config.get("colmap_analysis_target_classes", []))

    stage0_on = bool(config.get("enable_stage0_scan", config.get("ENABLE_STAGE0_SCAN", True)))
    stage3_on = bool(config.get("enable_stage3_refinement", config.get("ENABLE_STAGE3_REFINEMENT", True)))
    if minimal_mode:
        stage_plan = "Stage1->Stage2(minimal_v1)"
    elif pose_backend == "colmap" and policy == "stage1_only":
        stage_plan = "Stage1 only"
    elif pose_backend == "colmap" and policy == "stage2_relaxed":
        if selection_profile == "no_vo_coverage":
            stage_plan = "Stage1->Stage1.5->Stage2->Stage2.5(no_vo_coverage)"
        else:
            stage_plan = "Stage1->Stage0->Stage2(relaxed)" if stage0_on else "Stage1->Stage2(relaxed)"
    else:
        stage_plan = "Stage1->" + ("0->" if stage0_on else "") + "2" + ("->3" if stage3_on else "")

    return {
        "pose_backend": pose_backend,
        "pipeline_mode": pipeline_mode,
        "minimal_mode": minimal_mode,
        "policy": policy,
        "selection_profile": selection_profile,
        "target_mode": target_mode,
        "target_min": target_min,
        "target_max": target_max,
        "nms_window_sec": nms_window,
        "colmap_enable_stage0": colmap_enable_stage0,
        "colmap_motion_aware_selection": colmap_motion_aware_selection,
        "colmap_nms_motion_window_ratio": colmap_nms_motion_window_ratio,
        "colmap_stage1_adaptive_threshold": colmap_stage1_adaptive_threshold,
        "colmap_stage1_min_candidates_per_bin": colmap_stage1_min_candidates_per_bin,
        "colmap_stage1_max_candidates": colmap_stage1_max_candidates,
        "colmap_stage2_entry_budget": colmap_stage2_entry_budget,
        "colmap_stage2_entry_min_gap": colmap_stage2_entry_min_gap,
        "colmap_diversity_ssim_threshold": colmap_diversity_ssim_threshold,
        "colmap_diversity_phash_hamming": colmap_diversity_phash_hamming,
        "colmap_final_target_policy": colmap_final_target_policy,
        "colmap_final_soft_min": colmap_final_soft_min,
        "colmap_final_soft_max": colmap_final_soft_max,
        "colmap_no_supplement_on_low_quality": colmap_no_supplement_on_low_quality,
        "rig_policy": rig_policy,
        "rig_seed_opk_deg": list(rig_seed_opk),
        "workspace_scope": workspace_scope,
        "reuse_db": reuse_db,
        "analysis_mask_profile": analysis_mask_profile,
        "sparse_model_pick_policy": sparse_model_pick_policy,
        "colmap_input_subset_enabled": colmap_input_subset_enabled,
        "colmap_input_gate_method": colmap_input_gate_method,
        "colmap_input_gate_strength": colmap_input_gate_strength,
        "colmap_input_min_keep_ratio": colmap_input_min_keep_ratio,
        "colmap_input_max_gap_rescue_frames": colmap_input_max_gap_rescue_frames,
        "disabled_components": [
            "stage0",
            "stage1_5",
            "stage3",
            "dynamic_mask",
            "retarget",
            "vo_dependent_selection",
        ] if minimal_mode else [],
        "effective_stage_plan": stage_plan,
    }


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
    from core.stage_temp_store import StageTempStore
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
    run_id = str(config.get("analysis_run_id", config.get("ANALYSIS_RUN_ID", "")) or "").strip()
    if args.resume and not run_id:
        logger.error("--resume を使う場合は --analysis-run-id を指定してください")
        sys.exit(1)
    if not run_id:
        run_id = str(uuid.uuid4())
    config["analysis_run_id"] = run_id
    config["ANALYSIS_RUN_ID"] = run_id
    output_dir = resolve_output_dir(video_path, args.output)
    pose_backend = str(config.get("pose_backend", "vo") or "vo").strip().lower()
    if pose_backend not in {"vo", "colmap"}:
        pose_backend = "vo"
    config["pose_backend"] = pose_backend
    colmap_keyframe_runtime = _apply_colmap_keyframe_runtime(config)
    colmap_path = str(config.get("colmap_path", "colmap") or "colmap").strip() or "colmap"
    config["colmap_path"] = colmap_path
    if pose_backend == "colmap":
        ok, detail = _validate_colmap_executable(colmap_path)
        if not ok:
            logger.error(
                f"{detail}. "
                "COLMAPを再インストールするか、`--colmap-path`で有効な実行ファイルを指定してください。"
            )
            sys.exit(2)

    logger.info("=" * 60)
    logger.info("360Split - CLIモード")
    logger.info("=" * 60)
    logger.info(f"入力動画: {video_path}")
    logger.info(f"出力先:   {output_dir}")
    logger.info(f"analysis_run_id: {run_id}")
    logger.info(
        "Pose backend: "
        f"{pose_backend} "
        f"(export={str(config.get('pose_export_format', 'internal') or 'internal').lower()}, "
        f"trans_th={float(config.get('pose_select_translation_threshold', 1.2)):.2f}, "
        f"rot_th={float(config.get('pose_select_rotation_threshold_deg', 5.0)):.2f}, "
        f"min_obs={int(config.get('pose_select_min_observations', 30))})"
    )
    logger.info(
        "COLMAP keyframe policy: "
        f"{colmap_keyframe_runtime['policy']} "
        f"(pipeline={colmap_keyframe_runtime.get('pipeline_mode', 'legacy')}, "
        f"minimal={'ON' if colmap_keyframe_runtime.get('minimal_mode', False) else 'OFF'}, "
        f"profile={colmap_keyframe_runtime.get('selection_profile', 'legacy')}, "
        f"mode={colmap_keyframe_runtime['target_mode']}, "
        f"target={colmap_keyframe_runtime['target_min']}-{colmap_keyframe_runtime['target_max']}, "
        f"nms={colmap_keyframe_runtime['nms_window_sec']:.2f}s, "
        f"stage0={'ON' if colmap_keyframe_runtime.get('colmap_enable_stage0', True) else 'OFF'}, "
        f"motion_aware={'ON' if colmap_keyframe_runtime.get('colmap_motion_aware_selection', True) else 'OFF'}"
        f"@ratio={float(colmap_keyframe_runtime.get('colmap_nms_motion_window_ratio', 0.5)):.2f}, "
        f"rig={colmap_keyframe_runtime['rig_policy']}@opk={colmap_keyframe_runtime['rig_seed_opk_deg']}, "
        f"workspace_scope={colmap_keyframe_runtime['workspace_scope']}, "
        f"reuse_db={colmap_keyframe_runtime['reuse_db']}, "
        f"mask_profile={colmap_keyframe_runtime['analysis_mask_profile']}, "
        f"plan={colmap_keyframe_runtime['effective_stage_plan']})"
    )
    if bool(config.get("resume_enabled", False)):
        logger.info("resume: 有効")
    if bool(config.get("keep_temp_on_success", False)):
        logger.info("keep_temp: 有効")
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
    logger.info(
        "品質フィルタ: "
        f"{'ON' if bool(config.get('quality_filter_enabled', True)) else 'OFF'} "
        f"(threshold={float(config.get('quality_threshold', 0.50)):.2f}, "
        f"roi={str(config.get('quality_roi_mode', 'circle'))}:{float(config.get('quality_roi_ratio', 0.40)):.2f}, "
        f"abs_lap_min={float(config.get('quality_abs_laplacian_min', 35.0)):.1f}, "
        f"orb={'ON' if bool(config.get('quality_use_orb', True)) else 'OFF'})"
    )
    logger.info("-" * 60)

    loader = create_loader(video_path, args, is_front_rear, is_osv, config=config)

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

    stage_store = StageTempStore(run_id=run_id)
    keyframes = selector.select_keyframes(
        loader,
        progress_callback=progress_callback,
        frame_log_callback=frame_log_callback,
        stage_temp_store=stage_store,
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

    exported_image_paths: List[Path] = []
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
                exported_image_paths.append(filepath)

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
            exported_image_paths.append(filepath)

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

    pose_summary: Dict[str, Any] = {
        "enabled": True,
        "backend": pose_backend,
        "trajectory_count": 0,
        "selected_count": 0,
        "selection_stats": {},
        "pose_trajectory_csv": None,
        "metashape_csv": None,
        "selected_images_dir": None,
        "selected_images_list": None,
        "copied_count": 0,
        "diagnostics": {},
        "raw_log_paths": {},
        "failure_reason": None,
    }
    pose_image_root = stereo_images_root if is_stereo_mode else output_dir
    try:
        exported_entries = _build_exported_entries(pose_image_root, exported_image_paths)
        frame_metrics_map = _build_frame_metrics_map(frame_metrics_records)
        colmap_preview_frame_indices = [
            int(v)
            for v in list(
                getattr(selector, "last_selection_runtime", {}).get(
                    "stage2_colmap_preview_indices",
                    [],
                )
                or []
            )
            if isinstance(v, (int, float))
        ]
        if not str(config.get("colmap_workspace", "") or "").strip():
            config["colmap_workspace"] = str((output_dir / "pose_colmap").resolve())

        def _pose_log(msg: str):
            logger.info(msg)

        pose_payload = run_pose_pipeline(
            image_dir=str(pose_image_root),
            output_dir=str(output_dir),
            config=config,
            context={
                "log_callback": _pose_log,
                "calibration_runtime": dict(calibration_runtime or {}),
                "exported_entries": exported_entries,
                "frame_metrics_map": frame_metrics_map,
                "vo_trajectory_csv": str(output_dir / "vo_trajectory.csv"),
                "colmap_workspace": str(config.get("colmap_workspace", "") or "").strip() or str(output_dir / "pose_colmap"),
                "colmap_db_path": str(config.get("colmap_db_path", "") or "").strip(),
                "analysis_run_id": str(run_id),
                "colmap_workspace_scope": str(config.get("colmap_workspace_scope", "run_scoped") or "run_scoped"),
                "colmap_reuse_db": bool(config.get("colmap_reuse_db", False)),
                "colmap_rig_policy": str(config.get("colmap_rig_policy", "lr_opk") or "lr_opk"),
                "colmap_rig_seed_opk_deg": list(config.get("colmap_rig_seed_opk_deg", [0.0, 0.0, 180.0])),
                "colmap_sparse_model_pick_policy": str(config.get("colmap_sparse_model_pick_policy", "registered_then_coverage") or "registered_then_coverage"),
                "colmap_input_subset_enabled": bool(config.get("colmap_input_subset_enabled", True)),
                "colmap_input_gate_method": str(config.get("colmap_input_gate_method", "homography_degeneracy_v1") or "homography_degeneracy_v1"),
                "colmap_input_gate_strength": str(config.get("colmap_input_gate_strength", "medium") or "medium"),
                "colmap_input_min_keep_ratio": float(config.get("colmap_input_min_keep_ratio", 0.20)),
                "colmap_input_max_gap_rescue_frames": int(config.get("colmap_input_max_gap_rescue_frames", 150)),
                "colmap_preview_frame_indices": colmap_preview_frame_indices,
            },
        )
        pose_result = pose_payload.get("result")
        pose_summary.update(
            {
                "trajectory_count": int(pose_payload.get("trajectory_count", 0)),
                "selected_count": int(pose_payload.get("selected_count", 0)),
                "selection_stats": dict(pose_payload.get("selection_stats", {})),
                "pose_trajectory_csv": pose_payload.get("pose_trajectory_csv"),
                "metashape_csv": pose_payload.get("metashape_csv"),
                "selected_images_dir": pose_payload.get("selected_images_dir"),
                "selected_images_list": pose_payload.get("selected_images_list"),
                "copied_count": int(pose_payload.get("copied_count", 0)),
                "diagnostics": dict(getattr(pose_result, "diagnostics", {}) or {}),
                "raw_log_paths": dict(getattr(pose_result, "raw_log_paths", {}) or {}),
            }
        )
        logger.info(
            "Pose summary: "
            f"backend={pose_summary['backend']}, "
            f"trajectory={pose_summary['trajectory_count']}, "
            f"selected={pose_summary['selected_count']}, "
            f"selected_frames={pose_summary.get('selection_stats', {}).get('selected_frame_count')}, "
            f"spatial_post={pose_summary.get('selection_stats', {}).get('spatial_post_filter', {}).get('applied')}, "
            f"copied={pose_summary['copied_count']}"
        )
    except Exception as e:
        pose_summary["failure_reason"] = str(e)
        logger.error(f"Pose推定に失敗しました: {e}")
        if pose_backend == "colmap":
            loader.close()
            sys.exit(2)

    colmap_dir = None
    if bool(config.get("colmap_format", False)):
        try:
            colmap_dir = _write_colmap_bundle(
                output_dir=output_dir,
                image_paths=exported_image_paths,
                width=int(meta.width),
                height=int(meta.height),
                calibration_runtime=dict(calibration_runtime or {}),
            )
            if colmap_dir is not None:
                logger.info(f"COLMAP互換出力: {colmap_dir}")
        except Exception as e:
            logger.warning(f"COLMAP互換出力に失敗しました: {e}")

    quality_records = list(getattr(selector, "stage1_quality_records", []) or [])
    quality_summary = summarize_quality_records(quality_records) if quality_records else {
        "total_records": 0,
        "passed_records": 0,
        "pass_ratio": 0.0,
        "drop_reason_counts": {},
        "quality_min": None,
        "quality_max": None,
        "quality_median": None,
    }

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
        "quality_filter": {
            "enabled": bool(config.get("quality_filter_enabled", True)),
            "threshold": float(config.get("quality_threshold", 0.50)),
            "roi_mode": str(config.get("quality_roi_mode", "circle")),
            "roi_ratio": float(config.get("quality_roi_ratio", 0.40)),
            "abs_laplacian_min": float(config.get("quality_abs_laplacian_min", 35.0)),
            "use_orb": bool(config.get("quality_use_orb", True)),
            "tenengrad_scale": float(config.get("quality_tenengrad_scale", 1.0)),
            "weights": {
                "sharpness": float(config.get("quality_weight_sharpness", 0.40)),
                "tenengrad": float(config.get("quality_weight_tenengrad", 0.30)),
                "exposure": float(config.get("quality_weight_exposure", 0.15)),
                "keypoints": float(config.get("quality_weight_keypoints", 0.15)),
            },
            "norm_percentiles": {
                "p_low": float(config.get("quality_norm_p_low", 10.0)),
                "p_high": float(config.get("quality_norm_p_high", 90.0)),
            },
            "debug": bool(config.get("quality_debug", False)),
        },
        "pipeline_runtime": {
            "analysis_run_id": run_id,
            "resume_enabled": bool(config.get("resume_enabled", False)),
            "keep_temp_on_success": bool(config.get("keep_temp_on_success", False)),
            "flow_downscale": float(config.get("flow_downscale", 1.0)),
            "stage3_disable_traj_when_vo_unreliable": bool(
                config.get("stage3_disable_traj_when_vo_unreliable", True)
            ),
            "stage3_vo_valid_ratio_threshold": float(config.get("stage3_vo_valid_ratio_threshold", 0.50)),
        },
        "keyframe_policy": str(
            getattr(selector, "last_selection_runtime", {}).get(
                "policy",
                colmap_keyframe_runtime.get("policy", "legacy"),
            )
        ),
        "keyframe_target_mode": str(
            getattr(selector, "last_selection_runtime", {}).get(
                "target_mode",
                colmap_keyframe_runtime.get("target_mode", "fixed"),
            )
        ),
        "effective_stage_plan": str(
            getattr(selector, "last_selection_runtime", {}).get(
                "effective_stage_plan",
                colmap_keyframe_runtime.get("effective_stage_plan", "unknown"),
            )
        ),
        "stage1_candidates_raw": int(
            getattr(selector, "last_selection_runtime", {}).get("stage1_candidates_raw", 0)
        ),
        "stage1_candidates_effective": int(
            getattr(selector, "last_selection_runtime", {}).get("stage1_candidates_effective", 0)
        ),
        "stage1_adaptive_threshold_base": float(
            getattr(selector, "last_selection_runtime", {}).get("stage1_adaptive_threshold_base", 0.0)
        ),
        "stage1_adaptive_threshold_effective": float(
            getattr(selector, "last_selection_runtime", {}).get("stage1_adaptive_threshold_effective", 0.0)
        ),
        "stage1_bin_floor_added_count": int(
            getattr(selector, "last_selection_runtime", {}).get("stage1_bin_floor_added_count", 0)
        ),
        "motion_median_step": float(
            getattr(selector, "last_selection_runtime", {}).get("motion_median_step", 0.0)
        ),
        "effective_motion_window": float(
            getattr(selector, "last_selection_runtime", {}).get("effective_motion_window", 0.0)
        ),
        "motion_bins_occupied_before": int(
            getattr(selector, "last_selection_runtime", {}).get("motion_bins_occupied_before", 0)
        ),
        "motion_bins_occupied_after": int(
            getattr(selector, "last_selection_runtime", {}).get("motion_bins_occupied_after", 0)
        ),
        "stage2_drop_reason_counts": round_json_friendly(
            getattr(selector, "last_selection_runtime", {}).get("stage2_drop_reason_counts", {})
        ),
        "pre_retarget_count": int(
            getattr(selector, "last_selection_runtime", {}).get("pre_retarget_count", len(keyframes))
        ),
        "post_retarget_count": int(
            getattr(selector, "last_selection_runtime", {}).get("post_retarget_count", len(keyframes))
        ),
        "retarget_reason": str(
            getattr(selector, "last_selection_runtime", {}).get("retarget_reason", "n/a")
        ),
        "effective_target_min": int(
            getattr(selector, "last_selection_runtime", {}).get(
                "effective_target_min",
                colmap_keyframe_runtime.get("target_min", 120),
            )
        ),
        "effective_target_max": int(
            getattr(selector, "last_selection_runtime", {}).get(
                "effective_target_max",
                colmap_keyframe_runtime.get("target_max", 240),
            )
        ),
        "auto_target": round_json_friendly(
            getattr(selector, "last_selection_runtime", {}).get("auto_target", {})
        ),
        "coverage_before": round_json_friendly(
            getattr(selector, "last_selection_runtime", {}).get("coverage_before", {})
        ),
        "coverage_after": round_json_friendly(
            getattr(selector, "last_selection_runtime", {}).get("coverage_after", {})
        ),
        "colmap_runtime": {
            "pipeline_mode": str(colmap_keyframe_runtime.get("pipeline_mode", "legacy")),
            "minimal_mode": bool(colmap_keyframe_runtime.get("minimal_mode", False)),
            "disabled_components": list(colmap_keyframe_runtime.get("disabled_components", [])),
            "target_mode": str(colmap_keyframe_runtime.get("target_mode", "fixed")),
            "enable_stage0": bool(colmap_keyframe_runtime.get("colmap_enable_stage0", True)),
            "motion_aware_selection": bool(colmap_keyframe_runtime.get("colmap_motion_aware_selection", True)),
            "nms_motion_window_ratio": float(colmap_keyframe_runtime.get("colmap_nms_motion_window_ratio", 0.5)),
            "stage1_adaptive_threshold": bool(colmap_keyframe_runtime.get("colmap_stage1_adaptive_threshold", True)),
            "stage1_min_candidates_per_bin": int(colmap_keyframe_runtime.get("colmap_stage1_min_candidates_per_bin", 3)),
            "stage1_max_candidates": int(colmap_keyframe_runtime.get("colmap_stage1_max_candidates", 360)),
            "rig_policy": str(colmap_keyframe_runtime.get("rig_policy", "off")),
            "rig_seed_opk_deg": list(colmap_keyframe_runtime.get("rig_seed_opk_deg", [0.0, 0.0, 180.0])),
            "workspace_scope": str(colmap_keyframe_runtime.get("workspace_scope", "run_scoped")),
            "reuse_db": bool(colmap_keyframe_runtime.get("reuse_db", False)),
            "analysis_mask_profile": str(colmap_keyframe_runtime.get("analysis_mask_profile", "legacy")),
        },
        "pose": pose_summary,
        "colmap": {
            "enabled": bool(config.get("colmap_format", False)),
            "output_dir": str(colmap_dir) if colmap_dir is not None else None,
        },
        "quality_summary": round_json_friendly(quality_summary),
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
    if quality_records:
        quality_json_path, quality_csv_path, quality_diag = write_quality_metrics(output_dir, quality_records)
        logger.info(f"品質メトリクス(JSON): {quality_json_path}")
        logger.info(f"品質メトリクス(CSV): {quality_csv_path}")
        logger.info(
            "品質 summary: "
            f"total={quality_diag['total_records']}, "
            f"passed={quality_diag['passed_records']}, "
            f"pass_ratio={quality_diag['pass_ratio']:.3f}"
        )
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
            f"conf_mean={vo_diag['vo_confidence_mean']:.3f}, "
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
