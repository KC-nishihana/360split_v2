#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.visual_odometry.calibration import load_calibration_xml
from core.visual_odometry.calibration_check import run_calibration_check
from utils.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration undistortion check tool")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--calib-xml", type=str, default=None)
    p.add_argument("--front-video", type=str, default=None)
    p.add_argument("--rear-video", type=str, default=None)
    p.add_argument("--front-calib-xml", type=str, default=None)
    p.add_argument("--rear-calib-xml", type=str, default=None)
    p.add_argument("--calib-model", type=str, default="auto", choices=["auto", "opencv", "fisheye"])
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--roi-ratio", type=float, default=0.6)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _video_size(path: str) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (0, 0)
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    is_mono = bool(args.video)
    is_pair = bool(args.front_video and args.rear_video)
    if not is_mono and not is_pair:
        logger.error("Either --video or --front-video/--rear-video is required")
        return 2

    if is_mono:
        if not args.calib_xml:
            logger.error("--calib-xml is required in monocular mode")
            return 2
        if not Path(args.video).exists():
            logger.error(f"video not found: {args.video}")
            return 2
        if not Path(args.calib_xml).exists():
            logger.error(f"calibration XML not found: {args.calib_xml}")
            return 2
        calib = load_calibration_xml(
            args.calib_xml,
            model_hint=args.calib_model,
            fallback_image_size=_video_size(args.video),
            logger=logger,
        )
        run_calibration_check(
            out_dir=str(out),
            video_path=args.video,
            calib=calib,
            frame_index=args.frame,
            roi_ratio=args.roi_ratio,
            logger=logger,
        )
        return 0

    # paired
    if not Path(args.front_video).exists() or not Path(args.rear_video).exists():
        logger.error(f"paired videos not found: front={args.front_video}, rear={args.rear_video}")
        return 2
    if not args.front_calib_xml and not args.calib_xml:
        logger.error("--front-calib-xml (or fallback --calib-xml) is required")
        return 2
    if not args.rear_calib_xml and not args.calib_xml:
        logger.error("--rear-calib-xml (or fallback --calib-xml) is required")
        return 2

    front_xml = args.front_calib_xml or args.calib_xml
    rear_xml = args.rear_calib_xml or args.calib_xml
    front_calib = load_calibration_xml(
        front_xml,
        model_hint=args.calib_model,
        fallback_image_size=_video_size(args.front_video),
        logger=logger,
    )
    rear_calib = load_calibration_xml(
        rear_xml,
        model_hint=args.calib_model,
        fallback_image_size=_video_size(args.rear_video),
        logger=logger,
    )
    run_calibration_check(
        out_dir=str(out),
        front_video_path=args.front_video,
        rear_video_path=args.rear_video,
        front_calib=front_calib,
        rear_calib=rear_calib,
        frame_index=args.frame,
        roi_ratio=args.roi_ratio,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
