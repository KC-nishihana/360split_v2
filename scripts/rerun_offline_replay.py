#!/usr/bin/env python3
"""
Offline replay utility: CSV/JSON -> Rerun stream/.rrd
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.rerun_logger import RerunKeyframeLogger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline replay logger for Rerun")
    p.add_argument("--input", required=True, help="CSV or JSON input path")
    p.add_argument("--rrd", required=True, help="Output .rrd path")
    p.add_argument("--spawn", action="store_true", default=False, help="Spawn Rerun Viewer")
    p.add_argument("--image-root", default=None, help="Base directory for relative image paths")
    p.add_argument(
        "--points-on-keyframe-only",
        action="store_true",
        default=False,
        help="Only send point clouds when keyframe_flag=1",
    )
    return p.parse_args()


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _read_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("frames"), list):
            return data["frames"]
        if isinstance(data.get("records"), list):
            return data["records"]
    raise ValueError("JSONの形式が不正です。list または {frames:[...]} を指定してください。")


def _resolve_image_path(record: Dict, image_root: Optional[Path]) -> Optional[Path]:
    for key in ("image_path", "img_path", "frame_path", "frame_png"):
        value = record.get(key)
        if value:
            p = Path(str(value))
            if p.exists():
                return p
            if image_root is not None:
                q = image_root / p
                if q.exists():
                    return q
    return None


def _extract_metrics(record: Dict) -> Dict[str, float]:
    metrics = {}
    nested = record.get("metrics")
    if isinstance(nested, dict):
        for k, v in nested.items():
            metrics[str(k)] = _to_float(v)

    metric_keys = (
        "translation_delta",
        "rotation_delta",
        "laplacian_var",
        "match_count",
        "overlap_ratio",
        "exposure_ratio",
        "keyframe_flag",
        "combined_score",
    )
    for k in metric_keys:
        if k in record:
            metrics[k] = _to_float(record.get(k))

    for k, v in record.items():
        if str(k).startswith("m_"):
            metrics[str(k)[2:]] = _to_float(v)

    return metrics


def _extract_points(record: Dict) -> Optional[np.ndarray]:
    points_value = record.get("points_world")
    if isinstance(points_value, list):
        arr = np.asarray(points_value, dtype=np.float32)
        if arr.size > 0:
            return arr.reshape(-1, 3)

    points_path = record.get("points_path")
    if points_path:
        p = Path(str(points_path))
        if p.exists() and p.suffix.lower() == ".npy":
            arr = np.load(str(p))
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size > 0:
                return arr.reshape(-1, 3)
    return None


def iter_log_payloads(records: Iterable[Dict], image_root: Optional[Path]):
    for idx, record in enumerate(records):
        frame_idx = _to_int(record.get("frame_index", record.get("frame_idx", idx)), idx)

        image_path = _resolve_image_path(record, image_root)
        image = None
        if image_path is not None:
            image = cv2.imread(str(image_path))

        t_xyz = (
            record.get("t_xyz")
            if record.get("t_xyz") is not None
            else [record.get("tx", frame_idx), record.get("ty", 0.0), record.get("tz", 0.0)]
        )
        q_wxyz = (
            record.get("q_wxyz")
            if record.get("q_wxyz") is not None
            else [record.get("qw", 1.0), record.get("qx", 0.0), record.get("qy", 0.0), record.get("qz", 0.0)]
        )

        metrics = _extract_metrics(record)
        is_keyframe = _to_bool(record.get("is_keyframe", metrics.get("keyframe_flag", 0)))
        if "keyframe_flag" not in metrics:
            metrics["keyframe_flag"] = 1.0 if is_keyframe else 0.0

        yield {
            "frame_idx": frame_idx,
            "image": image,
            "t_xyz": t_xyz,
            "q_wxyz": q_wxyz,
            "is_keyframe": is_keyframe,
            "metrics": metrics,
            "points_world": _extract_points(record),
        }


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    image_root = Path(args.image_root) if args.image_root else None

    records = _read_records(input_path)
    logger = RerunKeyframeLogger(
        app_id="keyframe_check_offline",
        spawn=bool(args.spawn),
        save_path=args.rrd,
        timeline_name="frame",
    )
    if not logger.enabled:
        return 1

    for payload in iter_log_payloads(records, image_root):
        points_world = payload["points_world"]
        if args.points_on_keyframe_only and (not payload["is_keyframe"]):
            points_world = None
        logger.log_frame(
            frame_idx=payload["frame_idx"],
            img=payload["image"],
            t_xyz=payload["t_xyz"],
            q_wxyz=payload["q_wxyz"],
            is_keyframe=payload["is_keyframe"],
            metrics=payload["metrics"],
            points_world=points_world,
        )

    print(f"[OK] replayed {len(records)} records -> {args.rrd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
