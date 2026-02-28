from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .types import PoseRecord


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q /= n
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_ypr_deg(r: np.ndarray) -> tuple[float, float, float]:
    # ZYX Euler: yaw(z), pitch(y), roll(x)
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    singular = sy < 1e-9
    if not singular:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:
        roll = float(np.arctan2(-r[1, 2], r[1, 1]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        yaw = 0.0
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def write_internal_pose_csv(path: Path, poses: Iterable[PoseRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "filename",
                "qw",
                "qx",
                "qy",
                "qz",
                "tx",
                "ty",
                "tz",
                "confidence",
                "observations",
            ],
        )
        writer.writeheader()
        for pose in poses:
            writer.writerow(
                {
                    "frame_index": int(pose.frame_index),
                    "filename": str(pose.filename),
                    "qw": float(pose.qw),
                    "qx": float(pose.qx),
                    "qy": float(pose.qy),
                    "qz": float(pose.qz),
                    "tx": float(pose.tx),
                    "ty": float(pose.ty),
                    "tz": float(pose.tz),
                    "confidence": float(pose.confidence),
                    "observations": int(pose.observations),
                }
            )
    return path


def write_metashape_csv(path: Path, poses: Iterable[PoseRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "x", "y", "z", "yaw", "pitch", "roll"],
        )
        writer.writeheader()
        for pose in poses:
            r = _quat_to_rotmat(pose.qw, pose.qx, pose.qy, pose.qz)
            yaw, pitch, roll = _rotmat_to_ypr_deg(r)
            writer.writerow(
                {
                    "filename": str(pose.filename),
                    "x": float(pose.tx),
                    "y": float(pose.ty),
                    "z": float(pose.tz),
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "roll": float(roll),
                }
            )
    return path


def export_selected_images(
    image_root: Path,
    selected_poses: List[PoseRecord],
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_list_path = output_dir.parent / "selected_images.txt"
    copied_paths: List[str] = []

    with selected_list_path.open("w", encoding="utf-8") as list_f:
        for pose in selected_poses:
            rel = Path(str(pose.filename))
            src = image_root / rel
            if not src.exists():
                continue
            dst = output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rel_text = str(rel)
            copied_paths.append(rel_text)
            list_f.write(rel_text + "\n")

    return {
        "selected_list_path": str(selected_list_path),
        "copied_count": len(copied_paths),
        "copied": copied_paths,
    }
