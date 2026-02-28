from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .types import PoseRecord


@dataclass
class PoseSelectionConfig:
    translation_threshold: float = 1.2
    rotation_threshold_deg: float = 5.0
    min_observations: int = 30
    enable_translation: bool = True
    enable_rotation: bool = True
    enable_observations: bool = False


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    qn = _quat_normalize(q)
    return np.array([qn[0], -qn[1], -qn[2], -qn[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _relative_rotation_deg(q_from: np.ndarray, q_to: np.ndarray) -> float:
    q_rel = _quat_multiply(_quat_inverse(q_from), q_to)
    q_rel = _quat_normalize(q_rel)
    w = float(np.clip(abs(q_rel[0]), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def _median_step_norm(poses: List[PoseRecord]) -> float:
    if len(poses) < 2:
        return 1.0
    steps: List[float] = []
    for i in range(1, len(poses)):
        a = poses[i - 1]
        b = poses[i]
        d = np.array([b.tx - a.tx, b.ty - a.ty, b.tz - a.tz], dtype=np.float64)
        steps.append(float(np.linalg.norm(d)))
    arr = np.asarray([s for s in steps if s > 0.0], dtype=np.float64)
    if arr.size == 0:
        return 1.0
    med = float(np.median(arr))
    return med if med > 1e-12 else 1.0


def select_required_poses(poses: List[PoseRecord], cfg: PoseSelectionConfig) -> Dict[str, object]:
    ordered = sorted(poses, key=lambda p: (int(p.frame_index), str(p.filename)))
    if not ordered:
        return {
            "selected": [],
            "stats": {
                "trajectory_count": 0,
                "selected_count": 0,
                "median_step": 1.0,
                "mean_translation_norm": 0.0,
                "mean_rotation_deg": 0.0,
            },
        }

    # If all gates are disabled, keep all trajectories.
    if not (cfg.enable_translation or cfg.enable_rotation or cfg.enable_observations):
        return {
            "selected": list(ordered),
            "stats": {
                "trajectory_count": len(ordered),
                "selected_count": len(ordered),
                "median_step": _median_step_norm(ordered),
                "mean_translation_norm": 0.0,
                "mean_rotation_deg": 0.0,
            },
        }

    median_step = _median_step_norm(ordered)
    selected: List[PoseRecord] = [ordered[0]]
    trans_hist: List[float] = []
    rot_hist: List[float] = []

    for pose in ordered[1:]:
        last = selected[-1]
        d = np.array([pose.tx - last.tx, pose.ty - last.ty, pose.tz - last.tz], dtype=np.float64)
        step = float(np.linalg.norm(d))
        step_norm = float(step / max(median_step, 1e-12))

        q_last = np.asarray([last.qw, last.qx, last.qy, last.qz], dtype=np.float64)
        q_cur = np.asarray([pose.qw, pose.qx, pose.qy, pose.qz], dtype=np.float64)
        rot_deg = _relative_rotation_deg(q_last, q_cur)

        trans_ok = bool(cfg.enable_translation and step_norm >= float(cfg.translation_threshold))
        rot_ok = bool(cfg.enable_rotation and rot_deg >= float(cfg.rotation_threshold_deg))
        obs_ok = bool(cfg.enable_observations and int(pose.observations) >= int(cfg.min_observations))

        if trans_ok or rot_ok or obs_ok:
            selected.append(pose)
            trans_hist.append(step_norm)
            rot_hist.append(rot_deg)

    return {
        "selected": selected,
        "stats": {
            "trajectory_count": len(ordered),
            "selected_count": len(selected),
            "median_step": median_step,
            "mean_translation_norm": float(np.mean(np.asarray(trans_hist, dtype=np.float64))) if trans_hist else 0.0,
            "mean_rotation_deg": float(np.mean(np.asarray(rot_hist, dtype=np.float64))) if rot_hist else 0.0,
        },
    }
