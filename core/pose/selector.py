from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    enable_spatial_post_filter: bool = False
    spatial_cell_scale: float = 3.0
    spatial_min_distance_scale: float = 1.2
    spatial_max_per_cell: int = 1
    spatial_max_gap_frames: int = 150
    spatial_min_keep_ratio: float = 0.45
    spatial_max_keep_ratio: float = 0.80


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


def _sensor_priority(filename: str) -> int:
    rel = str(filename or "").replace("\\", "/").strip("/")
    if not rel:
        return 9
    first = rel.split("/")[0].upper()
    if first == "F":
        return 0
    if first == "L":
        return 1
    if first == "R":
        return 2
    return 3


def _representative_pose(frame_poses: List[PoseRecord]) -> PoseRecord:
    ordered = sorted(
        frame_poses,
        key=lambda p: (_sensor_priority(str(p.filename)), str(p.filename)),
    )
    return ordered[0]


def _spatial_cell(pose: PoseRecord, cell_size: float) -> Tuple[int, int, int]:
    s = float(max(cell_size, 1e-9))
    return (
        int(np.floor(float(pose.tx) / s)),
        int(np.floor(float(pose.ty) / s)),
        int(np.floor(float(pose.tz) / s)),
    )


def _pose_distance(a: PoseRecord, b: PoseRecord) -> float:
    d = np.array([float(a.tx - b.tx), float(a.ty - b.ty), float(a.tz - b.tz)], dtype=np.float64)
    return float(np.linalg.norm(d))


def _apply_spatial_post_filter(poses: List[PoseRecord], cfg: PoseSelectionConfig) -> Tuple[List[PoseRecord], Dict[str, float]]:
    if len(poses) <= 2:
        return list(poses), {"applied": 0.0, "before_count": float(len(poses)), "after_count": float(len(poses))}

    ordered = sorted(poses, key=lambda p: int(p.frame_index))
    median_step = _median_step_norm(ordered)
    cell_size = float(max(1e-9, median_step * max(0.1, float(cfg.spatial_cell_scale))))
    min_dist = float(max(1e-9, median_step * max(0.1, float(cfg.spatial_min_distance_scale))))
    max_per_cell = max(1, int(cfg.spatial_max_per_cell))
    max_gap_frames = max(1, int(cfg.spatial_max_gap_frames))
    min_keep_ratio = float(np.clip(float(cfg.spatial_min_keep_ratio), 0.0, 1.0))
    max_keep_ratio = float(np.clip(float(cfg.spatial_max_keep_ratio), min_keep_ratio, 1.0))

    target_min_keep = max(1, int(round(len(ordered) * min_keep_ratio)))
    target_max_keep = max(target_min_keep, int(round(len(ordered) * max_keep_ratio)))
    target_max_keep = min(target_max_keep, len(ordered))

    selected: List[PoseRecord] = [ordered[0]]
    skipped: List[PoseRecord] = []
    cell_counts: Dict[Tuple[int, int, int], int] = {}
    first_cell = _spatial_cell(ordered[0], cell_size)
    cell_counts[first_cell] = 1

    for pose in ordered[1:]:
        if int(pose.frame_index) - int(selected[-1].frame_index) >= max_gap_frames:
            selected.append(pose)
            c = _spatial_cell(pose, cell_size)
            cell_counts[c] = int(cell_counts.get(c, 0) + 1)
            continue
        c = _spatial_cell(pose, cell_size)
        if int(cell_counts.get(c, 0)) >= max_per_cell:
            skipped.append(pose)
            continue
        nearest = min((_pose_distance(pose, kept) for kept in selected), default=0.0)
        if float(nearest) >= min_dist:
            selected.append(pose)
            cell_counts[c] = int(cell_counts.get(c, 0) + 1)
        else:
            skipped.append(pose)

    # Ensure temporal coverage endpoint.
    if int(selected[-1].frame_index) != int(ordered[-1].frame_index):
        selected.append(ordered[-1])

    # Keep ratio floor rescue by farthest-point insertion.
    remaining = [p for p in ordered if int(p.frame_index) not in {int(s.frame_index) for s in selected}]
    while len(selected) < target_min_keep and remaining:
        best = max(
            remaining,
            key=lambda p: (
                min((_pose_distance(p, kept) for kept in selected), default=0.0),
                int(p.frame_index),
            ),
        )
        selected.append(best)
        remaining = [p for p in remaining if int(p.frame_index) != int(best.frame_index)]

    selected = sorted({int(p.frame_index): p for p in selected}.values(), key=lambda p: int(p.frame_index))
    if len(selected) > target_max_keep:
        pick = np.linspace(0, len(selected) - 1, num=target_max_keep, dtype=np.float64)
        selected = [selected[int(round(v))] for v in pick.tolist()]
        selected = sorted({int(p.frame_index): p for p in selected}.values(), key=lambda p: int(p.frame_index))
        if int(selected[0].frame_index) != int(ordered[0].frame_index):
            selected[0] = ordered[0]
        if int(selected[-1].frame_index) != int(ordered[-1].frame_index):
            selected[-1] = ordered[-1]

    return selected, {
        "applied": 1.0,
        "before_count": float(len(ordered)),
        "after_count": float(len(selected)),
        "cell_size": float(cell_size),
        "min_dist": float(min_dist),
        "target_min_keep": float(target_min_keep),
        "target_max_keep": float(target_max_keep),
    }


def select_required_poses(poses: List[PoseRecord], cfg: PoseSelectionConfig) -> Dict[str, object]:
    ordered_all = sorted(poses, key=lambda p: (int(p.frame_index), str(p.filename)))
    if not ordered_all:
        return {
            "selected": [],
            "stats": {
                "trajectory_count": 0,
                "selected_count": 0,
                "trajectory_frame_count": 0,
                "selected_frame_count": 0,
                "median_step": 1.0,
                "mean_translation_norm": 0.0,
                "mean_rotation_deg": 0.0,
            },
        }

    by_frame: Dict[int, List[PoseRecord]] = {}
    for pose in ordered_all:
        by_frame.setdefault(int(pose.frame_index), []).append(pose)
    ordered = [_representative_pose(by_frame[idx]) for idx in sorted(by_frame.keys())]

    median_step = _median_step_norm(ordered)
    selected_frames: List[PoseRecord] = [ordered[0]]
    trans_hist: List[float] = []
    rot_hist: List[float] = []

    if cfg.enable_translation or cfg.enable_rotation or cfg.enable_observations:
        for pose in ordered[1:]:
            last = selected_frames[-1]
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
                selected_frames.append(pose)
                trans_hist.append(step_norm)
                rot_hist.append(rot_deg)
    else:
        selected_frames = list(ordered)

    spatial_stats: Dict[str, float] = {"applied": 0.0}
    if bool(cfg.enable_spatial_post_filter):
        selected_frames, spatial_stats = _apply_spatial_post_filter(selected_frames, cfg)

    selected_frame_set = {int(p.frame_index) for p in selected_frames}
    selected = [p for p in ordered_all if int(p.frame_index) in selected_frame_set]

    return {
        "selected": selected,
        "stats": {
            "trajectory_count": len(ordered_all),
            "selected_count": len(selected),
            "trajectory_frame_count": len(ordered),
            "selected_frame_count": len(selected_frame_set),
            "median_step": median_step,
            "mean_translation_norm": float(np.mean(np.asarray(trans_hist, dtype=np.float64))) if trans_hist else 0.0,
            "mean_rotation_deg": float(np.mean(np.asarray(rot_hist, dtype=np.float64))) if rot_hist else 0.0,
            "spatial_post_filter": dict(spatial_stats),
        },
    }
