from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np


@dataclass
class TrajectorySample:
    frame_idx: int
    vo_valid: bool
    t_dir: List[float]
    step_proxy: float
    r_rel_q_wxyz: List[float]


def _quat_wxyz_to_rotmat(q_wxyz: List[float]) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (q / n).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_wxyz(r: np.ndarray) -> List[float]:
    m = np.asarray(r, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    q = q / n
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _as_sample(item: object) -> TrajectorySample:
    if isinstance(item, Mapping):
        m = item
        return TrajectorySample(
            frame_idx=int(m.get("frame_idx", 0)),
            vo_valid=bool(m.get("vo_valid", False)),
            t_dir=[float(x) for x in m.get("t_dir", [0.0, 0.0, 0.0])],
            step_proxy=float(m.get("step_proxy", 0.0)),
            r_rel_q_wxyz=[float(x) for x in m.get("r_rel_q_wxyz", [1.0, 0.0, 0.0, 0.0])],
        )
    return TrajectorySample(
        frame_idx=int(getattr(item, "frame_idx", 0)),
        vo_valid=bool(getattr(item, "vo_valid", False)),
        t_dir=[float(x) for x in getattr(item, "t_dir", [0.0, 0.0, 0.0])],
        step_proxy=float(getattr(item, "step_proxy", 0.0)),
        r_rel_q_wxyz=[float(x) for x in getattr(item, "r_rel_q_wxyz", [1.0, 0.0, 0.0, 0.0])],
    )


def integrate_relative_trajectory(
    samples: Iterable[object],
    t_sign: float = 1.0,
) -> Dict[int, Dict[str, object]]:
    seq = [_as_sample(x) for x in samples]
    if not seq:
        return {}

    valid_steps = [float(max(0.0, s.step_proxy)) for s in seq if s.vo_valid and float(s.step_proxy) > 0.0]
    step_median = float(np.median(np.asarray(valid_steps, dtype=np.float64))) if valid_steps else 1.0
    if step_median <= 1e-12:
        step_median = 1.0

    r_world = np.eye(3, dtype=np.float64)
    p_world = np.zeros(3, dtype=np.float64)
    out: MutableMapping[int, Dict[str, object]] = {}
    for sample in seq:
        step_norm = 0.0
        if sample.vo_valid:
            r_world = r_world @ _quat_wxyz_to_rotmat(sample.r_rel_q_wxyz)
            t_dir = np.asarray(sample.t_dir, dtype=np.float64).reshape(3)
            t_norm = float(np.linalg.norm(t_dir))
            if t_norm > 1e-12:
                d_world = r_world @ ((float(t_sign) / t_norm) * t_dir)
                step_norm = float(max(0.0, sample.step_proxy) / step_median)
                p_world = p_world + step_norm * d_world
        out[int(sample.frame_idx)] = {
            "t_xyz": [float(p_world[0]), float(p_world[1]), float(p_world[2])],
            "q_wxyz": _rotmat_to_quat_wxyz(r_world),
            "step_norm": float(step_norm),
        }
    return dict(out)

