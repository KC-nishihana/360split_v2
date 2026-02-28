from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base import PoseEstimator
from .types import PoseEstimationResult, PoseRecord

_FRAME_PAT = re.compile(r"keyframe_(\d+)")


def _parse_frame_idx(name: str) -> int:
    m = _FRAME_PAT.search(name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1


def _pose_from_metric_row(filename: str, metrics_row: Dict[str, Any]) -> Optional[PoseRecord]:
    t_xyz = metrics_row.get("t_xyz")
    q_wxyz = metrics_row.get("q_wxyz")
    if not (isinstance(t_xyz, list) and len(t_xyz) == 3 and isinstance(q_wxyz, list) and len(q_wxyz) == 4):
        return None
    frame_idx = int(metrics_row.get("frame_index", _parse_frame_idx(filename)))
    m = metrics_row.get("metrics", {}) if isinstance(metrics_row.get("metrics"), dict) else {}
    return PoseRecord(
        frame_index=frame_idx,
        filename=filename,
        qw=float(q_wxyz[0]),
        qx=float(q_wxyz[1]),
        qy=float(q_wxyz[2]),
        qz=float(q_wxyz[3]),
        tx=float(t_xyz[0]),
        ty=float(t_xyz[1]),
        tz=float(t_xyz[2]),
        confidence=float(m.get("vo_confidence", 0.0)),
        observations=0,
    )


def _iter_vo_csv(path: Path) -> Iterable[PoseRecord]:
    if not path.exists():
        return []
    out: List[PoseRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row.get("frame_idx", "0"))
                q_w = row.get("q_w")
                q_x = row.get("q_x")
                q_y = row.get("q_y")
                q_z = row.get("q_z")
                t_x = row.get("t_x")
                t_y = row.get("t_y")
                t_z = row.get("t_z")
                if any(v in {"", None} for v in (q_w, q_x, q_y, q_z, t_x, t_y, t_z)):
                    continue
                out.append(
                    PoseRecord(
                        frame_index=frame_idx,
                        filename=f"frame_{frame_idx:06d}.png",
                        qw=float(q_w),
                        qx=float(q_x),
                        qy=float(q_y),
                        qz=float(q_z),
                        tx=float(t_x),
                        ty=float(t_y),
                        tz=float(t_z),
                        confidence=float(row.get("vo_confidence", 0.0) or 0.0),
                        observations=0,
                    )
                )
            except Exception:
                continue
    return out


class VOPoseEstimator(PoseEstimator):
    def estimate(self, image_dir: str, context: Optional[Dict[str, Any]] = None) -> PoseEstimationResult:
        ctx = dict(context or {})
        poses: List[PoseRecord] = []

        exported_entries = list(ctx.get("exported_entries", []) or [])
        frame_metrics_map: Dict[int, Dict[str, Any]] = dict(ctx.get("frame_metrics_map", {}) or {})

        for entry in exported_entries:
            filename = str(entry.get("filename", "") or "")
            frame_idx = int(entry.get("frame_index", _parse_frame_idx(filename)))
            metrics_row = frame_metrics_map.get(frame_idx)
            if not metrics_row:
                continue
            pose = _pose_from_metric_row(filename, metrics_row)
            if pose is not None:
                poses.append(pose)

        if not poses:
            vo_csv = Path(str(ctx.get("vo_trajectory_csv", "") or "").strip())
            poses = list(_iter_vo_csv(vo_csv)) if vo_csv else []

        poses.sort(key=lambda p: (int(p.frame_index), str(p.filename)))
        diagnostics = {
            "source": "frame_metrics" if exported_entries and frame_metrics_map else "vo_trajectory_csv",
            "pose_count": len(poses),
        }
        return PoseEstimationResult(poses=poses, backend="vo", diagnostics=diagnostics, raw_log_paths={})
