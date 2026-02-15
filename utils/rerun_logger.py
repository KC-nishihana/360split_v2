"""
Rerun logging utilities for keyframe validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def _as_float32_vec3(value: Optional[Sequence[float]], fallback: Sequence[float]) -> np.ndarray:
    if value is None:
        return np.asarray(fallback, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        return np.asarray(fallback, dtype=np.float32)
    return arr


def _as_float32_quat_wxyz(value: Optional[Sequence[float]]) -> np.ndarray:
    if value is None:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 4:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return arr


class RerunKeyframeLogger:
    """
    Thin wrapper around rerun SDK with optional no-op fallback.
    """

    def __init__(
        self,
        app_id: str = "keyframe_check",
        spawn: bool = False,
        save_path: Optional[str] = None,
        timeline_name: str = "frame",
    ) -> None:
        self.timeline_name = timeline_name
        self._trajectory: list[np.ndarray] = []
        self._keyframes: list[np.ndarray] = []
        self._rr = None

        try:
            import rerun as rr  # type: ignore
            self._rr = rr
        except ImportError:
            logger.warning("rerun が見つかりません。Rerunログは無効化されます。`pip install rerun-sdk` を実行してください。")
            return

        self._rr.init(app_id, spawn=spawn)
        if save_path:
            save_target = Path(save_path)
            save_target.parent.mkdir(parents=True, exist_ok=True)
            self._rr.save(str(save_target))
            logger.info(f"Rerunログ保存先: {save_target}")

    @property
    def enabled(self) -> bool:
        return self._rr is not None

    def _set_frame_time(self, frame_idx: int) -> None:
        if self._rr is None:
            return
        self._rr.set_time(self.timeline_name, sequence=int(frame_idx))

    def _log_scalar(self, path: str, value: float) -> None:
        if self._rr is None:
            return
        if hasattr(self._rr, "Scalar"):
            self._rr.log(path, self._rr.Scalar(float(value)))
        else:
            self._rr.log_scalar(path, float(value))

    def _make_quaternion(self, q_wxyz: np.ndarray):
        if self._rr is None:
            return None
        try:
            return self._rr.Quaternion(wxyz=q_wxyz)
        except TypeError:
            # Fallback for SDK variants expecting XYZW.
            return self._rr.Quaternion(xyzw=np.asarray([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32))

    def log_frame(
        self,
        frame_idx: int,
        img: Optional[np.ndarray],
        t_xyz: Optional[Sequence[float]] = None,
        q_wxyz: Optional[Sequence[float]] = None,
        is_keyframe: bool = False,
        metrics: Optional[Dict[str, float]] = None,
        points_world: Optional[np.ndarray] = None,
    ) -> None:
        """
        Log one frame worth of entities into rerun.
        """
        if self._rr is None:
            return

        translation = _as_float32_vec3(t_xyz, fallback=(float(frame_idx), 0.0, 0.0))
        quat_wxyz = _as_float32_quat_wxyz(q_wxyz)

        self._set_frame_time(frame_idx)

        if img is not None:
            self._rr.log("cam/image", self._rr.Image(img))

        self._rr.log(
            "world/cam",
            self._rr.Transform3D(
                translation=translation,
                rotation=self._make_quaternion(quat_wxyz),
            ),
        )

        self._trajectory.append(translation.astype(np.float32))
        traj = np.asarray(self._trajectory, dtype=np.float32)
        traj_colors = np.tile(np.asarray([[70, 130, 255]], dtype=np.uint8), (traj.shape[0], 1))
        self._rr.log("world/trajectory", self._rr.Points3D(traj, colors=traj_colors))

        if is_keyframe:
            self._keyframes.append(translation.astype(np.float32))
        if self._keyframes:
            kf = np.asarray(self._keyframes, dtype=np.float32)
            kf_colors = np.tile(np.asarray([[255, 80, 80]], dtype=np.uint8), (kf.shape[0], 1))
            self._rr.log("world/keyframes", self._rr.Points3D(kf, colors=kf_colors))

        for name, value in (metrics or {}).items():
            if value is None:
                continue
            try:
                self._log_scalar(f"metrics/{name}", float(value))
            except (TypeError, ValueError):
                continue

        if points_world is not None:
            pts = np.asarray(points_world, dtype=np.float32)
            if pts.size > 0:
                self._rr.log("world/points", self._rr.Points3D(pts.reshape(-1, 3)))
