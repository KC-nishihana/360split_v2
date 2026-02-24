"""
OSV/dual-fisheye cross5 split helper.

This module centralizes the split logic that was prototyped in
`test/extract_views_dualstream_cross5.py` so CLI/GUI can share it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from core.visual_odometry.calibration import calibration_from_dict


@dataclass
class Cross5SplitConfig:
    size: int = 1600
    hfov: float = 80.0
    vfov: float = 80.0
    cross_yaw_deg: float = 50.5
    cross_pitch_deg: float = 50.5
    cross_inward_deg: float = 10.0
    inward_up_deg: Optional[float] = 25.0
    inward_down_deg: Optional[float] = 25.0
    inward_left_deg: Optional[float] = 25.0
    inward_right_deg: Optional[float] = 25.0
    valid_ratio_thresh: float = 0.90
    mask_erode: int = 1
    circle_inner_ratio: float = 0.98


def _remap_view_label(name: str) -> str:
    # Keep behavior compatible with the cross5 prototype script.
    swapped = {
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
    }
    return swapped.get(name, name)


def _make_knew(out_w: int, out_h: int, h_fov_deg: float, v_fov_deg: float) -> np.ndarray:
    fx = out_w / (2.0 * math.tan(math.radians(h_fov_deg) / 2.0))
    fy = out_h / (2.0 * math.tan(math.radians(v_fov_deg) / 2.0))
    cx = out_w / 2.0
    cy = out_h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _r_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _build_cross5_views(cfg: Cross5SplitConfig):
    i_side = float(cfg.cross_inward_deg)
    i_up = i_side if cfg.inward_up_deg is None else float(cfg.inward_up_deg)
    i_down = i_side if cfg.inward_down_deg is None else float(cfg.inward_down_deg)
    i_left = i_side if cfg.inward_left_deg is None else float(cfg.inward_left_deg)
    i_right = i_side if cfg.inward_right_deg is None else float(cfg.inward_right_deg)
    right_yaw = max(0.0, float(cfg.cross_yaw_deg) - i_right)
    left_yaw = max(0.0, float(cfg.cross_yaw_deg) - i_left)
    up_pitch = max(0.0, float(cfg.cross_pitch_deg) - i_up)
    down_pitch = max(0.0, float(cfg.cross_pitch_deg) - i_down)
    return [
        ("front", 0.0, 0.0, 0.0),
        ("right", +right_yaw, 0.0, 0.0),
        ("left", -left_yaw, 0.0, 0.0),
        ("up", 0.0, +up_pitch, 0.0),
        ("down", 0.0, -down_pitch, 0.0),
    ]


def _undistort_map(
    k: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    knew: np.ndarray,
    out_size: Tuple[int, int],
    model: str,
) -> Tuple[np.ndarray, np.ndarray]:
    out_w, out_h = out_size
    if model == "fisheye":
        d4 = np.asarray(d, dtype=np.float64).reshape(-1)
        if d4.size != 4:
            raise ValueError(f"fisheye model expects 4 coeffs, got {d4.size}")
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            np.asarray(k, dtype=np.float64),
            d4.reshape(4, 1),
            r,
            knew,
            (out_w, out_h),
            m1type=cv2.CV_32FC1,
        )
        return map1, map2
    return cv2.initUndistortRectifyMap(
        np.asarray(k, dtype=np.float64),
        np.asarray(d, dtype=np.float64).reshape(-1),
        r,
        knew,
        (out_w, out_h),
        m1type=cv2.CV_32FC1,
    )


def _make_valid_mask(
    mapx: np.ndarray,
    mapy: np.ndarray,
    src_w: int,
    src_h: int,
    cfg: Cross5SplitConfig,
    circle_cx: float,
    circle_cy: float,
    circle_radius: float,
) -> np.ndarray:
    finite = np.isfinite(mapx) & np.isfinite(mapy)
    inside_rect = finite & (mapx >= 0.0) & (mapx <= (src_w - 1)) & (mapy >= 0.0) & (mapy <= (src_h - 1))

    inner = max(0.1, min(1.0, float(cfg.circle_inner_ratio)))
    use_r = max(1.0, float(circle_radius) * inner)
    inside_circle = np.zeros_like(finite, dtype=bool)
    if np.any(finite):
        mapx64 = mapx[finite].astype(np.float64)
        mapy64 = mapy[finite].astype(np.float64)
        dx = mapx64 - float(circle_cx)
        dy = mapy64 - float(circle_cy)
        inside_circle_finite = (dx * dx + dy * dy) <= (use_r * use_r)
        inside_circle[finite] = inside_circle_finite

    mask = (inside_rect & inside_circle).astype(np.uint8)
    if cfg.mask_erode > 0:
        k = max(1, int(cfg.mask_erode))
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)

    k_local = max(3, int(cfg.mask_erode) * 2 + 1)
    if (k_local % 2) == 0:
        k_local += 1
    local_ratio = cv2.blur(mask.astype(np.float32), (k_local, k_local))
    return (local_ratio >= float(cfg.valid_ratio_thresh)).astype(np.uint8) * 255


class Cross5FisheyeSplitter:
    def __init__(self, calibration: dict, cfg: Optional[Cross5SplitConfig] = None):
        calib = calibration_from_dict(calibration)
        if calib is None:
            raise ValueError("valid calibration is required for cross5 split")
        self.k = np.asarray(calib.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.d = np.asarray(calib.dist, dtype=np.float64).reshape(-1)
        self.model = str(calib.model or "auto").strip().lower()
        if self.model == "auto":
            self.model = "fisheye" if self.d.size == 4 else "opencv"
        self.cfg = cfg or Cross5SplitConfig()
        self._maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._last_shape: Optional[Tuple[int, int]] = None

    def _ensure_maps(self, src_h: int, src_w: int) -> None:
        key_shape = (src_h, src_w)
        if self._last_shape == key_shape and self._maps:
            return
        out_size = (int(self.cfg.size), int(self.cfg.size))
        knew = _make_knew(out_size[0], out_size[1], self.cfg.hfov, self.cfg.vfov)
        maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, yaw, pitch, roll in _build_cross5_views(self.cfg):
            r = _r_from_ypr(yaw, pitch, roll)
            maps[_remap_view_label(name)] = _undistort_map(self.k, self.d, r, knew, out_size, self.model)
        self._maps = maps
        self._last_shape = key_shape

    def split_image(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        src_h, src_w = frame.shape[:2]
        self._ensure_maps(src_h, src_w)
        out: Dict[str, np.ndarray] = {}
        for label, (mapx, mapy) in self._maps.items():
            out[label] = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out

    def split_image_with_valid_mask(self, frame: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        src_h, src_w = frame.shape[:2]
        self._ensure_maps(src_h, src_w)
        circle_cx = float(self.k[0, 2])
        circle_cy = float(self.k[1, 2])
        circle_radius = min(circle_cx, circle_cy, (src_w - 1) - circle_cx, (src_h - 1) - circle_cy)

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for label, (mapx, mapy) in self._maps.items():
            valid = _make_valid_mask(mapx, mapy, src_w, src_h, self.cfg, circle_cx, circle_cy, circle_radius)
            img = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            img = cv2.bitwise_and(img, img, mask=valid)
            out[label] = (img, valid)
        return out

    def project_mask(self, fisheye_mask: np.ndarray) -> Dict[str, np.ndarray]:
        src_h, src_w = fisheye_mask.shape[:2]
        self._ensure_maps(src_h, src_w)
        out: Dict[str, np.ndarray] = {}
        for label, (mapx, mapy) in self._maps.items():
            remapped = cv2.remap(
                fisheye_mask,
                mapx,
                mapy,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if remapped.ndim == 3:
                remapped = cv2.cvtColor(remapped, cv2.COLOR_BGR2GRAY)
            out[label] = (remapped > 127).astype(np.uint8) * 255
        return out
