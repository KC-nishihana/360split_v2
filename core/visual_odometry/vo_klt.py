from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .calibration import CalibrationData


@dataclass
class VOMetrics:
    vo_valid: bool
    match_count: int
    tracked_count: int
    inlier_ratio: float
    vo_confidence: float
    feature_uniformity: float
    track_sufficiency: float
    pose_plausibility: float
    rotation_delta_deg: float
    translation_delta_rel: float
    step_proxy: float
    t_dir: List[float]
    r_rel_q_wxyz: List[float]
    essential_method_used: str


def _invalid_metrics() -> VOMetrics:
    return VOMetrics(
        vo_valid=False,
        match_count=0,
        tracked_count=0,
        inlier_ratio=0.0,
        vo_confidence=0.0,
        feature_uniformity=0.0,
        track_sufficiency=0.0,
        pose_plausibility=0.0,
        rotation_delta_deg=0.0,
        translation_delta_rel=0.0,
        step_proxy=0.0,
        t_dir=[0.0, 0.0, 0.0],
        r_rel_q_wxyz=[1.0, 0.0, 0.0, 0.0],
        essential_method_used="none",
    )


def _rotation_matrix_to_quaternion_wxyz(r: np.ndarray) -> List[float]:
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


class KLTVisualOdometry:
    def __init__(
        self,
        max_features: int = 600,
        quality_level: float = 0.01,
        min_distance: float = 8.0,
        min_track_points: int = 24,
        ransac_threshold: float = 1.0,
        center_roi_ratio: float = 0.6,
        downscale_long_edge: int = 1000,
        fast_fail_inlier_ratio: float = 0.12,
        step_proxy_clip_px: float = 80.0,
        essential_method: str = "auto",
        subpixel_refine: bool = True,
    ):
        self.max_features = int(max(50, max_features))
        self.quality_level = float(max(1e-4, quality_level))
        self.min_distance = float(max(1.0, min_distance))
        self.min_track_points = int(max(8, min_track_points))
        self.ransac_threshold = float(max(0.1, ransac_threshold))
        self.center_roi_ratio = float(np.clip(center_roi_ratio, 0.05, 1.0))
        self.downscale_long_edge = int(max(0, downscale_long_edge))
        self.fast_fail_inlier_ratio = float(np.clip(fast_fail_inlier_ratio, 0.0, 1.0))
        self.step_proxy_clip_px = float(max(0.0, step_proxy_clip_px))
        essential_mode = str(essential_method or "auto").strip().lower()
        self.essential_method = essential_mode if essential_mode in {"auto", "ransac", "magsac"} else "auto"
        self.subpixel_refine = bool(subpixel_refine)
        self._has_magsac = bool(hasattr(cv2, "USAC_MAGSAC"))
        self._prev_step_proxy: Optional[float] = None

    def _resize_if_needed(
        self,
        frame: np.ndarray,
        calibration: Optional[CalibrationData],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = frame.shape[:2]
        long_edge = max(h, w)
        if self.downscale_long_edge <= 0 or long_edge <= self.downscale_long_edge:
            if calibration is None:
                return frame, None
            return frame, np.asarray(calibration.camera_matrix, dtype=np.float64).copy()

        scale = float(self.downscale_long_edge) / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if calibration is None:
            return resized, None
        k = np.asarray(calibration.camera_matrix, dtype=np.float64).copy()
        k[0, 0] *= scale
        k[1, 1] *= scale
        k[0, 2] *= scale
        k[1, 2] *= scale
        return resized, k

    def _build_center_mask(self, shape: Tuple[int, int], ratio: float) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        radius = int(max(1.0, min(w, h) * 0.5 * float(np.clip(ratio, 0.05, 1.0))))
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask

    def _undistort_points(
        self,
        pts: np.ndarray,
        calibration: CalibrationData,
        k_use: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        dist = np.asarray(calibration.dist, dtype=np.float64).reshape(-1, 1)
        if calibration.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, k_use, dist, P=k_use)
        else:
            out = cv2.undistortPoints(pts, k_use, dist, P=k_use)
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _feature_uniformity(points_xy: np.ndarray, shape: Tuple[int, int]) -> float:
        if points_xy is None or len(points_xy) <= 0:
            return 0.0
        h, w = shape
        if h <= 1 or w <= 1:
            return 0.0
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        q1 = np.logical_and(pts[:, 0] < (w * 0.5), pts[:, 1] < (h * 0.5))
        q2 = np.logical_and(pts[:, 0] >= (w * 0.5), pts[:, 1] < (h * 0.5))
        q3 = np.logical_and(pts[:, 0] < (w * 0.5), pts[:, 1] >= (h * 0.5))
        q4 = np.logical_and(pts[:, 0] >= (w * 0.5), pts[:, 1] >= (h * 0.5))
        counts = np.asarray(
            [
                float(np.count_nonzero(q1)),
                float(np.count_nonzero(q2)),
                float(np.count_nonzero(q3)),
                float(np.count_nonzero(q4)),
            ],
            dtype=np.float64,
        )
        total = float(np.sum(counts))
        if total <= 0.0:
            return 0.0
        p = counts / total
        p = p[p > 1e-12]
        if p.size == 0:
            return 0.0
        entropy = float(-np.sum(p * np.log(p)))
        norm = float(np.log(4.0))
        if norm <= 1e-12:
            return 0.0
        return float(np.clip(entropy / norm, 0.0, 1.0))

    def _compute_pose_plausibility(self, rot_deg: float, step_proxy: float) -> float:
        rot_term = float(np.clip(1.0 - abs(float(rot_deg)) / 30.0, 0.0, 1.0))
        if self._prev_step_proxy is None or self._prev_step_proxy <= 1e-12 or step_proxy <= 1e-12:
            step_term = 0.5
        else:
            step_term = float(np.clip(1.0 - abs(np.log(step_proxy) - np.log(self._prev_step_proxy)) / 1.0, 0.0, 1.0))
        self._prev_step_proxy = float(max(0.0, step_proxy))
        return float(np.clip(0.5 * rot_term + 0.5 * step_term, 0.0, 1.0))

    def _find_essential_with_fallback(
        self,
        pts1_u: np.ndarray,
        pts2_u: np.ndarray,
        k_essential: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        total = max(1, int(len(pts1_u)))

        def _run(method_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
            if method_name == "magsac":
                if not self._has_magsac:
                    return None, None, 0
                method_flag = int(getattr(cv2, "USAC_MAGSAC"))
            else:
                method_flag = int(cv2.RANSAC)
            try:
                e_local, mask_local = cv2.findEssentialMat(
                    pts1_u,
                    pts2_u,
                    cameraMatrix=k_essential,
                    method=method_flag,
                    prob=0.999,
                    threshold=self.ransac_threshold,
                )
            except Exception:
                return None, None, 0
            if e_local is None or mask_local is None:
                return None, None, 0
            inliers_local = int(np.count_nonzero(mask_local))
            return e_local, mask_local, inliers_local

        mode = self.essential_method
        if mode == "ransac":
            e, mask, _ = _run("ransac")
            return e, mask, "ransac"
        if mode == "magsac":
            e, mask, _ = _run("magsac")
            if e is not None and mask is not None:
                return e, mask, "magsac"
            e, mask, _ = _run("ransac")
            return e, mask, "ransac_fallback"

        e_r, m_r, in_r = _run("ransac")
        if e_r is None or m_r is None:
            e_m, m_m, _ = _run("magsac")
            if e_m is not None and m_m is not None:
                return e_m, m_m, "magsac_auto"
            return None, None, "none"
        ratio_r = float(in_r) / float(total)
        if ratio_r >= 0.30:
            return e_r, m_r, "ransac_auto"
        e_m, m_m, in_m = _run("magsac")
        if e_m is not None and m_m is not None and int(in_m) >= int(in_r):
            return e_m, m_m, "magsac_auto"
        return e_r, m_r, "ransac_auto"

    def estimate(
        self,
        frame_prev: np.ndarray,
        frame_cur: np.ndarray,
        calibration: Optional[CalibrationData] = None,
        center_roi_ratio: Optional[float] = None,
    ) -> VOMetrics:
        if frame_prev is None or frame_cur is None:
            return _invalid_metrics()

        roi_ratio = self.center_roi_ratio if center_roi_ratio is None else float(center_roi_ratio)
        prev_rs, k_use = self._resize_if_needed(frame_prev, calibration)
        cur_rs, _ = self._resize_if_needed(frame_cur, calibration)
        if prev_rs is None or cur_rs is None:
            return _invalid_metrics()

        gray1 = cv2.cvtColor(prev_rs, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cur_rs, cv2.COLOR_BGR2GRAY)
        mask = self._build_center_mask(gray1.shape[:2], ratio=roi_ratio)
        corners = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7,
        )
        if corners is None or len(corners) < self.min_track_points:
            return _invalid_metrics()

        if self.subpixel_refine:
            try:
                corners = cv2.cornerSubPix(
                    gray1,
                    corners.astype(np.float32),
                    winSize=(5, 5),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                )
            except Exception:
                pass

        pts2, status, _ = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            corners,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if pts2 is None or status is None:
            return _invalid_metrics()

        status = status.reshape(-1).astype(bool)
        pts1 = corners.reshape(-1, 2)[status]
        pts2 = pts2.reshape(-1, 2)[status]
        if len(pts1) < self.min_track_points:
            return _invalid_metrics()

        if self.subpixel_refine:
            try:
                pts2_refined = cv2.cornerSubPix(
                    gray2,
                    pts2.reshape(-1, 1, 2).astype(np.float32),
                    winSize=(5, 5),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                )
                if pts2_refined is not None:
                    pts2 = pts2_refined.reshape(-1, 2)
            except Exception:
                pass

        if calibration is not None and k_use is not None:
            pts1_u = self._undistort_points(pts1, calibration=calibration, k_use=k_use)
            pts2_u = self._undistort_points(pts2, calibration=calibration, k_use=k_use)
            k_essential = np.asarray(k_use, dtype=np.float64)
        else:
            pts1_u = np.asarray(pts1, dtype=np.float64)
            pts2_u = np.asarray(pts2, dtype=np.float64)
            h, w = gray1.shape[:2]
            fx = fy = float(max(h, w))
            cx = float(w * 0.5)
            cy = float(h * 0.5)
            k_essential = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        e, mask_e, method_used = self._find_essential_with_fallback(pts1_u, pts2_u, k_essential)
        if e is None or mask_e is None or int(mask_e.sum()) < self.min_track_points:
            return _invalid_metrics()

        try:
            _, r, t, mask_pose = cv2.recoverPose(e, pts1_u, pts2_u, k_essential, mask=mask_e)
        except Exception:
            return _invalid_metrics()
        if r is None or t is None or mask_pose is None:
            return _invalid_metrics()

        inlier_count = int(np.count_nonzero(mask_pose))
        total_count = int(len(pts1_u))
        if inlier_count <= 0 or total_count <= 0:
            return _invalid_metrics()
        tracked_count = int(len(pts1))
        inlier_ratio = float(np.clip(inlier_count / float(total_count), 0.0, 1.0))
        if inlier_ratio < self.fast_fail_inlier_ratio:
            return _invalid_metrics()
        inlier_mask = np.asarray(mask_pose).reshape(-1).astype(bool)

        trace_val = float(np.trace(r))
        cos_theta = float(np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0))
        rot_deg = float(np.degrees(np.arccos(cos_theta)))
        t_vec = np.asarray(t, dtype=np.float64).reshape(3)
        t_norm = float(np.linalg.norm(t_vec))
        if t_norm > 1e-12:
            t_dir = (t_vec / t_norm).tolist()
        else:
            t_dir = [0.0, 0.0, 0.0]
        if np.count_nonzero(inlier_mask) > 0:
            delta = np.asarray(pts2[inlier_mask] - pts1[inlier_mask], dtype=np.float64)
            step_proxy = float(np.median(np.linalg.norm(delta, axis=1)))
        else:
            step_proxy = 0.0
        if self.step_proxy_clip_px > 0.0:
            step_proxy = min(step_proxy, self.step_proxy_clip_px)
        q_wxyz = _rotation_matrix_to_quaternion_wxyz(r)
        uniformity = self._feature_uniformity(pts1[inlier_mask] if np.count_nonzero(inlier_mask) > 0 else pts1, gray1.shape[:2])
        track_sufficiency = float(np.clip(float(inlier_count) / float(max(self.min_track_points, 1)), 0.0, 1.0))
        pose_plausibility = self._compute_pose_plausibility(rot_deg, float(max(0.0, step_proxy)))
        vo_confidence = float(
            np.clip(
                0.4 * inlier_ratio + 0.2 * uniformity + 0.2 * track_sufficiency + 0.2 * pose_plausibility,
                0.0,
                1.0,
            )
        )

        return VOMetrics(
            vo_valid=True,
            match_count=inlier_count,
            tracked_count=tracked_count,
            inlier_ratio=inlier_ratio,
            vo_confidence=vo_confidence,
            feature_uniformity=uniformity,
            track_sufficiency=track_sufficiency,
            pose_plausibility=pose_plausibility,
            rotation_delta_deg=rot_deg,
            translation_delta_rel=t_norm,
            step_proxy=float(max(0.0, step_proxy)),
            t_dir=[float(x) for x in t_dir],
            r_rel_q_wxyz=q_wxyz,
            essential_method_used=method_used,
        )
