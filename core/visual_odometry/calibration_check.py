from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import CalibrationData, calibration_to_dict, summarize_calibration


def _undistort(frame: np.ndarray, calib: CalibrationData) -> np.ndarray:
    k = np.asarray(calib.camera_matrix, dtype=np.float64)
    d = np.asarray(calib.dist, dtype=np.float64).reshape(-1, 1)
    if calib.model == "fisheye":
        return cv2.fisheye.undistortImage(frame, k, d, Knew=k)
    return cv2.undistort(frame, k, d)


def _center_circle_mask(shape: Tuple[int, int], roi_ratio: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    cy = h // 2
    radius = int(max(1.0, min(w, h) * 0.5 * float(np.clip(roi_ratio, 0.05, 1.0))))
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def _line_length_score(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    edges = cv2.Canny(gray, 60, 180)
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=30,
        maxLineGap=6,
    )
    if lines is None:
        return 0.0
    total = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        total += float(np.hypot(x2 - x1, y2 - y1))
    return total


def _score_frame(orig: np.ndarray, undist: np.ndarray, roi_ratio: float) -> Dict[str, float]:
    gray_o = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_u = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    center_mask = _center_circle_mask(gray_o.shape[:2], roi_ratio=roi_ratio)

    full_before = _line_length_score(gray_o, None)
    full_after = _line_length_score(gray_u, None)
    center_before = _line_length_score(gray_o, center_mask)
    center_after = _line_length_score(gray_u, center_mask)

    def _ratio(after: float, before: float) -> float:
        if before <= 1e-9:
            return 0.0 if after <= 1e-9 else 999.0
        return after / before

    return {
        "full_before": float(full_before),
        "full_after": float(full_after),
        "full_ratio": float(_ratio(full_after, full_before)),
        "center_before": float(center_before),
        "center_after": float(center_after),
        "center_ratio": float(_ratio(center_after, center_before)),
    }


def _read_frame_at(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _sample_indices(video_path: str, frame_index: Optional[int]) -> List[int]:
    if frame_index is not None:
        return [max(0, int(frame_index))]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [0]
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if count <= 0:
        return [0]
    return sorted(set([0, count // 4, count // 2, (count * 3) // 4, max(0, count - 1)]))


def _save_compare(orig: np.ndarray, undist: np.ndarray, compare_path: Path) -> None:
    h1, w1 = orig.shape[:2]
    h2, w2 = undist.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = orig
    canvas[:h2, w1:w1 + w2] = undist
    cv2.putText(canvas, "orig", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, "undist", (w1 + 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(str(compare_path), canvas)


def run_calibration_check(
    out_dir: str,
    video_path: Optional[str] = None,
    calib: Optional[CalibrationData] = None,
    front_video_path: Optional[str] = None,
    rear_video_path: Optional[str] = None,
    front_calib: Optional[CalibrationData] = None,
    rear_calib: Optional[CalibrationData] = None,
    frame_index: Optional[int] = None,
    roi_ratio: float = 0.6,
    logger=None,
) -> Dict:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report: Dict = {
        "note": "Calibration check is a visual/reference aid only; it does not prove calibration correctness.",
        "roi_ratio": float(np.clip(roi_ratio, 0.05, 1.0)),
        "mode": "",
        "entries": [],
    }

    if video_path:
        report["mode"] = "monocular"
        report["calibration"] = summarize_calibration(calib)
        indices = _sample_indices(video_path, frame_index)
        for idx in indices:
            frame = _read_frame_at(video_path, idx)
            if frame is None:
                continue
            undist = _undistort(frame, calib) if calib is not None else frame.copy()
            prefix = f"frame_{idx:06d}"
            orig_path = output / f"{prefix}_orig.png"
            und_path = output / f"{prefix}_undist.png"
            cmp_path = output / f"{prefix}_compare.png"
            cv2.imwrite(str(orig_path), frame)
            cv2.imwrite(str(und_path), undist)
            _save_compare(frame, undist, cmp_path)
            scores = _score_frame(frame, undist, roi_ratio=report["roi_ratio"])
            report["entries"].append(
                {
                    "frame_index": int(idx),
                    "orig_path": str(orig_path),
                    "undist_path": str(und_path),
                    "compare_path": str(cmp_path),
                    "scores": scores,
                }
            )
    else:
        report["mode"] = "front_rear"
        report["front_calibration"] = summarize_calibration(front_calib)
        report["rear_calibration"] = summarize_calibration(rear_calib)
        base_video = front_video_path or rear_video_path or ""
        indices = _sample_indices(base_video, frame_index)
        for idx in indices:
            pair_items = [
                ("front", front_video_path, front_calib),
                ("rear", rear_video_path, rear_calib),
            ]
            for label, path, c in pair_items:
                if not path:
                    continue
                frame = _read_frame_at(path, idx)
                if frame is None:
                    continue
                undist = _undistort(frame, c) if c is not None else frame.copy()
                prefix = f"{label}_frame_{idx:06d}"
                orig_path = output / f"{prefix}_orig.png"
                und_path = output / f"{prefix}_undist.png"
                cmp_path = output / f"{prefix}_compare.png"
                cv2.imwrite(str(orig_path), frame)
                cv2.imwrite(str(und_path), undist)
                _save_compare(frame, undist, cmp_path)
                scores = _score_frame(frame, undist, roi_ratio=report["roi_ratio"])
                report["entries"].append(
                    {
                        "camera": label,
                        "frame_index": int(idx),
                        "orig_path": str(orig_path),
                        "undist_path": str(und_path),
                        "compare_path": str(cmp_path),
                        "scores": scores,
                        "calibration": calibration_to_dict(c),
                    }
                )

    report_path = output / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if logger is not None:
        logger.info(f"calibration check report: {report_path}")
    report["report_path"] = str(report_path)
    return report

