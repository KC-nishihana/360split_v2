"""Per-frame quality metrics computation."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.quality_score import compute_raw_metrics
from core.adaptive_selector import AdaptiveSelector

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Quality metrics for a single projected image."""

    frame_idx: int
    timestamp: float
    stream: str
    direction: str
    laplacian_var: float
    mean_intensity: float
    clipped_high_ratio: float
    clipped_low_ratio: float
    exposure_score: float
    orb_keypoints: int
    ssim_prev: float  # SSIM with previous frame (same stream/direction)

    @property
    def filename(self) -> str:
        return f"{self.stream}_{self.direction}_{self.frame_idx:08d}.jpg"


METRICS_CSV_FIELDS = [f.name for f in fields(FrameMetrics)]


class MetricsComputer:
    """Compute quality metrics for projected pinhole images."""

    def __init__(self, orb_features: int = 2000, ssim_scale: float = 0.5) -> None:
        self._orb = cv2.ORB_create(nfeatures=orb_features)
        self._ssim_computer = AdaptiveSelector(ssim_scale=ssim_scale)
        self._prev_images: Dict[str, np.ndarray] = {}  # key: "stream_direction"

    def reset(self) -> None:
        """Clear previous frame cache."""
        self._prev_images.clear()

    def compute_frame(
        self,
        images: Dict[str, np.ndarray],
        frame_idx: int,
        timestamp: float,
        stream: str,
    ) -> List[FrameMetrics]:
        """Compute metrics for all directions of one frame.

        Parameters
        ----------
        images : dict mapping direction name to projected BGR image
        frame_idx : frame index
        timestamp : frame timestamp in seconds
        stream : "front" or "back"

        Returns
        -------
        List of FrameMetrics, one per direction
        """
        results: List[FrameMetrics] = []

        for direction, img in images.items():
            # Compute raw quality metrics (reuse existing)
            raw = compute_raw_metrics(img, roi_spec=None, use_orb=True, orb=self._orb)

            # Compute additional exposure details
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            gray_f = gray.astype(np.float64)
            total_pixels = max(gray_f.size, 1)

            mean_intensity = float(np.mean(gray_f))
            clipped_high_ratio = float(np.sum(gray_f >= 245.0) / total_pixels)
            clipped_low_ratio = float(np.sum(gray_f <= 16.0) / total_pixels)

            # SSIM with previous frame of same stream/direction
            prev_key = f"{stream}_{direction}"
            prev_img = self._prev_images.get(prev_key)
            if prev_img is not None:
                ssim_prev = float(self._ssim_computer.compute_ssim(prev_img, img))
            else:
                ssim_prev = 0.0  # No previous frame

            # Update previous frame cache
            self._prev_images[prev_key] = img

            metrics = FrameMetrics(
                frame_idx=frame_idx,
                timestamp=timestamp,
                stream=stream,
                direction=direction,
                laplacian_var=float(raw["laplacian_var"]),
                mean_intensity=mean_intensity,
                clipped_high_ratio=clipped_high_ratio,
                clipped_low_ratio=clipped_low_ratio,
                exposure_score=float(raw["exposure"]),
                orb_keypoints=int(raw["orb_keypoints"]),
                ssim_prev=ssim_prev,
            )
            results.append(metrics)

        return results


def write_metrics_csv(metrics: List[FrameMetrics], path: Path) -> None:
    """Write metrics_all.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        for m in metrics:
            row = {
                "frame_idx": m.frame_idx,
                "timestamp": f"{m.timestamp:.4f}",
                "stream": m.stream,
                "direction": m.direction,
                "laplacian_var": f"{m.laplacian_var:.2f}",
                "mean_intensity": f"{m.mean_intensity:.2f}",
                "clipped_high_ratio": f"{m.clipped_high_ratio:.6f}",
                "clipped_low_ratio": f"{m.clipped_low_ratio:.6f}",
                "exposure_score": f"{m.exposure_score:.4f}",
                "orb_keypoints": m.orb_keypoints,
                "ssim_prev": f"{m.ssim_prev:.6f}",
            }
            writer.writerow(row)
    logger.info(f"Wrote {len(metrics)} metric records to {path}")


def load_metrics_csv(path: Path) -> List[FrameMetrics]:
    """Load metrics_all.csv back into FrameMetrics list."""
    results: List[FrameMetrics] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(FrameMetrics(
                frame_idx=int(row["frame_idx"]),
                timestamp=float(row["timestamp"]),
                stream=row["stream"],
                direction=row["direction"],
                laplacian_var=float(row["laplacian_var"]),
                mean_intensity=float(row["mean_intensity"]),
                clipped_high_ratio=float(row["clipped_high_ratio"]),
                clipped_low_ratio=float(row["clipped_low_ratio"]),
                exposure_score=float(row["exposure_score"]),
                orb_keypoints=int(row["orb_keypoints"]),
                ssim_prev=float(row["ssim_prev"]),
            ))
    return results
