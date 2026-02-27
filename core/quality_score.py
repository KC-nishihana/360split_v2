"""Quality scoring utilities for Stage1 filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from core.accelerator import get_accelerator


@dataclass(frozen=True)
class QualityROISpec:
    mode: str = "circle"
    ratio: float = 0.40


DEFAULT_ROI = QualityROISpec(mode="circle", ratio=0.40)
DEFAULT_FIELDS: Tuple[str, ...] = ("laplacian_var", "tenengrad", "exposure", "orb_keypoints")


def parse_roi_spec(roi_spec: str | QualityROISpec | None) -> QualityROISpec:
    """Parse roi spec from 'circle:0.4' / 'rect:0.6'."""
    if isinstance(roi_spec, QualityROISpec):
        return QualityROISpec(
            mode=str(roi_spec.mode or "circle").strip().lower(),
            ratio=float(np.clip(roi_spec.ratio, 0.05, 1.0)),
        )
    if roi_spec is None:
        return DEFAULT_ROI
    text = str(roi_spec).strip().lower()
    if not text:
        return DEFAULT_ROI
    if ":" not in text:
        mode = text
        ratio = DEFAULT_ROI.ratio
    else:
        mode_text, ratio_text = text.split(":", 1)
        mode = mode_text.strip()
        try:
            ratio = float(ratio_text.strip())
        except (TypeError, ValueError):
            ratio = DEFAULT_ROI.ratio
    if mode not in {"circle", "rect"}:
        mode = "circle"
    ratio = float(np.clip(ratio, 0.05, 1.0))
    return QualityROISpec(mode=mode, ratio=ratio)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _build_roi_mask(gray: np.ndarray, spec: QualityROISpec) -> np.ndarray:
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return mask
    if spec.mode == "rect":
        ww = max(1, int(round(w * spec.ratio)))
        hh = max(1, int(round(h * spec.ratio)))
        x0 = max(0, (w - ww) // 2)
        y0 = max(0, (h - hh) // 2)
        x1 = min(w, x0 + ww)
        y1 = min(h, y0 + hh)
        mask[y0:y1, x0:x1] = 255
        return mask
    radius = max(1, int(round(min(w, h) * 0.5 * spec.ratio)))
    cx = w // 2
    cy = h // 2
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def _masked_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask > 0]
    if vals.size > 0:
        return vals
    return arr.reshape(-1)


def compute_raw_metrics(
    frame: np.ndarray,
    roi_spec: str | QualityROISpec | None = None,
    *,
    use_orb: bool = True,
    orb: Optional[cv2.ORB] = None,
) -> Dict[str, float]:
    """Compute raw quality metrics on ROI."""
    gray = _to_gray(frame)
    spec = parse_roi_spec(roi_spec)
    mask = _build_roi_mask(gray, spec)
    vals = _masked_values(gray.astype(np.float64), mask)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = _masked_values(lap, mask)
    lap_var = float(np.var(lap_vals)) if lap_vals.size > 0 else 0.0

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_map = (sobel_x * sobel_x) + (sobel_y * sobel_y)
    tenengrad_vals = _masked_values(tenengrad_map, mask)
    tenengrad = float(np.mean(tenengrad_vals)) if tenengrad_vals.size > 0 else 0.0

    black_clip = float(np.mean(vals <= 16.0)) if vals.size > 0 else 1.0
    white_clip = float(np.mean(vals >= 245.0)) if vals.size > 0 else 1.0
    contrast = float(np.std(vals)) if vals.size > 0 else 0.0
    contrast_norm = float(np.clip(contrast / 64.0, 0.0, 1.0))
    clip_penalty = float(np.clip(black_clip + white_clip, 0.0, 1.0))
    exposure = float(np.clip(0.7 * contrast_norm + 0.3 * (1.0 - clip_penalty), 0.0, 1.0))

    orb_keypoints = 0.0
    if use_orb:
        orb_obj = orb if orb is not None else cv2.ORB_create(nfeatures=2000)
        keypoints = orb_obj.detect(gray, mask=mask)
        orb_keypoints = float(len(keypoints) if keypoints is not None else 0.0)

    return {
        "laplacian_var": lap_var,
        "tenengrad": tenengrad,
        "exposure": exposure,
        "orb_keypoints": orb_keypoints,
        "black_clip": black_clip,
        "white_clip": white_clip,
        "contrast": contrast,
        "roi_pixel_count": float(int(np.count_nonzero(mask))),
    }


def compute_raw_metrics_batch(
    frames: List[np.ndarray],
    roi_spec: str | QualityROISpec | None = None,
    *,
    use_orb: bool = True,
    orb: Optional[cv2.ORB] = None,
    gpu_batch_enabled: bool = True,
) -> List[Dict[str, float]]:
    """Compute raw quality metrics for a batch of frames."""
    if not frames:
        return []

    grays = [_to_gray(f) for f in frames]
    spec = parse_roi_spec(roi_spec)
    accelerator = get_accelerator()

    full_roi = bool(spec.mode == "rect" and spec.ratio >= 0.999)
    if gpu_batch_enabled and full_roi:
        lap_vars: List[Optional[float]] = [float(v) for v in accelerator.batch_laplacian_var(grays)]
    else:
        lap_vars = [None] * len(grays)

    cache_masks: Dict[Tuple[int, int], np.ndarray] = {}
    out: List[Dict[str, float]] = []
    for gray, lap_var in zip(grays, lap_vars):
        shape_key = (int(gray.shape[0]), int(gray.shape[1]))
        mask = cache_masks.get(shape_key)
        if mask is None:
            mask = _build_roi_mask(gray, spec)
            cache_masks[shape_key] = mask

        vals = _masked_values(gray.astype(np.float64), mask)
        if lap_var is None:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vals = _masked_values(lap, mask)
            lap_var = float(np.var(lap_vals)) if lap_vals.size > 0 else 0.0
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad_map = (sobel_x * sobel_x) + (sobel_y * sobel_y)
        tenengrad_vals = _masked_values(tenengrad_map, mask)
        tenengrad = float(np.mean(tenengrad_vals)) if tenengrad_vals.size > 0 else 0.0

        black_clip = float(np.mean(vals <= 16.0)) if vals.size > 0 else 1.0
        white_clip = float(np.mean(vals >= 245.0)) if vals.size > 0 else 1.0
        contrast = float(np.std(vals)) if vals.size > 0 else 0.0
        contrast_norm = float(np.clip(contrast / 64.0, 0.0, 1.0))
        clip_penalty = float(np.clip(black_clip + white_clip, 0.0, 1.0))
        exposure = float(np.clip(0.7 * contrast_norm + 0.3 * (1.0 - clip_penalty), 0.0, 1.0))

        orb_keypoints = 0.0
        if use_orb:
            orb_obj = orb if orb is not None else cv2.ORB_create(nfeatures=2000)
            keypoints = orb_obj.detect(gray, mask=mask)
            orb_keypoints = float(len(keypoints) if keypoints is not None else 0.0)

        out.append(
            {
                "laplacian_var": float(lap_var),
                "tenengrad": tenengrad,
                "exposure": exposure,
                "orb_keypoints": orb_keypoints,
                "black_clip": black_clip,
                "white_clip": white_clip,
                "contrast": contrast,
                "roi_pixel_count": float(int(np.count_nonzero(mask))),
            }
        )
    return out


def normalize_batch_p10_p90(
    records: List[Dict[str, float]],
    *,
    p_low: float = 10.0,
    p_high: float = 90.0,
    fields: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Normalize records into [0, 1] with p_low/p_high percentiles."""
    target_fields = tuple(fields or DEFAULT_FIELDS)
    out = [dict(rec) for rec in records]
    stats: Dict[str, Dict[str, float]] = {}
    p_low = float(np.clip(p_low, 0.0, 100.0))
    p_high = float(np.clip(p_high, p_low, 100.0))

    for field in target_fields:
        values = np.asarray(
            [float(rec.get(field, 0.0)) for rec in records if np.isfinite(float(rec.get(field, 0.0)))],
            dtype=np.float64,
        )
        if values.size == 0:
            lo = 0.0
            hi = 1.0
        else:
            lo = float(np.percentile(values, p_low))
            hi = float(np.percentile(values, p_high))
        stats[field] = {"p_low": lo, "p_high": hi}

        span = hi - lo
        norm_key = f"norm_{field}"
        for rec in out:
            val = float(rec.get(field, 0.0))
            if not np.isfinite(val):
                rec[norm_key] = 0.0
                continue
            if span <= 1e-12:
                rec[norm_key] = 0.5
                continue
            rec[norm_key] = float(np.clip((val - lo) / span, 0.0, 1.0))

    return out, stats


def compose_quality(norm_metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Compose normalized metrics into final quality score [0, 1]."""
    w = weights or {}
    w_sharp = float(w.get("quality_weight_sharpness", w.get("sharpness", 0.40)))
    w_tene = float(w.get("quality_weight_tenengrad", w.get("tenengrad", 0.30)))
    w_expo = float(w.get("quality_weight_exposure", w.get("exposure", 0.15)))
    w_kp = float(w.get("quality_weight_keypoints", w.get("keypoints", 0.15)))

    sharp = float(norm_metrics.get("norm_laplacian_var", norm_metrics.get("laplacian_var", 0.0)))
    tene = float(norm_metrics.get("norm_tenengrad", norm_metrics.get("tenengrad", 0.0)))
    expo = float(norm_metrics.get("norm_exposure", norm_metrics.get("exposure", 0.0)))
    kp = float(norm_metrics.get("norm_orb_keypoints", norm_metrics.get("orb_keypoints", 0.0)))

    wsum = w_sharp + w_tene + w_expo + w_kp
    if wsum <= 1e-12:
        w_sharp, w_tene, w_expo, w_kp = 0.40, 0.30, 0.15, 0.15
        wsum = 1.0

    score = (w_sharp * sharp + w_tene * tene + w_expo * expo + w_kp * kp) / wsum
    return float(np.clip(score, 0.0, 1.0))


def apply_abs_guard(laplacian_var: float, abs_min: float) -> bool:
    """Absolute safety guard against very blurry frames."""
    return bool(float(laplacian_var) >= float(abs_min))


__all__ = [
    "QualityROISpec",
    "parse_roi_spec",
    "compute_raw_metrics",
    "compute_raw_metrics_batch",
    "normalize_batch_p10_p90",
    "compose_quality",
    "apply_abs_guard",
]
