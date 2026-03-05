"""2-tier keyframe selection (SfM strict / 3DGS relaxed)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from osv_keyframe_app.config import SelectionConfig, ThresholdConfig
from osv_keyframe_app.metrics import FrameMetrics

logger = logging.getLogger(__name__)


@dataclass
class BatchNorms:
    """P90-based normalization denominators computed from a full metrics batch.

    Adapts quality score scaling to the actual distribution of the video being
    processed, so that e.g. a high-quality outdoor clip and a dim indoor clip
    each have their best frames score near 1.0 rather than being unfairly
    penalised by fixed global constants.
    """

    laplacian_p90: float = 500.0   # divisor for laplacian_var normalisation
    tenengrad_p90: float = 2000.0  # divisor for tenengrad normalisation
    orb_p90: float = 500.0         # divisor for orb_keypoints normalisation


def compute_batch_norms(metrics: List[FrameMetrics]) -> BatchNorms:
    """Compute P90-based normalization denominators from all frame metrics.

    Falls back to fixed defaults when a metric has no positive values
    (e.g., all-black frames).
    """
    lap_vals = [m.laplacian_var for m in metrics if m.laplacian_var > 0]
    ten_vals = [m.tenengrad for m in metrics if m.tenengrad > 0]
    orb_vals = [m.orb_keypoints for m in metrics if m.orb_keypoints > 0]

    norms = BatchNorms(
        laplacian_p90=float(np.percentile(lap_vals, 90)) if lap_vals else 500.0,
        tenengrad_p90=float(np.percentile(ten_vals, 90)) if ten_vals else 2000.0,
        orb_p90=float(np.percentile(orb_vals, 90)) if orb_vals else 500.0,
    )
    logger.info(
        f"BatchNorms P90 — laplacian={norms.laplacian_p90:.1f}, "
        f"tenengrad={norms.tenengrad_p90:.1f}, orb={norms.orb_p90:.1f}"
    )
    return norms


@dataclass
class SelectedFrame:
    """A frame selected for output."""

    frame_idx: int
    stream: str
    direction: str
    tier: str  # "sfm" or "gs"
    reason: str
    score: float
    metrics: FrameMetrics


@dataclass
class SelectionResult:
    """Result of 2-tier selection."""

    sfm_frames: List[SelectedFrame] = field(default_factory=list)
    gs_frames: List[SelectedFrame] = field(default_factory=list)

    @property
    def sfm_count(self) -> int:
        return len(self.sfm_frames)

    @property
    def gs_count(self) -> int:
        return len(self.gs_frames)

    def counts_by_direction(self, tier: str) -> Dict[str, int]:
        """Count selected frames per direction for a tier."""
        frames = self.sfm_frames if tier == "sfm" else self.gs_frames
        counts: Dict[str, int] = defaultdict(int)
        for f in frames:
            counts[f.direction] += 1
        return dict(counts)

    def counts_by_stream_direction(self, tier: str) -> Dict[Tuple[str, str], int]:
        """Count selected frames per (stream, direction) pair."""
        frames = self.sfm_frames if tier == "sfm" else self.gs_frames
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for f in frames:
            counts[(f.stream, f.direction)] += 1
        return dict(counts)


def _compute_quality_score(
    m: FrameMetrics,
    norms: Optional[BatchNorms] = None,
) -> float:
    """Compute composite quality score [0, 1] from metrics.

    Weights:
      Laplacian sharpness  25%  (normalized by BatchNorms.laplacian_p90 or 500)
      Tenengrad            15%  (normalized by BatchNorms.tenengrad_p90 or 2000)
      Exposure             25%
      ORB keypoints        20%  (normalized by BatchNorms.orb_p90 or 500)
      Novelty (1-SSIM)     15%

    When ``norms`` is provided the denominators are video-adaptive (P90 of the
    full batch), so the best frames in any given clip score near 1.0 regardless
    of absolute metric magnitudes.  Passing ``None`` uses fixed fallback values
    for backward compatibility.
    """
    lap_d = norms.laplacian_p90 if norms is not None else 500.0
    ten_d = norms.tenengrad_p90 if norms is not None else 2000.0
    orb_d = norms.orb_p90       if norms is not None else 500.0

    sharpness_norm = float(np.clip(m.laplacian_var / max(lap_d, 1e-6), 0.0, 1.0))
    tenengrad_norm = float(np.clip(m.tenengrad     / max(ten_d, 1e-6), 0.0, 1.0))
    exposure_norm  = m.exposure_score
    orb_norm       = float(np.clip(m.orb_keypoints / max(orb_d, 1e-6), 0.0, 1.0))
    # Novelty: lower SSIM = more novel content
    novelty_norm   = float(np.clip(1.0 - m.ssim_prev, 0.0, 1.0))

    score = (
        0.25 * sharpness_norm
        + 0.15 * tenengrad_norm
        + 0.25 * exposure_norm
        + 0.20 * orb_norm
        + 0.15 * novelty_norm
    )
    return float(np.clip(score, 0.0, 1.0))


class TierSelector:
    """Select keyframes for one tier (SfM or 3DGS)."""

    def __init__(self, thresholds: ThresholdConfig, tier_name: str) -> None:
        self._t = thresholds
        self._tier = tier_name

    def select(
        self,
        all_metrics: List[FrameMetrics],
        norms: Optional[BatchNorms] = None,
    ) -> List[SelectedFrame]:
        """Select frames that pass quality thresholds.

        Parameters
        ----------
        norms : optional batch-adaptive normalization denominators produced by
            ``compute_batch_norms()``.  When provided, quality scores used for
            rescue ranking and max_total sorting adapt to the video's own metric
            distribution (P90-based) instead of fixed global constants.

        Steps:
        1. Filter by absolute thresholds (sharpness, tenengrad, exposure, orb)
        2. Filter by SSIM novelty (reject frames too similar to previous)
        3. Ensure per-direction minimum counts (rescue best frames if needed)
        4. Apply max_total limit if set
        """
        # Step 1: Absolute threshold filter
        passed: List[Tuple[FrameMetrics, str]] = []
        rejected_by_threshold: List[FrameMetrics] = []

        for m in all_metrics:
            reasons = []
            if m.laplacian_var < self._t.sharpness_min:
                reasons.append(f"sharpness({m.laplacian_var:.1f}<{self._t.sharpness_min})")
            if self._t.tenengrad_min > 0 and m.tenengrad < self._t.tenengrad_min:
                reasons.append(f"tenengrad({m.tenengrad:.1f}<{self._t.tenengrad_min})")
            if m.exposure_score < self._t.exposure_min:
                reasons.append(f"exposure({m.exposure_score:.3f}<{self._t.exposure_min})")
            if m.orb_keypoints < self._t.orb_min:
                reasons.append(f"orb({m.orb_keypoints}<{self._t.orb_min})")

            if reasons:
                rejected_by_threshold.append(m)
            else:
                passed.append((m, "threshold_pass"))

        # Step 2: SSIM novelty filter (reject too-similar adjacent frames)
        novel: List[Tuple[FrameMetrics, str]] = []
        for m, reason in passed:
            if m.ssim_prev > self._t.ssim_max and m.ssim_prev > 0:
                rejected_by_threshold.append(m)
            else:
                novel.append((m, "quality_and_novelty_pass"))

        # Step 3: Ensure per-direction minimum counts
        dir_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for m, _ in novel:
            dir_counts[(m.stream, m.direction)] += 1

        # Find all unique (stream, direction) pairs
        all_pairs: set[Tuple[str, str]] = set()
        for m in all_metrics:
            all_pairs.add((m.stream, m.direction))

        # Rescue frames for under-represented directions
        rescued: List[Tuple[FrameMetrics, str]] = []
        for pair in sorted(all_pairs):
            current = dir_counts.get(pair, 0)
            deficit = self._t.per_direction_min - current
            if deficit <= 0:
                continue

            # Get all rejected frames for this direction, sorted by quality
            candidates = [
                m for m in rejected_by_threshold
                if (m.stream, m.direction) == pair
            ]
            candidates.sort(key=lambda m: _compute_quality_score(m, norms), reverse=True)

            for m in candidates[:deficit]:
                rescued.append((m, f"rescued_for_{pair[0]}_{pair[1]}"))
                dir_counts[pair] = dir_counts.get(pair, 0) + 1

        # Combine and create SelectedFrame objects
        all_selected = novel + rescued
        results: List[SelectedFrame] = []
        for m, reason in all_selected:
            results.append(SelectedFrame(
                frame_idx=m.frame_idx,
                stream=m.stream,
                direction=m.direction,
                tier=self._tier,
                reason=reason,
                score=_compute_quality_score(m, norms),
                metrics=m,
            ))

        # Step 4: Apply max_total limit
        if self._t.max_total is not None and len(results) > self._t.max_total:
            results.sort(key=lambda sf: sf.score, reverse=True)
            results = results[:self._t.max_total]

        # Sort by frame_idx for consistent output
        results.sort(key=lambda sf: (sf.frame_idx, sf.stream, sf.direction))

        logger.info(
            f"{self._tier} selection: {len(results)} frames "
            f"(passed={len(novel)}, rescued={len(rescued)})"
        )
        return results


def select_two_tier(
    all_metrics: List[FrameMetrics],
    config: SelectionConfig,
) -> SelectionResult:
    """Run both SfM (strict) and 3DGS (relaxed) selection.

    3DGS result is a superset of SfM result (SfM frames always included in 3DGS).

    Computes P90-based batch normalization once from all metrics and shares it
    across both tier selectors so quality scores adapt to the video's own metric
    distribution.
    """
    norms = compute_batch_norms(all_metrics)

    sfm_selector = TierSelector(config.sfm, "sfm")
    gs_selector = TierSelector(config.gs, "gs")

    sfm_frames = sfm_selector.select(all_metrics, norms=norms)
    gs_frames = gs_selector.select(all_metrics, norms=norms)

    # Merge: ensure all SfM frames are included in 3DGS
    gs_keys = {(f.frame_idx, f.stream, f.direction) for f in gs_frames}
    for sf in sfm_frames:
        key = (sf.frame_idx, sf.stream, sf.direction)
        if key not in gs_keys:
            gs_frames.append(SelectedFrame(
                frame_idx=sf.frame_idx,
                stream=sf.stream,
                direction=sf.direction,
                tier="gs",
                reason="included_from_sfm",
                score=sf.score,
                metrics=sf.metrics,
            ))
            gs_keys.add(key)

    gs_frames.sort(key=lambda f: (f.frame_idx, f.stream, f.direction))

    result = SelectionResult(sfm_frames=sfm_frames, gs_frames=gs_frames)

    logger.info(
        f"Selection complete: SfM={result.sfm_count}, 3DGS={result.gs_count}"
    )
    return result
