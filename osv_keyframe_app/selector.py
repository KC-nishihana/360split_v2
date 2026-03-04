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


def _compute_quality_score(m: FrameMetrics) -> float:
    """Compute composite quality score [0, 1] from metrics."""
    # Normalize components to rough [0, 1] range
    sharpness_norm = float(np.clip(m.laplacian_var / 500.0, 0.0, 1.0))
    exposure_norm = m.exposure_score
    orb_norm = float(np.clip(m.orb_keypoints / 500.0, 0.0, 1.0))
    # Novelty: lower SSIM = more novel content
    novelty_norm = float(np.clip(1.0 - m.ssim_prev, 0.0, 1.0))

    # Weighted combination
    score = 0.35 * sharpness_norm + 0.25 * exposure_norm + 0.20 * orb_norm + 0.20 * novelty_norm
    return float(np.clip(score, 0.0, 1.0))


class TierSelector:
    """Select keyframes for one tier (SfM or 3DGS)."""

    def __init__(self, thresholds: ThresholdConfig, tier_name: str) -> None:
        self._t = thresholds
        self._tier = tier_name

    def select(self, all_metrics: List[FrameMetrics]) -> List[SelectedFrame]:
        """Select frames that pass quality thresholds.

        Steps:
        1. Filter by absolute thresholds (sharpness, exposure, orb)
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
            candidates.sort(key=_compute_quality_score, reverse=True)

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
                score=_compute_quality_score(m),
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
    """
    sfm_selector = TierSelector(config.sfm, "sfm")
    gs_selector = TierSelector(config.gs, "gs")

    sfm_frames = sfm_selector.select(all_metrics)
    gs_frames = gs_selector.select(all_metrics)

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
