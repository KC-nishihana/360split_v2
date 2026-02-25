"""Scoring helper module."""

from typing import Any, Dict, List


def compute_combined_score(selector: Any, quality_scores: Dict[str, float],
                           geometric_scores: Dict[str, float],
                           adaptive_scores: Dict[str, float]) -> float:
    return selector._compute_combined_score(quality_scores, geometric_scores, adaptive_scores)


def apply_nms(selector: Any, candidates: List[Any], fps: float = 30.0) -> List[Any]:
    return selector._apply_nms(candidates, fps=fps)
