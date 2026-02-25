"""Stage 3 refiner module."""

from typing import Any, Dict, List


def run_stage3_refiner(
    selector: Any,
    metadata: Any,
    stage2_candidates: List[Any],
    stage2_final: List[Any],
    stage2_records: List[Any],
    stage0_metrics: Dict[int, Dict[str, Any]],
    video_loader: Any,
) -> List[Any]:
    return selector._stage3_refine_with_trajectory(
        metadata=metadata,
        stage2_candidates=stage2_candidates,
        stage2_final=stage2_final,
        stage2_records=stage2_records,
        stage0_metrics=stage0_metrics,
        video_loader=video_loader,
    )
