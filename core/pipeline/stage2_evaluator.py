"""Stage 2 evaluator module."""

from typing import Any, Callable, Dict, List, Optional, Tuple


def run_stage2_evaluator(
    selector: Any,
    video_loader: Any,
    metadata: Any,
    stage1_candidates: List[Dict],
    progress_callback: Optional[Callable[[int, int], None]],
    frame_log_callback: Optional[Callable[[Dict[str, Any]], None]],
    stage0_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[List[Any], List[Any]]:
    return selector._stage2_precise_evaluation(
        video_loader,
        metadata,
        stage1_candidates,
        progress_callback,
        frame_log_callback,
        stage0_metrics,
    )
