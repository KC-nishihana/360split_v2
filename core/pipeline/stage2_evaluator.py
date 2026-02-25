"""Stage 2 evaluator module."""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
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
    if bool(selector.config.get("ENABLE_STAGE2_PIPELINE_PARALLEL", False)):
        # Deterministic producer-consumer skeleton:
        # Producer prepares immutable candidate stream in index order.
        # Consumer keeps original evaluation order/logic for identical results.
        queue: Queue = Queue(maxsize=32)

        def _producer() -> None:
            for item in stage1_candidates:
                queue.put(item)
            queue.put(None)

        ordered_candidates: List[Dict] = []
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_producer)
            while True:
                item = queue.get()
                if item is None:
                    break
                ordered_candidates.append(item)
        stage1_candidates = ordered_candidates

    return selector._stage2_precise_evaluation(
        video_loader,
        metadata,
        stage1_candidates,
        progress_callback,
        frame_log_callback,
        stage0_metrics,
    )
