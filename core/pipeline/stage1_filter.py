"""Stage 1 filtering module."""

from typing import Any, Callable, Dict, List, Optional


def run_stage1_filter(selector: Any, video_loader: Any, metadata: Any,
                      progress_callback: Optional[Callable[[int, int], None]]) -> List[Dict]:
    return selector._stage1_fast_filter(video_loader, metadata, progress_callback)
