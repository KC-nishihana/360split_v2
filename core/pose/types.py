from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PoseRecord:
    frame_index: int
    filename: str
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    confidence: float
    observations: int = 0


@dataclass
class PoseEstimationResult:
    poses: List[PoseRecord]
    backend: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    raw_log_paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
