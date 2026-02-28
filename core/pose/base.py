from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import PoseEstimationResult


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, image_dir: str, context: Optional[Dict[str, Any]] = None) -> PoseEstimationResult:
        raise NotImplementedError
