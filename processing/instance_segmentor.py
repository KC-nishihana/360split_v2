"""
インスタンスセグメンテーションモジュール

SAMベースの推論を試み、利用不可の場合は矩形マスクでフォールバックする。
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from core.accelerator import get_accelerator
from utils.logger import get_logger

logger = get_logger(__name__)


class InstanceSegmentor:
    """SAMを利用したインスタンスマスク生成"""

    def __init__(self, model_path: str = "sam3_t.pt", device: str = "auto"):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self._model = None
        self._sam_available = True

    def _resolve_device(self, device: str) -> str:
        if device and device != "auto":
            return device
        accel = get_accelerator()
        if accel.has_torch and accel.has_cuda:
            return "0"
        if accel.has_torch and accel.has_mps:
            return "mps"
        return "cpu"

    def _ensure_model(self):
        if self._model is not None or not self._sam_available:
            return
        try:
            from ultralytics import SAM
            self._model = SAM(self.model_path)
            logger.info(f"InstanceSegmentor初期化: model={self.model_path}, device={self.device}")
        except Exception as e:
            self._sam_available = False
            logger.warning(f"SAMの初期化に失敗したため矩形マスクへフォールバックします: {e}")

    @staticmethod
    def _box_mask(shape: Tuple[int, int], box: Sequence[int]) -> np.ndarray:
        h, w = shape
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

    def segment(self, frame: np.ndarray, boxes: List[Sequence[int]]) -> List[np.ndarray]:
        """
        各バウンディングボックスに対する2値マスクを返す（0/1）。
        """
        if not boxes:
            return []

        self._ensure_model()
        h, w = frame.shape[:2]
        result_masks: List[np.ndarray] = []

        if self._model is None:
            return [self._box_mask((h, w), box) for box in boxes]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for box in boxes:
            try:
                preds = self._model.predict(
                    source=rgb,
                    bboxes=[list(map(float, box))],
                    device=self.device,
                    verbose=False,
                )
                if preds and preds[0].masks is not None and len(preds[0].masks.data) > 0:
                    mask = preds[0].masks.data[0].detach().cpu().numpy()
                    result_masks.append((mask > 0.5).astype(np.uint8))
                else:
                    result_masks.append(self._box_mask((h, w), box))
            except Exception:
                result_masks.append(self._box_mask((h, w), box))

        return result_masks
