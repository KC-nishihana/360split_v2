"""
対象物検出モジュール

YOLOベースの物体検出をラップし、GUI設定で指定されたクラスのみを抽出する。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from core.accelerator import get_accelerator
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """検出結果"""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


class ObjectDetector:
    """YOLOによる物体検出"""

    # GUI表示ラベル -> COCO系クラス名の対応
    TARGET_CLASS_MAP: Dict[str, Set[str]] = {
        "人物": {"person"},
        "人": {"person"},
        "自転車": {"bicycle"},
        "バイク": {"motorcycle"},
        "車両": {"car", "bus", "truck"},
        "動物": {
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe"
        },
        "その他": set(),
    }

    def __init__(
        self,
        model_path: str = "yolo26n-seg.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)
        self._model = None
        self._class_name_to_id: Dict[str, int] = {}

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
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "ultralytics が見つかりません。`pip install ultralytics` を実行してください。"
            ) from e

        # モデルがローカルに無い場合、ultralytics側の仕組みで自動取得される
        self._model = YOLO(self.model_path)
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            self._class_name_to_id = {str(v): int(k) for k, v in names.items()}
        elif isinstance(names, list):
            self._class_name_to_id = {str(v): i for i, v in enumerate(names)}
        logger.info(f"ObjectDetector初期化: model={self.model_path}, device={self.device}")

    def _expand_target_ids(self, target_classes: Optional[List[str]]) -> Optional[Set[int]]:
        if not target_classes:
            return None

        all_known = set().union(
            *[v for k, v in self.TARGET_CLASS_MAP.items() if k != "その他"]
        )
        target_names: Set[str] = set()

        for label in target_classes:
            if label in self.TARGET_CLASS_MAP:
                target_names.update(self.TARGET_CLASS_MAP[label])
            else:
                # 直接クラス名が指定された場合（例: person）
                target_names.add(label)

        if "その他" in target_classes:
            model_names = set(self._class_name_to_id.keys())
            target_names.update(model_names - all_known)

        target_ids = {
            cid for name, cid in self._class_name_to_id.items()
            if name in target_names
        }
        return target_ids

    def detect(self, frame: np.ndarray, classes: Optional[List[str]] = None) -> List[Detection]:
        """
        フレームから物体を検出する。

        Parameters
        ----------
        frame : np.ndarray
            BGR画像
        classes : List[str], optional
            取得対象ラベル（GUIラベルまたはクラス名）
        """
        self._ensure_model()
        target_ids = self._expand_target_ids(classes)

        results = self._model.predict(
            source=frame,
            conf=float(self.confidence_threshold),
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: List[Detection] = []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()

        for box, class_id, score in zip(xyxy, cls, conf):
            if target_ids is not None and class_id not in target_ids:
                continue
            class_name = str(self._model.names.get(int(class_id), class_id))
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(score),
                )
            )
        return detections
