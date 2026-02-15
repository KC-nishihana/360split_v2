"""
対象マスク生成ユーティリティ

YOLO検出 + SAMセグメンテーション結果をOR結合し、
対象=黒(0), 背景=白(255)の2値マスクを生成する。
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from processing.instance_segmentor import InstanceSegmentor
from processing.object_detector import Detection, ObjectDetector
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TARGET_CLASSES = ["人物", "人", "自転車", "バイク", "車両", "動物", "その他"]


class TargetMaskGenerator:
    """フレームから対象マスクを生成し保存するヘルパー"""

    def __init__(
        self,
        yolo_model_path: str = "yolo26x-seg.pt",
        sam_model_path: str = "sam3_t.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ):
        self.detector = ObjectDetector(
            model_path=yolo_model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        self.segmentor = InstanceSegmentor(
            model_path=sam_model_path,
            device=device,
        )

    @staticmethod
    def _detections_to_binary_mask(
        frame_shape,
        detections: List[Detection],
        instance_masks: List[np.ndarray],
    ) -> np.ndarray:
        combined = np.zeros(frame_shape[:2], dtype=np.uint8)
        for det, mask in zip(detections, instance_masks):
            if mask.shape != combined.shape:
                continue
            _ = det
            combined = np.logical_or(combined, mask > 0).astype(np.uint8)
        return np.where(combined == 1, 0, 255).astype(np.uint8)

    def generate_mask(
        self,
        frame: np.ndarray,
        target_classes: Optional[List[str]],
    ) -> np.ndarray:
        """
        対象=黒(0), 背景=白(255)の2値マスクを生成する。
        """
        # クラス未選択時は白マスク
        if not target_classes:
            return np.full(frame.shape[:2], 255, dtype=np.uint8)

        detections = self.detector.detect(frame, classes=target_classes)
        if not detections:
            return np.full(frame.shape[:2], 255, dtype=np.uint8)

        boxes = [det.box for det in detections]
        masks = self.segmentor.segment(frame, boxes)
        if not masks:
            return np.full(frame.shape[:2], 255, dtype=np.uint8)

        return self._detections_to_binary_mask(frame.shape, detections, masks)

    @staticmethod
    def build_mask_path(
        image_path: Path,
        images_root: Path,
        masks_root: Path,
        add_suffix: bool = True,
        suffix: str = "_mask",
        mask_ext: str = "same",
    ) -> Path:
        """
        画像パスから対応するマスク出力パスを構築する。
        """
        rel = image_path.relative_to(images_root)
        stem = f"{image_path.stem}{suffix}" if add_suffix else image_path.stem
        if mask_ext == "same":
            ext = image_path.suffix
        else:
            ext = f".{mask_ext.lstrip('.')}"
        return (masks_root / rel.parent / f"{stem}{ext}")
