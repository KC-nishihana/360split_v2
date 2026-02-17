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

SKY_LABEL = "空"
DEFAULT_TARGET_CLASSES = ["人物", "人", "自転車", "バイク", "車両", "空", "動物", "その他"]


class TargetMaskGenerator:
    """フレームから対象マスクを生成し保存するヘルパー"""

    def __init__(
        self,
        yolo_model_path: str = "yolo26x-seg.pt",
        sam_model_path: str = "sam3.pt",
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

        target_set = set(target_classes)
        include_sky = SKY_LABEL in target_set
        non_sky_targets = [t for t in target_classes if t != SKY_LABEL]

        combined = np.zeros(frame.shape[:2], dtype=np.uint8)

        if non_sky_targets:
            detections = self.detector.detect(frame, classes=non_sky_targets)
            if detections:
                boxes = [det.box for det in detections]
                masks = self.segmentor.segment(frame, boxes)
                if masks:
                    object_binary = self._detections_to_binary_mask(frame.shape, detections, masks)
                    combined = np.logical_or(combined, object_binary == 0).astype(np.uint8)

        if include_sky:
            sky_mask = self._detect_sky_mask(frame)
            combined = np.logical_or(combined, sky_mask > 0).astype(np.uint8)

        if not np.any(combined):
            return np.full(frame.shape[:2], 255, dtype=np.uint8)
        return np.where(combined == 1, 0, 255).astype(np.uint8)

    @staticmethod
    def _detect_sky_mask(frame: np.ndarray) -> np.ndarray:
        """
        空領域の簡易推定マスク（0/1）。
        上半分優先 + HSVしきい値で青空/曇天を拾う。
        """
        import cv2

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # 青空（青系）と曇天（低彩度・高輝度）を候補化
        blue_sky = (h_ch >= 85) & (h_ch <= 140) & (s_ch >= 25) & (v_ch >= 70)
        cloudy_sky = (s_ch <= 50) & (v_ch >= 140)
        sky = np.logical_or(blue_sky, cloudy_sky)

        # 空は主に上側に現れる前提で重みづけ
        top_prior = np.zeros((h, w), dtype=bool)
        top_prior[: int(h * 0.65), :] = True
        sky = np.logical_and(sky, top_prior)

        sky_u8 = sky.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        sky_u8 = cv2.morphologyEx(sky_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        sky_u8 = cv2.morphologyEx(sky_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        return sky_u8

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
