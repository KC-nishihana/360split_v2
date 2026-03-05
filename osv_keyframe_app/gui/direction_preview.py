"""Direction preview grid showing projected pinhole images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QSlider, QVBoxLayout, QHBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 240


def _cv2_to_qpixmap(img: np.ndarray, max_size: int = THUMBNAIL_SIZE) -> QPixmap:
    """Convert BGR OpenCV image to QPixmap, resized to thumbnail."""
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class DirectionPreview(QWidget):
    """2x5 grid of thumbnail images for front/back x 5 directions."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._projected_dir: Optional[Path] = None
        self._frame_indices: List[int] = []
        self._streams = ["front", "back"]
        self._directions = ["front", "left", "right", "up", "down"]
        self._labels: Dict[str, QLabel] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        slider_layout.addWidget(self._frame_slider, stretch=1)
        self._frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self._frame_label)
        layout.addLayout(slider_layout)

        # Image grid: rows=streams, cols=directions
        grid = QGridLayout()

        # Column headers
        for col, direction in enumerate(self._directions):
            header = QLabel(direction)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("font-weight: bold;")
            grid.addWidget(header, 0, col + 1)

        # Row headers + image labels
        for row, stream in enumerate(self._streams):
            header = QLabel(stream)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("font-weight: bold;")
            grid.addWidget(header, row + 1, 0)

            for col, direction in enumerate(self._directions):
                label = QLabel()
                label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet("border: 1px solid #ccc; background: #222;")
                key = f"{stream}_{direction}"
                self._labels[key] = label
                grid.addWidget(label, row + 1, col + 1)

        layout.addLayout(grid)

    def set_projected_dir(self, projected_dir: Path, frame_indices: List[int]) -> None:
        """Set the directory containing projected images and available frame indices."""
        self._projected_dir = projected_dir
        self._frame_indices = sorted(set(frame_indices))

        self._frame_slider.setMaximum(max(0, len(self._frame_indices) - 1))
        self._frame_slider.setValue(0)
        self._on_frame_changed(0)

    def _on_frame_changed(self, slider_value: int) -> None:
        """Load and display thumbnails for the selected frame."""
        if not self._frame_indices or not self._projected_dir:
            self._frame_label.setText("0 / 0")
            return

        idx = max(0, min(slider_value, len(self._frame_indices) - 1))
        frame_idx = self._frame_indices[idx]
        self._frame_label.setText(f"{frame_idx} ({idx + 1}/{len(self._frame_indices)})")

        for stream in self._streams:
            for direction in self._directions:
                key = f"{stream}_{direction}"
                filename = f"{stream}_{direction}_{frame_idx:08d}.jpg"
                filepath = self._projected_dir / filename

                label = self._labels[key]
                if filepath.exists():
                    img = cv2.imread(str(filepath))
                    if img is not None:
                        pixmap = _cv2_to_qpixmap(img)
                        label.setPixmap(pixmap)
                        label.setStyleSheet("border: 2px solid #4CAF50; background: #222;")
                        continue

                label.setText("N/A")
                label.setStyleSheet("border: 1px solid #ccc; background: #222; color: #888;")

    def set_selection_highlight(
        self, selected_keys: set[tuple[int, str, str]]
    ) -> None:
        """Highlight selected frames with a green border, others with gray."""
        if not self._frame_indices:
            return

        idx = self._frame_slider.value()
        frame_idx = self._frame_indices[idx] if idx < len(self._frame_indices) else -1

        for stream in self._streams:
            for direction in self._directions:
                key = f"{stream}_{direction}"
                label = self._labels[key]
                if (frame_idx, stream, direction) in selected_keys:
                    label.setStyleSheet("border: 3px solid #4CAF50; background: #222;")
                else:
                    label.setStyleSheet("border: 1px solid #666; background: #222;")
