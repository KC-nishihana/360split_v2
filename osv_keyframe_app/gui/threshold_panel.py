"""Threshold sliders for SfM/3DGS selection with live count updates."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, Signal

from osv_keyframe_app.config import SelectionConfig, ThresholdConfig
from osv_keyframe_app.metrics import FrameMetrics
from osv_keyframe_app.selector import SelectionResult, select_two_tier

logger = logging.getLogger(__name__)


class _ThresholdSlider(QWidget):
    """A slider with spinbox and label for one threshold parameter."""

    valueChanged = Signal()

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        decimals: int = 2,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(label))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val * self._scale), int(max_val * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, stretch=1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(min_val, max_val)
        self._spin.setDecimals(decimals)
        self._spin.setValue(default)
        self._spin.setSingleStep(1.0 / self._scale)
        self._spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self._spin)

    def value(self) -> float:
        return self._spin.value()

    def _on_slider(self, val: int) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(val / self._scale)
        self._spin.blockSignals(False)
        self.valueChanged.emit()

    def _on_spin(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(val * self._scale))
        self._slider.blockSignals(False)
        self.valueChanged.emit()


class _TierControls(QGroupBox):
    """Controls for one tier (SfM or 3DGS)."""

    thresholdsChanged = Signal()

    def __init__(self, title: str, defaults: ThresholdConfig, parent=None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self.sharpness = _ThresholdSlider("Sharpness min:", 0, 500, defaults.sharpness_min, 1)
        self.exposure = _ThresholdSlider("Exposure min:", 0.0, 1.0, defaults.exposure_min, 2)
        self.orb = _ThresholdSlider("ORB min:", 0, 1000, defaults.orb_min, 0)
        self.ssim = _ThresholdSlider("SSIM max:", 0.5, 1.0, defaults.ssim_max, 3)
        self.per_dir_min = _ThresholdSlider("Per-dir min:", 0, 100, defaults.per_direction_min, 0)

        for slider in (self.sharpness, self.exposure, self.orb, self.ssim, self.per_dir_min):
            slider.valueChanged.connect(self.thresholdsChanged.emit)
            layout.addWidget(slider)

        self._count_label = QLabel("Selected: 0 frames")
        self._count_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._count_label)

    def get_thresholds(self) -> ThresholdConfig:
        return ThresholdConfig(
            sharpness_min=self.sharpness.value(),
            exposure_min=self.exposure.value(),
            orb_min=int(self.orb.value()),
            ssim_max=self.ssim.value(),
            per_direction_min=int(self.per_dir_min.value()),
        )

    def set_count(self, count: int) -> None:
        self._count_label.setText(f"Selected: {count} frames")


class ThresholdPanel(QWidget):
    """Panel with SfM/3DGS threshold sliders and live count updates."""

    selectionChanged = Signal(object)  # emits SelectionResult

    def __init__(self, config: SelectionConfig, parent=None) -> None:
        super().__init__(parent)
        self._metrics: List[FrameMetrics] = []
        self._setup_ui(config)

    def _setup_ui(self, config: SelectionConfig) -> None:
        layout = QVBoxLayout(self)

        # Two-column layout for SfM and 3DGS
        columns = QHBoxLayout()

        self._sfm_controls = _TierControls("SfM (Strict)", config.sfm)
        self._gs_controls = _TierControls("3DGS (Relaxed)", config.gs)

        self._sfm_controls.thresholdsChanged.connect(self._on_thresholds_changed)
        self._gs_controls.thresholdsChanged.connect(self._on_thresholds_changed)

        columns.addWidget(self._sfm_controls)
        columns.addWidget(self._gs_controls)
        layout.addLayout(columns)

        # Direction breakdown table
        self._breakdown_table = QTableWidget()
        self._breakdown_table.setColumnCount(3)
        self._breakdown_table.setHorizontalHeaderLabels(["Stream/Direction", "SfM", "3DGS"])
        self._breakdown_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._breakdown_table.setMaximumHeight(200)
        layout.addWidget(QLabel("Direction Breakdown:"))
        layout.addWidget(self._breakdown_table)

    def set_metrics(self, metrics: List[FrameMetrics]) -> None:
        """Set metrics data for live selection updates."""
        self._metrics = metrics
        self._on_thresholds_changed()

    def _on_thresholds_changed(self) -> None:
        """Recompute selection with current thresholds and update counts."""
        if not self._metrics:
            return

        config = SelectionConfig(
            sfm=self._sfm_controls.get_thresholds(),
            gs=self._gs_controls.get_thresholds(),
        )
        result = select_two_tier(self._metrics, config)

        self._sfm_controls.set_count(result.sfm_count)
        self._gs_controls.set_count(result.gs_count)

        self._update_breakdown(result)
        self.selectionChanged.emit(result)

    def _update_breakdown(self, result: SelectionResult) -> None:
        """Update the direction breakdown table."""
        sfm_counts = result.counts_by_stream_direction("sfm")
        gs_counts = result.counts_by_stream_direction("gs")

        all_keys = sorted(set(sfm_counts.keys()) | set(gs_counts.keys()))
        self._breakdown_table.setRowCount(len(all_keys))

        for i, key in enumerate(all_keys):
            stream, direction = key
            label = f"{stream}/{direction}"
            self._breakdown_table.setItem(i, 0, QTableWidgetItem(label))
            self._breakdown_table.setItem(i, 1, QTableWidgetItem(str(sfm_counts.get(key, 0))))
            self._breakdown_table.setItem(i, 2, QTableWidgetItem(str(gs_counts.get(key, 0))))
