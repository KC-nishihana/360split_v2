"""Metrics visualization tab with pyqtgraph time-series plots."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSplitter,
)
from PySide6.QtCore import Qt

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    pg = None

import numpy as np

from osv_keyframe_app.metrics import FrameMetrics

logger = logging.getLogger(__name__)

# Color palette for directions
DIRECTION_COLORS = {
    "front": (66, 133, 244),   # blue
    "left": (234, 67, 53),     # red
    "right": (52, 168, 83),    # green
    "back": (251, 188, 4),     # yellow
}

METRIC_LABELS = {
    "laplacian_var": "Sharpness (Laplacian Var)",
    "exposure_score": "Exposure Score",
    "orb_keypoints": "ORB Keypoints",
    "ssim_prev": "SSIM (Adjacent Frame)",
}


class MetricsTab(QWidget):
    """Tab showing time-series metric plots with filtering."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._metrics: List[FrameMetrics] = []
        self._plots: Dict[str, object] = {}
        self._curves: Dict[str, Dict[str, object]] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Filter bar
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Stream:"))
        self._stream_combo = QComboBox()
        self._stream_combo.addItems(["All", "front", "back"])
        self._stream_combo.currentTextChanged.connect(self._update_plots)
        filter_layout.addWidget(self._stream_combo)

        filter_layout.addWidget(QLabel("Direction:"))
        self._direction_combo = QComboBox()
        self._direction_combo.addItems(["All", "front", "left", "right", "back"])
        self._direction_combo.currentTextChanged.connect(self._update_plots)
        filter_layout.addWidget(self._direction_combo)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        if not HAS_PYQTGRAPH:
            layout.addWidget(QLabel("pyqtgraph is not installed. Install with: pip install pyqtgraph"))
            return

        # 4 plot widgets in a 2x2 grid via splitters
        outer_splitter = QSplitter(Qt.Orientation.Vertical)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        for metric_key, label in METRIC_LABELS.items():
            plot_widget = pg.PlotWidget(title=label)
            plot_widget.setLabel("bottom", "Frame Index")
            plot_widget.setLabel("left", label.split("(")[0].strip())
            plot_widget.addLegend(offset=(10, 10))
            plot_widget.showGrid(x=True, y=True, alpha=0.3)

            self._plots[metric_key] = plot_widget
            self._curves[metric_key] = {}

            if metric_key in ("laplacian_var", "exposure_score"):
                top_splitter.addWidget(plot_widget)
            else:
                bottom_splitter.addWidget(plot_widget)

        outer_splitter.addWidget(top_splitter)
        outer_splitter.addWidget(bottom_splitter)
        layout.addWidget(outer_splitter)

    def set_metrics(self, metrics: List[FrameMetrics]) -> None:
        """Load metrics data and update plots."""
        self._metrics = metrics
        self._update_plots()

    def _update_plots(self) -> None:
        """Redraw all plots based on current filter settings."""
        if not HAS_PYQTGRAPH or not self._metrics:
            return

        stream_filter = self._stream_combo.currentText()
        direction_filter = self._direction_combo.currentText()

        # Filter metrics
        filtered = self._metrics
        if stream_filter != "All":
            filtered = [m for m in filtered if m.stream == stream_filter]
        if direction_filter != "All":
            filtered = [m for m in filtered if m.direction == direction_filter]

        # Group by direction
        by_direction: Dict[str, List[FrameMetrics]] = defaultdict(list)
        for m in filtered:
            by_direction[m.direction].append(m)

        # Update each plot
        for metric_key, plot_widget in self._plots.items():
            # Clear old curves
            plot_widget.clear()
            self._curves[metric_key] = {}

            for direction, direction_metrics in sorted(by_direction.items()):
                color = DIRECTION_COLORS.get(direction, (128, 128, 128))
                x = np.array([m.frame_idx for m in direction_metrics])
                y = np.array([getattr(m, metric_key) for m in direction_metrics])

                # Sort by frame index
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]

                pen = pg.mkPen(color=color, width=1.5)
                curve = plot_widget.plot(
                    x, y, pen=pen, name=direction,
                    symbol="o", symbolSize=3, symbolBrush=color,
                )
                self._curves[metric_key][direction] = curve

    def clear(self) -> None:
        """Clear all plots."""
        self._metrics = []
        if HAS_PYQTGRAPH:
            for plot_widget in self._plots.values():
                plot_widget.clear()
