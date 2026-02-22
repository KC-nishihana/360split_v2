"""
擬似軌跡ビュー - 360Split v2 GUI
translation_delta / rotation_delta からデバッグ用の擬似軌跡を描画する。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class TrajectoryWidget(QWidget):
    """擬似軌跡を表示する簡易ビュー。"""

    frameSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._coords = np.zeros((0, 3), dtype=np.float32)
        self._frame_indices: List[int] = []
        self._is_stationary: List[bool] = []
        self._keyframe_set: set[int] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        head = QHBoxLayout()
        label = QLabel("擬似軌跡（VO実測ではありません）")
        label.setStyleSheet("color: #aaa; font-size: 11px;")
        head.addWidget(label)
        head.addStretch()

        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["XY", "XZ"])
        self._axis_combo.currentTextChanged.connect(self._redraw)
        head.addWidget(self._axis_combo)
        layout.addLayout(head)

        self._plot = None
        self._line_item = None
        self._scatter_normal = None
        self._scatter_stationary = None
        self._scatter_keyframe = None

        if HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True)
            self._plot = pg.PlotWidget()
            self._plot.setBackground('#1a1a2e')
            self._plot.showGrid(x=True, y=True, alpha=0.15)
            self._plot.setLabel('left', 'Y/Z')
            self._plot.setLabel('bottom', 'X')
            self._plot.setMouseEnabled(x=True, y=True)

            self._line_item = self._plot.plot([], [], pen=pg.mkPen('#6fa7ff', width=1.5))
            self._scatter_normal = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(90, 150, 255, 180), pen=None)
            self._scatter_stationary = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(140, 140, 140, 190), pen=None)
            self._scatter_keyframe = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 90, 90, 220), pen=None)

            self._scatter_normal.sigClicked.connect(self._on_point_clicked)
            self._scatter_stationary.sigClicked.connect(self._on_point_clicked)
            self._scatter_keyframe.sigClicked.connect(self._on_point_clicked)

            self._plot.addItem(self._line_item)
            self._plot.addItem(self._scatter_normal)
            self._plot.addItem(self._scatter_stationary)
            self._plot.addItem(self._scatter_keyframe)
            layout.addWidget(self._plot)
        else:
            fallback = QLabel("pyqtgraph が未インストールのため軌跡表示は無効です")
            fallback.setStyleSheet("color: #ff8800; background: #1e1e1e; padding: 10px;")
            layout.addWidget(fallback)

    def set_frame_data(self, frame_scores: Sequence[object], keyframe_indices: Sequence[int]):
        self._frame_indices = [int(getattr(s, 'frame_index', 0)) for s in frame_scores]
        if not self._frame_indices:
            self._coords = np.zeros((0, 3), dtype=np.float32)
            self._is_stationary = []
            self._keyframe_set = set()
            self._redraw()
            return

        flow_mag = np.asarray([float(getattr(s, 'flow_mag', 0.0)) for s in frame_scores], dtype=np.float32)
        rotation = np.asarray([float(getattr(s, 'rotation_delta', 0.0)) for s in frame_scores], dtype=np.float32)
        self._is_stationary = [bool(getattr(s, 'is_stationary', False)) for s in frame_scores]
        self._keyframe_set = {int(i) for i in keyframe_indices}

        # 擬似軌跡: flow を移動量、rotation を方位変化として積分
        dist = np.clip(flow_mag, 0.0, np.percentile(flow_mag, 95) if flow_mag.size > 0 else 0.0)
        dist = dist * 0.03
        dtheta = np.deg2rad(np.clip(rotation, 0.0, 30.0))

        x = np.zeros_like(dist)
        y = np.zeros_like(dist)
        z = np.zeros_like(dist)
        theta = 0.0
        for i in range(1, len(dist)):
            theta += float(dtheta[i])
            step = float(dist[i])
            x[i] = x[i - 1] + step * np.cos(theta)
            z[i] = z[i - 1] + step * np.sin(theta)
            y[i] = y[i - 1] + step * np.sin(theta * 0.25) * 0.5

        self._coords = np.column_stack([x, y, z]).astype(np.float32)
        self._redraw()

    def _selected_plane(self) -> tuple[np.ndarray, np.ndarray]:
        if self._coords.size == 0:
            return np.array([]), np.array([])
        if self._axis_combo.currentText() == "XY":
            return self._coords[:, 0], self._coords[:, 1]
        return self._coords[:, 0], self._coords[:, 2]

    def _redraw(self):
        if not HAS_PYQTGRAPH or self._plot is None:
            return
        xs, ys = self._selected_plane()
        if xs.size == 0:
            self._line_item.setData([], [])
            self._scatter_normal.setData([])
            self._scatter_stationary.setData([])
            self._scatter_keyframe.setData([])
            return

        self._line_item.setData(xs, ys)

        normal_pts = []
        stationary_pts = []
        key_pts = []
        for i, frame_idx in enumerate(self._frame_indices):
            point = {'pos': (float(xs[i]), float(ys[i])), 'data': int(frame_idx)}
            if frame_idx in self._keyframe_set:
                key_pts.append(point)
            elif i < len(self._is_stationary) and self._is_stationary[i]:
                stationary_pts.append(point)
            else:
                normal_pts.append(point)

        self._scatter_normal.setData(normal_pts)
        self._scatter_stationary.setData(stationary_pts)
        self._scatter_keyframe.setData(key_pts)
        self._plot.enableAutoRange()

    def _on_point_clicked(self, _item, points):
        if not points:
            return
        frame_idx = int(points[0].data())
        self.frameSelected.emit(frame_idx)
