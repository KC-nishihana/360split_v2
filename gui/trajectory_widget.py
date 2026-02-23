"""
軌跡ビュー - 360Split v2 GUI
VO相対軌跡（t_xyz）があればそれを優先し、無い場合は擬似軌跡を描画する。
2D平面表示と3D表示を切り替え可能。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except Exception:
    HAS_PYQTGRAPH = False
    pg = None  # type: ignore

try:
    import pyqtgraph.opengl as gl
    HAS_GL = HAS_PYQTGRAPH
except Exception:
    HAS_GL = False
    gl = None  # type: ignore


class TrajectoryWidget(QWidget):
    """軌跡を表示するビュー。2D/3D切替に対応。"""

    frameSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._coords = np.zeros((0, 3), dtype=np.float32)
        self._frame_indices: List[int] = []
        self._is_stationary: List[bool] = []
        self._keyframe_set: set[int] = set()
        self._runtime_points: List[np.ndarray] = []
        self._runtime_origin: Optional[np.ndarray] = None
        self._runtime_update_stride = 5

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        head = QHBoxLayout()
        self._title_label = QLabel("擬似軌跡（VO実測ではありません）")
        self._title_label.setStyleSheet("color: #aaa; font-size: 11px;")
        head.addWidget(self._title_label)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #8fa3b6; font-size: 11px;")
        head.addWidget(self._status_label)
        head.addStretch()

        self._view_mode_combo = QComboBox()
        self._view_mode_combo.addItems(["3D", "2D"])
        self._view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        head.addWidget(self._view_mode_combo)

        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["XY", "XZ"])
        self._axis_combo.currentTextChanged.connect(self._redraw)
        head.addWidget(self._axis_combo)
        layout.addLayout(head)

        self._plot = None
        self._fallback_label_2d: Optional[QLabel] = None
        self._view3d = None
        self._line3d = None
        self._fallback_label_3d: Optional[QLabel] = None
        self._line_item = None
        self._scatter_normal = None
        self._scatter_stationary = None
        self._scatter_keyframe = None
        self._world_colors = {"world": (0.2, 0.45, 1.0, 1.0)}

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
            self._fallback_label_2d = QLabel("pyqtgraph が未インストールのため2D軌跡表示は無効です")
            self._fallback_label_2d.setStyleSheet("color: #ff8800; background: #1e1e1e; padding: 10px;")
            layout.addWidget(self._fallback_label_2d)

        if HAS_GL:
            self._view3d = gl.GLViewWidget()
            self._view3d.setCameraPosition(distance=8)
            self._view3d.opts["center"] = pg.Vector(0, 0, 0)

            grid = gl.GLGridItem()
            grid.scale(1, 1, 1)
            self._view3d.addItem(grid)

            axis = gl.GLAxisItem()
            axis.setSize(1.5, 1.5, 1.5)
            self._view3d.addItem(axis)
            layout.addWidget(self._view3d)
        else:
            self._fallback_label_3d = QLabel("pyqtgraph.opengl が未インストールのため3D軌跡表示は無効です")
            self._fallback_label_3d.setStyleSheet("color: #ff8800; background: #1e1e1e; padding: 10px;")
            layout.addWidget(self._fallback_label_3d)

        default_mode = "3D" if HAS_GL else "2D"
        self.set_view_mode(default_mode)

    @staticmethod
    def _format_reason(reason: str) -> str:
        mapping = {
            "enabled": "有効",
            "calibration_unavailable": "calibration unavailable",
            "projection_mode_unsupported": "projection mode unsupported",
            "vo_disabled_by_config": "設定で無効",
            "frame_subsample_skip": "frame subsample skip",
            "estimate_failed_or_low_inlier": "estimate failed / low inlier",
            "frame_unavailable": "frame unavailable",
            "init": "init",
            "not_evaluated": "not evaluated",
            "unknown": "unknown",
        }
        return mapping.get(str(reason), str(reason))

    def set_view_mode(self, mode: str):
        mode_up = "3D" if str(mode).strip().upper() == "3D" else "2D"
        if self._view_mode_combo.currentText() != mode_up:
            self._view_mode_combo.blockSignals(True)
            self._view_mode_combo.setCurrentText(mode_up)
            self._view_mode_combo.blockSignals(False)
        self._apply_view_mode(mode_up)

    def _on_view_mode_changed(self, mode: str):
        self._apply_view_mode(mode)

    def _apply_view_mode(self, mode: str):
        mode_up = "3D" if str(mode).strip().upper() == "3D" else "2D"
        show_2d = mode_up == "2D"
        self._axis_combo.setVisible(show_2d)

        if self._plot is not None:
            self._plot.setVisible(show_2d)
        if self._fallback_label_2d is not None:
            self._fallback_label_2d.setVisible(show_2d)

        show_3d = mode_up == "3D"
        if self._view3d is not None:
            self._view3d.setVisible(show_3d)
        if self._fallback_label_3d is not None:
            self._fallback_label_3d.setVisible(show_3d)
        self._redraw()

    def reset_runtime_trajectory(self):
        self._runtime_points = []
        self._runtime_origin = None
        if self._line3d is not None and self._view3d is not None:
            self._view3d.removeItem(self._line3d)
            self._line3d = None
        self._redraw()

    def append_runtime_pose(self, payload: dict, force: bool = False):
        t_xyz = payload.get("t_xyz")
        if not isinstance(t_xyz, (list, tuple)) or len(t_xyz) != 3:
            return
        arr = np.asarray(t_xyz, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(arr)):
            return
        if self._runtime_origin is None:
            self._runtime_origin = arr.copy()
        rel = arr - self._runtime_origin
        self._runtime_points.append(rel.astype(np.float32))
        if len(self._runtime_points) < 2:
            return
        if not force and (len(self._runtime_points) % max(1, int(self._runtime_update_stride)) != 0):
            return
        self._update_3d_line(np.asarray(self._runtime_points, dtype=np.float32))

    def flush_runtime_trajectory(self):
        if len(self._runtime_points) < 2:
            return
        self._update_3d_line(np.asarray(self._runtime_points, dtype=np.float32))

    def _update_3d_line(self, points: np.ndarray):
        if points.size == 0 or not HAS_GL or self._view3d is None:
            return
        color = self._world_colors.get("world", (0.8, 0.8, 0.8, 1.0))
        if self._line3d is None:
            self._line3d = gl.GLLinePlotItem(
                pos=points,
                color=color,
                width=2,
                antialias=True,
                mode="line_strip",
            )
            self._view3d.addItem(self._line3d)
        else:
            self._line3d.setData(pos=points, color=color, width=2)

    @staticmethod
    def _derive_vo_summary(frame_scores: Sequence[object]) -> Dict[str, object]:
        if not frame_scores:
            return {
                "runtime_enabled": False,
                "runtime_reason": "not_evaluated",
                "attempted": 0,
                "valid": 0,
                "pose_valid": 0,
                "reason_counts": {},
            }
        attempted = 0
        valid = 0
        pose_valid = 0
        reasons = Counter()
        for s in frame_scores:
            attempted += 1 if bool(getattr(s, "vo_attempted", False)) else 0
            valid += 1 if bool(getattr(s, "vo_valid", False)) else 0
            pose_valid += 1 if bool(getattr(s, "vo_pose_valid", False)) else 0
            reasons[str(getattr(s, "vo_status_reason", "unknown"))] += 1
        top_reason = reasons.most_common(1)[0][0] if reasons else "unknown"
        runtime_enabled = any(r not in {"calibration_unavailable", "projection_mode_unsupported", "vo_disabled_by_config"} for r in reasons.keys())
        return {
            "runtime_enabled": bool(runtime_enabled),
            "runtime_reason": top_reason,
            "attempted": int(attempted),
            "valid": int(valid),
            "pose_valid": int(pose_valid),
            "reason_counts": dict(reasons),
        }

    def set_frame_data(
        self,
        frame_scores: Sequence[object],
        keyframe_indices: Sequence[int],
        vo_summary: Optional[Dict[str, object]] = None,
    ):
        self._frame_indices = [int(getattr(s, 'frame_index', 0)) for s in frame_scores]
        if not self._frame_indices:
            self._coords = np.zeros((0, 3), dtype=np.float32)
            self._is_stationary = []
            self._keyframe_set = set()
            self._title_label.setText("擬似軌跡（VO未使用）")
            self._status_label.setText("")
            self._redraw()
            return

        self._is_stationary = [bool(getattr(s, 'is_stationary', False)) for s in frame_scores]
        self._keyframe_set = {int(i) for i in keyframe_indices}

        summary = vo_summary or self._derive_vo_summary(frame_scores)
        coords = np.full((len(frame_scores), 3), np.nan, dtype=np.float32)
        valid_pose_count = 0
        for i, s in enumerate(frame_scores):
            if not bool(getattr(s, "vo_pose_valid", False)):
                continue
            t_xyz = getattr(s, 't_xyz', None)
            if isinstance(t_xyz, (list, tuple)) and len(t_xyz) == 3:
                arr = np.asarray(t_xyz, dtype=np.float32).reshape(3)
                if np.all(np.isfinite(arr)):
                    coords[i] = arr
                    valid_pose_count += 1

        if valid_pose_count >= 2:
            self._title_label.setText("相対軌跡（VO）")
            attempts = int(summary.get("attempted", 0))
            valid = int(summary.get("valid", 0))
            ratio = (valid / attempts) if attempts > 0 else 0.0
            self._status_label.setText(f"VO有効率: {ratio:.1%} ({valid}/{attempts})")
            if np.any(np.isfinite(coords[0])):
                first = coords[0].copy()
            else:
                first = np.zeros(3, dtype=np.float32)
            last = first.copy()
            for i in range(coords.shape[0]):
                if np.all(np.isfinite(coords[i])):
                    last = coords[i]
                else:
                    coords[i] = last
            self._coords = coords - first
            self._update_3d_line(self._coords)
        else:
            runtime_enabled = bool(summary.get("runtime_enabled", False))
            runtime_reason = str(summary.get("runtime_reason", "not_evaluated"))
            if not runtime_enabled and runtime_reason in {"calibration_unavailable", "projection_mode_unsupported", "vo_disabled_by_config"}:
                self._title_label.setText(f"VO無効（理由: {self._format_reason(runtime_reason)}）")
                self._status_label.setText("擬似軌跡を表示中")
            else:
                self._title_label.setText("擬似軌跡（VO未使用）")
                self._status_label.setText(f"状態: {self._format_reason(runtime_reason)}")
            flow_mag = np.asarray([float(getattr(s, 'flow_mag', 0.0)) for s in frame_scores], dtype=np.float32)
            rotation = np.asarray([float(getattr(s, 'rotation_delta', 0.0)) for s in frame_scores], dtype=np.float32)
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
        if HAS_PYQTGRAPH and self._plot is not None:
            xs, ys = self._selected_plane()
            if xs.size == 0:
                self._line_item.setData([], [])
                self._scatter_normal.setData([])
                self._scatter_stationary.setData([])
                self._scatter_keyframe.setData([])
            else:
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

        if len(self._runtime_points) >= 2:
            self._update_3d_line(np.asarray(self._runtime_points, dtype=np.float32))

    def _on_point_clicked(self, _item, points):
        if not points:
            return
        frame_idx = int(points[0].data())
        self.frameSelected.emit(frame_idx)
