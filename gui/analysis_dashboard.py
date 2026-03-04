"""Analysis dashboard widget for Stage1/2/3/COLMAP diagnostics and tuning."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.analysis_data_loader import summarize_colmap
from gui.parameter_recommender import generate_parameter_recommendations

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except Exception:
    HAS_PYQTGRAPH = False
    pg = None  # type: ignore


class AnalysisDashboardWidget(QWidget):
    """Stage-wise analytics dashboard with recommendation apply actions."""

    apply_settings_requested = Signal(dict)
    run_colmap_requested = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._run_id: str = "-"
        self._current_settings: Dict[str, Any] = {}
        self._latest_payload: Dict[str, Any] = {}
        self._batch_stage1_points: List[Dict[str, Any]] = []
        self._recommendations: List[Dict[str, Any]] = []
        self._colmap_candidates: List[Dict[str, Any]] = []

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self._run_label = QLabel("run_id: -")
        self._run_label.setStyleSheet("color: #d0d0d0; font-size: 12px; font-weight: bold;")
        root.addWidget(self._run_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        container = QWidget()
        self._scroll.setWidget(container)
        root.addWidget(self._scroll, stretch=1)

        content = QVBoxLayout(container)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)

        # Stage1 section
        self._stage1_group = QGroupBox("Stage1")
        s1_layout = QVBoxLayout(self._stage1_group)
        self._stage1_kpi = QLabel("pass_rate: -, p10/p90: -/-")
        s1_layout.addWidget(self._stage1_kpi)
        self._stage1_quality_plot = self._create_plot_widget("quality 時系列")
        s1_layout.addWidget(self._stage1_quality_plot)
        self._stage1_pass_plot = self._create_plot_widget("pass/fail")
        s1_layout.addWidget(self._stage1_pass_plot)
        self._stage1_drop_plot = self._create_plot_widget("drop_reason")
        s1_layout.addWidget(self._stage1_drop_plot)
        content.addWidget(self._stage1_group)

        # Stage2 section
        self._stage2_group = QGroupBox("Stage2")
        s2_layout = QVBoxLayout(self._stage2_group)
        self._stage2_kpi = QLabel("read_success_rate: -, final/target: -/-")
        s2_layout.addWidget(self._stage2_kpi)
        self._stage2_combined_plot = self._create_plot_widget("combined_stage2 時系列")
        s2_layout.addWidget(self._stage2_combined_plot)
        self._stage2_drop_plot = self._create_plot_widget("drop_reason")
        s2_layout.addWidget(self._stage2_drop_plot)
        self._stage2_novelty_plot = self._create_plot_widget("novelty 分布")
        s2_layout.addWidget(self._stage2_novelty_plot)
        content.addWidget(self._stage2_group)

        # Stage3 section
        self._stage3_group = QGroupBox("Stage3")
        s3_layout = QVBoxLayout(self._stage3_group)
        self._stage3_kpi = QLabel("alerts: -, coverage: -, contiguous_run: -")
        s3_layout.addWidget(self._stage3_kpi)
        self._stage3_c2c3_plot = self._create_plot_widget("combined_stage2 vs combined_stage3")
        s3_layout.addWidget(self._stage3_c2c3_plot)
        self._stage3_traj_risk_plot = self._create_plot_widget(
            "trajectory_consistency_effective vs stage0_motion_risk"
        )
        s3_layout.addWidget(self._stage3_traj_risk_plot)
        content.addWidget(self._stage3_group)

        # COLMAP section
        self._colmap_group = QGroupBox("COLMAP")
        c_layout = QVBoxLayout(self._colmap_group)
        self._colmap_kpi = QLabel("input_subset: -, trajectory/selected: -/-")
        c_layout.addWidget(self._colmap_kpi)
        self._colmap_input_subset_plot = self._create_plot_widget("input_subset")
        c_layout.addWidget(self._colmap_input_subset_plot)
        self._colmap_subset_drop_plot = self._create_plot_widget("subset drop_reason")
        c_layout.addWidget(self._colmap_subset_drop_plot)
        self._colmap_sparse_plot = self._create_plot_widget("sparse model candidates")
        c_layout.addWidget(self._colmap_sparse_plot)
        content.addWidget(self._colmap_group)

        # COLMAP manual run prep
        prep_group = QGroupBox("COLMAP実行準備")
        prep_layout = QVBoxLayout(prep_group)

        prep_row = QHBoxLayout()
        self._colmap_select_all_btn = QPushButton("全選択")
        self._colmap_select_all_btn.clicked.connect(lambda: self._set_colmap_selection_all(True))
        prep_row.addWidget(self._colmap_select_all_btn)

        self._colmap_clear_all_btn = QPushButton("全解除")
        self._colmap_clear_all_btn.clicked.connect(lambda: self._set_colmap_selection_all(False))
        prep_row.addWidget(self._colmap_clear_all_btn)

        self._colmap_run_btn = QPushButton("COLMAP実行")
        self._colmap_run_btn.clicked.connect(self._on_run_colmap_clicked)
        prep_row.addWidget(self._colmap_run_btn)

        self._colmap_candidate_count = QLabel("候補: 0")
        self._colmap_candidate_count.setStyleSheet("color: #a0a0a0;")
        prep_row.addWidget(self._colmap_candidate_count)
        prep_row.addStretch(1)
        prep_layout.addLayout(prep_row)

        self._colmap_table = QTableWidget(0, 3)
        self._colmap_table.setHorizontalHeaderLabels(["use", "frame_index", "score"])
        self._colmap_table.horizontalHeader().setStretchLastSection(True)
        self._colmap_table.setMinimumHeight(180)
        prep_layout.addWidget(self._colmap_table)

        content.addWidget(prep_group)

        # Recommendation table
        rec_group = QGroupBox("推奨パラメータ")
        rec_layout = QVBoxLayout(rec_group)

        row = QHBoxLayout()
        self._recompute_btn = QPushButton("推奨を再計算")
        self._recompute_btn.clicked.connect(self._on_recompute_recommendations)
        row.addWidget(self._recompute_btn)

        self._apply_all_btn = QPushButton("推奨を一括適用")
        self._apply_all_btn.clicked.connect(self._on_apply_all)
        row.addWidget(self._apply_all_btn)

        self._rec_count = QLabel("0件")
        self._rec_count.setStyleSheet("color: #a0a0a0;")
        row.addWidget(self._rec_count)
        row.addStretch(1)
        rec_layout.addLayout(row)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["stage", "parameter", "current", "suggested", "reason", "apply"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._table.setMinimumHeight(220)
        rec_layout.addWidget(self._table)

        content.addWidget(rec_group)

    def _create_plot_widget(self, title: str) -> QWidget:
        if not HAS_PYQTGRAPH:
            label = QLabel("pyqtgraph が未インストールです。pip install pyqtgraph を実行してください。")
            label.setWordWrap(True)
            label.setStyleSheet("color: #ff8800; background: #202020; border-radius: 4px; padding: 8px;")
            return label
        pg.setConfigOptions(antialias=True)
        plot = pg.PlotWidget()
        plot.setBackground("#1a1a2e")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setMinimumHeight(150)
        plot.setTitle(title)
        return plot

    @staticmethod
    def _fmt_num(value: Any, digits: int = 3) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    @staticmethod
    def _fmt_pct(value: Any) -> str:
        try:
            return f"{float(value) * 100.0:.1f}%"
        except Exception:
            return "-"

    @staticmethod
    def _plot_clear(widget: QWidget) -> None:
        if HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget):
            widget.clear()

    @staticmethod
    def _plot_message(widget: QWidget, title: str) -> None:
        if HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget):
            widget.clear()
            widget.setTitle(title)

    @staticmethod
    def _plot_series(widget: QWidget, x_vals: List[float], y_vals: List[float], *, title: str, color: str) -> None:
        if not (HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget)):
            return
        widget.clear()
        widget.setTitle(title)
        if not x_vals or not y_vals:
            return
        widget.plot(x_vals, y_vals, pen=pg.mkPen(color, width=1.8))

    @staticmethod
    def _plot_scatter(
        widget: QWidget,
        x_vals: List[float],
        y_vals: List[float],
        *,
        title: str,
        color: str,
    ) -> None:
        if not (HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget)):
            return
        widget.clear()
        widget.setTitle(title)
        if not x_vals or not y_vals:
            return
        scatter = pg.ScatterPlotItem(x=x_vals, y=y_vals, size=6, brush=pg.mkBrush(color), pen=None)
        widget.addItem(scatter)

    @staticmethod
    def _plot_bar(widget: QWidget, labels: List[str], values: List[float], *, title: str, color: str) -> None:
        if not (HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget)):
            return
        widget.clear()
        widget.setTitle(title)
        if not labels or not values:
            return
        x = list(range(len(labels)))
        bars = pg.BarGraphItem(x=x, height=values, width=0.65, brush=pg.mkBrush(color))
        widget.addItem(bars)
        axis = widget.getPlotItem().getAxis("bottom")
        axis.setTicks([[(i, labels[i]) for i in range(len(labels))]])

    @staticmethod
    def _plot_hist(widget: QWidget, values: List[float], *, title: str, bins: int = 16) -> None:
        if not (HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget)):
            return
        widget.clear()
        widget.setTitle(title)
        if not values:
            return
        low = min(values)
        high = max(values)
        if high <= low:
            low = low - 0.5
            high = high + 0.5
        width = (high - low) / float(max(1, bins))
        counts = [0 for _ in range(max(1, bins))]
        for v in values:
            idx = int((float(v) - low) / max(width, 1e-9))
            idx = max(0, min(len(counts) - 1, idx))
            counts[idx] += 1
        x = [low + (i * width) for i in range(len(counts))]
        bars = pg.BarGraphItem(x=x, height=counts, width=width * 0.9, brush=pg.mkBrush("#7cc8ff"))
        widget.addItem(bars)

    def reset_for_run(self, run_id: str, current_settings: Dict[str, Any]) -> None:
        self._run_id = str(run_id or "-")
        self._run_label.setText(f"run_id: {self._run_id}")
        self._current_settings = dict(current_settings or {})
        self._latest_payload = {}
        self._batch_stage1_points = []
        self._recommendations = []
        self._colmap_candidates = []
        self._render_all_sections()
        self._update_table([])
        self.set_colmap_candidates([])

    def set_current_settings(self, current_settings: Dict[str, Any]) -> None:
        self._current_settings = dict(current_settings or {})
        self._recompute_recommendations()

    def update_stage1_batch(self, batch: List[Any]) -> None:
        for item in list(batch or []):
            frame_idx = int(getattr(item, "frame_index", -1))
            quality = getattr(item, "quality", None)
            qv = float(quality) if quality is not None else 0.0
            if qv <= 0.0:
                sharp = float(getattr(item, "sharpness", 0.0))
                qv = max(0.0, min(1.0, sharp / 1000.0))
            self._batch_stage1_points.append({"frame_index": frame_idx, "quality": qv})

        has_stage1_artifact = bool(
            (self._latest_payload.get("stage1") or {}).get("records_count", 0)
        )
        if not has_stage1_artifact:
            self._render_stage1_from_batch()

    def update_stage_artifacts(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        next_payload = dict(payload)
        # Keep latest pose summary if poll payload doesn't carry it yet.
        old_colmap = self._latest_payload.get("colmap") if isinstance(self._latest_payload.get("colmap"), dict) else {}
        new_colmap = next_payload.get("colmap") if isinstance(next_payload.get("colmap"), dict) else {}
        if isinstance(old_colmap.get("pose_summary"), dict) and not isinstance(new_colmap.get("pose_summary"), dict):
            new_colmap["pose_summary"] = dict(old_colmap.get("pose_summary", {}))
            analysis_summary = next_payload.get("analysis_summary") if isinstance(next_payload.get("analysis_summary"), dict) else {}
            next_payload["colmap"] = summarize_colmap(analysis_summary, new_colmap.get("pose_summary"))

        self._latest_payload = next_payload
        self._render_all_sections()
        self._recompute_recommendations()

    def update_pose_summary(self, pose_summary: Dict[str, Any]) -> None:
        if not isinstance(pose_summary, dict):
            return
        summary = self._latest_payload.get("analysis_summary")
        analysis_summary = dict(summary) if isinstance(summary, dict) else {}
        self._latest_payload["analysis_summary"] = analysis_summary
        self._latest_payload["colmap"] = summarize_colmap(analysis_summary, pose_summary)
        self._render_colmap_section()
        self._recompute_recommendations()

    def _render_all_sections(self) -> None:
        stage1 = self._latest_payload.get("stage1") if isinstance(self._latest_payload.get("stage1"), dict) else {}
        if stage1.get("records_count", 0) > 0:
            self._render_stage1_section(stage1)
        else:
            self._render_stage1_from_batch()
        self._render_stage2_section()
        self._render_stage3_section()
        self._render_colmap_section()

    def _render_stage1_from_batch(self) -> None:
        if not self._batch_stage1_points:
            self._stage1_kpi.setText("pass_rate: -, p10/p90: -/-, records: 0")
            self._plot_message(self._stage1_quality_plot, "quality 時系列 (データ待ち)")
            self._plot_message(self._stage1_pass_plot, "pass/fail (データ待ち)")
            self._plot_message(self._stage1_drop_plot, "drop_reason (データ待ち)")
            return

        points = sorted(self._batch_stage1_points, key=lambda x: int(x.get("frame_index", -1)))
        x = [float(p.get("frame_index", 0)) for p in points]
        y = [float(p.get("quality", 0.0)) for p in points]
        p10 = sorted(y)[max(0, int(len(y) * 0.10) - 1)] if y else None
        p90 = sorted(y)[max(0, int(len(y) * 0.90) - 1)] if y else None
        self._stage1_kpi.setText(
            f"pass_rate: -, p10/p90: {self._fmt_num(p10, 3)}/{self._fmt_num(p90, 3)}, records: {len(points)} (batch)"
        )
        self._plot_series(self._stage1_quality_plot, x, y, title="quality 時系列", color="#66a3ff")
        self._plot_message(self._stage1_pass_plot, "pass/fail (batch段階では未確定)")
        self._plot_message(self._stage1_drop_plot, "drop_reason (batch段階では未確定)")

    def _render_stage1_section(self, stage1: Dict[str, Any]) -> None:
        quality_series = stage1.get("quality_series") if isinstance(stage1.get("quality_series"), list) else []
        pass_fail = stage1.get("pass_fail") if isinstance(stage1.get("pass_fail"), dict) else {}
        drop_counts = stage1.get("drop_reason_counts") if isinstance(stage1.get("drop_reason_counts"), dict) else {}

        x = [float(_get_num(r, "frame_index")) for r in quality_series]
        y = [float(_get_num(r, "quality")) for r in quality_series]
        self._plot_series(self._stage1_quality_plot, x, y, title="quality 時系列", color="#66a3ff")

        self._plot_bar(
            self._stage1_pass_plot,
            ["pass", "fail"],
            [float(_get_num(pass_fail, "pass")), float(_get_num(pass_fail, "fail"))],
            title="pass/fail",
            color="#87d37c",
        )

        labels = list(drop_counts.keys())
        values = [float(_get_num(drop_counts, k)) for k in labels]
        self._plot_bar(self._stage1_drop_plot, labels, values, title="drop_reason", color="#ffb347")

        self._stage1_kpi.setText(
            "pass_rate: "
            f"{self._fmt_pct(stage1.get('pass_rate'))}, "
            f"p10/p90: {self._fmt_num(stage1.get('quality_p10'), 3)}/{self._fmt_num(stage1.get('quality_p90'), 3)}, "
            f"records: {self._fmt_num(stage1.get('records_count'), 0)}"
        )

    def _render_stage2_section(self) -> None:
        stage2 = self._latest_payload.get("stage2") if isinstance(self._latest_payload.get("stage2"), dict) else {}
        combined = stage2.get("combined_stage2_series") if isinstance(stage2.get("combined_stage2_series"), list) else []
        drop_counts = stage2.get("drop_reason_counts") if isinstance(stage2.get("drop_reason_counts"), dict) else {}
        novelty_vals = stage2.get("novelty_values") if isinstance(stage2.get("novelty_values"), list) else []

        x = [float(_get_num(r, "frame_index")) for r in combined]
        y = [float(_get_num(r, "combined_stage2")) for r in combined]
        self._plot_series(self._stage2_combined_plot, x, y, title="combined_stage2 時系列", color="#f29c9c")

        labels = list(drop_counts.keys())
        values = [float(_get_num(drop_counts, k)) for k in labels]
        self._plot_bar(self._stage2_drop_plot, labels, values, title="drop_reason", color="#f7ca18")

        self._plot_hist(
            self._stage2_novelty_plot,
            [float(v) for v in novelty_vals if isinstance(v, (int, float))],
            title="novelty 分布",
            bins=16,
        )

        self._stage2_kpi.setText(
            "read_success_rate: "
            f"{self._fmt_pct(stage2.get('read_success_rate'))}, "
            f"final/target: {self._fmt_num(stage2.get('final_keyframes'), 0)}/"
            f"{self._fmt_num(stage2.get('target_min'), 0)}-{self._fmt_num(stage2.get('target_max'), 0)}, "
            f"preview: {self._fmt_num(stage2.get('stage2_colmap_preview_count'), 0)}"
        )

    def _render_stage3_section(self) -> None:
        stage3 = self._latest_payload.get("stage3") if isinstance(self._latest_payload.get("stage3"), dict) else {}
        c2c3 = stage3.get("stage2_vs_stage3") if isinstance(stage3.get("stage2_vs_stage3"), list) else []
        traj_risk = stage3.get("trajectory_vs_risk") if isinstance(stage3.get("trajectory_vs_risk"), list) else []
        diagnostics = stage3.get("diagnostics") if isinstance(stage3.get("diagnostics"), dict) else {}

        x1 = [float(_get_num(r, "combined_stage2")) for r in c2c3]
        y1 = [float(_get_num(r, "combined_stage3")) for r in c2c3]
        self._plot_scatter(
            self._stage3_c2c3_plot,
            x1,
            y1,
            title="combined_stage2 vs combined_stage3",
            color="#50e3c2",
        )

        x2 = [float(_get_num(r, "trajectory_consistency_effective")) for r in traj_risk]
        y2 = [float(_get_num(r, "stage0_motion_risk")) for r in traj_risk]
        self._plot_scatter(
            self._stage3_traj_risk_plot,
            x2,
            y2,
            title="trajectory_consistency_effective vs stage0_motion_risk",
            color="#ff7f7f",
        )

        alerts = diagnostics.get("alerts") if isinstance(diagnostics.get("alerts"), list) else []
        alerts_text = ", ".join(str(a) for a in alerts) if alerts else "none"
        self._stage3_kpi.setText(
            f"alerts: {alerts_text}, coverage_bins: {self._fmt_num(diagnostics.get('coverage_bins_occupied'), 0)}, "
            f"contiguous_run: {self._fmt_num(diagnostics.get('contiguous_run_max'), 0)}, "
            f"vo_valid_ratio: {self._fmt_pct(stage3.get('vo_valid_ratio'))}"
        )

    def _render_colmap_section(self) -> None:
        colmap = self._latest_payload.get("colmap") if isinstance(self._latest_payload.get("colmap"), dict) else {}
        input_subset = colmap.get("input_subset") if isinstance(colmap.get("input_subset"), dict) else {}
        subset_drop = colmap.get("subset_drop_reason_counts") if isinstance(colmap.get("subset_drop_reason_counts"), dict) else {}
        sparse_candidates = colmap.get("sparse_model_candidates") if isinstance(colmap.get("sparse_model_candidates"), list) else []

        input_count = _get_num(input_subset, "input_count")
        kept_count = _get_num(input_subset, "kept_count")
        kept_ratio = input_subset.get("kept_ratio")

        self._plot_bar(
            self._colmap_input_subset_plot,
            ["input", "kept"],
            [float(input_count), float(kept_count)],
            title="input_subset",
            color="#7fb3ff",
        )

        labels = list(subset_drop.keys())
        values = [float(_get_num(subset_drop, k)) for k in labels]
        self._plot_bar(self._colmap_subset_drop_plot, labels, values, title="subset drop_reason", color="#f5b7b1")

        self._plot_sparse_candidates(self._colmap_sparse_plot, sparse_candidates)

        self._colmap_kpi.setText(
            "input_subset: "
            f"{int(input_count)}/{int(kept_count)} ({self._fmt_pct(kept_ratio)}), "
            f"auto_relaxed={str(bool(input_subset.get('auto_relaxed', False))).lower()}, "
            f"trajectory/selected: {self._fmt_num(colmap.get('trajectory_count'), 0)}/"
            f"{self._fmt_num(colmap.get('selected_count'), 0)}"
        )

    @staticmethod
    def _plot_sparse_candidates(widget: QWidget, candidates: List[Dict[str, Any]]) -> None:
        if not (HAS_PYQTGRAPH and isinstance(widget, pg.PlotWidget)):
            return
        widget.clear()
        widget.setTitle("sparse model candidates")
        if not candidates:
            return

        rows = [c for c in candidates if isinstance(c, dict)]
        rows = sorted(rows, key=lambda c: int(_get_num(c, "model_id")))
        x = [int(_get_num(c, "model_id")) for c in rows]
        reg = [float(_get_num(c, "registered_count")) for c in rows]
        cov = [float(_get_num(c, "coverage_span")) for c in rows]

        widget.plot(x, reg, pen=pg.mkPen("#7DCEA0", width=1.8), symbol="o", symbolBrush="#7DCEA0")
        widget.plot(x, cov, pen=pg.mkPen("#F5B041", width=1.8), symbol="t", symbolBrush="#F5B041")

    def _on_recompute_recommendations(self) -> None:
        self._recompute_recommendations()

    def _recompute_recommendations(self) -> None:
        self._recommendations = generate_parameter_recommendations(self._latest_payload, self._current_settings)
        self._update_table(self._recommendations)

    def _update_table(self, recommendations: List[Dict[str, Any]]) -> None:
        self._table.setRowCount(len(recommendations))
        self._rec_count.setText(f"{len(recommendations)}件")

        for row_idx, rec in enumerate(recommendations):
            self._table.setItem(row_idx, 0, QTableWidgetItem(str(rec.get("stage", ""))))
            self._table.setItem(row_idx, 1, QTableWidgetItem(str(rec.get("parameter", ""))))
            self._table.setItem(row_idx, 2, QTableWidgetItem(self._fmt_num(rec.get("current"), 4)))
            self._table.setItem(row_idx, 3, QTableWidgetItem(self._fmt_num(rec.get("suggested"), 4)))
            self._table.setItem(row_idx, 4, QTableWidgetItem(str(rec.get("reason", ""))))

            btn = QPushButton("適用")
            setting_key = str(rec.get("setting_key", rec.get("parameter", "")))
            suggested = rec.get("suggested")
            btn.clicked.connect(
                lambda _checked=False, k=setting_key, v=suggested: self.apply_settings_requested.emit({k: v})
            )
            self._table.setCellWidget(row_idx, 5, btn)

        self._table.resizeColumnsToContents()

    def _on_apply_all(self) -> None:
        if not self._recommendations:
            return
        payload: Dict[str, Any] = {}
        for rec in self._recommendations:
            key = str(rec.get("setting_key", rec.get("parameter", "")))
            if not key:
                continue
            payload[key] = rec.get("suggested")
        if payload:
            self.apply_settings_requested.emit(payload)

    def set_colmap_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        normalized: List[Dict[str, Any]] = []
        for row in list(candidates or []):
            if not isinstance(row, dict):
                continue
            try:
                frame_idx = int(row.get("frame_index", -1))
            except Exception:
                continue
            if frame_idx < 0:
                continue
            score = row.get("score")
            try:
                score_v = float(score) if score is not None else 0.0
            except Exception:
                score_v = 0.0
            normalized.append({"frame_index": frame_idx, "score": score_v})

        normalized = sorted(normalized, key=lambda r: int(r.get("frame_index", -1)))
        self._colmap_candidates = normalized
        self._colmap_table.setRowCount(len(normalized))
        self._colmap_candidate_count.setText(f"候補: {len(normalized)}")

        for row_idx, row in enumerate(normalized):
            cb = QCheckBox()
            cb.setChecked(True)
            cb.setStyleSheet("margin-left: 6px;")
            self._colmap_table.setCellWidget(row_idx, 0, cb)
            self._colmap_table.setItem(row_idx, 1, QTableWidgetItem(str(int(row.get("frame_index", -1)))))
            self._colmap_table.setItem(row_idx, 2, QTableWidgetItem(self._fmt_num(row.get("score"), 4)))

        self._colmap_table.resizeColumnsToContents()

    def selected_colmap_frames(self) -> List[int]:
        selected: List[int] = []
        for row_idx in range(self._colmap_table.rowCount()):
            checked = False
            cb = self._colmap_table.cellWidget(row_idx, 0)
            if isinstance(cb, QCheckBox):
                checked = bool(cb.isChecked())
            if not checked:
                continue
            item = self._colmap_table.item(row_idx, 1)
            if item is None:
                continue
            try:
                selected.append(int(item.text()))
            except Exception:
                continue
        return selected

    def _set_colmap_selection_all(self, checked: bool) -> None:
        for row_idx in range(self._colmap_table.rowCount()):
            cb = self._colmap_table.cellWidget(row_idx, 0)
            if isinstance(cb, QCheckBox):
                cb.setChecked(bool(checked))

    def _on_run_colmap_clicked(self) -> None:
        self.run_colmap_requested.emit(self.selected_colmap_frames())



def _get_num(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)
