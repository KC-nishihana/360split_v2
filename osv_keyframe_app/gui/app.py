"""Main GUI application window."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
    QVBoxLayout, QWidget, QMenuBar, QStatusBar,
    QProgressBar, QFileDialog, QMessageBox, QLabel,
)
from PySide6.QtCore import Qt

from osv_keyframe_app.config import AppConfig
from osv_keyframe_app.metrics import FrameMetrics, load_metrics_csv
from osv_keyframe_app.pipeline import PipelineResult
from osv_keyframe_app.gui.metrics_tab import MetricsTab
from osv_keyframe_app.gui.threshold_panel import ThresholdPanel
from osv_keyframe_app.gui.direction_preview import DirectionPreview
from osv_keyframe_app.gui.colmap_tab import ColmapTab
from osv_keyframe_app.gui.workers import PipelineWorker

logger = logging.getLogger(__name__)


class OSVKeyframeMainWindow(QMainWindow):
    """Main application window for OSV Keyframe App."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._pipeline_worker: Optional[PipelineWorker] = None
        self._pipeline_result: Optional[PipelineResult] = None
        self.setWindowTitle("OSV Keyframe Extractor")
        self.setMinimumSize(1200, 800)
        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tab widget
        self._tabs = QTabWidget()

        # Metrics tab
        self._metrics_tab = MetricsTab()
        self._tabs.addTab(self._metrics_tab, "Metrics")

        # Selection tab (threshold sliders + direction preview)
        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)

        self._threshold_panel = ThresholdPanel(self._config.selection)
        selection_layout.addWidget(self._threshold_panel)

        self._direction_preview = DirectionPreview()
        selection_layout.addWidget(self._direction_preview)

        self._tabs.addTab(selection_widget, "Selection")

        # COLMAP tab
        self._colmap_tab = ColmapTab(self._config)
        self._tabs.addTab(self._colmap_tab, "COLMAP")

        layout.addWidget(self._tabs)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._status.addPermanentWidget(self._progress_bar)

        self._status.showMessage("Ready - Open an OSV file or load existing metrics")

    def _setup_menu(self) -> None:
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("File")

        open_osv = file_menu.addAction("Open OSV File...")
        open_osv.triggered.connect(self._open_osv)

        load_metrics = file_menu.addAction("Load Metrics CSV...")
        load_metrics.triggered.connect(self._load_metrics)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

    def _open_osv(self) -> None:
        """Open OSV file and run the pipeline."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OSV File", "", "OSV Files (*.osv);;All Files (*)",
        )
        if not path:
            return

        self._config.osv_path = path
        self._run_pipeline(path)

    def _load_metrics(self) -> None:
        """Load existing metrics CSV for visualization."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Metrics CSV", "", "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            metrics = load_metrics_csv(Path(path))
            self._apply_metrics(metrics)
            self._status.showMessage(f"Loaded {len(metrics)} metrics from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _run_pipeline(self, osv_path: str) -> None:
        """Run the full pipeline in a background thread."""
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Pipeline is already running.")
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status.showMessage("Running pipeline...")

        worker = PipelineWorker(self._config, osv_path)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_pipeline_finished)
        worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker = worker
        worker.start()

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress_bar.setValue(int(fraction * 100))
        self._status.showMessage(message)

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        self._pipeline_result = result
        self._progress_bar.setVisible(False)
        self._status.showMessage(
            f"Pipeline complete: {len(result.all_metrics)} metrics, "
            f"SfM={result.selection.sfm_count if result.selection else 0}, "
            f"3DGS={result.selection.gs_count if result.selection else 0}"
        )

        self._apply_metrics(result.all_metrics)

        # Set up direction preview
        frame_indices = sorted({m.frame_idx for m in result.all_metrics})
        self._direction_preview.set_projected_dir(result.projected_dir, frame_indices)

    def _on_pipeline_error(self, message: str) -> None:
        self._progress_bar.setVisible(False)
        self._status.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Pipeline Error", message)

    def _apply_metrics(self, metrics: list[FrameMetrics]) -> None:
        """Apply metrics data to all relevant widgets."""
        self._metrics_tab.set_metrics(metrics)
        self._threshold_panel.set_metrics(metrics)


def launch_gui(config: AppConfig) -> None:
    """Launch the GUI application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = OSVKeyframeMainWindow(config)
    window.show()
    sys.exit(app.exec())
