"""COLMAP control tab with run buttons, log display, and registration table."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QGroupBox, QMessageBox,
    QSplitter,
)
from PySide6.QtCore import Qt

from osv_keyframe_app.config import AppConfig
from osv_keyframe_app.colmap_runner import ColmapRunner, ColmapResult, RegistrationStatus
from osv_keyframe_app.gui.workers import (
    ColmapSfMWorker, ColmapRegistrationWorker, ColmapTriangulatorWorker,
)

logger = logging.getLogger(__name__)


class ColmapTab(QWidget):
    """Tab for COLMAP execution and result display."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._runner: Optional[ColmapRunner] = None
        self._current_worker = None
        self._registered: List[RegistrationStatus] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Control buttons
        btn_group = QGroupBox("COLMAP Operations")
        btn_layout = QHBoxLayout(btn_group)

        self._btn_sfm = QPushButton("Run SfM (mapper)")
        self._btn_sfm.clicked.connect(self._run_sfm)
        btn_layout.addWidget(self._btn_sfm)

        self._btn_register = QPushButton("Run Registration (3DGS)")
        self._btn_register.clicked.connect(self._run_registration)
        btn_layout.addWidget(self._btn_register)

        self._btn_triangulate = QPushButton("Run Triangulator")
        self._btn_triangulate.clicked.connect(self._run_triangulator)
        btn_layout.addWidget(self._btn_triangulate)

        layout.addWidget(btn_group)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._status_label)

        # Splitter for log and table
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Log display
        log_group = QGroupBox("COLMAP Log")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        log_layout.addWidget(self._log_text)
        splitter.addWidget(log_group)

        # Registration table
        table_group = QGroupBox("Registration Results")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Filename", "Registered", "Position (x,y,z)", "Rotation (qw,qx,qy,qz)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self._table)

        # Save CSV button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self._btn_save_csv = QPushButton("Save Registration CSV")
        self._btn_save_csv.clicked.connect(self._save_csv)
        self._btn_save_csv.setEnabled(False)
        save_layout.addWidget(self._btn_save_csv)
        table_layout.addLayout(save_layout)

        splitter.addWidget(table_group)
        layout.addWidget(splitter)

    def _get_runner(self) -> ColmapRunner:
        if self._runner is None:
            self._runner = ColmapRunner(self._config.colmap, self._config.projection)
        return self._runner

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._btn_sfm.setEnabled(enabled)
        self._btn_register.setEnabled(enabled)
        self._btn_triangulate.setEnabled(enabled)

    def _run_sfm(self) -> None:
        output_dir = Path(self._config.output_dir)
        image_dir = output_dir / "sfm" / "images"
        workspace = output_dir / self._config.colmap.workspace

        if not image_dir.exists() or not any(image_dir.iterdir()):
            QMessageBox.warning(self, "No Images", "SfM images not found. Run the pipeline first.")
            return

        self._set_buttons_enabled(False)
        self._status_label.setText("Running SfM...")
        self._log_text.clear()

        worker = ColmapSfMWorker(self._get_runner(), image_dir, workspace)
        worker.progress.connect(self._on_log_line)
        worker.finished.connect(self._on_sfm_finished)
        worker.error.connect(self._on_error)
        self._current_worker = worker
        worker.start()

    def _run_registration(self) -> None:
        output_dir = Path(self._config.output_dir)
        gs_image_dir = output_dir / "gs" / "images"
        sfm_workspace = output_dir / self._config.colmap.workspace

        if not gs_image_dir.exists() or not any(gs_image_dir.iterdir()):
            QMessageBox.warning(self, "No Images", "3DGS images not found. Run the pipeline first.")
            return

        db_path = sfm_workspace / "database.db"
        if not db_path.exists():
            QMessageBox.warning(self, "No SfM", "SfM database not found. Run SfM first.")
            return

        self._set_buttons_enabled(False)
        self._status_label.setText("Running image registration...")
        self._log_text.clear()

        worker = ColmapRegistrationWorker(
            self._get_runner(), gs_image_dir, sfm_workspace,
        )
        worker.progress.connect(self._on_log_line)
        worker.finished.connect(self._on_registration_finished)
        worker.error.connect(self._on_error)
        self._current_worker = worker
        worker.start()

    def _run_triangulator(self) -> None:
        output_dir = Path(self._config.output_dir)
        workspace = output_dir / self._config.colmap.workspace

        # Use combined image dirs
        image_dir = output_dir / "gs" / "images"
        if not image_dir.exists():
            image_dir = output_dir / "sfm" / "images"

        self._set_buttons_enabled(False)
        self._status_label.setText("Running point triangulator...")
        self._log_text.clear()

        worker = ColmapTriangulatorWorker(
            self._get_runner(), image_dir, workspace,
        )
        worker.progress.connect(self._on_log_line)
        worker.finished.connect(self._on_triangulator_finished)
        worker.error.connect(self._on_error)
        self._current_worker = worker
        worker.start()

    def _on_log_line(self, line: str) -> None:
        self._log_text.append(line)

    def _on_sfm_finished(self, result: ColmapResult) -> None:
        self._set_buttons_enabled(True)
        if result.success:
            self._status_label.setText(f"SfM complete: {result.message}")
            self._registered = result.registered
            self._update_table()
        else:
            self._status_label.setText(f"SfM failed: {result.message}")
            QMessageBox.warning(self, "SfM Failed", result.message)

    def _on_registration_finished(self, result: ColmapResult) -> None:
        self._set_buttons_enabled(True)
        if result.success:
            self._status_label.setText(f"Registration complete: {result.message}")
            self._registered = result.registered
            self._update_table()
        else:
            self._status_label.setText(f"Registration failed: {result.message}")
            QMessageBox.warning(self, "Registration Failed", result.message)

    def _on_triangulator_finished(self, result: ColmapResult) -> None:
        self._set_buttons_enabled(True)
        if result.success:
            self._status_label.setText(f"Triangulation complete: {result.message}")
        else:
            self._status_label.setText(f"Triangulation failed: {result.message}")
            QMessageBox.warning(self, "Triangulation Failed", result.message)

    def _on_error(self, message: str) -> None:
        self._set_buttons_enabled(True)
        self._status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "COLMAP Error", message)

    def _update_table(self) -> None:
        """Update registration results table."""
        self._table.setRowCount(len(self._registered))
        for i, reg in enumerate(self._registered):
            self._table.setItem(i, 0, QTableWidgetItem(reg.filename))

            status_item = QTableWidgetItem("Yes" if reg.registered else "No")
            if reg.registered:
                status_item.setForeground(Qt.GlobalColor.green)
            else:
                status_item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(i, 1, status_item)

            if reg.registered:
                pos = f"({reg.tx:.3f}, {reg.ty:.3f}, {reg.tz:.3f})"
                rot = f"({reg.qw:.4f}, {reg.qx:.4f}, {reg.qy:.4f}, {reg.qz:.4f})"
            else:
                pos = "-"
                rot = "-"
            self._table.setItem(i, 2, QTableWidgetItem(pos))
            self._table.setItem(i, 3, QTableWidgetItem(rot))

        self._btn_save_csv.setEnabled(bool(self._registered))

    def _save_csv(self) -> None:
        """Save registration results as CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Registration CSV", "registration.csv", "CSV Files (*.csv)",
        )
        if not path:
            return

        output_path = Path(path)
        all_images = [r.filename for r in self._registered]
        ColmapRunner.write_registration_csv(self._registered, all_images, output_path)
        self._status_label.setText(f"Saved: {output_path}")
