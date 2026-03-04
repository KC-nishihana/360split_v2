"""QThread workers for async pipeline and COLMAP execution."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from osv_keyframe_app.config import AppConfig
from osv_keyframe_app.pipeline import Pipeline, PipelineResult
from osv_keyframe_app.colmap_runner import ColmapRunner, ColmapResult

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Run the full pipeline in a background thread."""

    progress = Signal(float, str)  # (fraction, message)
    finished = Signal(object)  # PipelineResult or Exception
    error = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        osv_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._osv_path = osv_path

    def run(self) -> None:
        try:
            pipeline = Pipeline(self._config)
            result = pipeline.run(
                self._osv_path,
                on_progress=self._on_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.emit(fraction, message)


class ColmapSfMWorker(QThread):
    """Run COLMAP SfM (mapper) in a background thread."""

    progress = Signal(str)  # log line
    finished = Signal(object)  # ColmapResult
    error = Signal(str)

    def __init__(
        self,
        runner: ColmapRunner,
        image_dir: Path,
        workspace: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._image_dir = image_dir
        self._workspace = workspace

    def run(self) -> None:
        try:
            result = self._runner.run_sfm(
                self._image_dir,
                self._workspace,
                log_callback=self._on_log,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"COLMAP SfM error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))

    def _on_log(self, line: str) -> None:
        self.progress.emit(line)


class ColmapRegistrationWorker(QThread):
    """Run COLMAP image_registrator in a background thread."""

    progress = Signal(str)
    finished = Signal(object)  # ColmapResult
    error = Signal(str)

    def __init__(
        self,
        runner: ColmapRunner,
        gs_image_dir: Path,
        sfm_workspace: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._gs_image_dir = gs_image_dir
        self._sfm_workspace = sfm_workspace

    def run(self) -> None:
        try:
            result = self._runner.run_gs_registration(
                self._gs_image_dir,
                self._sfm_workspace,
                log_callback=self._on_log,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"COLMAP registration error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))

    def _on_log(self, line: str) -> None:
        self.progress.emit(line)


class ColmapTriangulatorWorker(QThread):
    """Run COLMAP point_triangulator in a background thread."""

    progress = Signal(str)
    finished = Signal(object)  # ColmapResult
    error = Signal(str)

    def __init__(
        self,
        runner: ColmapRunner,
        image_dir: Path,
        workspace: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._image_dir = image_dir
        self._workspace = workspace

    def run(self) -> None:
        try:
            result = self._runner.run_point_triangulator(
                self._image_dir,
                self._workspace,
                log_callback=self._on_log,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"COLMAP triangulator error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))

    def _on_log(self, line: str) -> None:
        self.progress.emit(line)
