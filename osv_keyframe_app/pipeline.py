"""Full pipeline orchestrator: OSV → split → project → metrics → select → output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from osv_keyframe_app.config import AppConfig
from osv_keyframe_app.osv_splitter import SplitResult, split_osv
from osv_keyframe_app.fisheye_projector import FisheyeProjector
from osv_keyframe_app.metrics import FrameMetrics, MetricsComputer, write_metrics_csv
from osv_keyframe_app.selector import SelectionResult, select_two_tier
from osv_keyframe_app.outputs import (
    copy_selected_images,
    write_config_json,
    write_manifest_csv,
)

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[float, str], None]]


@dataclass
class PipelineResult:
    """Result of the full pipeline execution."""

    split: SplitResult
    all_metrics: List[FrameMetrics] = field(default_factory=list)
    selection: Optional[SelectionResult] = None
    output_dir: Path = Path("out")
    projected_dir: Path = Path("out/projected")
    metrics_csv: Path = Path("out/metrics_all.csv")
    manifest_csv: Path = Path("out/manifest.csv")


class Pipeline:
    """Orchestrate the full OSV keyframe extraction pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._projector = FisheyeProjector(config)
        self._metrics = MetricsComputer()

    def run(
        self,
        osv_path: str | Path,
        on_progress: ProgressCallback = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Steps:
        1. Split OSV into front/back streams
        2. Iterate frames at configured FPS
        3. Project each frame to 4 directions → save intermediate images
        4. Compute metrics for all projected images
        5. Run 2-tier selection (SfM + 3DGS)
        6. Copy selected images to output directories
        7. Write metrics_all.csv, manifest.csv, config_used.json
        """
        osv_path = Path(osv_path)
        output_dir = Path(self._config.output_dir)
        projected_dir = output_dir / "projected"
        projected_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(
            split=SplitResult(Path(), Path(), 0.0, 0, 0, 0),
            output_dir=output_dir,
            projected_dir=projected_dir,
            metrics_csv=output_dir / "metrics_all.csv",
            manifest_csv=output_dir / "manifest.csv",
        )

        # Step 1: Split OSV
        self._report(on_progress, 0.0, "Splitting OSV streams...")
        split = split_osv(osv_path, output_dir / "streams")
        result.split = split

        # Step 2-4: Iterate, project, compute metrics
        self._report(on_progress, 0.05, "Extracting and projecting frames...")
        all_metrics: List[FrameMetrics] = []
        self._metrics.reset()

        frame_indices = self._compute_frame_indices(split)
        total_frames = len(frame_indices)

        if total_frames == 0:
            logger.warning("No frames to process")
            result.all_metrics = all_metrics
            return result

        cap_front = cv2.VideoCapture(str(split.front_path))
        cap_back = cv2.VideoCapture(str(split.back_path))

        try:
            for i, frame_idx in enumerate(frame_indices):
                progress = 0.05 + 0.80 * (i / total_frames)
                self._report(
                    on_progress, progress,
                    f"Processing frame {frame_idx} ({i + 1}/{total_frames})"
                )

                timestamp = frame_idx / max(split.fps, 1.0)

                # Read frames from both streams
                front_frame = self._read_frame(cap_front, frame_idx)
                back_frame = self._read_frame(cap_back, frame_idx)

                # Project and compute metrics for each stream
                for stream, frame in [("front", front_frame), ("back", back_frame)]:
                    if frame is None:
                        continue

                    projected = self._projector.project(frame, stream)

                    # Save projected images
                    for direction, img in projected.items():
                        filename = f"{stream}_{direction}_{frame_idx:08d}.jpg"
                        cv2.imwrite(str(projected_dir / filename), img)

                    # Compute metrics
                    frame_metrics = self._metrics.compute_frame(
                        projected, frame_idx, timestamp, stream,
                    )
                    all_metrics.extend(frame_metrics)

        finally:
            cap_front.release()
            cap_back.release()

        result.all_metrics = all_metrics

        # Step 4.5: Write metrics CSV
        self._report(on_progress, 0.85, "Writing metrics CSV...")
        write_metrics_csv(all_metrics, result.metrics_csv)

        # Step 5: Selection
        self._report(on_progress, 0.88, "Running 2-tier selection...")
        selection = select_two_tier(all_metrics, self._config.selection)
        result.selection = selection

        # Step 6: Copy selected images
        self._report(on_progress, 0.92, "Copying selected images...")
        copy_selected_images(selection, projected_dir, output_dir)

        # Step 7: Write manifest and config
        self._report(on_progress, 0.96, "Writing manifest and config...")
        write_manifest_csv(selection, len(all_metrics), result.manifest_csv)
        write_config_json(self._config, output_dir / "config_used.json")

        self._report(on_progress, 1.0, "Pipeline complete")
        logger.info(
            f"Pipeline complete: {len(all_metrics)} metrics, "
            f"SfM={selection.sfm_count}, 3DGS={selection.gs_count}"
        )

        return result

    def _compute_frame_indices(self, split: SplitResult) -> List[int]:
        """Compute frame indices to extract based on config."""
        cfg = self._config.extraction
        source_fps = split.fps if split.fps > 0 else 30.0

        # Frame step: extract at config.fps from source
        step = max(1, int(round(source_fps / cfg.fps)))
        start_frame = int(cfg.start_sec * source_fps)

        if cfg.end_sec is not None:
            end_frame = int(cfg.end_sec * source_fps)
        else:
            end_frame = split.frame_count

        end_frame = min(end_frame, split.frame_count)

        indices = list(range(start_frame, end_frame, step))

        if cfg.max_frames is not None:
            indices = indices[:cfg.max_frames]

        logger.info(
            f"Frame indices: {len(indices)} frames "
            f"(step={step}, range=[{start_frame}, {end_frame}))"
        )
        return indices

    @staticmethod
    def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame from a video capture."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame

    @staticmethod
    def _report(callback: ProgressCallback, progress: float, message: str) -> None:
        """Report progress via callback."""
        if callback is not None:
            callback(progress, message)
        logger.debug(f"[{progress:.0%}] {message}")
