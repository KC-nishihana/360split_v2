"""Output writers for CSV, manifest, and image files."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from osv_keyframe_app.config import AppConfig
from osv_keyframe_app.selector import SelectedFrame, SelectionResult

logger = logging.getLogger(__name__)


MANIFEST_FIELDS = [
    "frame_idx", "timestamp", "stream", "direction",
    "tier", "adopted", "reason", "score",
    "laplacian_var", "tenengrad", "mean_intensity", "clipped_high_ratio", "clipped_low_ratio",
    "exposure_score", "orb_keypoints", "ssim_prev",
    "filename",
]


def write_manifest_csv(
    result: SelectionResult,
    all_frame_count: int,
    path: Path,
) -> None:
    """Write manifest.csv with adoption status for all frames."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup of selected frames
    sfm_keys: Set[tuple] = set()
    gs_keys: Set[tuple] = set()
    frame_data: Dict[tuple, SelectedFrame] = {}

    for sf in result.sfm_frames:
        key = (sf.frame_idx, sf.stream, sf.direction)
        sfm_keys.add(key)
        frame_data[key] = sf

    for sf in result.gs_frames:
        key = (sf.frame_idx, sf.stream, sf.direction)
        gs_keys.add(key)
        if key not in frame_data:
            frame_data[key] = sf

    rows: List[Dict[str, Any]] = []
    for key, sf in sorted(frame_data.items()):
        m = sf.metrics
        if key in sfm_keys:
            tier = "sfm+gs"
        else:
            tier = "gs"

        rows.append({
            "frame_idx": m.frame_idx,
            "timestamp": f"{m.timestamp:.4f}",
            "stream": m.stream,
            "direction": m.direction,
            "tier": tier,
            "adopted": True,
            "reason": sf.reason,
            "score": f"{sf.score:.4f}",
            "laplacian_var": f"{m.laplacian_var:.2f}",
            "tenengrad": f"{m.tenengrad:.2f}",
            "mean_intensity": f"{m.mean_intensity:.2f}",
            "clipped_high_ratio": f"{m.clipped_high_ratio:.6f}",
            "clipped_low_ratio": f"{m.clipped_low_ratio:.6f}",
            "exposure_score": f"{m.exposure_score:.4f}",
            "orb_keypoints": m.orb_keypoints,
            "ssim_prev": f"{m.ssim_prev:.6f}",
            "filename": m.filename,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote manifest with {len(rows)} entries to {path}")


def copy_selected_images(
    result: SelectionResult,
    projected_dir: Path,
    output_dir: Path,
) -> None:
    """Copy selected images to out/sfm/images/ and out/gs/images/.

    Parameters
    ----------
    result : SelectionResult
    projected_dir : directory containing projected pinhole images
    output_dir : base output directory (will create sfm/images and gs/images under it)
    """
    sfm_dir = output_dir / "sfm" / "images"
    gs_dir = output_dir / "gs" / "images"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    gs_dir.mkdir(parents=True, exist_ok=True)

    sfm_count = 0
    for sf in result.sfm_frames:
        src = projected_dir / sf.metrics.filename
        if src.exists():
            shutil.copy2(src, sfm_dir / sf.metrics.filename)
            sfm_count += 1
        else:
            logger.warning(f"SfM source image not found: {src}")

    gs_count = 0
    for sf in result.gs_frames:
        src = projected_dir / sf.metrics.filename
        if src.exists():
            shutil.copy2(src, gs_dir / sf.metrics.filename)
            gs_count += 1
        else:
            logger.warning(f"3DGS source image not found: {src}")

    logger.info(f"Copied images: SfM={sfm_count}, 3DGS={gs_count}")


def write_config_json(config: AppConfig, path: Path) -> None:
    """Write config_used.json for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_json_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote config to {path}")
