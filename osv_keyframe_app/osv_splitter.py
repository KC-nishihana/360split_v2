"""Split OSV file into front/back video streams using ffmpeg."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Result of OSV stream splitting."""

    front_path: Path
    back_path: Path
    fps: float
    frame_count: int
    width: int
    height: int


def _probe_video(path: Path) -> dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"ffprobe failed for {path}: {e}")
        return {}


def _get_video_info(probe: dict) -> tuple[float, int, int, int]:
    """Extract fps, frame_count, width, height from ffprobe output."""
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            # Parse FPS from r_frame_rate (e.g. "30/1")
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / max(float(den), 1.0)
            else:
                fps = float(fps_str)
            frame_count = int(stream.get("nb_frames", 0))
            if frame_count == 0:
                # Estimate from duration
                duration = float(stream.get("duration", probe.get("format", {}).get("duration", 0)))
                frame_count = int(duration * fps) if duration > 0 else 0
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            return fps, frame_count, width, height
    return 30.0, 0, 0, 0


def split_osv(
    osv_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> SplitResult:
    """Split .osv into front/back streams using ffmpeg.

    Parameters
    ----------
    osv_path : path to OSV file
    output_dir : directory for output .mp4 files
    force : re-split even if output files exist

    Returns
    -------
    SplitResult with paths and metadata
    """
    osv_path = Path(osv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    front_path = output_dir / "front.mp4"
    back_path = output_dir / "back.mp4"

    if not force and front_path.exists() and back_path.exists():
        logger.info(f"Split files already exist, skipping: {front_path}, {back_path}")
        probe = _probe_video(front_path)
        fps, frame_count, width, height = _get_video_info(probe)
        return SplitResult(
            front_path=front_path, back_path=back_path,
            fps=fps, frame_count=frame_count, width=width, height=height,
        )

    # OSV: stream 0 = front, stream 1 = back
    cmd = [
        "ffmpeg", "-y", "-i", str(osv_path),
        "-map", "0:0", "-c", "copy", str(front_path),
        "-map", "0:1", "-c", "copy", str(back_path),
    ]
    logger.info(f"Splitting OSV: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg.\n"
            "macOS: brew install ffmpeg\n"
            "Ubuntu: sudo apt install ffmpeg\n"
            "Windows: https://ffmpeg.org/download.html"
        )

    probe = _probe_video(front_path)
    fps, frame_count, width, height = _get_video_info(probe)

    logger.info(
        f"OSV split complete: {width}x{height} @ {fps:.2f}fps, "
        f"{frame_count} frames per stream"
    )

    return SplitResult(
        front_path=front_path, back_path=back_path,
        fps=fps, frame_count=frame_count, width=width, height=height,
    )
