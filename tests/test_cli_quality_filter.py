from types import SimpleNamespace

import cv2
import numpy as np

from core.keyframe_selector import KeyframeSelector
from main import write_quality_metrics


def _write_video(path, frames, fps=30.0):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


class _DummyMonoLoader:
    def __init__(self, video_path: str):
        self._video_path = video_path
        self.is_paired = False


def _build_frames():
    h, w = 240, 320
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    blur = cv2.GaussianBlur(checker, (21, 21), 0)
    dark = (checker * 0.15).astype(np.uint8)
    mid = cv2.addWeighted(checker, 0.6, blur, 0.4, 0)
    return [checker, blur, dark, mid]


def test_quality_filter_default_on_and_writes_metrics(tmp_path):
    frames = _build_frames()
    video_path = tmp_path / "input.mp4"
    _write_video(video_path, frames)
    loader = _DummyMonoLoader(str(video_path))
    meta = SimpleNamespace(frame_count=len(frames), fps=30.0)

    selector = KeyframeSelector()
    _ = selector._stage1_fast_filter(loader, meta, progress_callback=None)

    assert len(selector.stage1_quality_records) > 0
    assert any(rec.get("quality") is not None for rec in selector.stage1_quality_records)

    json_path, csv_path, summary = write_quality_metrics(tmp_path, selector.stage1_quality_records)
    assert json_path.exists()
    assert csv_path.exists()
    assert summary["total_records"] == len(selector.stage1_quality_records)


def test_no_quality_filter_falls_back_to_legacy_thresholding(tmp_path):
    frames = _build_frames()
    video_path = tmp_path / "input_legacy.mp4"
    _write_video(video_path, frames)
    loader = _DummyMonoLoader(str(video_path))
    meta = SimpleNamespace(frame_count=len(frames), fps=30.0)

    selector = KeyframeSelector(
        config={
            "quality_filter_enabled": False,
            "laplacian_threshold": 0.0,
            "motion_blur_threshold": 1.0,
            "exposure_threshold": 0.0,
        }
    )
    candidates = selector._stage1_fast_filter(loader, meta, progress_callback=None)
    assert len(candidates) > 0
    assert all(isinstance(rec.get("quality"), float) for rec in selector.stage1_quality_records)
    assert all("legacy_quality_scores" in rec for rec in selector.stage1_quality_records)


def test_quality_threshold_is_monotonic(tmp_path):
    frames = _build_frames()
    video_path = tmp_path / "input_threshold.mp4"
    _write_video(video_path, frames)
    loader = _DummyMonoLoader(str(video_path))
    meta = SimpleNamespace(frame_count=len(frames), fps=30.0)

    selector_lo = KeyframeSelector(
        config={
            "quality_filter_enabled": True,
            "quality_threshold": 0.2,
        }
    )
    lo = selector_lo._stage1_fast_filter(loader, meta, progress_callback=None)

    selector_hi = KeyframeSelector(
        config={
            "quality_filter_enabled": True,
            "quality_threshold": 0.8,
        }
    )
    hi = selector_hi._stage1_fast_filter(loader, meta, progress_callback=None)

    assert len(hi) <= len(lo)
