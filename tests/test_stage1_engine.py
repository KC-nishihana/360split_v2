from pathlib import Path

import cv2
import numpy as np

from core.config_loader import ConfigManager
from core.stage1_engine import run_stage1_mono_scan


def _write_test_video(path: Path, n_frames: int = 16, w: int = 320, h: int = 180) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter")
    try:
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10 + i, 20), (120 + i, 130), (40 + i * 5, 160, 220 - i * 3), -1)
            cv2.putText(frame, f"f{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def test_stage1_engine_single_vs_multi_process_equivalence(tmp_path):
    video_path = tmp_path / "stage1_eq.mp4"
    _write_test_video(video_path)

    base = ConfigManager.default_config()
    base.update(
        {
            "quality_filter_enabled": True,
            "quality_use_orb": False,
            "stage1_metrics_batch_size": 4,
            "stage1_gpu_batch_enabled": False,
            "stage1_process_workers": 1,
            "stage1_grab_threshold": 30,
            "darwin_capture_backend": "auto",
        }
    )

    r1 = run_stage1_mono_scan(
        video_path=str(video_path),
        config=base,
        sample_interval=1,
    )

    multi = dict(base)
    multi["stage1_process_workers"] = 2
    r2 = run_stage1_mono_scan(
        video_path=str(video_path),
        config=multi,
        sample_interval=1,
    )

    rec1 = {int(r["frame_index"]): float(r.get("quality", 0.0) or 0.0) for r in r1["records"]}
    rec2 = {int(r["frame_index"]): float(r.get("quality", 0.0) or 0.0) for r in r2["records"]}
    assert rec1.keys() == rec2.keys()
    for k in rec1:
        assert abs(rec1[k] - rec2[k]) <= 1e-4

    c1 = [int(c["frame_idx"]) for c in r1["candidates"]]
    c2 = [int(c["frame_idx"]) for c in r2["candidates"]]
    assert c1 == c2
