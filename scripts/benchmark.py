#!/usr/bin/env python3
"""360Split benchmark utility."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import ConfigManager
from core.keyframe_selector import KeyframeSelector
from core.pipeline import run_stage1_filter, run_stage2_evaluator, run_stage3_refiner
from core.video_loader import VideoLoader
from utils.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="360Split benchmark")
    p.add_argument("--video", type=str, required=True, help="input video path")
    p.add_argument("--config", type=str, default=None, help="settings json")
    p.add_argument("--preset", type=str, choices=["outdoor", "indoor", "mixed"], default=None)
    p.add_argument("--stages", type=str, default="0,1,2,3", help="comma separated stage ids")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--compare", type=str, default=None, help="previous benchmark json")
    return p.parse_args()


def _load_config(config_path: str | None, preset_id: str | None) -> Dict[str, Any]:
    manager = ConfigManager()
    cfg = manager.default_config()
    if preset_id:
        cfg = manager.load_preset(preset_id, cfg)
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    cfg["enable_profile"] = True
    cfg["stage2_perf_profile"] = True
    return cfg


def _stage_result(stage: str, elapsed_s: float, frames: int) -> Dict[str, Any]:
    fps = float(frames / elapsed_s) if elapsed_s > 1e-9 else 0.0
    return {
        "stage": stage,
        "elapsed_seconds": float(elapsed_s),
        "frames": int(frames),
        "throughput_fps": fps,
    }


def run_benchmark(video_path: str, config: Dict[str, Any], enabled_stages: set[str]) -> Dict[str, Any]:
    loader = VideoLoader(config=config)
    loader.load(video_path)
    metadata = loader.get_metadata()
    if metadata is None:
        raise RuntimeError("metadata unavailable")

    selector = KeyframeSelector(config=config)

    stage0_metrics: Dict[int, Dict[str, Any]] = {}
    stage1_candidates: List[Dict[str, Any]] = []
    stage2_candidates: List[Any] = []
    stage2_records: List[Any] = []

    results: List[Dict[str, Any]] = []

    if "0" in enabled_stages:
        t0 = time.perf_counter()
        stage0_metrics = selector._stage0_lightweight_motion_scan(loader, metadata, None, None)
        results.append(_stage_result("0", time.perf_counter() - t0, len(stage0_metrics)))

    if "1" in enabled_stages or "2" in enabled_stages or "3" in enabled_stages:
        t1 = time.perf_counter()
        stage1_candidates = run_stage1_filter(selector, loader, metadata, None)
        sampled = len(range(0, int(metadata.frame_count), max(1, int(config.get("sample_interval", 1)))))
        results.append(_stage_result("1", time.perf_counter() - t1, sampled))

    if "2" in enabled_stages or "3" in enabled_stages:
        t2 = time.perf_counter()
        stage2_candidates, stage2_records = run_stage2_evaluator(
            selector,
            loader,
            metadata,
            stage1_candidates,
            None,
            None,
            stage0_metrics,
        )
        results.append(_stage_result("2", time.perf_counter() - t2, len(stage1_candidates)))

    keyframes_count = 0
    if "3" in enabled_stages:
        t3 = time.perf_counter()
        stage2_final = selector._enforce_max_interval(
            selector._apply_nms(stage2_candidates),
            metadata.fps,
            source_candidates=stage2_candidates,
        )
        stage3_candidates = run_stage3_refiner(
            selector,
            metadata=metadata,
            stage2_candidates=stage2_candidates,
            stage2_final=stage2_final,
            stage2_records=stage2_records,
            stage0_metrics=stage0_metrics,
            video_loader=loader,
        )
        keyframes = selector._enforce_max_interval(
            selector._apply_nms(stage3_candidates),
            metadata.fps,
            source_candidates=stage3_candidates,
        )
        keyframes_count = len(keyframes)
        results.append(_stage_result("3", time.perf_counter() - t3, len(stage2_candidates)))

    loader.close()

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "video": {
            "path": str(video_path),
            "frame_count": int(metadata.frame_count),
            "fps": float(metadata.fps),
            "width": int(metadata.width),
            "height": int(metadata.height),
        },
        "config_snapshot": {
            "opencv_thread_count": int(config.get("opencv_thread_count", 0)),
            "stage1_process_workers": int(config.get("stage1_process_workers", 0)),
            "stage1_prefetch_size": int(config.get("stage1_prefetch_size", 32)),
            "stage1_metrics_batch_size": int(config.get("stage1_metrics_batch_size", 64)),
            "stage1_gpu_batch_enabled": bool(config.get("stage1_gpu_batch_enabled", True)),
            "darwin_capture_backend": str(config.get("darwin_capture_backend", "auto")),
            "mps_min_pixels": int(config.get("mps_min_pixels", 256 * 256)),
        },
        "stages": results,
        "keyframes": int(keyframes_count),
    }


def _save_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _save_csv(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "elapsed_seconds", "frames", "throughput_fps"])
        writer.writeheader()
        for row in report.get("stages", []):
            writer.writerow(
                {
                    "stage": row.get("stage"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "frames": row.get("frames"),
                    "throughput_fps": row.get("throughput_fps"),
                }
            )


def _print_compare(current: Dict[str, Any], previous: Dict[str, Any]) -> None:
    prev_map = {str(s.get("stage")): s for s in previous.get("stages", [])}
    print("\\n=== Benchmark Compare ===")
    for row in current.get("stages", []):
        sid = str(row.get("stage"))
        prev = prev_map.get(sid)
        if not prev:
            continue
        cur_t = float(row.get("elapsed_seconds", 0.0))
        prev_t = float(prev.get("elapsed_seconds", 0.0))
        cur_fps = float(row.get("throughput_fps", 0.0))
        prev_fps = float(prev.get("throughput_fps", 0.0))
        dt = ((cur_t - prev_t) / prev_t * 100.0) if prev_t > 1e-9 else 0.0
        dfps = ((cur_fps - prev_fps) / prev_fps * 100.0) if prev_fps > 1e-9 else 0.0
        print(f"Stage {sid}: time {cur_t:.3f}s ({dt:+.1f}%), fps {cur_fps:.2f} ({dfps:+.1f}%)")


def main() -> int:
    args = _parse_args()
    enabled_stages = {s.strip() for s in str(args.stages).split(",") if s.strip()}
    cfg = _load_config(args.config, args.preset)
    report = run_benchmark(args.video, cfg, enabled_stages)

    print("=== 360Split Benchmark Report ===")
    print(f"Video: {report['video']['path']}")
    for row in report.get("stages", []):
        print(
            f"Stage {row['stage']}: {row['elapsed_seconds']:.3f}s, "
            f"frames={row['frames']}, throughput={row['throughput_fps']:.2f} fps"
        )

    if args.output_json:
        _save_json(Path(args.output_json), report)
        print(f"Saved JSON: {args.output_json}")
    if args.output_csv:
        _save_csv(Path(args.output_csv), report)
        print(f"Saved CSV: {args.output_csv}")

    if args.compare and Path(args.compare).exists():
        with open(args.compare, "r", encoding="utf-8") as f:
            prev = json.load(f)
        _print_compare(report, prev)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
