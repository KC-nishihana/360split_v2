"""Shared Stage1 scanning engine for GUI and selector."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.quality_evaluator import QualityEvaluator
from core.quality_score import (
    apply_abs_guard,
    compose_quality,
    compose_legacy_quality_proxy,
    compute_raw_metrics,
    compute_raw_metrics_batch,
    normalize_batch_p10_p90,
    parse_roi_spec,
)
from core.accelerator import get_accelerator
from core.video_loader import FramePrefetcher, create_video_capture


def _raw_metrics_worker(payload: Tuple[np.ndarray, str, bool, float]) -> Dict[str, float]:
    frame, roi_text, use_orb, tenengrad_scale = payload
    return compute_raw_metrics(
        frame,
        roi_spec=roi_text,
        use_orb=use_orb,
        tenengrad_scale=tenengrad_scale,
    )


def _compute_raw_batch(
    frames: List[np.ndarray],
    *,
    roi_text: str,
    use_orb: bool,
    process_workers: int,
    gpu_batch_enabled: bool,
    tenengrad_scale: float,
) -> List[Dict[str, float]]:
    if not frames:
        return []

    if process_workers > 1:
        payloads = [(f, roi_text, use_orb, tenengrad_scale) for f in frames]
        with ProcessPoolExecutor(max_workers=process_workers) as ex:
            return list(ex.map(_raw_metrics_worker, payloads))

    return compute_raw_metrics_batch(
        frames,
        roi_spec=roi_text,
        use_orb=use_orb,
        orb=None,
        gpu_batch_enabled=gpu_batch_enabled,
        tenengrad_scale=tenengrad_scale,
    )


def run_stage1_mono_scan(
    *,
    video_path: str,
    config: Dict[str, Any],
    sample_interval: int,
    is_running_cb: Optional[Callable[[], bool]] = None,
    on_progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Run Stage1 mono scan and return both candidates and records.

    Returns:
      {
        "candidates": List[{"frame_idx": int, "quality_scores": dict}],
        "records": List[dict],
      }
    """
    stop_cb = is_running_cb or (lambda: True)

    backend_pref = str(
        config.get("DARWIN_CAPTURE_BACKEND", config.get("darwin_capture_backend", "auto")) or "auto"
    )
    cap = create_video_capture(video_path, backend_preference=backend_pref)
    if not cap.isOpened():
        raise RuntimeError(f"ビデオを開けません: {video_path}")
    prefetch_size = int(max(1, config.get("STAGE1_PREFETCH_SIZE", config.get("stage1_prefetch_size", 32))))
    prefetcher = FramePrefetcher(
        video_path,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        prefetch_size=prefetch_size,
        backend_preference=backend_pref,
    )
    prefetcher.start()

    try:
        accel = get_accelerator()
        accel.configure_runtime(config)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        sample_interval = max(1, int(sample_interval))
        grab_threshold = int(config.get("STAGE1_GRAB_THRESHOLD", config.get("stage1_grab_threshold", 30)))
        use_grab = bool(sample_interval > 1 and sample_interval <= grab_threshold)

        frame_indices = list(range(0, total_frames, sample_interval))

        quality_filter_enabled = bool(config.get("QUALITY_FILTER_ENABLED", config.get("quality_filter_enabled", True)))
        quality_threshold = float(np.clip(config.get("QUALITY_THRESHOLD", config.get("quality_threshold", 0.50)), 0.0, 1.0))
        quality_abs_laplacian_min = float(
            max(0.0, config.get("QUALITY_ABS_LAPLACIAN_MIN", config.get("quality_abs_laplacian_min", 35.0)))
        )
        quality_use_orb = bool(config.get("QUALITY_USE_ORB", config.get("quality_use_orb", True)))
        quality_norm_p_low = float(np.clip(config.get("QUALITY_NORM_P_LOW", config.get("quality_norm_p_low", 10.0)), 0.0, 100.0))
        quality_norm_p_high = float(
            np.clip(
                config.get("QUALITY_NORM_P_HIGH", config.get("quality_norm_p_high", 90.0)),
                quality_norm_p_low,
                100.0,
            )
        )
        roi_mode = str(config.get("QUALITY_ROI_MODE", config.get("quality_roi_mode", "circle"))).strip().lower()
        roi_ratio = float(np.clip(config.get("QUALITY_ROI_RATIO", config.get("quality_roi_ratio", 0.40)), 0.05, 1.0))
        roi_spec = parse_roi_spec(f"{roi_mode}:{roi_ratio}")
        roi_text = f"{roi_spec.mode}:{roi_spec.ratio}"

        quality_weights = {
            "quality_weight_sharpness": float(config.get("QUALITY_WEIGHT_SHARPNESS", config.get("quality_weight_sharpness", 0.40))),
            "quality_weight_tenengrad": float(config.get("QUALITY_WEIGHT_TENENGRAD", config.get("quality_weight_tenengrad", 0.30))),
            "quality_weight_exposure": float(config.get("QUALITY_WEIGHT_EXPOSURE", config.get("quality_weight_exposure", 0.15))),
            "quality_weight_keypoints": float(config.get("QUALITY_WEIGHT_KEYPOINTS", config.get("quality_weight_keypoints", 0.15))),
        }
        quality_fields = ("laplacian_var", "tenengrad", "exposure", "orb_keypoints")
        quality_tenengrad_scale = float(
            np.clip(
                config.get("QUALITY_TENENGRAD_SCALE", config.get("quality_tenengrad_scale", 1.0)),
                0.1,
                1.0,
            )
        )

        stage1_metrics_batch_size = int(
            max(1, config.get("STAGE1_METRICS_BATCH_SIZE", config.get("stage1_metrics_batch_size", 64)))
        )
        stage1_gpu_batch_enabled = bool(
            config.get("STAGE1_GPU_BATCH_ENABLED", config.get("stage1_gpu_batch_enabled", True))
        )
        stage1_process_workers = int(
            max(0, config.get("STAGE1_PROCESS_WORKERS", config.get("stage1_process_workers", 0)))
        )
        if stage1_process_workers == 0:
            stage1_process_workers = max((cv2.getNumThreads() or 1) - 1, 1)

        stage1_eval_scale = float(config.get("stage1_eval_scale", config.get("STAGE1_EVAL_SCALE", 0.5)))
        stage1_eval_scale = max(0.1, min(1.0, stage1_eval_scale))
        evaluator = QualityEvaluator(
            eval_scale=stage1_eval_scale,
            motion_blur_method=str(config.get("MOTION_BLUR_METHOD", config.get("motion_blur_method", "legacy"))),
        )

        records: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []

        pending_frames: List[np.ndarray] = []
        pending_meta: List[Tuple[int, float]] = []
        raw_entries: List[Dict[str, Any]] = []

        last_read_idx = -1
        current_pos = -1
        if use_grab and frame_indices:
            first_idx = frame_indices[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
            current_pos = first_idx

        for i, frame_idx in enumerate(frame_indices):
            if not stop_cb():
                break

            frame = prefetcher.get_frame(frame_idx, timeout=0.01)
            if frame is None:
                if use_grab:
                    while current_pos < frame_idx:
                        ok = cap.grab()
                        if not ok:
                            break
                        current_pos += 1
                    if current_pos != frame_idx:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        current_pos = frame_idx
                else:
                    if frame_idx != last_read_idx + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if use_grab:
                    current_pos += 1
                last_read_idx = frame_idx
                if not ret or frame is None:
                    continue
                prefetcher.set_position(frame_idx + 1)

            timestamp = float(frame_idx / max(fps, 1e-6))

            if quality_filter_enabled:
                pending_frames.append(frame)
                pending_meta.append((int(frame_idx), timestamp))
                if len(pending_frames) >= stage1_metrics_batch_size:
                    raws = _compute_raw_batch(
                        pending_frames,
                        roi_text=roi_text,
                        use_orb=quality_use_orb,
                        process_workers=stage1_process_workers,
                        gpu_batch_enabled=stage1_gpu_batch_enabled,
                        tenengrad_scale=quality_tenengrad_scale,
                    )
                    for (fidx, ts), raw in zip(pending_meta, raws):
                        raw_entries.append({"frame_idx": fidx, "timestamp": ts, "raw": raw})
                    pending_frames.clear()
                    pending_meta.clear()
            else:
                quality_scores = evaluator.evaluate_stage1_fast(frame)
                quality_proxy = compose_legacy_quality_proxy(
                    quality_scores,
                    laplacian_threshold=float(config.get("LAPLACIAN_THRESHOLD", config.get("laplacian_threshold", 100.0))),
                    motion_blur_threshold=float(config.get("MOTION_BLUR_THRESHOLD", config.get("motion_blur_threshold", 0.3))),
                    exposure_threshold=float(config.get("EXPOSURE_THRESHOLD", config.get("exposure_threshold", 0.35))),
                )
                passes = bool(
                    quality_scores["sharpness"] >= float(config.get("LAPLACIAN_THRESHOLD", config.get("laplacian_threshold", 100.0)))
                    and quality_scores["motion_blur"] <= float(config.get("MOTION_BLUR_THRESHOLD", config.get("motion_blur_threshold", 0.3)))
                    and quality_scores["exposure"] >= float(config.get("EXPOSURE_THRESHOLD", config.get("exposure_threshold", 0.35)))
                )
                quality_scores["quality"] = float(quality_proxy)
                quality_scores["passes_threshold"] = passes
                if passes:
                    candidates.append({"frame_idx": int(frame_idx), "quality_scores": dict(quality_scores)})
                records.append(
                    {
                        "frame_index": int(frame_idx),
                        "timestamp": timestamp,
                        "quality": float(quality_proxy),
                        "is_pass": passes,
                        "drop_reason": "pass" if passes else "legacy_threshold",
                        "legacy_quality_scores": dict(quality_scores),
                        "raw_metrics": {},
                        "norm_metrics": {},
                    }
                )

            if on_progress_cb is not None:
                on_progress_cb(i + 1, len(frame_indices))

        if quality_filter_enabled:
            if pending_frames:
                raws = _compute_raw_batch(
                    pending_frames,
                    roi_text=roi_text,
                    use_orb=quality_use_orb,
                    process_workers=stage1_process_workers,
                    gpu_batch_enabled=stage1_gpu_batch_enabled,
                    tenengrad_scale=quality_tenengrad_scale,
                )
                for (fidx, ts), raw in zip(pending_meta, raws):
                    raw_entries.append({"frame_idx": fidx, "timestamp": ts, "raw": raw})

            raw_list = [e["raw"] for e in raw_entries]
            norm_list, _stats = normalize_batch_p10_p90(
                raw_list,
                p_low=quality_norm_p_low,
                p_high=quality_norm_p_high,
                fields=quality_fields,
            )
            for entry, norm_rec in zip(raw_entries, norm_list):
                frame_idx = int(entry["frame_idx"])
                raw = entry["raw"]
                quality = compose_quality(norm_rec, quality_weights)
                abs_ok = apply_abs_guard(raw.get("laplacian_var", 0.0), quality_abs_laplacian_min)
                quality_ok = bool(quality >= quality_threshold)
                passes = bool(quality_ok and abs_ok)
                drop_reason = "pass"
                if not abs_ok:
                    drop_reason = "abs_laplacian_guard"
                elif not quality_ok:
                    drop_reason = "quality_below_threshold"

                quality_scores = {
                    "sharpness": float(raw.get("laplacian_var", 0.0)),
                    "exposure": float(raw.get("exposure", 0.0)),
                    "motion_blur": 0.0,
                    "softmax_depth": 0.0,
                    "quality": float(quality),
                    "raw_metrics": dict(raw),
                    "norm_metrics": {k: float(norm_rec.get(f"norm_{k}", 0.0)) for k in quality_fields},
                    "passes_threshold": passes,
                }
                if passes:
                    candidates.append({"frame_idx": frame_idx, "quality_scores": dict(quality_scores)})
                records.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp": float(entry["timestamp"]),
                        "quality": float(quality),
                        "is_pass": passes,
                        "drop_reason": drop_reason,
                        "raw_metrics": dict(raw),
                        "norm_metrics": {k: float(norm_rec.get(f"norm_{k}", 0.0)) for k in quality_fields},
                    }
                )

        return {"candidates": candidates, "records": records}
    finally:
        prefetcher.stop()
        cap.release()
