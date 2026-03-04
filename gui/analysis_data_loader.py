"""Utilities for loading and aggregating analysis artifacts for the dashboard."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        # Keep polling; partially-written files are expected during runtime.
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    # Skip malformed/incomplete lines and retry on the next poll.
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except Exception:
        return []
    return out


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pp = max(0.0, min(100.0, float(p))) / 100.0
    idx = pp * (len(vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    w = idx - lo
    return float(vals[lo] * (1.0 - w) + vals[hi] * w)


def _stage1_sharpness(row: Dict[str, Any]) -> Optional[float]:
    if isinstance(row.get("legacy_quality_scores"), dict):
        return _to_float(row["legacy_quality_scores"].get("sharpness"), 0.0)
    if isinstance(row.get("raw_metrics"), dict):
        return _to_float(row["raw_metrics"].get("laplacian_var"), 0.0)
    lens_vals: List[float] = []
    for key in ("lens_a_raw", "lens_b_raw"):
        lens_raw = row.get(key)
        if isinstance(lens_raw, dict):
            lens_vals.append(_to_float(lens_raw.get("laplacian_var"), 0.0))
    if lens_vals:
        return max(lens_vals)
    if "sharpness" in row:
        return _to_float(row.get("sharpness"), 0.0)
    return None


def _stage2_combined_stage2(row: Dict[str, Any]) -> Optional[float]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        if "combined_stage2" in metrics:
            return _to_float(metrics.get("combined_stage2"), 0.0)
        if "combined_score" in metrics:
            return _to_float(metrics.get("combined_score"), 0.0)
    if "combined_score" in row:
        return _to_float(row.get("combined_score"), 0.0)
    return None


def _stage2_combined_stage3(row: Dict[str, Any]) -> Optional[float]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and "combined_stage3" in metrics:
        return _to_float(metrics.get("combined_stage3"), 0.0)
    if "combined_score" in row:
        return _to_float(row.get("combined_score"), 0.0)
    return None


def _counter_ratio(counter: Counter, key: str, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return float(counter.get(key, 0) / float(denom))


def summarize_stage1(stage1_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(stage1_rows)
    if not rows:
        return {
            "records_count": 0,
            "quality_series": [],
            "pass_fail": {"pass": 0, "fail": 0},
            "drop_reason_counts": {},
            "pass_rate": None,
            "quality_p10": None,
            "quality_p90": None,
            "abs_laplacian_guard_ratio": None,
        }

    quality_series: List[Dict[str, Any]] = []
    quality_vals: List[float] = []
    drop_counter: Counter = Counter()
    pass_count = 0

    for row in rows:
        frame_idx = _to_int(row.get("frame_index"), -1)
        quality = row.get("quality")
        if quality is None and isinstance(row.get("quality_scores"), dict):
            quality = row["quality_scores"].get("quality")
        q = _to_float(quality, 0.0)
        quality_vals.append(q)
        pass_flag = bool(row.get("is_pass", False))
        if pass_flag:
            pass_count += 1
        reason = str(row.get("drop_reason", "unknown") or "unknown")
        drop_counter[reason] += 1
        quality_series.append(
            {
                "frame_index": frame_idx,
                "quality": q,
                "is_pass": pass_flag,
                "drop_reason": reason,
                "sharpness": _stage1_sharpness(row),
            }
        )

    total = len(rows)
    return {
        "records_count": total,
        "quality_series": sorted(quality_series, key=lambda x: int(x.get("frame_index", -1))),
        "pass_fail": {"pass": int(pass_count), "fail": int(max(0, total - pass_count))},
        "drop_reason_counts": dict(drop_counter),
        "pass_rate": float(pass_count / float(total)) if total > 0 else None,
        "quality_p10": _percentile(quality_vals, 10.0),
        "quality_p90": _percentile(quality_vals, 90.0),
        "abs_laplacian_guard_ratio": _counter_ratio(drop_counter, "abs_laplacian_guard", total),
    }


def summarize_stage2(
    stage2_rows: Iterable[Dict[str, Any]],
    stage2_preview_rows: Iterable[Dict[str, Any]],
    analysis_summary: Dict[str, Any],
) -> Dict[str, Any]:
    rows = list(stage2_rows)
    preview_rows = list(stage2_preview_rows)

    drop_counter: Counter = Counter()
    combined_stage2_series: List[Dict[str, Any]] = []
    novelty_vals: List[float] = []
    read_success = 0

    for row in rows:
        frame_idx = _to_int(row.get("frame_index"), -1)
        drop_reason = str(row.get("drop_reason", "unknown") or "unknown")
        drop_counter[drop_reason] += 1
        if drop_reason != "read_fail":
            read_success += 1

        c2 = _stage2_combined_stage2(row)
        if c2 is not None:
            combined_stage2_series.append({"frame_index": frame_idx, "combined_stage2": float(c2)})

    for row in preview_rows:
        novelty_vals.append(_to_float(row.get("novelty"), 0.0))

    summary_stage_counts = analysis_summary.get("stage_counts")
    summary_colmap_runtime = analysis_summary.get("colmap_runtime")
    stage_counts = summary_stage_counts if isinstance(summary_stage_counts, dict) else {}
    colmap_runtime = summary_colmap_runtime if isinstance(summary_colmap_runtime, dict) else {}

    total = len(rows)
    read_success_rate = float(read_success / float(total)) if total > 0 else None
    stage2_preview_drop_counts = analysis_summary.get("stage2_colmap_preview_drop_reason_counts")

    return {
        "records_count": total,
        "combined_stage2_series": sorted(combined_stage2_series, key=lambda x: int(x.get("frame_index", -1))),
        "drop_reason_counts": dict(drop_counter),
        "novelty_values": novelty_vals,
        "read_success_rate": read_success_rate,
        "ssim_skip_ratio": _counter_ratio(drop_counter, "ssim_skip", total),
        "min_interval_ratio": _counter_ratio(drop_counter, "min_interval", total),
        "final_keyframes": _to_int(stage_counts.get("final_keyframes"), 0),
        "target_min": _to_int(colmap_runtime.get("target_min"), 0),
        "target_max": _to_int(colmap_runtime.get("target_max"), 0),
        "stage2_colmap_preview_count": _to_int(
            analysis_summary.get("stage2_colmap_preview_count", len(preview_rows)),
            len(preview_rows),
        ),
        "stage2_colmap_preview_max_gap": analysis_summary.get("stage2_colmap_preview_max_gap"),
        "stage2_colmap_preview_drop_reason_counts": (
            dict(stage2_preview_drop_counts) if isinstance(stage2_preview_drop_counts, dict) else {}
        ),
    }


def summarize_stage3(
    stage2_rows: Iterable[Dict[str, Any]],
    stage3_diagnostics: Dict[str, Any],
    analysis_summary: Dict[str, Any],
) -> Dict[str, Any]:
    rows = list(stage2_rows)
    c2_c3_points: List[Dict[str, Any]] = []
    traj_risk_points: List[Dict[str, Any]] = []
    vo_attempted = 0
    vo_valid = 0

    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        frame_idx = _to_int(row.get("frame_index"), -1)
        c2 = _stage2_combined_stage2(row)
        c3 = _stage2_combined_stage3(row)
        if c2 is not None and c3 is not None:
            c2_c3_points.append(
                {
                    "frame_index": frame_idx,
                    "combined_stage2": float(c2),
                    "combined_stage3": float(c3),
                }
            )

        traj_eff = metrics.get("trajectory_consistency_effective")
        risk = metrics.get("stage0_motion_risk")
        if traj_eff is not None and risk is not None:
            traj_risk_points.append(
                {
                    "frame_index": frame_idx,
                    "trajectory_consistency_effective": _to_float(traj_eff, 0.0),
                    "stage0_motion_risk": _to_float(risk, 0.0),
                }
            )

        attempted_flag = _to_float(metrics.get("vo_attempted"), 0.0)
        valid_flag = _to_float(metrics.get("vo_valid"), 0.0)
        vo_attempted += 1 if attempted_flag > 0.5 else 0
        vo_valid += 1 if valid_flag > 0.5 else 0

    summary_diag = analysis_summary.get("stage3_diagnostics")
    diagnostics = dict(summary_diag) if isinstance(summary_diag, dict) else {}
    if stage3_diagnostics:
        diagnostics.update(stage3_diagnostics)

    vo_valid_ratio = float(vo_valid / float(vo_attempted)) if vo_attempted > 0 else None

    return {
        "stage2_vs_stage3": c2_c3_points,
        "trajectory_vs_risk": traj_risk_points,
        "diagnostics": diagnostics,
        "vo_attempted": int(vo_attempted),
        "vo_valid": int(vo_valid),
        "vo_valid_ratio": vo_valid_ratio,
    }


def summarize_colmap(analysis_summary: Dict[str, Any], pose_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pose = dict(pose_summary or {})
    diagnostics = pose.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    subset = diagnostics.get("colmap_input_subset")
    input_subset = dict(subset) if isinstance(subset, dict) else {}

    sparse_selection = diagnostics.get("sparse_model_selection")
    sparse_selection_dict = dict(sparse_selection) if isinstance(sparse_selection, dict) else {}
    sparse_candidates = sparse_selection_dict.get("candidates")
    if not isinstance(sparse_candidates, list):
        sparse_candidates = []

    preview_drop_counts_raw = analysis_summary.get("stage2_colmap_preview_drop_reason_counts")
    preview_drop_counts = (
        dict(preview_drop_counts_raw) if isinstance(preview_drop_counts_raw, dict) else {}
    )
    preview_drop_total = sum(int(v) for v in preview_drop_counts.values() if isinstance(v, (int, float)))
    dense_neighbor_ratio = (
        float(_to_float(preview_drop_counts.get("dense_neighbor_low_novelty"), 0.0) / float(preview_drop_total))
        if preview_drop_total > 0
        else None
    )

    summary_colmap_runtime = analysis_summary.get("colmap_runtime")
    colmap_runtime = summary_colmap_runtime if isinstance(summary_colmap_runtime, dict) else {}

    return {
        "pose_summary": pose,
        "diagnostics": diagnostics,
        "input_subset": input_subset,
        "subset_drop_reason_counts": dict(input_subset.get("drop_reason_counts", {}))
        if isinstance(input_subset.get("drop_reason_counts"), dict)
        else {},
        "sparse_model_selection": sparse_selection_dict,
        "sparse_model_candidates": [c for c in sparse_candidates if isinstance(c, dict)],
        "trajectory_count": _to_int(pose.get("trajectory_count"), 0),
        "selected_count": _to_int(pose.get("selected_count"), 0),
        "stage2_colmap_preview_count": _to_int(analysis_summary.get("stage2_colmap_preview_count"), 0),
        "stage2_colmap_preview_max_gap": analysis_summary.get("stage2_colmap_preview_max_gap"),
        "preview_drop_reason_counts": preview_drop_counts,
        "dense_neighbor_low_novelty_ratio": dense_neighbor_ratio,
        "final_keyframes": _to_int(
            (analysis_summary.get("stage_counts") or {}).get("final_keyframes", analysis_summary.get("final_keyframes", 0)),
            0,
        ),
        "target_min": _to_int(colmap_runtime.get("target_min"), 0),
        "target_max": _to_int(colmap_runtime.get("target_max"), 0),
        "total_frames": _to_int(analysis_summary.get("total_frames"), 0),
    }


def load_analysis_artifacts(run_dir: Path, pose_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load and summarize latest run artifacts from ~/.360split/tmp_runs/<run_id>."""
    run_path = Path(run_dir)
    stage1_rows = _load_jsonl(run_path / "stage1_records.jsonl")
    stage2_rows = _load_jsonl(run_path / "stage2_records.jsonl")
    stage2_preview_rows = _load_jsonl(run_path / "stage2_colmap_preview.jsonl")
    stage3_diag = _load_json(run_path / "stage3_diagnostics.json")
    analysis_summary = _load_json(run_path / "analysis_summary.json")

    payload = {
        "run_id": run_path.name,
        "run_dir": str(run_path),
        "artifacts": {
            "stage1_records": (run_path / "stage1_records.jsonl").exists(),
            "stage2_records": (run_path / "stage2_records.jsonl").exists(),
            "stage2_colmap_preview": (run_path / "stage2_colmap_preview.jsonl").exists(),
            "stage3_diagnostics": (run_path / "stage3_diagnostics.json").exists(),
            "analysis_summary": (run_path / "analysis_summary.json").exists(),
        },
        "analysis_summary": analysis_summary,
    }
    payload["stage1"] = summarize_stage1(stage1_rows)
    payload["stage2"] = summarize_stage2(stage2_rows, stage2_preview_rows, analysis_summary)
    payload["stage3"] = summarize_stage3(stage2_rows, stage3_diag, analysis_summary)
    payload["colmap"] = summarize_colmap(analysis_summary, pose_summary)
    return payload
