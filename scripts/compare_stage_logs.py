#!/usr/bin/env python3
"""Compare key stage logs between two runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_SUMMARY_KEYS = ("stage_counts", "stage_plan", "retarget", "selection_runtime")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _extract_indices(rows: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for row in rows:
        raw = row.get("frame_idx", row.get("frame_index", None))
        if isinstance(raw, (int, float)):
            idx = int(raw)
            if idx >= 0:
                out.append(idx)
    return sorted(out)


def _index_stats(indices: List[int]) -> Dict[str, Any]:
    if not indices:
        return {"count": 0, "first": None, "last": None, "max_gap": None}
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    return {
        "count": int(len(indices)),
        "first": int(indices[0]),
        "last": int(indices[-1]),
        "max_gap": int(max(gaps)) if gaps else 0,
    }


def _bin_pass_ratio(indices: List[int], total_frames: int, bin_size: int = 100) -> Dict[str, float]:
    if total_frames <= 0:
        return {}
    bucket_count = max(1, (total_frames + bin_size - 1) // bin_size)
    hits = [0] * bucket_count
    totals = [0] * bucket_count
    for b in range(bucket_count):
        start = b * bin_size
        end = min(total_frames, start + bin_size)
        totals[b] = max(0, end - start)
    for idx in indices:
        if 0 <= idx < total_frames:
            hits[idx // bin_size] += 1
    out: Dict[str, float] = {}
    for b in range(bucket_count):
        if totals[b] <= 0:
            continue
        start = b * bin_size
        end = start + bin_size - 1
        out[f"{start:04d}-{end:04d}"] = float(hits[b] / totals[b])
    return out


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(len(sa | sb), 1))


def _missing_keys(summary: Dict[str, Any]) -> List[str]:
    return [k for k in REQUIRED_SUMMARY_KEYS if k not in summary]


def _lens_matrix_from_records(rows: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, int]:
    out = {"both_pass": 0, "a_only_pass": 0, "b_only_pass": 0, "neither": 0}
    for row in rows:
        qa = row.get("quality_lens_a")
        qb = row.get("quality_lens_b")
        if not isinstance(qa, (int, float)) or not isinstance(qb, (int, float)):
            continue
        pass_a = float(qa) >= threshold
        pass_b = float(qb) >= threshold
        if pass_a and pass_b:
            out["both_pass"] += 1
        elif pass_a and not pass_b:
            out["a_only_pass"] += 1
        elif pass_b and not pass_a:
            out["b_only_pass"] += 1
        else:
            out["neither"] += 1
    return out


def _summary_matrix(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, int]:
    matrix = summary.get("stage1_lens_pass_matrix")
    if isinstance(matrix, dict) and matrix:
        return {
            "both_pass": int(matrix.get("both_pass", 0)),
            "a_only_pass": int(matrix.get("a_only_pass", 0)),
            "b_only_pass": int(matrix.get("b_only_pass", 0)),
            "neither": int(matrix.get("neither", 0)),
        }
    return _lens_matrix_from_records(records, threshold=0.5)


def _print_kv(label: str, before: Any, after: Any) -> None:
    print(f"{label}: before={before}  after={after}")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compare Stage logs from two runs")
    parser.add_argument("--before", required=True, help="previous run log dir")
    parser.add_argument("--after", required=True, help="new run log dir")
    args = parser.parse_args()

    before_dir = Path(args.before).expanduser().resolve()
    after_dir = Path(args.after).expanduser().resolve()

    before_summary = _load_json(before_dir / "analysis_summary.json")
    after_summary = _load_json(after_dir / "analysis_summary.json")
    before_s1_records = _load_jsonl(before_dir / "stage1_records.jsonl")
    after_s1_records = _load_jsonl(after_dir / "stage1_records.jsonl")
    before_s1 = _extract_indices(_load_jsonl(before_dir / "stage1_candidates.jsonl"))
    after_s1 = _extract_indices(_load_jsonl(after_dir / "stage1_candidates.jsonl"))
    before_s3 = _extract_indices(_load_jsonl(before_dir / "stage3_keyframes.jsonl"))
    after_s3 = _extract_indices(_load_jsonl(after_dir / "stage3_keyframes.jsonl"))

    before_stage1_stats = _index_stats(before_s1)
    after_stage1_stats = _index_stats(after_s1)
    before_stage3_stats = _index_stats(before_s3)
    after_stage3_stats = _index_stats(after_s3)

    before_matrix = _summary_matrix(before_summary, before_s1_records)
    after_matrix = _summary_matrix(after_summary, after_s1_records)

    total_before = int(before_summary.get("total_frames", 0) or 0)
    total_after = int(after_summary.get("total_frames", 0) or 0)
    before_bins = _bin_pass_ratio(before_s1, total_before, bin_size=100) if total_before > 0 else {}
    after_bins = _bin_pass_ratio(after_s1, total_after, bin_size=100) if total_after > 0 else {}

    print("== Stage1 Count/Ratio ==")
    before_ratio = float(before_stage1_stats["count"] / max(total_before, 1)) if total_before > 0 else None
    after_ratio = float(after_stage1_stats["count"] / max(total_after, 1)) if total_after > 0 else None
    _print_kv("stage1_count", before_stage1_stats["count"], after_stage1_stats["count"])
    _print_kv("stage1_ratio", before_ratio, after_ratio)

    print("\n== Stage1 Lens Pass Matrix ==")
    for k in ("both_pass", "a_only_pass", "b_only_pass", "neither"):
        _print_kv(k, before_matrix.get(k, 0), after_matrix.get(k, 0))

    print("\n== Stage1 Distribution ==")
    _print_kv("first_frame", before_stage1_stats["first"], after_stage1_stats["first"])
    _print_kv("last_frame", before_stage1_stats["last"], after_stage1_stats["last"])
    _print_kv("max_gap", before_stage1_stats["max_gap"], after_stage1_stats["max_gap"])

    print("\n== Stage1 100-Frame Bin Pass Ratio (after - before) ==")
    all_bins = sorted(set(before_bins.keys()) | set(after_bins.keys()))
    for key in all_bins:
        b = float(before_bins.get(key, 0.0))
        a = float(after_bins.get(key, 0.0))
        print(f"{key}: before={b:.4f}  after={a:.4f}  delta={a - b:+.4f}")

    print("\n== Candidate Overlap ==")
    print(f"stage1_jaccard={_jaccard(before_s1, after_s1):.4f}")
    print(f"stage3_jaccard={_jaccard(before_s3, after_s3):.4f}")

    before_only_s1 = sorted(set(before_s1) - set(after_s1))
    after_only_s1 = sorted(set(after_s1) - set(before_s1))
    print(f"stage1_only_before_sample={before_only_s1[:20]}")
    print(f"stage1_only_after_sample={after_only_s1[:20]}")

    print("\n== Stage3 Distribution ==")
    _print_kv("stage3_count", before_stage3_stats["count"], after_stage3_stats["count"])
    _print_kv("stage3_first", before_stage3_stats["first"], after_stage3_stats["first"])
    _print_kv("stage3_last", before_stage3_stats["last"], after_stage3_stats["last"])
    _print_kv("stage3_max_gap", before_stage3_stats["max_gap"], after_stage3_stats["max_gap"])

    print("\n== Summary Key Check ==")
    missing_before = _missing_keys(before_summary)
    missing_after = _missing_keys(after_summary)
    print(f"missing_before={missing_before}")
    print(f"missing_after={missing_after}")

    cond_stage1_count = after_stage1_stats["count"] >= before_stage1_stats["count"]
    cond_stage1_range = (
        before_stage1_stats["first"] is not None
        and before_stage1_stats["last"] is not None
        and after_stage1_stats["first"] is not None
        and after_stage1_stats["last"] is not None
        and int(after_stage1_stats["first"]) < int(before_stage1_stats["first"])
        and int(after_stage1_stats["last"]) > int(before_stage1_stats["last"])
    )
    cond_summary_keys = len(missing_after) == 0

    print("\n== Acceptance ==")
    print(f"[{'PASS' if cond_stage1_count else 'FAIL'}] stage1_count_not_decrease")
    print(f"[{'PASS' if cond_stage1_range else 'FAIL'}] stage1_range_expanded")
    print(f"[{'PASS' if cond_summary_keys else 'FAIL'}] summary_required_keys")

    overall = bool(cond_stage1_count and cond_summary_keys)
    print(f"\nOVERALL={'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(_main())

