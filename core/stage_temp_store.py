"""Temporary Stage I/O store for per-run pipeline artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class StageTempStore:
    """Persist stage outputs to temporary files under ~/.360split/tmp_runs/<run_id>."""

    STAGE1_CANDIDATES_FILE = "stage1_candidates.jsonl"
    STAGE1_CANDIDATES_EFFECTIVE_FILE = "stage1_candidates_effective.jsonl"
    STAGE15_CANDIDATES_FILE = "stage15_candidates.jsonl"
    STAGE1_RECORDS_FILE = "stage1_records.jsonl"
    STAGE0_METRICS_FILE = "stage0_metrics.jsonl"
    STAGE2_CANDIDATES_FILE = "stage2_candidates.jsonl"
    STAGE2_RECORDS_FILE = "stage2_records.jsonl"
    STAGE3_KEYFRAMES_FILE = "stage3_keyframes.jsonl"
    ANALYSIS_SUMMARY_FILE = "analysis_summary.json"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, run_id: str, root_dir: Optional[Path] = None):
        rid = str(run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required")
        rid = rid.replace("/", "_").replace("\\", "_")

        base = Path(root_dir) if root_dir is not None else (Path.home() / ".360split" / "tmp_runs")
        self.run_id = rid
        self.run_dir = base / rid
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / self.MANIFEST_FILE

        if not self.manifest_path.exists():
            self._manifest: Dict[str, Any] = {
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "completed_stages": [],
                "stage_files": {},
                "stage_counts": {},
                "failed": False,
                "failed_stage": None,
                "error": None,
                "cleanup_target": str(self.run_dir),
                "cleanup_status": "pending",
                "resume_enabled": False,
                "resumed": False,
                "resume_count": 0,
                "last_resume_at": None,
            }
            self._write_manifest()
        else:
            self._manifest = self._read_manifest()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data

    def _write_manifest(self) -> None:
        self._manifest["updated_at"] = self._now_iso()
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _jsonl_path(self, filename: str) -> Path:
        return self.run_dir / filename

    def _write_jsonl(self, filename: str, rows: Iterable[Dict[str, Any]]) -> Path:
        path = self._jsonl_path(filename)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=self._json_default))
                f.write("\n")
        return path

    def _read_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        path = self._jsonl_path(filename)
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                value = json.loads(text)
                if isinstance(value, dict):
                    out.append(value)
        return out

    def has_stage1(self) -> bool:
        return (self.run_dir / self.STAGE1_CANDIDATES_FILE).exists() and (self.run_dir / self.STAGE1_RECORDS_FILE).exists()

    def has_stage0(self) -> bool:
        return (self.run_dir / self.STAGE0_METRICS_FILE).exists()

    def has_stage2(self) -> bool:
        return (self.run_dir / self.STAGE2_CANDIDATES_FILE).exists() and (self.run_dir / self.STAGE2_RECORDS_FILE).exists()

    def has_stage3(self) -> bool:
        return (self.run_dir / self.STAGE3_KEYFRAMES_FILE).exists()

    def is_stage_done(self, stage: str) -> bool:
        done = list(self._manifest.get("completed_stages", []))
        return str(stage) in done

    def save_stage1(self, candidates: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, str]:
        p1 = self._write_jsonl(self.STAGE1_CANDIDATES_FILE, candidates)
        p2 = self._write_jsonl(self.STAGE1_RECORDS_FILE, records)
        return {"candidates": str(p1), "records": str(p2)}

    def save_stage1_effective(self, candidates: List[Dict[str, Any]]) -> str:
        p = self._write_jsonl(self.STAGE1_CANDIDATES_EFFECTIVE_FILE, candidates)
        return str(p)

    def save_stage15(self, candidates: List[Dict[str, Any]]) -> str:
        p = self._write_jsonl(self.STAGE15_CANDIDATES_FILE, candidates)
        return str(p)

    def load_stage1(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self._read_jsonl(self.STAGE1_CANDIDATES_FILE), self._read_jsonl(self.STAGE1_RECORDS_FILE)

    def load_stage1_effective(self) -> List[Dict[str, Any]]:
        return self._read_jsonl(self.STAGE1_CANDIDATES_EFFECTIVE_FILE)

    def load_stage15(self) -> List[Dict[str, Any]]:
        return self._read_jsonl(self.STAGE15_CANDIDATES_FILE)

    def save_stage0(self, metrics: Dict[int, Dict[str, Any]]) -> str:
        rows = []
        for idx in sorted(metrics.keys()):
            row = {"frame_index": int(idx)}
            value = metrics.get(idx, {}) or {}
            if isinstance(value, dict):
                row.update(value)
            rows.append(row)
        p = self._write_jsonl(self.STAGE0_METRICS_FILE, rows)
        return str(p)

    def load_stage0(self) -> Dict[int, Dict[str, Any]]:
        rows = self._read_jsonl(self.STAGE0_METRICS_FILE)
        out: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            idx = int(row.get("frame_index", -1))
            if idx < 0:
                continue
            data = dict(row)
            data.pop("frame_index", None)
            out[idx] = data
        return out

    def save_stage2(self, candidates: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, str]:
        p1 = self._write_jsonl(self.STAGE2_CANDIDATES_FILE, candidates)
        p2 = self._write_jsonl(self.STAGE2_RECORDS_FILE, records)
        return {"candidates": str(p1), "records": str(p2)}

    def load_stage2(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self._read_jsonl(self.STAGE2_CANDIDATES_FILE), self._read_jsonl(self.STAGE2_RECORDS_FILE)

    def save_stage3(self, keyframes: List[Dict[str, Any]]) -> str:
        p = self._write_jsonl(self.STAGE3_KEYFRAMES_FILE, keyframes)
        return str(p)

    def load_stage3(self) -> List[Dict[str, Any]]:
        return self._read_jsonl(self.STAGE3_KEYFRAMES_FILE)

    def save_analysis_summary(self, summary: Dict[str, Any]) -> str:
        path = self.run_dir / self.ANALYSIS_SUMMARY_FILE
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=self._json_default)
        self._manifest["analysis_summary_file"] = str(path)
        self._write_manifest()
        return str(path)

    def mark_stage_done(
        self,
        stage: str,
        *,
        files: Optional[Dict[str, str]] = None,
        counts: Optional[Dict[str, int]] = None,
    ) -> None:
        stage_name = str(stage)
        completed = list(self._manifest.get("completed_stages", []))
        if stage_name not in completed:
            completed.append(stage_name)
        self._manifest["completed_stages"] = completed
        if files:
            sf = dict(self._manifest.get("stage_files", {}))
            sf[stage_name] = dict(files)
            self._manifest["stage_files"] = sf
        if counts:
            sc = dict(self._manifest.get("stage_counts", {}))
            sc[stage_name] = {k: int(v) for k, v in counts.items()}
            self._manifest["stage_counts"] = sc
        self._write_manifest()

    def mark_failed(self, stage: str, error: str) -> None:
        self._manifest["failed"] = True
        self._manifest["failed_stage"] = str(stage)
        self._manifest["error"] = str(error)
        self._manifest["cleanup_status"] = "retained_on_error"
        self._write_manifest()

    def mark_retained_on_success(self) -> None:
        self._manifest["failed"] = False
        self._manifest["failed_stage"] = None
        self._manifest["error"] = None
        self._manifest["cleanup_status"] = "retained_on_success"
        self._write_manifest()

    def record_resume_state(self, *, enabled: bool) -> None:
        self._manifest["resume_enabled"] = bool(enabled)
        if bool(enabled) and (
            self.has_stage1() or self.has_stage0() or self.has_stage2() or self.has_stage3()
        ):
            self._manifest["resumed"] = True
            self._manifest["resume_count"] = int(self._manifest.get("resume_count", 0)) + 1
            self._manifest["last_resume_at"] = self._now_iso()
        self._write_manifest()

    def cleanup_on_success(self) -> None:
        self._manifest["failed"] = False
        self._manifest["failed_stage"] = None
        self._manifest["error"] = None
        self._manifest["cleanup_status"] = "removed"
        self._write_manifest()
        shutil.rmtree(self.run_dir, ignore_errors=True)


__all__ = ["StageTempStore"]
