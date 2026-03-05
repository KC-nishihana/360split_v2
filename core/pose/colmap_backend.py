from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import PoseEstimator
from .types import PoseEstimationResult, PoseRecord

_FRAME_PAT = re.compile(r"keyframe_(\d+)")


def _parse_frame_idx(name: str) -> int:
    m = _FRAME_PAT.search(name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / n
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(r: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(r, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    q /= n
    if q[0] < 0.0:
        q *= -1.0
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def opk_deg_to_rotmat(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    om = float(np.deg2rad(omega_deg))
    ph = float(np.deg2rad(phi_deg))
    ka = float(np.deg2rad(kappa_deg))
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(om), -np.sin(om)],
            [0.0, np.sin(om), np.cos(om)],
        ],
        dtype=np.float64,
    )
    ry = np.asarray(
        [
            [np.cos(ph), 0.0, np.sin(ph)],
            [0.0, 1.0, 0.0],
            [-np.sin(ph), 0.0, np.cos(ph)],
        ],
        dtype=np.float64,
    )
    rz = np.asarray(
        [
            [np.cos(ka), -np.sin(ka), 0.0],
            [np.sin(ka), np.cos(ka), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Metashape OPK convention seed: R = Rz(kappa) * Ry(phi) * Rx(omega)
    return rz @ ry @ rx


def parse_colmap_images_txt(path: Path) -> List[PoseRecord]:
    poses: List[PoseRecord] = []
    if not path.exists():
        return poses

    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        points_line = ""
        if i < len(lines):
            points_line = lines[i].strip()
            i += 1

        try:
            qw, qx, qy, qz = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            tx, ty, tz = (float(parts[5]), float(parts[6]), float(parts[7]))
            name = str(parts[9])
        except Exception:
            continue

        observations = 0
        if points_line and not points_line.startswith("#"):
            pts = points_line.split()
            for k in range(2, len(pts), 3):
                try:
                    if int(float(pts[k])) != -1:
                        observations += 1
                except Exception:
                    continue

        # COLMAP stores world-to-camera: Xc = Rcw * Xw + tcw.
        # Convert to camera center in world coordinates.
        r_cw = quat_to_rotmat(qw, qx, qy, qz)
        t_cw = np.asarray([tx, ty, tz], dtype=np.float64)
        r_wc = r_cw.T
        c_w = -r_wc @ t_cw
        q_wc = rotmat_to_quat(r_wc)

        poses.append(
            PoseRecord(
                frame_index=_parse_frame_idx(name),
                filename=name,
                qw=float(q_wc[0]),
                qx=float(q_wc[1]),
                qy=float(q_wc[2]),
                qz=float(q_wc[3]),
                tx=float(c_w[0]),
                ty=float(c_w[1]),
                tz=float(c_w[2]),
                confidence=1.0,
                observations=int(observations),
            )
        )

    poses.sort(key=lambda p: (int(p.frame_index), str(p.filename)))
    return poses


class COLMAPPoseEstimator(PoseEstimator):
    def __init__(
        self,
        colmap_path: str = "colmap",
        workspace: Optional[str] = None,
        db_path: Optional[str] = None,
        clear_db: bool = False,
    ):
        self.colmap_path = str(colmap_path or "colmap")
        self.workspace = workspace
        self.db_path = db_path
        self.clear_db = bool(clear_db)

    def _resolve_colmap_binary(self) -> str:
        requested = str(self.colmap_path or "colmap").strip() or "colmap"
        p = Path(requested)
        candidates: List[str] = []
        if p.is_absolute() or os.sep in requested:
            candidates.append(str(p))
        else:
            found = shutil.which(requested)
            if found:
                candidates.append(found)
            if sys.platform == "win32":
                exe = requested if requested.lower().endswith(".exe") else f"{requested}.exe"
                candidates.extend(
                    [
                        str(Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "COLMAP" / exe),
                        str(Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Public")) / "COLMAP" / exe),
                    ]
                )
            else:
                candidates.extend(
                    [
                        f"/opt/homebrew/bin/{requested}",
                        f"/usr/local/bin/{requested}",
                    ]
                )

        seen = set()
        deduped: List[str] = []
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            deduped.append(c)

        if not deduped:
            if sys.platform == "win32":
                install_hint = "Install COLMAP from https://colmap.github.io/install.html or pass --colmap-path."
            elif sys.platform == "darwin":
                install_hint = "Install with `brew install colmap` or pass --colmap-path."
            else:
                install_hint = "Install with `sudo apt install colmap` or pass --colmap-path."
            raise RuntimeError(f"COLMAP is not installed or not in PATH. {install_hint}")

        errors: List[str] = []
        for c in deduped:
            cp = Path(c)
            if not cp.exists():
                errors.append(f"{c}: not found")
                continue
            try:
                proc = subprocess.run(
                    [c, "-h"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                if int(proc.returncode) == 0:
                    return c
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                combined_err = " ".join(detail).lower()
                if "library not loaded" in combined_err or "dyld" in combined_err:
                    raise RuntimeError(
                        f"COLMAP cannot load a required shared library (dyld error).\n"
                        f"  Fix: brew reinstall glog && brew reinstall colmap\n"
                        f"  Detail: {detail[-1] if detail else 'see stderr'}"
                    )
                tail = detail[-1] if detail else f"returncode={proc.returncode}"
                errors.append(f"{c}: {tail}")
            except Exception as e:
                errors.append(f"{c}: {e}")
                continue

        raise RuntimeError(
            "COLMAP binary is present but not runnable. "
            + " | ".join(errors)
        )

    @staticmethod
    def _command_supports_option(colmap_bin: str, command: str, option_name: str) -> bool:
        try:
            proc = subprocess.run(
                [colmap_bin, command, "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            text = str(proc.stdout or "")
            return option_name in text
        except Exception:
            return False

    def _gpu_option_names(self, colmap_bin: str, matcher_command: str) -> Tuple[str, str]:
        # COLMAP 3.13+ uses FeatureExtraction/FeatureMatching.use_gpu,
        # while older builds may expose SiftExtraction/SiftMatching.use_gpu.
        if self._command_supports_option(colmap_bin, "feature_extractor", "--FeatureExtraction.use_gpu"):
            extract_opt = "--FeatureExtraction.use_gpu"
        else:
            extract_opt = "--SiftExtraction.use_gpu"

        if self._command_supports_option(colmap_bin, matcher_command, "--FeatureMatching.use_gpu"):
            match_opt = "--FeatureMatching.use_gpu"
        else:
            match_opt = "--SiftMatching.use_gpu"
        return extract_opt, match_opt

    @staticmethod
    def _camera_params_from_runtime(runtime: Dict[str, Any]) -> Optional[str]:
        try:
            c = runtime.get("mono") or runtime.get("front") or runtime.get("rear")
            if not isinstance(c, dict):
                return None
            k = np.asarray(c.get("camera_matrix", []), dtype=np.float64).reshape(3, 3)
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
            if fx <= 0.0 or fy <= 0.0:
                return None
            # OPENCV expects fx,fy,cx,cy,k1,k2,p1,p2
            return f"{fx:.8f},{fy:.8f},{cx:.8f},{cy:.8f},0,0,0,0"
        except Exception:
            return None

    def _run_cmd(
        self,
        cmd: List[str],
        cwd: Path,
        step_name: str,
        log_file,
        on_log: Optional[Callable[[str], None]],
    ) -> Dict[str, Any]:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_error = ""
        captured: List[str] = []

        assert process.stdout is not None
        for line in process.stdout:
            msg = line.rstrip("\n")
            captured.append(msg)
            log_file.write(f"[{step_name}] {msg}\n")
            if on_log:
                on_log(f"[{step_name}] {msg}")
            low = msg.lower()
            if "error" in low or "failed" in low or "library not loaded" in low or "dyld" in low:
                last_error = msg

        ret = process.wait()
        return {
            "returncode": int(ret),
            "last_error_line": last_error,
            "lines": captured,
            "command": " ".join(cmd),
        }

    @staticmethod
    def _normalize_db_path(db_raw: Any, workspace: Path) -> Path:
        db_text = str(db_raw).strip() if db_raw is not None else ""
        if not db_text:
            return workspace / "database.db"
        db_path = Path(db_text)
        looks_like_dir = (
            db_text.endswith(os.sep)
            or db_text.endswith("/")
            or db_text.endswith("\\")
            or db_path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}
        )
        if db_path.exists() and db_path.is_dir():
            looks_like_dir = True
        if looks_like_dir:
            db_path = db_path / "database.db"
        return db_path

    def _resolve_workspace(self, image_root: Path, ctx: Dict[str, Any]) -> Tuple[Path, str]:
        workspace_raw = ctx.get("colmap_workspace", self.workspace)
        workspace_text = str(workspace_raw).strip() if workspace_raw is not None else ""
        base_workspace = Path(workspace_text) if workspace_text else (image_root.parent / "pose_colmap")
        scope = str(ctx.get("colmap_workspace_scope", "run_scoped") or "run_scoped").strip().lower()
        if scope not in {"shared", "run_scoped"}:
            scope = "run_scoped"
        run_id = str(ctx.get("analysis_run_id", "") or "").strip()
        if scope == "run_scoped":
            if not run_id:
                run_id = str(uuid.uuid4())
            workspace = base_workspace / run_id
        else:
            workspace = base_workspace
        return workspace, scope

    @staticmethod
    def _normalize_rig_name(name: str, sensor: str) -> str:
        sensor_upper = sensor.upper()
        p = Path(name)
        stem = p.stem
        if stem.endswith(f"_{sensor_upper}"):
            stem = stem[: -(len(sensor_upper) + 1)]
        return f"{stem}{p.suffix}"

    @staticmethod
    def _canonicalize_pose_filename(
        filename: str,
        *,
        image_root: Path,
        image_path_for_colmap: Path,
    ) -> str:
        raw = str(filename or "").replace("\\", "/").strip()
        if not raw:
            return raw
        root_candidates = [image_path_for_colmap, image_root]
        path_candidates: List[Path] = []
        p = Path(raw)
        if p.is_absolute():
            path_candidates.append(p)
        else:
            for base in root_candidates:
                path_candidates.append(base / p)

        for cand in path_candidates:
            try:
                resolved = cand.resolve()
            except Exception:
                continue
            for base in root_candidates:
                try:
                    rel = resolved.relative_to(base.resolve())
                except Exception:
                    continue
                rel_text = str(rel).replace("\\", "/")
                if rel_text and not rel_text.startswith("../"):
                    return rel_text

        parts = [x for x in raw.split("/") if x and x not in {".", ".."}]
        for sensor in ("L", "R", "F"):
            if sensor in parts:
                idx = parts.index(sensor)
                if idx < len(parts) - 1:
                    return "/".join(parts[idx:])
        if "images" in parts:
            idx = parts.index("images")
            if idx < len(parts) - 1:
                return "/".join(parts[idx + 1 :])
        return Path(raw).name

    def _prepare_rig_image_root(
        self,
        image_root: Path,
        workspace: Path,
    ) -> Tuple[Path, Dict[str, str], Dict[str, Any]]:
        left_dir = image_root / "L"
        right_dir = image_root / "R"
        if not (left_dir.exists() and right_dir.exists() and left_dir.is_dir() and right_dir.is_dir()):
            return image_root, {}, {"enabled": False, "reason": "no_lr_folders"}

        left_files = [p for p in sorted(left_dir.glob("*")) if p.is_file()]
        right_files = [p for p in sorted(right_dir.glob("*")) if p.is_file()]
        left_map = {self._normalize_rig_name(p.name, "L"): p for p in left_files}
        right_map = {self._normalize_rig_name(p.name, "R"): p for p in right_files}
        common = sorted(set(left_map.keys()) & set(right_map.keys()))
        if not common:
            return image_root, {}, {"enabled": False, "reason": "no_lr_pairs"}

        rig_root = workspace / "rig_images"
        rig_left = rig_root / "L"
        rig_right = rig_root / "R"
        if rig_root.exists():
            shutil.rmtree(rig_root, ignore_errors=True)
        rig_left.mkdir(parents=True, exist_ok=True)
        rig_right.mkdir(parents=True, exist_ok=True)

        name_map: Dict[str, str] = {}
        copied = 0
        for norm_name in common:
            src_l = left_map[norm_name]
            src_r = right_map[norm_name]
            dst_l = rig_left / norm_name
            dst_r = rig_right / norm_name
            try:
                shutil.copy2(src_l, dst_l)
                shutil.copy2(src_r, dst_r)
            except Exception:
                continue
            name_map[f"L/{norm_name}"] = f"L/{src_l.name}"
            name_map[f"R/{norm_name}"] = f"R/{src_r.name}"
            copied += 1

        if copied <= 0:
            return image_root, {}, {"enabled": False, "reason": "pair_copy_failed"}
        return rig_root, name_map, {"enabled": True, "pairs": int(copied)}

    @staticmethod
    def _find_latest_sparse_model_dir(sparse_dir: Path) -> Optional[Path]:
        numeric_dirs: List[Tuple[int, Path]] = []
        for p in sparse_dir.glob("*"):
            if not p.is_dir():
                continue
            try:
                numeric_dirs.append((int(p.name), p))
            except ValueError:
                continue
        if not numeric_dirs:
            return None
        numeric_dirs.sort(key=lambda x: x[0])
        return numeric_dirs[-1][1]

    @staticmethod
    def _sensor_from_filename(filename: str) -> str:
        rel = str(filename or "").replace("\\", "/").strip("/")
        if not rel:
            return ""
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0].upper() in {"L", "R", "F"}:
            return parts[0].upper()
        return ""

    def _collect_exported_entries(self, image_root: Path, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        exported_entries = list(ctx.get("exported_entries", []) or [])
        rows: List[Dict[str, Any]] = []
        if not exported_entries:
            for p in sorted(image_root.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    rel = str(p.relative_to(image_root)).replace("\\", "/")
                except Exception:
                    rel = p.name
                exported_entries.append(
                    {
                        "filename": rel,
                        "frame_index": _parse_frame_idx(rel),
                        "abs_path": str(p),
                    }
                )

        for entry in exported_entries:
            if not isinstance(entry, dict):
                continue
            rel_name = str(entry.get("filename", "") or "").replace("\\", "/").strip()
            if not rel_name:
                continue
            raw_idx = entry.get("frame_index", None)
            if isinstance(raw_idx, (int, float)):
                frame_index = int(raw_idx)
            else:
                frame_index = int(_parse_frame_idx(rel_name))
            if frame_index < 0:
                continue
            abs_path_raw = str(entry.get("abs_path", "") or "").strip()
            abs_path = Path(abs_path_raw) if abs_path_raw else (image_root / rel_name)
            try:
                abs_path = abs_path.resolve()
            except Exception:
                abs_path = (image_root / rel_name).resolve()
            if not abs_path.exists() or not abs_path.is_file():
                continue
            rows.append(
                {
                    "filename": rel_name,
                    "frame_index": int(frame_index),
                    "abs_path": abs_path,
                    "sensor": self._sensor_from_filename(rel_name),
                }
            )
        rows.sort(key=lambda x: (int(x["frame_index"]), str(x["filename"])))
        return rows

    @staticmethod
    def _gate_thresholds(strength: str) -> Dict[str, float]:
        s = str(strength or "medium").strip().lower()
        if s == "weak":
            return {
                "min_good_matches": 60.0,
                "min_f_inliers": 24.0,
                "h_over_f_max": 1.60,
                "min_median_displacement_px": 4.5,
            }
        if s == "strong":
            return {
                "min_good_matches": 100.0,
                "min_f_inliers": 45.0,
                "h_over_f_max": 1.20,
                "min_median_displacement_px": 7.5,
            }
        return {
            "min_good_matches": 80.0,
            "min_f_inliers": 35.0,
            "h_over_f_max": 1.35,
            "min_median_displacement_px": 6.0,
        }

    @staticmethod
    def _relax_gate_thresholds(base: Dict[str, float], rounds: int) -> Dict[str, float]:
        relax_rounds = max(0, int(rounds))
        out = dict(base)
        for _ in range(relax_rounds):
            out["min_good_matches"] = max(12.0, float(out["min_good_matches"]) * 0.8)
            out["min_f_inliers"] = max(10.0, float(out["min_f_inliers"]) * 0.8)
            out["h_over_f_max"] = float(out["h_over_f_max"]) * 1.15
        return out

    @staticmethod
    def _safe_imread_gray(path: Path, cache: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
        key = str(path)
        if key in cache:
            return cache[key]
        try:
            gray = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        except Exception:
            gray = None
        cache[key] = gray
        return gray

    @staticmethod
    def _evaluate_gate_pair(
        prev_path: Path,
        curr_path: Path,
        thresholds: Dict[str, float],
        orb: Any,
        matcher: Any,
        gray_cache: Dict[str, Optional[np.ndarray]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "pass": False,
            "reason": "unknown",
            "good_matches": 0,
            "f_inliers": 0,
            "h_inliers": 0,
            "h_over_f": 0.0,
            "median_displacement_px": 0.0,
        }
        prev_gray = COLMAPPoseEstimator._safe_imread_gray(prev_path, gray_cache)
        curr_gray = COLMAPPoseEstimator._safe_imread_gray(curr_path, gray_cache)
        if prev_gray is None or curr_gray is None:
            out["reason"] = "read_fail"
            return out

        kp1, desc1 = orb.detectAndCompute(prev_gray, None)
        kp2, desc2 = orb.detectAndCompute(curr_gray, None)
        if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
            out["reason"] = "feature_fail"
            return out

        try:
            knn = matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            out["reason"] = "match_fail"
            return out

        good: List[Any] = []
        for pair in knn:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < 0.75 * n.distance:
                good.append(m)

        out["good_matches"] = int(len(good))
        if len(good) < int(round(float(thresholds["min_good_matches"]))):
            out["reason"] = "low_matches"
            return out

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        if pts1.shape[0] < 8:
            out["reason"] = "low_matches"
            return out

        fm_method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
        try:
            _F, mask_f = cv2.findFundamentalMat(
                pts1,
                pts2,
                method=fm_method,
                ransacReprojThreshold=1.5,
                confidence=0.999,
                maxIters=10000,
            )
        except cv2.error:
            mask_f = None
        if mask_f is None:
            out["reason"] = "f_estimation_fail"
            return out

        f_mask = np.asarray(mask_f).reshape(-1).astype(bool)
        f_inliers = int(np.sum(f_mask))
        out["f_inliers"] = int(f_inliers)
        if f_inliers < int(round(float(thresholds["min_f_inliers"]))):
            out["reason"] = "low_f_inliers"
            return out

        try:
            _H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        except cv2.error:
            mask_h = None
        h_inliers = int(np.sum(np.asarray(mask_h).reshape(-1).astype(bool))) if mask_h is not None else 0
        out["h_inliers"] = int(h_inliers)
        h_over_f = float(h_inliers / max(f_inliers, 1))
        out["h_over_f"] = float(h_over_f)
        if h_over_f > float(thresholds["h_over_f_max"]):
            out["reason"] = "homography_dominant"
            return out

        pts1_in = pts1[f_mask] if f_mask.shape[0] == pts1.shape[0] else pts1
        pts2_in = pts2[f_mask] if f_mask.shape[0] == pts2.shape[0] else pts2
        if pts1_in.shape[0] <= 0 or pts2_in.shape[0] <= 0:
            pts1_in = pts1
            pts2_in = pts2
        displacement = np.linalg.norm(pts2_in - pts1_in, axis=1)
        median_disp = float(np.median(displacement)) if displacement.size > 0 else 0.0
        out["median_displacement_px"] = float(median_disp)
        if median_disp < float(thresholds["min_median_displacement_px"]):
            out["reason"] = "low_parallax"
            return out

        out["pass"] = True
        out["reason"] = "pass"
        return out

    @staticmethod
    def _index_stats(indices: List[int]) -> Dict[str, Any]:
        if not indices:
            return {"first": None, "last": None, "max_gap": None}
        gaps = [b - a for a, b in zip(indices, indices[1:])]
        return {
            "first": int(indices[0]),
            "last": int(indices[-1]),
            "max_gap": int(max(gaps)) if gaps else 0,
        }

    def _run_subset_gate(
        self,
        frame_rows: List[Dict[str, Any]],
        *,
        paired_mode: bool,
        thresholds: Dict[str, float],
        max_gap_rescue_frames: int,
    ) -> Dict[str, Any]:
        if not frame_rows:
            return {
                "kept_indices": [],
                "drop_reason_counts": {},
                "rescue_kept_count": 0,
            }

        orb = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        gray_cache: Dict[str, Optional[np.ndarray]] = {}
        keep_indices: List[int] = [0]
        drop_reason_counts: Dict[str, int] = {}
        rescue_kept = 0

        def _drop(reason: str) -> None:
            drop_reason_counts[reason] = int(drop_reason_counts.get(reason, 0) + 1)

        for cur_idx in range(1, len(frame_rows)):
            last_idx = keep_indices[-1]
            prev_row = frame_rows[last_idx]
            curr_row = frame_rows[cur_idx]
            frame_gap = int(curr_row["frame_index"]) - int(prev_row["frame_index"])
            if frame_gap >= int(max_gap_rescue_frames):
                keep_indices.append(cur_idx)
                rescue_kept += 1
                continue

            if paired_mode:
                lens_results: Dict[str, Dict[str, Any]] = {}
                for lens in ("L", "R"):
                    prev_entry = prev_row["by_sensor"].get(lens)
                    curr_entry = curr_row["by_sensor"].get(lens)
                    if prev_entry is None or curr_entry is None:
                        lens_results[lens] = {"pass": False, "reason": "missing_lens"}
                        continue
                    lens_results[lens] = self._evaluate_gate_pair(
                        prev_entry["abs_path"],
                        curr_entry["abs_path"],
                        thresholds,
                        orb,
                        matcher,
                        gray_cache,
                    )
                if lens_results.get("L", {}).get("pass") or lens_results.get("R", {}).get("pass"):
                    keep_indices.append(cur_idx)
                    continue
                fail_reasons = [str(v.get("reason", "unknown")) for v in lens_results.values()]
                if "homography_dominant" in fail_reasons:
                    _drop("both_lens_homography_dominant")
                elif "low_parallax" in fail_reasons:
                    _drop("both_lens_low_parallax")
                elif "low_f_inliers" in fail_reasons or "f_estimation_fail" in fail_reasons:
                    _drop("both_lens_low_f_inliers")
                elif "low_matches" in fail_reasons or "feature_fail" in fail_reasons:
                    _drop("both_lens_low_matches")
                elif "read_fail" in fail_reasons:
                    _drop("both_lens_read_fail")
                else:
                    _drop("both_lens_fail")
            else:
                prev_entry = prev_row.get("mono_entry")
                curr_entry = curr_row.get("mono_entry")
                if prev_entry is None or curr_entry is None:
                    _drop("mono_missing_entry")
                    continue
                result = self._evaluate_gate_pair(
                    prev_entry["abs_path"],
                    curr_entry["abs_path"],
                    thresholds,
                    orb,
                    matcher,
                    gray_cache,
                )
                if bool(result.get("pass", False)):
                    keep_indices.append(cur_idx)
                else:
                    _drop(str(result.get("reason", "mono_fail")))

        return {
            "kept_indices": keep_indices,
            "drop_reason_counts": drop_reason_counts,
            "rescue_kept_count": int(rescue_kept),
        }

    @staticmethod
    def _safe_link_or_copy(src: Path, dst: Path, *, force_copy: bool = False) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if force_copy:
            shutil.copy2(src, dst)
            return
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    def _prepare_colmap_input_subset(
        self,
        image_root: Path,
        workspace: Path,
        ctx: Dict[str, Any],
        on_log: Optional[Callable[[str], None]],
    ) -> Tuple[Path, Dict[str, Any]]:
        enabled = bool(ctx.get("colmap_input_subset_enabled", True))
        gate_method = str(ctx.get("colmap_input_gate_method", "homography_degeneracy_v1") or "homography_degeneracy_v1").strip().lower()
        gate_strength = str(ctx.get("colmap_input_gate_strength", "medium") or "medium").strip().lower()
        min_keep_ratio = float(max(0.0, min(1.0, ctx.get("colmap_input_min_keep_ratio", 0.20))))
        max_gap_rescue_frames = int(max(1, ctx.get("colmap_input_max_gap_rescue_frames", 150)))
        force_copy = bool(ctx.get("colmap_input_subset_force_copy", False))
        preview_seed_raw = ctx.get("colmap_preview_frame_indices", [])
        preview_seed_indices = sorted(
            {
                int(v)
                for v in (preview_seed_raw if isinstance(preview_seed_raw, (list, tuple)) else [])
                if isinstance(v, (int, float)) and int(v) >= 0
            }
        )

        diagnostics: Dict[str, Any] = {
            "enabled": bool(enabled),
            "gate_method": gate_method,
            "gate_strength": gate_strength,
            "input_count": 0,
            "kept_count": 0,
            "kept_ratio": 0.0,
            "drop_reason_counts": {},
            "first": None,
            "last": None,
            "max_gap": None,
            "auto_relaxed": False,
            "auto_relaxed_rounds": 0,
            "subset_image_path": "",
            "subset_force_copy": bool(force_copy),
            "paired_mode": False,
            "min_keep_ratio": float(min_keep_ratio),
            "max_gap_rescue_frames": int(max_gap_rescue_frames),
            "preview_seed_count": int(len(preview_seed_indices)),
            "preview_seed_coverage": self._index_stats(preview_seed_indices),
            "preview_seed_applied": False,
            "preview_seed_matched_count": 0,
            "preview_seed_matched_ratio": 0.0,
            "input_count_before_seed": 0,
        }

        if not enabled:
            diagnostics["reason"] = "disabled"
            return image_root, diagnostics
        if gate_method == "off":
            diagnostics["reason"] = "gate_off"
            return image_root, diagnostics

        exported = self._collect_exported_entries(image_root, ctx)
        grouped: Dict[int, Dict[str, Any]] = {}
        for row in exported:
            idx = int(row["frame_index"])
            bucket = grouped.setdefault(idx, {"frame_index": idx, "entries": [], "by_sensor": {}})
            bucket["entries"].append(row)
            sensor = str(row.get("sensor", "")).upper()
            if sensor in {"L", "R", "F"} and sensor not in bucket["by_sensor"]:
                bucket["by_sensor"][sensor] = row
        if not grouped:
            diagnostics["reason"] = "no_entries"
            return image_root, diagnostics

        for bucket in grouped.values():
            bucket["entries"] = sorted(bucket["entries"], key=lambda x: str(x["filename"]))
            bucket["mono_entry"] = bucket["entries"][0] if bucket["entries"] else None

        frame_rows_all = sorted(grouped.values(), key=lambda x: int(x["frame_index"]))
        paired_rows = [row for row in frame_rows_all if "L" in row["by_sensor"] and "R" in row["by_sensor"]]
        paired_mode = bool(len(paired_rows) >= 2)
        frame_rows = paired_rows if paired_mode else frame_rows_all
        diagnostics["paired_mode"] = bool(paired_mode)
        input_count_before_seed = int(len(frame_rows))
        diagnostics["input_count_before_seed"] = int(input_count_before_seed)
        diagnostics["input_count"] = int(input_count_before_seed)
        if preview_seed_indices:
            seed_set = set(preview_seed_indices)
            seeded_rows = [row for row in frame_rows if int(row["frame_index"]) in seed_set]
            diagnostics["preview_seed_matched_count"] = int(len(seeded_rows))
            diagnostics["preview_seed_matched_ratio"] = float(
                len(seeded_rows) / max(len(frame_rows), 1)
            )
            diagnostics["preview_seed_coverage"] = self._index_stats(
                [int(r["frame_index"]) for r in seeded_rows]
            )
            if len(seeded_rows) >= 2:
                frame_rows = seeded_rows
                diagnostics["preview_seed_applied"] = True
            else:
                diagnostics["preview_seed_applied"] = False
                diagnostics["preview_seed_reason"] = "insufficient_seed_match"
        diagnostics["input_count"] = int(len(frame_rows))
        if len(frame_rows) <= 1:
            diagnostics["reason"] = "insufficient_frames"
            return image_root, diagnostics

        base_thresholds = self._gate_thresholds(gate_strength)
        best_gate: Optional[Dict[str, Any]] = None
        best_thresholds: Dict[str, float] = dict(base_thresholds)
        for relax_round in range(0, 3):
            thresholds = self._relax_gate_thresholds(base_thresholds, relax_round)
            gate_out = self._run_subset_gate(
                frame_rows,
                paired_mode=paired_mode,
                thresholds=thresholds,
                max_gap_rescue_frames=max_gap_rescue_frames,
            )
            kept_count = len(gate_out["kept_indices"])
            kept_ratio = float(kept_count / max(len(frame_rows), 1))
            best_gate = gate_out
            best_thresholds = thresholds
            if kept_ratio >= min_keep_ratio:
                diagnostics["auto_relaxed"] = bool(relax_round > 0)
                diagnostics["auto_relaxed_rounds"] = int(relax_round)
                break
            if relax_round == 2:
                diagnostics["auto_relaxed"] = True
                diagnostics["auto_relaxed_rounds"] = int(relax_round)

        if best_gate is None:
            diagnostics["reason"] = "gate_failed"
            return image_root, diagnostics

        selected_rows = [frame_rows[i] for i in best_gate["kept_indices"] if 0 <= int(i) < len(frame_rows)]
        selected_frame_indices = sorted(int(row["frame_index"]) for row in selected_rows)
        diagnostics["kept_count"] = int(len(selected_frame_indices))
        diagnostics["kept_ratio"] = float(len(selected_frame_indices) / max(len(frame_rows), 1))
        diagnostics["drop_reason_counts"] = dict(best_gate.get("drop_reason_counts", {}))
        diagnostics["rescue_kept_count"] = int(best_gate.get("rescue_kept_count", 0))
        diagnostics.update(self._index_stats(selected_frame_indices))
        diagnostics["thresholds_effective"] = {
            "min_good_matches": int(round(best_thresholds["min_good_matches"])),
            "min_f_inliers": int(round(best_thresholds["min_f_inliers"])),
            "h_over_f_max": float(best_thresholds["h_over_f_max"]),
            "min_median_displacement_px": float(best_thresholds["min_median_displacement_px"]),
        }

        if len(selected_rows) <= 0:
            diagnostics["reason"] = "empty_after_gate"
            return image_root, diagnostics

        subset_root = workspace / "colmap_input_subset"
        if subset_root.exists():
            shutil.rmtree(subset_root, ignore_errors=True)
        subset_root.mkdir(parents=True, exist_ok=True)

        copied = 0
        for row in selected_rows:
            if paired_mode:
                entries = [row["by_sensor"].get("L"), row["by_sensor"].get("R")]
                entries = [e for e in entries if e is not None]
            else:
                entries = list(row["entries"])
            for entry in entries:
                rel = str(entry["filename"]).replace("\\", "/")
                dst = subset_root / rel
                self._safe_link_or_copy(Path(entry["abs_path"]), dst, force_copy=force_copy)
                copied += 1

        if copied <= 0:
            diagnostics["reason"] = "subset_write_failed"
            return image_root, diagnostics

        diagnostics["subset_image_path"] = str(subset_root)
        diagnostics["subset_file_count"] = int(copied)
        if on_log:
            on_log(
                "colmap_input_subset, "
                f"enabled={diagnostics['enabled']}, "
                f"paired={diagnostics['paired_mode']}, "
                f"input={diagnostics['input_count']}, kept={diagnostics['kept_count']}, "
                f"kept_ratio={diagnostics['kept_ratio']:.3f}, "
                f"preview_seed_count={diagnostics['preview_seed_count']}, "
                f"preview_seed_applied={diagnostics['preview_seed_applied']}, "
                f"first={diagnostics['first']}, last={diagnostics['last']}, "
                f"max_gap={diagnostics['max_gap']}, "
                f"auto_relaxed={diagnostics['auto_relaxed']}, rounds={diagnostics['auto_relaxed_rounds']}"
            )
        return subset_root, diagnostics

    def _evaluate_sparse_model_candidate(
        self,
        *,
        colmap_bin: str,
        workspace: Path,
        model_dir: Path,
        model_id: int,
        log_file,
        on_log: Optional[Callable[[str], None]],
    ) -> Optional[Dict[str, Any]]:
        eval_txt_dir = workspace / "_sparse_eval_txt" / str(model_id)
        if eval_txt_dir.exists():
            shutil.rmtree(eval_txt_dir, ignore_errors=True)
        eval_txt_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(eval_txt_dir),
            "--output_type",
            "TXT",
        ]
        info = self._run_cmd(cmd, workspace, f"model_converter_eval_{model_id}", log_file, on_log)
        if int(info.get("returncode", 1)) != 0:
            return None

        poses = parse_colmap_images_txt(eval_txt_dir / "images.txt")
        frame_indices = sorted({int(p.frame_index) for p in poses if int(p.frame_index) >= 0})
        if frame_indices:
            first = int(frame_indices[0])
            last = int(frame_indices[-1])
            gaps = [b - a for a, b in zip(frame_indices, frame_indices[1:])]
            max_gap = int(max(gaps)) if gaps else 0
            span = int(max(0, last - first))
        else:
            first = None
            last = None
            max_gap = None
            span = 0
        return {
            "model_id": int(model_id),
            "model_dir": str(model_dir),
            "registered_count": int(len(poses)),
            "first_frame": first,
            "last_frame": last,
            "coverage_span": int(span),
            "max_gap": max_gap,
        }

    def _select_sparse_model(
        self,
        *,
        sparse_dir: Path,
        workspace: Path,
        colmap_bin: str,
        policy: str,
        log_file,
        on_log: Optional[Callable[[str], None]],
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        candidates: List[Tuple[int, Path]] = []
        for p in sparse_dir.glob("*"):
            if not p.is_dir():
                continue
            try:
                candidates.append((int(p.name), p))
            except ValueError:
                continue
        if not candidates:
            return None, {"policy": policy, "candidates": [], "selected_model": None}
        candidates.sort(key=lambda x: x[0])

        if policy == "latest_legacy":
            picked = candidates[-1]
            return picked[1], {
                "policy": policy,
                "candidates": [{"model_id": int(picked[0]), "model_dir": str(picked[1])}],
                "selected_model": {"model_id": int(picked[0]), "model_dir": str(picked[1])},
            }

        scored: List[Dict[str, Any]] = []
        for model_id, model_dir in candidates:
            row = self._evaluate_sparse_model_candidate(
                colmap_bin=colmap_bin,
                workspace=workspace,
                model_dir=model_dir,
                model_id=model_id,
                log_file=log_file,
                on_log=on_log,
            )
            if row is not None:
                scored.append(row)

        if not scored:
            picked = candidates[-1]
            return picked[1], {
                "policy": policy,
                "candidates": [],
                "selected_model": {"model_id": int(picked[0]), "model_dir": str(picked[1]), "fallback": True},
            }

        def _score_key(row: Dict[str, Any]) -> Tuple[int, int, int, int]:
            registered = int(row.get("registered_count", 0))
            span = int(row.get("coverage_span", 0))
            max_gap = int(row.get("max_gap", 10**9) if row.get("max_gap", None) is not None else 10**9)
            model_id = int(row.get("model_id", -1))
            if policy == "coverage_then_registered":
                return (span, registered, -max_gap, model_id)
            return (registered, span, -max_gap, model_id)

        chosen = max(scored, key=_score_key)
        selected_path = Path(str(chosen["model_dir"]))
        return selected_path, {
            "policy": policy,
            "candidates": sorted(scored, key=lambda x: int(x.get("model_id", -1))),
            "selected_model": dict(chosen),
        }

    def _write_lr_opk_rig_config(
        self,
        workspace: Path,
        seed_opk_deg: List[float],
    ) -> Path:
        r = opk_deg_to_rotmat(float(seed_opk_deg[0]), float(seed_opk_deg[1]), float(seed_opk_deg[2]))
        q = rotmat_to_quat(r)
        payload = [
            {
                "cameras": [
                    {"image_prefix": "L/", "ref_sensor": True},
                    {
                        "image_prefix": "R/",
                        "cam_from_rig_rotation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                        "cam_from_rig_translation": [0.0, 0.0, 0.0],
                    },
                ]
            }
        ]
        rig_config_path = workspace / "rig_config.json"
        with rig_config_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return rig_config_path

    def estimate(self, image_dir: str, context: Optional[Dict[str, Any]] = None) -> PoseEstimationResult:
        ctx = dict(context or {})
        on_log = ctx.get("log_callback")

        colmap_bin = self._resolve_colmap_binary()
        image_root = Path(image_dir).resolve()
        if not image_root.exists():
            raise RuntimeError(f"Image directory not found: {image_root}")

        workspace, workspace_scope = self._resolve_workspace(image_root, ctx)
        workspace.mkdir(parents=True, exist_ok=True)

        db_path = self._normalize_db_path(ctx.get("colmap_db_path", self.db_path), workspace)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        reuse_db = bool(ctx.get("colmap_reuse_db", not self.clear_db))
        clear_db = (not reuse_db) or bool(ctx.get("colmap_clear_db", False)) or self.clear_db

        if clear_db:
            if db_path.exists():
                db_path.unlink(missing_ok=True)

        sparse_dir = workspace / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        log_path = workspace / "colmap_pose.log"
        image_path_for_colmap = image_root
        rig_policy = str(ctx.get("colmap_rig_policy", "lr_opk") or "lr_opk").strip().lower()
        if rig_policy not in {"off", "lr_opk"}:
            rig_policy = "off"
        rig_seed_raw = ctx.get("colmap_rig_seed_opk_deg", [0.0, 0.0, 180.0])
        if isinstance(rig_seed_raw, str):
            rig_seed_raw = [v.strip() for v in rig_seed_raw.split(",") if v.strip()]
        if not isinstance(rig_seed_raw, (list, tuple)) or len(rig_seed_raw) != 3:
            rig_seed_raw = [0.0, 0.0, 180.0]
        try:
            rig_seed_opk = [float(rig_seed_raw[0]), float(rig_seed_raw[1]), float(rig_seed_raw[2])]
        except (TypeError, ValueError):
            rig_seed_opk = [0.0, 0.0, 180.0]
        filename_map: Dict[str, str] = {}
        rig_info: Dict[str, Any] = {"enabled": False, "policy": rig_policy}
        if rig_policy == "lr_opk":
            image_path_for_colmap, filename_map, rig_info = self._prepare_rig_image_root(image_root, workspace)

        sparse_pick_policy = str(
            ctx.get("colmap_sparse_model_pick_policy", "registered_then_coverage") or "registered_then_coverage"
        ).strip().lower()
        if sparse_pick_policy not in {"registered_then_coverage", "coverage_then_registered", "latest_legacy"}:
            sparse_pick_policy = "registered_then_coverage"

        subset_ctx = dict(ctx)
        if bool(rig_info.get("enabled", False)):
            # Rig images are normalized copies; exported_entries from pre-rig names should not be reused.
            subset_ctx["exported_entries"] = []
            # Keep L/R relative paths stable for rig_configurator image_prefix matching.
            subset_ctx["colmap_input_subset_force_copy"] = True
        subset_image_path, subset_diagnostics = self._prepare_colmap_input_subset(
            image_path_for_colmap,
            workspace,
            subset_ctx,
            on_log,
        )
        image_path_for_colmap = subset_image_path

        diagnostics: Dict[str, Any] = {
            "workspace": str(workspace),
            "workspace_scope": workspace_scope,
            "database_path": str(db_path),
            "reuse_db": bool(reuse_db),
            "image_path": str(image_path_for_colmap),
            "steps": [],
            "last_error_line": "",
            "log_path": str(log_path),
            "rig": dict(rig_info),
            "rig_seed_opk_deg": list(rig_seed_opk),
            "colmap_input_subset": dict(subset_diagnostics),
            "sparse_model_selection": {
                "policy": sparse_pick_policy,
                "candidates": [],
                "selected_model": None,
            },
        }

        camera_params = self._camera_params_from_runtime(dict(ctx.get("calibration_runtime", {}) or {}))

        with log_path.open("a", encoding="utf-8") as lf:
            matcher_cmd = "sequential_matcher"
            extract_gpu_opt, match_gpu_opt = self._gpu_option_names(colmap_bin, matcher_cmd)
            cmd1 = [
                colmap_bin,
                "feature_extractor",
                "--database_path",
                str(db_path),
                "--image_path",
                str(image_path_for_colmap),
                "--ImageReader.camera_model",
                "OPENCV",
                extract_gpu_opt,
                "0",
            ]
            if self._command_supports_option(colmap_bin, "feature_extractor", "--ImageReader.single_camera_per_folder"):
                cmd1.extend(["--ImageReader.single_camera_per_folder", "1"])
            else:
                cmd1.extend(["--ImageReader.single_camera", "1"])
            if camera_params:
                cmd1.extend(["--ImageReader.camera_params", camera_params])

            cmd2 = [colmap_bin, matcher_cmd, "--database_path", str(db_path), match_gpu_opt, "0"]
            if bool(rig_info.get("enabled", False)) and self._command_supports_option(colmap_bin, matcher_cmd, "--FeatureMatching.rig_verification"):
                cmd2.extend(["--FeatureMatching.rig_verification", "1"])
            if bool(rig_info.get("enabled", False)) and self._command_supports_option(colmap_bin, matcher_cmd, "--SequentialMatching.expand_rig_images"):
                cmd2.extend(["--SequentialMatching.expand_rig_images", "1"])
            cmd3 = [
                colmap_bin,
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(image_path_for_colmap),
                "--output_path",
                str(sparse_dir),
            ]
            if self._command_supports_option(colmap_bin, "mapper", "--Mapper.ba_refine_sensor_from_rig"):
                cmd3.extend(["--Mapper.ba_refine_sensor_from_rig", "1"])

            steps: List[Tuple[str, List[str]]] = [("feature_extractor", cmd1)]
            if bool(rig_info.get("enabled", False)):
                if not self._command_supports_option(colmap_bin, "rig_configurator", "--rig_config_path"):
                    raise RuntimeError(
                        "COLMAP rig_configurator is not available in this build. "
                        "Please use COLMAP 3.12+ or set --colmap-rig-policy off."
                    )
                rig_config_path = self._write_lr_opk_rig_config(workspace, rig_seed_opk)
                diagnostics["rig_config_path"] = str(rig_config_path)
                cmd_rig = [
                    colmap_bin,
                    "rig_configurator",
                    "--database_path",
                    str(db_path),
                    "--rig_config_path",
                    str(rig_config_path),
                ]
                steps.append(("rig_configurator", cmd_rig))
            steps.extend([(matcher_cmd, cmd2), ("mapper", cmd3)])

            for name, cmd in steps:
                info = self._run_cmd(cmd, workspace, name, lf, on_log)
                diagnostics["steps"].append(info)
                if info.get("last_error_line"):
                    diagnostics["last_error_line"] = str(info.get("last_error_line"))
                if int(info.get("returncode", 1)) != 0:
                    raise RuntimeError(
                        f"COLMAP {name} failed (code={info['returncode']}). "
                        f"{diagnostics.get('last_error_line') or 'See colmap log.'}"
                    )

            selected_sparse_model, sparse_selection_info = self._select_sparse_model(
                sparse_dir=sparse_dir,
                workspace=workspace,
                colmap_bin=colmap_bin,
                policy=sparse_pick_policy,
                log_file=lf,
                on_log=on_log,
            )
            diagnostics["sparse_model_selection"] = dict(sparse_selection_info)
            if selected_sparse_model is None:
                raise RuntimeError(
                    "COLMAP mapper did not produce sparse model. "
                    f"{diagnostics.get('last_error_line') or 'Try lowering frame count or verify image overlap.'}"
                )
            diagnostics["selected_sparse_model"] = str(selected_sparse_model)

            txt_dir = workspace / "sparse_txt"
            txt_dir.mkdir(parents=True, exist_ok=True)
            cmd4 = [
                colmap_bin,
                "model_converter",
                "--input_path",
                str(selected_sparse_model),
                "--output_path",
                str(txt_dir),
                "--output_type",
                "TXT",
            ]
            info = self._run_cmd(cmd4, workspace, "model_converter", lf, on_log)
            diagnostics["steps"].append(info)
            if info.get("last_error_line"):
                diagnostics["last_error_line"] = str(info.get("last_error_line"))
            if int(info.get("returncode", 1)) != 0:
                raise RuntimeError(
                    f"COLMAP model_converter failed (code={info['returncode']}). "
                    f"{diagnostics.get('last_error_line') or 'See colmap log.'}"
                )

        images_txt = workspace / "sparse_txt" / "images.txt"
        poses = parse_colmap_images_txt(images_txt)
        canonicalized_count = 0
        remapped_count = 0
        for pose in poses:
            original_name = str(pose.filename)
            canonical = self._canonicalize_pose_filename(
                original_name,
                image_root=image_root,
                image_path_for_colmap=image_path_for_colmap,
            )
            if canonical != original_name:
                canonicalized_count += 1
            mapped = filename_map.get(canonical, canonical) if filename_map else canonical
            if mapped != canonical:
                remapped_count += 1
            pose.filename = mapped
        diagnostics["pose_filename_canonicalized_count"] = int(canonicalized_count)
        diagnostics["pose_filename_remapped_count"] = int(remapped_count)
        diagnostics["pose_count"] = int(len(poses))
        if not poses:
            raise RuntimeError("COLMAP produced no camera poses in images.txt")

        return PoseEstimationResult(
            poses=poses,
            backend="colmap",
            diagnostics=diagnostics,
            raw_log_paths={"colmap_log": str(log_path), "images_txt": str(images_txt)},
            error=None,
        )
