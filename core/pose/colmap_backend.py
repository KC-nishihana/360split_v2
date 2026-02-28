from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
            raise RuntimeError(
                "COLMAP is not installed or not in PATH. Install with `brew install colmap` or pass --colmap-path."
            )

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
            if "error" in low or "failed" in low:
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

            latest_sparse_model = self._find_latest_sparse_model_dir(sparse_dir)
            if latest_sparse_model is None:
                raise RuntimeError(
                    "COLMAP mapper did not produce sparse model. "
                    f"{diagnostics.get('last_error_line') or 'Try lowering frame count or verify image overlap.'}"
                )
            diagnostics["selected_sparse_model"] = str(latest_sparse_model)

            txt_dir = workspace / "sparse_txt"
            txt_dir.mkdir(parents=True, exist_ok=True)
            cmd4 = [
                colmap_bin,
                "model_converter",
                "--input_path",
                str(latest_sparse_model),
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
        if filename_map:
            for pose in poses:
                pose.filename = filename_map.get(str(pose.filename), str(pose.filename))
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
