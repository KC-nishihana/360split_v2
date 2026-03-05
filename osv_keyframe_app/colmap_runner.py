"""COLMAP wrapper: mapper + image_registrator + point_triangulator."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from osv_keyframe_app.config import ColmapConfig, ProjectionConfig

logger = logging.getLogger(__name__)

LogCallback = Optional[Callable[[str], None]]


@dataclass
class RegistrationStatus:
    """Status of an image in COLMAP registration."""

    filename: str
    registered: bool
    qw: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0


@dataclass
class ColmapResult:
    """Result of a COLMAP run."""

    success: bool
    message: str
    sparse_model_path: Optional[Path] = None
    registered: List[RegistrationStatus] = field(default_factory=list)
    log: str = ""


class ColmapRunner:
    """Orchestrate COLMAP for SfM mapper + 3DGS image_registrator."""

    def __init__(
        self,
        config: ColmapConfig,
        projection: ProjectionConfig,
    ) -> None:
        self._config = config
        self._projection = projection
        self._colmap_bin = self._resolve_binary(config.binary_path)
        # Detect GPU option names: COLMAP 3.13+ renamed SiftExtraction/SiftMatching
        # to FeatureExtraction/FeatureMatching.
        self._extract_gpu_opt = (
            "--FeatureExtraction.use_gpu"
            if self._command_supports_option(self._colmap_bin, "feature_extractor", "--FeatureExtraction.use_gpu")
            else "--SiftExtraction.use_gpu"
        )
        self._match_gpu_opt = (
            "--FeatureMatching.use_gpu"
            if self._command_supports_option(self._colmap_bin, "sequential_matcher", "--FeatureMatching.use_gpu")
            else "--SiftMatching.use_gpu"
        )

    @staticmethod
    def _resolve_binary(binary_path: str) -> str:
        """Find COLMAP binary."""
        resolved = shutil.which(binary_path)
        if resolved:
            return resolved
        if Path(binary_path).exists():
            return str(binary_path)
        raise FileNotFoundError(
            f"COLMAP binary not found: {binary_path}\n"
            "Install COLMAP: https://colmap.github.io/install.html"
        )

    @staticmethod
    def _command_supports_option(colmap_bin: str, command: str, option_name: str) -> bool:
        """Return True if the given COLMAP command lists option_name in its help text."""
        try:
            proc = subprocess.run(
                [colmap_bin, command, "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            return option_name in (proc.stdout or "")
        except Exception:
            return False

    def _camera_params_str(self) -> str:
        """Build COLMAP camera params string from projection config."""
        p = self._projection
        return f"{p.fx:.6f},{p.fy:.6f},{p.cx:.6f},{p.cy:.6f}"

    def run_sfm(
        self,
        image_dir: Path,
        workspace: Path,
        log_callback: LogCallback = None,
    ) -> ColmapResult:
        """Run full SfM pipeline: feature_extractor → matcher → mapper.

        Parameters
        ----------
        image_dir : directory with SfM images
        workspace : COLMAP workspace directory
        log_callback : optional callback for log lines
        """
        workspace.mkdir(parents=True, exist_ok=True)
        db_path = workspace / "database.db"
        sparse_dir = workspace / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Remove old database for clean run
        if db_path.exists():
            db_path.unlink()

        full_log = []

        try:
            # Feature extraction
            self._run_feature_extractor(db_path, image_dir, log_callback, full_log)

            # Matching
            self._run_matcher(db_path, log_callback, full_log)

            # Mapper
            self._run_mapper(db_path, image_dir, sparse_dir, log_callback, full_log)

            # Convert to TXT
            model_dir = self._find_best_model(sparse_dir)
            if model_dir:
                txt_dir = workspace / "sparse_txt"
                self._run_model_converter(model_dir, txt_dir, log_callback, full_log)

                registered = self._parse_images_txt(txt_dir / "images.txt")
                return ColmapResult(
                    success=True,
                    message=f"SfM complete: {len(registered)} registered images",
                    sparse_model_path=model_dir,
                    registered=registered,
                    log="\n".join(full_log),
                )
            else:
                return ColmapResult(
                    success=False,
                    message="Mapper produced no model",
                    log="\n".join(full_log),
                )

        except RuntimeError as e:
            return ColmapResult(
                success=False,
                message=str(e),
                log="\n".join(full_log),
            )

    def run_gs_registration(
        self,
        gs_image_dir: Path,
        sfm_workspace: Path,
        log_callback: LogCallback = None,
    ) -> ColmapResult:
        """Register 3DGS images into existing SfM model.

        Uses image_registrator to add new images without BA/triangulation.

        Parameters
        ----------
        gs_image_dir : directory with 3DGS images (additional to SfM)
        sfm_workspace : workspace from previous run_sfm
        log_callback : optional callback for log lines
        """
        db_path = sfm_workspace / "database.db"
        sparse_dir = sfm_workspace / "sparse"
        full_log = []

        if not db_path.exists():
            return ColmapResult(
                success=False,
                message="SfM database not found. Run SfM first.",
            )

        model_dir = self._find_best_model(sparse_dir)
        if not model_dir:
            return ColmapResult(
                success=False,
                message="No SfM model found. Run SfM first.",
            )

        try:
            # Extract features for new images
            self._run_feature_extractor(db_path, gs_image_dir, log_callback, full_log)

            # Match new images against existing
            self._run_matcher(db_path, log_callback, full_log)

            # Run image_registrator
            registered_dir = sfm_workspace / "sparse_registered"
            registered_dir.mkdir(parents=True, exist_ok=True)
            self._run_image_registrator(
                db_path, model_dir, registered_dir, log_callback, full_log,
            )

            # Convert to TXT
            txt_dir = sfm_workspace / "registered_txt"
            self._run_model_converter(registered_dir, txt_dir, log_callback, full_log)

            registered = self._parse_images_txt(txt_dir / "images.txt")
            return ColmapResult(
                success=True,
                message=f"Registration complete: {len(registered)} images",
                sparse_model_path=registered_dir,
                registered=registered,
                log="\n".join(full_log),
            )

        except RuntimeError as e:
            return ColmapResult(
                success=False,
                message=str(e),
                log="\n".join(full_log),
            )

    def run_point_triangulator(
        self,
        image_dir: Path,
        workspace: Path,
        input_model: Optional[Path] = None,
        log_callback: LogCallback = None,
    ) -> ColmapResult:
        """Run point_triangulator on an existing model."""
        db_path = workspace / "database.db"
        full_log = []

        if input_model is None:
            input_model = self._find_best_model(workspace / "sparse")
            if not input_model:
                input_model = workspace / "sparse_registered"

        output_dir = workspace / "sparse_triangulated"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._run_cmd(
                [
                    self._colmap_bin, "point_triangulator",
                    "--database_path", str(db_path),
                    "--image_path", str(image_dir),
                    "--input_path", str(input_model),
                    "--output_path", str(output_dir),
                ],
                log_callback, full_log,
            )
            return ColmapResult(
                success=True,
                message="Point triangulation complete",
                sparse_model_path=output_dir,
                log="\n".join(full_log),
            )
        except RuntimeError as e:
            return ColmapResult(
                success=False,
                message=str(e),
                log="\n".join(full_log),
            )

    # ── Internal COLMAP subprocess calls ──

    def _run_feature_extractor(
        self, db_path: Path, image_dir: Path,
        log_callback: LogCallback, full_log: list,
    ) -> None:
        cmd = [
            self._colmap_bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", self._config.camera_model,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", self._camera_params_str(),
        ]
        if not self._config.use_gpu:
            cmd.extend([self._extract_gpu_opt, "0"])
        self._run_cmd(cmd, log_callback, full_log)

    def _run_matcher(
        self, db_path: Path,
        log_callback: LogCallback, full_log: list,
    ) -> None:
        method = self._config.matching_method
        cmd = [
            self._colmap_bin, f"{method}_matcher",
            "--database_path", str(db_path),
        ]
        if not self._config.use_gpu:
            cmd.extend([self._match_gpu_opt, "0"])
        self._run_cmd(cmd, log_callback, full_log)

    def _run_mapper(
        self, db_path: Path, image_dir: Path, sparse_dir: Path,
        log_callback: LogCallback, full_log: list,
    ) -> None:
        cmd = [
            self._colmap_bin, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_global_max_num_iterations",
            str(self._config.mapper_ba_iterations),
        ]
        self._run_cmd(cmd, log_callback, full_log)

    def _run_image_registrator(
        self, db_path: Path, input_model: Path, output_dir: Path,
        log_callback: LogCallback, full_log: list,
    ) -> None:
        cmd = [
            self._colmap_bin, "image_registrator",
            "--database_path", str(db_path),
            "--input_path", str(input_model),
            "--output_path", str(output_dir),
        ]
        self._run_cmd(cmd, log_callback, full_log)

    def _run_model_converter(
        self, input_dir: Path, output_dir: Path,
        log_callback: LogCallback, full_log: list,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._colmap_bin, "model_converter",
            "--input_path", str(input_dir),
            "--output_path", str(output_dir),
            "--output_type", "TXT",
        ]
        self._run_cmd(cmd, log_callback, full_log)

    @staticmethod
    def _run_cmd(
        cmd: List[str],
        log_callback: LogCallback,
        full_log: list,
    ) -> None:
        """Run a subprocess command with logging."""
        cmd_str = " ".join(cmd)
        logger.info(f"Running: {cmd_str}")
        if log_callback:
            log_callback(f"$ {cmd_str}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                full_log.append(line)
                if log_callback:
                    log_callback(line)

            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"COLMAP command failed (exit {proc.returncode}): {cmd[1]}"
                )

        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {cmd[0]}")

    @staticmethod
    def _find_best_model(sparse_dir: Path) -> Optional[Path]:
        """Find the best (largest) sparse model in the COLMAP output."""
        if not sparse_dir.exists():
            return None
        models = sorted(sparse_dir.iterdir())
        models = [m for m in models if m.is_dir()]
        if not models:
            # Check if sparse_dir itself contains model files
            if (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists():
                return sparse_dir
            return None
        # Return the first model (typically "0")
        return models[0]

    @staticmethod
    def _parse_images_txt(images_txt: Path) -> List[RegistrationStatus]:
        """Parse COLMAP images.txt to get registration status."""
        results: List[RegistrationStatus] = []
        if not images_txt.exists():
            return results

        with open(images_txt, encoding="utf-8") as f:
            lines = f.readlines()

        # Skip comment lines (start with #)
        i = 0
        while i < len(lines) and lines[i].startswith("#"):
            i += 1

        # Every other line is image data, alternating with point data
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue

            parts = line.split()
            if len(parts) < 10:
                # Skip point2D lines (variable length)
                continue

            try:
                # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                tx = float(parts[5])
                ty = float(parts[6])
                tz = float(parts[7])
                name = parts[9]

                results.append(RegistrationStatus(
                    filename=name,
                    registered=True,
                    qw=qw, qx=qx, qy=qy, qz=qz,
                    tx=tx, ty=ty, tz=tz,
                ))
            except (ValueError, IndexError):
                continue

            # Skip the next line (2D points)
            i += 1

        return results

    @staticmethod
    def write_registration_csv(
        registered: List[RegistrationStatus],
        all_images: List[str],
        path: Path,
    ) -> None:
        """Write registration status CSV including unregistered images."""
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        reg_names = {r.filename for r in registered}

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "registered", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            ])
            writer.writeheader()

            for r in registered:
                writer.writerow({
                    "filename": r.filename,
                    "registered": True,
                    "qw": f"{r.qw:.8f}",
                    "qx": f"{r.qx:.8f}",
                    "qy": f"{r.qy:.8f}",
                    "qz": f"{r.qz:.8f}",
                    "tx": f"{r.tx:.6f}",
                    "ty": f"{r.ty:.6f}",
                    "tz": f"{r.tz:.6f}",
                })

            for name in sorted(all_images):
                if name not in reg_names:
                    writer.writerow({
                        "filename": name,
                        "registered": False,
                        "qw": "", "qx": "", "qy": "", "qz": "",
                        "tx": "", "ty": "", "tz": "",
                    })

        logger.info(
            f"Registration CSV: {len(registered)} registered, "
            f"{len(all_images) - len(reg_names)} unregistered → {path}"
        )
