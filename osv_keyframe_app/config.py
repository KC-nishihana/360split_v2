"""YAML-based configuration for OSV Keyframe App."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class CameraConfig:
    """Camera intrinsic parameters (fisheye or opencv model)."""

    K: List[List[float]]
    D: List[float]
    image_size: List[int]  # [width, height]
    model: str = "fisheye"

    def __post_init__(self) -> None:
        if len(self.K) != 3 or any(len(row) != 3 for row in self.K):
            raise ValueError("K must be a 3x3 matrix")
        if self.model == "fisheye" and len(self.D) != 4:
            raise ValueError(f"fisheye model expects 4 distortion coefficients, got {len(self.D)}")
        if len(self.image_size) != 2:
            raise ValueError("image_size must be [width, height]")

    @property
    def K_np(self) -> np.ndarray:
        return np.array(self.K, dtype=np.float64)

    @property
    def D_np(self) -> np.ndarray:
        return np.array(self.D, dtype=np.float64)


@dataclass
class DirectionConfig:
    """Projection direction (yaw/pitch/roll in degrees)."""

    name: str
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


DEFAULT_DIRECTIONS = [
    DirectionConfig(name="front", yaw_deg=0.0),
    DirectionConfig(name="left", yaw_deg=-90.0),
    DirectionConfig(name="right", yaw_deg=90.0),
    DirectionConfig(name="back", yaw_deg=180.0),
]


@dataclass
class ProjectionConfig:
    """Pinhole projection parameters."""

    directions: List[DirectionConfig] = field(default_factory=lambda: list(DEFAULT_DIRECTIONS))
    hfov_deg: float = 90.0
    vfov_deg: float = 90.0
    output_size: List[int] = field(default_factory=lambda: [1600, 1600])

    def __post_init__(self) -> None:
        if len(self.output_size) != 2:
            raise ValueError("output_size must be [width, height]")
        if not (1.0 <= self.hfov_deg <= 180.0):
            raise ValueError(f"hfov_deg must be in [1, 180], got {self.hfov_deg}")
        if not (1.0 <= self.vfov_deg <= 180.0):
            raise ValueError(f"vfov_deg must be in [1, 180], got {self.vfov_deg}")

    @property
    def output_width(self) -> int:
        return int(self.output_size[0])

    @property
    def output_height(self) -> int:
        return int(self.output_size[1])

    @property
    def fx(self) -> float:
        return self.output_width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.output_height / (2.0 * math.tan(math.radians(self.vfov_deg) / 2.0))

    @property
    def cx(self) -> float:
        return self.output_width / 2.0

    @property
    def cy(self) -> float:
        return self.output_height / 2.0


@dataclass
class ExtractionConfig:
    """Frame extraction parameters."""

    fps: float = 2.0
    start_sec: float = 0.0
    end_sec: Optional[float] = None
    max_frames: Optional[int] = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.start_sec < 0:
            raise ValueError(f"start_sec must be non-negative, got {self.start_sec}")


@dataclass
class ThresholdConfig:
    """Selection thresholds for one tier (SfM or 3DGS)."""

    sharpness_min: float = 100.0
    exposure_min: float = 0.3
    orb_min: int = 50
    ssim_max: float = 0.95
    per_direction_min: int = 10
    max_total: Optional[int] = None


@dataclass
class SelectionConfig:
    """2-tier selection configuration."""

    sfm: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(
        sharpness_min=150.0, exposure_min=0.4, orb_min=100,
        ssim_max=0.92, per_direction_min=20,
    ))
    gs: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(
        sharpness_min=80.0, exposure_min=0.3, orb_min=50,
        ssim_max=0.96, per_direction_min=10,
    ))


@dataclass
class ColmapConfig:
    """COLMAP execution settings."""

    enabled: bool = False
    binary_path: str = "colmap"
    workspace: str = "colmap_workspace"
    camera_model: str = "PINHOLE"
    use_gpu: bool = False
    matching_method: str = "sequential"
    mapper_ba_iterations: int = 100


@dataclass
class AppConfig:
    """Top-level application configuration."""

    camera_front: CameraConfig = field(default_factory=lambda: CameraConfig(
        K=[[1200, 0, 960], [0, 1200, 960], [0, 0, 1]],
        D=[0.0, 0.0, 0.0, 0.0],
        image_size=[1920, 1920],
    ))
    camera_back: CameraConfig = field(default_factory=lambda: CameraConfig(
        K=[[1200, 0, 960], [0, 1200, 960], [0, 0, 1]],
        D=[0.0, 0.0, 0.0, 0.0],
        image_size=[1920, 1920],
    ))
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    output_dir: str = "out"
    osv_path: Optional[str] = None
    project_name: str = "osv"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("Config must be a YAML mapping")
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        """Build AppConfig from a parsed dict."""
        kwargs: Dict[str, Any] = {}

        for cam_key in ("camera_front", "camera_back"):
            if cam_key in d:
                kwargs[cam_key] = CameraConfig(**d[cam_key])

        if "projection" in d:
            proj = dict(d["projection"])
            if "directions" in proj:
                proj["directions"] = [
                    DirectionConfig(**dd) if isinstance(dd, dict) else dd
                    for dd in proj["directions"]
                ]
            kwargs["projection"] = ProjectionConfig(**proj)

        if "extraction" in d:
            kwargs["extraction"] = ExtractionConfig(**d["extraction"])

        if "selection" in d:
            sel = d["selection"]
            sfm = ThresholdConfig(**sel["sfm"]) if "sfm" in sel else ThresholdConfig()
            gs = ThresholdConfig(**sel["gs"]) if "gs" in sel else ThresholdConfig()
            kwargs["selection"] = SelectionConfig(sfm=sfm, gs=gs)

        if "colmap" in d:
            kwargs["colmap"] = ColmapConfig(**d["colmap"])

        for key in ("output_dir", "osv_path", "project_name"):
            if key in d:
                kwargs[key] = d[key]

        return cls(**kwargs)

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for config_used.json."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            return obj

        raw = asdict(self)
        return json.loads(json.dumps(raw, default=_convert, ensure_ascii=False))

    def get_camera(self, stream: str) -> CameraConfig:
        """Get camera config for a stream name."""
        if stream == "front":
            return self.camera_front
        elif stream == "back":
            return self.camera_back
        raise ValueError(f"Unknown stream: {stream}")
