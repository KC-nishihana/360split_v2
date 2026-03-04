"""Fisheye-to-pinhole projection with cached remap maps."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

from processing.fisheye_splitter import _make_knew, _r_from_ypr, _undistort_map
from osv_keyframe_app.config import AppConfig, CameraConfig, DirectionConfig

logger = logging.getLogger(__name__)


class FisheyeProjector:
    """Project fisheye frames to N pinhole-direction images.

    Builds and caches remap maps for each (stream, direction) pair.
    For 2 streams x 4 directions = 8 map pairs, built once at init.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._maps: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._directions = config.projection.directions
        self._build_maps()

    def _build_maps(self) -> None:
        """Build all remap map pairs eagerly at init."""
        proj = self._config.projection
        out_size = (proj.output_width, proj.output_height)
        knew = _make_knew(proj.output_width, proj.output_height, proj.hfov_deg, proj.vfov_deg)

        for stream in ("front", "back"):
            cam = self._config.get_camera(stream)
            K = cam.K_np
            D = cam.D_np

            for direction in self._directions:
                R = _r_from_ypr(direction.yaw_deg, direction.pitch_deg, direction.roll_deg)
                map1, map2 = _undistort_map(K, D, R, knew, out_size, cam.model)
                self._maps[(stream, direction.name)] = (map1, map2)

        logger.info(
            f"Built {len(self._maps)} remap maps "
            f"({out_size[0]}x{out_size[1]}, "
            f"hfov={proj.hfov_deg}, vfov={proj.vfov_deg})"
        )

    def project(self, frame: np.ndarray, stream: str) -> Dict[str, np.ndarray]:
        """Project a single fisheye frame to all configured directions.

        Parameters
        ----------
        frame : BGR fisheye image
        stream : "front" or "back"

        Returns
        -------
        Dict mapping direction name to projected pinhole image
        """
        results: Dict[str, np.ndarray] = {}
        for direction in self._directions:
            key = (stream, direction.name)
            map1, map2 = self._maps[key]
            results[direction.name] = cv2.remap(
                frame, map1, map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return results

    def get_pinhole_params(self) -> Dict[str, float]:
        """Get pinhole camera parameters for COLMAP (PINHOLE model).

        Returns fx, fy, cx, cy computed from projection config.
        """
        proj = self._config.projection
        return {
            "fx": proj.fx,
            "fy": proj.fy,
            "cx": proj.cx,
            "cy": proj.cy,
            "width": proj.output_width,
            "height": proj.output_height,
        }

    @property
    def direction_names(self) -> list[str]:
        """List of direction names in order."""
        return [d.name for d in self._directions]

    @property
    def stream_names(self) -> list[str]:
        return ["front", "back"]
