"""
リグ関連データモデル。

前後魚眼やステレオなど、複数レンズ構成で共有する
キャリブレーション/変換行列情報を保持する。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class LensIntrinsics:
    """単一レンズの内部パラメータ。"""
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_width: int
    image_height: int
    model: str = "fisheye"


@dataclass
class RigCalibration:
    """2レンズリグの内部・外部パラメータ。"""
    lens_a: LensIntrinsics
    lens_b: LensIntrinsics
    rotation_ab: np.ndarray
    translation_ab: np.ndarray
    reprojection_error: float = 0.0


@dataclass
class RigTransforms:
    """スティッチング等の変換行列。"""
    matrices: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RigMetadata:
    """VideoMetadataから参照するリグ情報。"""
    rig_type: str = "monocular"
    calibration: Optional[RigCalibration] = None
    transforms: RigTransforms = field(default_factory=RigTransforms)
