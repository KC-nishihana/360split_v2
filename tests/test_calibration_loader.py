from pathlib import Path

import pytest

from core.visual_odometry.calibration import load_calibration_xml


def test_load_agisoft_fisheye_xml() -> None:
    xml = Path(__file__).resolve().parents[1] / "calib" / "cam1_agisoft.xml"
    calib = load_calibration_xml(str(xml), model_hint="fisheye")
    assert calib is not None
    assert calib.model == "fisheye"
    assert calib.image_size == (3840, 3840)
    assert calib.dist.shape[0] == 4
    # Agisoft schema stores f, b1 where fx = f + b1 and fy = f.
    assert calib.camera_matrix[0, 0] == pytest.approx(1047.5471925995, rel=1e-6)
    assert calib.camera_matrix[1, 1] == pytest.approx(1048.7178278271, rel=1e-6)


def test_load_agisoft_as_opencv_model() -> None:
    xml = Path(__file__).resolve().parents[1] / "calib" / "cam1_agisoft.xml"
    calib = load_calibration_xml(str(xml), model_hint="opencv")
    assert calib is not None
    assert calib.model == "opencv"
    assert calib.dist.shape[0] == 5
