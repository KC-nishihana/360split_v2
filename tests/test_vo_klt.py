import cv2
import numpy as np

from core.visual_odometry.vo_klt import KLTVisualOdometry


def _make_textured_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(360, 640), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def _translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    mat = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(frame, mat, (w, h), flags=cv2.INTER_LINEAR)


def test_vo_klt_confidence_range_and_fields():
    prev = _make_textured_frame(1)
    cur = _translate(prev, 4.0, 3.0)

    vo = KLTVisualOdometry(
        max_features=1000,
        min_track_points=24,
        essential_method="auto",
        subpixel_refine=True,
    )
    m = vo.estimate(prev, cur, calibration=None)

    assert 0.0 <= float(m.vo_confidence) <= 1.0
    assert 0.0 <= float(m.feature_uniformity) <= 1.0
    assert 0.0 <= float(m.track_sufficiency) <= 1.0
    assert 0.0 <= float(m.pose_plausibility) <= 1.0
    assert isinstance(m.essential_method_used, str)


def test_vo_klt_essential_method_switches():
    prev = _make_textured_frame(2)
    cur = _translate(prev, 3.0, 2.0)

    methods = ["ransac", "auto", "magsac"]
    for method in methods:
        vo = KLTVisualOdometry(
            max_features=1000,
            min_track_points=20,
            essential_method=method,
            subpixel_refine=False,
        )
        m = vo.estimate(prev, cur, calibration=None)
        assert isinstance(m.essential_method_used, str)
        if method == "ransac":
            assert m.essential_method_used in {"ransac", "none"}
        elif method == "auto":
            assert m.essential_method_used in {"ransac_auto", "magsac_auto", "none"}
        else:
            assert m.essential_method_used in {"magsac", "ransac_fallback", "none"}


def test_vo_klt_subpixel_toggle_runs():
    prev = _make_textured_frame(3)
    cur = _translate(prev, 5.0, 1.0)

    vo_off = KLTVisualOdometry(subpixel_refine=False)
    vo_on = KLTVisualOdometry(subpixel_refine=True)

    m_off = vo_off.estimate(prev, cur, calibration=None)
    m_on = vo_on.estimate(prev, cur, calibration=None)

    assert 0.0 <= float(m_off.vo_confidence) <= 1.0
    assert 0.0 <= float(m_on.vo_confidence) <= 1.0
