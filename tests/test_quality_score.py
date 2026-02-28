import pytest
import math

from core.quality_score import (
    apply_abs_guard,
    compose_quality,
    compose_legacy_quality_proxy,
    compute_raw_metrics,
    compute_raw_metrics_batch,
    normalize_batch_p10_p90,
)


def test_compute_raw_metrics_roi_returns_finite(checkerboard_bgr):
    metrics = compute_raw_metrics(checkerboard_bgr, roi_spec="circle:0.4", use_orb=True)
    for key in (
        "laplacian_var",
        "tenengrad",
        "exposure",
        "orb_keypoints",
        "black_clip",
        "white_clip",
        "contrast",
    ):
        assert key in metrics
        assert math.isfinite(float(metrics[key]))


def test_compute_raw_metrics_batch_matches_single(checkerboard_bgr, gradient_bgr):
    frames = [checkerboard_bgr, gradient_bgr]
    singles = [compute_raw_metrics(f, roi_spec="circle:0.4", use_orb=False) for f in frames]
    batched = compute_raw_metrics_batch(frames, roi_spec="circle:0.4", use_orb=False, gpu_batch_enabled=False)
    assert len(singles) == len(batched)
    for a, b in zip(singles, batched):
        assert a["laplacian_var"] == pytest.approx(b["laplacian_var"], rel=1e-6, abs=1e-6)
        assert a["tenengrad"] == pytest.approx(b["tenengrad"], rel=1e-6, abs=1e-6)
        assert a["exposure"] == pytest.approx(b["exposure"], rel=1e-6, abs=1e-6)


def test_normalize_batch_p10_p90_bounds():
    records = [
        {"laplacian_var": 10.0, "tenengrad": 10.0, "exposure": 0.1, "orb_keypoints": 5.0},
        {"laplacian_var": 20.0, "tenengrad": 20.0, "exposure": 0.5, "orb_keypoints": 10.0},
        {"laplacian_var": 30.0, "tenengrad": 30.0, "exposure": 0.9, "orb_keypoints": 20.0},
    ]
    norm, _stats = normalize_batch_p10_p90(records, p_low=10, p_high=90)
    for rec in norm:
        assert 0.0 <= rec["norm_laplacian_var"] <= 1.0
        assert 0.0 <= rec["norm_tenengrad"] <= 1.0
        assert 0.0 <= rec["norm_exposure"] <= 1.0
        assert 0.0 <= rec["norm_orb_keypoints"] <= 1.0


def test_compose_quality_range():
    metrics = {
        "norm_laplacian_var": 0.8,
        "norm_tenengrad": 0.7,
        "norm_exposure": 0.6,
        "norm_orb_keypoints": 0.5,
    }
    score = compose_quality(metrics, {})
    assert 0.0 <= score <= 1.0


def test_apply_abs_guard():
    assert apply_abs_guard(50.0, 35.0) is True
    assert apply_abs_guard(10.0, 35.0) is False


def test_pair_quality_uses_min():
    lens_a = {"norm_laplacian_var": 0.9, "norm_tenengrad": 0.9, "norm_exposure": 0.9, "norm_orb_keypoints": 0.9}
    lens_b = {"norm_laplacian_var": 0.2, "norm_tenengrad": 0.2, "norm_exposure": 0.2, "norm_orb_keypoints": 0.2}
    q_a = compose_quality(lens_a, {})
    q_b = compose_quality(lens_b, {})
    assert min(q_a, q_b) == pytest.approx(q_b, abs=1e-9)


def test_tenengrad_downscale_keeps_reasonable_range(checkerboard_bgr):
    full = compute_raw_metrics(checkerboard_bgr, roi_spec="circle:0.4", use_orb=False, tenengrad_scale=1.0)
    down = compute_raw_metrics(checkerboard_bgr, roi_spec="circle:0.4", use_orb=False, tenengrad_scale=0.5)
    assert down["tenengrad"] >= 0.0
    assert full["tenengrad"] >= 0.0
    if full["tenengrad"] > 1e-6:
        ratio = down["tenengrad"] / full["tenengrad"]
        assert 0.2 <= ratio <= 5.0


def test_compose_legacy_quality_proxy_range():
    score = compose_legacy_quality_proxy(
        {"sharpness": 120.0, "motion_blur": 0.2, "exposure": 0.4},
        laplacian_threshold=100.0,
        motion_blur_threshold=0.3,
        exposure_threshold=0.35,
    )
    assert 0.0 <= score <= 1.0
