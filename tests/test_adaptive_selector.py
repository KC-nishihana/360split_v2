from core.adaptive_selector import AdaptiveSelector


def test_compute_ssim_identical_is_high(checkerboard_bgr):
    selector = AdaptiveSelector(ssim_scale=1.0)
    ssim = selector.compute_ssim(checkerboard_bgr, checkerboard_bgr)
    assert 0.99 <= ssim <= 1.0


def test_optical_flow_downscale_returns_finite(checkerboard_bgr):
    selector = AdaptiveSelector(flow_downscale=0.5)
    flow = selector.compute_optical_flow_magnitude(checkerboard_bgr, checkerboard_bgr)
    assert flow >= 0.0
