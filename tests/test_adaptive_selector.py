from core.adaptive_selector import AdaptiveSelector


def test_compute_ssim_identical_is_high(checkerboard_bgr):
    selector = AdaptiveSelector(ssim_scale=1.0)
    ssim = selector.compute_ssim(checkerboard_bgr, checkerboard_bgr)
    assert 0.99 <= ssim <= 1.0
