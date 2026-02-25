import numpy as np

from core.quality_evaluator import QualityEvaluator


def test_quality_evaluate_keys(gradient_bgr):
    q = QualityEvaluator(eval_scale=1.0)
    out = q.evaluate(gradient_bgr)
    assert {"sharpness", "motion_blur", "exposure", "softmax_depth"}.issubset(out.keys())


def test_quality_stage1_fast_range(checkerboard_bgr):
    q = QualityEvaluator(eval_scale=0.5)
    out = q.evaluate_stage1_fast(checkerboard_bgr)
    assert out["sharpness"] >= 0.0
    assert 0.0 <= out["motion_blur"] <= 1.0
    assert 0.0 <= out["exposure"] <= 1.0


def test_quality_motion_blur_modes(noise_bgr):
    legacy = QualityEvaluator(eval_scale=1.0, motion_blur_method="legacy").evaluate_stage1_fast(noise_bgr)
    fft = QualityEvaluator(eval_scale=1.0, motion_blur_method="fft_hybrid").evaluate_stage1_fast(noise_bgr)
    assert 0.0 <= legacy["motion_blur"] <= 1.0
    assert 0.0 <= fft["motion_blur"] <= 1.0
