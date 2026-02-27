import pytest

from config import KeyframeConfig


def test_keyframe_config_round_trip():
    cfg = KeyframeConfig()
    d = cfg.to_selector_dict()
    restored = KeyframeConfig.from_dict(d)
    assert restored.to_selector_dict()["WEIGHT_SHARPNESS"] == pytest.approx(d["WEIGHT_SHARPNESS"], abs=1e-6)
    assert restored.to_selector_dict()["SOFTMAX_BETA"] == pytest.approx(d["SOFTMAX_BETA"], abs=1e-6)
    assert restored.to_selector_dict()["QUALITY_FILTER_ENABLED"] is True
    assert restored.to_selector_dict()["QUALITY_THRESHOLD"] == pytest.approx(d["QUALITY_THRESHOLD"], abs=1e-6)
    assert restored.to_selector_dict()["QUALITY_ROI_RATIO"] == pytest.approx(d["QUALITY_ROI_RATIO"], abs=1e-6)
    assert restored.to_selector_dict()["VO_ESSENTIAL_METHOD"] == "auto"
    assert restored.to_selector_dict()["VO_SUBPIXEL_REFINE"] is True
    assert restored.to_selector_dict()["VO_ADAPTIVE_SUBSAMPLE"] is False
    assert restored.to_selector_dict()["VO_SUBSAMPLE_MIN"] == 1
    assert restored.to_selector_dict()["OPENCV_THREAD_COUNT"] == 0
    assert restored.to_selector_dict()["STAGE1_PROCESS_WORKERS"] == 0
    assert restored.to_selector_dict()["STAGE1_PREFETCH_SIZE"] == 32
    assert restored.to_selector_dict()["STAGE1_METRICS_BATCH_SIZE"] == 64
    assert restored.to_selector_dict()["STAGE1_GPU_BATCH_ENABLED"] is True
    assert restored.to_selector_dict()["DARWIN_CAPTURE_BACKEND"] == "auto"
    assert restored.to_selector_dict()["MPS_MIN_PIXELS"] == 256 * 256


def test_weight_validation_raises():
    bad = {
        "weight_sharpness": 0.5,
        "weight_exposure": 0.5,
        "weight_geometric": 0.5,
        "weight_content": 0.5,
    }
    with pytest.raises(ValueError):
        KeyframeConfig.from_dict(bad)
