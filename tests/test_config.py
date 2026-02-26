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


def test_weight_validation_raises():
    bad = {
        "weight_sharpness": 0.5,
        "weight_exposure": 0.5,
        "weight_geometric": 0.5,
        "weight_content": 0.5,
    }
    with pytest.raises(ValueError):
        KeyframeConfig.from_dict(bad)
