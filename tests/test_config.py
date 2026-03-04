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
    assert restored.to_selector_dict()["STAGE1_LR_MERGE_MODE"] == "asymmetric_sky_v1"
    assert restored.to_selector_dict()["STAGE1_LR_ASYM_WEAK_FLOOR"] == pytest.approx(0.35, abs=1e-6)
    assert restored.to_selector_dict()["STAGE1_LR_SKY_RATIO_THRESHOLD"] == pytest.approx(0.55, abs=1e-6)
    assert restored.to_selector_dict()["STAGE1_LR_SKY_RATIO_DIFF_THRESHOLD"] == pytest.approx(0.20, abs=1e-6)
    assert restored.to_selector_dict()["STAGE1_LR_QUALITY_GAP_THRESHOLD"] == pytest.approx(0.15, abs=1e-6)
    assert restored.to_selector_dict()["STAGE1_LR_SEMANTIC_SKY_ENABLED"] is True
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
    assert restored.to_selector_dict()["POSE_BACKEND"] == "vo"
    assert restored.to_selector_dict()["NMS_TIME_WINDOW"] == pytest.approx(1.0, abs=1e-6)
    assert restored.to_selector_dict()["COLMAP_KEYFRAME_POLICY"] == ""
    assert restored.to_selector_dict()["COLMAP_PIPELINE_MODE"] == ""
    assert restored.to_selector_dict()["COLMAP_KEYFRAME_TARGET_MODE"] == "auto"
    assert restored.to_selector_dict()["COLMAP_KEYFRAME_TARGET_MIN"] == 120
    assert restored.to_selector_dict()["COLMAP_KEYFRAME_TARGET_MAX"] == 240
    assert restored.to_selector_dict()["COLMAP_NMS_WINDOW_SEC"] == pytest.approx(0.35, abs=1e-6)
    assert restored.to_selector_dict()["COLMAP_ENABLE_STAGE0"] is True
    assert restored.to_selector_dict()["COLMAP_MOTION_AWARE_SELECTION"] is True
    assert restored.to_selector_dict()["COLMAP_NMS_MOTION_WINDOW_RATIO"] == pytest.approx(0.5, abs=1e-6)
    assert restored.to_selector_dict()["COLMAP_STAGE1_ADAPTIVE_THRESHOLD"] is True
    assert restored.to_selector_dict()["COLMAP_STAGE1_MIN_CANDIDATES_PER_BIN"] == 3
    assert restored.to_selector_dict()["COLMAP_STAGE1_MAX_CANDIDATES"] == 360
    assert restored.to_selector_dict()["COLMAP_SELECTION_PROFILE"] == ""
    assert restored.to_selector_dict()["COLMAP_STAGE2_ENTRY_BUDGET"] == 180
    assert restored.to_selector_dict()["COLMAP_STAGE2_ENTRY_MIN_GAP"] == 3
    assert restored.to_selector_dict()["COLMAP_DIVERSITY_SSIM_THRESHOLD"] == pytest.approx(0.93, abs=1e-6)
    assert restored.to_selector_dict()["COLMAP_DIVERSITY_PHASH_HAMMING"] == 10
    assert restored.to_selector_dict()["COLMAP_FINAL_TARGET_POLICY"] == "soft_auto"
    assert restored.to_selector_dict()["COLMAP_FINAL_SOFT_MIN"] == 80
    assert restored.to_selector_dict()["COLMAP_FINAL_SOFT_MAX"] == 220
    assert restored.to_selector_dict()["COLMAP_NO_SUPPLEMENT_ON_LOW_QUALITY"] is True
    assert restored.to_selector_dict()["COLMAP_RIG_POLICY"] == "lr_opk"
    assert restored.to_selector_dict()["COLMAP_RIG_SEED_OPK_DEG"] == [0.0, 0.0, 180.0]
    assert restored.to_selector_dict()["COLMAP_WORKSPACE_SCOPE"] == "run_scoped"
    assert restored.to_selector_dict()["COLMAP_REUSE_DB"] is False
    assert restored.to_selector_dict()["COLMAP_ANALYSIS_MASK_PROFILE"] == "colmap_safe"
    assert restored.to_selector_dict()["COLMAP_SPARSE_MODEL_PICK_POLICY"] == "registered_then_coverage"
    assert restored.to_selector_dict()["COLMAP_INPUT_SUBSET_ENABLED"] is True
    assert restored.to_selector_dict()["COLMAP_INPUT_GATE_METHOD"] == "homography_degeneracy_v1"
    assert restored.to_selector_dict()["COLMAP_INPUT_GATE_STRENGTH"] == "medium"
    assert restored.to_selector_dict()["COLMAP_INPUT_MIN_KEEP_RATIO"] == pytest.approx(0.20, abs=1e-6)
    assert restored.to_selector_dict()["COLMAP_INPUT_MAX_GAP_RESCUE_FRAMES"] == 150
    assert restored.to_selector_dict()["POSE_EXPORT_FORMAT"] == "internal"
    assert restored.to_selector_dict()["POSE_SELECT_ENABLE_TRANSLATION"] is True


def test_weight_validation_raises():
    bad = {
        "weight_sharpness": 0.5,
        "weight_exposure": 0.5,
        "weight_geometric": 0.5,
        "weight_content": 0.5,
    }
    with pytest.raises(ValueError):
        KeyframeConfig.from_dict(bad)
