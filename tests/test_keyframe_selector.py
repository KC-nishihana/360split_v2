from core.keyframe_selector import KeyframeSelector, KeyframeInfo


def test_combined_score_range():
    selector = KeyframeSelector(config={"WEIGHT_SHARPNESS": 0.3, "WEIGHT_EXPOSURE": 0.15, "WEIGHT_GEOMETRIC": 0.3, "WEIGHT_CONTENT": 0.25})
    score = selector._compute_combined_score(
        quality_scores={"sharpness": 0.6, "exposure": 0.7},
        geometric_scores={"gric": 0.8},
        adaptive_scores={"ssim": 0.4},
    )
    assert 0.0 <= score <= 1.0


def test_apply_nms_prefers_high_score():
    selector = KeyframeSelector()
    cands = [
        KeyframeInfo(frame_index=10, timestamp=1.0, combined_score=0.5),
        KeyframeInfo(frame_index=12, timestamp=1.1, combined_score=0.9),
        KeyframeInfo(frame_index=40, timestamp=3.0, combined_score=0.4),
    ]
    selected = selector._apply_nms(cands)
    idx = [k.frame_index for k in selected]
    assert 12 in idx
