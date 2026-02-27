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


def test_stage0_adaptive_subsample_changes_effective_interval():
    import cv2
    import numpy as np
    from types import SimpleNamespace

    class _Loader:
        def __init__(self, frames):
            self.frames = frames
            self.is_paired = False

        def get_frame(self, idx):
            if 0 <= idx < len(self.frames):
                return self.frames[idx]
            return None

    class _VO:
        def estimate(self, *_args, **_kwargs):
            return SimpleNamespace(
                vo_valid=True,
                step_proxy=4.0,
                t_dir=[1.0, 0.0, 0.0],
                r_rel_q_wxyz=[1.0, 0.0, 0.0, 0.0],
                rotation_delta_deg=2.0,
                translation_delta_rel=1.0,
                match_count=40,
                tracked_count=60,
                inlier_ratio=0.7,
                vo_confidence=0.8,
                feature_uniformity=0.7,
                track_sufficiency=1.0,
                pose_plausibility=0.9,
                essential_method_used="ransac",
            )

    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(12):
        dx = 0 if i < 6 else i * 3
        mat = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0]])
        frames.append(cv2.warpAffine(base, mat, (300, 200), flags=cv2.INTER_LINEAR))

    selector = KeyframeSelector(
        config={
            "VO_ENABLED": True,
            "VO_FRAME_SUBSAMPLE": 4,
            "VO_ADAPTIVE_SUBSAMPLE": True,
            "VO_SUBSAMPLE_MIN": 1,
            "STAGE0_STRIDE": 1,
        }
    )
    selector.vo_estimator = _VO()
    selector._resolve_vo_runtime = lambda **_kwargs: (True, None, "enabled")
    loader = _Loader(frames)
    metadata = SimpleNamespace(frame_count=len(frames), rig_calibration=None)
    out = selector._stage0_lightweight_motion_scan(loader, metadata, progress_callback=None)
    values = [int(v.get("vo_effective_subsample", 4)) for v in out.values()]
    assert values
    assert min(values) <= 2
    assert max(values) >= 3
