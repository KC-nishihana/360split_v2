from types import SimpleNamespace

from core.keyframe_selector import KeyframeSelector, KeyframeInfo
from core.stage_temp_store import StageTempStore


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


def test_colmap_stage2_relaxed_disables_stage0_stage3(monkeypatch, tmp_path):
    order = []

    class _Loader:
        def __init__(self):
            self.is_paired = False
            self.is_stereo = False
            self.rig_type = "monocular"
            self._meta = SimpleNamespace(frame_count=30, fps=30.0, rig_type="monocular", rig_calibration=None)

        def get_metadata(self):
            return self._meta

    def fake_stage1(self, _video_loader, _metadata, _progress_cb):
        order.append("1")
        self.stage1_quality_records = [{"frame_index": 5, "timestamp": 0.166, "quality": 0.8, "is_pass": True, "drop_reason": "pass"}]
        return [{"frame_idx": 5, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}}]

    def fake_stage0(self, _video_loader, _metadata, _progress_cb, _frame_log_cb=None):
        order.append("0")
        return {}

    def fake_stage2(self, _video_loader, _metadata, stage1_candidates, _progress_cb, _frame_log_cb=None, _stage0_metrics=None):
        order.append("2")
        idx = int(stage1_candidates[0]["frame_idx"])
        return [KeyframeInfo(frame_index=idx, timestamp=idx / 30.0, combined_score=0.8)], []

    def fake_stage3(self, metadata, stage2_candidates, stage2_final, stage2_records, stage0_metrics, video_loader):
        order.append("3")
        return list(stage2_candidates)

    selector = KeyframeSelector(
        config={
            "pose_backend": "colmap",
            "colmap_keyframe_policy": "stage2_relaxed",
            "colmap_keyframe_target_min": 1,
            "colmap_keyframe_target_max": 5,
        }
    )
    monkeypatch.setattr(selector, "_stage1_fast_filter", fake_stage1.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage0_lightweight_motion_scan", fake_stage0.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage2_precise_evaluation", fake_stage2.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage3_refine_with_trajectory", fake_stage3.__get__(selector, KeyframeSelector))

    store = StageTempStore("run-colmap-stage2-relaxed", root_dir=tmp_path)
    monkeypatch.setattr(store, "cleanup_on_success", lambda: None)
    out = selector.select_keyframes(_Loader(), stage_temp_store=store)

    assert len(out) == 1
    assert order == ["1", "2"]
    assert selector.last_selection_runtime["policy"] == "stage2_relaxed"
    assert selector.last_selection_runtime["effective_stage_plan"] == "Stage1->Stage2(relaxed)"


def test_apply_nms_uses_configured_time_window():
    selector = KeyframeSelector(config={"nms_time_window": 0.1})
    cands = [
        KeyframeInfo(frame_index=10, timestamp=1.0, combined_score=0.5),
        KeyframeInfo(frame_index=12, timestamp=1.15, combined_score=0.4),
    ]
    selected = selector._apply_nms(cands)
    assert len(selected) == 2


def test_colmap_retarget_downsample_and_supplement():
    selector = KeyframeSelector()
    stage1_candidates = [{"frame_idx": i, "quality_scores": {"quality": 0.6}} for i in range(0, 300)]

    many = [
        KeyframeInfo(frame_index=i, timestamp=i / 30.0, combined_score=0.7)
        for i in range(0, 300, 1)
    ]
    downsampled, down_info = selector._retarget_keyframes_for_colmap(
        many,
        stage1_candidates,
        total_frames=300,
        fps=30.0,
        target_mode="fixed",
        target_min=120,
        target_max=240,
    )
    assert len(downsampled) == 240
    assert "downsample_to_max" in down_info["retarget_reason"]

    few = [
        KeyframeInfo(frame_index=i * 10, timestamp=(i * 10) / 30.0, combined_score=0.7)
        for i in range(5)
    ]
    supplemented, sup_info = selector._retarget_keyframes_for_colmap(
        few,
        stage1_candidates,
        total_frames=300,
        fps=30.0,
        target_mode="fixed",
        target_min=30,
        target_max=240,
    )
    assert len(supplemented) >= 30
    assert "supplement_to_min" in sup_info["retarget_reason"]


def test_colmap_retarget_auto_mode_stays_within_bounds():
    selector = KeyframeSelector()
    stage1_candidates = [{"frame_idx": i, "quality_scores": {"quality": 0.6}} for i in range(0, 500, 2)]
    keyframes = [
        KeyframeInfo(frame_index=i, timestamp=i / 30.0, combined_score=0.7)
        for i in range(0, 500, 4)
    ]
    out, info = selector._retarget_keyframes_for_colmap(
        keyframes,
        stage1_candidates,
        total_frames=500,
        fps=30.0,
        target_mode="auto",
        target_min=120,
        target_max=240,
        stage2_records=[],
    )
    assert info["target_mode"] == "auto"
    assert info["effective_target_min"] >= 120
    assert info["effective_target_max"] <= 240
    assert info["effective_target_min"] <= len(out) <= info["effective_target_max"]
