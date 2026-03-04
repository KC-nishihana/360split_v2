from types import SimpleNamespace

from core.keyframe_selector import KeyframeSelector, KeyframeInfo, Stage2FrameRecord
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


def test_colmap_stage2_relaxed_enables_stage0_and_disables_stage3(monkeypatch, tmp_path):
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
            "colmap_pipeline_mode": "legacy",
            "colmap_keyframe_policy": "stage2_relaxed",
            "colmap_selection_profile": "legacy",
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
    assert order == ["1", "0", "2"]
    assert selector.last_selection_runtime["policy"] == "stage2_relaxed"
    assert selector.last_selection_runtime["effective_stage_plan"] == "Stage1->Stage0->Stage2(relaxed)"


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


def test_stage15_budget_reduces_dense_cluster():
    selector = KeyframeSelector()
    stage1 = []
    for i in range(100, 220):
        stage1.append({"frame_idx": i, "quality_scores": {"quality": 0.8}})
    for i in range(0, 100, 10):
        stage1.append({"frame_idx": i, "quality_scores": {"quality": 0.7}})
    stage1 = sorted(stage1, key=lambda x: x["frame_idx"])

    class _Loader:
        is_paired = False

        @staticmethod
        def get_frame(idx):
            import numpy as np
            v = int(idx % 255)
            return np.full((64, 64, 3), v, dtype=np.uint8)

    out, info, _trace = selector._run_stage15_entry_budget(
        stage1,
        total_frames=240,
        video_loader=_Loader(),
        entry_budget=60,
        min_gap=3,
        diversity_ssim_threshold=0.95,
        diversity_phash_hamming=8,
    )
    assert len(out) <= 60
    assert int(info["entry_count"]) == len(out)
    assert int(info["coverage_bins_after"]) > 0


def test_stage15_preserves_time_bin_coverage():
    selector = KeyframeSelector()
    stage1 = [{"frame_idx": i, "quality_scores": {"quality": 0.6}} for i in range(0, 240, 4)]

    class _Loader:
        is_paired = False

        @staticmethod
        def get_frame(_idx):
            import numpy as np
            return np.zeros((64, 64, 3), dtype=np.uint8)

    out, info, _trace = selector._run_stage15_entry_budget(
        stage1,
        total_frames=240,
        video_loader=_Loader(),
        entry_budget=48,
        min_gap=0,
        diversity_ssim_threshold=1.0,
        diversity_phash_hamming=0,
    )
    assert len(out) <= 48
    assert int(info["coverage_bins_after"]) >= 20


def test_final_soft_auto_no_forced_supplement():
    selector = KeyframeSelector(config={"quality_threshold": 0.9})
    base = [
        KeyframeInfo(frame_index=i * 20, timestamp=(i * 20) / 30.0, combined_score=0.7)
        for i in range(3)
    ]
    stage1_candidates = [{"frame_idx": i * 10, "quality_scores": {"quality": 0.2}} for i in range(30)]
    out, info = selector._retarget_keyframes_for_colmap(
        base,
        stage1_candidates,
        total_frames=300,
        fps=30.0,
        target_mode="fixed",
        target_min=120,
        target_max=240,
        final_target_policy="soft_auto",
        final_soft_min=10,
        final_soft_max=50,
        no_supplement_on_low_quality=True,
    )
    assert len(out) == len(base)
    assert info["final_reject_reason"] == "under_target_quality_guard"


def test_no_vo_profile_ignores_vo_metrics():
    selector = KeyframeSelector(config={"pose_backend": "colmap", "colmap_selection_profile": "no_vo_coverage"})
    runtime = selector._resolve_colmap_keyframe_runtime()
    assert runtime["selection_profile"] == "no_vo_coverage"
    assert runtime["colmap_motion_aware_selection"] is False
    assert runtime["force_stage0_off"] is True


def test_build_cumulative_motion_map_interpolation():
    selector = KeyframeSelector()
    stage0_metrics = {
        0: {"flow_mag_light": 0.0},
        10: {"flow_mag_light": 2.0},
        20: {"flow_mag_light": 4.0},
    }
    motion_map, info = selector._build_cumulative_motion_map(stage0_metrics, total_frames=21)
    assert len(motion_map) == 21
    assert info["sample_count"] == 3
    assert info["motion_median_step"] > 0.0
    assert motion_map[0] <= motion_map[5] <= motion_map[10] <= motion_map[20]


def test_apply_nms_with_motion_window():
    selector = KeyframeSelector()
    cands = [
        KeyframeInfo(frame_index=10, timestamp=1.0, combined_score=0.9),
        KeyframeInfo(frame_index=100, timestamp=10.0, combined_score=0.8),
    ]
    selected = selector._apply_nms(
        cands,
        time_window=0.05,
        cumulative_motion_map={10: 5.0, 100: 5.1},
        motion_window=0.5,
        motion_aware_selection=True,
    )
    assert len(selected) == 1
    assert selected[0].frame_index == 10


def test_motion_distributed_downsample_keeps_edges():
    selector = KeyframeSelector()
    keyframes = [KeyframeInfo(frame_index=i, timestamp=i / 30.0, combined_score=0.7) for i in [0, 10, 20, 30, 40, 50]]
    down = selector._motion_distributed_downsample(
        keyframes,
        target_count=3,
        cumulative_motion_map={0: 0.0, 10: 1.0, 20: 2.0, 30: 4.0, 40: 7.0, 50: 10.0},
    )
    idx = [k.frame_index for k in down]
    assert len(idx) == 3
    assert idx[0] == 0
    assert idx[-1] == 50


def test_retarget_supplement_motion_distance_priority():
    selector = KeyframeSelector()
    existing = [
        KeyframeInfo(frame_index=0, timestamp=0.0, combined_score=0.8),
        KeyframeInfo(frame_index=100, timestamp=100 / 30.0, combined_score=0.7),
    ]
    stage1_candidates = [
        {"frame_idx": 0, "quality_scores": {"quality": 0.7}},
        {"frame_idx": 50, "quality_scores": {"quality": 0.7}},
        {"frame_idx": 90, "quality_scores": {"quality": 0.7}},
        {"frame_idx": 100, "quality_scores": {"quality": 0.7}},
    ]
    out, _info = selector._retarget_keyframes_for_colmap(
        existing,
        stage1_candidates,
        total_frames=120,
        fps=30.0,
        target_mode="fixed",
        target_min=3,
        target_max=10,
        cumulative_motion_map={0: 0.0, 50: 99.0, 90: 50.0, 100: 100.0},
    )
    idx = [k.frame_index for k in out]
    assert 90 in idx
    assert 50 not in idx


def test_stage2_drop_reason_annotation_min_interval():
    import numpy as np

    class _Cap:
        def __init__(self, frames):
            self.frames = frames
            self.pos = 0

        def set(self, _prop, value):
            self.pos = int(value)

        def read(self):
            if 0 <= self.pos < len(self.frames):
                out = self.frames[self.pos]
                self.pos += 1
                return True, out.copy()
            return False, None

        def release(self):
            return None

    class _Loader:
        is_paired = False
        is_stereo = False
        rig_type = "monocular"
        _video_path = "dummy.mp4"

    selector = KeyframeSelector(config={"MIN_KEYFRAME_INTERVAL": 5, "ENABLE_DYNAMIC_MASK_REMOVAL": False})
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    selector._open_independent_capture = lambda _path: _Cap(frames)

    metadata = SimpleNamespace(frame_count=4, fps=30.0, rig_calibration=None)
    stage1_candidates = [
        {"frame_idx": 0, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}},
        {"frame_idx": 1, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}},
    ]
    _cands, records = selector._stage2_precise_evaluation(
        _Loader(),
        metadata,
        stage1_candidates,
        progress_callback=None,
    )
    reasons = [r.drop_reason for r in records]
    assert "selected" in reasons
    assert "min_interval" in reasons


def test_colmap_minimal_runtime_defaults():
    selector = KeyframeSelector(config={"pose_backend": "colmap"})
    runtime = selector._resolve_colmap_keyframe_runtime()
    assert runtime["pipeline_mode"] == "minimal_v1"
    assert runtime["minimal_mode"] is True
    assert runtime["force_stage0_off"] is True
    assert runtime["force_stage3_off"] is True
    assert runtime["colmap_shortcut"] is False
    assert runtime["effective_stage_plan"] == "Stage1->Stage2(minimal_v1)"


def test_stage2_minimal_selects_all_read_success():
    import numpy as np

    class _Cap:
        def __init__(self, frames):
            self.frames = frames
            self.pos = 0

        def set(self, _prop, value):
            self.pos = int(value)

        def read(self):
            if 0 <= self.pos < len(self.frames):
                out = self.frames[self.pos]
                self.pos += 1
                return True, out.copy()
            return False, None

        def release(self):
            return None

    class _Loader:
        is_paired = False
        is_stereo = False
        rig_type = "monocular"
        _video_path = "dummy.mp4"

    selector = KeyframeSelector(config={"MIN_KEYFRAME_INTERVAL": 99, "ENABLE_DYNAMIC_MASK_REMOVAL": True})
    frames = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 16, dtype=np.uint8)]
    selector._open_independent_capture = lambda _path: _Cap(frames)
    metadata = SimpleNamespace(frame_count=6, fps=30.0, rig_calibration=None)
    stage1_candidates = [
        {"frame_idx": 0, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}},
        {"frame_idx": 1, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}},
        {"frame_idx": 5, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}},
    ]
    out, records = selector._stage2_minimal_motion_only_evaluation(
        _Loader(),
        metadata,
        stage1_candidates,
        progress_callback=None,
    )
    assert [k.frame_index for k in out] == [0, 1]
    assert any(r.drop_reason == "read_fail" for r in records)
    assert all(r.drop_reason in {"selected", "read_fail"} for r in records)
    assert "optical_flow" in out[1].adaptive_scores


def _run_paired_stage1_with_mocked_scores(monkeypatch, scores, sky_ratios, config_extra=None):
    import numpy as np
    import core.keyframe_selector as ks_mod

    class _Loader:
        is_paired = True

        @staticmethod
        def get_frame_pair(_idx):
            return (
                np.full((32, 32, 3), 80, dtype=np.uint8),
                np.full((32, 32, 3), 120, dtype=np.uint8),
            )

    selector = KeyframeSelector(
        config={
            "SAMPLE_INTERVAL": 1,
            "QUALITY_FILTER_ENABLED": True,
            "ENABLE_FISHEYE_BORDER_MASK": False,
            **(config_extra or {}),
        }
    )
    selector._open_independent_pair_captures = lambda _loader: (None, None)
    score_iter = iter(scores)
    sky_iter = iter(sky_ratios)
    monkeypatch.setattr(ks_mod, "compose_quality", lambda *_args, **_kwargs: float(next(score_iter)))
    monkeypatch.setattr(ks_mod, "apply_abs_guard", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(selector, "_estimate_sky_ratio", lambda *_args, **_kwargs: float(next(sky_iter)))
    selector._stage1_fast_filter(_Loader(), SimpleNamespace(frame_count=1, fps=30.0), progress_callback=None)
    return selector


def test_stage1_lr_strict_min_keeps_legacy_behavior(monkeypatch):
    selector = _run_paired_stage1_with_mocked_scores(
        monkeypatch,
        scores=[0.80, 0.30],
        sky_ratios=[0.75, 0.10],
        config_extra={"STAGE1_LR_MERGE_MODE": "strict_min"},
    )
    assert len(selector.stage1_quality_records) == 1
    rec = selector.stage1_quality_records[0]
    assert rec["is_pass"] is False
    assert rec["drop_reason"] == "quality_below_threshold"
    assert rec["quality_merge_strategy"] == "strict_min"
    assert rec["lr_merge_mode_applied"] == "strict_min"


def test_stage1_lr_asymmetric_mode_selects_high_quality_side(monkeypatch):
    selector = _run_paired_stage1_with_mocked_scores(
        monkeypatch,
        scores=[0.80, 0.40],
        sky_ratios=[0.75, 0.10],
        config_extra={"STAGE1_LR_MERGE_MODE": "asymmetric_sky_v1"},
    )
    rec = selector.stage1_quality_records[0]
    assert rec["is_pass"] is True
    assert rec["quality"] == 0.80
    assert rec["quality_merge_strategy"] == "asymmetric_max_with_weak_floor"
    assert rec["lr_merge_mode_applied"] == "asymmetric_sky_v1"
    assert rec["lr_asym_eligible"] is True


def test_stage1_lr_asymmetric_mode_rejects_below_weak_floor(monkeypatch):
    selector = _run_paired_stage1_with_mocked_scores(
        monkeypatch,
        scores=[0.80, 0.34],
        sky_ratios=[0.75, 0.10],
        config_extra={
            "STAGE1_LR_MERGE_MODE": "asymmetric_sky_v1",
            "STAGE1_LR_ASYM_WEAK_FLOOR": 0.35,
        },
    )
    rec = selector.stage1_quality_records[0]
    assert rec["is_pass"] is False
    assert rec["drop_reason"] == "lr_weak_floor"
    assert rec["quality_merge_strategy"] == "asymmetric_max_with_weak_floor"
    assert rec["lr_asym_eligible"] is True


def test_stage1_lr_auto_relax_when_sky_threshold_unreachable(monkeypatch):
    selector = _run_paired_stage1_with_mocked_scores(
        monkeypatch,
        scores=[0.80, 0.31],
        sky_ratios=[0.52, 0.22],
        config_extra={
            "STAGE1_LR_MERGE_MODE": "asymmetric_sky_v1",
            "STAGE1_LR_SKY_RATIO_THRESHOLD": 0.55,
            "STAGE1_LR_SKY_RATIO_DIFF_THRESHOLD": 0.20,
            "STAGE1_LR_QUALITY_GAP_THRESHOLD": 0.15,
            "STAGE1_LR_ASYM_WEAK_FLOOR": 0.35,
        },
    )
    rec = selector.stage1_quality_records[0]
    assert rec["lr_auto_relaxed"] is True
    assert rec["lr_sky_ratio_threshold"] <= 0.35 + 1e-9
    assert rec["lr_weak_floor"] <= 0.30 + 1e-9
    assert rec["is_pass"] is True
    assert rec["quality_merge_strategy"] == "asymmetric_max_with_weak_floor"


def test_stage2_colmap_preview_v1_is_deterministic_and_distributed():
    selector = KeyframeSelector(config={"pose_backend": "colmap", "colmap_pipeline_mode": "minimal_v1"})
    records = []
    for idx in range(0, 2310, 5):
        flow = float((idx % 240) / 8.0)
        ssim_pair = float(max(0.2, min(0.98, 0.95 - ((idx % 180) / 360.0))))
        quality = float(max(0.1, min(0.98, 0.45 + ((idx % 120) / 300.0))))
        records.append(
            Stage2FrameRecord(
                frame_index=idx,
                frame=None,
                quality_scores={"quality": quality},
                geometric_scores={},
                adaptive_scores={"ssim_pair": ssim_pair, "optical_flow": flow},
                metrics={"flow_mag": flow},
                is_candidate=True,
                drop_reason="selected",
            )
        )
    rows_a, summary_a = selector._build_stage2_colmap_preview_v1(records, total_frames=2310)
    rows_b, summary_b = selector._build_stage2_colmap_preview_v1(records, total_frames=2310)
    idx_a = [int(r["frame_index"]) for r in rows_a]
    idx_b = [int(r["frame_index"]) for r in rows_b]
    assert idx_a == idx_b
    assert 180 <= len(idx_a) <= 800
    assert int(summary_a.get("bins_occupied", 0)) >= 20
    assert idx_a[0] <= 100
    assert idx_a[-1] >= 2100
    assert int(summary_a.get("max_gap", 9999)) <= 180
    assert summary_a == summary_b


def test_stage3_diagnostics_v1_detects_cluster():
    selector = KeyframeSelector(config={"pose_backend": "colmap", "colmap_pipeline_mode": "minimal_v1"})
    keyframes = [
        KeyframeInfo(frame_index=idx, timestamp=idx / 30.0, quality_scores={"quality": 0.25}, adaptive_scores={"ssim_pair": 0.99})
        for idx in range(1000, 1110)
    ]
    records = [
        Stage2FrameRecord(
            frame_index=idx,
            frame=None,
            quality_scores={"quality": 0.25},
            geometric_scores={},
            adaptive_scores={"ssim_pair": 0.99, "optical_flow": 0.2},
            metrics={"flow_mag": 0.2},
            is_candidate=True,
            drop_reason="selected",
        )
        for idx in range(1000, 1110)
    ]
    diag = selector._compute_stage3_diagnostics_v1(
        keyframes=keyframes,
        stage2_records=records,
        total_frames=2310,
    )
    assert bool(diag.get("cluster_alert")) is True
    assert int(diag.get("contiguous_run_max", 0)) >= 80
    assert isinstance(diag.get("alerts", []), list)
