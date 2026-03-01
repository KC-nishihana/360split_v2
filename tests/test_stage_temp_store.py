from types import SimpleNamespace

from core.keyframe_selector import KeyframeInfo, KeyframeSelector, Stage2FrameRecord
from core.stage_temp_store import StageTempStore


class _DummyLoader:
    def __init__(self, frame_count: int = 10, fps: float = 30.0):
        self.is_paired = False
        self.is_stereo = False
        self.rig_type = "monocular"
        self._meta = SimpleNamespace(frame_count=frame_count, fps=fps, rig_type="monocular", rig_calibration=None)

    def get_metadata(self):
        return self._meta


def test_stage_temp_store_roundtrip_and_cleanup(tmp_path):
    store = StageTempStore("run-roundtrip", root_dir=tmp_path)

    cands = [{"frame_idx": 1, "quality_scores": {"quality": 0.8}}]
    recs = [{"frame_index": 1, "quality": 0.8, "is_pass": True}]
    store.save_stage1(cands, recs)
    store.save_stage1_effective(cands)
    lc, lr = store.load_stage1()
    assert lc == cands
    assert lr == recs
    assert store.load_stage1_effective() == cands

    s0 = {1: {"motion_risk": 0.2, "vo_status_reason": "ok"}}
    store.save_stage0(s0)
    assert store.load_stage0() == s0

    s2c = [{"frame_index": 1, "combined_score": 0.9}]
    s2r = [{"frame_index": 1, "metrics": {"combined_stage2": 0.9}}]
    store.save_stage2(s2c, s2r)
    assert store.load_stage2() == (s2c, s2r)

    s3 = [{"frame_index": 1, "combined_score": 0.9}]
    store.save_stage3(s3)
    assert store.load_stage3() == s3

    store.mark_stage_done("1", files={"candidates": "x"}, counts={"records": 1})
    assert store.manifest_path.exists()

    run_dir = store.run_dir
    store.cleanup_on_success()
    assert not run_dir.exists()


def test_stage_temp_store_mark_failed_keeps_artifacts(tmp_path):
    store = StageTempStore("run-failed", root_dir=tmp_path)
    store.save_stage1([], [])
    store.mark_failed("2", "boom")

    assert store.run_dir.exists()
    manifest = store._read_manifest()
    assert manifest.get("failed") is True
    assert manifest.get("failed_stage") == "2"
    assert "boom" in str(manifest.get("error", ""))


def test_select_keyframes_stage_order_and_temp_outputs(monkeypatch, tmp_path):
    order = []

    def fake_stage1(self, _video_loader, _metadata, _progress_cb):
        order.append("1")
        self.stage1_quality_records = [
            {"frame_index": 2, "timestamp": 0.0, "quality": 0.8, "is_pass": True, "drop_reason": "pass"}
        ]
        return [{"frame_idx": 2, "quality_scores": {"quality": 0.8, "sharpness": 100.0, "exposure": 0.8}}]

    def fake_stage0(self, _video_loader, _metadata, _progress_cb, _frame_log_cb=None):
        order.append("0")
        return {2: {"motion_risk": 0.1, "vo_status_reason": "ok"}}

    def fake_stage2(self, _video_loader, _metadata, stage1_candidates, _progress_cb, _frame_log_cb=None, _stage0_metrics=None):
        order.append("2")
        idx = int(stage1_candidates[0]["frame_idx"])
        candidates = [
            KeyframeInfo(
                frame_index=idx,
                timestamp=idx / 30.0,
                quality_scores={"quality": 0.8, "sharpness": 100.0, "exposure": 0.8},
                geometric_scores={"gric": 0.7},
                adaptive_scores={"ssim": 0.7},
                combined_score=0.8,
            )
        ]
        records = [
            Stage2FrameRecord(
                frame_index=idx,
                frame=None,
                quality_scores={"quality": 0.8},
                geometric_scores={"gric": 0.7},
                adaptive_scores={"ssim": 0.7},
                metrics={"combined_stage2": 0.8, "combined_stage3": 0.8},
                is_candidate=True,
            )
        ]
        return candidates, records

    def fake_stage3(self, metadata, stage2_candidates, stage2_final, stage2_records, stage0_metrics, video_loader):
        order.append("3")
        return list(stage2_candidates)

    selector = KeyframeSelector()
    monkeypatch.setattr(selector, "_stage1_fast_filter", fake_stage1.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage0_lightweight_motion_scan", fake_stage0.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage2_precise_evaluation", fake_stage2.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage3_refine_with_trajectory", fake_stage3.__get__(selector, KeyframeSelector))

    store = StageTempStore("run-order", root_dir=tmp_path)
    monkeypatch.setattr(store, "cleanup_on_success", lambda: None)

    out = selector.select_keyframes(_DummyLoader(), stage_temp_store=store)

    assert len(out) == 1
    assert order == ["1", "0", "2", "3"]
    assert (store.run_dir / store.STAGE1_CANDIDATES_FILE).exists()
    assert (store.run_dir / store.STAGE1_RECORDS_FILE).exists()
    assert (store.run_dir / store.STAGE0_METRICS_FILE).exists()
    assert (store.run_dir / store.STAGE2_CANDIDATES_FILE).exists()
    assert (store.run_dir / store.STAGE2_RECORDS_FILE).exists()
    assert (store.run_dir / store.STAGE3_KEYFRAMES_FILE).exists()


def test_select_keyframes_reuses_stage1_temp(monkeypatch, tmp_path):
    order = []

    def should_not_run_stage1(*_args, **_kwargs):
        raise AssertionError("Stage1 should not run when temp artifact exists")

    def fake_stage0(self, _video_loader, _metadata, _progress_cb, _frame_log_cb=None):
        order.append("0")
        return {3: {"motion_risk": 0.2, "vo_status_reason": "ok"}}

    def fake_stage2(self, _video_loader, _metadata, stage1_candidates, _progress_cb, _frame_log_cb=None, _stage0_metrics=None):
        order.append("2")
        idx = int(stage1_candidates[0]["frame_idx"])
        return [
            KeyframeInfo(frame_index=idx, timestamp=0.1, quality_scores={}, geometric_scores={}, adaptive_scores={}, combined_score=0.6)
        ], [
            Stage2FrameRecord(frame_index=idx, frame=None, quality_scores={}, geometric_scores={}, adaptive_scores={}, metrics={"combined_stage2": 0.6, "combined_stage3": 0.6})
        ]

    def fake_stage3(self, metadata, stage2_candidates, stage2_final, stage2_records, stage0_metrics, video_loader):
        order.append("3")
        return list(stage2_candidates)

    selector = KeyframeSelector(config={"resume_enabled": True})
    monkeypatch.setattr(selector, "_stage1_fast_filter", should_not_run_stage1)
    monkeypatch.setattr(selector, "_stage0_lightweight_motion_scan", fake_stage0.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage2_precise_evaluation", fake_stage2.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage3_refine_with_trajectory", fake_stage3.__get__(selector, KeyframeSelector))

    store = StageTempStore("run-reuse", root_dir=tmp_path)
    store.save_stage1(
        [{"frame_idx": 3, "quality_scores": {"quality": 0.7, "sharpness": 80.0, "exposure": 0.7}}],
        [{"frame_index": 3, "timestamp": 0.1, "quality": 0.7, "is_pass": True, "drop_reason": "pass"}],
    )
    monkeypatch.setattr(store, "cleanup_on_success", lambda: None)

    out = selector.select_keyframes(_DummyLoader(), stage_temp_store=store)
    assert len(out) == 1
    assert order == ["0", "2", "3"]


def test_select_keyframes_reuses_stage0_temp_when_resume(monkeypatch, tmp_path):
    flags = {"stage0_called": False}

    def fake_stage1(self, _video_loader, _metadata, _progress_cb):
        self.stage1_quality_records = [{"frame_index": 2, "timestamp": 0.0, "quality": 0.8, "is_pass": True, "drop_reason": "pass"}]
        return [{"frame_idx": 2, "quality_scores": {"quality": 0.8}}]

    def should_not_run_stage0(*_args, **_kwargs):
        flags["stage0_called"] = True
        raise AssertionError("Stage0 should not run when temp artifact exists with resume")

    def fake_stage2(self, _video_loader, _metadata, stage1_candidates, _progress_cb, _frame_log_cb=None, _stage0_metrics=None):
        idx = int(stage1_candidates[0]["frame_idx"])
        return [KeyframeInfo(frame_index=idx, timestamp=0.1, combined_score=0.6)], [
            Stage2FrameRecord(frame_index=idx, frame=None, quality_scores={}, geometric_scores={}, adaptive_scores={}, metrics={"combined_stage2": 0.6, "combined_stage3": 0.6})
        ]

    def fake_stage3(self, metadata, stage2_candidates, stage2_final, stage2_records, stage0_metrics, video_loader):
        return list(stage2_candidates)

    selector = KeyframeSelector(config={"resume_enabled": True})
    monkeypatch.setattr(selector, "_stage1_fast_filter", fake_stage1.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage0_lightweight_motion_scan", should_not_run_stage0)
    monkeypatch.setattr(selector, "_stage2_precise_evaluation", fake_stage2.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage3_refine_with_trajectory", fake_stage3.__get__(selector, KeyframeSelector))

    store = StageTempStore("run-reuse-stage0", root_dir=tmp_path)
    store.save_stage0({2: {"motion_risk": 0.1, "vo_status_reason": "ok"}})
    monkeypatch.setattr(store, "cleanup_on_success", lambda: None)

    out = selector.select_keyframes(_DummyLoader(), stage_temp_store=store)
    assert len(out) == 1
    assert flags["stage0_called"] is False


def test_select_keyframes_stage1_empty_exits_before_stage0(monkeypatch, tmp_path):
    flags = {"stage0_called": False}

    def fake_stage1(self, _video_loader, _metadata, _progress_cb):
        self.stage1_quality_records = []
        return []

    def fake_stage0(*_args, **_kwargs):
        flags["stage0_called"] = True
        return {}

    selector = KeyframeSelector()
    monkeypatch.setattr(selector, "_stage1_fast_filter", fake_stage1.__get__(selector, KeyframeSelector))
    monkeypatch.setattr(selector, "_stage0_lightweight_motion_scan", fake_stage0)

    store = StageTempStore("run-empty", root_dir=tmp_path)
    monkeypatch.setattr(store, "cleanup_on_success", lambda: None)

    out = selector.select_keyframes(_DummyLoader(), stage_temp_store=store)
    assert out == []
    assert flags["stage0_called"] is False
