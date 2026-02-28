from types import SimpleNamespace

import cv2
import numpy as np

from core.keyframe_selector import KeyframeInfo, KeyframeSelector, Stage2FrameRecord


class _DummyLoader:
    def __init__(self, frames):
        self.frames = frames
        self.is_paired = False

    def get_frame(self, idx: int):
        if 0 <= idx < len(self.frames):
            return self.frames[idx]
        return None


class _FakeVOEstimator:
    def estimate(self, *_args, **_kwargs):
        return SimpleNamespace(
            vo_valid=True,
            step_proxy=5.0,
            t_dir=[1.0, 0.0, 0.0],
            r_rel_q_wxyz=[1.0, 0.0, 0.0, 0.0],
            rotation_delta_deg=3.0,
            translation_delta_rel=1.0,
            match_count=60,
            tracked_count=90,
            inlier_ratio=0.72,
            vo_confidence=0.81,
            feature_uniformity=0.66,
            track_sufficiency=1.0,
            pose_plausibility=0.88,
            essential_method_used="ransac",
        )


class _FailVOEstimator:
    def estimate(self, *_args, **_kwargs):
        return SimpleNamespace(
            vo_valid=False,
            step_proxy=0.0,
            t_dir=[0.0, 0.0, 0.0],
            r_rel_q_wxyz=[1.0, 0.0, 0.0, 0.0],
            rotation_delta_deg=0.0,
            translation_delta_rel=0.0,
            match_count=0,
            tracked_count=0,
            inlier_ratio=0.0,
            vo_confidence=0.0,
            feature_uniformity=0.0,
            track_sufficiency=0.0,
            pose_plausibility=0.0,
            essential_method_used="none",
        )


def _make_frames(n: int = 8):
    h, w = 240, 320
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    out = []
    for i in range(n):
        mat = np.float32([[1.0, 0.0, float(i * 2)], [0.0, 1.0, float(i)]])
        out.append(cv2.warpAffine(base, mat, (w, h), flags=cv2.INTER_LINEAR))
    return out


def test_stage0_and_stage3_emit_vo_confidence_metrics():
    frames = _make_frames(10)
    loader = _DummyLoader(frames)
    selector = KeyframeSelector(
        config={
            "VO_ENABLED": True,
            "VO_FRAME_SUBSAMPLE": 3,
            "VO_ADAPTIVE_SUBSAMPLE": True,
            "VO_SUBSAMPLE_MIN": 1,
        }
    )
    selector.vo_estimator = _FakeVOEstimator()
    selector._resolve_vo_runtime = lambda **_kwargs: (True, None, "enabled")

    metadata = SimpleNamespace(frame_count=len(frames), rig_calibration=None, fps=30.0)
    stage0 = selector._stage0_lightweight_motion_scan(loader, metadata, progress_callback=None)
    assert stage0
    for m in stage0.values():
        assert 0.0 <= float(m.get("vo_confidence", 0.0)) <= 1.0

    candidates = [
        KeyframeInfo(frame_index=1, timestamp=0.03, combined_score=0.6),
        KeyframeInfo(frame_index=4, timestamp=0.13, combined_score=0.7),
        KeyframeInfo(frame_index=7, timestamp=0.23, combined_score=0.65),
    ]
    records = [
        Stage2FrameRecord(
            frame_index=i,
            frame=None,
            quality_scores={},
            geometric_scores={},
            adaptive_scores={},
            metrics={
                "combined_stage2": 0.6,
                "combined_stage3": 0.6,
                "trajectory_consistency": 0.5,
                "vo_confidence": 0.0,
            },
        )
        for i in [1, 4, 7]
    ]

    refined = selector._stage3_refine_with_trajectory(
        metadata=metadata,
        stage2_candidates=candidates,
        stage2_final=list(candidates),
        stage2_records=records,
        stage0_metrics=stage0,
        video_loader=loader,
    )
    assert refined
    for rec in records:
        assert 0.0 <= float(rec.metrics.get("vo_confidence", 0.0)) <= 1.0
        assert "trajectory_consistency_effective" in rec.metrics


def test_stage3_disables_traj_weight_when_vo_unreliable():
    frames = _make_frames(8)
    loader = _DummyLoader(frames)
    selector = KeyframeSelector(
        config={
            "VO_ENABLED": True,
            "STAGE3_DISABLE_TRAJ_WHEN_VO_UNRELIABLE": True,
            "STAGE3_VO_VALID_RATIO_THRESHOLD": 0.8,
            "VO_FRAME_SUBSAMPLE": 1,
        }
    )
    selector.vo_estimator = _FailVOEstimator()
    selector._resolve_vo_runtime = lambda **_kwargs: (True, None, "enabled")

    metadata = SimpleNamespace(frame_count=len(frames), rig_calibration=None, fps=30.0)
    candidates = [
        KeyframeInfo(frame_index=1, timestamp=0.03, combined_score=0.6),
        KeyframeInfo(frame_index=3, timestamp=0.10, combined_score=0.7),
    ]
    records = [
        Stage2FrameRecord(frame_index=1, frame=None, quality_scores={}, geometric_scores={}, adaptive_scores={}, metrics={}),
        Stage2FrameRecord(frame_index=3, frame=None, quality_scores={}, geometric_scores={}, adaptive_scores={}, metrics={}),
    ]

    refined = selector._stage3_refine_with_trajectory(
        metadata=metadata,
        stage2_candidates=candidates,
        stage2_final=list(candidates),
        stage2_records=records,
        stage0_metrics={1: {"motion_risk": 0.0}, 3: {"motion_risk": 0.0}},
        video_loader=loader,
    )
    assert refined
    assert all(float(kf.stage3_scores.get("stage3_w_traj", -1.0)) == 0.0 for kf in refined)
