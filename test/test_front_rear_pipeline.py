import numpy as np

from core.keyframe_selector import KeyframeSelector
from core.video_loader import VideoMetadata
from processing.fisheye_rig import FisheyeRigProcessor


class DummyFrontRearLoader:
    def __init__(self):
        self._metadata = VideoMetadata(
            fps=1.0,
            frame_count=3,
            width=64,
            height=32,
            duration=3.0,
            codec="dummy",
            rig_type="front_rear"
        )
        self.is_paired = True
        self.is_stereo = False
        self.rig_type = "front_rear"
        self.frames = {
            0: (
                np.full((32, 64, 3), 80, dtype=np.uint8),
                np.full((32, 64, 3), 90, dtype=np.uint8),
            ),
            1: (
                np.full((32, 64, 3), 100, dtype=np.uint8),
                np.full((32, 64, 3), 105, dtype=np.uint8),
            ),
            2: (
                np.full((32, 64, 3), 130, dtype=np.uint8),
                np.full((32, 64, 3), 140, dtype=np.uint8),
            ),
        }

    def get_metadata(self):
        return self._metadata

    def get_frame_pair(self, idx):
        return self.frames.get(idx, (None, None))

    def get_frame(self, idx):
        return self.frames[idx][0]


def test_pair_quality_requires_both_lenses():
    selector = KeyframeSelector(config={
        "laplacian_threshold": 50.0,
        "motion_blur_threshold": 0.4,
        "exposure_threshold": 0.4,
    })

    def fake_eval(frame, beta=5.0):
        mean_v = float(frame.mean())
        return {
            "sharpness": 100.0,
            "motion_blur": 0.2,
            "exposure": 0.8 if mean_v > 80 else 0.2,
            "softmax_depth": 0.5,
        }

    selector.quality_evaluator.evaluate = fake_eval

    a = np.full((10, 10, 3), 120, dtype=np.uint8)
    b = np.full((10, 10, 3), 60, dtype=np.uint8)
    score = selector._compute_quality_score_pair(a, b)
    assert score["passes_threshold"] is False


def test_pair_optical_flow_uses_max_of_two_lenses():
    selector = KeyframeSelector(config={
        "min_keyframe_interval": 1,
        "ssim_change_threshold": 0.99,
        "enable_rig_stitching": False,
        "pair_motion_aggregation": "max",
    })
    loader = DummyFrontRearLoader()

    selector.quality_evaluator.evaluate = lambda frame, beta=5.0: {
        "sharpness": 200.0,
        "motion_blur": 0.1,
        "exposure": 0.8,
        "softmax_depth": 0.5,
    }

    selector.geometric_evaluator.evaluate = lambda *args, **kwargs: {
        "gric": 0.6,
        "feature_distribution_1": 0.5,
        "feature_distribution_2": 0.5,
        "feature_match_count": 80,
        "ray_dispersion": 0.5,
    }

    selector.adaptive_selector.evaluate = lambda *args, **kwargs: {
        "ssim": 0.2,
        "optical_flow": 1.0,
        "momentum": 0.0,
    }

    call_state = {"i": 0}

    def fake_flow(*args, **kwargs):
        call_state["i"] += 1
        return 2.0 if call_state["i"] % 2 == 1 else 7.5

    selector.adaptive_selector.compute_optical_flow_magnitude = fake_flow

    keyframes = selector.select_keyframes(loader)
    assert len(keyframes) >= 2
    assert keyframes[1].adaptive_scores["optical_flow"] == 7.5
    assert keyframes[1].adaptive_scores["optical_flow_lens_a"] == 2.0
    assert keyframes[1].adaptive_scores["optical_flow_lens_b"] == 7.5


def test_fisheye_stitch_and_feature_extraction_runs():
    processor = FisheyeRigProcessor()
    front = np.zeros((120, 240, 3), dtype=np.uint8)
    rear = np.zeros((120, 240, 3), dtype=np.uint8)
    front[:, :120] = 255
    rear[:, 120:] = 180

    stitched, seam = processor.stitch_to_equirect(front, rear, None, (256, 128))
    feats = processor.extract_360_features(stitched, seam_mask=seam, method="orb")

    assert stitched.shape == (128, 256, 3)
    assert seam.shape == (128, 256)
    assert feats.seam_keypoint_count >= 0
