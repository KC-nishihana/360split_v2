import numpy as np

from processing.target_mask_generator import TargetMaskGenerator


def test_detect_sky_mask_removes_isolated_bright_ground_region():
    frame = np.full((120, 160, 3), 30, dtype=np.uint8)

    # 上側の曇天空
    frame[:44, :] = (215, 215, 215)

    # 地平線付近の建物/樹木を想定した暗い帯で上下を分離
    frame[44:52, :] = (70, 70, 70)

    # 下側の明るいコンクリート領域（誤検出させたくない）
    frame[60:100, 20:140] = (210, 210, 210)

    mask = TargetMaskGenerator._detect_sky_mask(frame)

    assert mask[10, 80] == 1
    assert mask[80, 80] == 0


def test_detect_sky_mask_handles_empty_frame_shape():
    frame = np.zeros((0, 0, 3), dtype=np.uint8)
    mask = TargetMaskGenerator._detect_sky_mask(frame)
    assert mask.shape == (0, 0)
