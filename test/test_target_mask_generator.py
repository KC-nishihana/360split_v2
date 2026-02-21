import numpy as np

from processing.mask_processor import MaskProcessor
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


def test_create_fisheye_valid_mask_shape_dtype_and_range():
    processor = MaskProcessor()
    mask = processor.create_fisheye_valid_mask(200, 100, radius_ratio=0.94)

    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[50, 100] == 255
    assert mask[0, 0] == 0


def test_create_fisheye_valid_mask_offset_changes_center():
    processor = MaskProcessor()
    centered = processor.create_fisheye_valid_mask(240, 120, radius_ratio=0.90, offset_x=0, offset_y=0)
    shifted = processor.create_fisheye_valid_mask(240, 120, radius_ratio=0.90, offset_x=20, offset_y=0)

    assert centered[60, 120] == 255
    assert shifted[60, 120] == 255
    assert centered[60, 179] == 0
    assert shifted[60, 179] == 255


def test_apply_valid_region_mask_blacks_outside_and_keeps_inside():
    processor = MaskProcessor()
    frame = np.full((80, 120, 3), 120, dtype=np.uint8)
    valid_mask = processor.create_fisheye_valid_mask(120, 80, radius_ratio=0.94)

    masked = processor.apply_valid_region_mask(frame, valid_mask, fill_value=0)

    assert np.array_equal(masked[40, 60], frame[40, 60])
    assert np.array_equal(masked[0, 0], np.array([0, 0, 0], dtype=np.uint8))
