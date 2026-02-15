import numpy as np

from processing.stitching import StitchingProcessor


def test_stitch_fast_handles_mixed_widths_without_shape_error():
    processor = StitchingProcessor()
    left = np.full((6, 10, 3), 10, dtype=np.uint8)
    right = np.full((6, 8, 3), 200, dtype=np.uint8)

    stitched = processor.stitch_fast([left, right], mode="horizontal")

    expected_width = 10 + 8 - int(10 * 0.15)
    assert stitched.shape == (6, expected_width, 3)
    # 右端は2枚目画像の非オーバーラップ領域が残る
    assert np.all(stitched[:, -3:, :] == 200)


def test_stitch_depth_aware_handles_mixed_widths_without_shape_error():
    processor = StitchingProcessor()
    left = np.full((6, 10, 3), 20, dtype=np.uint8)
    right = np.full((6, 8, 3), 180, dtype=np.uint8)

    stitched = processor.stitch_depth_aware([left, right])

    expected_width = 10 + 8 - int(10 * 0.2)
    assert stitched.shape == (6, expected_width, 3)
    assert np.all(stitched[:, -3:, :] == 180)
