from pathlib import Path

import cv2
import numpy as np

from main import _write_colmap_bundle


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_write_colmap_bundle_creates_expected_files(tmp_path):
    image_a = tmp_path / "keyframe_000001.png"
    image_b = tmp_path / "images" / "L" / "keyframe_000002_L.png"
    _write_image(image_a)
    _write_image(image_b)

    out_dir = tmp_path / "out"
    colmap_dir = _write_colmap_bundle(
        output_dir=out_dir,
        image_paths=[image_a, image_b],
        width=200,
        height=120,
        calibration_runtime={},
    )

    assert colmap_dir is not None
    assert (colmap_dir / "cameras.txt").exists()
    assert (colmap_dir / "image_list.txt").exists()
    images = list((colmap_dir / "images").glob("*"))
    assert len(images) == 2
