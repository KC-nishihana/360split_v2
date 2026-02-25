import numpy as np
import cv2
import pytest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def gradient_bgr() -> np.ndarray:
    h, w = 240, 320
    row = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def checkerboard_bgr() -> np.ndarray:
    h, w, cell = 240, 320, 20
    yy, xx = np.indices((h, w))
    img = (((xx // cell) + (yy // cell)) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def translated_pair(checkerboard_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = checkerboard_bgr.shape[:2]
    M = np.float32([[1, 0, 8], [0, 1, 5]])
    shifted = cv2.warpAffine(checkerboard_bgr, M, (w, h))
    return checkerboard_bgr, shifted


@pytest.fixture
def noise_bgr() -> np.ndarray:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    return arr
