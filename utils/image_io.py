"""
画像I/Oユーティリティ

Windows環境で日本語など非ASCII文字を含むパスでも安全に画像保存する。
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


PathLike = Union[str, Path]


def write_image(path: PathLike,
                image: np.ndarray,
                params: Optional[Sequence[int]] = None) -> bool:
    """
    Unicodeパス対応で画像を保存する。

    OpenCVの `cv2.imwrite` は環境によりUnicodeパスで失敗するため、
    `cv2.imencode` + `numpy.ndarray.tofile` で保存する。
    """
    try:
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        if not ext:
            logger.warning(f"画像保存失敗: 拡張子がありません ({path_obj})")
            return False

        ok, encoded = cv2.imencode(ext, image, list(params or []))
        if not ok:
            logger.warning(f"画像エンコード失敗: {path_obj}")
            return False

        encoded.tofile(str(path_obj))
        return True
    except Exception as e:
        logger.warning(f"画像保存失敗 ({path}): {e}")
        return False

