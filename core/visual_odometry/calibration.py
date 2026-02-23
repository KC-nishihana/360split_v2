from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CalibrationData:
    camera_matrix: np.ndarray
    dist: np.ndarray
    model: str
    image_size: Tuple[int, int]
    xml_path: str


def _read_first_matrix(fs: cv2.FileStorage, names: list[str]) -> Optional[np.ndarray]:
    for name in names:
        node = fs.getNode(name)
        if node.empty():
            continue
        mat = node.mat()
        if mat is not None:
            arr = np.asarray(mat, dtype=np.float64)
            if arr.size > 0:
                return arr
    return None


def _read_first_int(fs: cv2.FileStorage, names: list[str]) -> Optional[int]:
    for name in names:
        node = fs.getNode(name)
        if node.empty():
            continue
        try:
            value = int(node.real())
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _normalize_model(model_hint: str, dist_size: int) -> str:
    hint = str(model_hint or "auto").strip().lower()
    if hint in ("opencv", "fisheye"):
        return hint
    if hint != "auto":
        return "opencv"
    return "fisheye" if dist_size == 4 else "opencv"


def calibration_to_dict(calib: Optional[CalibrationData]) -> Optional[Dict[str, Any]]:
    if calib is None:
        return None
    return {
        "camera_matrix": np.asarray(calib.camera_matrix, dtype=np.float64).tolist(),
        "dist": np.asarray(calib.dist, dtype=np.float64).reshape(-1).tolist(),
        "model": str(calib.model),
        "image_size": [int(calib.image_size[0]), int(calib.image_size[1])],
        "xml_path": str(calib.xml_path),
    }


def calibration_from_dict(data: Optional[Dict[str, Any]]) -> Optional[CalibrationData]:
    if not data:
        return None
    try:
        k = np.asarray(data.get("camera_matrix", []), dtype=np.float64).reshape(3, 3)
        d = np.asarray(data.get("dist", []), dtype=np.float64).reshape(-1)
        size_raw = data.get("image_size", [0, 0])
        if isinstance(size_raw, (tuple, list)) and len(size_raw) == 2:
            size = (int(size_raw[0]), int(size_raw[1]))
        else:
            size = (0, 0)
        return CalibrationData(
            camera_matrix=k,
            dist=d,
            model=str(data.get("model", "opencv")).strip().lower(),
            image_size=size,
            xml_path=str(data.get("xml_path", "")),
        )
    except Exception:
        return None


def summarize_calibration(calib: Optional[CalibrationData]) -> Dict[str, Any]:
    if calib is None:
        return {"enabled": False}
    k = np.asarray(calib.camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(calib.dist, dtype=np.float64).reshape(-1)
    return {
        "enabled": True,
        "xml_path": str(calib.xml_path),
        "model": str(calib.model),
        "image_size": [int(calib.image_size[0]), int(calib.image_size[1])],
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "dist": [float(x) for x in d.tolist()],
        "camera_matrix": k.tolist(),
    }


def log_calibration_summary(logger, label: str, calib: Optional[CalibrationData]) -> None:
    if calib is None:
        logger.warning(f"{label}: calibration unavailable (VO disabled)")
        return
    info = summarize_calibration(calib)
    logger.info(
        f"{label}: model={info['model']}, size={info['image_size'][0]}x{info['image_size'][1]}, "
        f"fx={info['fx']:.4f}, fy={info['fy']:.4f}, cx={info['cx']:.4f}, cy={info['cy']:.4f}, "
        f"dist={info['dist']}, xml={info['xml_path']}"
    )


def load_calibration_xml(
    xml_path: str,
    model_hint: str = "auto",
    fallback_image_size: Optional[Tuple[int, int]] = None,
    logger=None,
) -> Optional[CalibrationData]:
    log = logger
    try:
        path = Path(xml_path).expanduser()
        if not path.exists():
            if log:
                log.warning(f"calibration XML not found: {xml_path}")
            return None

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            if log:
                log.warning(f"failed to open calibration XML: {xml_path}")
            return None

        try:
            k = _read_first_matrix(fs, ["Camera_Matrix", "camera_matrix", "K", "cameraMatrix"])
            d = _read_first_matrix(
                fs,
                ["Distortion_Coefficients", "distortion_coefficients", "distortion", "distCoeffs", "D"],
            )
            if k is None or d is None:
                if log:
                    log.warning(f"missing Camera_Matrix/Distortion_Coefficients in XML: {xml_path}")
                return None

            k = np.asarray(k, dtype=np.float64).reshape(3, 3)
            d = np.asarray(d, dtype=np.float64).reshape(-1)

            w = _read_first_int(fs, ["image_Width", "image_width", "width", "ImageWidth"])
            h = _read_first_int(fs, ["image_Height", "image_height", "height", "ImageHeight"])
            if (w is None or h is None) and fallback_image_size is not None:
                w = int(fallback_image_size[0])
                h = int(fallback_image_size[1])

            if w is None or h is None:
                w, h = 0, 0

            model = _normalize_model(model_hint=model_hint, dist_size=int(d.size))
            return CalibrationData(
                camera_matrix=k,
                dist=d,
                model=model,
                image_size=(int(w), int(h)),
                xml_path=str(path),
            )
        finally:
            fs.release()
    except Exception as e:
        if log:
            log.warning(f"calibration XML load failed: path={xml_path}, err={e}")
        return None

