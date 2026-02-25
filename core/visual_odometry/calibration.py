from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

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


def _read_xml_float(root: ET.Element, key: str, default: Optional[float] = None) -> Optional[float]:
    node = root.find(key)
    if node is None or node.text is None:
        return default
    try:
        return float(node.text.strip())
    except (TypeError, ValueError):
        return default


def _read_xml_int(root: ET.Element, key: str, default: Optional[int] = None) -> Optional[int]:
    node = root.find(key)
    if node is None or node.text is None:
        return default
    try:
        return int(float(node.text.strip()))
    except (TypeError, ValueError):
        return default


def _load_opencv_filestorage_xml(
    path: Path,
    model_hint: str,
    fallback_image_size: Optional[Tuple[int, int]],
) -> Optional[CalibrationData]:
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except Exception:
        return None

    if not fs.isOpened():
        return None

    try:
        k = _read_first_matrix(fs, ["Camera_Matrix", "camera_matrix", "K", "cameraMatrix"])
        d = _read_first_matrix(
            fs,
            ["Distortion_Coefficients", "distortion_coefficients", "distortion", "distCoeffs", "D"],
        )
        if k is None or d is None:
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


def _load_agisoft_calibration_xml(path: Path, model_hint: str) -> Optional[CalibrationData]:
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return None

    if root.tag.strip().lower() != "calibration":
        return None

    projection = ((root.findtext("projection") or "").strip().lower())
    width = _read_xml_int(root, "width")
    height = _read_xml_int(root, "height")
    f = _read_xml_float(root, "f")
    if width is None or height is None or width <= 0 or height <= 0 or f is None:
        return None

    # Agisoft offsets are relative to image center.
    cx_offset = _read_xml_float(root, "cx", 0.0) or 0.0
    cy_offset = _read_xml_float(root, "cy", 0.0) or 0.0
    b1 = _read_xml_float(root, "b1", 0.0) or 0.0
    b2 = _read_xml_float(root, "b2", 0.0) or 0.0
    k1 = _read_xml_float(root, "k1", 0.0) or 0.0
    k2 = _read_xml_float(root, "k2", 0.0) or 0.0
    k3 = _read_xml_float(root, "k3", 0.0) or 0.0
    k4 = _read_xml_float(root, "k4", 0.0) or 0.0
    p1 = _read_xml_float(root, "p1", 0.0) or 0.0
    p2 = _read_xml_float(root, "p2", 0.0) or 0.0

    fx = float(f + b1)
    fy = float(f)
    cx = float(width * 0.5 + cx_offset)
    cy = float(height * 0.5 + cy_offset)
    camera_matrix = np.array(
        [[fx, float(b2), cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    hint = str(model_hint or "auto").strip().lower()
    if hint not in ("opencv", "fisheye", "auto"):
        hint = "auto"
    if hint == "auto":
        hint = "fisheye" if "fisheye" in projection else "opencv"

    if hint == "fisheye":
        dist = np.array([k1, k2, k3, k4], dtype=np.float64)
        model = "fisheye"
    else:
        # Keep OpenCV-compatible (k1,k2,p1,p2,k3) ordering.
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        model = "opencv"

    return CalibrationData(
        camera_matrix=camera_matrix,
        dist=dist,
        model=model,
        image_size=(int(width), int(height)),
        xml_path=str(path),
    )


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

        calib = _load_opencv_filestorage_xml(
            path=path,
            model_hint=model_hint,
            fallback_image_size=fallback_image_size,
        )
        if calib is not None:
            return calib

        calib = _load_agisoft_calibration_xml(path=path, model_hint=model_hint)
        if calib is not None:
            if log:
                log.info(f"Agisoft calibration XML detected: {path}")
            return calib

        if log:
            log.warning(f"unsupported calibration XML format: {path}")
        return None
    except Exception as e:
        if log:
            log.warning(f"calibration XML load failed: path={xml_path}, err={e}")
        return None
