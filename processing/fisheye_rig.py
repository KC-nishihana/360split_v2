"""
前後魚眼リグ処理モジュール。

- 魚眼リグの内部/外部パラメータ推定
- 前後魚眼からEquirectangular 360画像の生成
- 全周囲特徴抽出（SIFT/ORB）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import Equirect360Config
from core.rig_models import LensIntrinsics, RigCalibration
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PanoramaFeatures:
    """360画像特徴点の結果。"""
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    seam_keypoint_count: int


class FisheyeRigProcessor:
    """前後魚眼リグの校正とスティッチング処理。"""

    def __init__(self, equirect_config: Optional[Equirect360Config] = None):
        self.equirect_config = equirect_config or Equirect360Config()

    def calibrate_from_checkerboard(
        self,
        front_images: List[np.ndarray],
        rear_images: List[np.ndarray],
        checkerboard_size: Tuple[int, int],
        square_size: float,
    ) -> RigCalibration:
        """
        チェッカーボード画像列から前後魚眼リグをキャリブレーションする。

        Parameters
        ----------
        front_images/rear_images:
            同期した前後画像列
        checkerboard_size:
            (cols, rows) の内部コーナー数
        square_size:
            1マスのサイズ（メートルなど任意単位）
        """
        if len(front_images) != len(rear_images):
            raise ValueError("front_images と rear_images の枚数が一致していません")
        if len(front_images) < 5:
            raise ValueError("キャリブレーションには最低5ペア以上の画像が必要です")

        objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []
        imgpoints_f = []
        imgpoints_r = []

        for front, rear in zip(front_images, rear_images):
            gray_f = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rear, cv2.COLOR_BGR2GRAY)

            ok_f, corners_f = cv2.findChessboardCorners(gray_f, checkerboard_size)
            ok_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_size)
            if not ok_f or not ok_r:
                continue

            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
            corners_f = cv2.cornerSubPix(gray_f, corners_f, (5, 5), (-1, -1), term)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), term)

            objpoints.append(objp)
            imgpoints_f.append(corners_f.reshape(1, -1, 2))
            imgpoints_r.append(corners_r.reshape(1, -1, 2))

        if len(objpoints) < 5:
            raise RuntimeError("有効なチェッカーボード検出ペアが不足しています")

        h, w = front_images[0].shape[:2]
        Kf = np.eye(3)
        Df = np.zeros((4, 1))
        Kr = np.eye(3)
        Dr = np.zeros((4, 1))

        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

        err_f, Kf, Df, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints_f, (w, h), Kf, Df, None, None, flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
        err_r, Kr, Dr, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints_r, (w, h), Kr, Dr, None, None, flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )

        stereo_flags = (
            cv2.fisheye.CALIB_FIX_INTRINSIC +
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        _, _, _, _, _, R, T = cv2.fisheye.stereoCalibrate(
            objpoints,
            imgpoints_f,
            imgpoints_r,
            Kf,
            Df,
            Kr,
            Dr,
            (w, h),
            None,
            None,
            stereo_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )

        calibration = RigCalibration(
            lens_a=LensIntrinsics(Kf, Df, w, h, model="fisheye"),
            lens_b=LensIntrinsics(Kr, Dr, w, h, model="fisheye"),
            rotation_ab=R,
            translation_ab=T,
            reprojection_error=float((err_f + err_r) / 2.0)
        )
        return calibration

    def stitch_to_equirect(
        self,
        front_frame: np.ndarray,
        rear_frame: np.ndarray,
        calibration: Optional[RigCalibration],
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前後魚眼画像からEquirectangular 360画像を合成する。

        Returns
        -------
        stitched:
            合成済み Equirectangular 画像
        seam_mask:
            前後重複（縫い目近傍）領域マスク
        """
        out_w, out_h = output_size

        front_eq = self._fisheye_to_equirect(front_frame, calibration.lens_a if calibration else None, out_w, out_h)
        rear_eq = self._fisheye_to_equirect(rear_frame, calibration.lens_b if calibration else None, out_w, out_h)

        # 前半分をfront、後半分をrearで合成（簡易シームブレンディング）
        x = np.linspace(0.0, 1.0, out_w, dtype=np.float32)
        front_weight = np.where(x <= 0.5, 1.0, 0.0)

        # 0度と180度近傍を滑らかにブレンド
        blend_width = max(8, out_w // 40)
        for center in (0, out_w // 2, out_w - 1):
            left = max(0, center - blend_width)
            right = min(out_w, center + blend_width)
            ramp = np.linspace(1.0, 0.0, right - left, dtype=np.float32)
            if center == 0 or center == out_w - 1:
                front_weight[left:right] = np.maximum(front_weight[left:right], ramp)
            else:
                front_weight[left:right] = np.minimum(front_weight[left:right], ramp)

        front_weight = np.tile(front_weight[np.newaxis, :], (out_h, 1))
        stitched = (
            front_eq.astype(np.float32) * front_weight[..., None] +
            rear_eq.astype(np.float32) * (1.0 - front_weight[..., None])
        )

        seam_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        seam_half = max(4, out_w // 80)
        seam_cols = [0, out_w // 2, out_w - 1]
        for c in seam_cols:
            l = max(0, c - seam_half)
            r = min(out_w, c + seam_half)
            seam_mask[:, l:r] = 255

        return np.clip(stitched, 0, 255).astype(np.uint8), seam_mask

    def extract_360_features(
        self,
        equirect: np.ndarray,
        seam_mask: Optional[np.ndarray] = None,
        method: str = "orb",
    ) -> PanoramaFeatures:
        """360画像から特徴点を抽出し、シーム付近の特徴点数を返す。"""
        gray = cv2.cvtColor(equirect, cv2.COLOR_BGR2GRAY) if equirect.ndim == 3 else equirect

        if method.lower() == "sift" and hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create()
        else:
            detector = cv2.ORB_create(nfeatures=5000)

        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        if self.equirect_config.enable_polar_mask:
            polar = int(gray.shape[0] * self.equirect_config.mask_polar_ratio)
            mask[:polar, :] = 0
            mask[-polar:, :] = 0

        keypoints, descriptors = detector.detectAndCompute(gray, mask)

        seam_keypoint_count = 0
        if seam_mask is not None and keypoints:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= y < seam_mask.shape[0] and 0 <= x < seam_mask.shape[1] and seam_mask[y, x] > 0:
                    seam_keypoint_count += 1

        return PanoramaFeatures(
            keypoints=keypoints or [],
            descriptors=descriptors,
            seam_keypoint_count=seam_keypoint_count
        )

    def _fisheye_to_equirect(
        self,
        frame: np.ndarray,
        intrinsics: Optional[LensIntrinsics],
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """魚眼画像を簡易Equirectへ展開。"""
        h, w = frame.shape[:2]

        # 球面方向ベクトル
        theta = np.linspace(-np.pi, np.pi, out_w, dtype=np.float32)
        phi = np.linspace(np.pi / 2, -np.pi / 2, out_h, dtype=np.float32)
        theta_v, phi_v = np.meshgrid(theta, phi)

        x = np.cos(phi_v) * np.cos(theta_v)
        y = np.sin(phi_v)
        z = np.cos(phi_v) * np.sin(theta_v)

        # 前方半球のみ使用
        valid = x > 0
        xn = z / np.maximum(x, 1e-8)
        yn = y / np.maximum(x, 1e-8)

        r = np.sqrt(xn * xn + yn * yn)
        theta_f = np.arctan(r)

        if intrinsics is not None:
            fx = intrinsics.camera_matrix[0, 0]
            fy = intrinsics.camera_matrix[1, 1]
            cx = intrinsics.camera_matrix[0, 2]
            cy = intrinsics.camera_matrix[1, 2]
        else:
            fx = fy = min(h, w) * 0.5
            cx = w * 0.5
            cy = h * 0.5

        scale = np.where(r > 1e-8, theta_f / r, 0.0)
        map_x = (fx * xn * scale + cx).astype(np.float32)
        map_y = (fy * yn * scale + cy).astype(np.float32)

        map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

        projected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        projected[~valid] = 0
        return projected
