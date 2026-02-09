"""
Equirectangular投影処理モジュール（最適化版）
360度画像のEquirectangular形式と立方体マップ、透視投影間の変換処理
cv2.remap() + UV キャッシング + GPU加速対応
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import logging

try:
    from ..core.accelerator import get_accelerator
    HAS_ACCELERATOR = True
except (ImportError, ValueError):
    HAS_ACCELERATOR = False

logger = logging.getLogger('360split')


class EquirectangularProcessor:
    """
    360度Equirectangular画像の処理クラス（最適化版）
    立方体マップ、透視投影への変換、球面カバレッジ計算など

    最適化：
    - cv2.remap() で 10x+ 高速化
    - UV マップキャッシング
    - GPU 加速対応
    - ベクトル化されたカバレッジ計算
    """

    def __init__(self):
        """初期化"""
        # UV マップキャッシュ
        self._cubemap_cache = {}  # key: (src_w, src_h, face_size, face_name)
        self._perspective_cache = {}  # key: (src_w, src_h, out_w, out_h, fov, yaw, pitch)

        # アクセレータ初期化
        self.accelerator = None
        if HAS_ACCELERATOR:
            try:
                self.accelerator = get_accelerator()
            except Exception as e:
                logger.warning(f"アクセレータ初期化失敗: {e}")

    # ===== 立方体マップ変換 =====

    def to_cubemap(self, equirect_image: np.ndarray, face_size: int = 1024) -> Dict[str, np.ndarray]:
        """
        Equirectangular画像を6つの立方体マップ面に変換（最適化版）

        cv2.remap() と UV マップキャッシングを使用。

        Args:
            equirect_image: 入力Equirectangular画像 (H x W x 3)
            face_size: 出力立方体面のサイズ（ピクセル）

        Returns:
            6つの立方体面を含む辞書 {'front', 'back', 'left', 'right', 'up', 'down'}
            各面は (face_size x face_size x 3) の配列
        """
        height, width = equirect_image.shape[:2]

        # 立方体面の基底ベクトル定義
        faces = {
            'front': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'back': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            'right': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            'left': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            'up': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            'down': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        }

        cubemap = {}

        for face_name, basis in faces.items():
            # キャッシュをチェック
            cache_key = (width, height, face_size, face_name)
            if cache_key in self._cubemap_cache:
                map_x, map_y = self._cubemap_cache[cache_key]
            else:
                # UV マップを計算
                map_x, map_y = self._compute_cubemap_uv(
                    width, height, face_size, basis
                )
                # キャッシュに保存
                self._cubemap_cache[cache_key] = (map_x, map_y)

            # GPU remap または CPU remap
            if self.accelerator and hasattr(self.accelerator, 'gpu_remap'):
                try:
                    face = self.accelerator.gpu_remap(
                        equirect_image, map_x, map_y, cv2.INTER_LINEAR
                    )
                except Exception as e:
                    logger.debug(f"GPU remap失敗、CPUでフォールバック: {e}")
                    face = cv2.remap(equirect_image, map_x, map_y,
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            else:
                face = cv2.remap(equirect_image, map_x, map_y,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

            cubemap[face_name] = face.astype(np.uint8)

        return cubemap

    def _compute_cubemap_uv(self, src_width: int, src_height: int,
                            face_size: int, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        立方体マップ用の UV マップを計算

        Args:
            src_width: ソース画像幅
            src_height: ソース画像高さ
            face_size: 立方体面サイズ
            basis: 基底ベクトル (3x3)

        Returns:
            (map_x, map_y): cv2.remap() 用の座標マップ
        """
        # 立方体面上のピクセル座標をメッシュ生成
        x = np.linspace(-1, 1, face_size, dtype=np.float32)
        y = np.linspace(-1, 1, face_size, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # 立方体面座標系から3D方向ベクトルへ変換
        directions = np.zeros((face_size, face_size, 3), dtype=np.float32)
        directions[:, :, 0] = basis[0, 0] * xv + basis[1, 0]
        directions[:, :, 1] = basis[0, 1] * xv + basis[1, 1]
        directions[:, :, 2] = basis[0, 2] * xv + basis[1, 2]

        directions += basis[2:3] * yv[:, :, np.newaxis]

        # 3D方向ベクトルを正規化
        norms = np.sqrt(np.sum(directions**2, axis=2, keepdims=True))
        directions = directions / (norms + 1e-8)

        # 球面座標 (theta, phi) に変換
        theta = np.arctan2(directions[:, :, 2], directions[:, :, 0])
        phi = np.arcsin(np.clip(directions[:, :, 1], -1, 1))

        # Equirectangular画像座標に変換
        map_x = ((theta + np.pi) / (2 * np.pi) * src_width).astype(np.float32)
        map_y = ((np.pi / 2 - phi) / np.pi * src_height).astype(np.float32)

        # 360度ラップを処理
        map_x = np.fmod(map_x + src_width, src_width)
        map_x = np.clip(map_x, 0, src_width - 1.001).astype(np.float32)
        map_y = np.clip(map_y, 0, src_height - 1.001).astype(np.float32)

        return map_x, map_y

    def from_cubemap(self, faces_dict: Dict[str, np.ndarray], output_width: int = 4096) -> np.ndarray:
        """
        6つの立方体マップ面からEquirectangular画像を再構成

        Args:
            faces_dict: 6つの立方体面を含む辞書
            output_width: 出力Equirectangular画像の幅

        Returns:
            再構成されたEquirectangular画像 (output_height x output_width x 3)
        """
        output_height = output_width // 2

        # 出力画像初期化
        equirect = np.zeros((output_height, output_width, 3), dtype=np.float32)
        weights = np.zeros((output_height, output_width), dtype=np.float32)

        # 各立方体面の基底ベクトル
        face_bases = {
            'front': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'back': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            'right': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            'left': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            'up': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            'down': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        }

        # Equirectangular座標をメッシュ生成
        theta = np.linspace(-np.pi, np.pi, output_width, dtype=np.float32)
        phi = np.linspace(np.pi / 2, -np.pi / 2, output_height, dtype=np.float32)
        theta_v, phi_v = np.meshgrid(theta, phi)

        # 球面座標から3D方向ベクトルへ
        x = np.cos(phi_v) * np.cos(theta_v)
        y = np.sin(phi_v)
        z = np.cos(phi_v) * np.sin(theta_v)

        # 各立方体面に対して逆マッピング
        for face_name, basis in face_bases.items():
            face_image = faces_dict[face_name].astype(np.float32)
            face_size = face_image.shape[0]

            # 3Dベクトルを立方体面座標に投影
            x_face = x * basis[0, 0] + y * basis[1, 0] + z * basis[2, 0]
            y_face = x * basis[0, 1] + y * basis[1, 1] + z * basis[2, 1]
            z_face = x * basis[0, 2] + y * basis[1, 2] + z * basis[2, 2]

            # 面に対して有効な領域
            valid = (z_face > 0)

            # 立方体面座標系に変換
            face_u = x_face / (z_face + 1e-8)
            face_v = y_face / (z_face + 1e-8)

            # ピクセル座標に変換
            face_x = (face_u + 1) / 2 * (face_size - 1)
            face_y = (-face_v + 1) / 2 * (face_size - 1)

            # 有効範囲をチェック
            in_bounds = valid & (face_x >= 0) & (face_x < face_size - 1) & \
                       (face_y >= 0) & (face_y < face_size - 1)

            # cv2.remap で高速補間
            map_x = face_x.astype(np.float32)
            map_y = face_y.astype(np.float32)
            sampled = cv2.remap(face_image, map_x, map_y, cv2.INTER_LINEAR)

            # Equirectangular画像に合成
            equirect[in_bounds] += sampled[in_bounds]
            weights[in_bounds] += 1

        # 正規化
        weights = np.maximum(weights, 1e-8)
        equirect = equirect / weights[:, :, np.newaxis]

        return np.clip(equirect, 0, 255).astype(np.uint8)

    # ===== 透視投影変換 =====

    def to_perspective(self, equirect_image: np.ndarray, yaw: float, pitch: float,
                      fov: float = 90.0, output_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """
        Equirectangular画像から透視投影画像を抽出（最適化版）

        指定された方向と視野角で透視投影を計算。
        UV マップキャッシングと cv2.remap() を使用。

        Args:
            equirect_image: 入力Equirectangular画像 (H x W x 3)
            yaw: ヨー角（度、-180から180）
            pitch: ピッチ角（度、-90から90）
            fov: 視野角（度）
            output_size: 出力画像サイズ (height, width)

        Returns:
            透視投影画像 (height x width x 3)
        """
        height, width = equirect_image.shape[:2]
        out_h, out_w = output_size

        # キャッシュをチェック
        cache_key = (width, height, out_w, out_h, fov, yaw, pitch)
        if cache_key in self._perspective_cache:
            map_x, map_y = self._perspective_cache[cache_key]
        else:
            # UV マップを計算
            map_x, map_y = self._compute_perspective_uv(
                width, height, out_w, out_h, yaw, pitch, fov
            )
            # キャッシュに保存
            self._perspective_cache[cache_key] = (map_x, map_y)

        # GPU remap または CPU remap
        if self.accelerator and hasattr(self.accelerator, 'gpu_remap'):
            try:
                perspective = self.accelerator.gpu_remap(
                    equirect_image, map_x, map_y, cv2.INTER_LINEAR
                )
            except Exception as e:
                logger.debug(f"GPU remap失敗、CPUでフォールバック: {e}")
                perspective = cv2.remap(equirect_image, map_x, map_y,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        else:
            perspective = cv2.remap(equirect_image, map_x, map_y,
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        return perspective.astype(np.uint8)

    def _compute_perspective_uv(self, src_width: int, src_height: int,
                               out_width: int, out_height: int,
                               yaw: float, pitch: float, fov: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        透視投影用の UV マップを計算

        Args:
            src_width: ソース画像幅
            src_height: ソース画像高さ
            out_width: 出力幅
            out_height: 出力高さ
            yaw: ヨー角（度）
            pitch: ピッチ角（度）
            fov: 視野角（度）

        Returns:
            (map_x, map_y): cv2.remap() 用の座標マップ
        """
        # 度からラジアンに変換
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        fov_rad = np.radians(fov)

        # 透視投影平面上のピクセル座標をメッシュ生成
        x = np.linspace(-1, 1, out_width, dtype=np.float32)
        y = np.linspace(-1, 1, out_height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # 焦点距離を計算
        focal_length = 1.0 / np.tan(fov_rad / 2)

        # カメラ座標系での3D方向
        cam_x = xv
        cam_y = yv
        cam_z = focal_length * np.ones_like(xv, dtype=np.float32)

        # ヨー回転（Y軸周り）
        rot_y_cos = np.cos(yaw_rad)
        rot_y_sin = np.sin(yaw_rad)

        rotated_x = cam_x * rot_y_cos + cam_z * rot_y_sin
        rotated_z = -cam_x * rot_y_sin + cam_z * rot_y_cos

        # ピッチ回転（X軸周り）
        rot_p_cos = np.cos(pitch_rad)
        rot_p_sin = np.sin(pitch_rad)

        rotated_y = cam_y * rot_p_cos - rotated_z * rot_p_sin
        rotated_z = cam_y * rot_p_sin + rotated_z * rot_p_cos

        # 3D方向を正規化
        norms = np.sqrt(rotated_x**2 + rotated_y**2 + rotated_z**2)
        rotated_x = rotated_x / (norms + 1e-8)
        rotated_y = rotated_y / (norms + 1e-8)
        rotated_z = rotated_z / (norms + 1e-8)

        # 球面座標に変換
        theta = np.arctan2(rotated_z, rotated_x)
        phi = np.arcsin(np.clip(rotated_y, -1, 1))

        # Equirectangular座標に変換
        map_x = ((theta + np.pi) / (2 * np.pi) * src_width).astype(np.float32)
        map_y = ((np.pi / 2 - phi) / np.pi * src_height).astype(np.float32)

        # 360度ラップを処理
        map_x = np.fmod(map_x + src_width, src_width)
        map_x = np.clip(map_x, 0, src_width - 1.001).astype(np.float32)
        map_y = np.clip(map_y, 0, src_height - 1.001).astype(np.float32)

        return map_x, map_y

    # ===== カバレッジ計算 =====

    def compute_coverage_map(self, keyframe_poses: List[Tuple[float, float]],
                            image_width: int, image_height: int) -> np.ndarray:
        """
        複数のキーフレーム視点から球面のカバレッジヒートマップを計算（ベクトル化版）

        Args:
            keyframe_poses: [(yaw, pitch), ...] のキーフレーム姿勢リスト
            image_width: Equirectangular画像の幅
            image_height: Equirectangular画像の高さ

        Returns:
            カバレッジヒートマップ (0から1の値) (height x width)
        """
        coverage = np.zeros((image_height, image_width), dtype=np.float32)

        if not keyframe_poses:
            return coverage

        # Equirectangular座標をメッシュ生成
        theta = np.linspace(-np.pi, np.pi, image_width, dtype=np.float32)
        phi = np.linspace(np.pi / 2, -np.pi / 2, image_height, dtype=np.float32)
        theta_v, phi_v = np.meshgrid(theta, phi)

        # 球面座標から3D方向
        x = np.cos(phi_v) * np.cos(theta_v)
        y = np.sin(phi_v)
        z = np.cos(phi_v) * np.sin(theta_v)

        # 全キーフレームに対してベクトル化計算
        fov = 90.0
        fov_rad = np.radians(fov)
        max_angle = fov_rad / 2

        for yaw_deg, pitch_deg in keyframe_poses:
            # カメラ視線方向を計算
            yaw_rad = np.radians(yaw_deg)
            pitch_rad = np.radians(pitch_deg)

            cam_forward_x = np.cos(pitch_rad) * np.cos(yaw_rad)
            cam_forward_y = np.sin(pitch_rad)
            cam_forward_z = np.cos(pitch_rad) * np.sin(yaw_rad)

            # 内積（ベクトル化）
            cos_angle = x * cam_forward_x + y * cam_forward_y + z * cam_forward_z

            # 視野角内の領域に対してガウス重み付け
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            weight = np.exp(-4 * (angle / max_angle)**2)
            weight[cos_angle < np.cos(max_angle)] = 0

            coverage += weight

        # 正規化
        coverage = np.clip(coverage / len(keyframe_poses), 0, 1)

        return coverage

    # ===== 球面座標とベクトル変換 =====

    def get_viewing_direction(self, equirect_width: int, equirect_height: int,
                             x: float, y: float) -> np.ndarray:
        """
        Equirectangular画像のピクセル座標を3D方向ベクトルに変換

        Args:
            equirect_width: 画像幅
            equirect_height: 画像高さ
            x: ピクセルX座標（0からwidth-1）
            y: ピクセルY座標（0からheight-1）

        Returns:
            正規化された3D方向ベクトル [x, y, z]
        """
        # ピクセル座標を球面座標に変換
        theta = (x / equirect_width) * 2 * np.pi - np.pi
        phi = (equirect_height / 2 - y) / equirect_height * np.pi - np.pi / 2

        # 球面座標から3D方向に変換
        cos_phi = np.cos(phi)
        direction = np.array([
            cos_phi * np.cos(theta),
            np.sin(phi),
            cos_phi * np.sin(theta)
        ], dtype=np.float32)

        return direction / (np.linalg.norm(direction) + 1e-8)

    def compute_overlap(self, pose1: Tuple[float, float], pose2: Tuple[float, float],
                       fov: float = 90.0) -> float:
        """
        2つの視点方向間の角度オーバーラップを計算

        視野角内で重なっている領域の割合を返す

        Args:
            pose1: (yaw, pitch) - 最初の姿勢（度）
            pose2: (yaw, pitch) - 2番目の姿勢（度）
            fov: 視野角（度）

        Returns:
            オーバーラップ度 (0から1)
        """
        yaw1, pitch1 = pose1
        yaw2, pitch2 = pose2

        # 方向ベクトルを計算
        yaw1_rad = np.radians(yaw1)
        pitch1_rad = np.radians(pitch1)
        yaw2_rad = np.radians(yaw2)
        pitch2_rad = np.radians(pitch2)

        dir1 = np.array([
            np.cos(pitch1_rad) * np.cos(yaw1_rad),
            np.sin(pitch1_rad),
            np.cos(pitch1_rad) * np.sin(yaw1_rad)
        ], dtype=np.float32)

        dir2 = np.array([
            np.cos(pitch2_rad) * np.cos(yaw2_rad),
            np.sin(pitch2_rad),
            np.cos(pitch2_rad) * np.sin(yaw2_rad)
        ], dtype=np.float32)

        # 2つの方向間の角度
        cos_angle = np.clip(np.dot(dir1, dir2), -1, 1)
        angle = np.arccos(cos_angle)

        # 視野角から計算
        fov_rad = np.radians(fov)
        max_angle = fov_rad / 2

        # オーバーラップ度を計算
        if angle > 2 * max_angle:
            return 0.0

        overlap = 1.0 - (angle / (2 * max_angle))
        return float(np.clip(overlap, 0.0, 1.0))
