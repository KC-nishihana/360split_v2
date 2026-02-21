"""
マスク処理モジュール
カメラマンと機材の除去、および領域マスキング処理
"""

import numpy as np
import cv2
from typing import Callable, List, Tuple, Optional, Union


class MaskProcessor:
    """
    マスク生成と適用を行うクラス
    写真家・機材の検出と除去、カバレッジマスク生成
    """

    def __init__(self):
        """初期化"""
        pass

    # ===== マスク生成 =====

    def create_nadir_mask(self, width: int, height: int, radius_ratio: float = 0.15) -> np.ndarray:
        """
        Equirectangular画像の底部（nadir）領域用の円形マスクを生成
        三脚や撮影者がこの領域に映り込むことが多い

        Args:
            width: 画像幅
            height: 画像高さ
            radius_ratio: マスク半径の高さに対する比率（0-1）

        Returns:
            バイナリマスク (height x width)、マスク領域は1、背景は0
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # Equirectangular画像では底部中央がnadir（nadir region）
        # 底部の中央付近に円形マスクを適用
        center_x = width // 2
        center_y = height - int(height * radius_ratio)

        radius = int(height * radius_ratio)

        # 円形マスクを描画
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)

        return mask

    def create_zenith_mask(self, width: int, height: int, radius_ratio: float = 0.05) -> np.ndarray:
        """
        Equirectangular画像の頂部（zenith）領域用のマスクを生成
        頂部はノイズが多い領域

        Args:
            width: 画像幅
            height: 画像高さ
            radius_ratio: マスク半径の高さに対する比率（0-1）

        Returns:
            バイナリマスク (height x width)、マスク領域は1、背景は0
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # 頂部の中央
        center_x = width // 2
        center_y = int(height * radius_ratio)

        radius = int(height * radius_ratio)

        # 円形マスク
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)

        return mask

    def create_fisheye_valid_mask(
        self,
        width: int,
        height: int,
        radius_ratio: float = 0.94,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> np.ndarray:
        """
        魚眼画像の有効領域（レンズ内）マスクを生成する。

        Returns:
            uint8 マスク (H x W), 255=有効領域, 0=無効領域
        """
        if width <= 0 or height <= 0:
            return np.zeros((max(height, 0), max(width, 0)), dtype=np.uint8)

        ratio = float(np.clip(radius_ratio, 0.0, 1.0))
        radius = int(min(width, height) * 0.5 * ratio)

        cx = int(np.clip((width // 2) + int(offset_x), 0, max(width - 1, 0)))
        cy = int(np.clip((height // 2) + int(offset_y), 0, max(height - 1, 0)))

        mask = np.zeros((height, width), dtype=np.uint8)
        if radius > 0:
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask

    def create_equipment_mask(self, frame: np.ndarray, method: str = 'threshold') -> np.ndarray:
        """
        フレーム内の機材を自動検出してマスクを生成

        Args:
            frame: 入力フレーム (H x W x 3) BGR画像
            method: 検出方法
                'threshold': 色・輝度しきい値による簡単な検出
                'contour': 輪郭ベースの形状検出

        Returns:
            検出された機材領域のバイナリマスク (height x width)
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if method == 'threshold':
            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 暗い領域（黒い機材）を検出
            # 機材は通常暗い色
            dark_mask = gray < 60
            mask[dark_mask] = 1

            # 極端に明るい領域（白飛び）も除外候補
            bright_mask = gray > 240
            mask[bright_mask] = 1

        elif method == 'contour':
            # エッジ検出
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # キャニーエッジ検出
            edges = cv2.Canny(blurred, 50, 150)

            # 輪郭を抽出
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 既知の形状（三脚など）に基づいて輪郭を分析
            # ここでは簡略版：小さすぎたり大きすぎたりしない輪郭を機材と判定
            for contour in contours:
                area = cv2.contourArea(contour)

                # 面積がフレーム面積の0.5%から20%の範囲
                frame_area = height * width
                if frame_area * 0.005 < area < frame_area * 0.2:
                    # 輪郭が直線的または機械的な形状か判定
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

                    # 4～8辺の多角形を機材と判定
                    if 4 <= len(approx) <= 8:
                        cv2.drawContours(mask, [contour], 0, 1, -1)

        return mask

    # ===== マスク適用 =====

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray, value: Union[int, Tuple[int, int, int]] = 0) -> np.ndarray:
        """
        フレームにマスクを適用
        マスク領域を指定値に設定

        Args:
            frame: 入力フレーム (H x W x 3)
            mask: バイナリマスク (H x W)
            value: マスク領域に設定する値（スカラーまたはRGB）

        Returns:
            マスク適用後のフレーム
        """
        result = frame.copy()

        if isinstance(value, (tuple, list)):
            # RGB値を指定
            result[mask == 1] = value
        else:
            # グレースケール値を指定
            result[mask == 1] = value

        return result

    def apply_valid_region_mask(
        self,
        frame: np.ndarray,
        valid_mask: np.ndarray,
        fill_value: Union[int, Tuple[int, int, int]] = 0,
    ) -> np.ndarray:
        """
        有効領域マスク(255=有効, 0=無効)を使って無効領域を塗りつぶす。
        """
        if frame is None:
            return frame
        if valid_mask is None:
            return frame.copy()

        mask = valid_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        result = frame.copy()
        invalid = mask == 0
        if isinstance(fill_value, (tuple, list)):
            result[invalid] = tuple(fill_value)
        else:
            result[invalid] = fill_value
        return result

    def dilate_mask(self, mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        マスクを膨張させて覆う領域を拡張
        これによりマスク境界の近い領域も確実に覆う

        Args:
            mask: バイナリマスク
            kernel_size: 膨張カーネルのサイズ

        Returns:
            膨張されたマスク
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=1)

        return dilated

    def combine_masks(self, *masks: np.ndarray) -> np.ndarray:
        """
        複数のマスクを論理ORで結合

        Args:
            *masks: 結合対象のマスク群

        Returns:
            結合されたマスク
        """
        if not masks:
            return None

        combined = masks[0].copy()

        for mask in masks[1:]:
            combined = np.logical_or(combined, mask)

        return combined.astype(np.uint8)

    # ===== マスク可視化 =====

    def visualize_mask(self, frame: np.ndarray, mask: np.ndarray,
                      color: Tuple[int, int, int] = (0, 0, 255),
                      alpha: float = 0.5) -> np.ndarray:
        """
        マスクをフレーム上に可視化

        Args:
            frame: 入力フレーム (H x W x 3)
            mask: バイナリマスク (H x W)
            color: オーバーレイ色 (B, G, R)
            alpha: 透明度（0-1）

        Returns:
            可視化されたフレーム
        """
        result = frame.copy().astype(np.float32)

        # マスク領域に色をオーバーレイ
        overlay = frame.copy().astype(np.float32)
        overlay[mask == 1] = color

        # アルファブレンド
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ===== 損失重み計算 =====

    def compute_masked_loss_weight(self, mask: np.ndarray) -> np.ndarray:
        """
        Masked Diffusion Loss用のピクセル単位の重み行列を計算
        マスク領域（mask==1）は重み0、背景は重み1

        Args:
            mask: バイナリマスク (H x W)

        Returns:
            重み行列 (H x W)、値は0または1
        """
        weight = np.ones_like(mask, dtype=np.float32)
        weight[mask == 1] = 0.0

        return weight

    # ===== 動きに基づく検出 =====

    def detect_moving_objects(self, frames: List[np.ndarray], threshold: int = 30) -> np.ndarray:
        """
        複数フレームのバックグラウンド差分で動く物体（撮影者など）を検出

        Args:
            frames: フレームのリスト [(H x W x 3), ...]
            threshold: 差分のしきい値（0-255）

        Returns:
            動いている領域のマスク (H x W)
        """
        if len(frames) < 2:
            return np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.uint8)

        height, width = frames[0].shape[:2]
        motion_mask = np.zeros((height, width), dtype=np.uint8)

        # グレースケール変換
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # 背景画像として最初のフレームを使用
        background = gray_frames[0].astype(np.float32)

        # 複数フレームとの差分を計算
        for gray_frame in gray_frames[1:]:
            frame_float = gray_frame.astype(np.float32)

            # フレーム間の絶対差
            diff = np.abs(frame_float - background)

            # しきい値以上の領域をマスク
            motion_region = diff > threshold
            motion_mask = np.logical_or(motion_mask, motion_region)

            # 背景を更新（指数移動平均）
            background = 0.7 * background + 0.3 * frame_float

        # モルフォロジー処理でノイズを除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        return motion_mask

    def apply_inpaint_hook(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        inpaint_hook: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        将来の動画インペイント連携用フック。

        Args:
            frame: 入力フレーム (H x W x 3)
            mask: インペイント対象マスク (H x W), 1=対象
            inpaint_hook: 外部インペイント関数。未指定時はそのまま返す

        Returns:
            インペイント後フレーム（hook未設定時は入力フレーム）
        """
        if inpaint_hook is None:
            return frame
        try:
            return inpaint_hook(frame, mask)
        except Exception:
            # フック失敗時も本処理を継続する
            return frame

    # ===== マスク分析ユーティリティ =====

    def get_mask_statistics(self, mask: np.ndarray) -> dict:
        """
        マスクの統計情報を取得

        Args:
            mask: バイナリマスク

        Returns:
            統計情報の辞書
        """
        total_pixels = mask.size
        masked_pixels = np.count_nonzero(mask)

        stats = {
            'total_pixels': total_pixels,
            'masked_pixels': masked_pixels,
            'coverage_ratio': masked_pixels / total_pixels if total_pixels > 0 else 0.0
        }

        return stats

    def expand_mask_by_distance(self, mask: np.ndarray, distance_pixels: int) -> np.ndarray:
        """
        マスクを距離によって膨張させる

        Args:
            mask: バイナリマスク
            distance_pixels: 膨張距離（ピクセル単位）

        Returns:
            距離膨張されたマスク
        """
        # 距離変換を計算
        dist_transform = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_PRECISE)

        # 距離がdistance_pixels以下の領域を拡張マスクとする
        expanded = (dist_transform <= distance_pixels).astype(np.uint8)

        return expanded

    def smooth_mask_boundaries(self, mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        マスクの境界をスムージング

        Args:
            mask: バイナリマスク
            kernel_size: スムージングカーネルサイズ

        Returns:
            境界がスムーズなマスク
        """
        # マスクを浮動小数点に変換
        mask_float = mask.astype(np.float32)

        # ガウシアンフィルタでスムージング
        smoothed = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)

        # 0.5でしきい値処理
        result = (smoothed > 0.5).astype(np.uint8)

        return result

    def create_soft_mask(self, mask: np.ndarray, blur_kernel_size: int = 31) -> np.ndarray:
        """
        ソフトマスク（グラデーション）を生成
        マスク領域は1、背景は0、境界はグラデーション

        Args:
            mask: バイナリマスク
            blur_kernel_size: ブラーカーネルサイズ

        Returns:
            ソフトマスク (H x W) 浮動小数点値
        """
        # バイナリマスクをブラー
        soft = mask.astype(np.float32)
        soft = cv2.GaussianBlur(soft, (blur_kernel_size, blur_kernel_size), 0)

        # 正規化
        soft = np.clip(soft, 0, 1)

        return soft

    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        マスクを反転（0と1を入れ替え）

        Args:
            mask: バイナリマスク

        Returns:
            反転されたマスク
        """
        return 1 - mask
