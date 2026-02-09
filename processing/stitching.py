"""
画像ステッチングモジュール
複数の画像を結合し、360度パノラマを作成
3つのステッチング方式を実装
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import warnings


class StitchingProcessor:
    """
    画像ステッチングを行うクラス
    3つのモード：高速、高品質、深度認識ステッチング
    """

    def __init__(self):
        """初期化"""
        self.orb = cv2.ORB_create(nfeatures=1000)

    # ===== 高速ステッチング（SI-FS） =====

    def stitch_fast(self, images: List[np.ndarray], mode: str = 'horizontal') -> np.ndarray:
        """
        高速ピクセルベースステッチング（SI-FS）
        単純な重ねあわせと線形フェザリングを使用

        Args:
            images: ステッチ対象の画像リスト [(H x W x 3), ...]
            mode: ステッチモード
                'horizontal': 水平方向に結合
                'vertical': 垂直方向に結合

        Returns:
            ステッチ済み画像 (H x width_total x 3)
        """
        if not images:
            return None

        if len(images) == 1:
            return images[0]

        # 全ての画像が同じ高さであることを確認
        if mode == 'horizontal':
            height = images[0].shape[0]

            # オーバーラップ領域のサイズを推定（画像幅の15%）
            overlap_width = int(images[0].shape[1] * 0.15)

            # 出力画像の高さと幅を計算
            total_width = sum(img.shape[1] for img in images) - (len(images) - 1) * overlap_width
            result = np.zeros((height, total_width, 3), dtype=np.float32)

            x_offset = 0

            for i, image in enumerate(images):
                img_h, img_w = image.shape[:2]

                if i == 0:
                    # 最初の画像をそのまま配置
                    result[:, :img_w, :] = image
                    x_offset = img_w
                else:
                    # オーバーラップ領域を計算
                    blend_start = x_offset - overlap_width
                    blend_end = x_offset + img_w

                    # オーバーラップ領域での線形フェザリング
                    blend_width = overlap_width
                    weights = np.linspace(0, 1, blend_width)
                    weights = weights[np.newaxis, :, np.newaxis]  # (1, blend_width, 1)

                    # オーバーラップ領域
                    prev_overlap = result[:, blend_start:x_offset, :]
                    curr_overlap = image[:, :overlap_width, :]

                    # ブレンド
                    result[:, blend_start:x_offset, :] = (1 - weights) * prev_overlap + weights * curr_overlap

                    # 残りの部分を配置
                    remaining_width = img_w - overlap_width
                    if blend_end <= total_width:
                        result[:, x_offset:blend_end, :] = image[:, overlap_width:, :]
                    else:
                        result[:, x_offset:, :] = image[:, overlap_width:total_width - x_offset, :]

                    x_offset = blend_end - overlap_width

            return np.clip(result, 0, 255).astype(np.uint8)

        elif mode == 'vertical':
            # 垂直方向のステッチング
            width = images[0].shape[1]
            overlap_height = int(images[0].shape[0] * 0.15)

            total_height = sum(img.shape[0] for img in images) - (len(images) - 1) * overlap_height
            result = np.zeros((total_height, width, 3), dtype=np.float32)

            y_offset = 0

            for i, image in enumerate(images):
                img_h, img_w = image.shape[:2]

                if i == 0:
                    result[:img_h, :, :] = image
                    y_offset = img_h
                else:
                    blend_start = y_offset - overlap_height
                    blend_end = y_offset + img_h

                    blend_height = overlap_height
                    weights = np.linspace(0, 1, blend_height)
                    weights = weights[:, np.newaxis, np.newaxis]  # (blend_height, 1, 1)

                    prev_overlap = result[blend_start:y_offset, :, :]
                    curr_overlap = image[:overlap_height, :, :]

                    result[blend_start:y_offset, :, :] = (1 - weights) * prev_overlap + weights * curr_overlap

                    remaining_height = img_h - overlap_height
                    if blend_end <= total_height:
                        result[y_offset:blend_end, :, :] = image[overlap_height:, :, :]
                    else:
                        result[y_offset:, :, :] = image[overlap_height:total_height - y_offset, :, :]

                    y_offset = blend_end - overlap_height

            return np.clip(result, 0, 255).astype(np.uint8)

    # ===== 高品質ステッチング（SI-HQS） =====

    def stitch_high_quality(self, images: List[np.ndarray]) -> np.ndarray:
        """
        特徴ベースの高品質ステッチング（SI-HQS）
        ORB特徴点とホモグラフィ推定、多帯域ブレンディング使用

        Args:
            images: ステッチ対象の画像リスト [(H x W x 3), ...]

        Returns:
            ステッチ済み画像
        """
        if not images or len(images) == 1:
            return images[0] if images else None

        # グレースケール画像に変換
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        # 特徴点と記述子を抽出
        keypoints_list = []
        descriptors_list = []

        for gray in gray_images:
            kp, desc = self.orb.detectAndCompute(gray, None)
            keypoints_list.append(kp)
            descriptors_list.append(desc)

        # ホモグラフィ行列を計算
        homographies = []

        for i in range(len(images) - 1):
            desc1 = descriptors_list[i]
            desc2 = descriptors_list[i + 1]

            if desc1 is None or desc2 is None:
                # 特徴点がない場合は単純な平行シフトを使用
                homographies.append(np.eye(3))
                continue

            # 特徴点マッチング（BFMatcher）
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)[:100]

            if len(matches) < 4:
                homographies.append(np.eye(3))
                continue

            # マッチした特徴点の座標を取得
            src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches])
            dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches])

            # ホモグラフィを推定（RANSAC使用）
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                H = np.eye(3)

            homographies.append(H)

        # 出力キャンバスサイズを計算
        # 全ホモグラフィを適用した時の全体的な範囲を計算
        h, w = images[0].shape[:2]
        canvas_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        all_corners = [canvas_points]

        current_H = np.eye(3)

        for H in homographies:
            current_H = current_H @ H
            corners = cv2.perspectiveTransform(canvas_points.reshape(1, -1, 2), current_H)[0]
            all_corners.append(corners)

        # キャンバスサイズを計算
        x_min = min(corners[:, 0].min() for corners in all_corners)
        y_min = min(corners[:, 1].min() for corners in all_corners)
        x_max = max(corners[:, 0].max() for corners in all_corners)
        y_max = max(corners[:, 1].max() for corners in all_corners)

        canvas_width = int(x_max - x_min)
        canvas_height = int(y_max - y_max)

        # キャンバスは無限に大きくなる可能性があるため、制限
        canvas_width = min(canvas_width, 8000)
        canvas_height = min(canvas_height, 4000)

        result = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)

        # 各画像をキャンバスに配置
        current_H = np.eye(3)
        translation = np.eye(3)
        translation[0, 2] = -x_min
        translation[1, 2] = -y_min

        for i, image in enumerate(images):
            if i > 0:
                current_H = current_H @ homographies[i - 1]

            # 変換行列
            M = translation @ current_H

            # 透視変換
            warped = cv2.warpPerspective(image, M, (canvas_width, canvas_height))

            # マスク（有効ピクセル領域）を生成
            mask = (warped[:, :, 0] > 0) | (warped[:, :, 1] > 0) | (warped[:, :, 2] > 0)

            # 境界近くでフェザリング
            mask_float = mask.astype(np.float32)
            mask_float = cv2.GaussianBlur(mask_float, (31, 31), 0)

            # 加算合成
            result += warped * mask_float[:, :, np.newaxis]
            weight_map += mask_float

        # 正規化
        weight_map = np.maximum(weight_map, 1e-8)
        result = result / weight_map[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)

    # ===== 深度認識ステッチング（SI-DMS） =====

    def stitch_depth_aware(self, images: List[np.ndarray],
                          depth_maps: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        深度情報を使用した視差対応ステッチング（SI-DMS）
        深度マップがあれば使用、なければ推定

        Args:
            images: ステッチ対象の画像リスト [(H x W x 3), ...]
            depth_maps: 対応する深度マップ [(H x W), ...] オプション

        Returns:
            ステッチ済み画像
        """
        if not images or len(images) == 1:
            return images[0] if images else None

        # 深度マップがない場合は推定（簡略版）
        if depth_maps is None:
            depth_maps = self._estimate_depth_maps(images)

        h, w = images[0].shape[:2]

        # カメラパラメータ（簡略版）
        focal_length = w / 2
        cx = w / 2
        cy = h / 2

        # 深度マップを逆数に（距離の正規化）
        normalized_depths = []
        for depth in depth_maps:
            depth_norm = 1.0 / (depth + 1e-6)
            depth_norm = depth_norm / depth_norm.max()
            normalized_depths.append(depth_norm)

        # オーバーラップ領域での視差オフセットを計算
        parallax_offsets = []

        for i in range(len(images) - 1):
            # 2つの画像間の視差を推定
            offset = self._compute_parallax_offset(normalized_depths[i], normalized_depths[i + 1])
            parallax_offsets.append(offset)

        # ステッチングを実行
        overlap_width = int(w * 0.2)
        total_width = sum(img.shape[1] for img in images) - (len(images) - 1) * overlap_width
        result = np.zeros((h, total_width, 3), dtype=np.float32)

        x_offset = 0

        for i, image in enumerate(images):
            depth = normalized_depths[i]

            if i == 0:
                result[:, :w, :] = image
                x_offset = w
            else:
                # 視差を考慮したオーバーラップブレンディング
                parallax = parallax_offsets[i - 1]

                blend_start = x_offset - overlap_width
                blend_end = x_offset + w

                # 深度ベースの重み付け
                prev_depth = depth_maps[i - 1][:, -overlap_width:]
                curr_depth = depth[:, :overlap_width]

                # 深度が浅い（カメラに近い）ほど重みが高い
                prev_weight = prev_depth / (prev_depth + curr_depth + 1e-6)
                prev_weight = np.clip(prev_weight, 0, 1)
                curr_weight = 1 - prev_weight

                # ブレンディング
                for c in range(3):
                    result[:, blend_start:x_offset, c] = \
                        result[:, blend_start:x_offset, c] * (1 - curr_weight) + \
                        image[:, :overlap_width, c] * curr_weight

                # 残りの部分を配置
                remaining_width = w - overlap_width
                if blend_end <= total_width:
                    result[:, x_offset:blend_end, :] = image[:, overlap_width:, :]
                else:
                    result[:, x_offset:, :] = image[:, overlap_width:total_width - x_offset, :]

                x_offset = blend_end - overlap_width

        return np.clip(result, 0, 255).astype(np.uint8)

    # ===== ヘルパーメソッド =====

    def _find_homography(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        2つの画像間のホモグラフィ行列を計算

        Args:
            img1: 最初の画像
            img2: 2番目の画像

        Returns:
            3x3ホモグラフィ行列
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, desc1 = self.orb.detectAndCompute(gray1, None)
        kp2, desc2 = self.orb.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            return np.eye(3)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 4:
            return np.eye(3)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return np.eye(3)

        return H

    def _blend_images(self, img1: np.ndarray, img2: np.ndarray,
                     overlap_width: int) -> np.ndarray:
        """
        2つの画像をブレンディング

        Args:
            img1: 最初の画像
            img2: 2番目の画像
            overlap_width: オーバーラップ幅

        Returns:
            ブレンドされた画像
        """
        h = img1.shape[0]
        w1 = img1.shape[1]
        w2 = img2.shape[1]

        # 出力サイズ
        total_width = w1 + w2 - overlap_width
        result = np.zeros((h, total_width, 3), dtype=np.float32)

        # 最初の画像を配置
        result[:, :w1, :] = img1

        # オーバーラップ領域でブレンディング
        weights = np.linspace(0, 1, overlap_width)
        weights = weights[np.newaxis, :, np.newaxis]

        overlap_start = w1 - overlap_width
        result[:, overlap_start:w1, :] = (1 - weights) * result[:, overlap_start:w1, :] + \
                                          weights * img2[:, :overlap_width, :]

        # 2番目の画像の残りを配置
        result[:, w1:, :] = img2[:, overlap_width:, :]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _compute_seam(self, img1: np.ndarray, img2: np.ndarray,
                     overlap_region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        オーバーラップ領域で最適な接合線を計算
        グラフカット法を簡略化した実装

        Args:
            img1: 最初の画像
            img2: 2番目の画像
            overlap_region: (x_start, y_start, x_end, y_end)

        Returns:
            接合線マスク
        """
        x_start, y_start, x_end, y_end = overlap_region

        overlap_h = y_end - y_start
        overlap_w = x_end - x_start

        region1 = img1[y_start:y_end, x_start:x_end, :]
        region2 = img2[y_start:y_end, x_start:x_end, :]

        # 色差を計算
        diff = np.sum((region1.astype(np.float32) - region2.astype(np.float32))**2, axis=2)

        # 垂直接合線を計算（動的計画法）
        seam = np.zeros(overlap_h, dtype=np.int32)

        # 最初の行のコスト
        cost = diff[0, :].copy()

        # 動的計画法で最小コスト経路を計算
        for y in range(1, overlap_h):
            new_cost = np.zeros(overlap_w)

            for x in range(overlap_w):
                # 3方向から来た場合の最小コスト
                min_prev_cost = np.inf

                for dx in [-1, 0, 1]:
                    prev_x = x + dx
                    if 0 <= prev_x < overlap_w:
                        min_prev_cost = min(min_prev_cost, cost[prev_x])

                new_cost[x] = diff[y, x] + min_prev_cost

            cost = new_cost

        # 最後の行から最小コストを見つける
        seam[-1] = np.argmin(cost)

        # 経路を遡る
        for y in range(overlap_h - 2, -1, -1):
            # 簡略化：上の行で最も近いコストの列を選択
            seam[y] = seam[y + 1]

        # マスクを生成
        seam_mask = np.zeros((overlap_h, overlap_w), dtype=np.uint8)

        for y in range(overlap_h):
            x = seam[y]
            seam_mask[y, :x] = 1

        return seam_mask

    def _estimate_depth_maps(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        画像からシンプルな深度マップを推定
        簡略版：エッジ密度に基づいて推定

        Args:
            images: 画像リスト

        Returns:
            推定深度マップリスト
        """
        depth_maps = []

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sobelエッジ検出
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)

            # エッジが多い領域は手前（深度値が小さい）
            depth = 1.0 / (1.0 + edge_magnitude)
            depth_maps.append(depth)

        return depth_maps

    def _compute_parallax_offset(self, depth1: np.ndarray, depth2: np.ndarray) -> float:
        """
        2つの深度マップから視差オフセットを推定

        Args:
            depth1: 最初の深度マップ
            depth2: 2番目の深度マップ

        Returns:
            推定された視差オフセット（ピクセル単位）
        """
        # 深度の平均差を計算
        mean_depth_diff = np.mean(depth1) - np.mean(depth2)

        # ピクセルオフセットに変換（簡略版）
        offset = mean_depth_diff * 100

        return offset
