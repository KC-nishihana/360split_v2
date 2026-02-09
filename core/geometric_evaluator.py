"""
幾何学的評価モジュール - 360Split用
特徴点ベースの幾何学的性質を評価
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger('360split')


class GeometricEvaluator:
    """
    フレーム間の幾何学的性質の評価

    特徴点検出とマッチング、ホモグラフィ/基礎行列の比較、
    特徴点分布、3D光線の分散を評価する
    """

    def __init__(self, use_sift: bool = False):
        """
        初期化

        Parameters:
        -----------
        use_sift : bool
            SIFTを使用するか（False=ORB）
        """
        self.use_sift = use_sift

        if use_sift:
            try:
                self.detector = cv2.SIFT_create()
            except AttributeError:
                logger.warning("SIFTが利用できません。ORBを使用します")
                self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            self.detector = cv2.ORB_create(nfeatures=5000)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _detect_and_compute(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        特徴点を検出して記述子を計算

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）

        Returns:
        --------
        tuple
            (キーポイント, 記述子) のタプル
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray,
                       kp1: List, kp2: List) -> List[Tuple[int, int]]:
        """
        特徴点をマッチング

        Parameters:
        -----------
        desc1, desc2 : np.ndarray
            記述子
        kp1, kp2 : list
            キーポイント

        Returns:
        --------
        list
            マッチしたインデックスペアのリスト
        """
        if desc1 is None or desc2 is None:
            return []

        # K-NN マッチングで最良の2つを取得
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio testでフィルタリング
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))

        return good_matches

    def compute_gric_score(self, frame1: np.ndarray, frame2: np.ndarray,
                          ransac_threshold: float = 3.0) -> float:
        """
        GRIC スコア計算

        ホモグラフィと基礎行列のフィッティングを比較して、
        シーン内の視差（パラレックス）の度合いを評価。

        GRICスコア = min(error_H, error_F) / max(error_H, error_F)

        スコアが低いほど（視差が大きいほど）より良い

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        ransac_threshold : float
            RANSAC再投影誤差閾値

        Returns:
        --------
        float
            GRICスコア（0-1、低いほど視差あり）
        """
        kp1, desc1 = self._detect_and_compute(frame1)
        kp2, desc2 = self._detect_and_compute(frame2)

        matches = self._match_features(desc1, desc2, kp1, kp2)

        if len(matches) < 8:  # ホモグラフィと基礎行列計算に最低8点必要
            return 1.0  # 視差なし

        # マッチ点を座標に変換
        pts1 = np.float32([kp1[m[0]].pt for m in matches])
        pts2 = np.float32([kp2[m[1]].pt for m in matches])

        # ホモグラフィ行列を計算
        H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)

        # 基礎行列を計算
        F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_threshold)

        # 再投影誤差を計算
        error_H = np.inf
        error_F = np.inf

        if H is not None and mask_H is not None:
            # ホモグラフィの再投影誤差
            pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
            pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

            pts1_proj = (H @ pts1_h.T).T
            pts1_proj = pts1_proj[:, :2] / (pts1_proj[:, 2:3] + 1e-6)

            error_H = np.mean(np.linalg.norm(pts1_proj - pts2, axis=1))

        if F is not None and mask_F is not None:
            # 基礎行列の幾何学的誤差（Sampson distance）
            pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
            pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

            # エピポーラ線の計算
            l2 = (F @ pts1_h.T).T  # pts1対応のエピポーラ線
            l1 = (F.T @ pts2_h.T).T  # pts2対応のエピポーラ線

            # Sampson distance
            Fx1 = (F @ pts1_h.T).T
            Ftx2 = (F.T @ pts2_h.T).T

            x2Fx1 = np.sum(pts2_h * Fx1, axis=1)

            denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
            denom = np.maximum(denom, 1e-6)

            error_F = np.mean(x2Fx1 ** 2 / denom)

        # GRIC比率を計算
        if error_H < 1e-6 or error_F < 1e-6:
            gric_ratio = 0.0  # 視差なし（どちらかのモデルが完全にフィット）
        else:
            gric_ratio = min(error_H, error_F) / max(error_H, error_F)

        return float(np.clip(gric_ratio, 0.0, 1.0))

    def compute_feature_distribution(self, frame: np.ndarray) -> float:
        """
        特徴点分布スコア

        画像をグリッドに分割し、各セルの特徴点数の
        分布エントロピーを計算。均等に分布していほど
        スコアが高い。

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム

        Returns:
        --------
        float
            分布スコア（0-1、高いほど均等分布）
        """
        kp, _ = self._detect_and_compute(frame)

        if len(kp) == 0:
            return 0.0

        h, w = frame.shape[:2]

        # 画像を4x4のグリッドに分割
        grid_cols, grid_rows = 4, 4
        cell_width = w / grid_cols
        cell_height = h / grid_rows

        # 各セルの特徴点数をカウント
        cell_counts = np.zeros((grid_rows, grid_cols))

        for kp_point in kp:
            x, y = kp_point.pt
            col = int(x / cell_width)
            row = int(y / cell_height)
            col = min(col, grid_cols - 1)
            row = min(row, grid_rows - 1)
            cell_counts[row, col] += 1

        # セル数の分布を正規化
        cell_counts = cell_counts.flatten()
        total_count = np.sum(cell_counts)

        if total_count == 0:
            return 0.0

        # 確率分布として正規化
        prob_dist = cell_counts / total_count

        # エントロピーを計算（0 = 集中、高 = 均等分布）
        # 最大エントロピーは log(grid_cols * grid_rows)
        entropy = -np.sum(prob_dist[prob_dist > 0] * np.log(prob_dist[prob_dist > 0] + 1e-10))
        max_entropy = np.log(grid_rows * grid_cols)

        # 正規化されたスコア
        distribution_score = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(np.clip(distribution_score, 0.0, 1.0))

    def compute_feature_match_count(self, frame1: np.ndarray, frame2: np.ndarray) -> int:
        """
        フレーム間のロバスト特徴マッチ数を計算

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム

        Returns:
        --------
        int
            マッチした特徴点数
        """
        kp1, desc1 = self._detect_and_compute(frame1)
        kp2, desc2 = self._detect_and_compute(frame2)

        matches = self._match_features(desc1, desc2, kp1, kp2)

        return len(matches)

    def compute_ray_dispersion(self, frame: np.ndarray,
                              is_equirectangular: bool = False) -> float:
        """
        特徴点光線の分散スコア

        特徴点に対応する光線（画像座標をカメラ光線に変換）の
        3D空間における分散を計算。高いほど多様な方向から
        観察している。

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム
        is_equirectangular : bool
            エクイレクタングラ画像か

        Returns:
        --------
        float
            光線分散スコア（0-1）
        """
        kp, _ = self._detect_and_compute(frame)

        if len(kp) < 4:
            return 0.0

        h, w = frame.shape[:2]

        # 正規化画像座標に変換
        rays = []
        for kp_point in kp[:100]:  # 計算量制限
            x, y = kp_point.pt

            if is_equirectangular:
                # エクイレクタングラ画像の場合
                # 経度（-π to π）と緯度（-π/2 to π/2）に変換
                lon = 2 * np.pi * (x / w) - np.pi
                lat = np.pi * (y / h) - np.pi / 2

                # 3D単位ベクトルに変換（球面座標から直交座標）
                ray = np.array([
                    np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat)
                ])
            else:
                # 通常の画像の場合
                # 正規化座標（-1 to 1）
                nx = 2 * (x / w) - 1
                ny = 2 * (y / h) - 1

                # Plücker座標の概念を簡略化して使用
                ray = np.array([nx, ny, 1.0])
                ray = ray / (np.linalg.norm(ray) + 1e-6)

            rays.append(ray)

        rays = np.array(rays)

        # 光線の分散を計算（共分散行列の固有値）
        covariance = np.cov(rays.T)
        eigenvalues = np.linalg.eigvals(covariance)
        eigenvalues = np.real(eigenvalues)

        # 分散スコア（固有値のバランス）
        # 3つの固有値が等しいほど分散が高い
        max_eig = np.max(eigenvalues)
        if max_eig < 1e-6:
            return 0.0

        normalized_eigs = eigenvalues / max_eig

        # 均等性指標（1に近いほど均等）
        dispersion_score = np.min(normalized_eigs)

        return float(np.clip(dispersion_score, 0.0, 1.0))

    def evaluate(self, frame1: np.ndarray, frame2: np.ndarray,
                min_matches: int = 30,
                gric_threshold: float = 0.8) -> Dict[str, float]:
        """
        フレーム間の幾何学的性質を総合評価

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        min_matches : int
            評価に必要な最小マッチ数
        gric_threshold : float
            GRIC閾値

        Returns:
        --------
        dict
            評価スコア辞書：
            - 'gric': GRICスコア
            - 'feature_distribution_1': フレーム1の特徴点分布
            - 'feature_distribution_2': フレーム2の特徴点分布
            - 'feature_match_count': マッチ数
            - 'ray_dispersion': 光線分散スコア
        """
        gric_score = self.compute_gric_score(frame1, frame2)
        dist1 = self.compute_feature_distribution(frame1)
        dist2 = self.compute_feature_distribution(frame2)
        match_count = self.compute_feature_match_count(frame1, frame2)
        ray_disp = self.compute_ray_dispersion(frame1)

        return {
            'gric': gric_score,
            'feature_distribution_1': dist1,
            'feature_distribution_2': dist2,
            'feature_match_count': match_count,
            'ray_dispersion': ray_disp
        }
