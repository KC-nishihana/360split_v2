"""
幾何学的評価モジュール - 360Split v2
GRIC (Geometric Robust Information Criterion) ベースの視差評価

GRIC = sum(rho(r_i^2, sigma)) + lambda1 * d * n + lambda2 * k

ホモグラフィH vs 基礎行列Fを比較して視差（パラレックス）を判定。
カスタム例外による縮退・推定失敗の明示的な通知。
360°ポーラーマスクによる歪み領域の除外。
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from core.accelerator import get_accelerator
from core.exceptions import (
    GeometricDegeneracyError,
    EstimationFailureError,
    InsufficientFeaturesError
)
from config import GRICConfig, Equirect360Config

from utils.logger import get_logger
logger = get_logger(__name__)


class FeatureCache:
    """
    特徴点検出結果をLRUキャッシュで管理

    フレームインデックスをキーとして、キーポイントと記述子を保存。
    最大50エントリを保持し、古いものは自動削除される。
    """

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self.cache = OrderedDict()

    def get(self, frame_idx: int) -> Optional[Tuple[List, np.ndarray]]:
        if frame_idx not in self.cache:
            return None
        self.cache.move_to_end(frame_idx)
        return self.cache[frame_idx]

    def put(self, frame_idx: int, keypoints: List, descriptors: np.ndarray):
        if frame_idx in self.cache:
            self.cache.move_to_end(frame_idx)
        else:
            self.cache[frame_idx] = (keypoints, descriptors)
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


class GeometricEvaluator:
    """
    GRICベースのフレーム間幾何学的評価 (v2)

    2Dモデル（ホモグラフィH）と3Dモデル（基礎行列F）の
    両方をRANSACで推定し、GRICで比較。

    GRIC判定:
    - GRIC_F < GRIC_H → 有効な並進運動（3D再構成に有用）→ 高スコア
    - GRIC_H <= GRIC_F → 回転のみ/平面シーン → 低スコア or 例外

    360°特有処理:
    - 天頂/天底ポーラーマスクによる歪み領域の除外

    最適化:
    - 特徴記述子のLRUキャッシング
    - FLANNマッチャーによる高速化
    - ベクトル化された残差・GRIC計算
    """

    def __init__(self, use_sift: bool = False,
                 gric_config: GRICConfig = None,
                 equirect_config: Equirect360Config = None):
        """
        初期化

        Parameters:
        -----------
        use_sift : bool
            SIFTを使用するか（False=ORB）
        gric_config : GRICConfig, optional
            GRIC計算パラメータ。Noneの場合はデフォルト値
        equirect_config : Equirect360Config, optional
            360°処理設定。Noneの場合はデフォルト値
        """
        self.use_sift = use_sift
        self.accelerator = get_accelerator()

        # GRIC設定
        self.gric_config = gric_config or GRICConfig()
        # 360°設定
        self.equirect_config = equirect_config or Equirect360Config()

        # 特徴検出器を初期化
        if use_sift:
            try:
                self.detector = cv2.SIFT_create()
            except AttributeError:
                logger.warning("SIFTが利用できません。ORBを使用します")
                self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            self.detector = cv2.ORB_create(nfeatures=5000)

        # FLANN マッチャーを初期化
        self._init_matcher()

        # 特徴記述子キャッシュ
        self.feature_cache = FeatureCache(max_entries=50)

    def _init_matcher(self):
        """
        最適なマッチャーを初期化

        FLANN (Fast Approximate Nearest Neighbors) を優先し、
        失敗時はBFMatcherにフォールバック
        """
        try:
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=12,
                key_size=20,
                multi_probe_level=2
            )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.use_flann = True
            logger.info("FLANN マッチャーを初期化")
        except Exception as e:
            logger.warning(f"FLANN初期化失敗、BFMatcherを使用: {e}")
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.use_flann = False

    def _create_polar_mask(self, h: int, w: int) -> np.ndarray:
        """
        360°画像用ポーラーマスクを生成

        天頂（上端）と天底（下端）の歪みが大きい領域をマスクする。

        Parameters:
        -----------
        h, w : int
            画像の高さと幅

        Returns:
        --------
        np.ndarray
            マスク画像 (uint8, 0=除外, 255=有効)
        """
        mask = np.ones((h, w), dtype=np.uint8) * 255
        margin = int(h * self.equirect_config.mask_polar_ratio)
        if margin > 0:
            mask[:margin, :] = 0       # 天頂マスク
            mask[h - margin:, :] = 0   # 天底マスク
        return mask

    def _detect_and_compute_cached(self, frame: np.ndarray,
                                   frame_idx: Optional[int] = None,
                                   use_polar_mask: bool = False,
                                   external_mask: Optional[np.ndarray] = None) -> Tuple[List, Optional[np.ndarray]]:
        """
        特徴点を検出して記述子を計算（キャッシュ・ポーラーマスク対応）

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）
        frame_idx : int, optional
            フレームインデックス（キャッシュキー）
        use_polar_mask : bool
            360°ポーラーマスクを適用するか

        Returns:
        --------
        tuple
            (キーポイント, 記述子) のタプル
        """
        use_cache = frame_idx is not None and external_mask is None
        # キャッシュをチェック
        if use_cache:
            cached = self.feature_cache.get(frame_idx)
            if cached is not None:
                return cached

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # ポーラーマスク適用
        mask = None
        if use_polar_mask and self.equirect_config.enable_polar_mask:
            h, w = gray.shape[:2]
            mask = self._create_polar_mask(h, w)
        if external_mask is not None:
            ext = self._normalize_external_feature_mask(external_mask, gray.shape[:2])
            mask = ext if mask is None else cv2.bitwise_and(mask, ext)

        # 特徴点検出と記述子計算
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)

        # キャッシュに保存
        if use_cache:
            self.feature_cache.put(
                frame_idx, keypoints,
                descriptors if descriptors is not None else np.array([])
            )

        return keypoints, descriptors

    @staticmethod
    def _normalize_external_feature_mask(
        external_mask: np.ndarray,
        expected_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        外部マスクを detectAndCompute 用に正規化する。
        戻り値は 255=有効, 0=無効。
        """
        if external_mask is None:
            return np.ones(expected_shape, dtype=np.uint8) * 255

        mask = external_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != expected_shape:
            mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = mask.astype(np.uint8)
        # 0/1 の場合は 1=除外領域として扱う
        if np.max(mask) <= 1:
            valid = (mask == 0).astype(np.uint8) * 255
            return valid
        # 0/255 の場合は 0=除外, 255=有効を想定
        valid = np.where(mask > 0, 255, 0).astype(np.uint8)
        return valid

    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray,
                       kp1: List, kp2: List) -> List[Tuple[int, int]]:
        """
        特徴点をマッチング（Lowe's ratio test）

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
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append((m.queryIdx, m.trainIdx))
            return good_matches
        except Exception as e:
            logger.warning(f"特徴点マッチング失敗: {e}")
            return []

    def _compute_reprojection_errors_H(self, pts1: np.ndarray, pts2: np.ndarray,
                                        H: np.ndarray) -> np.ndarray:
        """
        ホモグラフィの対称転送誤差を計算

        Parameters:
        -----------
        pts1, pts2 : np.ndarray
            対応点座標 (N, 2)
        H : np.ndarray
            ホモグラフィ行列 (3, 3)

        Returns:
        --------
        np.ndarray
            各点の対称転送誤差 (N,)
        """
        n = pts1.shape[0]
        if n == 0:
            return np.array([], dtype=np.float64)
        if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return np.full(n, np.inf, dtype=np.float64)

        # Forward: pts1 → H → pts2_pred
        pts1_h = np.hstack([pts1, np.ones((n, 1), dtype=pts1.dtype)])
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            pts2_pred = (H @ pts1_h.T).T
            w = pts2_pred[:, 2:3]
            w = np.where(np.abs(w) < 1e-10, 1e-10, w)
            pts2_pred = pts2_pred[:, :2] / w
            err_forward = np.sum((pts2_pred - pts2) ** 2, axis=1)
        err_forward = np.where(np.isfinite(err_forward), err_forward, np.inf)

        # Backward: pts2 → H_inv → pts1_pred
        try:
            H_inv = np.linalg.inv(H)
            if not np.all(np.isfinite(H_inv)):
                err_backward = np.full(n, np.inf, dtype=np.float64)
            else:
                pts2_h = np.hstack([pts2, np.ones((n, 1), dtype=pts2.dtype)])
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    pts1_pred = (H_inv @ pts2_h.T).T
                    w = pts1_pred[:, 2:3]
                    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
                    pts1_pred = pts1_pred[:, :2] / w
                    err_backward = np.sum((pts1_pred - pts1) ** 2, axis=1)
                err_backward = np.where(np.isfinite(err_backward), err_backward, np.inf)
        except np.linalg.LinAlgError:
            err_backward = err_forward

        return (err_forward + err_backward) / 2.0

    def _compute_sampson_errors_F(self, pts1: np.ndarray, pts2: np.ndarray,
                                   F: np.ndarray) -> np.ndarray:
        """
        基礎行列のSampson距離を計算

        Parameters:
        -----------
        pts1, pts2 : np.ndarray
            対応点座標 (N, 2)
        F : np.ndarray
            基礎行列 (3, 3)

        Returns:
        --------
        np.ndarray
            各点のSampson距離 (N,)
        """
        n = pts1.shape[0]
        if n == 0:
            return np.array([], dtype=np.float64)
        if F is None or F.shape != (3, 3) or not np.all(np.isfinite(F)):
            return np.full(n, np.inf, dtype=np.float64)
        pts1_h = np.hstack([pts1, np.ones((n, 1), dtype=pts1.dtype)])
        pts2_h = np.hstack([pts2, np.ones((n, 1), dtype=pts2.dtype)])

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Fx1 = (F @ pts1_h.T).T
            Ftx2 = (F.T @ pts2_h.T).T

            x2tFx1 = np.sum(pts2_h * Fx1, axis=1)

            denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
            denom = np.maximum(denom, 1e-10)
            sampson = x2tFx1**2 / denom

        return np.where(np.isfinite(sampson), sampson, np.inf)

    def _compute_gric(self, residuals: np.ndarray, sigma: float,
                      d_model: int, k_model: int, n_points: int,
                      lambda1: float, lambda2: float) -> float:
        """
        GRIC値を計算 (Torr 1998)

        GRIC = sum_i rho(e_i^2) + lambda1 * d_model * n + lambda2 * k_model

        rho(e^2) = min(e^2 / sigma^2, 2*(r - d_model))
        r = 4 (2D-2D対応のデータ空間次元)

        Parameters:
        -----------
        residuals : np.ndarray
            各対応点の残差 (N,)
        sigma : float
            残差標準偏差推定値
        d_model : int
            モデルの多様体次元 (H=2, F=3)
        k_model : int
            モデルパラメータ数 (H=8, F=7)
        n_points : int
            対応点数
        lambda1, lambda2 : float
            正則化係数

        Returns:
        --------
        float
            GRIC値
        """
        r = 4  # データ空間の次元（2D-2D対応）
        penalty = 2.0 * (r - d_model)

        # ロバスト誤差関数
        sigma_sq = sigma ** 2
        if sigma_sq < 1e-10:
            sigma_sq = 1e-10
        normalized_residuals = residuals / sigma_sq
        rho_values = np.minimum(normalized_residuals, penalty)

        gric = np.sum(rho_values) + lambda1 * d_model * n_points + lambda2 * k_model
        return float(gric)

    def compute_gric_score(self, frame1: np.ndarray, frame2: np.ndarray,
                           frame1_idx: Optional[int] = None,
                           frame2_idx: Optional[int] = None,
                           frame1_mask: Optional[np.ndarray] = None,
                           frame2_mask: Optional[np.ndarray] = None) -> float:
        """
        GRICベースの視差スコアを計算

        ホモグラフィHと基礎行列Fの両方をRANSACで推定し、
        それぞれのGRIC値を比較して視差の有無を判定する。

        判定ロジック:
        - GRIC_F < GRIC_H → 有効な視差あり（並進運動）→ 高スコア
        - GRIC_H <= GRIC_F → 回転のみ/平面 → 低スコア

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        frame1_idx, frame2_idx : int, optional
            フレームインデックス（キャッシュ用）

        Returns:
        --------
        float
            視差スコア (0.0-1.0, 高いほど有効な視差)

        Raises:
        -------
        InsufficientFeaturesError
            マッチ数が min_matches 未満の場合
        EstimationFailureError
            H/F行列の推定に失敗した場合
        GeometricDegeneracyError
            回転のみ/平面シーンと判定された場合
        """
        cfg = self.gric_config

        # 特徴点検出（360°ポーラーマスク適用）
        kp1, desc1 = self._detect_and_compute_cached(
            frame1, frame_idx=frame1_idx, use_polar_mask=True, external_mask=frame1_mask
        )
        kp2, desc2 = self._detect_and_compute_cached(
            frame2, frame_idx=frame2_idx, use_polar_mask=True, external_mask=frame2_mask
        )

        # マッチング
        matches = self._match_features(desc1, desc2, kp1, kp2)

        # 最小マッチ数チェック
        if len(matches) < cfg.min_matches:
            raise InsufficientFeaturesError(
                match_count=len(matches),
                required_count=cfg.min_matches
            )

        # 座標変換
        pts1 = np.float32([kp1[m[0]].pt for m in matches])
        pts2 = np.float32([kp2[m[1]].pt for m in matches])
        n_points = len(pts1)

        # ===== ホモグラフィH推定 =====
        try:
            H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, cfg.ransac_threshold)
        except cv2.error as e:
            raise EstimationFailureError(
                reason=f"ホモグラフィ推定でOpenCVエラー: {e}",
                match_count=n_points,
                required_count=cfg.min_matches
            )
        if H is None or mask_H is None:
            raise EstimationFailureError(
                reason="ホモグラフィ推定失敗（RANSAC収束不可）",
                match_count=n_points,
                required_count=cfg.min_matches
            )

        # ===== 基礎行列F推定 =====
        try:
            F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, cfg.ransac_threshold)
        except cv2.error as e:
            # OpenCV内部のアサーション失敗（特定の点配置で発生し得る）
            raise EstimationFailureError(
                reason=f"基礎行列推定でOpenCVエラー: {e}",
                match_count=n_points,
                required_count=cfg.min_matches
            )
        if F is None or mask_F is None:
            raise EstimationFailureError(
                reason="基礎行列推定失敗（RANSAC収束不可）",
                match_count=n_points,
                required_count=cfg.min_matches
            )

        # findFundamentalMatは複数解を返すことがある（9x3等）→ 先頭3x3を使用
        if F.shape[0] > 3:
            logger.debug(f"基礎行列が複数解 ({F.shape}) → 先頭3x3を使用")
            F = F[:3, :]

        # インライア率
        inlier_ratio_H = float(np.sum(mask_H.ravel())) / n_points
        inlier_ratio_F = float(np.sum(mask_F.ravel())) / n_points

        # 両方のインライア率が最低要件を満たさない場合
        if inlier_ratio_H < cfg.min_inlier_ratio and inlier_ratio_F < cfg.min_inlier_ratio:
            raise EstimationFailureError(
                reason=f"インライア率不足 (H={inlier_ratio_H:.2f}, F={inlier_ratio_F:.2f})",
                match_count=n_points,
                required_count=cfg.min_matches
            )

        # ===== 残差計算 =====
        errors_H = self._compute_reprojection_errors_H(pts1, pts2, H)
        errors_F = self._compute_sampson_errors_F(pts1, pts2, F)

        # ===== GRIC計算 =====
        # H: d=2 (2DOF/点), k=8 | F: d=3 (3DOF/点), k=7
        gric_H = self._compute_gric(
            errors_H, cfg.sigma, d_model=2, k_model=8,
            n_points=n_points, lambda1=cfg.lambda1, lambda2=cfg.lambda2
        )
        gric_F = self._compute_gric(
            errors_F, cfg.sigma, d_model=3, k_model=7,
            n_points=n_points, lambda1=cfg.lambda1, lambda2=cfg.lambda2
        )

        logger.debug(
            f"GRIC: H={gric_H:.2f}, F={gric_F:.2f}, "
            f"inlier_H={inlier_ratio_H:.3f}, inlier_F={inlier_ratio_F:.3f}, "
            f"matches={n_points}"
        )

        # ===== 縮退判定: Hのインライア率が非常に高い =====
        if inlier_ratio_H >= cfg.degeneracy_threshold:
            raise GeometricDegeneracyError(
                message="Hインライア率が高すぎる（純回転/平面シーン）",
                gric_h=gric_H,
                gric_f=gric_F,
                inlier_ratio_h=inlier_ratio_H
            )

        # ===== GRIC比較による視差スコア算出 =====
        if gric_F < gric_H:
            # Fモデルが優位 → 有効な並進運動（視差あり）
            gric_sum = gric_H + gric_F + 1e-10
            ratio = (gric_H - gric_F) / gric_sum
            score = float(np.clip(0.5 + ratio, 0.3, 1.0))
        else:
            # Hモデルが優位 → 回転のみ/平面の傾向
            gric_sum = gric_H + gric_F + 1e-10
            ratio = (gric_F - gric_H) / gric_sum
            score = float(np.clip(0.5 - ratio, 0.0, 0.5))

            # 明確にHが優位（スコアが非常に低い）場合は縮退
            if score < 0.15:
                raise GeometricDegeneracyError(
                    message="GRIC判定: ホモグラフィが明確に優位（視差不十分）",
                    gric_h=gric_H,
                    gric_f=gric_F,
                    inlier_ratio_h=inlier_ratio_H
                )

        return score

    def compute_feature_distribution(self, frame: np.ndarray,
                                     frame_idx: Optional[int] = None,
                                     feature_mask: Optional[np.ndarray] = None) -> float:
        """
        特徴点分布スコア

        画像をグリッドに分割し、各セルの特徴点数の
        分布エントロピーを計算。均等に分布しているほど
        スコアが高い。

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム
        frame_idx : int, optional
            フレームインデックス

        Returns:
        --------
        float
            分布スコア（0-1、高いほど均等分布）
        """
        kp, _ = self._detect_and_compute_cached(
            frame, frame_idx=frame_idx, use_polar_mask=True, external_mask=feature_mask
        )

        if len(kp) == 0:
            return 0.0

        h, w = frame.shape[:2]
        grid_cols, grid_rows = 4, 4

        # キーポイント座標を抽出（ベクトル化）
        kp_coords = np.array([kp_point.pt for kp_point in kp])
        x_coords = kp_coords[:, 0]
        y_coords = kp_coords[:, 1]

        # np.histogram2d() で分布を計算
        cell_counts, _, _ = np.histogram2d(
            x_coords, y_coords,
            bins=[grid_cols, grid_rows],
            range=[[0, w], [0, h]]
        )

        cell_counts = cell_counts.flatten()
        total_count = np.sum(cell_counts)

        if total_count == 0:
            return 0.0

        # 確率分布として正規化
        prob_dist = cell_counts / total_count

        # エントロピーを計算
        entropy = -np.sum(prob_dist[prob_dist > 0] * np.log(prob_dist[prob_dist > 0] + 1e-10))
        max_entropy = np.log(grid_rows * grid_cols)

        distribution_score = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(np.clip(distribution_score, 0.0, 1.0))

    def compute_feature_match_count(self, frame1: np.ndarray, frame2: np.ndarray,
                                    frame1_idx: Optional[int] = None,
                                    frame2_idx: Optional[int] = None,
                                    frame1_mask: Optional[np.ndarray] = None,
                                    frame2_mask: Optional[np.ndarray] = None) -> int:
        """
        フレーム間のロバスト特徴マッチ数を計算

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        frame1_idx, frame2_idx : int, optional
            フレームインデックス

        Returns:
        --------
        int
            マッチした特徴点数
        """
        kp1, desc1 = self._detect_and_compute_cached(
            frame1, frame_idx=frame1_idx, use_polar_mask=True, external_mask=frame1_mask
        )
        kp2, desc2 = self._detect_and_compute_cached(
            frame2, frame_idx=frame2_idx, use_polar_mask=True, external_mask=frame2_mask
        )

        matches = self._match_features(desc1, desc2, kp1, kp2)
        return len(matches)

    def compute_ray_dispersion(self, frame: np.ndarray,
                              is_equirectangular: bool = False,
                              frame_idx: Optional[int] = None,
                              feature_mask: Optional[np.ndarray] = None) -> float:
        """
        特徴点光線の分散スコア

        特徴点に対応する光線の3D空間における分散を計算。
        高いほど多様な方向から観察している。

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム
        is_equirectangular : bool
            エクイレクタングラ画像か
        frame_idx : int, optional
            フレームインデックス

        Returns:
        --------
        float
            光線分散スコア（0-1）
        """
        kp, _ = self._detect_and_compute_cached(
            frame, frame_idx=frame_idx, use_polar_mask=True, external_mask=feature_mask
        )

        if len(kp) < 4:
            return 0.0

        h, w = frame.shape[:2]

        # キーポイント数を制限
        kp_subset = kp[:500]

        kp_coords = np.array([kp_point.pt for kp_point in kp_subset])
        x_coords = kp_coords[:, 0]
        y_coords = kp_coords[:, 1]

        if is_equirectangular:
            lon = 2 * np.pi * (x_coords / w) - np.pi
            lat = np.pi * (y_coords / h) - np.pi / 2

            rays = np.column_stack([
                np.cos(lat) * np.cos(lon),
                np.cos(lat) * np.sin(lon),
                np.sin(lat)
            ])
        else:
            nx = 2 * (x_coords / w) - 1
            ny = 2 * (y_coords / h) - 1

            rays = np.column_stack([nx, ny, np.ones_like(nx)])
            norms = np.linalg.norm(rays, axis=1, keepdims=True) + 1e-6
            rays = rays / norms

        # 光線の分散を計算（共分散行列の固有値）
        covariance = np.cov(rays.T)
        eigenvalues = np.linalg.eigvals(covariance)
        eigenvalues = np.real(eigenvalues)

        max_eig = np.max(eigenvalues)
        if max_eig < 1e-6:
            return 0.0

        normalized_eigs = eigenvalues / max_eig
        dispersion_score = np.min(normalized_eigs)

        return float(np.clip(dispersion_score, 0.0, 1.0))

    def evaluate(self, frame1: np.ndarray, frame2: np.ndarray,
                frame1_idx: Optional[int] = None,
                frame2_idx: Optional[int] = None,
                min_matches: int = None,
                gric_threshold: float = None,
                frame1_mask: Optional[np.ndarray] = None,
                frame2_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        フレーム間の幾何学的性質を総合評価

        GRICスコア計算でカスタム例外が発生した場合、
        呼び出し元（KeyframeSelector）で適切にハンドリングされる。

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        frame1_idx, frame2_idx : int, optional
            フレームインデックス
        min_matches : int, optional
            (後方互換用、GRICConfigの値が優先)
        gric_threshold : float, optional
            (後方互換用、GRICConfigの値が優先)

        Returns:
        --------
        dict
            評価スコア辞書：
            - 'gric': GRICベースの視差スコア (0-1)
            - 'feature_distribution_1': フレーム1の特徴点分布
            - 'feature_distribution_2': フレーム2の特徴点分布
            - 'feature_match_count': マッチ数
            - 'ray_dispersion': 光線分散スコア

        Raises:
        -------
        GeometricDegeneracyError
            視差不十分（回転のみ/平面シーン）
        EstimationFailureError
            行列推定失敗
        InsufficientFeaturesError
            特徴点マッチ不足
        """
        # GRIC スコア計算（カスタム例外が発生し得る → 呼び出し元でハンドリング）
        gric_score = self.compute_gric_score(
            frame1, frame2,
            frame1_idx=frame1_idx,
            frame2_idx=frame2_idx,
            frame1_mask=frame1_mask,
            frame2_mask=frame2_mask,
        )

        # 補助スコア計算（OpenCVエラーに対する安全ネット付き）
        try:
            dist1 = self.compute_feature_distribution(
                frame1, frame_idx=frame1_idx, feature_mask=frame1_mask
            )
            dist2 = self.compute_feature_distribution(
                frame2, frame_idx=frame2_idx, feature_mask=frame2_mask
            )
            match_count = self.compute_feature_match_count(
                frame1, frame2,
                frame1_idx=frame1_idx,
                frame2_idx=frame2_idx,
                frame1_mask=frame1_mask,
                frame2_mask=frame2_mask,
            )
            ray_disp = self.compute_ray_dispersion(
                frame1, frame_idx=frame1_idx, feature_mask=frame1_mask
            )
        except cv2.error as e:
            logger.warning(f"補助スコア計算でOpenCVエラー（GRICスコアのみ使用）: {e}")
            dist1, dist2 = 0.5, 0.5
            match_count = 0
            ray_disp = 0.5

        return {
            'gric': gric_score,
            'feature_distribution_1': dist1,
            'feature_distribution_2': dist2,
            'feature_match_count': match_count,
            'ray_dispersion': ray_disp
        }
