"""
適応的選択モジュール - 360Split用
SSIM計算、光学フロー、カメラ動き推定
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger('360split')


class AdaptiveSelector:
    """
    フレーム間の相似度と動きに基づく適応的サンプリング制御

    SSIMで構造的な相似度を計算し、光学フローでカメラ動きを
    推定して、動きの大きさに応じてサンプリング間隔を調整する
    """

    @staticmethod
    def compute_ssim(frame1: np.ndarray, frame2: np.ndarray,
                    window_size: int = 11, sigma: float = 1.5) -> float:
        """
        構造的相似度（SSIM）を計算

        ガウシアンウィンドウを使用した標準的なSSIM計算を実装。
        skimageやscipy依存なしでnumpy/cv2のみを使用

        SSIM = (2*mu_x*mu_y + c1)(2*sigma_xy + c2) /
               ((mu_x^2 + mu_y^2 + c1)(sigma_x^2 + sigma_y^2 + c2))

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム（BGR形式またはグレースケール）
        window_size : int
            ガウシアンウィンドウサイズ（奇数）
        sigma : float
            ガウシアン標準偏差

        Returns:
        --------
        float
            SSIM値（-1～1、1が完全一致）
        """
        if frame1 is None or frame2 is None:
            return 0.0

        if frame1.shape != frame2.shape:
            return 0.0

        # グレースケール変換
        if len(frame1.shape) == 3:
            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            img1 = frame1
            img2 = frame2

        # float64に変換
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # ガウシアンウィンドウを生成
        kernel = cv2.getGaussianKernel(window_size, sigma)
        window = kernel @ kernel.T

        # 定数の定義
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # 平均値を計算
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # 分散と共分散を計算
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

        # SSIM mapを計算
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2

        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)

        # 平均SSIMを返す
        return float(np.mean(ssim_map))

    @staticmethod
    def compute_optical_flow_magnitude(frame1: np.ndarray, frame2: np.ndarray,
                                      method: str = 'farneback') -> float:
        """
        フレーム間の光学フロー大きさを計算

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        method : str
            光学フロー計算方式（'farneback' または 'lucas_kanade'）

        Returns:
        --------
        float
            平均光学フロー大きさ（0以上）
        """
        if frame1 is None or frame2 is None:
            return 0.0

        # グレースケール変換
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        if gray1.shape != gray2.shape:
            return 0.0

        try:
            if method == 'lucas_kanade':
                # ゴッドスピード光学フロー（疎フロー）
                corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100,
                                                 qualityLevel=0.01, minDistance=10)
                if corners is None or len(corners) == 0:
                    return 0.0

                flow, status, err = cv2.calcOpticalFlowPyrLK(
                    gray1, gray2, corners, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

                if status is None:
                    return 0.0

                # 有効なフローのみ取得
                valid_flow = flow[status.flatten() == 1]
                if len(valid_flow) == 0:
                    return 0.0

                magnitudes = np.linalg.norm(valid_flow, axis=1)

            else:  # farneback
                # Farnebäck光学フロー（密フロー）
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # フロー大きさを計算
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes = magnitude.flatten()

            mean_magnitude = float(np.mean(magnitudes))

        except Exception as e:
            logger.warning(f"光学フロー計算エラー: {e}")
            return 0.0

        return mean_magnitude

    @staticmethod
    def compute_camera_momentum(frames_window: List[np.ndarray],
                               method: str = 'farneback') -> float:
        """
        カメラ動き加速度を推定

        フレームウィンドウ内の連続する光学フロー大きさの
        変化率（加速度）を計算。大きい値は急加速を示す。

        Parameters:
        -----------
        frames_window : list of np.ndarray
            フレームのリスト（最低3フレーム推奨）
        method : str
            光学フロー計算方式

        Returns:
        --------
        float
            カメラ加速度（0以上）
        """
        if frames_window is None or len(frames_window) < 2:
            return 0.0

        # 連続フレーム間の光学フロー大きさを計算
        flow_magnitudes = []
        for i in range(len(frames_window) - 1):
            mag = AdaptiveSelector.compute_optical_flow_magnitude(
                frames_window[i], frames_window[i + 1], method
            )
            flow_magnitudes.append(mag)

        if len(flow_magnitudes) < 2:
            return 0.0

        # 動きの加速度を計算（光学フロー大きさの変化率）
        flow_magnitudes = np.array(flow_magnitudes)
        accelerations = np.abs(np.diff(flow_magnitudes))

        # 加速度の平均を返す
        momentum = float(np.mean(accelerations))

        return momentum

    @staticmethod
    def get_adaptive_interval(momentum: float, base_interval: int = 10,
                             boost_factor: float = 2.0,
                             min_interval: int = 5,
                             max_interval: int = 60) -> int:
        """
        カメラ加速度に基づいて適応的なサンプリング間隔を計算

        加速度が大きい場合（カメラが素早く動く場合）は
        サンプリング間隔を短くする（キーフレーム間隔を減らす）

        Parameters:
        -----------
        momentum : float
            カメラ加速度値
        base_interval : int
            基本サンプリング間隔（フレーム数）
        boost_factor : float
            加速度時の倍率調整係数
        min_interval : int
            最小間隔
        max_interval : int
            最大間隔

        Returns:
        --------
        int
            適応的なサンプリング間隔（フレーム数）
        """
        # 加速度をシグモイド関数でスケール
        # momentum = 0のとき間隔 = base_interval
        # momentum が大きいほど間隔が短くなる

        # シグモイド： 1 / (1 + exp(-x))を使用して
        # 加速度を0-1スケールに正規化
        scaled_momentum = 1.0 / (1.0 + np.exp(-momentum / 10.0))

        # 加速度に基づく調整係数（0.5-1.5の範囲）
        # momentum=0で1.0、momentum大で0.5に向かう
        adjustment = 1.5 - scaled_momentum  # 1.5 - (0-1) = 0.5-1.5

        # 適応間隔を計算
        adaptive_interval = base_interval / (adjustment * boost_factor)

        # 範囲に収める
        adaptive_interval = int(np.clip(adaptive_interval, min_interval, max_interval))

        return adaptive_interval

    @staticmethod
    def evaluate(frame1: np.ndarray, frame2: np.ndarray,
                frames_window: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """
        適応的選択用のスコアを計算

        Parameters:
        -----------
        frame1, frame2 : np.ndarray
            比較するフレーム
        frames_window : list of np.ndarray, optional
            カメラ加速度計算用のフレームウィンドウ

        Returns:
        --------
        dict
            評価スコア辞書：
            - 'ssim': SSIM相似度
            - 'optical_flow': 光学フロー大きさ
            - 'momentum': カメラ加速度
        """
        ssim = AdaptiveSelector.compute_ssim(frame1, frame2)
        optical_flow = AdaptiveSelector.compute_optical_flow_magnitude(frame1, frame2)

        momentum = 0.0
        if frames_window is not None:
            momentum = AdaptiveSelector.compute_camera_momentum(frames_window)

        return {
            'ssim': ssim,
            'optical_flow': optical_flow,
            'momentum': momentum
        }
