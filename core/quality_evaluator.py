"""
品質評価モジュール - 360Split用
画像の鮮明度、モーションブラー、露光を評価
"""

import cv2
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger('360split')


class QualityEvaluator:
    """
    フレーム品質の多面的評価

    鮮明度、モーションブラー、露光、深度スコアを計算して
    キーフレーム選択の品質スコアを提供する。
    """

    @staticmethod
    def compute_sharpness(frame: np.ndarray) -> float:
        """
        ラプラシアンフィルタによる鮮明度計算

        ラプラシアンフィルタを適用して計算した分散値で、
        大きいほど鮮明（エッジが多い）

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）

        Returns:
        --------
        float
            ラプラシアン分散スコア（0以上）
        """
        if frame is None or frame.size == 0:
            return 0.0

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # ラプラシアンフィルタ適用
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # 分散を計算（鮮明度指標）
        sharpness = laplacian.var()

        return float(sharpness)

    @staticmethod
    def compute_motion_blur(frame: np.ndarray) -> float:
        """
        モーションブラー検出スコア

        方向勾配分析によってモーションブラーを検出。
        水平方向と垂直方向の勾配比率を用いて
        モーション方向を推定し、対応する方向の
        勾配エネルギーが高い場合はモーションブラーと判定

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）

        Returns:
        --------
        float
            モーションブラースコア（0-1、大きいほどブラー多い）
        """
        if frame is None or frame.size == 0:
            return 0.0

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Sobelフィルタで勾配を計算
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 勾配エネルギーを計算
        energy_x = np.sum(np.abs(sobel_x))
        energy_y = np.sum(np.abs(sobel_y))

        # 総エネルギー
        total_energy = energy_x + energy_y
        if total_energy < 1e-6:
            return 0.0

        # 方向バランスを計算（0.5に近いほどモーションブラーなし）
        direction_ratio = energy_x / total_energy

        # 偏った方向性があるほどモーションブラースコアが高い
        # 最も偏った場合（0 or 1）で1.0、バランスしている場合は0に近い
        motion_blur_score = abs(direction_ratio - 0.5) * 2.0

        return float(np.clip(motion_blur_score, 0.0, 1.0))

    @staticmethod
    def compute_exposure_score(frame: np.ndarray) -> float:
        """
        露光スコア計算

        フレームの平均輝度を評価し、適切な露光の場合は
        スコアが高く、暗すぎるまたは明るすぎる場合は低い
        スコアを返す

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）

        Returns:
        --------
        float
            露光スコア（0-1、1が最適）
        """
        if frame is None or frame.size == 0:
            return 0.0

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 平均輝度を計算
        mean_brightness = float(np.mean(gray))

        # 最適輝度は128付近（0-255の中央値）
        # ガウス分布で評価
        optimal_brightness = 128.0
        sigma = 80.0  # 標準偏差

        # 正規分布に基づくスコア
        exposure_score = np.exp(-((mean_brightness - optimal_brightness) ** 2) / (2 * sigma ** 2))

        return float(exposure_score)

    @staticmethod
    def compute_softmax_depth_score(frame: np.ndarray, beta: float = 5.0) -> float:
        """
        Softmax-scaling深度スコア計算

        勾配ベースの深度代理値を用いて、Softmax-scalingの概念を
        実装。エッジ信頼度を重みとして、加重平均深度を計算する：

        score = log(sum(w_i * exp(beta * w_i) * d_i) / sum(w_i * exp(beta * w_i)))

        ここで w_i はエッジ信頼度、d_i は深度代理値

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）
        beta : float
            温度パラメータ（大きいほどシャープな重み付け）

        Returns:
        --------
        float
            Softmax-scaling深度スコア（0-1に正規化）
        """
        if frame is None or frame.size == 0:
            return 0.0

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 勾配を計算（深度代理値として使用）
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 勾配の大きさを計算
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 勾配方向を計算（エッジ信頼度として）
        edge_direction = np.arctan2(sobely, sobelx)

        # エッジ信頼度を正規化（0-1）
        edge_confidence = gradient_magnitude / (np.max(gradient_magnitude) + 1e-6)

        # 深度代理値（勾配の大きさを0-1に正規化）
        depth_proxy = edge_confidence

        # フラットン化してベクトル化計算
        w = edge_confidence.flatten()
        d = depth_proxy.flatten()

        # ゼロを除外
        nonzero_mask = w > 0.01
        if not np.any(nonzero_mask):
            return 0.5  # デフォルト値

        w = w[nonzero_mask]
        d = d[nonzero_mask]

        # Softmax-scaling深度計算
        exp_beta_w = np.exp(beta * w)
        numerator = np.sum(w * exp_beta_w * d)
        denominator = np.sum(w * exp_beta_w)

        if denominator < 1e-6:
            return 0.5

        softmax_depth = numerator / denominator

        # ログスケールで計算（安定性）
        if softmax_depth > 0:
            score = np.log(softmax_depth + 1.0) / np.log(2.0)  # log2スケール
        else:
            score = 0.0

        # 0-1に正規化
        score = float(np.clip(score, 0.0, 1.0))

        return score

    @staticmethod
    def evaluate(frame: np.ndarray, beta: float = 5.0) -> Dict[str, float]:
        """
        フレームの全品質スコアを計算

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）
        beta : float
            Softmax-scaling深度の温度パラメータ

        Returns:
        --------
        dict
            以下のキーを持つ評価スコア辞書：
            - 'sharpness': 鮮明度スコア
            - 'motion_blur': モーションブラースコア
            - 'exposure': 露光スコア
            - 'softmax_depth': Softmax-scaling深度スコア
        """
        if frame is None:
            return {
                'sharpness': 0.0,
                'motion_blur': 0.0,
                'exposure': 0.0,
                'softmax_depth': 0.0
            }

        return {
            'sharpness': QualityEvaluator.compute_sharpness(frame),
            'motion_blur': QualityEvaluator.compute_motion_blur(frame),
            'exposure': QualityEvaluator.compute_exposure_score(frame),
            'softmax_depth': QualityEvaluator.compute_softmax_depth_score(frame, beta)
        }
