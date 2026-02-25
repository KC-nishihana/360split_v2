"""
品質評価モジュール - 360Split用
画像の鮮明度、モーションブラー、露光を評価。
GPU加速、数値安定性、メモリ効率化を実装。
"""

import cv2
import numpy as np
from typing import Dict, Optional

from core.accelerator import get_accelerator

from utils.logger import get_logger
logger = get_logger(__name__)


class QualityEvaluator:
    """
    フレーム品質の多面的評価

    鮮明度、モーションブラー、露光、深度スコアを計算して
    キーフレーム選択の品質スコアを提供する。

    主な最適化：
    - グレースケール変換の共有（evaluate()で1回のみ）
    - Sobel勾配の計算共有
    - GPU加速（ラプラシアン、フィルタ操作）
    - ダウンスケール評価オプション
    - Log-sum-exp トリックによる数値安定性
    """

    def __init__(self, eval_scale: float = 0.5, motion_blur_method: str = "legacy"):
        """
        初期化

        Parameters:
        -----------
        eval_scale : float
            評価スケール（1.0 = 元解像度、0.5 = 1/2解像度、0.25 = 1/4解像度）
            デフォルト0.5 = 1/2解像度で高速化
        """
        self._eval_scale = np.clip(eval_scale, 0.1, 1.0)
        self._accel = get_accelerator()
        self._motion_blur_method = str(motion_blur_method or "legacy").strip().lower()
        if self._motion_blur_method not in {"legacy", "angle_hist", "fft_hybrid"}:
            self._motion_blur_method = "legacy"

    def evaluate(self, frame: np.ndarray, beta: float = 5.0) -> Dict[str, float]:
        """
        フレームの全品質スコアを計算（インスタンスメソッド）

        最適化：
        1. グレースケール変換は1回のみ
        2. Sobel勾配も1回計算して複数メソッドで共有
        3. GPU加速（ラプラシアン）
        4. ダウンスケール処理による高速化

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
        if frame is None or frame.size == 0:
            return {
                'sharpness': 0.0,
                'motion_blur': 0.0,
                'exposure': 0.0,
                'softmax_depth': 0.0
            }

        # ダウンスケール処理
        if self._eval_scale < 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * self._eval_scale), int(w * self._eval_scale)
            frame_work = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_work = frame

        # グレースケール変換は1回のみ
        gray = cv2.cvtColor(frame_work, cv2.COLOR_BGR2GRAY) if len(frame_work.shape) == 3 else frame_work

        # Sobel勾配を計算（複数メソッドで共有）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 各品質指標を計算
        sharpness = self._compute_sharpness(gray)
        motion_blur = self._compute_motion_blur(sobel_x, sobel_y, gray=gray, method=self._motion_blur_method)
        exposure = self._compute_exposure_score(gray)
        softmax_depth = self._compute_softmax_depth_score(sobel_x, sobel_y, beta)

        return {
            'sharpness': sharpness,
            'motion_blur': motion_blur,
            'exposure': exposure,
            'softmax_depth': softmax_depth
        }

    def evaluate_stage1_fast(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Stage1用の軽量品質評価（softmax_depth計算を省略）

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）

        Returns:
        --------
        dict
            Stage1判定に必要な品質スコア辞書
        """
        if frame is None or frame.size == 0:
            return {
                'sharpness': 0.0,
                'motion_blur': 0.0,
                'exposure': 0.0,
                'softmax_depth': 0.0,
            }

        if self._eval_scale < 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * self._eval_scale), int(w * self._eval_scale)
            frame_work = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_work = frame

        gray = cv2.cvtColor(frame_work, cv2.COLOR_BGR2GRAY) if len(frame_work.shape) == 3 else frame_work
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        return {
            'sharpness': self._compute_sharpness(gray),
            'motion_blur': self._compute_motion_blur(sobel_x, sobel_y, gray=gray, method=self._motion_blur_method),
            'exposure': self._compute_exposure_score(gray),
            'softmax_depth': 0.0,
        }

    @staticmethod
    def evaluate_frame(frame: np.ndarray, beta: float = 5.0,
                      eval_scale: float = 0.5) -> Dict[str, float]:
        """
        スタティックメソッド（クラスレベルの便利メソッド）

        インスタンスを作成して evaluate を呼び出し。
        既存APIとの互換性を保つためのラッパー。

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR形式）
        beta : float
            Softmax-scaling深度の温度パラメータ
        eval_scale : float
            評価スケール

        Returns:
        --------
        dict
            品質評価スコア辞書
        """
        evaluator = QualityEvaluator(eval_scale=eval_scale)
        return evaluator.evaluate(frame, beta=beta)

    def _compute_sharpness(self, gray: np.ndarray) -> float:
        """
        ラプラシアンフィルタによる鮮明度計算

        ラプラシアンフィルタを適用して計算した分散値で、
        大きいほど鮮明（エッジが多い）。
        GPU加速を試行。

        Parameters:
        -----------
        gray : np.ndarray
            グレースケール画像

        Returns:
        --------
        float
            ラプラシアン分散スコア（0以上）
        """
        if gray is None or gray.size == 0:
            return 0.0

        # GPU加速版ラプラシアン（利用可能な場合）
        sharpness = self._accel.compute_laplacian_var(gray)

        return float(sharpness)

    def _compute_motion_blur(
        self,
        sobel_x: np.ndarray,
        sobel_y: np.ndarray,
        gray: Optional[np.ndarray] = None,
        method: str = "legacy",
    ) -> float:
        """
        モーションブラー検出スコア（共有Sobel版）

        方向勾配分析によってモーションブラーを検出。
        水平方向と垂直方向の勾配比率を用いて
        モーション方向を推定し、対応する方向の
        勾配エネルギーが高い場合はモーションブラーと判定。

        Parameters:
        -----------
        sobel_x : np.ndarray
            水平方向Sobel勾配
        sobel_y : np.ndarray
            垂直方向Sobel勾配

        Returns:
        --------
        float
            モーションブラースコア（0-1、大きいほどブラー多い）
        """
        if sobel_x is None or sobel_y is None:
            return 0.0

        method = str(method or "legacy").strip().lower()
        if method not in {"legacy", "angle_hist", "fft_hybrid"}:
            method = "legacy"

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
        legacy_score = float(np.clip(abs(direction_ratio - 0.5) * 2.0, 0.0, 1.0))
        if method == "legacy":
            return legacy_score

        angle_score = self._compute_motion_blur_angle_hist(sobel_x, sobel_y)
        if method == "angle_hist":
            return angle_score

        if gray is None or gray.size == 0:
            return float(np.clip(0.5 * legacy_score + 0.5 * angle_score, 0.0, 1.0))
        fft_score = self._compute_motion_blur_fft(gray)
        # legacy互換を維持しつつ、方向性/高周波劣化を加味
        return float(np.clip(0.25 * legacy_score + 0.35 * angle_score + 0.40 * fft_score, 0.0, 1.0))

    def _compute_motion_blur_angle_hist(self, sobel_x: np.ndarray, sobel_y: np.ndarray) -> float:
        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        if mag.size == 0:
            return 0.0
        angle = (np.arctan2(sobel_y, sobel_x) + np.pi) % np.pi
        bins = 18
        hist, _ = np.histogram(angle, bins=bins, range=(0.0, np.pi), weights=mag)
        total = float(np.sum(hist))
        if total <= 1e-6:
            return 0.0
        p = hist / total
        entropy = -np.sum(p[p > 0.0] * np.log2(p[p > 0.0]))
        max_entropy = np.log2(bins)
        concentration = 1.0 - float(entropy / max(max_entropy, 1e-6))
        return float(np.clip(concentration, 0.0, 1.0))

    def _compute_motion_blur_fft(self, gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        # 360動画で向き依存を抑えるため4象限で集計
        hs = [0, h // 2, h]
        ws = [0, w // 2, w]
        scores = []
        for y0, y1 in ((hs[0], hs[1]), (hs[1], hs[2])):
            for x0, x1 in ((ws[0], ws[1]), (ws[1], ws[2])):
                if y1 - y0 < 8 or x1 - x0 < 8:
                    continue
                patch = gray[y0:y1, x0:x1].astype(np.float32)
                f = np.fft.fftshift(np.fft.fft2(patch))
                power = np.log1p(np.abs(f))
                ph, pw = power.shape
                cy, cx = ph // 2, pw // 2
                yy, xx = np.ogrid[:ph, :pw]
                rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                r0 = 0.15 * min(ph, pw)
                r1 = 0.40 * min(ph, pw)
                high = power[rr >= r1]
                mid = power[(rr >= r0) & (rr < r1)]
                if high.size == 0 or mid.size == 0:
                    continue
                ratio = float(np.mean(high) / max(np.mean(mid), 1e-6))
                scores.append(float(np.clip(1.0 - ratio, 0.0, 1.0)))
        if not scores:
            return 0.0
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _compute_exposure_score(self, gray: np.ndarray) -> float:
        """
        露光スコア計算

        フレームの平均輝度を評価し、適切な露光の場合は
        スコアが高く、暗すぎるまたは明るすぎる場合は低い
        スコアを返す

        Parameters:
        -----------
        gray : np.ndarray
            グレースケール画像

        Returns:
        --------
        float
            露光スコア（0-1、1が最適）
        """
        if gray is None or gray.size == 0:
            return 0.0

        # 平均輝度を計算
        mean_brightness = float(np.mean(gray))

        # 最適輝度は128付近（0-255の中央値）
        # ガウス分布で評価
        optimal_brightness = 128.0
        sigma = 80.0  # 標準偏差

        # 正規分布に基づくスコア
        exposure_score = np.exp(-((mean_brightness - optimal_brightness) ** 2) / (2 * sigma ** 2))

        return float(exposure_score)

    def _compute_softmax_depth_score(self, sobel_x: np.ndarray,
                                     sobel_y: np.ndarray,
                                     beta: float = 5.0) -> float:
        """
        Softmax-scaling深度スコア計算（共有Sobel版、数値安定化）

        勾配ベースの深度代理値を用いて、Softmax-scalingの概念を
        実装。エッジ信頼度を重みとして、加重平均深度を計算する：

        score = log(sum(w_i * exp(beta * w_i) * d_i) / sum(w_i * exp(beta * w_i)))

        Log-sum-exp トリックを使用して数値安定性を確保。

        Parameters:
        -----------
        sobel_x : np.ndarray
            水平方向Sobel勾配
        sobel_y : np.ndarray
            垂直方向Sobel勾配
        beta : float
            温度パラメータ（大きいほどシャープな重み付け）

        Returns:
        --------
        float
            Softmax-scaling深度スコア（0-1に正規化）
        """
        if sobel_x is None or sobel_y is None:
            return 0.5

        # 勾配の大きさを計算
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # エッジ信頼度を正規化（0-1）
        max_grad = np.max(gradient_magnitude)
        if max_grad < 1e-6:
            return 0.5

        edge_confidence = gradient_magnitude / max_grad

        # 深度代理値（勾配の大きさを0-1に正規化）
        depth_proxy = edge_confidence

        # フラット化してベクトル化計算
        w = edge_confidence.flatten()
        d = depth_proxy.flatten()

        # ゼロを除外
        nonzero_mask = w > 0.01
        if not np.any(nonzero_mask):
            return 0.5  # デフォルト値

        w = w[nonzero_mask]
        d = d[nonzero_mask]

        # Log-sum-exp トリック for numerical stability
        # exp(beta * w) は大きな値になるため、log-sum-exp を使用
        beta_w = beta * w

        # 最大値を保存（数値安定化）
        max_beta_w = np.max(beta_w)

        # exp(beta * w - max_beta_w) を計算（オーバーフロー防止）
        exp_beta_w_stable = np.exp(beta_w - max_beta_w)

        # 加重合計（数値安定版）
        numerator = np.sum(w * exp_beta_w_stable * d)
        denominator = np.sum(w * exp_beta_w_stable)

        if denominator < 1e-6:
            return 0.5

        softmax_depth = numerator / denominator

        # ログスケール正規化（安定性）
        if softmax_depth > 0:
            score = np.log(softmax_depth + 1.0) / np.log(2.0)  # log2スケール
        else:
            score = 0.0

        # 0-1に正規化
        score = float(np.clip(score, 0.0, 1.0))

        return score
