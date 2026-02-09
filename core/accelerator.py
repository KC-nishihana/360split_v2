"""
ハードウェアアクセラレータ抽象化レイヤー - 360Split

Apple Silicon (Metal/MPS) と Windows (CUDA) の両環境で
最適なバックエンドを自動検出・選択する統一API。

検出優先順位:
  macOS:  Metal/MPS (PyTorch) → OpenCV CPU最適化 → NumPy
  Windows: CUDA (cv2.cuda / PyTorch) → OpenCV CPU最適化 → NumPy
"""

import platform
import logging
import os
from enum import Enum
from typing import Optional
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger('360split')


class AccelBackend(Enum):
    """利用可能なアクセラレーションバックエンド"""
    CUDA = "cuda"               # NVIDIA CUDA (Windows/Linux)
    MPS = "mps"                 # Apple Metal Performance Shaders
    OPENCV_CUDA = "opencv_cuda" # OpenCV CUDAモジュール
    CPU_OPTIMIZED = "cpu"       # マルチスレッドCPU
    NUMPY = "numpy"             # NumPyフォールバック


class Accelerator:
    """
    ハードウェアアクセラレータ管理クラス

    環境を自動検出し、利用可能な最速のバックエンドを選択する。
    シングルトンパターンでアプリケーション全体で共有。
    """

    _instance: Optional['Accelerator'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._system = platform.system()            # Darwin, Windows, Linux
        self._machine = platform.machine()           # arm64, x86_64, AMD64
        self._backend = AccelBackend.NUMPY
        self._torch_device = None
        self._torch_available = False
        self._opencv_cuda_available = False
        self._cuda_device_name = ""
        self._mps_available = False
        self._num_threads = os.cpu_count() or 4

        self._detect_hardware()
        self._configure_opencv_threads()

        logger.info(f"アクセラレータ初期化完了: backend={self._backend.value}, "
                    f"system={self._system}/{self._machine}, threads={self._num_threads}")

    def _detect_hardware(self):
        """ハードウェアとライブラリの検出"""

        # --- OpenCV CUDAモジュール検出 ---
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self._opencv_cuda_available = True
                dev = cv2.cuda.getDevice()
                self._cuda_device_name = f"CUDA Device {dev}"
                logger.info(f"OpenCV CUDA 検出: {self._cuda_device_name}")
        except Exception:
            pass

        # --- PyTorch検出 ---
        try:
            import torch
            self._torch_available = True

            # CUDA (Windows/Linux, NVIDIA GPU)
            if torch.cuda.is_available():
                self._torch_device = torch.device("cuda")
                self._cuda_device_name = torch.cuda.get_device_name(0)
                self._backend = AccelBackend.CUDA
                logger.info(f"PyTorch CUDA 検出: {self._cuda_device_name}")
                return

            # MPS (macOS, Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._torch_device = torch.device("mps")
                self._mps_available = True
                self._backend = AccelBackend.MPS
                logger.info("PyTorch MPS (Apple Metal) 検出")
                return

        except ImportError:
            logger.info("PyTorchが未インストール（CPU最適化モードで動作）")

        # --- OpenCV CUDAフォールバック ---
        if self._opencv_cuda_available:
            self._backend = AccelBackend.OPENCV_CUDA
            return

        # --- CPU最適化 ---
        self._backend = AccelBackend.CPU_OPTIMIZED

    def _configure_opencv_threads(self):
        """OpenCVのスレッド数を最適化"""
        # Apple Siliconでは効率コア+パフォーマンスコアの構成を考慮
        if self._system == "Darwin" and self._machine == "arm64":
            # Apple Silicon: パフォーマンスコア数を推定（全コアの半分）
            optimal_threads = max(self._num_threads // 2, 4)
        else:
            optimal_threads = self._num_threads

        cv2.setNumThreads(optimal_threads)
        logger.info(f"OpenCVスレッド数: {optimal_threads}")

    # === プロパティ ===

    @property
    def backend(self) -> AccelBackend:
        """現在のバックエンド"""
        return self._backend

    @property
    def has_gpu(self) -> bool:
        """GPU利用可能か"""
        return self._backend in (AccelBackend.CUDA, AccelBackend.MPS, AccelBackend.OPENCV_CUDA)

    @property
    def has_torch(self) -> bool:
        """PyTorch利用可能か"""
        return self._torch_available

    @property
    def has_cuda(self) -> bool:
        """CUDA利用可能か"""
        return self._backend == AccelBackend.CUDA or self._opencv_cuda_available

    @property
    def has_mps(self) -> bool:
        """Apple Metal MPS利用可能か"""
        return self._mps_available

    @property
    def torch_device(self):
        """PyTorchデバイス（未インストール時はNone）"""
        return self._torch_device

    @property
    def num_threads(self) -> int:
        """最適スレッド数"""
        return self._num_threads

    @property
    def device_name(self) -> str:
        """デバイス名の文字列表現"""
        if self._backend == AccelBackend.CUDA:
            return f"CUDA: {self._cuda_device_name}"
        elif self._backend == AccelBackend.MPS:
            return f"Apple Metal (MPS): {self._machine}"
        elif self._backend == AccelBackend.OPENCV_CUDA:
            return f"OpenCV CUDA: {self._cuda_device_name}"
        else:
            return f"CPU ({self._num_threads} threads)"

    # === GPU画像処理ユーティリティ ===

    def to_gpu(self, frame: np.ndarray):
        """
        フレームをGPUメモリに転送

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム (BGR, uint8)

        Returns:
        --------
        GPU上のフレーム（バックエンド依存の型）
        """
        if self._backend == AccelBackend.OPENCV_CUDA or (
            self._backend == AccelBackend.CUDA and self._opencv_cuda_available
        ):
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(frame)
            return gpu_mat

        if self._torch_available and self._torch_device is not None:
            import torch
            tensor = torch.from_numpy(frame).to(self._torch_device)
            return tensor

        return frame

    def to_cpu(self, gpu_frame) -> np.ndarray:
        """
        GPUフレームをCPUに戻す

        Parameters:
        -----------
        gpu_frame : GPU上のフレーム

        Returns:
        --------
        np.ndarray
            CPUフレーム
        """
        if isinstance(gpu_frame, np.ndarray):
            return gpu_frame

        if hasattr(gpu_frame, 'download'):
            # cv2.cuda_GpuMat
            return gpu_frame.download()

        if hasattr(gpu_frame, 'cpu'):
            # torch.Tensor
            return gpu_frame.cpu().numpy()

        return np.array(gpu_frame)

    def gpu_cvtColor(self, frame, code: int) -> np.ndarray:
        """
        GPU高速化された色空間変換

        Parameters:
        -----------
        frame : np.ndarray or GpuMat
            入力フレーム
        code : int
            OpenCV色変換コード

        Returns:
        --------
        np.ndarray
            変換後フレーム
        """
        if self._opencv_cuda_available and isinstance(frame, np.ndarray):
            try:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(frame)
                result = cv2.cuda.cvtColor(gpu_mat, code)
                return result.download()
            except Exception:
                pass

        return cv2.cvtColor(frame, code)

    def gpu_filter2D(self, src: np.ndarray, ddepth: int,
                     kernel: np.ndarray) -> np.ndarray:
        """
        GPU高速化された2Dフィルタ

        Parameters:
        -----------
        src : np.ndarray
            入力画像
        ddepth : int
            出力深度
        kernel : np.ndarray
            畳み込みカーネル

        Returns:
        --------
        np.ndarray
            フィルタ適用後画像
        """
        if self._opencv_cuda_available:
            try:
                gpu_src = cv2.cuda_GpuMat()
                gpu_src.upload(src)
                conv = cv2.cuda.createLinearFilter(
                    src.dtype, ddepth, kernel
                )
                gpu_dst = conv.apply(gpu_src)
                return gpu_dst.download()
            except Exception:
                pass

        return cv2.filter2D(src, ddepth, kernel)

    def gpu_resize(self, src: np.ndarray, dsize: tuple,
                   interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        GPU高速化されたリサイズ

        Parameters:
        -----------
        src : np.ndarray
            入力画像
        dsize : tuple
            出力サイズ (width, height)
        interpolation : int
            補間方法

        Returns:
        --------
        np.ndarray
            リサイズ後画像
        """
        if self._opencv_cuda_available:
            try:
                gpu_src = cv2.cuda_GpuMat()
                gpu_src.upload(src)
                gpu_dst = cv2.cuda.resize(gpu_src, dsize,
                                          interpolation=interpolation)
                return gpu_dst.download()
            except Exception:
                pass

        return cv2.resize(src, dsize, interpolation=interpolation)

    def gpu_remap(self, src: np.ndarray, map_x: np.ndarray,
                  map_y: np.ndarray,
                  interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        GPU高速化されたリマップ（360度投影変換用）

        Parameters:
        -----------
        src : np.ndarray
            入力画像
        map_x, map_y : np.ndarray
            リマップテーブル
        interpolation : int
            補間方法

        Returns:
        --------
        np.ndarray
            リマップ後画像
        """
        if self._opencv_cuda_available:
            try:
                gpu_src = cv2.cuda_GpuMat()
                gpu_src.upload(src)
                gpu_mx = cv2.cuda_GpuMat()
                gpu_mx.upload(map_x.astype(np.float32))
                gpu_my = cv2.cuda_GpuMat()
                gpu_my.upload(map_y.astype(np.float32))
                gpu_dst = cv2.cuda.remap(gpu_src, gpu_mx, gpu_my,
                                         interpolation=interpolation)
                return gpu_dst.download()
            except Exception:
                pass

        return cv2.remap(src, map_x.astype(np.float32),
                         map_y.astype(np.float32), interpolation)

    def compute_laplacian_var(self, gray: np.ndarray) -> float:
        """
        GPU高速化されたラプラシアン分散計算（鮮明度評価用）

        Parameters:
        -----------
        gray : np.ndarray
            グレースケール画像

        Returns:
        --------
        float
            ラプラシアン分散値
        """
        if self._opencv_cuda_available:
            try:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                laplacian_filter = cv2.cuda.createLaplacianFilter(
                    cv2.CV_8U, cv2.CV_32F, ksize=3
                )
                gpu_lap = laplacian_filter.apply(gpu_gray)
                lap = gpu_lap.download()
                return float(lap.var())
            except Exception:
                pass

        if self._torch_available and self._torch_device is not None:
            try:
                import torch
                import torch.nn.functional as F

                kernel = torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    dtype=torch.float32, device=self._torch_device
                ).unsqueeze(0).unsqueeze(0)

                img = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                img = img.to(self._torch_device)
                lap = F.conv2d(img, kernel, padding=1)
                return float(lap.var().cpu().item())
            except Exception:
                pass

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        return float(lap.var())

    def compute_optical_flow_sparse(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray,
        max_corners: int = 200
    ) -> float:
        """
        スパースオプティカルフロー計算（Dense比で10-50倍高速）

        Parameters:
        -----------
        prev_gray, curr_gray : np.ndarray
            前後のグレースケールフレーム
        max_corners : int
            追跡特徴点数

        Returns:
        --------
        float
            平均フロー大きさ
        """
        # 特徴点検出
        pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=max_corners,
            qualityLevel=0.01, minDistance=10
        )

        if pts is None or len(pts) < 5:
            return 0.0

        # Lucas-Kanadeスパースフロー
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None,
            winSize=(21, 21), maxLevel=3
        )

        if pts_next is None:
            return 0.0

        # 有効な追跡点のみ
        good_mask = status.ravel() == 1
        if good_mask.sum() < 3:
            return 0.0

        pts_prev = pts[good_mask]
        pts_next = pts_next[good_mask]

        # フロー大きさの平均
        flow = pts_next - pts_prev
        magnitudes = np.sqrt(flow[:, 0, 0] ** 2 + flow[:, 0, 1] ** 2)

        return float(np.mean(magnitudes))

    def batch_ssim(self, frames: list, reference: np.ndarray) -> list:
        """
        バッチSSIM計算（PyTorch利用時はGPUバッチ処理）

        Parameters:
        -----------
        frames : list of np.ndarray
            比較フレームリスト（グレースケール）
        reference : np.ndarray
            基準フレーム（グレースケール）

        Returns:
        --------
        list of float
            各フレームのSSIMスコア
        """
        if self._torch_available and self._torch_device is not None and len(frames) > 1:
            try:
                return self._batch_ssim_torch(frames, reference)
            except Exception:
                pass

        # CPU フォールバック: 個別計算
        from core.adaptive_selector import AdaptiveSelector
        selector = AdaptiveSelector()
        return [selector.compute_ssim(ref=reference, frame=f) for f in frames]

    def _batch_ssim_torch(self, frames: list, reference: np.ndarray) -> list:
        """PyTorch GPUバッチSSIM"""
        import torch
        import torch.nn.functional as F

        device = self._torch_device
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

        # Gaussianカーネル
        k = 11
        sigma = 1.5
        coords = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = (g.unsqueeze(0) * g.unsqueeze(1))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # 基準フレームをテンソル化
        ref_t = torch.from_numpy(reference.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        results = []
        # バッチ処理
        batch = torch.stack([
            torch.from_numpy(f.astype(np.float32)) for f in frames
        ]).unsqueeze(1).to(device)

        mu_ref = F.conv2d(ref_t, kernel, padding=k // 2)
        mu_batch = F.conv2d(batch, kernel, padding=k // 2)

        mu_ref_sq = mu_ref ** 2
        mu_batch_sq = mu_batch ** 2
        mu_cross = mu_ref * mu_batch

        sigma_ref_sq = F.conv2d(ref_t ** 2, kernel, padding=k // 2) - mu_ref_sq
        sigma_batch_sq = F.conv2d(batch ** 2, kernel, padding=k // 2) - mu_batch_sq
        sigma_cross = F.conv2d(ref_t * batch, kernel, padding=k // 2) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_ref_sq + mu_batch_sq + C1) * (sigma_ref_sq + sigma_batch_sq + C2))

        # 各フレームの平均SSIMを計算
        for i in range(len(frames)):
            results.append(float(ssim_map[i].mean().cpu().item()))

        return results


# === モジュールレベルのアクセサ ===

@lru_cache(maxsize=1)
def get_accelerator() -> Accelerator:
    """シングルトンアクセラレータインスタンスを取得"""
    return Accelerator()
