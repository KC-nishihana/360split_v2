"""
ビデオ読み込みモジュール - 360Split用
OpenCVを使用したビデオフレーム抽出とメタデータ管理
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger('360split')


@dataclass
class VideoMetadata:
    """
    ビデオメタデータ格納クラス

    Attributes:
    -----------
    fps : float
        フレームレート（フレーム/秒）
    frame_count : int
        総フレーム数
    width : int
        フレーム幅（ピクセル）
    height : int
        フレーム高さ（ピクセル）
    duration : float
        ビデオ総長（秒）
    codec : str
        ビデオコーデック
    """
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    codec: Optional[str] = None


class VideoLoader:
    """
    ビデオファイル読み込みとフレーム抽出

    OpenCVを使用してビデオファイルを開き、フレーム単位での
    アクセスとバッチ抽出をサポート。コンテキストマネージャ対応。
    """

    def __init__(self):
        """初期化"""
        self._cap = None
        self._metadata = None
        self._current_frame_idx = -1
        self._video_path = None

    def load(self, path: str) -> VideoMetadata:
        """
        ビデオファイルを開く

        Parameters:
        -----------
        path : str
            ビデオファイルパス

        Returns:
        --------
        VideoMetadata
            ビデオメタデータ

        Raises:
        -------
        FileNotFoundError
            ファイルが見つからない場合
        RuntimeError
            ビデオファイルが開けない場合
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ビデオファイルが見つかりません: {path}")

        # 既存のキャプチャを閉じる
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"ビデオファイルを開けません: {path}")

        # メタデータを取得
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # コーデック情報を取得
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = self._fourcc_to_string(fourcc)

        # 総フレーム数が0の場合は警告
        if frame_count == 0:
            logger.warning(f"フレーム数を取得できません: {path}")

        duration = frame_count / fps if fps > 0 else 0

        self._metadata = VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
            codec=codec
        )

        self._video_path = path
        self._current_frame_idx = -1

        logger.info(
            f"ビデオ読み込み完了: {path.name} | "
            f"{width}x{height} @ {fps:.2f}fps | "
            f"{frame_count}フレーム ({duration:.2f}秒)"
        )

        return self._metadata

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        指定フレーム番号のフレームを取得

        Parameters:
        -----------
        index : int
            フレーム番号（0ベース）

        Returns:
        --------
        np.ndarray or None
            BGR形式のフレーム。読み込み失敗時はNone
        """
        if self._cap is None:
            raise RuntimeError("ビデオが読み込まれていません")

        if index < 0 or index >= self._metadata.frame_count:
            logger.warning(f"フレーム番号が範囲外です: {index}")
            return None

        # フレームにシーク
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()

        if not ret:
            logger.warning(f"フレーム読み込み失敗: {index}")
            return None

        self._current_frame_idx = index
        return frame

    def get_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """
        指定時刻のフレームを取得

        Parameters:
        -----------
        seconds : float
            時刻（秒）

        Returns:
        --------
        np.ndarray or None
            BGR形式のフレーム。読み込み失敗時はNone
        """
        if self._metadata is None:
            raise RuntimeError("ビデオが読み込まれていません")

        frame_index = int(seconds * self._metadata.fps)
        return self.get_frame(frame_index)

    def extract_frames(self, start: int = 0, end: Optional[int] = None,
                      step: int = 1) -> List[np.ndarray]:
        """
        フレーム範囲を抽出

        Parameters:
        -----------
        start : int
            開始フレーム番号
        end : int, optional
            終了フレーム番号。Noneの場合は最後のフレーム
        step : int
            ステップ数（1=すべてのフレーム、2=1フレーム置き）

        Returns:
        --------
        list of np.ndarray
            抽出されたフレームのリスト
        """
        if self._cap is None:
            raise RuntimeError("ビデオが読み込まれていません")

        if end is None:
            end = self._metadata.frame_count

        frames = []
        for idx in range(start, min(end, self._metadata.frame_count), step):
            frame = self.get_frame(idx)
            if frame is not None:
                frames.append(frame)

        logger.info(f"{len(frames)}フレーム抽出完了 (range: {start}-{end}, step: {step})")
        return frames

    def get_metadata(self) -> Optional[VideoMetadata]:
        """
        ビデオメタデータを取得

        Returns:
        --------
        VideoMetadata or None
            メタデータ。読み込まれていない場合はNone
        """
        return self._metadata

    @property
    def fps(self) -> float:
        """フレームレート"""
        return self._metadata.fps if self._metadata else 0

    @property
    def frame_count(self) -> int:
        """総フレーム数"""
        return self._metadata.frame_count if self._metadata else 0

    @property
    def width(self) -> int:
        """フレーム幅"""
        return self._metadata.width if self._metadata else 0

    @property
    def height(self) -> int:
        """フレーム高さ"""
        return self._metadata.height if self._metadata else 0

    @property
    def duration(self) -> float:
        """ビデオ総長（秒）"""
        return self._metadata.duration if self._metadata else 0

    def _fourcc_to_string(self, fourcc: int) -> str:
        """
        FourCC値を文字列に変換

        Parameters:
        -----------
        fourcc : int
            FourCC値

        Returns:
        --------
        str
            コーデック文字列
        """
        try:
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except (ValueError, OverflowError):
            return "unknown"

    def __enter__(self):
        """コンテキストマネージャエントリ"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャ終了時にリソース解放"""
        self.close()
        return False

    def close(self):
        """ビデオファイルをクローズ"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._metadata = None
            logger.info("ビデオファイルをクローズしました")

    def __del__(self):
        """デストラクタ"""
        self.close()
