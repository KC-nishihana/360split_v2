"""
ビデオ読み込みモジュール - 360Split用
OpenCVを使用したビデオフレーム抽出とメタデータ管理。
ハードウェアアクセラレーション、LRUフレームキャッシュ、バックグラウンドプリフェッチャー実装。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import threading
import queue
import platform
import time
import subprocess

from core.accelerator import get_accelerator

from utils.logger import get_logger
logger = get_logger(__name__)


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


class FrameBuffer:
    """
    LRUキャッシュベースのフレームバッファ

    最大100フレームを保持するLRU (Least Recently Used) キャッシュ。
    頻繁にアクセスされるフレームをメモリに保持して、シークオーバーヘッドを削減。
    """

    def __init__(self, max_size: int = 100):
        """
        初期化

        Parameters:
        -----------
        max_size : int
            最大キャッシュフレーム数
        """
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max_size = max_size

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        キャッシュからフレームを取得（アクセス順序を更新）

        Parameters:
        -----------
        frame_idx : int
            フレーム番号

        Returns:
        --------
        np.ndarray or None
            キャッシュ内のフレーム、なければNone
        """
        if frame_idx not in self._cache:
            return None

        # LRU: 最近アクセスしたものを最後に移動
        self._cache.move_to_end(frame_idx)
        return self._cache[frame_idx]

    def put(self, frame_idx: int, frame: np.ndarray) -> None:
        """
        フレームをキャッシュに追加（満杯時は古いフレームを削除）

        Parameters:
        -----------
        frame_idx : int
            フレーム番号
        frame : np.ndarray
            フレームデータ
        """
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
        else:
            self._cache[frame_idx] = frame
            # サイズ超過時は最も古いフレームを削除
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """
        キャッシュを全クリア
        """
        self._cache.clear()

    def size(self) -> int:
        """
        現在のキャッシュサイズ

        Returns:
        --------
        int
            キャッシュ内のフレーム数
        """
        return len(self._cache)


class FramePrefetcher:
    """
    バックグラウンドフレームプリフェッチャー

    Producer-Consumerパターンで、バックグラウンドスレッドが
    次のフレームを先読みして queue.Queue に格納。
    シーク操作を最小化して読み込み性能を向上させる。
    """

    def __init__(self, video_path: str, frame_count: int, prefetch_size: int = 10):
        """
        初期化

        Parameters:
        -----------
        video_path : str
            ビデオファイルパス
        frame_count : int
            総フレーム数
        prefetch_size : int
            先読みキューサイズ
        """
        self._video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = frame_count
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self._current_idx = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """
        プリフェッチスレッドを開始
        """
        if self._thread is not None:
            return
        if self._frame_count <= 0:
            return
        self._cap = cv2.VideoCapture(self._video_path)
        if self._cap is None or not self._cap.isOpened():
            logger.warning(f"プリフェッチ用キャプチャを開けません: {self._video_path}")
            self._cap = None
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()
        logger.debug("フレームプリフェッチャー開始")

    def stop(self) -> None:
        """
        プリフェッチスレッドを停止
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        # キューをクリア
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        logger.debug("フレームプリフェッチャー停止")

    def set_position(self, frame_idx: int) -> None:
        """
        プリフェッチ位置を設定（シーク時に呼出）

        Parameters:
        -----------
        frame_idx : int
            次に読み込むフレーム番号
        """
        with self._lock:
            self._current_idx = frame_idx
            # キューをクリア（古いデータを破棄）
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass

    def get_frame(self, frame_idx: int, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        プリフェッチキューからフレームを取得

        Parameters:
        -----------
        frame_idx : int
            要求フレーム番号
        timeout : float
            タイムアウト時間（秒）

        Returns:
        --------
        np.ndarray or None
            キューから取得したフレーム（期待フレームと一致した場合）
        """
        if timeout <= 0:
            timeout = 0.001

        skipped = []
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                idx, frame = self._queue.get(timeout=remaining)
                if idx == frame_idx:
                    return frame
                skipped.append((idx, frame))
        except queue.Empty:
            pass
        finally:
            for item in skipped:
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    break
        return None

    def _prefetch_worker(self) -> None:
        """
        バックグラウンドプリフェッチワーカースレッド
        """
        while not self._stop_event.is_set():
            try:
                if self._cap is None:
                    break
                with self._lock:
                    idx = self._current_idx
                    self._current_idx += 1

                if idx >= self._frame_count:
                    self._stop_event.set()
                    break

                # プリフェッチ専用キャプチャで読み込み（メインキャプチャと競合させない）
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    try:
                        self._queue.put((idx, frame), timeout=1.0)
                    except queue.Full:
                        # キューが満杯の場合は待機
                        pass
                else:
                    # 読み込み失敗時は位置をリセット
                    with self._lock:
                        self._current_idx = idx + 1

            except Exception as e:
                logger.warning(f"プリフェッチワーカーエラー: {e}")
                break


class VideoLoader:
    """
    ビデオファイル読み込みとフレーム抽出

    OpenCVを使用してビデオファイルを開き、フレーム単位でのアクセス
    とバッチ抽出をサポート。LRUフレームキャッシュとバックグラウンド
    プリフェッチャーを備える。ハードウェアアクセラレーション対応。
    コンテキストマネージャ対応。
    """

    def __init__(self):
        """
        初期化
        """
        self._cap = None
        self._metadata = None
        self._current_frame_idx = -1
        self._video_path = None
        self._accel = get_accelerator()
        self._frame_buffer = FrameBuffer(max_size=100)
        self._prefetcher: Optional[FramePrefetcher] = None

    def load(self, path: str) -> VideoMetadata:
        """
        ビデオファイルを開く

        ハードウェアアクセラレーション対応：
        - Windows/CUDA環境では cv2.cudacodec.VideoReader を試行
        - macOS では cv2.CAP_FFMPEG を使用
        - フォールバック: 標準的なソフトウェアデコード

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

        # 既存のリソースをクリーンアップ
        if self._cap is not None:
            self.close()

        # ハードウェアアクセラレーション付きオープンを試行
        self._cap = self._try_hardware_accelerated_open(str(path))

        if self._cap is None or not self._cap.isOpened():
            # フォールバック: 標準デコード
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError(f"ビデオファイルを開けません: {path}")
            logger.info(f"ソフトウェアデコードで開きました: {path}")
        else:
            logger.info(f"ハードウェアアクセラレーションで開きました: {path}")

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
        self._frame_buffer.clear()

        # プリフェッチャーを初期化
        if self._prefetcher is not None:
            self._prefetcher.stop()
        self._prefetcher = FramePrefetcher(str(path), frame_count, prefetch_size=10)
        self._prefetcher.start()

        logger.info(
            f"ビデオ読み込み完了: {path.name} | "
            f"{width}x{height} @ {fps:.2f}fps | "
            f"{frame_count}フレーム ({duration:.2f}秒) | "
            f"アクセラレータ: {self._accel.device_name}"
        )

        return self._metadata

    def _try_hardware_accelerated_open(self, path: str) -> Optional[cv2.VideoCapture]:
        """
        ハードウェアアクセラレーション付きビデオオープンを試行

        Parameters:
        -----------
        path : str
            ビデオファイルパス

        Returns:
        --------
        cv2.VideoCapture or None
            成功時はキャプチャオブジェクト、失敗時はNone
        """
        system = platform.system()

        # Windows/CUDA: cudacodec を試行
        if system == "Windows" and self._accel.has_cuda:
            try:
                if hasattr(cv2, 'cudacodec'):
                    reader = cv2.cudacodec.createVideoReader(path)
                    if reader is not None:
                        # ダミー読み込みで初期化確認
                        success, frame = reader.nextFrame()
                        if success:
                            # 正常に動作。再度オープン
                            reader = cv2.cudacodec.createVideoReader(path)
                            return reader
            except Exception as e:
                logger.debug(f"CUDA VideoReader 失敗: {e}")

        # macOS: CAP_FFMPEG ヒント
        if system == "Darwin":
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    return cap
            except Exception as e:
                logger.debug(f"macOS CAP_FFMPEG 失敗: {e}")

        return None

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        指定フレーム番号のフレームを取得

        LRUキャッシュを優先的に確認し、キャッシュミス時のみ
        ファイルシークとプリフェッチキューを使用。

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

        # LRUキャッシュから確認
        cached_frame = self._frame_buffer.get(index)
        if cached_frame is not None:
            self._current_frame_idx = index
            return cached_frame

        # プリフェッチキューから確認（シーク不要）
        if self._prefetcher is not None:
            prefetched_frame = self._prefetcher.get_frame(index, timeout=0.1)
            if prefetched_frame is not None:
                self._frame_buffer.put(index, prefetched_frame.copy())
                self._current_frame_idx = index
                return prefetched_frame

        # ファイルシーク + 読み込み
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()

        if not ret:
            logger.warning(f"フレーム読み込み失敗: {index}")
            return None

        # キャッシュに保存
        self._frame_buffer.put(index, frame.copy())
        self._current_frame_idx = index

        # プリフェッチ位置を更新
        if self._prefetcher is not None:
            self._prefetcher.set_position(index + 1)

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

    def extract_frames_batch(self, indices: List[int]) -> Dict[int, np.ndarray]:
        """
        バッチフレーム抽出（最適化版）

        指定インデックスを昇順にソートして、シークを最小化して
        フレームを読み込む。クレームキャッシュとプリフェッチャーを活用。

        Parameters:
        -----------
        indices : list of int
            抽出対象のフレーム番号リスト

        Returns:
        --------
        dict
            {フレーム番号: フレーム} の辞書

        Example:
        --------
        >>> loader.load("video.mp4")
        >>> frames = loader.extract_frames_batch([0, 10, 20, 50, 100])
        >>> # フレームは昇順で読み込まれ、シーク回数が最小化される
        """
        if self._cap is None:
            raise RuntimeError("ビデオが読み込まれていません")

        if not indices:
            return {}

        result: Dict[int, np.ndarray] = {}

        # インデックスをソート（シークを最小化）
        sorted_indices = sorted(set(indices))

        logger.debug(f"バッチ抽出開始: {len(sorted_indices)}フレーム")

        for idx in sorted_indices:
            frame = self.get_frame(idx)
            if frame is not None:
                result[idx] = frame

        logger.info(f"{len(result)}フレームバッチ抽出完了")
        return result

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

    @property
    def cache_stats(self) -> Dict[str, int]:
        """
        キャッシュ統計情報を取得

        Returns:
        --------
        dict
            キャッシュ統計（'cached_frames'など）
        """
        return {
            'cached_frames': self._frame_buffer.size()
        }

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
        """
        ビデオファイルをクローズ

        フレームバッファとプリフェッチャーを解放。
        """
        if self._prefetcher is not None:
            self._prefetcher.stop()
            self._prefetcher = None

        if self._frame_buffer is not None:
            self._frame_buffer.clear()

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._metadata = None
            logger.info("ビデオファイルをクローズしました")

    def __del__(self):
        """デストラクタ"""
        self.close()


# ==============================================================================
# OSV (Omnidirectional Stereo Video) サポート
# ==============================================================================

class DualVideoLoader:
    """
    OSVファイル対応デュアルストリームローダー

    .osv ファイルから左右（Left/Right）の映像ストリームを分離し、
    フレーム単位で完全に同期して読み込む。既存の VideoLoader と
    互換性のあるインターフェースを提供。

    処理フロー:
    1. ffmpeg で .osv を left_eye.mp4 / right_eye.mp4 に分離
    2. 2つの VideoCapture で同期読み込み
    3. キーフレーム判定は Left 画像を代表として使用
    4. エクスポート時は L/R ペアで保存

    Attributes:
    -----------
    osv_path : str
        元の .osv ファイルパス
    left_path : str
        分離後の左目映像パス
    right_path : str
        分離後の右目映像パス
    """

    def __init__(self, temp_dir: str = "temp_streams"):
        """
        初期化

        Parameters:
        -----------
        temp_dir : str
            一時ストリーム保存ディレクトリ
        """
        self.osv_path: Optional[str] = None
        self.left_path: Optional[str] = None
        self.right_path: Optional[str] = None
        self.temp_dir = Path(temp_dir)

        self.cap_l: Optional[cv2.VideoCapture] = None
        self.cap_r: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._current_frame_idx = -1

        # フレームバッファ（L/R 両方をキャッシュ）
        self._frame_buffer_l = FrameBuffer(max_size=100)
        self._frame_buffer_r = FrameBuffer(max_size=100)

    def load(self, osv_path: str) -> VideoMetadata:
        """
        OSV ファイルを読み込み、左右ストリームに分離

        Parameters:
        -----------
        osv_path : str
            .osv ファイルパス

        Returns:
        --------
        VideoMetadata
            左目ストリームのメタデータ（代表として使用）

        Raises:
        -------
        FileNotFoundError
            OSV ファイルが見つからない場合
        RuntimeError
            ストリーム分離または読み込みに失敗した場合
        """
        osv_path_obj = Path(osv_path)
        if not osv_path_obj.exists():
            raise FileNotFoundError(f"OSV ファイルが見つかりません: {osv_path}")

        self.osv_path = str(osv_path_obj)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 一時ファイルパス
        stem = osv_path_obj.stem
        self.left_path = str(self.temp_dir / f"{stem}_left_eye.mp4")
        self.right_path = str(self.temp_dir / f"{stem}_right_eye.mp4")

        # ストリーム分離（キャッシュがない場合のみ）
        if not Path(self.left_path).exists() or not Path(self.right_path).exists():
            logger.info(f"OSV ストリームを分離中: {osv_path}")
            self._split_osv_streams()

        # 左右のストリームを開く
        self.cap_l = cv2.VideoCapture(self.left_path)
        self.cap_r = cv2.VideoCapture(self.right_path)

        if not self.cap_l.isOpened() or not self.cap_r.isOpened():
            raise RuntimeError(f"OSV ストリームを開けません: {osv_path}")

        # メタデータを取得（左目を代表として使用）
        fps = self.cap_l.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        # 右目のフレーム数と一致確認
        frame_count_r = int(self.cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count != frame_count_r:
            logger.warning(
                f"左右のフレーム数が不一致: L={frame_count}, R={frame_count_r}"
            )

        self._metadata = VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
            codec="osv_dual_stream"
        )

        logger.info(
            f"OSV 読み込み完了: {osv_path_obj.name} | "
            f"{width}x{height} @ {fps:.2f}fps | "
            f"{frame_count}フレーム (L/R ペア)"
        )

        return self._metadata

    def _split_osv_streams(self):
        """
        ffmpeg を使用して OSV ファイルを左右ストリームに分離

        Raises:
        -------
        RuntimeError
            ffmpeg 実行に失敗した場合
        """
        import subprocess

        try:
            # OSV ファイルは通常、ストリーム0=Left, ストリーム1=Right
            cmd = [
                "ffmpeg", "-y", "-i", self.osv_path,
                "-map", "0:0", "-c", "copy", self.left_path,
                "-map", "0:1", "-c", "copy", self.right_path
            ]

            logger.debug(f"ffmpeg コマンド: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"OSV ストリーム分離完了: L={self.left_path}, R={self.right_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg エラー: {e.stderr}")
            raise RuntimeError(f"OSV ストリーム分離に失敗: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg が見つかりません。ffmpeg をインストールしてください。\n"
                "macOS: brew install ffmpeg\n"
                "Ubuntu: sudo apt install ffmpeg\n"
                "Windows: https://ffmpeg.org/download.html"
            )

    def get_frame_pair(self, index: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        左右のフレームペアを同期して取得

        Parameters:
        -----------
        index : int
            フレーム番号（0ベース）

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] or (None, None)
            (左目フレーム, 右目フレーム) のタプル
        """
        if self.cap_l is None or self.cap_r is None:
            raise RuntimeError("OSV が読み込まれていません")

        if index < 0 or index >= self._metadata.frame_count:
            logger.warning(f"フレーム番号が範囲外です: {index}")
            return None, None

        # キャッシュから確認
        cached_l = self._frame_buffer_l.get(index)
        cached_r = self._frame_buffer_r.get(index)
        if cached_l is not None and cached_r is not None:
            self._current_frame_idx = index
            return cached_l, cached_r

        # 左右を同期して読み込み
        self.cap_l.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.cap_r.set(cv2.CAP_PROP_POS_FRAMES, index)

        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()

        if not ret_l or not ret_r:
            logger.warning(f"フレームペア読み込み失敗: {index}")
            return None, None

        # キャッシュに保存
        self._frame_buffer_l.put(index, frame_l.copy())
        self._frame_buffer_r.put(index, frame_r.copy())
        self._current_frame_idx = index

        return frame_l, frame_r

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        左目フレームのみを取得（VideoLoader 互換）

        Parameters:
        -----------
        index : int
            フレーム番号

        Returns:
        --------
        np.ndarray or None
            左目フレーム
        """
        frame_l, _ = self.get_frame_pair(index)
        return frame_l

    def get_metadata(self) -> Optional[VideoMetadata]:
        """メタデータを取得"""
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

    @property
    def is_stereo(self) -> bool:
        """ステレオ（OSV）であるかを判定"""
        return True

    def close(self):
        """リソースを解放"""
        if self.cap_l is not None:
            self.cap_l.release()
            self.cap_l = None

        if self.cap_r is not None:
            self.cap_r.release()
            self.cap_r = None

        self._frame_buffer_l.clear()
        self._frame_buffer_r.clear()
        self._metadata = None
        logger.info("OSV ファイルをクローズしました")

    def __enter__(self):
        """コンテキストマネージャエントリ"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャ終了"""
        self.close()
        return False

    def __del__(self):
        """デストラクタ"""
        self.close()
