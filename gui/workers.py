"""
非同期解析ワーカー - 360Split v2 GUI
GUIのフリーズを防ぐため、重い処理を別スレッドで実行する QThread クラス群。

AnalysisWorker:
  - Stage 1 (簡易解析): 全フレームの品質スコア(Sharpness/Exposure)を高速計算
  - Stage 2 (詳細解析): GRIC/SSIM をユーザー操作で開始
  - フレームごとのスコアをプログレッシブに返す

ExportWorker:
  - キーフレーム画像のバッチエクスポート
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from PySide6.QtCore import QThread, Signal, QObject

from utils.logger import get_logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# データ転送用の軽量クラス
# ---------------------------------------------------------------------------

@dataclass
class FrameScoreData:
    """1フレーム分のスコアデータ（GUI転送用）"""
    frame_index: int
    timestamp: float
    sharpness: float = 0.0
    exposure: float = 0.0
    motion_blur: float = 0.0
    gric: float = 0.0
    ssim: float = 1.0
    combined: float = 0.0
    is_keyframe: bool = False


# ---------------------------------------------------------------------------
# Stage 1: 簡易解析ワーカー
# ---------------------------------------------------------------------------

class Stage1Worker(QThread):
    """
    Stage 1: 高速品質フィルタリング（全フレーム）

    全フレームに対して品質スコア (Sharpness, Exposure, MotionBlur) を計算し、
    プログレッシブに結果を返す。GUIはこのデータで即座にグラフ描画を開始できる。

    Signals
    -------
    progress : Signal(int, int, str)
        (current, total, message)
    frame_scores : Signal(list)
        バッチ単位の FrameScoreData リスト
    finished_scores : Signal(list)
        全フレームスコア完了 (FrameScoreData リスト全体)
    error : Signal(str)
    """

    progress = Signal(int, int, str)
    frame_scores = Signal(list)          # バッチごとに送出
    finished_scores = Signal(list)       # 完了時に全データ送出
    error = Signal(str)

    def __init__(self, video_path: str, config: dict = None,
                 sample_interval: int = 1, parent: QObject = None):
        super().__init__(parent)
        self.video_path = video_path
        self.config = config or {}
        self.sample_interval = self.config.get('SAMPLE_INTERVAL', sample_interval)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            from core.quality_evaluator import QualityEvaluator

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"ビデオを開けません: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            evaluator = QualityEvaluator(eval_scale=0.5)

            all_scores: List[FrameScoreData] = []
            batch: List[FrameScoreData] = []
            batch_size = int(self.config.get('STAGE1_BATCH_SIZE', 32))
            beta = float(self.config.get('SOFTMAX_BETA', 5.0))

            frame_indices = list(range(0, total_frames, self.sample_interval))
            last_read = -1

            for count, frame_idx in enumerate(frame_indices):
                if not self._is_running:
                    break

                # シーク最適化
                if frame_idx != last_read + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                last_read = frame_idx

                if not ret or frame is None:
                    continue

                scores = evaluator.evaluate(frame, beta=beta)

                fsd = FrameScoreData(
                    frame_index=frame_idx,
                    timestamp=frame_idx / fps,
                    sharpness=scores.get('sharpness', 0.0),
                    exposure=scores.get('exposure', 0.0),
                    motion_blur=scores.get('motion_blur', 0.0),
                )
                all_scores.append(fsd)
                batch.append(fsd)

                # バッチ送出
                if len(batch) >= batch_size:
                    self.frame_scores.emit(list(batch))
                    batch.clear()
                    self.progress.emit(count + 1, len(frame_indices),
                                       f"Stage 1: {count+1}/{len(frame_indices)}")

            # 残りバッチ
            if batch:
                self.frame_scores.emit(list(batch))

            cap.release()

            if self._is_running:
                self.progress.emit(len(frame_indices), len(frame_indices),
                                   "Stage 1 完了")
                self.finished_scores.emit(all_scores)

        except Exception as e:
            logger.exception("Stage 1 ワーカーエラー")
            self.error.emit(f"Stage 1 エラー: {e}")


# ---------------------------------------------------------------------------
# Stage 2: 詳細解析ワーカー
# ---------------------------------------------------------------------------

class Stage2Worker(QThread):
    """
    Stage 2: 精密幾何学的評価（候補フレームのみ）

    Stage 1 で得たスコアに基づき、閾値通過フレームのみ GRIC/SSIM を計算。
    ユーザーが「詳細解析」ボタンを押した時に実行。

    Signals
    -------
    progress : Signal(int, int, str)
    keyframes_found : Signal(list)
        KeyframeInfo リスト
    frame_scores_updated : Signal(list)
        更新された FrameScoreData リスト
    finished : Signal()
    error : Signal(str)
    """

    progress = Signal(int, int, str)
    keyframes_found = Signal(list)
    frame_scores_updated = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, video_path: str, stage1_scores: List[FrameScoreData],
                 config: dict = None, parent: QObject = None):
        super().__init__(parent)
        self.video_path = video_path
        self.stage1_scores = stage1_scores
        self.config = config or {}
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            from core.video_loader import VideoLoader
            from core.keyframe_selector import KeyframeSelector

            loader = VideoLoader()
            loader.load(self.video_path)
            metadata = loader.get_metadata()

            selector = KeyframeSelector(config=self.config)

            def progress_cb(current, total, message=""):
                if not self._is_running:
                    return
                self.progress.emit(current, total,
                                   f"Stage 2: {current}/{total} {message}")

            keyframes = selector.select_keyframes(
                loader, progress_callback=progress_cb
            )

            loader.close()

            if not self._is_running:
                return

            # Stage1スコアにGRIC/SSIMを反映
            kf_indices = set()
            for kf in keyframes:
                kf_indices.add(kf.frame_index)

            updated_scores: List[FrameScoreData] = []
            for s1 in self.stage1_scores:
                fsd = FrameScoreData(
                    frame_index=s1.frame_index,
                    timestamp=s1.timestamp,
                    sharpness=s1.sharpness,
                    exposure=s1.exposure,
                    motion_blur=s1.motion_blur,
                )
                # キーフレーム情報があれば追加
                for kf in keyframes:
                    if kf.frame_index == s1.frame_index:
                        fsd.gric = kf.geometric_scores.get('gric', 0.0)
                        fsd.ssim = kf.adaptive_scores.get('ssim', 1.0)
                        fsd.combined = kf.combined_score
                        fsd.is_keyframe = True
                        break
                updated_scores.append(fsd)

            self.frame_scores_updated.emit(updated_scores)
            self.keyframes_found.emit(keyframes)
            self.progress.emit(1, 1, "Stage 2 完了")
            self.finished.emit()

        except Exception as e:
            logger.exception("Stage 2 ワーカーエラー")
            self.error.emit(f"Stage 2 エラー: {e}")


# ---------------------------------------------------------------------------
# フルパイプラインワーカー（Stage 1 + Stage 2 を連続実行）
# ---------------------------------------------------------------------------

class FullAnalysisWorker(QThread):
    """
    Stage 1 + Stage 2 を連続実行するワーカー。

    ユーザーが「分析実行」ボタンを1回押すだけで全パイプラインを走らせたい場合に使う。

    Signals
    -------
    progress : Signal(int, int, str)
    stage1_batch : Signal(list)
        Stage1 バッチ結果 (FrameScoreData リスト)
    stage1_finished : Signal(list)
        Stage1 完了 (FrameScoreData 全体)
    keyframes_found : Signal(list)
        KeyframeInfo リスト
    finished : Signal()
    error : Signal(str)
    """

    progress = Signal(int, int, str)
    stage1_batch = Signal(list)
    stage1_finished = Signal(list)
    keyframes_found = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, video_path: str, config: dict = None,
                 parent: QObject = None):
        super().__init__(parent)
        self.video_path = video_path
        self.config = config or {}
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            from core.video_loader import VideoLoader
            from core.keyframe_selector import KeyframeSelector
            from core.quality_evaluator import QualityEvaluator

            # ----- Stage 1: 高速品質スキャン -----
            self.progress.emit(0, 100, "Stage 1: 品質スキャン開始...")

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"ビデオを開けません: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            evaluator = QualityEvaluator(eval_scale=0.5)
            beta = float(self.config.get('SOFTMAX_BETA', 5.0))

            sample_interval = int(self.config.get('SAMPLE_INTERVAL', 1))
            batch_size = int(self.config.get('STAGE1_BATCH_SIZE', 32))
            frame_indices = list(range(0, total_frames, sample_interval))

            all_scores: List[FrameScoreData] = []
            batch: List[FrameScoreData] = []
            last_read = -1

            for count, frame_idx in enumerate(frame_indices):
                if not self._is_running:
                    cap.release()
                    return

                if frame_idx != last_read + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                last_read = frame_idx

                if not ret or frame is None:
                    continue

                scores = evaluator.evaluate(frame, beta=beta)
                fsd = FrameScoreData(
                    frame_index=frame_idx,
                    timestamp=frame_idx / fps,
                    sharpness=scores.get('sharpness', 0.0),
                    exposure=scores.get('exposure', 0.0),
                    motion_blur=scores.get('motion_blur', 0.0),
                )
                all_scores.append(fsd)
                batch.append(fsd)

                if len(batch) >= batch_size:
                    self.stage1_batch.emit(list(batch))
                    batch.clear()
                    pct = int((count + 1) / len(frame_indices) * 50)  # 0-50%
                    self.progress.emit(pct, 100,
                                       f"Stage 1: {count+1}/{len(frame_indices)}")

            if batch:
                self.stage1_batch.emit(list(batch))
            cap.release()

            if not self._is_running:
                return

            self.stage1_finished.emit(all_scores)
            self.progress.emit(50, 100, "Stage 1 完了。Stage 2 開始...")

            # ----- Stage 2: 精密評価 -----
            loader = VideoLoader()
            loader.load(self.video_path)

            selector = KeyframeSelector(config=self.config)

            def progress_cb(current, total, message=""):
                if not self._is_running:
                    return
                pct = 50 + int(current / max(total, 1) * 50)  # 50-100%
                self.progress.emit(pct, 100,
                                   f"Stage 2: {current}/{total}")

            keyframes = selector.select_keyframes(
                loader, progress_callback=progress_cb
            )
            loader.close()

            if not self._is_running:
                return

            self.keyframes_found.emit(keyframes)
            self.progress.emit(100, 100, "解析完了")
            self.finished.emit()

        except Exception as e:
            logger.exception("解析ワーカーエラー")
            self.error.emit(f"解析エラー: {e}")


# ---------------------------------------------------------------------------
# エクスポートワーカー
# ---------------------------------------------------------------------------

class ExportWorker(QThread):
    """
    キーフレーム画像のバッチエクスポート

    Signals
    -------
    progress : Signal(int, int, str)
    finished : Signal(int)
        エクスポートしたファイル数
    error : Signal(str)
    """

    progress = Signal(int, int, str)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, video_path: str, frame_indices: List[int],
                 output_dir: str, format: str = 'png',
                 jpeg_quality: int = 95, prefix: str = 'keyframe',
                 parent: QObject = None):
        super().__init__(parent)
        self.video_path = video_path
        self.frame_indices = sorted(frame_indices)
        self.output_dir = output_dir
        self.format = format.lower()
        self.jpeg_quality = jpeg_quality
        self.prefix = prefix
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"ビデオを開けません: {self.video_path}")
                return

            total = len(self.frame_indices)
            exported = 0

            for i, frame_idx in enumerate(self.frame_indices):
                if not self._is_running:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret and frame is not None:
                    ext = 'jpg' if self.format in ('jpg', 'jpeg') else self.format
                    filename = f"{self.prefix}_{frame_idx:06d}.{ext}"
                    filepath = output_path / filename

                    if ext == 'jpg':
                        cv2.imwrite(str(filepath), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    else:
                        cv2.imwrite(str(filepath), frame)

                    exported += 1

                self.progress.emit(i + 1, total,
                                   f"エクスポート: {i+1}/{total}")

            cap.release()
            self.finished.emit(exported)

        except Exception as e:
            logger.exception("エクスポートワーカーエラー")
            self.error.emit(f"エクスポートエラー: {e}")
