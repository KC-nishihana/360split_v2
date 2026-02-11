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
from typing import List
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal, QObject

from utils.logger import get_logger
from utils.image_io import write_image
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
        cap = None
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

            if self._is_running:
                self.progress.emit(len(frame_indices), len(frame_indices),
                                   "Stage 1 完了")
                self.finished_scores.emit(all_scores)

        except Exception as e:
            logger.exception("Stage 1 ワーカーエラー")
            self.error.emit(f"Stage 1 エラー: {e}")
        finally:
            if cap is not None:
                cap.release()


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

    360度処理（Equirectangular変換）とマスク処理に対応。

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
                 # 360度処理設定
                 enable_equirect: bool = False,
                 equirect_width: int = 4096,
                 equirect_height: int = 2048,
                 enable_polar_mask: bool = False,
                 mask_polar_ratio: float = 0.10,
                 # Cubemap 出力設定
                 enable_cubemap: bool = False,
                 cubemap_face_size: int = 1024,
                 # Perspective 出力設定
                 enable_perspective: bool = False,
                 perspective_fov: float = 90.0,
                 perspective_yaw_list: List[float] = None,
                 perspective_pitch_list: List[float] = None,
                 perspective_size: tuple = (1024, 1024),
                 # マスク処理設定
                 enable_nadir_mask: bool = False,
                 nadir_mask_radius: int = 100,
                 enable_equipment_detection: bool = False,
                 mask_dilation_size: int = 15,
                 parent: QObject = None):
        super().__init__(parent)
        self.video_path = video_path
        self.frame_indices = sorted(frame_indices)
        self.output_dir = output_dir
        self.format = format.lower()
        self.jpeg_quality = jpeg_quality
        self.prefix = prefix

        # 360度処理設定
        self.enable_equirect = enable_equirect
        self.equirect_width = equirect_width
        self.equirect_height = equirect_height
        self.enable_polar_mask = enable_polar_mask
        self.mask_polar_ratio = mask_polar_ratio

        # Cubemap 出力設定
        self.enable_cubemap = enable_cubemap
        self.cubemap_face_size = cubemap_face_size

        # Perspective 出力設定
        self.enable_perspective = enable_perspective
        self.perspective_fov = perspective_fov
        self.perspective_yaw_list = perspective_yaw_list or [0.0, 90.0, 180.0, -90.0]
        self.perspective_pitch_list = perspective_pitch_list or [0.0]
        self.perspective_size = perspective_size

        # マスク処理設定
        self.enable_nadir_mask = enable_nadir_mask
        self.nadir_mask_radius = nadir_mask_radius
        self.enable_equipment_detection = enable_equipment_detection
        self.mask_dilation_size = mask_dilation_size

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

            # 360度処理とマスク処理のプロセッサを初期化
            equirect_processor = None
            mask_processor = None

            needs_equirect = self.enable_equirect or self.enable_cubemap or self.enable_perspective
            if needs_equirect:
                try:
                    from processing.equirectangular import EquirectangularProcessor
                    equirect_processor = EquirectangularProcessor()
                    logger.info("360度処理を有効化しました")
                except ImportError as e:
                    logger.warning(f"EquirectangularProcessor のインポートに失敗: {e}")
                    self.enable_equirect = False
                    self.enable_cubemap = False
                    self.enable_perspective = False

            if self.enable_nadir_mask or self.enable_equipment_detection:
                try:
                    from processing.mask_processor import MaskProcessor
                    mask_processor = MaskProcessor()
                    logger.info("マスク処理を有効化しました")
                except ImportError as e:
                    logger.warning(f"MaskProcessor のインポートに失敗: {e}")
                    self.enable_nadir_mask = False
                    self.enable_equipment_detection = False

            total = len(self.frame_indices)
            exported = 0

            for i, frame_idx in enumerate(self.frame_indices):
                if not self._is_running:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret and frame is not None:
                    processed_frame = frame

                    # 360度処理を適用
                    if self.enable_equirect and equirect_processor:
                        try:
                            # Equirectangular変換（リサイズ + ポーラーマスク）
                            if processed_frame.shape[1] != self.equirect_width or \
                               processed_frame.shape[0] != self.equirect_height:
                                processed_frame = cv2.resize(
                                    processed_frame,
                                    (self.equirect_width, self.equirect_height),
                                    interpolation=cv2.INTER_LANCZOS4
                                )

                            # ポーラーマスク適用
                            if self.enable_polar_mask:
                                h, w = processed_frame.shape[:2]
                                mask_h = int(h * self.mask_polar_ratio)
                                # 天頂（上部）をマスク
                                processed_frame[:mask_h, :] = 0
                                # 天底（下部）をマスク
                                processed_frame[-mask_h:, :] = 0

                        except Exception as e:
                            logger.warning(f"360度処理エラー（フレーム {frame_idx}）: {e}")

                    # マスク処理を適用
                    if mask_processor:
                        try:
                            # ナディアマスク
                            if self.enable_nadir_mask:
                                h, w = processed_frame.shape[:2]
                                center_x, center_y = w // 2, h - 1
                                mask = np.ones((h, w), dtype=np.uint8) * 255
                                cv2.circle(mask, (center_x, center_y),
                                         self.nadir_mask_radius, 0, -1)
                                processed_frame = cv2.bitwise_and(
                                    processed_frame, processed_frame, mask=mask
                                )

                            # 装備検出（簡易的な実装：下部の一定領域をマスク）
                            if self.enable_equipment_detection:
                                h, w = processed_frame.shape[:2]
                                equipment_mask = np.ones((h, w), dtype=np.uint8) * 255
                                equipment_h = int(h * 0.2)  # 下部20%
                                equipment_mask[-equipment_h:, :] = 0
                                # 膨張処理
                                if self.mask_dilation_size > 0:
                                    kernel = np.ones(
                                        (self.mask_dilation_size, self.mask_dilation_size),
                                        np.uint8
                                    )
                                    equipment_mask = cv2.erode(
                                        equipment_mask, kernel, iterations=1
                                    )
                                processed_frame = cv2.bitwise_and(
                                    processed_frame, processed_frame, mask=equipment_mask
                                )

                        except Exception as e:
                            logger.warning(f"マスク処理エラー（フレーム {frame_idx}）: {e}")

                    # ファイルを保存
                    ext = 'jpg' if self.format in ('jpg', 'jpeg') else self.format
                    filename = f"{self.prefix}_{frame_idx:06d}.{ext}"
                    filepath = output_path / filename

                    if ext == 'jpg':
                        saved = write_image(
                            filepath,
                            processed_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                        )
                    else:
                        saved = write_image(filepath, processed_frame)

                    if not saved:
                        logger.warning(f"保存失敗（フレーム {frame_idx}）: {filepath}")
                        continue

                    exported += 1

                    # --- Cubemap 出力 ---
                    if self.enable_cubemap and equirect_processor:
                        try:
                            cubemap_dir = output_path / "cubemap" / f"frame_{frame_idx:06d}"
                            cubemap_dir.mkdir(parents=True, exist_ok=True)
                            faces = equirect_processor.to_cubemap(
                                processed_frame, self.cubemap_face_size
                            )
                            for face_name, face_img in faces.items():
                                face_path = cubemap_dir / f"{face_name}.{ext}"
                                if ext == 'jpg':
                                    if not write_image(
                                        face_path,
                                        face_img,
                                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                                    ):
                                        logger.warning(f"Cubemap保存失敗: {face_path}")
                                else:
                                    if not write_image(face_path, face_img):
                                        logger.warning(f"Cubemap保存失敗: {face_path}")
                            logger.debug(f"Cubemap 出力: {cubemap_dir}")
                        except Exception as e:
                            logger.warning(f"Cubemap 出力エラー（フレーム {frame_idx}）: {e}")

                    # --- Perspective 出力 ---
                    if self.enable_perspective and equirect_processor:
                        try:
                            persp_dir = output_path / "perspective" / f"frame_{frame_idx:06d}"
                            persp_dir.mkdir(parents=True, exist_ok=True)
                            for yaw in self.perspective_yaw_list:
                                for pitch in self.perspective_pitch_list:
                                    persp_img = equirect_processor.to_perspective(
                                        processed_frame,
                                        yaw=yaw, pitch=pitch,
                                        fov=self.perspective_fov,
                                        output_size=self.perspective_size
                                    )
                                    name = f"y{yaw:+.0f}_p{pitch:+.0f}.{ext}"
                                    persp_path = persp_dir / name
                                    if ext == 'jpg':
                                        if not write_image(
                                            persp_path,
                                            persp_img,
                                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                                        ):
                                            logger.warning(f"Perspective保存失敗: {persp_path}")
                                    else:
                                        if not write_image(persp_path, persp_img):
                                            logger.warning(f"Perspective保存失敗: {persp_path}")
                            logger.debug(f"Perspective 出力: {persp_dir}")
                        except Exception as e:
                            logger.warning(f"Perspective 出力エラー（フレーム {frame_idx}）: {e}")

                self.progress.emit(i + 1, total,
                                   f"エクスポート: {i+1}/{total}")

            cap.release()
            self.finished.emit(exported)

        except Exception as e:
            logger.exception("エクスポートワーカーエラー")
            self.error.emit(f"エクスポートエラー: {e}")
