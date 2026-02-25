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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

from PySide6.QtCore import QThread, Signal, QObject

from utils.logger import get_logger
from utils.image_io import write_image
from utils.rerun_logger import RerunKeyframeLogger
from core.visual_odometry.calibration import (
    calibration_to_dict,
    load_calibration_xml,
)
from processing.fisheye_splitter import Cross5FisheyeSplitter, Cross5SplitConfig
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
    flow_mag: float = 0.0
    translation_delta: float = 0.0
    rotation_delta: float = 0.0
    match_count: float = 0.0
    stage0_motion_risk: float = 0.0
    trajectory_consistency: float = 0.5
    combined_stage2: float = 0.0
    combined_stage3: float = 0.0
    stage3_selected: bool = False
    is_stationary: bool = False
    is_keyframe: bool = False
    t_xyz: Optional[List[float]] = None
    q_wxyz: Optional[List[float]] = None
    vo_status_reason: str = "not_evaluated"
    vo_pose_valid: bool = False
    vo_attempted: bool = False
    vo_valid: bool = False


def _run_stage1_scan(
    video_path: str,
    config: dict,
    sample_interval: int,
    is_running_cb,
    on_batch_cb=None,
    on_progress_cb=None,
) -> List[FrameScoreData]:
    from core.quality_evaluator import QualityEvaluator

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"ビデオを開けません: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stage1_eval_scale = float(config.get('stage1_eval_scale', config.get('STAGE1_EVAL_SCALE', 0.5)))
        stage1_eval_scale = max(0.1, min(1.0, stage1_eval_scale))
        evaluator = QualityEvaluator(
            eval_scale=stage1_eval_scale,
            motion_blur_method=str(config.get("MOTION_BLUR_METHOD", config.get("motion_blur_method", "legacy"))),
        )

        all_scores: List[FrameScoreData] = []
        batch: List[FrameScoreData] = []
        batch_size = int(config.get('STAGE1_BATCH_SIZE', 32))
        frame_indices = list(range(0, total_frames, sample_interval))
        grab_threshold = int(config.get('stage1_grab_threshold', config.get('STAGE1_GRAB_THRESHOLD', 30)))
        use_grab = (sample_interval > 1 and sample_interval <= grab_threshold)
        last_read = -1
        current_pos = -1

        if use_grab and frame_indices:
            first_idx = frame_indices[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
            current_pos = first_idx

        for count, frame_idx in enumerate(frame_indices):
            if not is_running_cb():
                break

            if use_grab:
                while current_pos < frame_idx:
                    ok = cap.grab()
                    if not ok:
                        break
                    current_pos += 1
                if current_pos != frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    current_pos = frame_idx
            else:
                if frame_idx != last_read + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if use_grab:
                current_pos += 1
            last_read = frame_idx

            if not ret or frame is None:
                continue

            scores = evaluator.evaluate_stage1_fast(frame)
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
                if on_batch_cb is not None:
                    on_batch_cb(list(batch))
                batch.clear()
            if on_progress_cb is not None:
                on_progress_cb(count + 1, len(frame_indices))

        if batch and on_batch_cb is not None:
            on_batch_cb(list(batch))
        return all_scores
    finally:
        cap.release()


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
            logger.info(
                "worker_start, worker=Stage1Worker,"
                f" video_path={self.video_path},"
                f" sample_interval={int(self.sample_interval)},"
                f" batch_size={int(self.config.get('STAGE1_BATCH_SIZE', 32))}"
            )
            def _on_progress(current: int, total: int) -> None:
                self.progress.emit(current, total, f"Stage 1: {current}/{max(total, 1)}")

            all_scores = _run_stage1_scan(
                video_path=self.video_path,
                config=self.config,
                sample_interval=int(self.sample_interval),
                is_running_cb=lambda: self._is_running,
                on_batch_cb=lambda items: self.frame_scores.emit(items),
                on_progress_cb=_on_progress,
            )
            if self._is_running:
                self.progress.emit(len(all_scores), max(len(all_scores), 1), "Stage 1 完了")
                self.finished_scores.emit(all_scores)
                logger.info(
                    "worker_finished, worker=Stage1Worker,"
                    f" scanned_frames={len(all_scores)}"
                )
        except (RuntimeError, cv2.error, ValueError, TypeError, OSError) as e:
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
    analysis_finished : Signal()
    error : Signal(str)
    """

    progress = Signal(int, int, str)
    keyframes_found = Signal(list)
    frame_scores_updated = Signal(list)
    trajectory_updated = Signal(dict)
    analysis_finished = Signal()
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
            from core.video_loader import VideoLoader, DualVideoLoader
            from core.keyframe_selector import KeyframeSelector
            stage_label = str(self.config.get("analysis_stage_label", "Stage 2"))
            logger.info(
                "worker_start, worker=Stage2Worker,"
                f" stage_label={stage_label},"
                f" analysis_mode={str(self.config.get('analysis_mode', 'full'))},"
                f" video_path={self.video_path},"
                f" stage1_scores={len(self.stage1_scores)},"
                f" enable_stage0_scan={bool(self.config.get('enable_stage0_scan', self.config.get('ENABLE_STAGE0_SCAN', True)))},"
                f" enable_stage3_refinement={bool(self.config.get('enable_stage3_refinement', self.config.get('ENABLE_STAGE3_REFINEMENT', True)))},"
                f" stage0_stride={int(self.config.get('stage0_stride', self.config.get('STAGE0_STRIDE', 5)))}"
            )

            if str(self.video_path).lower().endswith(".osv"):
                loader = DualVideoLoader()
                loader.load(self.video_path)
                logger.info("Stage2Worker: OSV入力をDualVideoLoaderで解析します（paired mode）")
            else:
                loader = VideoLoader()
                loader.load(self.video_path)
            _ensure_runtime_calibration(self.config, loader)

            selector = KeyframeSelector(config=self.config)
            rerun_logger = None
            metrics_map: Dict[int, Dict[str, float]] = {}
            pose_map: Dict[int, Dict[str, Optional[List[float]]]] = {}
            if bool(self.config.get("enable_rerun_logging", False)):
                rerun_logger = RerunKeyframeLogger(
                    app_id="keyframe_check_gui_stage2",
                    spawn=bool(self.config.get("rerun_spawn", True)),
                    save_path=self.config.get("rerun_save_path"),
                    timeline_name="frame",
                )

            def progress_cb(current, total, message=""):
                if not self._is_running:
                    return
                self.progress.emit(current, total,
                                   f"{stage_label}: {current}/{total} {message}")

            def frame_log_cb(payload: dict):
                if not self._is_running:
                    return
                frame_idx = int(payload.get("frame_index", 0))
                metrics_map[frame_idx] = dict(payload.get("metrics", {}))
                t_xyz = payload.get("t_xyz")
                q_wxyz = payload.get("q_wxyz")
                t_xyz_list = [float(v) for v in t_xyz] if isinstance(t_xyz, (list, tuple)) and len(t_xyz) == 3 else None
                q_wxyz_list = [float(v) for v in q_wxyz] if isinstance(q_wxyz, (list, tuple)) and len(q_wxyz) == 4 else None
                pose_map[frame_idx] = {
                    "t_xyz": t_xyz_list,
                    "q_wxyz": q_wxyz_list,
                }
                if t_xyz_list is not None:
                    self.trajectory_updated.emit(
                        {
                            "frame_index": frame_idx,
                            "t_xyz": t_xyz_list,
                            "q_wxyz": q_wxyz_list,
                            "is_keyframe": bool(payload.get("is_keyframe", False)),
                            "is_stationary": bool(float(metrics_map[frame_idx].get("is_stationary", 0.0)) > 0.5),
                        }
                    )
                if rerun_logger is None or not rerun_logger.enabled:
                    return
                rerun_logger.log_frame(
                    frame_idx=frame_idx,
                    img=payload.get("frame"),
                    t_xyz=payload.get("t_xyz"),
                    q_wxyz=payload.get("q_wxyz"),
                    is_keyframe=bool(payload.get("is_keyframe", False)),
                    metrics=payload.get("metrics", {}),
                    points_world=payload.get("points_world"),
                )

            keyframes = selector.select_keyframes(
                loader,
                progress_callback=progress_cb,
                frame_log_callback=frame_log_cb,
            )

            loader.close()

            if not self._is_running:
                return

            # Stage1スコアにGRIC/SSIMを反映（O(N_stage1 + N_keyframes)）
            keyframe_map = {kf.frame_index: kf for kf in keyframes}

            updated_scores: List[FrameScoreData] = []
            for s1 in self.stage1_scores:
                fsd = FrameScoreData(
                    frame_index=s1.frame_index,
                    timestamp=s1.timestamp,
                    sharpness=s1.sharpness,
                    exposure=s1.exposure,
                    motion_blur=s1.motion_blur,
                )
                m = metrics_map.get(s1.frame_index, {})
                p = pose_map.get(s1.frame_index, {})
                fsd.flow_mag = float(m.get('flow_mag', 0.0))
                fsd.translation_delta = float(m.get('translation_delta', 0.0))
                fsd.rotation_delta = float(m.get('rotation_delta', 0.0))
                fsd.match_count = float(m.get('match_count', 0.0))
                fsd.stage0_motion_risk = float(m.get('stage0_motion_risk', 0.0))
                fsd.trajectory_consistency = float(m.get('trajectory_consistency', 0.5))
                fsd.combined_stage2 = float(m.get('combined_stage2', fsd.combined))
                fsd.combined_stage3 = float(m.get('combined_stage3', fsd.combined))
                fsd.stage3_selected = bool(float(m.get('stage3_selected_flag', 0.0)) > 0.5)
                fsd.is_stationary = bool(float(m.get('is_stationary', 0.0)) > 0.5)
                fsd.t_xyz = p.get("t_xyz")
                fsd.q_wxyz = p.get("q_wxyz")
                fsd.vo_status_reason = str(m.get('vo_status_reason', 'not_evaluated'))
                fsd.vo_pose_valid = bool(float(m.get('vo_pose_valid', 0.0)) > 0.5)
                fsd.vo_attempted = bool(float(m.get('vo_attempted', 0.0)) > 0.5)
                fsd.vo_valid = bool(float(m.get('vo_valid', 0.0)) > 0.5)
                kf = keyframe_map.get(s1.frame_index)
                if kf is not None:
                    fsd.gric = kf.geometric_scores.get('gric', 0.0)
                    fsd.ssim = kf.adaptive_scores.get('ssim', 1.0)
                    fsd.combined = kf.combined_score
                    fsd.combined_stage3 = kf.combined_score
                    fsd.is_keyframe = True
                updated_scores.append(fsd)

            self.frame_scores_updated.emit(updated_scores)
            self.keyframes_found.emit(keyframes)
            self.progress.emit(1, 1, f"{stage_label} 完了")
            self.analysis_finished.emit()
            logger.info(
                "worker_finished, worker=Stage2Worker,"
                f" stage_label={stage_label},"
                f" keyframes={len(keyframes)},"
                f" updated_scores={len(updated_scores)}"
            )

        except Exception as e:
            logger.exception("Stage 2 ワーカーエラー")
            self.error.emit(f"Stage 2 エラー: {e}")


# ---------------------------------------------------------------------------
# 統合解析ワーカー（GUI解析の単一実行経路）
# ---------------------------------------------------------------------------

class UnifiedAnalysisWorker(QThread):
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
    analysis_finished : Signal()
    error : Signal(str)
    """

    progress = Signal(int, int, str)
    stage1_batch = Signal(list)
    stage1_finished = Signal(list)
    frame_scores_updated = Signal(list)
    trajectory_updated = Signal(dict)
    keyframes_found = Signal(list)
    analysis_finished = Signal()
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
            from core.video_loader import VideoLoader, DualVideoLoader
            from core.keyframe_selector import KeyframeSelector
            run_id = str(self.config.get("analysis_run_id", "n/a"))
            logger.info(
                "worker_start, worker=UnifiedAnalysisWorker,"
                f" analysis_run_id={run_id},"
                f" video_path={self.video_path},"
                f" enable_stage0_scan={bool(self.config.get('enable_stage0_scan', self.config.get('ENABLE_STAGE0_SCAN', True)))},"
                f" enable_stage3_refinement={bool(self.config.get('enable_stage3_refinement', self.config.get('ENABLE_STAGE3_REFINEMENT', True)))}"
            )

            # ----- Stage 1: 高速品質スキャン -----
            self.progress.emit(0, 100, "Stage 1: 品質スキャン開始...")

            sample_interval = int(self.config.get('SAMPLE_INTERVAL', 1))
            stage1_weight_pct = int(self.config.get('UNIFIED_STAGE1_PROGRESS_WEIGHT', 35))
            stage1_weight_pct = max(10, min(70, stage1_weight_pct))
            progress_state = {"last_total": 1}

            def _on_progress(current: int, total: int) -> None:
                progress_state["last_total"] = max(total, 1)
                pct = int(current / max(total, 1) * stage1_weight_pct)
                self.progress.emit(pct, 100, f"Stage 1: {current}/{max(total, 1)}")

            all_scores = _run_stage1_scan(
                video_path=self.video_path,
                config=self.config,
                sample_interval=sample_interval,
                is_running_cb=lambda: self._is_running,
                on_batch_cb=lambda items: self.stage1_batch.emit(items),
                on_progress_cb=_on_progress,
            )

            if not self._is_running:
                return

            self.stage1_finished.emit(all_scores)
            self.progress.emit(stage1_weight_pct, 100, "Stage 1 完了。Stage 2/3 開始...")

            # ----- Stage 2: 精密評価 -----
            if str(self.video_path).lower().endswith(".osv"):
                loader = DualVideoLoader()
                loader.load(self.video_path)
                logger.info("UnifiedAnalysisWorker: OSV入力をDualVideoLoaderで解析します（paired mode）")
            else:
                loader = VideoLoader()
                loader.load(self.video_path)
            _ensure_runtime_calibration(self.config, loader)

            selector = KeyframeSelector(config=self.config)
            rerun_logger = None
            metrics_map: Dict[int, Dict[str, float]] = {}
            pose_map: Dict[int, Dict[str, Optional[List[float]]]] = {}
            if bool(self.config.get("enable_rerun_logging", False)):
                rerun_logger = RerunKeyframeLogger(
                    app_id="keyframe_check_gui_full",
                    spawn=bool(self.config.get("rerun_spawn", True)),
                    save_path=self.config.get("rerun_save_path"),
                    timeline_name="frame",
                )

            def progress_cb(current, total, message=""):
                if not self._is_running:
                    return
                pct = stage1_weight_pct + int(current / max(total, 1) * (100 - stage1_weight_pct))
                pct = max(stage1_weight_pct, min(100, pct))
                self.progress.emit(pct, 100,
                                   f"解析: {current}/{total} {message}")

            def frame_log_cb(payload: dict):
                if not self._is_running:
                    return
                frame_idx = int(payload.get("frame_index", 0))
                metrics_map[frame_idx] = dict(payload.get("metrics", {}))
                t_xyz = payload.get("t_xyz")
                q_wxyz = payload.get("q_wxyz")
                t_xyz_list = [float(v) for v in t_xyz] if isinstance(t_xyz, (list, tuple)) and len(t_xyz) == 3 else None
                q_wxyz_list = [float(v) for v in q_wxyz] if isinstance(q_wxyz, (list, tuple)) and len(q_wxyz) == 4 else None
                pose_map[frame_idx] = {
                    "t_xyz": t_xyz_list,
                    "q_wxyz": q_wxyz_list,
                }
                if t_xyz_list is not None:
                    self.trajectory_updated.emit(
                        {
                            "frame_index": frame_idx,
                            "t_xyz": t_xyz_list,
                            "q_wxyz": q_wxyz_list,
                            "is_keyframe": bool(payload.get("is_keyframe", False)),
                            "is_stationary": bool(float(metrics_map[frame_idx].get("is_stationary", 0.0)) > 0.5),
                        }
                    )
                if rerun_logger is None or not rerun_logger.enabled:
                    return
                rerun_logger.log_frame(
                    frame_idx=frame_idx,
                    img=payload.get("frame"),
                    t_xyz=payload.get("t_xyz"),
                    q_wxyz=payload.get("q_wxyz"),
                    is_keyframe=bool(payload.get("is_keyframe", False)),
                    metrics=payload.get("metrics", {}),
                    points_world=payload.get("points_world"),
                )

            keyframes = selector.select_keyframes(
                loader,
                progress_callback=progress_cb,
                frame_log_callback=frame_log_cb,
            )
            loader.close()

            if not self._is_running:
                return

            keyframe_map = {kf.frame_index: kf for kf in keyframes}
            updated_scores: List[FrameScoreData] = []
            for s1 in all_scores:
                fsd = FrameScoreData(
                    frame_index=s1.frame_index,
                    timestamp=s1.timestamp,
                    sharpness=s1.sharpness,
                    exposure=s1.exposure,
                    motion_blur=s1.motion_blur,
                )
                m = metrics_map.get(s1.frame_index, {})
                p = pose_map.get(s1.frame_index, {})
                fsd.flow_mag = float(m.get('flow_mag', 0.0))
                fsd.translation_delta = float(m.get('translation_delta', 0.0))
                fsd.rotation_delta = float(m.get('rotation_delta', 0.0))
                fsd.match_count = float(m.get('match_count', 0.0))
                fsd.stage0_motion_risk = float(m.get('stage0_motion_risk', 0.0))
                fsd.trajectory_consistency = float(m.get('trajectory_consistency', 0.5))
                fsd.combined_stage2 = float(m.get('combined_stage2', fsd.combined))
                fsd.combined_stage3 = float(m.get('combined_stage3', fsd.combined))
                fsd.stage3_selected = bool(float(m.get('stage3_selected_flag', 0.0)) > 0.5)
                fsd.is_stationary = bool(float(m.get('is_stationary', 0.0)) > 0.5)
                fsd.t_xyz = p.get("t_xyz")
                fsd.q_wxyz = p.get("q_wxyz")
                fsd.vo_status_reason = str(m.get('vo_status_reason', 'not_evaluated'))
                fsd.vo_pose_valid = bool(float(m.get('vo_pose_valid', 0.0)) > 0.5)
                fsd.vo_attempted = bool(float(m.get('vo_attempted', 0.0)) > 0.5)
                fsd.vo_valid = bool(float(m.get('vo_valid', 0.0)) > 0.5)
                kf = keyframe_map.get(s1.frame_index)
                if kf is not None:
                    fsd.gric = kf.geometric_scores.get('gric', 0.0)
                    fsd.ssim = kf.adaptive_scores.get('ssim', 1.0)
                    fsd.combined = kf.combined_score
                    fsd.combined_stage3 = kf.combined_score
                    fsd.is_keyframe = True
                updated_scores.append(fsd)

            self.frame_scores_updated.emit(updated_scores)
            self.keyframes_found.emit(keyframes)
            self.progress.emit(100, 100, "解析完了")
            self.analysis_finished.emit()
            logger.info(
                "worker_finished, worker=UnifiedAnalysisWorker,"
                f" analysis_run_id={run_id},"
                f" keyframes={len(keyframes)},"
                f" updated_scores={len(updated_scores)}"
            )

        except Exception as e:
            logger.exception("解析ワーカーエラー")
            self.error.emit(f"解析エラー: {e}")


class FullAnalysisWorker(UnifiedAnalysisWorker):
    """後方互換ラッパー。GUIは UnifiedAnalysisWorker を使用する。"""
    pass


def _normalize_path_value(raw_path: object) -> str:
    s = str(raw_path or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    if not s:
        return ""
    if os.name != "nt" and "\\" in s:
        s = s.replace("\\", "/")
    s = os.path.expandvars(os.path.expanduser(s))
    return s


def _resolve_calibration_path(raw_path: object, search_roots: List[Path]) -> str:
    norm = _normalize_path_value(raw_path)
    if not norm:
        return ""
    p = Path(norm)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([root / p for root in search_roots])
        candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # Fallback: keep expanded input for logger/load attempt
    return str(p)


def _build_runtime_calibration(config: dict, loader) -> Dict[str, object]:
    """GUI解析用に calibration_runtime を構築する（CLIと同等の最小構成）。"""
    meta = loader.get_metadata()
    if meta is None:
        return {"model_hint": "auto", "mono": None, "front": None, "rear": None}

    model_hint = str(config.get("calib_model", config.get("CALIB_MODEL", "auto")) or "auto").strip().lower()
    fallback_size = (int(meta.width), int(meta.height))
    video_path = str(getattr(loader, "_video_path", "") or "")
    search_roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    if video_path:
        search_roots.append(Path(video_path).expanduser().resolve().parent)

    mono_xml_raw = config.get("calib_xml", config.get("CALIB_XML", ""))
    front_xml_raw = config.get("front_calib_xml", config.get("FRONT_CALIB_XML", ""))
    rear_xml_raw = config.get("rear_calib_xml", config.get("REAR_CALIB_XML", ""))
    mono_xml = _resolve_calibration_path(mono_xml_raw, search_roots) or None
    front_xml = _resolve_calibration_path(front_xml_raw, search_roots) or mono_xml
    rear_xml = _resolve_calibration_path(rear_xml_raw, search_roots) or mono_xml

    mono = None
    front = None
    rear = None

    is_paired = bool(getattr(loader, "is_paired", False))
    if is_paired:
        if front_xml:
            front = load_calibration_xml(front_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
        if rear_xml:
            rear = load_calibration_xml(rear_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
        if front is None and mono_xml:
            front = load_calibration_xml(mono_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
        if rear is None and mono_xml:
            rear = load_calibration_xml(mono_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)
    else:
        if mono_xml:
            mono = load_calibration_xml(mono_xml, model_hint=model_hint, fallback_image_size=fallback_size, logger=logger)

    return {
        "model_hint": model_hint,
        "mono": calibration_to_dict(mono),
        "front": calibration_to_dict(front),
        "rear": calibration_to_dict(rear),
    }


def _log_runtime_calibration_status(runtime: Dict[str, object]) -> None:
    mono_ok = runtime.get("mono") is not None
    front_ok = runtime.get("front") is not None
    rear_ok = runtime.get("rear") is not None
    logger.info(
        "gui calibration_runtime: "
        f"mono={'OK' if mono_ok else 'NONE'}, "
        f"front={'OK' if front_ok else 'NONE'}, "
        f"rear={'OK' if rear_ok else 'NONE'}"
    )


def _log_runtime_calibration_inputs(config: dict) -> None:
    mono_xml = _normalize_path_value(config.get("calib_xml", config.get("CALIB_XML", "")))
    front_xml = _normalize_path_value(config.get("front_calib_xml", config.get("FRONT_CALIB_XML", "")))
    rear_xml = _normalize_path_value(config.get("rear_calib_xml", config.get("REAR_CALIB_XML", "")))
    model = str(config.get("calib_model", config.get("CALIB_MODEL", "auto")) or "auto")
    logger.info(
        "gui calibration inputs: "
        f"calib_model={model}, mono='{mono_xml or '-'}', front='{front_xml or '-'}', rear='{rear_xml or '-'}'"
    )


def _ensure_runtime_calibration(config: dict, loader) -> None:
    _log_runtime_calibration_inputs(config)
    projection_mode = str(config.get("projection_mode", config.get("PROJECTION_MODE", "")) or "").strip().lower()
    if projection_mode in {"cubemap", "equirectangular"}:
        # GUI設定の projection_mode は主にエクスポート指定。
        # 解析時VOは元フレーム上で計算するため、VO無効化条件としては扱わない。
        config["_analysis_projection_mode_original"] = projection_mode
        config["projection_mode"] = "auto"
        config["PROJECTION_MODE"] = "auto"
        logger.info(
            f"gui analysis: projection_mode '{projection_mode}' is treated as export setting; "
            "VO runtime check uses 'auto'"
        )

    runtime = _build_runtime_calibration(config, loader)
    config["calibration_runtime"] = runtime
    _log_runtime_calibration_status(runtime)


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
                 enable_stereo_stitch: bool = False,
                 stitching_mode: str = "Fast",
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
                 enable_fisheye_border_mask: bool = True,
                 fisheye_mask_radius_ratio: float = 0.94,
                 fisheye_mask_center_offset_x: int = 0,
                 fisheye_mask_center_offset_y: int = 0,
                 # 対象検出マスク設定
                 enable_target_mask_generation: bool = False,
                 target_classes: Optional[List[str]] = None,
                 yolo_model_path: str = "yolo26n-seg.pt",
                 sam_model_path: str = "sam3_t.pt",
                 confidence_threshold: float = 0.25,
                 detection_device: str = "auto",
                 mask_output_dirname: str = "masks",
                 mask_add_suffix: bool = True,
                 mask_suffix: str = "_mask",
                 mask_output_format: str = "same",
                 dynamic_mask_use_motion_diff: bool = True,
                 dynamic_mask_motion_frames: int = 3,
                 dynamic_mask_motion_threshold: int = 30,
                 dynamic_mask_dilation_size: int = 5,
                 dynamic_mask_use_yolo_sam: bool = True,
                 dynamic_mask_target_classes: Optional[List[str]] = None,
                 dynamic_mask_inpaint_enabled: bool = False,
                 dynamic_mask_inpaint_module: str = "",
                 precomputed_analysis_masks: Optional[Dict[int, np.ndarray]] = None,
                 use_precomputed_analysis_masks: bool = False,
                 export_runtime_config: Optional[Dict[str, object]] = None,
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
        self.enable_stereo_stitch = enable_stereo_stitch
        self.stitching_mode = stitching_mode
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
        self.enable_fisheye_border_mask = bool(enable_fisheye_border_mask)
        self.fisheye_mask_radius_ratio = float(fisheye_mask_radius_ratio)
        self.fisheye_mask_center_offset_x = int(fisheye_mask_center_offset_x)
        self.fisheye_mask_center_offset_y = int(fisheye_mask_center_offset_y)
        self.enable_target_mask_generation = enable_target_mask_generation
        self.target_classes = target_classes or []
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.confidence_threshold = confidence_threshold
        self.detection_device = detection_device
        self.mask_output_dirname = mask_output_dirname or "masks"
        self.mask_add_suffix = mask_add_suffix
        self.mask_suffix = mask_suffix or "_mask"
        self.mask_output_format = (mask_output_format or "same").lower()
        self.dynamic_mask_use_motion_diff = bool(dynamic_mask_use_motion_diff)
        self.dynamic_mask_motion_frames = max(2, int(dynamic_mask_motion_frames))
        self.dynamic_mask_motion_threshold = int(dynamic_mask_motion_threshold)
        self.dynamic_mask_dilation_size = max(0, int(dynamic_mask_dilation_size))
        self.dynamic_mask_use_yolo_sam = bool(dynamic_mask_use_yolo_sam)
        self.dynamic_mask_target_classes = list(dynamic_mask_target_classes or [])
        self.dynamic_mask_inpaint_enabled = bool(dynamic_mask_inpaint_enabled)
        self.dynamic_mask_inpaint_module = str(dynamic_mask_inpaint_module or "").strip()
        self.precomputed_analysis_masks = dict(precomputed_analysis_masks or {})
        self.use_precomputed_analysis_masks = bool(use_precomputed_analysis_masks)
        self.export_runtime_config = dict(export_runtime_config or {})
        self._sparse_frame_export = False
        self._effective_motion_diff = self.dynamic_mask_use_motion_diff

        # ステレオ（OSV）設定
        self.is_stereo = False
        self.stereo_left_path = None
        self.stereo_right_path = None

        self._is_running = True

    def set_stereo_paths(self, left_path: str, right_path: str):
        """ステレオペア出力用のパスを設定"""
        self.is_stereo = True
        self.stereo_left_path = left_path
        self.stereo_right_path = right_path

    def stop(self):
        self._is_running = False

    def _build_export_calibration_runtime(self, frame_size: tuple[int, int]) -> Dict[str, object]:
        runtime = self.export_runtime_config.get("calibration_runtime")
        if isinstance(runtime, dict):
            return runtime

        cfg = self.export_runtime_config if isinstance(self.export_runtime_config, dict) else {}
        model_hint = str(cfg.get("calib_model", cfg.get("CALIB_MODEL", "auto")) or "auto").strip().lower()
        mono_xml_raw = cfg.get("calib_xml", cfg.get("CALIB_XML", ""))
        front_xml_raw = cfg.get("front_calib_xml", cfg.get("FRONT_CALIB_XML", ""))
        rear_xml_raw = cfg.get("rear_calib_xml", cfg.get("REAR_CALIB_XML", ""))
        search_roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
        if self.video_path:
            search_roots.append(Path(self.video_path).expanduser().resolve().parent)
        mono_xml = _resolve_calibration_path(mono_xml_raw, search_roots) or None
        front_xml = _resolve_calibration_path(front_xml_raw, search_roots) or mono_xml
        rear_xml = _resolve_calibration_path(rear_xml_raw, search_roots) or mono_xml

        mono = load_calibration_xml(mono_xml, model_hint=model_hint, fallback_image_size=frame_size, logger=logger) if mono_xml else None
        front = load_calibration_xml(front_xml, model_hint=model_hint, fallback_image_size=frame_size, logger=logger) if front_xml else None
        rear = load_calibration_xml(rear_xml, model_hint=model_hint, fallback_image_size=frame_size, logger=logger) if rear_xml else None
        if front is None and mono is not None:
            front = mono
        if rear is None and mono is not None:
            rear = mono
        return {
            "model_hint": model_hint,
            "mono": calibration_to_dict(mono),
            "front": calibration_to_dict(front),
            "rear": calibration_to_dict(rear),
        }

    def _create_cross5_splitters(self, frame_size: tuple[int, int]) -> Dict[str, Cross5FisheyeSplitter]:
        if not self.is_stereo:
            return {}
        split_enabled = bool(self.export_runtime_config.get("enable_split_views", True))
        if not split_enabled:
            logger.info("Export cross5 splitter: 分割出力は無効")
            return {}
        split_cfg = Cross5SplitConfig(
            size=int(self.export_runtime_config.get("split_view_size", 1600)),
            hfov=float(self.export_runtime_config.get("split_view_hfov", 80.0)),
            vfov=float(self.export_runtime_config.get("split_view_vfov", 80.0)),
            cross_yaw_deg=float(self.export_runtime_config.get("split_cross_yaw_deg", 50.5)),
            cross_pitch_deg=float(self.export_runtime_config.get("split_cross_pitch_deg", 50.5)),
            cross_inward_deg=float(self.export_runtime_config.get("split_cross_inward_deg", 10.0)),
            inward_up_deg=float(self.export_runtime_config.get("split_inward_up_deg", 25.0)),
            inward_down_deg=float(self.export_runtime_config.get("split_inward_down_deg", 25.0)),
            inward_left_deg=float(self.export_runtime_config.get("split_inward_left_deg", 25.0)),
            inward_right_deg=float(self.export_runtime_config.get("split_inward_right_deg", 25.0)),
        )
        runtime = self._build_export_calibration_runtime(frame_size)
        front_dict = runtime.get("front") or runtime.get("mono")
        rear_dict = runtime.get("rear") or runtime.get("mono")
        splitters: Dict[str, Cross5FisheyeSplitter] = {}
        try:
            if front_dict:
                splitters["_L"] = Cross5FisheyeSplitter(front_dict, cfg=split_cfg)
            if rear_dict:
                splitters["_R"] = Cross5FisheyeSplitter(rear_dict, cfg=split_cfg)
        except Exception as e:
            logger.warning(f"Export cross5 splitter初期化失敗: {e}")
            return {}
        if not splitters:
            logger.warning("Export cross5 splitter: キャリブレーション未設定のため分割をスキップ")
        return splitters

    def _save_target_mask(
        self,
        images_root: Path,
        masks_root: Path,
        image_path: Path,
        frame_idx: int,
        frame: np.ndarray,
        mask_generator,
        motion_frames: Optional[List[np.ndarray]] = None,
        force_mask_reanalysis: bool = False,
        flatten_stereo_lr: bool = False,
    ) -> bool:
        from processing.target_mask_generator import TargetMaskGenerator

        mask_path = TargetMaskGenerator.build_mask_path(
            image_path=image_path,
            images_root=images_root,
            masks_root=masks_root,
            add_suffix=self.mask_add_suffix,
            suffix=self.mask_suffix,
            mask_ext=self.mask_output_format,
            flatten_stereo_lr=flatten_stereo_lr,
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        binary_mask = None
        if self.use_precomputed_analysis_masks and not force_mask_reanalysis:
            cached_mask = self.precomputed_analysis_masks.get(int(frame_idx))
            if cached_mask is not None:
                if cached_mask.shape[:2] != frame.shape[:2]:
                    cached_mask = cv2.resize(
                        cached_mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                binary_mask = cached_mask.astype(np.uint8)

        if binary_mask is None:
            if mask_generator is None:
                return False
            safe_motion_frames = motion_frames
            if self.use_precomputed_analysis_masks and not force_mask_reanalysis:
                # 解析済みマスク再利用が欠損したフレームは履歴を使わず単フレームでフォールバック。
                safe_motion_frames = [frame]
            if self._sparse_frame_export:
                safe_motion_frames = [frame]
            classes_for_detection = (
                self.dynamic_mask_target_classes or self.target_classes
                if self.dynamic_mask_use_yolo_sam
                else []
            )
            binary_mask = mask_generator.generate_mask(
                frame,
                classes_for_detection,
                motion_frames=safe_motion_frames,
            )
        if self.is_stereo and self.enable_fisheye_border_mask:
            h, w = frame.shape[:2]
            valid_mask = np.zeros((h, w), dtype=np.uint8)
            radius = int(min(w, h) * 0.5 * float(np.clip(self.fisheye_mask_radius_ratio, 0.0, 1.0)))
            cx = int(np.clip((w // 2) + self.fisheye_mask_center_offset_x, 0, max(w - 1, 0)))
            cy = int(np.clip((h // 2) + self.fisheye_mask_center_offset_y, 0, max(h - 1, 0)))
            if radius > 0:
                cv2.circle(valid_mask, (cx, cy), radius, 255, -1)
            binary_mask = cv2.bitwise_and(binary_mask.astype(np.uint8), valid_mask)
        if self.dynamic_mask_inpaint_enabled:
            if mask_generator is not None:
                _ = mask_generator.run_inpaint_hook(frame, binary_mask)
        ext = mask_path.suffix.lower().lstrip(".")
        if ext in ("jpg", "jpeg"):
            return write_image(
                mask_path,
                binary_mask,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
        return write_image(mask_path, binary_mask)

    def _build_flat_split_mask_path(self, masks_root: Path, split_image_path: Path) -> Path:
        stem = f"{split_image_path.stem}{self.mask_suffix}" if self.mask_add_suffix else split_image_path.stem
        if self.mask_output_format == "same":
            ext = split_image_path.suffix
        else:
            ext = f".{self.mask_output_format.lstrip('.')}"
        return masks_root / f"{stem}{ext}"

    def run(self):
        write_executor = None
        try:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            stereo_images_root = output_path / "images"
            masks_root = output_path / self.mask_output_dirname
            max_write_workers = int(self.export_runtime_config.get("export_write_workers", min(4, os.cpu_count() or 2)))
            max_write_workers = max(1, min(8, max_write_workers))
            if max_write_workers > 1:
                write_executor = ThreadPoolExecutor(max_workers=max_write_workers)

            # ステレオ判定
            if self.is_stereo and self.stereo_left_path and self.stereo_right_path:
                # ステレオペア出力モード
                cap_l = cv2.VideoCapture(self.stereo_left_path)
                cap_r = cv2.VideoCapture(self.stereo_right_path)
                if not cap_l.isOpened() or not cap_r.isOpened():
                    self.error.emit(f"ステレオストリームを開けません")
                    return
                logger.info("ステレオペア出力モードで実行")
                cap = None  # 単眼キャプチャは使用しない
            else:
                # 通常の単眼モード
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.error.emit(f"ビデオを開けません: {self.video_path}")
                    return
                cap_l = cap_r = None

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

            stitch_processor = None
            if self.is_stereo and self.enable_stereo_stitch:
                try:
                    from processing.stitching import StitchingProcessor
                    stitch_processor = StitchingProcessor()
                    logger.info(f"ステレオステッチングを有効化しました: mode={self.stitching_mode}")
                except ImportError as e:
                    logger.warning(f"StitchingProcessor のインポートに失敗: {e}")
                    self.enable_stereo_stitch = False

            sorted_idx = sorted(int(i) for i in self.frame_indices)
            self._sparse_frame_export = any(
                (b - a) > 1 for a, b in zip(sorted_idx, sorted_idx[1:])
            )
            self._effective_motion_diff = bool(self.dynamic_mask_use_motion_diff)
            if self._sparse_frame_export and self._effective_motion_diff:
                logger.info("疎フレーム出力のため MotionDiff を自動無効化します")
                self._effective_motion_diff = False

            target_mask_generator = None
            target_mask_cls = None
            if self.enable_target_mask_generation:
                try:
                    needs_reanalysis = bool(self.enable_cubemap or self.enable_perspective or self.is_stereo)
                    can_reuse_only = (
                        self.use_precomputed_analysis_masks
                        and not needs_reanalysis
                        and all(int(idx) in self.precomputed_analysis_masks for idx in self.frame_indices)
                    )
                    if can_reuse_only:
                        logger.info("対象マスク出力は解析時マスクを再利用（再解析なし）")
                    else:
                        if self.use_precomputed_analysis_masks and not needs_reanalysis:
                            missing = [
                                int(idx) for idx in self.frame_indices
                                if int(idx) not in self.precomputed_analysis_masks
                            ]
                            if missing:
                                logger.warning(
                                    "解析時マスクが不足しているため一部フレームは再解析します: "
                                    f"missing={len(missing)}"
                                )
                        from processing.target_mask_generator import TargetMaskGenerator
                        target_mask_cls = TargetMaskGenerator
                        inpaint_hook = None
                        if self.dynamic_mask_inpaint_enabled and self.dynamic_mask_inpaint_module:
                            try:
                                mod = __import__(self.dynamic_mask_inpaint_module, fromlist=['inpaint_frame'])
                                hook = getattr(mod, 'inpaint_frame', None)
                                if callable(hook):
                                    inpaint_hook = hook
                            except Exception as e:
                                logger.warning(f"インペイントモジュール読み込み失敗: {e}")

                        target_mask_generator = TargetMaskGenerator(
                            yolo_model_path=self.yolo_model_path,
                            sam_model_path=self.sam_model_path,
                            confidence_threshold=self.confidence_threshold,
                            device=self.detection_device,
                            enable_motion_detection=self._effective_motion_diff,
                            motion_history_frames=self.dynamic_mask_motion_frames,
                            motion_threshold=self.dynamic_mask_motion_threshold,
                            motion_mask_dilation_size=self.dynamic_mask_dilation_size,
                            enable_mask_inpaint=self.dynamic_mask_inpaint_enabled,
                            inpaint_hook=inpaint_hook,
                        )
                        if "空" in (self.dynamic_mask_target_classes or self.target_classes):
                            logger.warning("対象クラスに「空」が含まれています。広域マスクになりやすいです。")
                        logger.info(
                            "対象マスク生成を有効化: "
                            f"classes={self.target_classes}, yolo={self.yolo_model_path}, sam={self.sam_model_path}"
                        )
                except Exception as e:
                    logger.warning(f"対象マスク生成の初期化に失敗したため無効化します: {e}")
                    self.enable_target_mask_generation = False

            splitters: Dict[str, Cross5FisheyeSplitter] = {}
            if self.is_stereo:
                frame_size = (
                    int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap_l is not None else 0,
                    int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap_l is not None else 0,
                )
                if frame_size[0] > 0 and frame_size[1] > 0:
                    splitters = self._create_cross5_splitters((frame_size[0], frame_size[1]))

            total = len(self.frame_indices)
            exported = 0
            stream_histories = {}
            stream_last_indices = {}

            for i, frame_idx in enumerate(self.frame_indices):
                if not self._is_running:
                    break

                # フレーム読み込み（ステレオ/単眼）
                if self.is_stereo:
                    # ステレオペア読み込み
                    cap_l.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret_l, frame_l = cap_l.read()
                    ret_r, frame_r = cap_r.read()

                    if not ret_l or not ret_r or frame_l is None or frame_r is None:
                        logger.warning(f"ステレオフレーム読み込み失敗: {frame_idx}")
                        continue

                    if self.enable_stereo_stitch and stitch_processor is not None:
                        try:
                            mode = str(self.stitching_mode).lower()
                            if mode == "high quality (hq)":
                                stitched = stitch_processor.stitch_high_quality([frame_l, frame_r])
                            elif mode == "depth-aware":
                                stitched = stitch_processor.stitch_depth_aware([frame_l, frame_r])
                            else:
                                stitched = stitch_processor.stitch_fast([frame_l, frame_r], mode='horizontal')
                            frames_to_process = [(stitched, '')]
                        except Exception as e:
                            logger.warning(f"ステッチング失敗（フレーム {frame_idx}）: {e}")
                            frames_to_process = [(frame_l, '_L'), (frame_r, '_R')]
                    else:
                        frames_to_process = [(frame_l, '_L'), (frame_r, '_R')]
                else:
                    # 単眼読み込み
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        logger.warning(f"フレーム読み込み失敗: {frame_idx}")
                        continue

                    frames_to_process = [(frame, '')]

                    # 各フレーム（L/R または単眼）を処理
                for frame, suffix in frames_to_process:
                    processed_frame = frame
                    force_mask_reanalysis = bool(
                        self.enable_cubemap or self.enable_perspective or self.is_stereo
                    )

                    # ファイル拡張子を決定（ステレオ/非ステレオ両方で使用）
                    ext = 'jpg' if self.format in ('jpg', 'jpeg') else self.format

                    # ステレオ・非ステッチ時は L/R を分離保存
                    if self.is_stereo and (not self.enable_stereo_stitch or suffix):
                        # パノラマ画像のみを L/ または R/ フォルダに保存
                        output_subdir = stereo_images_root / suffix.strip('_')  # 'L' or 'R'
                        output_subdir.mkdir(parents=True, exist_ok=True)

                        # ファイル名にサフィックスあり (_L or _R)
                        filename = f"{self.prefix}_{frame_idx:06d}{suffix}.{ext}"
                        filepath = output_subdir / filename

                        if ext == 'jpg':
                            saved = write_image(
                                filepath,
                                processed_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                            )
                        else:
                            saved = write_image(filepath, processed_frame)

                        if not saved:
                            logger.warning(f"保存失敗（フレーム {frame_idx}{suffix}）: {filepath}")
                            continue

                        fisheye_mask_path = None
                        if self.enable_target_mask_generation:
                            try:
                                history_key = suffix or "mono"
                                motion_history = stream_histories.get(history_key)
                                if motion_history is None:
                                    motion_history = deque(maxlen=max(2, self.dynamic_mask_motion_frames))
                                    stream_histories[history_key] = motion_history
                                prev_idx = stream_last_indices.get(history_key)
                                if prev_idx is not None and abs(int(frame_idx) - int(prev_idx)) > 1:
                                    motion_history.clear()
                                motion_history.append(processed_frame.copy())
                                stream_last_indices[history_key] = int(frame_idx)
                                motion_frames = list(motion_history)
                                if not self._save_target_mask(
                                    stereo_images_root,
                                    masks_root,
                                    filepath,
                                    frame_idx,
                                    processed_frame,
                                    target_mask_generator,
                                    motion_frames,
                                    force_mask_reanalysis,
                                    flatten_stereo_lr=True,
                                ):
                                    logger.warning(f"対象マスク保存失敗: {filepath}")
                                elif target_mask_cls is not None:
                                    fisheye_mask_path = target_mask_cls.build_mask_path(
                                        image_path=filepath,
                                        images_root=stereo_images_root,
                                        masks_root=masks_root,
                                        add_suffix=self.mask_add_suffix,
                                        suffix=self.mask_suffix,
                                        mask_ext=self.mask_output_format,
                                        flatten_stereo_lr=True,
                                    )
                            except Exception as e:
                                logger.warning(f"対象マスク生成エラー（フレーム {frame_idx}{suffix}）: {e}")

                        splitter = splitters.get(suffix)
                        if splitter is not None:
                            try:
                                split_views = splitter.split_image_with_valid_mask(processed_frame)
                                projected_masks = {}
                                if fisheye_mask_path is not None and fisheye_mask_path.exists():
                                    fisheye_mask_img = cv2.imread(str(fisheye_mask_path), cv2.IMREAD_GRAYSCALE)
                                    if fisheye_mask_img is not None:
                                        projected_masks = splitter.project_mask(fisheye_mask_img)
                                split_write_futures = []
                                for view_name, (view_img, _valid) in split_views.items():
                                    split_dir = stereo_images_root / f"{suffix.strip('_')}_{view_name}"
                                    split_dir.mkdir(parents=True, exist_ok=True)
                                    split_path = split_dir / f"{self.prefix}_{frame_idx:06d}{suffix}_{view_name}.{ext}"
                                    if write_executor is not None:
                                        if ext == 'jpg':
                                            split_write_futures.append((
                                                write_executor.submit(
                                                write_image,
                                                split_path,
                                                view_img,
                                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                                                ),
                                                split_path,
                                                "分割画像",
                                            ))
                                        else:
                                            split_write_futures.append((
                                                write_executor.submit(write_image, split_path, view_img),
                                                split_path,
                                                "分割画像",
                                            ))
                                        split_saved = True
                                    elif ext == 'jpg':
                                        split_saved = write_image(
                                            split_path,
                                            view_img,
                                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                                        )
                                    else:
                                        split_saved = write_image(split_path, view_img)
                                    if not split_saved:
                                        logger.warning(f"分割画像保存失敗: {split_path}")
                                        continue
                                    if not projected_masks or target_mask_cls is None:
                                        continue
                                    split_mask = projected_masks.get(view_name)
                                    if split_mask is None:
                                        continue
                                    split_mask_path = self._build_flat_split_mask_path(masks_root, split_path)
                                    split_mask_path.parent.mkdir(parents=True, exist_ok=True)
                                    if write_executor is not None:
                                        split_write_futures.append((
                                            write_executor.submit(write_image, split_mask_path, split_mask),
                                            split_mask_path,
                                            "分割マスク",
                                        ))
                                    else:
                                        saved_mask = write_image(split_mask_path, split_mask)
                                        if not saved_mask:
                                            logger.warning(f"分割マスク保存失敗: {split_mask_path}")
                                if write_executor is not None:
                                    for fut, out_path, kind in split_write_futures:
                                        if not bool(fut.result()):
                                            logger.warning(f"{kind}保存失敗: {out_path}")
                            except Exception as e:
                                logger.warning(f"cross5分割出力エラー（フレーム {frame_idx}{suffix}）: {e}")

                        exported += 1
                        continue  # ステレオの場合はここで次のフレームへ

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

                    # ファイルを保存（ステレオの場合は _L / _R サフィックス付き）
                    filename = f"{self.prefix}_{frame_idx:06d}{suffix}.{ext}"
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

                    if self.enable_target_mask_generation:
                        try:
                            history_key = suffix or "mono"
                            motion_history = stream_histories.get(history_key)
                            if motion_history is None:
                                motion_history = deque(maxlen=max(2, self.dynamic_mask_motion_frames))
                                stream_histories[history_key] = motion_history
                            prev_idx = stream_last_indices.get(history_key)
                            if prev_idx is not None and abs(int(frame_idx) - int(prev_idx)) > 1:
                                motion_history.clear()
                            motion_history.append(processed_frame.copy())
                            stream_last_indices[history_key] = int(frame_idx)
                            motion_frames = list(motion_history)
                            if not self._save_target_mask(
                                output_path,
                                masks_root,
                                filepath,
                                frame_idx,
                                processed_frame,
                                target_mask_generator,
                                motion_frames,
                                force_mask_reanalysis,
                            ):
                                logger.warning(f"対象マスク保存失敗: {filepath}")
                        except Exception as e:
                            logger.warning(f"対象マスク生成エラー（フレーム {frame_idx}{suffix}）: {e}")

                    exported += 1

                    # --- Cubemap 出力 ---
                    if self.enable_cubemap and equirect_processor:
                        try:
                            faces = equirect_processor.to_cubemap(
                                processed_frame, self.cubemap_face_size
                            )
                            cubemap_futures = []
                            for face_name, face_img in faces.items():
                                # 方向ごとのフォルダ: cubemap/front/, cubemap/back/ など
                                face_dir = output_path / "cubemap" / face_name
                                face_dir.mkdir(parents=True, exist_ok=True)

                                # ファイル名: keyframe_NNNNNN_front.jpg（サフィックスあり）
                                face_path = face_dir / f"keyframe_{frame_idx:06d}_{face_name}.{ext}"
                                if write_executor is not None:
                                    if ext == 'jpg':
                                        fut = write_executor.submit(
                                            write_image,
                                            face_path,
                                            face_img,
                                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                                        )
                                    else:
                                        fut = write_executor.submit(write_image, face_path, face_img)
                                    cubemap_futures.append((fut, face_path))
                                elif ext == 'jpg':
                                    if not write_image(
                                        face_path,
                                        face_img,
                                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                                    ):
                                        logger.warning(f"Cubemap保存失敗: {face_path}")
                                else:
                                    if not write_image(face_path, face_img):
                                        logger.warning(f"Cubemap保存失敗: {face_path}")
                            for fut, face_path in cubemap_futures:
                                if not bool(fut.result()):
                                    logger.warning(f"Cubemap保存失敗: {face_path}")
                            logger.debug(f"Cubemap 出力完了（フレーム {frame_idx}）")
                        except Exception as e:
                            logger.warning(f"Cubemap 出力エラー（フレーム {frame_idx}）: {e}")

                    # --- Perspective 出力 ---
                    if self.enable_perspective and equirect_processor:
                        try:
                            perspective_futures = []
                            for yaw in self.perspective_yaw_list:
                                for pitch in self.perspective_pitch_list:
                                    persp_img = equirect_processor.to_perspective(
                                        processed_frame,
                                        yaw=yaw, pitch=pitch,
                                        fov=self.perspective_fov,
                                        output_size=self.perspective_size
                                    )
                                    # 角度ごとのフォルダ: perspective/y+0_p+0/, perspective/y+90_p+0/ など
                                    angle_name = f"y{yaw:+.0f}_p{pitch:+.0f}"
                                    angle_dir = output_path / "perspective" / angle_name
                                    angle_dir.mkdir(parents=True, exist_ok=True)

                                    # ファイル名: keyframe_NNNNNN_y+0_p+0.jpg（サフィックスあり）
                                    persp_path = angle_dir / f"keyframe_{frame_idx:06d}_{angle_name}.{ext}"
                                    if write_executor is not None:
                                        if ext == 'jpg':
                                            fut = write_executor.submit(
                                                write_image,
                                                persp_path,
                                                persp_img,
                                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                                            )
                                        else:
                                            fut = write_executor.submit(write_image, persp_path, persp_img)
                                        perspective_futures.append((fut, persp_path))
                                    elif ext == 'jpg':
                                        if not write_image(
                                            persp_path,
                                            persp_img,
                                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                                        ):
                                            logger.warning(f"Perspective保存失敗: {persp_path}")
                                    else:
                                        if not write_image(persp_path, persp_img):
                                            logger.warning(f"Perspective保存失敗: {persp_path}")
                            for fut, persp_path in perspective_futures:
                                if not bool(fut.result()):
                                    logger.warning(f"Perspective保存失敗: {persp_path}")
                            logger.debug(f"Perspective 出力完了（フレーム {frame_idx}）")
                        except Exception as e:
                            logger.warning(f"Perspective 出力エラー（フレーム {frame_idx}）: {e}")

                self.progress.emit(i + 1, total,
                                   f"エクスポート: {i+1}/{total}")

            # キャプチャのクリーンアップ
            if self.is_stereo:
                if cap_l is not None:
                    cap_l.release()
                if cap_r is not None:
                    cap_r.release()
            else:
                if cap is not None:
                    cap.release()
            if write_executor is not None:
                write_executor.shutdown(wait=True)

            self.finished.emit(exported)

        except Exception as e:
            logger.exception("エクスポートワーカーエラー")
            self.error.emit(f"エクスポートエラー: {e}")
        finally:
            if write_executor is not None:
                write_executor.shutdown(wait=False)


class GenerateMasksWorker(QThread):
    """
    既存画像群に対して対象マスクを生成するワーカー。

    `ExportWorker` とは独立して、既に出力済みの images ディレクトリへ後処理適用できる。
    """

    progress = Signal(int, int, str)
    finished = Signal(int)
    error = Signal(str)

    def __init__(
        self,
        image_paths: List[str],
        images_root: str,
        target_classes: Optional[List[str]] = None,
        yolo_model_path: str = "yolo26n-seg.pt",
        sam_model_path: str = "sam3_t.pt",
        confidence_threshold: float = 0.25,
        detection_device: str = "auto",
        mask_output_dirname: str = "masks",
        mask_add_suffix: bool = True,
        mask_suffix: str = "_mask",
        mask_output_format: str = "same",
        jpeg_quality: int = 95,
        dynamic_mask_use_motion_diff: bool = True,
        dynamic_mask_motion_frames: int = 3,
        dynamic_mask_motion_threshold: int = 30,
        dynamic_mask_dilation_size: int = 5,
        dynamic_mask_use_yolo_sam: bool = True,
        dynamic_mask_target_classes: Optional[List[str]] = None,
        dynamic_mask_inpaint_enabled: bool = False,
        dynamic_mask_inpaint_module: str = "",
        parent: QObject = None,
    ):
        super().__init__(parent)
        self.image_paths = [Path(p) for p in image_paths]
        self.images_root = Path(images_root)
        self.target_classes = target_classes or []
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.confidence_threshold = confidence_threshold
        self.detection_device = detection_device
        self.mask_output_dirname = mask_output_dirname
        self.mask_add_suffix = mask_add_suffix
        self.mask_suffix = mask_suffix
        self.mask_output_format = mask_output_format
        self.jpeg_quality = jpeg_quality
        self.dynamic_mask_use_motion_diff = bool(dynamic_mask_use_motion_diff)
        self.dynamic_mask_motion_frames = max(2, int(dynamic_mask_motion_frames))
        self.dynamic_mask_motion_threshold = int(dynamic_mask_motion_threshold)
        self.dynamic_mask_dilation_size = max(0, int(dynamic_mask_dilation_size))
        self.dynamic_mask_use_yolo_sam = bool(dynamic_mask_use_yolo_sam)
        self.dynamic_mask_target_classes = list(dynamic_mask_target_classes or [])
        self.dynamic_mask_inpaint_enabled = bool(dynamic_mask_inpaint_enabled)
        self.dynamic_mask_inpaint_module = str(dynamic_mask_inpaint_module or "").strip()
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            from processing.target_mask_generator import TargetMaskGenerator

            inpaint_hook = None
            if self.dynamic_mask_inpaint_enabled and self.dynamic_mask_inpaint_module:
                try:
                    mod = __import__(self.dynamic_mask_inpaint_module, fromlist=['inpaint_frame'])
                    hook = getattr(mod, 'inpaint_frame', None)
                    if callable(hook):
                        inpaint_hook = hook
                except Exception as e:
                    logger.warning(f"インペイントモジュール読み込み失敗: {e}")

            generator = TargetMaskGenerator(
                yolo_model_path=self.yolo_model_path,
                sam_model_path=self.sam_model_path,
                confidence_threshold=self.confidence_threshold,
                device=self.detection_device,
                enable_motion_detection=self.dynamic_mask_use_motion_diff,
                motion_history_frames=self.dynamic_mask_motion_frames,
                motion_threshold=self.dynamic_mask_motion_threshold,
                motion_mask_dilation_size=self.dynamic_mask_dilation_size,
                enable_mask_inpaint=self.dynamic_mask_inpaint_enabled,
                inpaint_hook=inpaint_hook,
            )
            masks_root = self.images_root / self.mask_output_dirname
            total = len(self.image_paths)
            count = 0
            frame_history = deque(maxlen=max(2, self.dynamic_mask_motion_frames))

            for i, img_path in enumerate(self.image_paths):
                if not self._is_running:
                    break
                frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame_history.append(frame)

                classes_for_detection = (
                    self.dynamic_mask_target_classes or self.target_classes
                    if self.dynamic_mask_use_yolo_sam
                    else []
                )
                mask = generator.generate_mask(
                    frame,
                    classes_for_detection,
                    motion_frames=list(frame_history),
                )
                out_path = generator.build_mask_path(
                    image_path=img_path,
                    images_root=self.images_root,
                    masks_root=masks_root,
                    add_suffix=self.mask_add_suffix,
                    suffix=self.mask_suffix,
                    mask_ext=self.mask_output_format,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ext = out_path.suffix.lower().lstrip(".")
                if ext in ("jpg", "jpeg"):
                    saved = write_image(
                        out_path, mask, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                else:
                    saved = write_image(out_path, mask)

                if saved:
                    count += 1
                self.progress.emit(i + 1, total, f"マスク生成: {i+1}/{total}")

            self.finished.emit(count)
        except Exception as e:
            logger.exception("GenerateMasksWorker エラー")
            self.error.emit(f"マスク生成エラー: {e}")
