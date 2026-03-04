"""
メインウィンドウ - 360Split v2 GUI
全ウィジェットを統合するメインアプリケーションウィンドウ。

レイアウト:
  中央: VideoPlayerWidget
  下部: TimelineWidget (pyqtgraph スコアグラフ)
  右側ドック: SettingsPanel + KeyframeListWidget (タブ切り替え)

機能:
  - ドラッグ＆ドロップでの動画読み込み
  - メニューバー: ファイル(F), 表示(V)
  - 単一解析実行（Stage1→0→2→3）
  - Live Preview (パラメータ変更 → 判定再実行)
"""

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, List
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QTabWidget, QSplitter,
    QFileDialog, QMenuBar, QToolBar, QStatusBar,
    QMessageBox, QProgressBar, QLabel
)
from PySide6.QtCore import Qt, QSize, QUrl, QTimer
from PySide6.QtGui import QKeySequence, QAction, QDragEnterEvent, QDropEvent

from gui.video_player import VideoPlayerWidget
from gui.timeline_widget import TimelineWidget
from gui.trajectory_widget import TrajectoryWidget
from gui.settings_panel import SettingsPanel
from gui.settings_dialog import SettingsDialog
from gui.keyframe_panel import KeyframePanel
from gui.log_panel import LogPanel
from gui.analysis_dashboard import AnalysisDashboardWidget
from gui.analysis_data_loader import load_analysis_artifacts
from gui.export_dialog import ExportDialog
from gui.workers import UnifiedAnalysisWorker, PoseOnlyWorker, ExportWorker, FrameScoreData

from config import KeyframeConfig, NormalizationConfig

from utils.logger import get_logger
logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    360Split v2 メインウィンドウ

    全ウィジェットのライフサイクルとシグナル接続を管理。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("360Split v2 — キーフレーム抽出ツール")
        self.setGeometry(80, 60, 1600, 950)
        self.setAcceptDrops(True)

        # 状態
        self.video_path: Optional[str] = None
        self._stage1_scores: List[FrameScoreData] = []
        self._analysis_worker: Optional[UnifiedAnalysisWorker] = None
        self._pose_only_worker: Optional[PoseOnlyWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._analysis_masks: Dict[int, object] = {}
        self._trajectory_left: bool = False
        self._active_stage_mode_label: str = "解析"
        self._analysis_run_id: Optional[str] = None
        self._last_vo_summary: Dict[str, object] = {}
        self._last_pose_summary: Dict[str, object] = {}
        self._trajectory_runtime_counter: int = 0
        self._trajectory_runtime_stride: int = 5
        self._analysis_artifact_run_dir: Optional[Path] = None
        self._analysis_artifact_mtime: Dict[str, int] = {}
        self._analysis_artifact_payload: Dict[str, Any] = {}
        self._last_analysis_config: Dict[str, Any] = {}
        self._colmap_pending_context: Optional[Dict[str, Any]] = None
        self._colmap_run_action: Optional[QAction] = None
        self._analysis_artifact_poll_timer = QTimer(self)
        self._analysis_artifact_poll_timer.setInterval(400)
        self._analysis_artifact_poll_timer.timeout.connect(self._poll_analysis_artifacts)

        # ステレオ（OSV）対応
        self.is_stereo: bool = False
        self.stereo_left_path: Optional[str] = None
        self.stereo_right_path: Optional[str] = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_dock()
        self._setup_connections()
        self._apply_stylesheet()

        self.statusBar().showMessage("準備完了 — 動画ファイルをドラッグ＆ドロップで読み込めます")

    # ==================================================================
    # UI レイアウト
    # ==================================================================

    def _setup_ui(self):
        """中央ウィジェット: ビデオ + タイムライン"""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.setChildrenCollapsible(False)
        layout.addWidget(self._main_splitter, stretch=1)

        # ビデオプレーヤー
        self.video_player = VideoPlayerWidget()
        self._main_splitter.addWidget(self.video_player)

        # タイムライン
        self.timeline = TimelineWidget()
        self.timeline.setMinimumHeight(220)
        self.timeline.setMaximumHeight(360)

        # 擬似軌跡ビュー
        self.trajectory = TrajectoryWidget()
        self.trajectory.setMinimumSize(200, 200)

        # 下段: スコア + 擬似軌跡（左右配置を切り替え可能）
        self._bottom_splitter = QSplitter(Qt.Horizontal)
        self._bottom_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._bottom_splitter)
        self._rebuild_bottom_splitter()

        # 初期比率: ビデオを大きめ
        self._main_splitter.setStretchFactor(0, 4)
        self._main_splitter.setStretchFactor(1, 2)
        self._bottom_splitter.setStretchFactor(0, 3)
        self._bottom_splitter.setStretchFactor(1, 1)

        # ステータスバーにプログレスバーを追加
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(250)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self.statusBar().addPermanentWidget(self._progress_label)

    def _setup_dock(self):
        """右側ドック: 設定パネル + キーフレーム一覧 + 解析ログ (タブ切り替え)"""
        dock = QDockWidget("パネル", self)
        dock.setMinimumWidth(300)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )

        tab_widget = QTabWidget()

        # タブ 1: 設定パネル
        self.settings_panel = SettingsPanel()
        tab_widget.addTab(self.settings_panel, "⚙ 設定")

        # タブ 2: キーフレーム一覧
        self.keyframe_list = KeyframePanel()
        tab_widget.addTab(self.keyframe_list, "📋 キーフレーム")

        # タブ 3: 解析ログ
        self.log_panel = LogPanel()
        tab_widget.addTab(self.log_panel, "🧾 解析ログ")

        # タブ 4: 解析ダッシュボード
        self.analysis_dashboard = AnalysisDashboardWidget()
        tab_widget.addTab(self.analysis_dashboard, "📊 解析ダッシュボード")

        dock.setWidget(tab_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _setup_menu(self):
        menubar = self.menuBar()

        # ファイル(F)
        file_menu = menubar.addMenu("ファイル(&F)")

        open_action = QAction("開く(&O)...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(lambda: self.open_video())
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("キーフレームをエクスポート(&E)...", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_action.triggered.connect(lambda: self.export_keyframes())
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("終了(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 編集(E)
        edit_menu = menubar.addMenu("編集(&E)")

        settings_action = QAction("設定...(&S)", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._open_settings_dialog)
        edit_menu.addAction(settings_action)

        # 表示(V)
        view_menu = menubar.addMenu("表示(&V)")

        grid_action = QAction("グリッドオーバーレイ(&G)", self)
        grid_action.setCheckable(True)
        grid_action.triggered.connect(
            lambda checked: self.video_player.set_grid_overlay(checked)
        )
        view_menu.addAction(grid_action)

        traj_left_action = QAction("擬似軌跡を左に配置", self)
        traj_left_action.setCheckable(True)
        traj_left_action.setChecked(self._trajectory_left)
        traj_left_action.triggered.connect(self._set_trajectory_left)
        view_menu.addAction(traj_left_action)

        reset_layout_action = QAction("レイアウトをリセット", self)
        reset_layout_action.triggered.connect(self._reset_layout_sizes)
        view_menu.addAction(reset_layout_action)

        # 解析(A)
        analysis_menu = menubar.addMenu("解析(&A)")
        run_action = QAction("解析実行(&R)", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(lambda: self._run_analysis(trigger_source="menu:run"))
        analysis_menu.addAction(run_action)

    def _setup_toolbar(self):
        tb = self.addToolBar("メイン")
        tb.setIconSize(QSize(20, 20))

        tb.addAction("📂 開く", self.open_video)
        tb.addSeparator()
        tb.addAction("🚀 解析実行", lambda: self._run_analysis(trigger_source="toolbar:run"))
        self._colmap_run_action = tb.addAction("⏭ COLMAP実行", self._on_toolbar_run_colmap)
        if self._colmap_run_action:
            self._colmap_run_action.setEnabled(False)
        tb.addSeparator()
        tb.addAction("💾 エクスポート", self.export_keyframes)

    def _rebuild_bottom_splitter(self):
        """下段splitterの並び順を更新"""
        if self._bottom_splitter.indexOf(self.timeline) != -1:
            self.timeline.setParent(None)
        if self._bottom_splitter.indexOf(self.trajectory) != -1:
            self.trajectory.setParent(None)

        if self._trajectory_left:
            self._bottom_splitter.addWidget(self.trajectory)
            self._bottom_splitter.addWidget(self.timeline)
        else:
            self._bottom_splitter.addWidget(self.timeline)
            self._bottom_splitter.addWidget(self.trajectory)
        self._bottom_splitter.setSizes([900, 320])

    def _set_trajectory_left(self, checked: bool):
        self._trajectory_left = bool(checked)
        self._rebuild_bottom_splitter()

    def _reset_layout_sizes(self):
        self._main_splitter.setSizes([700, 320])
        self._bottom_splitter.setSizes([900, 320])

    def _setup_connections(self):
        """全シグナル/スロットを接続"""
        # ビデオプレーヤー → タイムライン同期
        self.video_player.frame_changed.connect(self.timeline.set_position)
        self.video_player.keyframe_marked.connect(self._on_manual_mark)

        # タイムライン → ビデオプレーヤー同期
        self.timeline.positionChanged.connect(self.video_player.seek_to_frame)
        self.timeline.keyframeClicked.connect(self.video_player.seek_to_frame)
        self.timeline.keyframeAddRequested.connect(self._on_manual_mark)
        self.timeline.keyframeRemoveRequested.connect(self._on_manual_unmark)
        self.trajectory.frameSelected.connect(self.video_player.seek_to_frame)

        # キーフレーム一覧 → ビデオプレーヤー
        self.keyframe_list.keyframe_selected.connect(self.video_player.seek_to_frame)
        self.keyframe_list.keyframe_deleted.connect(self._on_keyframe_deleted)

        # 設定パネル → Live Preview
        self.settings_panel.setting_changed.connect(self._on_live_preview)
        self.settings_panel.run_analysis_requested.connect(
            lambda: self._run_analysis(trigger_source="settings_panel:run_analysis")
        )
        self.settings_panel.run_stage2_requested.connect(
            lambda: self._run_analysis(trigger_source="settings_panel:run_stage2_legacy")
        )
        self.settings_panel.open_settings_requested.connect(self._open_settings_dialog)
        self.analysis_dashboard.apply_settings_requested.connect(self._on_apply_dashboard_settings)
        self.analysis_dashboard.run_colmap_requested.connect(self._on_dashboard_run_colmap)

    def _log_analysis_request(self, trigger_source: str, stage_mode_label: str, config: Dict, overrides: Dict):
        run_id = str(config.get("analysis_run_id", "n/a"))
        logger.info(
            "analysis_request,"
            f" analysis_run_id={run_id},"
            f" trigger={trigger_source},"
            f" stage_mode={stage_mode_label},"
            f" has_stage1_scores={bool(self._stage1_scores)},"
            f" stage1_scores_count={len(self._stage1_scores)},"
            f" overrides={json.dumps(overrides, ensure_ascii=False, sort_keys=True)},"
            f" enable_stage0_scan={bool(config.get('enable_stage0_scan', config.get('ENABLE_STAGE0_SCAN', True)))},"
            f" enable_stage3_refinement={bool(config.get('enable_stage3_refinement', config.get('ENABLE_STAGE3_REFINEMENT', True)))},"
            f" stage0_stride={int(config.get('stage0_stride', config.get('STAGE0_STRIDE', 5)))}"
        )

    def _reset_colmap_pending_state(self, clear_candidates: bool = True) -> None:
        self._colmap_pending_context = None
        if self._colmap_run_action is not None:
            self._colmap_run_action.setEnabled(False)
        if clear_candidates:
            self.analysis_dashboard.set_colmap_candidates([])

    # ==================================================================
    # ドラッグ＆ドロップ
    # ==================================================================

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.osv')):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.osv')):
                self._load_video(path)
                return

    # ==================================================================
    # ビデオ読み込み
    # ==================================================================

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ビデオファイルを開く", "",
            "ビデオ (*.mp4 *.mov *.avi *.mkv *.webm *.osv);;すべて (*)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        try:
            # 進行中ワーカーを停止し、前回解析の反映競合を防ぐ
            self._stop_workers()

            # 前回解析結果を先にクリア
            self._stage1_scores.clear()
            self._analysis_masks.clear()
            self._last_pose_summary = {}
            self.keyframe_list.clear()
            self.timeline.set_keyframes([], [])
            self.timeline.set_stationary_ranges([])
            self.timeline.set_score_data([], [])
            self.trajectory.set_frame_data([], [])
            self.trajectory.reset_runtime_trajectory()
            self.video_player.set_keyframe_indices([])
            self.analysis_dashboard.reset_for_run("-", self.settings_panel.get_selector_dict())
            self._reset_colmap_pending_state(clear_candidates=False)

            self.video_path = path

            # OSV ファイル判定
            if path.lower().endswith('.osv'):
                # DualVideoLoader でストリーム分離
                from core.video_loader import DualVideoLoader

                logger.info(f"OSV ファイルを検出: {path}")
                loader = DualVideoLoader()
                metadata = loader.load(path)

                # ステレオ情報を保存
                self.is_stereo = True
                self.stereo_left_path = loader.left_path
                self.stereo_right_path = loader.right_path

                # ステレオビデオプレーヤーに読み込み
                metadata = self.video_player.load_video_stereo(loader.left_path, loader.right_path)

                logger.info(f"ステレオモード有効化: L={loader.left_path}, R={loader.right_path}")
                status_msg = (
                    f"読み込み完了（OSV - ステレオ）: {Path(path).name}  "
                    f"({metadata.width}×{metadata.height}, "
                    f"{metadata.fps:.1f}fps, {metadata.frame_count}フレーム) - ステレオ表示可能"
                )
            else:
                # 通常の単眼ビデオ
                self.is_stereo = False
                self.stereo_left_path = None
                self.stereo_right_path = None

                metadata = self.video_player.load_video(path)

                status_msg = (
                    f"読み込み完了: {Path(path).name}  "
                    f"({metadata.width}×{metadata.height}, "
                    f"{metadata.fps:.1f}fps, {metadata.frame_count}フレーム)"
                )

            self.timeline.set_duration(metadata.frame_count, metadata.fps)
            self.keyframe_list.set_video_path(path)

            self.statusBar().showMessage(status_msg)

        except Exception as e:
            logger.exception(f"ビデオ読み込みエラー: {path}")
            QMessageBox.critical(self, "エラー", f"読み込み失敗:\n{e}")

    def _open_settings_dialog(self):
        """
        設定ダイアログを開く

        プリセット選択、詳細パラメータ調整、出力設定などを行うダイアログを表示します。

        Note:
        -----
        settings_dialog (モーダルダイアログ) で OK が押されると:
        1. 設定が ~/.360split/settings.json に保存される
        2. settings_panel (右サイドパネル) が自動的に再読み込みされる
        3. Live Preview が更新されて変更が反映される
        """
        dialog = SettingsDialog(self)

        # 現在の設定を読み込み
        current_settings = self.settings_panel.get_config().to_selector_dict()

        if dialog.exec():
            # OKが押された場合、設定をsettings_panelに反映
            # （SettingsDialogは自動的に設定を保存するので、ここでは何もしない）
            logger.info("設定ダイアログが適用されました")

            # 設定パネルをリロード（保存された設定を反映）
            self.settings_panel.reload_from_file()
            self.analysis_dashboard.set_current_settings(self.settings_panel.get_selector_dict())
            logger.info("設定パネルを再読み込みしました")

    # ==================================================================
    # 解析実行（Stage1->0->2->3）
    # ==================================================================

    def _run_analysis(self, trigger_source: str = "unknown"):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "ビデオを先に開いてください")
            return

        self._stop_workers()
        self._reset_colmap_pending_state(clear_candidates=True)
        self._stage1_scores.clear()
        self._analysis_masks.clear()
        self._trajectory_runtime_counter = 0
        self._last_pose_summary = {}
        self.trajectory.reset_runtime_trajectory()

        config = self.settings_panel.get_selector_dict()
        pose_backend = str(config.get("POSE_BACKEND", config.get("pose_backend", "vo")) or "vo").strip().lower()
        if pose_backend not in {"vo", "colmap"}:
            pose_backend = "vo"
        raw_pipeline_mode = str(
            config.get("COLMAP_PIPELINE_MODE", config.get("colmap_pipeline_mode", "")) or ""
        ).strip().lower()
        if raw_pipeline_mode not in {"", "legacy", "minimal_v1"}:
            raw_pipeline_mode = ""
        pipeline_mode = raw_pipeline_mode if raw_pipeline_mode else ("minimal_v1" if pose_backend == "colmap" else "legacy")
        minimal_mode = bool(pose_backend == "colmap" and pipeline_mode == "minimal_v1")
        raw_policy = str(
            config.get("COLMAP_KEYFRAME_POLICY", config.get("colmap_keyframe_policy", "")) or ""
        ).strip().lower()
        if raw_policy not in {"", "legacy", "stage2_relaxed", "stage1_only"}:
            raw_policy = ""
        keyframe_policy = raw_policy if raw_policy else ("stage2_relaxed" if pose_backend == "colmap" else "legacy")
        raw_target_mode = str(
            config.get("COLMAP_KEYFRAME_TARGET_MODE", config.get("colmap_keyframe_target_mode", "")) or ""
        ).strip().lower()
        if raw_target_mode not in {"", "fixed", "auto"}:
            raw_target_mode = ""
        target_mode = raw_target_mode if raw_target_mode else ("auto" if pose_backend == "colmap" else "fixed")
        mask_profile = str(
            config.get("COLMAP_ANALYSIS_MASK_PROFILE", config.get("colmap_analysis_mask_profile", "")) or ""
        ).strip().lower()
        if mask_profile not in {"legacy", "colmap_safe"}:
            mask_profile = "colmap_safe" if pose_backend == "colmap" else "legacy"
        config["pose_backend"] = pose_backend
        config["POSE_BACKEND"] = pose_backend
        config["colmap_pipeline_mode"] = pipeline_mode
        config["COLMAP_PIPELINE_MODE"] = pipeline_mode
        config["colmap_minimal_mode"] = minimal_mode
        config["COLMAP_MINIMAL_MODE"] = minimal_mode
        config["colmap_keyframe_policy"] = keyframe_policy
        config["COLMAP_KEYFRAME_POLICY"] = keyframe_policy
        config["colmap_keyframe_target_mode"] = target_mode
        config["COLMAP_KEYFRAME_TARGET_MODE"] = target_mode
        config["colmap_analysis_mask_profile"] = mask_profile
        config["COLMAP_ANALYSIS_MASK_PROFILE"] = mask_profile

        if pose_backend == "colmap" and minimal_mode:
            config["enable_stage0_scan"] = False
            config["ENABLE_STAGE0_SCAN"] = False
            config["enable_stage3_refinement"] = False
            config["ENABLE_STAGE3_REFINEMENT"] = False
            config["enable_dynamic_mask_removal"] = False
            config["ENABLE_DYNAMIC_MASK_REMOVAL"] = False
        elif pose_backend == "colmap" and keyframe_policy != "legacy":
            config["enable_stage0_scan"] = False
            config["ENABLE_STAGE0_SCAN"] = False
            config["enable_stage3_refinement"] = False
            config["ENABLE_STAGE3_REFINEMENT"] = False
        if pose_backend == "colmap" and mask_profile == "colmap_safe":
            classes = config.get("dynamic_mask_target_classes", config.get("DYNAMIC_MASK_TARGET_CLASSES", []))
            if not isinstance(classes, list):
                classes = list(classes) if classes else []
            filtered = [c for c in classes if str(c) != "空"]
            config["colmap_analysis_target_classes"] = filtered
            config["COLMAP_ANALYSIS_TARGET_CLASSES"] = list(filtered)
        else:
            classes = config.get("dynamic_mask_target_classes", config.get("DYNAMIC_MASK_TARGET_CLASSES", []))
            if not isinstance(classes, list):
                classes = list(classes) if classes else []
            config["colmap_analysis_target_classes"] = list(classes)
            config["COLMAP_ANALYSIS_TARGET_CLASSES"] = list(classes)

        stage0_on = bool(config.get("enable_stage0_scan", config.get("ENABLE_STAGE0_SCAN", True)))
        stage3_on = bool(config.get("enable_stage3_refinement", config.get("ENABLE_STAGE3_REFINEMENT", True)))
        if minimal_mode:
            stage_mode_label = "COLMAP Minimal v1 (Stage1->Stage2)"
        elif pose_backend == "colmap" and keyframe_policy == "stage1_only":
            stage_mode_label = "Unified(Stage1 only)"
        elif pose_backend == "colmap" and keyframe_policy == "stage2_relaxed":
            stage_mode_label = f"Unified(Stage1->2 relaxed, target={target_mode})"
        else:
            stage_mode_label = f"Unified(Stage1->{'0->' if stage0_on else ''}2{'->3' if stage3_on else ''})"
        self._analysis_run_id = str(uuid.uuid4())
        config["analysis_mode"] = "full"
        config["analysis_run_id"] = self._analysis_run_id
        config["ANALYSIS_RUN_ID"] = self._analysis_run_id
        self._last_analysis_config = dict(config)
        self.analysis_dashboard.reset_for_run(self._analysis_run_id, self.settings_panel.get_selector_dict())
        self._start_analysis_artifact_poll(self._analysis_run_id)
        self._log_analysis_request(
            trigger_source=trigger_source,
            stage_mode_label=stage_mode_label,
            config=config,
            overrides={},
        )
        self._active_stage_mode_label = "解析"
        pause_before_colmap = bool(pose_backend == "colmap")
        self._analysis_worker = UnifiedAnalysisWorker(
            self.video_path,
            config=config,
            pause_before_colmap=pause_before_colmap,
        )
        self._analysis_worker.progress.connect(self._on_progress)
        self._analysis_worker.stage1_batch.connect(self._on_stage1_batch)
        self._analysis_worker.stage1_finished.connect(self._on_stage1_finished)
        self._analysis_worker.frame_scores_updated.connect(self._on_scores_updated)
        self._analysis_worker.trajectory_updated.connect(self._on_trajectory_updated)
        self._analysis_worker.keyframes_found.connect(self._on_keyframes_found)
        self._analysis_worker.pose_pending.connect(self._on_pose_pending)
        self._analysis_worker.pose_finished.connect(self._on_pose_finished)
        self._analysis_worker.analysis_finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_error)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self.statusBar().showMessage("解析実行中... (Analysis→Pose)")
        self._analysis_worker.start()

    def _on_stage1_batch(self, batch: list):
        """Stage 1 バッチ結果をプログレッシブにグラフ追加"""
        self._stage1_scores.extend(batch)
        self.analysis_dashboard.update_stage1_batch(list(batch))
        norm_factor = 1000.0  # NormalizationConfig.SHARPNESS_NORM_FACTOR
        indices = [s.frame_index for s in batch]
        sharpness = [min(s.sharpness / norm_factor, 1.0) for s in batch]
        self.timeline.append_score_batch(indices, sharpness)

    def _on_stage1_finished(self, all_scores: list):
        """Stage 1 完了"""
        self._stage1_scores = all_scores
        # 統合解析中は Stage1 は中間段なのでプログレスバーを維持
        if not (self._analysis_worker and self._analysis_worker.isRunning()):
            self._progress_bar.setVisible(False)

        # 全データでグラフを更新
        norm_factor = 1000.0
        indices = [s.frame_index for s in all_scores]
        sharpness = [min(s.sharpness / norm_factor, 1.0) for s in all_scores]
        vo_conf = [float(getattr(s, "vo_confidence", 0.0)) for s in all_scores]
        vo_low, vo_mid = self._get_vo_conf_thresholds()
        self.timeline.set_score_data(
            indices,
            sharpness,
            vo_confidence=vo_conf,
            vo_confidence_low_threshold=vo_low,
            vo_confidence_mid_threshold=vo_mid,
        )
        self.timeline.set_stationary_ranges([])
        vo_summary = self._compute_vo_summary(all_scores)
        self._last_vo_summary = vo_summary
        self.trajectory.set_frame_data(all_scores, self.keyframe_list.keyframe_frames, vo_summary=vo_summary)

        self.statusBar().showMessage(
            f"Stage 1 完了: {len(all_scores)} フレームをスキャン。"
            "同一実行内で Stage2/3 に進みます。"
        )

    def _on_scores_updated(self, updated: list):
        """解析でGRIC/SSIM付きスコア更新"""
        self._stage1_scores = updated
        norm_factor = 1000.0
        indices = [s.frame_index for s in updated]
        sharpness = [min(s.sharpness / norm_factor, 1.0) for s in updated]
        gric = [s.gric for s in updated]
        ssim_change = [1.0 - s.ssim for s in updated]
        vo_conf = [float(getattr(s, "vo_confidence", 0.0)) for s in updated]
        vo_low, vo_mid = self._get_vo_conf_thresholds()

        # GRICデータがあるか確認
        has_gric = any(g > 0 for g in gric)
        has_ssim = any(sc > 0 for sc in ssim_change)

        self.timeline.set_score_data(
            indices, sharpness,
            gric=gric if has_gric else None,
            ssim_change=ssim_change if has_ssim else None,
            vo_confidence=vo_conf,
            vo_confidence_low_threshold=vo_low,
            vo_confidence_mid_threshold=vo_mid,
        )
        self.timeline.set_stationary_ranges(self._extract_stationary_ranges(updated))
        vo_summary = self._compute_vo_summary(updated)
        self._last_vo_summary = vo_summary
        self.trajectory.set_frame_data(updated, self.keyframe_list.keyframe_frames, vo_summary=vo_summary)

    def _on_analysis_finished(self):
        self._poll_analysis_artifacts()
        self._stop_analysis_artifact_poll()
        self.trajectory.flush_runtime_trajectory()
        self._progress_bar.setVisible(False)
        n = len(self.keyframe_list.keyframe_frames)
        logger.info(
            "analysis_finished,"
            f" analysis_run_id={self._analysis_run_id or 'n/a'},"
            f" stage_mode={self._active_stage_mode_label},"
            f" keyframes={n}"
        )
        attempts = int(self._last_vo_summary.get("attempted", 0))
        valid = int(self._last_vo_summary.get("valid", 0))
        pose_valid = int(self._last_vo_summary.get("pose_valid", 0))
        ratio = (valid / attempts) if attempts > 0 else 0.0
        pose_backend = str(self._last_pose_summary.get("backend", "vo"))
        pose_traj = int(self._last_pose_summary.get("trajectory_count", 0))
        pose_sel = int(self._last_pose_summary.get("selected_count", 0))
        pose_fail = str(self._last_pose_summary.get("failure_reason", "") or "")
        has_pending_colmap = bool(self._colmap_pending_context)
        if has_pending_colmap and pose_backend == "colmap":
            self.statusBar().showMessage(
                f"解析完了: {n} キーフレーム / VO有効率 {ratio:.1%} ({valid}/{attempts})。"
                "COLMAPは手動実行待機中です。"
            )
            QMessageBox.information(
                self,
                "解析完了（COLMAP待機）",
                f"解析完了: {n} キーフレームを検出しました\n"
                f"VO有効率: {ratio:.1%} ({valid}/{attempts})\n"
                "COLMAPはまだ未実行です。\n"
                "解析ダッシュボードまたはツールバーの「⏭ COLMAP実行」から実行してください。",
            )
            return

        self.statusBar().showMessage(
            f"解析完了: {n} キーフレーム / VO有効率 {ratio:.1%} ({valid}/{attempts}), "
            f"VO pose={pose_valid}, backend={pose_backend}, pose={pose_traj}, selected={pose_sel}"
        )
        pose_line = f"Pose backend: {pose_backend} / trajectory: {pose_traj} / selected: {pose_sel}"
        if pose_fail:
            pose_line += f"\\nPose失敗: {pose_fail}"
        QMessageBox.information(
            self,
            "完了",
            f"解析完了: {n} キーフレームを検出しました\n"
            f"VO有効率: {ratio:.1%} ({valid}/{attempts})\n"
            f"VO pose有効点: {pose_valid}\n"
            f"{pose_line}",
        )

    def _on_pose_finished(self, payload: dict):
        self._last_pose_summary = dict(payload or {})
        self.trajectory.set_pose_summary(self._last_pose_summary)
        self.analysis_dashboard.update_pose_summary(self._last_pose_summary)
        backend = str(self._last_pose_summary.get("backend", "vo"))
        traj = int(self._last_pose_summary.get("trajectory_count", 0))
        selected = int(self._last_pose_summary.get("selected_count", 0))
        fail = str(self._last_pose_summary.get("failure_reason", "") or "")
        if fail:
            logger.warning(f"[COLMAP_LAST_ERROR] {fail}")
        else:
            logger.info(f"pose_finished, backend={backend}, trajectory={traj}, selected={selected}")
        if backend == "colmap" and self._colmap_pending_context:
            self.statusBar().showMessage(
                f"COLMAP完了: trajectory={traj}, selected={selected}。選択を変更して再実行できます。"
            )

    def _on_pose_pending(self, payload: dict):
        info = dict(payload or {})
        run_id = str(info.get("run_id", self._analysis_run_id or "")).strip()
        preview_indices = [
            int(v) for v in list(info.get("stage2_colmap_preview_indices", []) or []) if isinstance(v, (int, float))
        ]
        candidates = self._collect_colmap_candidates(run_id)
        self._colmap_pending_context = {
            "run_id": run_id,
            "backend": "colmap",
            "stage2_colmap_preview_indices": preview_indices,
            "final_keyframes_count": int(info.get("final_keyframes_count", len(candidates))),
            "candidates": list(candidates),
            "config": dict(self._last_analysis_config or self.settings_panel.get_selector_dict()),
            "video_path": str(self.video_path or ""),
        }
        self.analysis_dashboard.set_colmap_candidates(candidates)
        if self._colmap_run_action is not None:
            self._colmap_run_action.setEnabled(bool(candidates))
        self._last_pose_summary = {
            "enabled": True,
            "backend": "colmap",
            "trajectory_count": 0,
            "selected_count": 0,
            "selection_stats": {},
            "failure_reason": None,
            "pending": True,
        }
        self.statusBar().showMessage(
            f"解析完了。COLMAP実行待機中（候補 {len(candidates)} 件）。ダッシュボードまたはツールバーから実行してください。"
        )
        logger.info(
            "pose_pending,"
            f" analysis_run_id={run_id},"
            f" candidates={len(candidates)},"
            f" preview_indices={len(preview_indices)}"
        )

    def _on_dashboard_run_colmap(self, selected_frames: list):
        self._run_pending_colmap(
            selected_frames=selected_frames,
            trigger_source="dashboard:run_colmap",
        )

    def _on_toolbar_run_colmap(self):
        self._run_pending_colmap(
            selected_frames=self.analysis_dashboard.selected_colmap_frames(),
            trigger_source="toolbar:run_colmap",
        )

    def _run_pending_colmap(self, selected_frames: Optional[List[int]], trigger_source: str) -> None:
        if self._pose_only_worker and self._pose_only_worker.isRunning():
            QMessageBox.warning(self, "警告", "COLMAP実行中です。完了後に再実行してください。")
            return
        pending = self._colmap_pending_context or {}
        if not pending:
            QMessageBox.warning(self, "警告", "COLMAP実行待機中の解析結果がありません。先に解析を実行してください。")
            return
        if not self.video_path:
            QMessageBox.warning(self, "警告", "ビデオが読み込まれていません。")
            return
        selected = sorted({int(v) for v in list(selected_frames or []) if isinstance(v, (int, float)) and int(v) >= 0})
        if not selected:
            QMessageBox.warning(self, "警告", "COLMAP採用画像が0件です。1件以上選択してください。")
            return
        run_id = str(pending.get("run_id", self._analysis_run_id or "")).strip()
        if not run_id:
            QMessageBox.warning(self, "警告", "run_id が不明なため COLMAP を実行できません。")
            return
        cfg = dict(pending.get("config") or self._last_analysis_config or self.settings_panel.get_selector_dict())
        cfg["pose_backend"] = "colmap"
        cfg["POSE_BACKEND"] = "colmap"
        preview_indices = [
            int(v)
            for v in list(pending.get("stage2_colmap_preview_indices", []) or [])
            if isinstance(v, (int, float))
        ]

        self._pose_only_worker = PoseOnlyWorker(
            video_path=self.video_path,
            run_id=run_id,
            selected_frame_indices=selected,
            config=cfg,
            colmap_preview_frame_indices=preview_indices,
        )
        self._pose_only_worker.progress.connect(self._on_progress)
        self._pose_only_worker.pose_finished.connect(self._on_pose_finished)
        self._pose_only_worker.error.connect(self._on_error)
        self._pose_only_worker.finished.connect(self._on_pose_only_finished)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self.statusBar().showMessage(f"COLMAP実行中... ({len(selected)} 枚, source={trigger_source})")
        logger.info(
            "pose_only_request,"
            f" analysis_run_id={run_id},"
            f" trigger={trigger_source},"
            f" selected={len(selected)},"
            f" preview={len(preview_indices)}"
        )
        self._pose_only_worker.start()

    def _on_pose_only_finished(self):
        self._progress_bar.setVisible(False)
        self._pose_only_worker = None

    def _collect_colmap_candidates(self, run_id: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        frames = list(self.keyframe_list.keyframe_frames or [])
        scores = list(self.keyframe_list.keyframe_scores or [])
        if frames:
            for i, frame_idx in enumerate(frames):
                score = scores[i] if i < len(scores) else 0.0
                rows.append({"frame_index": int(frame_idx), "score": float(score)})
        else:
            rows = self._load_stage3_keyframes_from_tmp_run(run_id)

        dedup: Dict[int, float] = {}
        for row in rows:
            try:
                frame_idx = int(row.get("frame_index", -1))
            except Exception:
                continue
            if frame_idx < 0:
                continue
            try:
                score = float(row.get("score", 0.0))
            except Exception:
                score = 0.0
            dedup[frame_idx] = score
        out = [{"frame_index": idx, "score": dedup[idx]} for idx in sorted(dedup.keys())]
        return out

    @staticmethod
    def _load_stage3_keyframes_from_tmp_run(run_id: str) -> List[Dict[str, Any]]:
        rid = str(run_id or "").strip()
        if not rid:
            return []
        path = Path.home() / ".360split" / "tmp_runs" / rid / "stage3_keyframes.jsonl"
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        item = json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(item, dict):
                        continue
                    frame_idx = item.get("frame_index")
                    if not isinstance(frame_idx, (int, float)):
                        continue
                    score = item.get("combined_stage3", item.get("combined_score", item.get("score", 0.0)))
                    try:
                        score_v = float(score)
                    except Exception:
                        score_v = 0.0
                    rows.append({"frame_index": int(frame_idx), "score": score_v})
        except Exception:
            return []
        return rows

    # ==================================================================
    # 共通コールバック
    # ==================================================================

    def _on_progress(self, current: int, total: int, message: str = ""):
        pct = int(current / max(total, 1) * 100)
        self._progress_bar.setValue(pct)
        self._progress_label.setText(message)
        if message:
            self.statusBar().showMessage(message)

    def _on_keyframes_found(self, keyframes):
        """キーフレーム検出結果を全ウィジェットに反映"""
        frames = [kf.frame_index for kf in keyframes]
        scores = [kf.combined_score for kf in keyframes]
        self._analysis_masks = {
            int(kf.frame_index): kf.dynamic_mask
            for kf in keyframes
            if getattr(kf, "dynamic_mask", None) is not None
        }
        self.keyframe_list.set_keyframes(frames, scores)
        self._apply_keyframe_view_state(frames, scores)
        self._suggest_keyframe_gaps()

    def _on_trajectory_updated(self, payload: dict):
        self._trajectory_runtime_counter += 1
        stride = max(1, int(self._trajectory_runtime_stride))
        is_keyframe = bool(payload.get("is_keyframe", False))
        force = is_keyframe or (self._trajectory_runtime_counter % stride == 0)
        self.trajectory.append_runtime_pose(payload, force=force)

    def _on_error(self, msg: str):
        self._stop_analysis_artifact_poll()
        self._progress_bar.setVisible(False)
        self.statusBar().showMessage(f"エラー: {msg}")
        QMessageBox.critical(self, "エラー", msg)

    def _on_manual_mark(self, frame_idx: int):
        """手動キーフレームマーク"""
        if frame_idx not in self.keyframe_list.keyframe_frames:
            self.keyframe_list.keyframe_frames.append(frame_idx)
            self.keyframe_list.keyframe_scores.append(self._compute_manual_stage3_score(frame_idx))
            self._refresh_manual_keyframes()
            self._suggest_keyframe_gaps()

    def _on_manual_unmark(self, frame_idx: int):
        if frame_idx in self.keyframe_list.keyframe_frames:
            idx = self.keyframe_list.keyframe_frames.index(frame_idx)
            self.keyframe_list.keyframe_frames.pop(idx)
            if idx < len(self.keyframe_list.keyframe_scores):
                self.keyframe_list.keyframe_scores.pop(idx)
            self._refresh_manual_keyframes()
            self._suggest_keyframe_gaps()

    def _on_keyframe_deleted(self, frame_idx: int):
        self.timeline.remove_keyframe(frame_idx)
        self._refresh_manual_keyframes()
        self._suggest_keyframe_gaps()

    def _refresh_manual_keyframes(self):
        pairs = sorted(
            zip(self.keyframe_list.keyframe_frames, self.keyframe_list.keyframe_scores),
            key=lambda x: int(x[0]),
        )
        self.keyframe_list.keyframe_frames = [int(p[0]) for p in pairs]
        self.keyframe_list.keyframe_scores = [float(p[1]) for p in pairs]
        self.keyframe_list._load_thumbnail_images()
        self.keyframe_list._update_display()
        self._apply_keyframe_view_state(self.keyframe_list.keyframe_frames, self.keyframe_list.keyframe_scores)

    def _compute_manual_stage3_score(self, frame_idx: int) -> float:
        if not self._stage1_scores:
            return 0.5
        score = next((s for s in self._stage1_scores if int(getattr(s, "frame_index", -1)) == int(frame_idx)), None)
        if score is None:
            return 0.5
        cfg = self.settings_panel.get_selector_dict()
        w_base = float(cfg.get("STAGE3_WEIGHT_BASE", 0.70))
        w_traj = float(cfg.get("STAGE3_WEIGHT_TRAJECTORY", 0.25))
        w_risk = float(cfg.get("STAGE3_WEIGHT_STAGE0_RISK", 0.05))
        base = float(getattr(score, "combined_stage2", getattr(score, "combined", 0.5)))
        traj = float(np.clip(getattr(score, "trajectory_consistency", 0.5), 0.0, 1.0))
        vo_conf = float(np.clip(getattr(score, "vo_confidence", 0.0), 0.0, 1.0))
        traj_effective = float(np.clip(traj * (0.5 + 0.5 * vo_conf), 0.0, 1.0))
        risk = float(np.clip(getattr(score, "stage0_motion_risk", 0.0), 0.0, 1.0))
        return float(np.clip(w_base * base + w_traj * traj_effective - w_risk * risk, 0.0, 1.0))

    def _suggest_keyframe_gaps(self):
        if not self._stage1_scores or len(self.keyframe_list.keyframe_frames) < 2:
            return
        cfg = self.settings_panel.get_selector_dict()
        max_interval = int(cfg.get("MAX_KEYFRAME_INTERVAL", 60))
        gap_threshold = int(max(2, round(max_interval * 1.5)))
        frames_sorted = sorted(int(f) for f in self.keyframe_list.keyframe_frames)
        for left, right in zip(frames_sorted[:-1], frames_sorted[1:]):
            if (right - left) <= gap_threshold:
                continue
            candidates = [
                s for s in self._stage1_scores
                if left < int(getattr(s, "frame_index", -1)) < right
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda s: self._compute_manual_stage3_score(int(getattr(s, "frame_index", -1))))
            best_idx = int(getattr(best, "frame_index", -1))
            self.statusBar().showMessage(
                f"キーフレーム間隔が大きい区間を検出: {left}-{right}。候補フレーム {best_idx} を右クリックで追加できます。"
            )
            break

    def _apply_keyframe_view_state(self, frames: list[int], scores: list[float]):
        self.timeline.set_keyframes(frames, scores)
        self.video_player.set_keyframe_indices(frames)
        self.trajectory.set_frame_data(self._stage1_scores, frames, vo_summary=self._compute_vo_summary(self._stage1_scores))

    # ==================================================================
    # Live Preview
    # ==================================================================

    def _on_live_preview(self, config_dict: dict):
        """
        設定パネルのパラメータ変更時に呼ばれる。
        再解析は走らせず、既存の _stage1_scores を使って
        閾値ベースのフィルタリングのみ再実行する。
        """
        if not self._stage1_scores:
            return

        quality_enabled = bool(config_dict.get('QUALITY_FILTER_ENABLED', True))
        quality_th = float(config_dict.get('QUALITY_THRESHOLD', 0.50))
        abs_lap_min = float(config_dict.get('QUALITY_ABS_LAPLACIAN_MIN', 35.0))
        lap_th = config_dict.get('LAPLACIAN_THRESHOLD', 100.0)
        blur_th = config_dict.get('MOTION_BLUR_THRESHOLD', 0.3)
        min_interval = config_dict.get('MIN_KEYFRAME_INTERVAL', 5)

        # 簡易フィルタリング
        candidates = []
        last_kf = -min_interval
        for s in self._stage1_scores:
            if quality_enabled:
                q = float(getattr(s, "quality", 0.0))
                lap = float(getattr(s, "sharpness", 0.0))
                passes = bool(q >= quality_th and lap >= abs_lap_min)
            else:
                passes = bool(s.sharpness >= lap_th and s.motion_blur <= blur_th)
            if not passes:
                continue
            if s.frame_index - last_kf >= min_interval:
                candidates.append(s.frame_index)
                last_kf = s.frame_index

        # マーカーだけ更新（スコアは仮に0.5）
        scores = [0.5] * len(candidates)
        self.timeline.set_keyframes(candidates, scores)
        self.video_player.set_keyframe_indices(candidates)

        mode_text = f"quality>={quality_th:.2f}" if quality_enabled else f"lap>={lap_th:.1f}, blur<={blur_th:.2f}"
        self.statusBar().showMessage(f"Live Preview: {len(candidates)} フレームが閾値を通過 ({mode_text})")

    @staticmethod
    def _extract_stationary_ranges(scores: List[FrameScoreData]) -> List[tuple[int, int]]:
        ranges: List[tuple[int, int]] = []
        start_idx: Optional[int] = None
        end_idx: Optional[int] = None
        for score in scores:
            if bool(getattr(score, "is_stationary", False)):
                if start_idx is None:
                    start_idx = int(score.frame_index)
                end_idx = int(score.frame_index)
            elif start_idx is not None and end_idx is not None:
                ranges.append((start_idx, end_idx))
                start_idx = None
                end_idx = None
        if start_idx is not None and end_idx is not None:
            ranges.append((start_idx, end_idx))
        return ranges

    def _get_vo_conf_thresholds(self) -> tuple[float, float]:
        cfg = self.settings_panel.get_selector_dict()
        low = float(np.clip(cfg.get("VO_CONFIDENCE_LOW_THRESHOLD", 0.35), 0.0, 1.0))
        mid = float(np.clip(cfg.get("VO_CONFIDENCE_MID_THRESHOLD", 0.55), low, 1.0))
        return low, mid

    @staticmethod
    def _compute_vo_summary(scores: List[FrameScoreData]) -> Dict[str, object]:
        if not scores:
            return {
                "runtime_enabled": False,
                "runtime_reason": "not_evaluated",
                "attempted": 0,
                "valid": 0,
                "pose_valid": 0,
                "reason_counts": {},
            }
        attempted = 0
        valid = 0
        pose_valid = 0
        reasons = Counter()
        for score in scores:
            attempted += 1 if bool(getattr(score, "vo_attempted", False)) else 0
            valid += 1 if bool(getattr(score, "vo_valid", False)) else 0
            pose_valid += 1 if bool(getattr(score, "vo_pose_valid", False)) else 0
            reasons[str(getattr(score, "vo_status_reason", "unknown"))] += 1
        top_reason = reasons.most_common(1)[0][0] if reasons else "unknown"
        runtime_enabled = any(
            reason not in {"calibration_unavailable", "projection_mode_unsupported", "vo_disabled_by_config"}
            for reason in reasons.keys()
        )
        return {
            "runtime_enabled": bool(runtime_enabled),
            "runtime_reason": str(top_reason),
            "attempted": int(attempted),
            "valid": int(valid),
            "pose_valid": int(pose_valid),
            "reason_counts": dict(reasons),
        }

    # ==================================================================
    # エクスポート
    # ==================================================================

    def export_keyframes(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "ビデオを先に開いてください")
            return

        selected = self.keyframe_list.get_selected_keyframes()
        if not selected:
            # 全キーフレームを対象にする
            selected = list(self.keyframe_list.keyframe_frames)
        if not selected:
            QMessageBox.warning(self, "警告", "エクスポートするキーフレームがありません")
            return

        # エクスポートダイアログを表示
        dlg = ExportDialog(self, num_keyframes=len(selected))
        if not dlg.exec():
            return

        s = dlg.get_settings()
        export_dir = s["output_dir"]
        if not export_dir:
            return

        self._export_worker = ExportWorker(
            self.video_path, selected, export_dir,
            format=s["output_format"],
            jpeg_quality=s["jpeg_quality"],
            prefix=s["prefix"],
            # 360度処理設定
            enable_equirect=s["enable_equirect"],
            equirect_width=s["equirect_width"],
            equirect_height=s["equirect_height"],
            enable_stereo_stitch=s["enable_stereo_stitch"],
            stitching_mode=s["stitching_mode"],
            enable_polar_mask=s["enable_polar_mask"],
            mask_polar_ratio=s["mask_polar_ratio"],
            # Cubemap 出力
            enable_cubemap=s["enable_cubemap"],
            cubemap_face_size=s["cubemap_face_size"],
            # Perspective 出力
            enable_perspective=s["enable_perspective"],
            perspective_fov=s["perspective_fov"],
            perspective_yaw_list=s["perspective_yaw_list"],
            perspective_pitch_list=s["perspective_pitch_list"],
            perspective_size=tuple(s["perspective_size"]),
            # マスク処理設定
            enable_nadir_mask=s["enable_nadir_mask"],
            nadir_mask_radius=s["nadir_mask_radius"],
            enable_equipment_detection=s["enable_equipment_detection"],
            mask_dilation_size=s["mask_dilation_size"],
            enable_fisheye_border_mask=s["enable_fisheye_border_mask"],
            fisheye_mask_radius_ratio=s["fisheye_mask_radius_ratio"],
            fisheye_mask_center_offset_x=s["fisheye_mask_center_offset_x"],
            fisheye_mask_center_offset_y=s["fisheye_mask_center_offset_y"],
            # 対象検出マスク
            enable_target_mask_generation=s["enable_target_mask_generation"],
            target_classes=s["target_classes"],
            yolo_model_path=s["yolo_model_path"],
            sam_model_path=s["sam_model_path"],
            confidence_threshold=s["confidence_threshold"],
            detection_device=s["detection_device"],
            mask_output_dirname=s["mask_output_dirname"],
            mask_add_suffix=s["mask_add_suffix"],
            mask_suffix=s["mask_suffix"],
            mask_output_format=s["mask_output_format"],
            dynamic_mask_use_motion_diff=s["dynamic_mask_use_motion_diff"],
            dynamic_mask_motion_frames=s["dynamic_mask_motion_frames"],
            dynamic_mask_motion_threshold=s["dynamic_mask_motion_threshold"],
            dynamic_mask_dilation_size=s["dynamic_mask_dilation_size"],
            dynamic_mask_use_yolo_sam=s["dynamic_mask_use_yolo_sam"],
            dynamic_mask_target_classes=s["dynamic_mask_target_classes"],
            dynamic_mask_inpaint_enabled=s["dynamic_mask_inpaint_enabled"],
            dynamic_mask_inpaint_module=s["dynamic_mask_inpaint_module"],
            precomputed_analysis_masks={
                idx: self._analysis_masks[idx]
                for idx in selected
                if idx in self._analysis_masks
            },
            use_precomputed_analysis_masks=bool(s.get("enable_dynamic_mask_removal", False)),
            export_runtime_config=s,
        )

        # ステレオ（OSV）対応: 左右ストリームパスを設定
        if self.is_stereo and self.stereo_left_path and self.stereo_right_path:
            self._export_worker.set_stereo_paths(self.stereo_left_path, self.stereo_right_path)
            logger.info(f"エクスポート: ステレオペア出力モード（L/R）")

        self._export_worker.progress.connect(self._on_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_error)

        # 出力内容のサマリをステータスバーに表示
        modes = []
        if s["enable_cubemap"]:
            modes.append(f"Cubemap({s['cubemap_face_size']}px)")
        if s["enable_perspective"]:
            ny = len(s["perspective_yaw_list"])
            np_ = len(s["perspective_pitch_list"])
            modes.append(f"Perspective({ny}×{np_}方向)")
        mode_str = " + ".join(modes) if modes else "元画像のみ"

        self._progress_bar.setVisible(True)
        self.statusBar().showMessage(f"エクスポート中... [{mode_str}]")
        self._export_worker.start()

    def _on_export_finished(self, count: int):
        self._progress_bar.setVisible(False)
        self.statusBar().showMessage(f"エクスポート完了: {count} ファイル")
        QMessageBox.information(self, "完了", f"{count} 個のキーフレームをエクスポートしました")

    # ==================================================================
    # ワーカー管理
    # ==================================================================

    def _stop_workers(self):
        self._stop_analysis_artifact_poll()
        for w in [self._analysis_worker, self._pose_only_worker, self._export_worker]:
            if w and w.isRunning():
                w.stop()
                w.wait(3000)

    def _start_analysis_artifact_poll(self, run_id: str) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            self._stop_analysis_artifact_poll()
            return
        self._analysis_artifact_run_dir = Path.home() / ".360split" / "tmp_runs" / rid
        self._analysis_artifact_mtime = {}
        self._analysis_artifact_payload = {}
        self._analysis_artifact_poll_timer.start()

    def _stop_analysis_artifact_poll(self) -> None:
        if self._analysis_artifact_poll_timer.isActive():
            self._analysis_artifact_poll_timer.stop()
        self._analysis_artifact_run_dir = None
        self._analysis_artifact_mtime = {}

    def _poll_analysis_artifacts(self) -> None:
        run_dir = self._analysis_artifact_run_dir
        if run_dir is None:
            return
        artifact_files = [
            run_dir / "stage1_records.jsonl",
            run_dir / "stage2_records.jsonl",
            run_dir / "stage2_colmap_preview.jsonl",
            run_dir / "stage3_diagnostics.json",
            run_dir / "analysis_summary.json",
        ]
        changed = False
        for path in artifact_files:
            if not path.exists():
                continue
            try:
                mtime_ns = int(path.stat().st_mtime_ns)
            except OSError:
                continue
            key = str(path)
            if self._analysis_artifact_mtime.get(key) != mtime_ns:
                self._analysis_artifact_mtime[key] = mtime_ns
                changed = True
        if not changed:
            return
        try:
            payload = load_analysis_artifacts(run_dir, pose_summary=self._last_pose_summary)
        except Exception:
            # 途中書きファイル等は次回ポーリングで再試行する。
            return
        self._analysis_artifact_payload = dict(payload)
        self.analysis_dashboard.update_stage_artifacts(self._analysis_artifact_payload)

    @staticmethod
    def _canonical_setting_key(key: str) -> str:
        k = str(key or "").strip()
        if k == "ssim_change_threshold":
            return "ssim_threshold"
        return k

    def _on_apply_dashboard_settings(self, payload: dict) -> None:
        if not isinstance(payload, dict) or not payload:
            return
        settings_dir = Path.home() / ".360split"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_file = settings_dir / "settings.json"

        current: Dict[str, Any] = {}
        if settings_file.exists():
            try:
                with settings_file.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    current = dict(loaded)
            except Exception:
                current = {}

        updates: Dict[str, Any] = {}
        for raw_key, value in payload.items():
            key = self._canonical_setting_key(str(raw_key))
            if not key:
                continue
            if current.get(key) != value:
                updates[key] = value

        if not updates:
            self.statusBar().showMessage("更新対象の差分はありません")
            return

        merged = dict(current)
        merged.update(updates)
        merged = SettingsDialog._strip_vo_legacy_keys(merged)

        try:
            with settings_file.open("w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("ダッシュボード設定保存エラー")
            QMessageBox.critical(self, "エラー", f"設定保存に失敗しました:\n{e}")
            return

        self.settings_panel.reload_from_file()
        self.analysis_dashboard.set_current_settings(self.settings_panel.get_selector_dict())
        self.statusBar().showMessage("設定適用済み。再解析を実行してください")

    def closeEvent(self, event):
        self._stop_workers()
        self.settings_panel.save_settings()
        super().closeEvent(event)

    # ==================================================================
    # スタイルシート
    # ==================================================================

    def _apply_stylesheet(self):
        self.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; color: #ffffff; }

        QMenuBar { background-color: #2d2d2d; color: #fff; border-bottom: 1px solid #3d3d3d; }
        QMenuBar::item:selected { background-color: #3d3d3d; }
        QMenu { background-color: #2d2d2d; color: #fff; border: 1px solid #3d3d3d; }
        QMenu::item:selected { background-color: #404080; }

        QToolBar { background-color: #2d2d2d; border-bottom: 1px solid #3d3d3d; spacing: 4px; padding: 4px; }

        QStatusBar { background: #2d2d2d; color: #fff; border-top: 1px solid #3d3d3d; }

        QDockWidget { color: #fff; }
        QDockWidget::title { background: #2d2d2d; padding: 6px; }

        QTabBar::tab { background: #3d3d3d; color: #fff; padding: 8px 16px; border: none; }
        QTabBar::tab:selected { background: #505080; border-bottom: 2px solid #5080d0; }
        QTabWidget::pane { border: 1px solid #3d3d3d; }

        QLabel { color: #fff; }

        QPushButton {
            background-color: #404080; color: #fff;
            border: 1px solid #5080d0; border-radius: 3px;
            padding: 4px 12px; font-weight: bold;
        }
        QPushButton:hover { background-color: #5080d0; }
        QPushButton:pressed { background-color: #3d6ead; }

        QSlider::groove:horizontal { background: #3d3d3d; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal {
            background: #5080d0; border: 1px solid #6090e0;
            width: 14px; margin: -4px 0; border-radius: 7px;
        }

        QComboBox { background: #3d3d3d; color: #fff; border: 1px solid #404040; border-radius: 3px; padding: 4px; }
        QSpinBox, QDoubleSpinBox { background: #3d3d3d; color: #fff; border: 1px solid #404040; border-radius: 3px; padding: 4px; }

        QCheckBox { color: #fff; spacing: 5px; }
        QCheckBox::indicator { width: 16px; height: 16px; }
        QCheckBox::indicator:unchecked { background: #3d3d3d; border: 1px solid #505050; }
        QCheckBox::indicator:checked { background: #5080d0; border: 1px solid #5080d0; }

        QScrollBar:vertical { background: #2d2d2d; width: 10px; }
        QScrollBar::handle:vertical { background: #505050; border-radius: 5px; min-height: 20px; }
        QScrollBar:horizontal { background: #2d2d2d; height: 10px; }
        QScrollBar::handle:horizontal { background: #505050; border-radius: 5px; min-width: 20px; }

        QProgressBar { background: #3d3d3d; color: #fff; border: 1px solid #404040; border-radius: 3px; text-align: center; }
        QProgressBar::chunk { background: #5080d0; }

        QGroupBox { color: #ddd; border: 1px solid #3d3d3d; border-radius: 4px; margin-top: 8px; padding-top: 16px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)
