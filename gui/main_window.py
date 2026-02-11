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
  - Stage 1 / Stage 2 分離解析
  - Live Preview (パラメータ変更 → 判定再実行)
"""

import json
from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QTabWidget,
    QFileDialog, QMenuBar, QToolBar, QStatusBar,
    QMessageBox, QProgressBar, QLabel
)
from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QKeySequence, QAction, QDragEnterEvent, QDropEvent

from gui.video_player import VideoPlayerWidget
from gui.timeline_widget import TimelineWidget
from gui.settings_panel import SettingsPanel
from gui.settings_dialog import SettingsDialog
from gui.keyframe_list import KeyframeListWidget
from gui.export_dialog import ExportDialog
from gui.workers import Stage1Worker, Stage2Worker, FullAnalysisWorker, ExportWorker, FrameScoreData

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
        self._stage1_worker: Optional[Stage1Worker] = None
        self._stage2_worker: Optional[Stage2Worker] = None
        self._full_worker: Optional[FullAnalysisWorker] = None
        self._export_worker: Optional[ExportWorker] = None

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

        # ビデオプレーヤー
        self.video_player = VideoPlayerWidget()
        layout.addWidget(self.video_player, stretch=1)

        # タイムライン
        self.timeline = TimelineWidget()
        layout.addWidget(self.timeline, stretch=0)

        # ステータスバーにプログレスバーを追加
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(250)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self.statusBar().addPermanentWidget(self._progress_label)

    def _setup_dock(self):
        """右側ドック: 設定パネル + キーフレーム一覧 (タブ切り替え)"""
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
        self.keyframe_list = KeyframeListWidget()
        tab_widget.addTab(self.keyframe_list, "📋 キーフレーム")

        dock.setWidget(tab_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _setup_menu(self):
        menubar = self.menuBar()

        # ファイル(F)
        file_menu = menubar.addMenu("ファイル(&F)")

        open_action = QAction("開く(&O)...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("キーフレームをエクスポート(&E)...", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_action.triggered.connect(self.export_keyframes)
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

        # 解析(A)
        analysis_menu = menubar.addMenu("解析(&A)")

        stage1_action = QAction("簡易解析 (Stage 1)(&1)", self)
        stage1_action.setShortcut(QKeySequence("Ctrl+1"))
        stage1_action.triggered.connect(self._run_stage1)
        analysis_menu.addAction(stage1_action)

        stage2_action = QAction("詳細解析 (Stage 2)(&2)", self)
        stage2_action.setShortcut(QKeySequence("Ctrl+2"))
        stage2_action.triggered.connect(self._run_stage2)
        analysis_menu.addAction(stage2_action)

        analysis_menu.addSeparator()

        full_action = QAction("フル解析 (Stage 1+2)(&R)", self)
        full_action.setShortcut(QKeySequence("Ctrl+R"))
        full_action.triggered.connect(self._run_full_analysis)
        analysis_menu.addAction(full_action)

    def _setup_toolbar(self):
        tb = self.addToolBar("メイン")
        tb.setIconSize(QSize(20, 20))

        tb.addAction("📂 開く", self.open_video)
        tb.addSeparator()
        tb.addAction("⚡ 簡易解析", self._run_stage1)
        tb.addAction("🔬 詳細解析", self._run_stage2)
        tb.addAction("🚀 フル解析", self._run_full_analysis)
        tb.addSeparator()
        tb.addAction("💾 エクスポート", self.export_keyframes)

    def _setup_connections(self):
        """全シグナル/スロットを接続"""
        # ビデオプレーヤー → タイムライン同期
        self.video_player.frame_changed.connect(self.timeline.set_position)
        self.video_player.keyframe_marked.connect(self._on_manual_mark)

        # タイムライン → ビデオプレーヤー同期
        self.timeline.positionChanged.connect(self.video_player.seek_to_frame)
        self.timeline.keyframeClicked.connect(self.video_player.seek_to_frame)

        # キーフレーム一覧 → ビデオプレーヤー
        self.keyframe_list.keyframe_selected.connect(self.video_player.seek_to_frame)
        self.keyframe_list.keyframe_deleted.connect(self.timeline.remove_keyframe)

        # 設定パネル → Live Preview
        self.settings_panel.setting_changed.connect(self._on_live_preview)
        self.settings_panel.run_stage2_requested.connect(self._run_stage2)

    # ==================================================================
    # ドラッグ＆ドロップ
    # ==================================================================

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                self._load_video(path)
                return

    # ==================================================================
    # ビデオ読み込み
    # ==================================================================

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ビデオファイルを開く", "",
            "ビデオ (*.mp4 *.mov *.avi *.mkv *.webm);;すべて (*)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        try:
            self.video_path = path
            metadata = self.video_player.load_video(path)
            self.timeline.set_duration(metadata.frame_count, metadata.fps)
            self.keyframe_list.set_video_path(path)
            self.keyframe_list.clear()
            self._stage1_scores.clear()

            self.statusBar().showMessage(
                f"読み込み完了: {Path(path).name}  "
                f"({metadata.width}×{metadata.height}, "
                f"{metadata.fps:.1f}fps, {metadata.frame_count}フレーム)"
            )
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
            logger.info("設定パネルを再読み込みしました")

    # ==================================================================
    # Stage 1: 簡易解析
    # ==================================================================

    def _run_stage1(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "ビデオを先に開いてください")
            return

        self._stop_workers()
        self._stage1_scores.clear()

        config = self.settings_panel.get_selector_dict()
        self._stage1_worker = Stage1Worker(self.video_path, config=config)
        self._stage1_worker.progress.connect(self._on_progress)
        self._stage1_worker.frame_scores.connect(self._on_stage1_batch)
        self._stage1_worker.finished_scores.connect(self._on_stage1_finished)
        self._stage1_worker.error.connect(self._on_error)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self.statusBar().showMessage("Stage 1: 品質スキャン中...")
        self._stage1_worker.start()

    def _on_stage1_batch(self, batch: list):
        """Stage 1 バッチ結果をプログレッシブにグラフ追加"""
        self._stage1_scores.extend(batch)
        norm_factor = 1000.0  # NormalizationConfig.SHARPNESS_NORM_FACTOR
        indices = [s.frame_index for s in batch]
        sharpness = [min(s.sharpness / norm_factor, 1.0) for s in batch]
        self.timeline.append_score_batch(indices, sharpness)

    def _on_stage1_finished(self, all_scores: list):
        """Stage 1 完了"""
        self._stage1_scores = all_scores
        self._progress_bar.setVisible(False)

        # 全データでグラフを更新
        norm_factor = 1000.0
        indices = [s.frame_index for s in all_scores]
        sharpness = [min(s.sharpness / norm_factor, 1.0) for s in all_scores]
        self.timeline.set_score_data(indices, sharpness)

        self.statusBar().showMessage(
            f"Stage 1 完了: {len(all_scores)} フレームをスキャン。"
            "「詳細解析」で GRIC/SSIM を計算できます。"
        )

    # ==================================================================
    # Stage 2: 詳細解析
    # ==================================================================

    def _run_stage2(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "ビデオを先に開いてください")
            return

        self._stop_workers()

        config = self.settings_panel.get_selector_dict()
        self._stage2_worker = Stage2Worker(
            self.video_path, self._stage1_scores, config=config
        )
        self._stage2_worker.progress.connect(self._on_progress)
        self._stage2_worker.keyframes_found.connect(self._on_keyframes_found)
        self._stage2_worker.frame_scores_updated.connect(self._on_scores_updated)
        self._stage2_worker.finished.connect(self._on_stage2_finished)
        self._stage2_worker.error.connect(self._on_error)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self.statusBar().showMessage("Stage 2: 精密評価中...")
        self._stage2_worker.start()

    def _on_scores_updated(self, updated: list):
        """Stage 2 でGRIC/SSIM付きスコア更新"""
        self._stage1_scores = updated
        norm_factor = 1000.0
        indices = [s.frame_index for s in updated]
        sharpness = [min(s.sharpness / norm_factor, 1.0) for s in updated]
        gric = [s.gric for s in updated]
        ssim_change = [1.0 - s.ssim for s in updated]

        # GRICデータがあるか確認
        has_gric = any(g > 0 for g in gric)
        has_ssim = any(sc > 0 for sc in ssim_change)

        self.timeline.set_score_data(
            indices, sharpness,
            gric=gric if has_gric else None,
            ssim_change=ssim_change if has_ssim else None
        )

    def _on_stage2_finished(self):
        self._progress_bar.setVisible(False)
        n = len(self.keyframe_list.keyframe_frames)
        self.statusBar().showMessage(f"Stage 2 完了: {n} キーフレーム検出")

    # ==================================================================
    # フル解析 (Stage 1 + 2)
    # ==================================================================

    def _run_full_analysis(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "ビデオを先に開いてください")
            return

        self._stop_workers()
        self._stage1_scores.clear()

        config = self.settings_panel.get_selector_dict()
        self._full_worker = FullAnalysisWorker(self.video_path, config=config)
        self._full_worker.progress.connect(self._on_progress)
        self._full_worker.stage1_batch.connect(self._on_stage1_batch)
        self._full_worker.stage1_finished.connect(self._on_stage1_finished)
        self._full_worker.keyframes_found.connect(self._on_keyframes_found)
        self._full_worker.finished.connect(self._on_full_finished)
        self._full_worker.error.connect(self._on_error)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self.statusBar().showMessage("フル解析開始 (Stage 1 + 2)...")
        self._full_worker.start()

    def _on_full_finished(self):
        self._progress_bar.setVisible(False)
        n = len(self.keyframe_list.keyframe_frames)
        self.statusBar().showMessage(f"解析完了: {n} キーフレーム検出")
        QMessageBox.information(self, "完了", f"解析完了: {n} キーフレームを検出しました")

    # ==================================================================
    # 共通コールバック
    # ==================================================================

    def _on_progress(self, current: int, total: int, message: str = ""):
        pct = int(current / max(total, 1) * 100)
        self._progress_bar.setValue(pct)
        self._progress_label.setText(message)

    def _on_keyframes_found(self, keyframes):
        """キーフレーム検出結果を全ウィジェットに反映"""
        frames = [kf.frame_index for kf in keyframes]
        scores = [kf.combined_score for kf in keyframes]

        self.timeline.set_keyframes(frames, scores)
        self.keyframe_list.set_keyframes(frames, scores)
        self.video_player.set_keyframe_indices(frames)

    def _on_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self.statusBar().showMessage(f"エラー: {msg}")
        QMessageBox.critical(self, "エラー", msg)

    def _on_manual_mark(self, frame_idx: int):
        """手動キーフレームマーク"""
        if frame_idx not in self.keyframe_list.keyframe_frames:
            self.keyframe_list.keyframe_frames.append(frame_idx)
            self.keyframe_list.keyframe_scores.append(0.5)
            self.keyframe_list._load_thumbnails()
            self.keyframe_list._update_display()
            self.timeline.set_keyframes(
                self.keyframe_list.keyframe_frames,
                self.keyframe_list.keyframe_scores
            )
            self.video_player.set_keyframe_indices(self.keyframe_list.keyframe_frames)

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

        lap_th = config_dict.get('LAPLACIAN_THRESHOLD', 100.0)
        blur_th = config_dict.get('MOTION_BLUR_THRESHOLD', 0.3)
        min_interval = config_dict.get('MIN_KEYFRAME_INTERVAL', 5)

        # 簡易フィルタリング
        candidates = []
        last_kf = -min_interval
        for s in self._stage1_scores:
            if s.sharpness >= lap_th and s.motion_blur <= blur_th:
                if s.frame_index - last_kf >= min_interval:
                    candidates.append(s.frame_index)
                    last_kf = s.frame_index

        # マーカーだけ更新（スコアは仮に0.5）
        scores = [0.5] * len(candidates)
        self.timeline.set_keyframes(candidates, scores)
        self.video_player.set_keyframe_indices(candidates)

        self.statusBar().showMessage(
            f"Live Preview: {len(candidates)} フレームが閾値を通過"
        )

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
            mask_dilation_size=s["mask_dilation_size"]
        )
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
        for w in [self._stage1_worker, self._stage2_worker, self._full_worker, self._export_worker]:
            if w and w.isRunning():
                w.stop()
                w.wait(3000)

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
