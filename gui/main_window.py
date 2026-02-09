"""
メインウィンドウ - 360Split GUI
PySide6を使用したメインアプリケーションウィンドウ
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QMenuBar, QToolBar, QStatusBar, QMessageBox,
    QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QAction

from gui.video_player import VideoPlayerWidget
from gui.timeline_widget import TimelineWidget
from gui.keyframe_panel import KeyframePanel
from gui.settings_dialog import SettingsDialog

logger = logging.getLogger('360split')


class AnalysisWorker(QThread):
    """
    キーフレーム分析用ワーカースレッド

    バックグラウンドでキーフレーム選択分析を実行し、
    進捗状況とスコア情報をシグナルで通知する。

    Signals:
    --------
    progress : Signal(int)
        分析進捗率（0-100）
    keyframes_found : Signal(list, list)
        検出されたキーフレーム情報（フレーム番号リスト、スコアリスト）
    quality_data : Signal(list)
        全フレームの品質スコアリスト
    finished : Signal()
        分析完了シグナル
    error : Signal(str)
        エラーメッセージ
    """

    progress = Signal(int)
    keyframes_found = Signal(list, list)
    quality_data = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, video_path: str, quality_evaluator, keyframe_selector=None):
        """
        ワーカーの初期化

        Parameters:
        -----------
        video_path : str
            分析対象のビデオファイルパス
        quality_evaluator : QualityEvaluator
            品質評価エンジン
        keyframe_selector : KeyframeSelector, optional
            キーフレーム選択エンジン（Noneの場合は簡易版を使用）
        """
        super().__init__()
        self.video_path = video_path
        self.quality_evaluator = quality_evaluator
        self.keyframe_selector = keyframe_selector
        self._is_running = True

    def run(self):
        """
        バックグラウンド分析を実行
        """
        try:
            import cv2
            import numpy as np
            from config import SOFTMAX_BETA

            # ビデオ読み込み
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("ビデオファイルを開けません")
                return

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            quality_scores = []
            keyframe_frames = []
            keyframe_scores = []
            prev_frame = None
            ssim_history = []

            for frame_idx in range(frame_count):
                if not self._is_running:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # 品質スコア計算
                quality_dict = self.quality_evaluator.evaluate(frame, SOFTMAX_BETA)
                combined_score = (
                    quality_dict.get('sharpness', 0) * 0.30 +
                    quality_dict.get('exposure', 0) * 0.15 +
                    quality_dict.get('softmax_depth', 0) * 0.30 +
                    (1.0 - quality_dict.get('motion_blur', 0)) * 0.25
                )
                quality_scores.append(combined_score)

                # シンプルなキーフレーム選択：スコアが閾値を超えた場合
                if combined_score > 0.65:
                    if not keyframe_frames or frame_idx - keyframe_frames[-1] >= 5:
                        keyframe_frames.append(frame_idx)
                        keyframe_scores.append(combined_score)

                prev_frame = frame

                # 進捗を報告
                progress = int((frame_idx + 1) / frame_count * 100)
                self.progress.emit(progress)

            cap.release()

            # 結果を送信
            self.quality_data.emit(quality_scores)
            self.keyframes_found.emit(keyframe_frames, keyframe_scores)
            self.finished.emit()

        except Exception as e:
            logger.exception("分析エラー")
            self.error.emit(f"分析エラー: {str(e)}")

    def stop(self):
        """
        分析を停止
        """
        self._is_running = False


class MainWindow(QMainWindow):
    """
    メインアプリケーションウィンドウ

    360Split の主要UIコンポーネントを統合し、
    ビデオプレーヤー、タイムライン、キーフレームパネルを
    管理するメインウィンドウ。

    Features:
    ---------
    - ダークテーマUIデザイン
    - メニューバー（ファイル、編集、処理、表示、ヘルプ）
    - ツールバー（主要機能へのアクセス）
    - ビデオプレーヤーウィジェット
    - カスタムタイムラインウィジェット
    - キーフレームパネル
    - ステータスバー
    - バックグラウンド分析スレッド
    """

    def __init__(self):
        """
        メインウィンドウの初期化
        """
        super().__init__()
        self.setWindowTitle("360Split - キーフレーム抽出ツール")
        self.setGeometry(100, 100, 1600, 900)

        # ロジックモジュール
        self.video_path: Optional[str] = None
        self.current_video_metadata = None
        self.analysis_worker: Optional[AnalysisWorker] = None

        # UI要素の初期化
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_stylesheet()
        self._setup_connections()

        # ステータス表示用
        self.statusBar().showMessage("準備完了")

        logger.info("メインウィンドウの初期化完了")

    def _setup_ui(self):
        """
        UIレイアウトの構築
        """
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # メイン分割レイアウト（ビデオ＋キーフレーム）
        main_splitter = QSplitter(Qt.Horizontal)

        # 左側：ビデオプレーヤー
        self.video_player = VideoPlayerWidget()
        main_splitter.addWidget(self.video_player)

        # 右側：キーフレームパネル
        self.keyframe_panel = KeyframePanel()
        main_splitter.addWidget(self.keyframe_panel)

        # 初期分割比率
        main_splitter.setSizes([1100, 500])

        main_layout.addWidget(main_splitter, stretch=1)

        # 下部：タイムラインウィジェット
        self.timeline = TimelineWidget()
        main_layout.addWidget(self.timeline, stretch=0)

        # ステータスバーの設定
        self.status_label = self.statusBar()

    def _setup_menu(self):
        """
        メニューバーの構築
        """
        menubar = self.menuBar()

        # === ファイル(File) メニュー ===
        file_menu = menubar.addMenu("ファイル(File)")

        open_action = QAction("ビデオを開く(&O)", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("キーフレームをエクスポート(&E)", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_action.triggered.connect(self.export_keyframes)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("終了(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # === 編集(Edit) メニュー ===
        edit_menu = menubar.addMenu("編集(Edit)")

        undo_action = QAction("元に戻す(&U)", self)
        undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("やり直す(&R)", self)
        redo_action.setShortcut(QKeySequence.Redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction("すべて選択(&A)", self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        select_all_action.triggered.connect(self.keyframe_panel.select_all)
        edit_menu.addAction(select_all_action)

        deselect_all_action = QAction("選択解除(&D)", self)
        deselect_all_action.triggered.connect(self.keyframe_panel.deselect_all)
        edit_menu.addAction(deselect_all_action)

        edit_menu.addSeparator()

        settings_action = QAction("設定(&S)...", self)
        settings_action.setShortcut(QKeySequence.Preferences)
        settings_action.triggered.connect(self._open_settings)
        edit_menu.addAction(settings_action)

        # === 処理(Processing) メニュー ===
        process_menu = menubar.addMenu("処理(Processing)")

        analyze_action = QAction("キーフレーム分析を実行(&R)", self)
        analyze_action.setShortcut(QKeySequence("Ctrl+R"))
        analyze_action.triggered.connect(self.run_keyframe_selection)
        process_menu.addAction(analyze_action)

        process_menu.addSeparator()

        # === 表示(View) メニュー ===
        view_menu = menubar.addMenu("表示(View)")

        grid_action = QAction("グリッドオーバーレイを表示(&G)", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(False)
        grid_action.triggered.connect(
            lambda checked: self.video_player.set_grid_overlay(checked)
        )
        view_menu.addAction(grid_action)

        # === ヘルプ(Help) メニュー ===
        help_menu = menubar.addMenu("ヘルプ(Help)")

        about_action = QAction("360Split について(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """
        ツールバーの構築
        """
        toolbar = self.addToolBar("メインツールバー")
        toolbar.setIconSize(QSize(24, 24))

        # ビデオを開く
        open_action = QAction("ビデオを開く", self)
        open_action.triggered.connect(self.open_video)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # 分析実行
        run_action = QAction("分析を実行", self)
        run_action.triggered.connect(self.run_keyframe_selection)
        toolbar.addAction(run_action)

        toolbar.addSeparator()

        # エクスポート
        export_action = QAction("キーフレームをエクスポート", self)
        export_action.triggered.connect(self.export_keyframes)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # 設定
        settings_action = QAction("設定", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)

    def _setup_stylesheet(self):
        """
        ダークテーマスタイルシートの適用
        """
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }

        QMenuBar {
            background-color: #2d2d2d;
            color: #ffffff;
            border-bottom: 1px solid #3d3d3d;
        }

        QMenuBar::item:selected {
            background-color: #3d3d3d;
        }

        QMenu {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3d3d3d;
        }

        QMenu::item:selected {
            background-color: #404080;
        }

        QToolBar {
            background-color: #2d2d2d;
            border-bottom: 1px solid #3d3d3d;
            spacing: 5px;
            padding: 5px;
        }

        QToolBar::separator {
            background-color: #3d3d3d;
            width: 1px;
            margin: 5px 0;
        }

        QStatusBar {
            background-color: #2d2d2d;
            color: #ffffff;
            border-top: 1px solid #3d3d3d;
        }

        QLabel {
            color: #ffffff;
        }

        QLineEdit {
            background-color: #3d3d3d;
            color: #ffffff;
            border: 1px solid #404040;
            border-radius: 3px;
            padding: 5px;
        }

        QLineEdit:focus {
            border: 1px solid #5080d0;
        }

        QPushButton {
            background-color: #404080;
            color: #ffffff;
            border: 1px solid #5080d0;
            border-radius: 3px;
            padding: 5px 15px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #5080d0;
        }

        QPushButton:pressed {
            background-color: #3d6ead;
        }

        QScrollBar:vertical {
            background-color: #2d2d2d;
            width: 12px;
            border: none;
        }

        QScrollBar::handle:vertical {
            background-color: #505050;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #606060;
        }

        QScrollBar:horizontal {
            background-color: #2d2d2d;
            height: 12px;
            border: none;
        }

        QScrollBar::handle:horizontal {
            background-color: #505050;
            border-radius: 6px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #606060;
        }

        QComboBox {
            background-color: #3d3d3d;
            color: #ffffff;
            border: 1px solid #404040;
            border-radius: 3px;
            padding: 5px;
        }

        QComboBox:focus {
            border: 1px solid #5080d0;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QSlider::groove:horizontal {
            background-color: #3d3d3d;
            height: 6px;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background-color: #5080d0;
            border: 1px solid #6090e0;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }

        QSlider::handle:horizontal:hover {
            background-color: #6090e0;
        }

        QTabBar::tab {
            background-color: #3d3d3d;
            color: #ffffff;
            padding: 8px 20px;
            border: none;
        }

        QTabBar::tab:selected {
            background-color: #505080;
            border-bottom: 2px solid #5080d0;
        }

        QTabWidget::pane {
            border: 1px solid #3d3d3d;
        }

        QDialog {
            background-color: #1e1e1e;
            color: #ffffff;
        }

        QProgressDialog {
            background-color: #2d2d2d;
        }

        QProgressBar {
            background-color: #3d3d3d;
            color: #ffffff;
            border: 1px solid #404040;
            border-radius: 3px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #5080d0;
        }

        QCheckBox {
            color: #ffffff;
            spacing: 5px;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }

        QCheckBox::indicator:unchecked {
            background-color: #3d3d3d;
            border: 1px solid #505050;
        }

        QCheckBox::indicator:checked {
            background-color: #5080d0;
            border: 1px solid #5080d0;
        }

        QSpinBox, QDoubleSpinBox {
            background-color: #3d3d3d;
            color: #ffffff;
            border: 1px solid #404040;
            border-radius: 3px;
            padding: 5px;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def _setup_connections(self):
        """
        UIコンポーネント間のシグナル/スロット接続
        """
        # ビデオプレーヤーのシグナル
        self.video_player.frame_changed.connect(self._on_frame_changed)
        self.video_player.playback_speed_changed.connect(self._on_speed_changed)

        # キーフレームパネルのシグナル
        self.keyframe_panel.keyframe_selected.connect(self._on_keyframe_selected)
        self.keyframe_panel.keyframe_deleted.connect(self._on_keyframe_deleted)

        # タイムラインのシグナル
        self.timeline.positionChanged.connect(self._on_timeline_position_changed)
        self.timeline.keyframeClicked.connect(self._on_timeline_keyframe_clicked)

    def open_video(self):
        """
        ビデオファイルを開くダイアログを表示し、ビデオを読み込む

        サポート形式: mp4, mov, avi, mkv
        """
        file_filter = "ビデオファイル (*.mp4 *.mov *.avi *.mkv);;すべてのファイル (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "ビデオファイルを開く",
            "",
            file_filter
        )

        if not file_path:
            return

        try:
            # ビデオを読み込む
            self.video_path = file_path
            metadata = self.video_player.load_video(file_path)
            self.current_video_metadata = metadata

            # タイムラインを初期化
            self.timeline.set_duration(metadata.frame_count, metadata.fps)

            # キーフレームパネルをクリア
            self.keyframe_panel.clear()

            # ステータス更新
            self.update_status(
                f"ビデオ読み込み完了: {Path(file_path).name} "
                f"({metadata.width}x{metadata.height}, {metadata.fps:.1f}fps, "
                f"{metadata.frame_count}フレーム)"
            )

            logger.info(f"ビデオ読み込み: {file_path}")

        except Exception as e:
            logger.exception(f"ビデオ読み込みエラー: {file_path}")
            QMessageBox.critical(
                self,
                "エラー",
                f"ビデオファイルの読み込みに失敗しました:\n{str(e)}"
            )

    def run_keyframe_selection(self):
        """
        キーフレーム選択分析をバックグラウンドスレッドで実行

        分析中はプログレスダイアログを表示し、
        完了後にタイムラインとキーフレームパネルを更新する。
        """
        if not self.video_path:
            QMessageBox.warning(
                self,
                "警告",
                "ビデオファイルを先に開いてください"
            )
            return

        # 品質評価エンジンの初期化
        from core.quality_evaluator import QualityEvaluator
        evaluator = QualityEvaluator()

        # 分析ワーカーの作成
        self.analysis_worker = AnalysisWorker(self.video_path, evaluator)
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.keyframes_found.connect(self._on_keyframes_found)
        self.analysis_worker.quality_data.connect(self._on_quality_data_received)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.error.connect(self._on_analysis_error)

        # プログレスダイアログ
        self.progress_dialog = QProgressDialog(
            "キーフレーム分析を実行中...",
            "キャンセル",
            0,
            100,
            self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setStyleSheet(self.styleSheet())
        self.progress_dialog.canceled.connect(self.analysis_worker.stop)

        # ワーカーを開始
        self.analysis_worker.start()
        self.update_status("キーフレーム分析を実行中...")

    def export_keyframes(self):
        """
        選択されたキーフレームを指定ディレクトリにエクスポート

        PNG形式で保存。設定で形式を変更可能。
        """
        if not self.video_path:
            QMessageBox.warning(
                self,
                "警告",
                "ビデオファイルを先に開いてください"
            )
            return

        keyframes = self.keyframe_panel.get_selected_keyframes()
        if not keyframes:
            QMessageBox.warning(
                self,
                "警告",
                "エクスポートするキーフレームを選択してください"
            )
            return

        # エクスポート先ディレクトリの選択
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "エクスポート先ディレクトリを選択"
        )

        if not export_dir:
            return

        try:
            import cv2
            from pathlib import Path

            export_path = Path(export_dir)
            export_path.mkdir(exist_ok=True)

            # ビデオから各キーフレームを抽出
            cap = cv2.VideoCapture(self.video_path)
            exported_count = 0

            for frame_idx in keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    output_file = export_path / f"keyframe_{frame_idx:05d}.png"
                    cv2.imwrite(str(output_file), frame)
                    exported_count += 1

            cap.release()

            self.update_status(
                f"キーフレーム {exported_count} 個をエクスポートしました: {export_dir}"
            )
            QMessageBox.information(
                self,
                "完了",
                f"{exported_count} 個のキーフレームをエクスポートしました"
            )

            logger.info(f"キーフレームをエクスポート: {exported_count}個 -> {export_dir}")

        except Exception as e:
            logger.exception("エクスポートエラー")
            QMessageBox.critical(
                self,
                "エラー",
                f"エクスポート中にエラーが発生しました:\n{str(e)}"
            )

    def update_status(self, message: str):
        """
        ステータスバーを更新

        Parameters:
        -----------
        message : str
            表示するメッセージ
        """
        self.status_label.showMessage(message)
        logger.info(f"ステータス: {message}")

    def _open_settings(self):
        """
        設定ダイアログを開く
        """
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()

    def _show_about(self):
        """
        アプリケーション情報ダイアログを表示
        """
        QMessageBox.information(
            self,
            "360Split について",
            "360Split v1.0\n\n"
            "360度動画キーフレーム抽出ツール\n"
            "photogrammetry/3GS再構成用\n\n"
            "© 2024 360Split Project"
        )

    # === シグナルハンドラー ===

    def _on_frame_changed(self, frame_idx: int):
        """
        ビデオプレーヤーのフレーム変更時のコールバック
        """
        self.timeline.set_position(frame_idx)

    def _on_speed_changed(self, speed: float):
        """
        再生速度変更時のコールバック
        """
        self.update_status(f"再生速度: {speed}x")

    def _on_keyframe_selected(self, frame_idx: int):
        """
        キーフレームパネルで選択されたキーフレームのコールバック
        """
        self.video_player.seek_to_frame(frame_idx)

    def _on_keyframe_deleted(self, frame_idx: int):
        """
        キーフレームが削除されたときのコールバック
        """
        self.timeline.remove_keyframe(frame_idx)

    def _on_timeline_position_changed(self, frame_idx: int):
        """
        タイムラインで位置が変更されたときのコールバック
        """
        self.video_player.seek_to_frame(frame_idx)

    def _on_timeline_keyframe_clicked(self, frame_idx: int):
        """
        タイムラインでキーフレームがクリックされたときのコールバック
        """
        self.video_player.seek_to_frame(frame_idx)

    def _on_analysis_progress(self, progress: int):
        """
        分析進捗時のコールバック
        """
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(progress)

    def _on_keyframes_found(self, frames: list, scores: list):
        """
        キーフレームが検出されたときのコールバック
        """
        self.timeline.set_keyframes(frames, scores)
        self.keyframe_panel.set_keyframes(frames, scores)

    def _on_quality_data_received(self, quality_scores: list):
        """
        品質スコアデータ受信時のコールバック
        """
        self.timeline.set_quality_data(quality_scores)

    def _on_analysis_finished(self):
        """
        分析完了時のコールバック
        """
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        self.update_status("キーフレーム分析が完了しました")
        QMessageBox.information(
            self,
            "完了",
            "キーフレーム分析が完了しました"
        )

    def _on_analysis_error(self, error_msg: str):
        """
        分析エラー時のコールバック
        """
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        self.update_status(f"エラー: {error_msg}")
        QMessageBox.critical(
            self,
            "エラー",
            f"分析中にエラーが発生しました:\n{error_msg}"
        )

    def closeEvent(self, event):
        """
        アプリケーション終了時の処理
        """
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.analysis_worker.wait()

        super().closeEvent(event)
