"""
メインウィンドウ - 360Split GUI
PySide6を使用したメインアプリケーションウィンドウ
"""

import sys
import json
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
    キーフレーム分析用ワーカースレッド（最適化版）

    KeyframeSelectorの2段階パイプラインを利用してバックグラウンドで
    キーフレーム選択分析を実行し、進捗状況をシグナルで通知する。

    最適化:
    - KeyframeSelectorの2段階パイプラインを活用
    - Stage 1で60-70%のフレームを高速フィルタ
    - Stage 2は候補フレームのみ精密評価

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

    def __init__(self, video_path: str, config: dict = None):
        """
        ワーカーの初期化

        Parameters:
        -----------
        video_path : str
            分析対象のビデオファイルパス
        config : dict, optional
            KeyframeSelectorに渡す設定辞書
        """
        super().__init__()
        self.video_path = video_path
        self.config = config
        self._is_running = True

    def run(self):
        """
        バックグラウンド分析を実行（最適化版KeyframeSelector使用）
        """
        try:
            from core.video_loader import VideoLoader
            from core.keyframe_selector import KeyframeSelector
            from core.accelerator import get_accelerator

            accel = get_accelerator()
            logger.info(f"分析開始 - デバイス: {accel.device_name}")

            # ビデオ読み込み
            loader = VideoLoader()
            loader.load(self.video_path)
            meta = loader.get_metadata()

            # KeyframeSelectorで2段階パイプライン実行（GUI設定を反映）
            selector = KeyframeSelector(config=self.config)

            def progress_callback(current, total, message=""):
                if not self._is_running:
                    return
                pct = int(current / total * 100) if total > 0 else 0
                self.progress.emit(pct)

            keyframes = selector.select_keyframes(
                loader,
                progress_callback=progress_callback
            )

            if not self._is_running:
                loader.close()
                return

            # 結果を分離
            keyframe_frames = [kf.frame_index for kf in keyframes]
            keyframe_scores = [kf.combined_score for kf in keyframes]

            # 品質スコアデータ（タイムライン表示用）
            # Stage 1のスコアがあればそれを使用
            quality_scores = getattr(selector, '_stage1_scores', [])
            if not quality_scores:
                quality_scores = [0.0] * meta.frame_count
                for kf in keyframes:
                    if 0 <= kf.frame_index < len(quality_scores):
                        quality_scores[kf.frame_index] = kf.combined_score

            loader.close()

            # 結果を送信
            self.quality_data.emit(quality_scores)
            self.keyframes_found.emit(keyframe_frames, keyframe_scores)
            self.finished.emit()

        except Exception as e:
            logger.exception("分析エラー")
            self.error.emit(f"分析エラー: {str(e)}")

    def stop(self):
        """分析を停止"""
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

            # キーフレームパネルにビデオパスを設定（サムネイル読み込み用）
            self.keyframe_panel.set_video_path(file_path)

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

        # GUI設定をKeyframeSelector用に変換して渡す
        selector_config = self._build_selector_config()

        # 分析ワーカーの作成（最適化版: KeyframeSelectorの2段階パイプライン使用）
        self.analysis_worker = AnalysisWorker(self.video_path, config=selector_config)
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

        設定ダイアログで指定された出力形式、JPEG品質、ファイル命名規則を使用。
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

        # GUI設定を読み込む
        gui_settings = self._load_gui_settings()
        output_format = gui_settings.get('output_image_format', 'png').lower()
        jpeg_quality = gui_settings.get('output_jpeg_quality', 95)
        naming_prefix = gui_settings.get('naming_prefix', 'keyframe')
        default_dir = gui_settings.get('output_directory', str(Path.home() / "360split_output"))

        # 拡張子のマッピング（設定値 → ファイル拡張子）
        format_ext_map = {
            'png': 'png',
            'jpeg': 'jpg',
            'jpg': 'jpg',
            'tiff': 'tiff',
            'tif': 'tiff',
        }
        file_ext = format_ext_map.get(output_format, 'png')

        # エクスポート先ディレクトリの選択（設定のデフォルトディレクトリを初期パスにする）
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "エクスポート先ディレクトリを選択",
            default_dir
        )

        if not export_dir:
            return

        try:
            import cv2

            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            # ビデオから各キーフレームを抽出
            cap = cv2.VideoCapture(self.video_path)
            exported_count = 0

            for frame_idx in sorted(keyframes):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # 設定に基づくファイル名生成
                    output_file = export_path / f"{naming_prefix}_{frame_idx:06d}.{file_ext}"

                    # 形式に応じた保存パラメータ
                    if file_ext == 'jpg':
                        cv2.imwrite(str(output_file), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    elif file_ext == 'tiff':
                        cv2.imwrite(str(output_file), frame)
                    else:  # png
                        cv2.imwrite(str(output_file), frame,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

                    exported_count += 1

            cap.release()

            self.update_status(
                f"キーフレーム {exported_count} 個をエクスポートしました: {export_dir}"
            )
            QMessageBox.information(
                self,
                "完了",
                f"{exported_count} 個のキーフレームを {file_ext.upper()} 形式でエクスポートしました\n"
                f"出力先: {export_dir}"
            )

            logger.info(
                f"キーフレームをエクスポート: {exported_count}個 -> {export_dir} "
                f"(形式: {file_ext}, プレフィックス: {naming_prefix})"
            )

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

    def _load_gui_settings(self) -> dict:
        """
        GUI設定ファイル (~/.360split/settings.json) を読み込む

        Returns:
        --------
        dict
            設定辞書。ファイルが存在しない場合はデフォルト値
        """
        settings_file = Path.home() / ".360split" / "settings.json"
        default_settings = {
            'weight_sharpness': 0.30,
            'weight_exposure': 0.15,
            'weight_geometric': 0.30,
            'weight_content': 0.25,
            'ssim_threshold': 0.85,
            'min_keyframe_interval': 5,
            'max_keyframe_interval': 60,
            'softmax_beta': 5.0,
            'output_image_format': 'png',
            'output_jpeg_quality': 95,
            'output_directory': str(Path.home() / "360split_output"),
            'naming_prefix': 'keyframe',
        }

        try:
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default_settings.update(loaded)
                logger.info(f"GUI設定を読み込みました: {settings_file}")
        except Exception as e:
            logger.warning(f"GUI設定読み込みエラー: {e} (デフォルト値を使用)")

        return default_settings

    def _build_selector_config(self) -> dict:
        """
        GUI設定をKeyframeSelector用の設定辞書に変換

        SettingsDialogの小文字キーをKeyframeSelectorの大文字キーにマッピングする。

        Returns:
        --------
        dict
            KeyframeSelectorのconfigに渡す設定辞書
        """
        gui_settings = self._load_gui_settings()

        selector_config = {
            'WEIGHT_SHARPNESS': gui_settings.get('weight_sharpness', 0.30),
            'WEIGHT_EXPOSURE': gui_settings.get('weight_exposure', 0.15),
            'WEIGHT_GEOMETRIC': gui_settings.get('weight_geometric', 0.30),
            'WEIGHT_CONTENT': gui_settings.get('weight_content', 0.25),
            'SSIM_CHANGE_THRESHOLD': gui_settings.get('ssim_threshold', 0.85),
            'MIN_KEYFRAME_INTERVAL': gui_settings.get('min_keyframe_interval', 5),
            'MAX_KEYFRAME_INTERVAL': gui_settings.get('max_keyframe_interval', 60),
            'SOFTMAX_BETA': gui_settings.get('softmax_beta', 5.0),
        }

        logger.info(
            f"分析設定: 重み[S={selector_config['WEIGHT_SHARPNESS']:.2f}, "
            f"E={selector_config['WEIGHT_EXPOSURE']:.2f}, "
            f"G={selector_config['WEIGHT_GEOMETRIC']:.2f}, "
            f"C={selector_config['WEIGHT_CONTENT']:.2f}], "
            f"SSIM閾値={selector_config['SSIM_CHANGE_THRESHOLD']:.2f}, "
            f"間隔={selector_config['MIN_KEYFRAME_INTERVAL']}-{selector_config['MAX_KEYFRAME_INTERVAL']}"
        )

        return selector_config

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
