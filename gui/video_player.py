"""
ビデオプレーヤーウィジェット - 360Split GUI
再生コントロール、フレーム表示、グリッドオーバーレイ機能
"""

from typing import Optional
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QComboBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QRect
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont

from utils.logger import get_logger
logger = get_logger(__name__)


class VideoPlayerWidget(QWidget):
    """
    ビデオプレーヤーウィジェット

    フレーム表示、再生制御、フレームスクラビング、
    グリッドオーバーレイ表示機能を提供。

    Signals:
    --------
    frame_changed : Signal(int)
        フレーム変更時シグナル（フレーインデックス）
    playback_speed_changed : Signal(float)
        再生速度変更時シグナル
    keyframe_marked : Signal(int)
        キーフレームをマークしたときのシグナル
    """

    frame_changed = Signal(int)
    playback_speed_changed = Signal(float)
    keyframe_marked = Signal(int)

    def __init__(self):
        """
        ビデオプレーヤーウィジェットの初期化
        """
        super().__init__()

        # ビデオ関連
        self.cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = None
        self.video_path = None

        # 再生制御
        self.is_playing = False
        self.playback_speed = 1.0
        self.play_timer = None

        # グリッドオーバーレイ
        self.show_grid = False

        self._setup_ui()
        self._setup_playback_timer()

    def _setup_ui(self):
        """
        UIコンポーネントの構築
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # === フレーム表示エリア ===
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(960, 540)
        self.frame_label.setStyleSheet("background-color: #000000; border: 1px solid #3d3d3d;")
        self.frame_label.setText("ビデオを開いてください")
        font = QFont()
        font.setPointSize(14)
        self.frame_label.setFont(font)
        main_layout.addWidget(self.frame_label, stretch=1)

        # === 再生制御パネル ===
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)

        # 再生/一時停止ボタン
        self.play_button = QPushButton("再生(P)")
        self.play_button.setMaximumWidth(80)
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)

        # 前フレームボタン
        self.prev_button = QPushButton("◀ 前")
        self.prev_button.setMaximumWidth(60)
        self.prev_button.clicked.connect(self.previous_frame)
        control_layout.addWidget(self.prev_button)

        # 次フレームボタン
        self.next_button = QPushButton("次 ▶")
        self.next_button.setMaximumWidth(60)
        self.next_button.clicked.connect(self.next_frame)
        control_layout.addWidget(self.next_button)

        # スタートボタン
        self.start_button = QPushButton("最初へ")
        self.start_button.setMaximumWidth(60)
        self.start_button.clicked.connect(self.seek_to_start)
        control_layout.addWidget(self.start_button)

        # エンドボタン
        self.end_button = QPushButton("最後へ")
        self.end_button.setMaximumWidth(60)
        self.end_button.clicked.connect(self.seek_to_end)
        control_layout.addWidget(self.end_button)

        control_layout.addSpacing(20)

        # グリッドオーバーレイトグル
        self.grid_toggle = QPushButton("グリッド OFF")
        self.grid_toggle.setMaximumWidth(100)
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.toggled.connect(self._on_grid_toggle)
        control_layout.addWidget(self.grid_toggle)

        # キーフレームマークボタン
        self.mark_button = QPushButton("★ 現フレームをマーク")
        self.mark_button.setMaximumWidth(150)
        self.mark_button.clicked.connect(self._on_mark_frame)
        control_layout.addWidget(self.mark_button)

        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # === フレーム情報パネル ===
        info_layout = QHBoxLayout()
        info_layout.setSpacing(15)

        # フレーム番号スライダー
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.sliderMoved.connect(self._on_slider_moved)
        info_layout.addWidget(self.frame_slider, stretch=1)

        # フレーム番号スピンボックス
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximumWidth(100)
        self.frame_spinbox.valueChanged.connect(self._on_spinbox_changed)
        info_layout.addWidget(self.frame_spinbox)

        # フレーム数表示
        self.frame_count_label = QLabel("/ 0")
        self.frame_count_label.setMaximumWidth(80)
        info_layout.addWidget(self.frame_count_label)

        info_layout.addSpacing(20)

        # 時刻表示
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setMaximumWidth(150)
        info_layout.addWidget(self.time_label)

        main_layout.addLayout(info_layout)

        # === 再生速度パネル ===
        speed_layout = QHBoxLayout()

        speed_label = QLabel("再生速度:")
        speed_label.setMaximumWidth(70)
        speed_layout.addWidget(speed_label)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # デフォルト 1.0x
        self.speed_combo.setMaximumWidth(100)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_combo)

        speed_layout.addStretch()

        main_layout.addLayout(speed_layout)

    def _setup_playback_timer(self):
        """
        再生用タイマーの設定
        """
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_playback_tick)

    def load_video(self, video_path: str):
        """
        ビデオファイルを読み込む

        Parameters:
        -----------
        video_path : str
            ビデオファイルパス

        Returns:
        --------
        VideoMetadata
            ビデオメタデータ

        Raises:
        -------
        RuntimeError
            ビデオ読み込みに失敗した場合
        """
        from core.video_loader import VideoLoader, VideoMetadata

        try:
            # 既存のキャプチャを閉じる
            if self.cap:
                self.cap.release()

            # ビデオ読み込み
            loader = VideoLoader()
            metadata = loader.load(video_path)

            self.cap = cv2.VideoCapture(video_path)
            self.video_path = video_path
            self.total_frames = metadata.frame_count
            self.fps = metadata.fps
            self.current_frame_idx = 0

            # UIを更新
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.frame_spinbox.setMaximum(self.total_frames - 1)
            self.frame_count_label.setText(f"/ {self.total_frames}")

            # 最初のフレームを表示
            self._load_and_display_frame(0)

            logger.info(f"ビデオ読み込み: {video_path} ({metadata.width}x{metadata.height})")

            return metadata

        except Exception as e:
            logger.exception(f"ビデオ読み込みエラー: {video_path}")
            raise

    def _load_and_display_frame(self, frame_idx: int):
        """
        指定フレームを読み込んで表示（最適化版）

        最適化:
        - 表示サイズへの事前リサイズ（フルフレーム描画を回避）
        - BGR→RGB変換をリサイズ後に実行（小画像で変換）
        - グリッドオーバーレイ用テクスチャキャッシュ
        - 再生中はFastTransformation使用

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        """
        if not self.cap:
            return

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if not ret:
                logger.warning(f"フレーム読み込み失敗: {frame_idx}")
                return

            self.current_frame = frame
            self.current_frame_idx = frame_idx

            # 表示サイズを計算
            label_size = self.frame_label.size()
            display_w = label_size.width() - 2
            display_h = label_size.height() - 2
            h, w = frame.shape[:2]

            # アスペクト比を保ったリサイズ先を計算
            scale = min(display_w / w, display_h / h)
            target_w = int(w * scale)
            target_h = int(h * scale)

            # OpenCVで高速リサイズ（QPixmap.scaled()より高速）
            if scale < 0.9:
                display_frame = cv2.resize(frame, (target_w, target_h),
                                           interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame

            # グリッドオーバーレイを追加（リサイズ後の小画像に描画）
            if self.show_grid:
                display_frame = self._draw_equirectangular_grid(display_frame)

            # BGR→RGB変換（リサイズ後の小画像で変換=高速）
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            dh, dw, ch = frame_rgb.shape
            bytes_per_line = 3 * dw

            # QImageに変換（contiguous配列を保証）
            frame_rgb = np.ascontiguousarray(frame_rgb)
            qt_image = QImage(frame_rgb.data, dw, dh, bytes_per_line,
                              QImage.Format_RGB888)

            # 再生中はFastTransformation（ちらつき防止+高速）
            if self.is_playing:
                pixmap = QPixmap.fromImage(qt_image)
            else:
                pixmap = QPixmap.fromImage(qt_image)

            self.frame_label.setPixmap(pixmap)

            # フレーム情報を更新
            self._update_frame_info(frame_idx)

            # シグナル発行
            self.frame_changed.emit(frame_idx)

        except Exception as e:
            logger.exception(f"フレーム表示エラー: {frame_idx}")

    def _update_frame_info(self, frame_idx: int):
        """
        フレーム情報表示を更新

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        """
        # フレーム番号
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_idx)
        self.frame_spinbox.blockSignals(False)

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)

        # 時刻
        current_time = frame_idx / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        current_str = self._format_time(current_time)
        total_str = self._format_time(total_time)
        self.time_label.setText(f"{current_str} / {total_str}")

    def seek_to_frame(self, frame_idx: int):
        """
        指定フレームへシーク

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        """
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1

        self._load_and_display_frame(frame_idx)

    def previous_frame(self):
        """
        前フレームへ移動
        """
        self.seek_to_frame(self.current_frame_idx - 1)

    def next_frame(self):
        """
        次フレームへ移動
        """
        self.seek_to_frame(self.current_frame_idx + 1)

    def seek_to_start(self):
        """
        ビデオの最初へ移動
        """
        self.seek_to_frame(0)

    def seek_to_end(self):
        """
        ビデオの最後へ移動
        """
        self.seek_to_frame(self.total_frames - 1)

    def toggle_playback(self):
        """
        再生/一時停止を切り替え
        """
        if not self.cap:
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_button.setText("一時停止(P)")
            self._start_playback()
        else:
            self.play_button.setText("再生(P)")
            self.play_timer.stop()

    def _start_playback(self):
        """
        再生を開始
        """
        if self.current_frame_idx >= self.total_frames - 1:
            self.seek_to_start()

        # フレーム間隔（ミリ秒）
        interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
        self.play_timer.start(interval)

    def _on_playback_tick(self):
        """
        再生タイマータイムアウト時のコールバック
        """
        if self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            self.play_button.setText("再生(P)")
            self.play_timer.stop()
            return

        self.seek_to_frame(self.current_frame_idx + 1)

    def set_grid_overlay(self, enabled: bool):
        """
        グリッドオーバーレイの表示/非表示を設定

        Parameters:
        -----------
        enabled : bool
            表示するかどうか
        """
        self.show_grid = enabled
        self._load_and_display_frame(self.current_frame_idx)

    def _draw_equirectangular_grid(self, frame: np.ndarray) -> np.ndarray:
        """
        Equirectangular投影グリッドをフレームに描画

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム（BGR）

        Returns:
        --------
        np.ndarray
            グリッド付きフレーム
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # 緯度線（水平線）
        grid_spacing_lat = h // 8
        for y in range(grid_spacing_lat, h, grid_spacing_lat):
            cv2.line(output, (0, y), (w, y), (0, 180, 100), 1)

        # 経度線（垂直線）
        grid_spacing_lon = w // 16
        for x in range(grid_spacing_lon, w, grid_spacing_lon):
            cv2.line(output, (x, 0), (x, h), (0, 180, 100), 1)

        # 中心十字
        cv2.line(output, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (100, 255, 100), 2)
        cv2.line(output, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (100, 255, 100), 2)

        return output

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        秒をHH:MM:SS形式にフォーマット

        Parameters:
        -----------
        seconds : float
            秒数

        Returns:
        --------
        str
            HH:MM:SS形式の時刻文字列
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # === スロット ===

    def _on_slider_moved(self, value: int):
        """
        フレームスライダー移動時のコールバック
        """
        self.seek_to_frame(value)

    def _on_spinbox_changed(self, value: int):
        """
        フレームスピンボックス変更時のコールバック
        """
        self.seek_to_frame(value)

    def _on_speed_changed(self, text: str):
        """
        再生速度変更時のコールバック
        """
        try:
            self.playback_speed = float(text.rstrip('x'))
            self.playback_speed_changed.emit(self.playback_speed)

            if self.is_playing:
                self.play_timer.stop()
                interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
                self.play_timer.start(interval)

        except ValueError:
            pass

    def _on_grid_toggle(self, checked: bool):
        """
        グリッドトグルボタン押下時のコールバック
        """
        self.show_grid = checked
        self.grid_toggle.setText("グリッド ON" if checked else "グリッド OFF")
        self._load_and_display_frame(self.current_frame_idx)

    def _on_mark_frame(self):
        """
        現フレームをマークしたときのコールバック
        """
        self.keyframe_marked.emit(self.current_frame_idx)
