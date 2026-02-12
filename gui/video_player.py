"""
ビデオプレーヤーウィジェット - 360Split v2 GUI
OpenCV + QPixmap による動画再生。

機能:
  - 再生 / 一時停止 / コマ送り / シーク
  - キーフレーム候補インジケータ（✅ Keyframe）
  - 360° Equirectangular 画像のドラッグ Pan（簡易ビューア）
  - グリッドオーバーレイ（Equirectangular 格子線）
"""

from typing import Optional, Set
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QComboBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen

from utils.logger import get_logger
logger = get_logger(__name__)


class VideoPlayerWidget(QWidget):
    """
    ビデオプレーヤーウィジェット

    Signals
    -------
    frame_changed : Signal(int)
        フレーム変更時
    playback_speed_changed : Signal(float)
        再生速度変更時
    keyframe_marked : Signal(int)
        ユーザーが現フレームを手動マーク
    """

    frame_changed = Signal(int)
    playback_speed_changed = Signal(float)
    keyframe_marked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ビデオ
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_idx: int = 0
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_frame: Optional[np.ndarray] = None
        self.video_path: Optional[str] = None

        # ステレオ（OSV）対応
        self.is_stereo: bool = False
        self.cap_l: Optional[cv2.VideoCapture] = None
        self.cap_r: Optional[cv2.VideoCapture] = None
        self.stereo_left_path: Optional[str] = None
        self.stereo_right_path: Optional[str] = None
        self.current_frame_l: Optional[np.ndarray] = None
        self.current_frame_r: Optional[np.ndarray] = None
        self.stereo_display_mode: str = "side_by_side"  # side_by_side, anaglyph, toggle, mono_left, mono_right

        # 再生制御
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.play_timer: Optional[QTimer] = None

        # キーフレーム候補セット
        self._keyframe_set: Set[int] = set()

        # グリッドオーバーレイ
        self.show_grid: bool = False

        # 360° パン
        self._pan_offset_x: int = 0       # ピクセルオフセット（横方向）
        self._pan_drag_start: Optional[QPoint] = None
        self._is_equirectangular: bool = False

        self._setup_ui()
        self._setup_playback_timer()

    # ==================================================================
    # UI
    # ==================================================================

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # === フレーム表示 ===
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 360)
        self.frame_label.setStyleSheet("background-color: #000; border: 1px solid #3d3d3d;")
        self.frame_label.setText("ビデオを開いてください")
        font = QFont()
        font.setPointSize(14)
        self.frame_label.setFont(font)
        main_layout.addWidget(self.frame_label, stretch=1)

        # === 再生制御 ===
        ctrl = QHBoxLayout()
        ctrl.setSpacing(5)

        self.play_button = QPushButton("▶ 再生")
        self.play_button.setFixedWidth(80)
        self.play_button.clicked.connect(self.toggle_playback)
        ctrl.addWidget(self.play_button)

        self.prev_button = QPushButton("◀")
        self.prev_button.setFixedWidth(40)
        self.prev_button.clicked.connect(self.previous_frame)
        ctrl.addWidget(self.prev_button)

        self.next_button = QPushButton("▶")
        self.next_button.setFixedWidth(40)
        self.next_button.clicked.connect(self.next_frame)
        ctrl.addWidget(self.next_button)

        self.start_button = QPushButton("⏮")
        self.start_button.setFixedWidth(40)
        self.start_button.clicked.connect(self.seek_to_start)
        ctrl.addWidget(self.start_button)

        self.end_button = QPushButton("⏭")
        self.end_button.setFixedWidth(40)
        self.end_button.clicked.connect(self.seek_to_end)
        ctrl.addWidget(self.end_button)

        ctrl.addSpacing(15)

        self.grid_toggle = QPushButton("Grid OFF")
        self.grid_toggle.setFixedWidth(80)
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.toggled.connect(self._on_grid_toggle)
        ctrl.addWidget(self.grid_toggle)

        # ステレオ表示モード切り替え
        ctrl.addWidget(QLabel("ステレオ:"))
        self.stereo_mode_combo = QComboBox()
        self.stereo_mode_combo.addItems([
            "Side-by-Side",
            "アナグリフ",
            "左のみ",
            "右のみ",
            "L/R切替"
        ])
        self.stereo_mode_combo.setCurrentIndex(0)
        self.stereo_mode_combo.setFixedWidth(120)
        self.stereo_mode_combo.currentIndexChanged.connect(self._on_stereo_mode_changed)
        self.stereo_mode_combo.setVisible(False)  # 初期は非表示
        ctrl.addWidget(self.stereo_mode_combo)

        self.mark_button = QPushButton("★ マーク")
        self.mark_button.setFixedWidth(80)
        self.mark_button.clicked.connect(lambda: self.keyframe_marked.emit(self.current_frame_idx))
        ctrl.addWidget(self.mark_button)

        ctrl.addStretch()

        # 再生速度
        ctrl.addWidget(QLabel("速度:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.setFixedWidth(70)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        ctrl.addWidget(self.speed_combo)

        main_layout.addLayout(ctrl)

        # === フレーム位置 ===
        pos_layout = QHBoxLayout()
        pos_layout.setSpacing(8)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.sliderMoved.connect(self._on_slider_moved)
        pos_layout.addWidget(self.frame_slider, stretch=1)

        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setFixedWidth(90)
        self.frame_spinbox.valueChanged.connect(self._on_spinbox_changed)
        pos_layout.addWidget(self.frame_spinbox)

        self.frame_count_label = QLabel("/ 0")
        self.frame_count_label.setFixedWidth(60)
        pos_layout.addWidget(self.frame_count_label)

        pos_layout.addSpacing(10)

        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFixedWidth(130)
        pos_layout.addWidget(self.time_label)

        main_layout.addLayout(pos_layout)

    def _setup_playback_timer(self):
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_playback_tick)

    # ==================================================================
    # 公開API
    # ==================================================================

    def load_video(self, video_path: str):
        """ビデオを読み込む。VideoMetadata を返す。"""
        from core.video_loader import VideoLoader, VideoMetadata

        if self.cap:
            self.cap.release()

        # ステレオモードをリセット
        self.is_stereo = False
        self.stereo_mode_combo.setVisible(False)

        loader = VideoLoader()
        metadata = loader.load(video_path)

        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.total_frames = metadata.frame_count
        self.fps = metadata.fps
        self.current_frame_idx = 0
        self._pan_offset_x = 0

        # Equirectangular 判定（2:1 アスペクト比）
        ratio = metadata.width / metadata.height if metadata.height > 0 else 1
        self._is_equirectangular = (1.9 <= ratio <= 2.1)

        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_spinbox.setMaximum(max(0, self.total_frames - 1))
        self.frame_count_label.setText(f"/ {self.total_frames}")

        self._load_and_display_frame(0)
        logger.info(f"ビデオ読み込み: {video_path}")
        return metadata

    def load_video_stereo(self, left_path: str, right_path: str):
        """
        ステレオビデオを読み込む（OSV用）

        Parameters
        ----------
        left_path : str
            左目ストリームのパス
        right_path : str
            右目ストリームのパス

        Returns
        -------
        VideoMetadata
            メタデータ（左目ストリームベース）
        """
        from core.video_loader import VideoLoader, VideoMetadata

        # 既存キャプチャをクリーンアップ
        if self.cap:
            self.cap.release()
        if self.cap_l:
            self.cap_l.release()
        if self.cap_r:
            self.cap_r.release()

        # メタデータ取得（左目ストリームから）
        loader = VideoLoader()
        metadata = loader.load(left_path)

        # ステレオキャプチャを開く
        self.cap_l = cv2.VideoCapture(left_path)
        self.cap_r = cv2.VideoCapture(right_path)
        self.stereo_left_path = left_path
        self.stereo_right_path = right_path
        self.is_stereo = True

        self.video_path = f"{left_path} (Stereo)"
        self.total_frames = metadata.frame_count
        self.fps = metadata.fps
        self.current_frame_idx = 0
        self._pan_offset_x = 0

        # Equirectangular 判定
        ratio = metadata.width / metadata.height if metadata.height > 0 else 1
        self._is_equirectangular = (1.9 <= ratio <= 2.1)

        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_spinbox.setMaximum(max(0, self.total_frames - 1))
        self.frame_count_label.setText(f"/ {self.total_frames}")

        # ステレオモードコンボボックスを表示
        self.stereo_mode_combo.setVisible(True)

        self._load_and_display_frame(0)
        logger.info(f"ステレオビデオ読み込み: L={left_path}, R={right_path}")
        return metadata

    def set_keyframe_indices(self, indices):
        """キーフレーム候補のフレーム番号セットを設定"""
        self._keyframe_set = set(indices)
        self._redraw_current()

    def seek_to_frame(self, frame_idx: int):
        """指定フレームへシーク"""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self._load_and_display_frame(frame_idx)

    def previous_frame(self):
        self.seek_to_frame(self.current_frame_idx - 1)

    def next_frame(self):
        self.seek_to_frame(self.current_frame_idx + 1)

    def seek_to_start(self):
        self.seek_to_frame(0)

    def seek_to_end(self):
        self.seek_to_frame(self.total_frames - 1)

    def toggle_playback(self):
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText("⏸ 停止")
            if self.current_frame_idx >= self.total_frames - 1:
                self.seek_to_start()
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
            self.play_timer.start(interval)
        else:
            self.play_button.setText("▶ 再生")
            self.play_timer.stop()

    def set_grid_overlay(self, enabled: bool):
        self.show_grid = enabled
        self.grid_toggle.setChecked(enabled)
        self._redraw_current()

    # ==================================================================
    # 描画
    # ==================================================================

    def _load_and_display_frame(self, frame_idx: int):
        if self.is_stereo:
            # ステレオモード
            if not self.cap_l or not self.cap_r:
                return

            self.cap_l.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()

            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                return

            self.current_frame_l = frame_l
            self.current_frame_r = frame_r
            self.current_frame = frame_l  # デフォルトは左
            self.current_frame_idx = frame_idx
            self._display_frame_stereo(frame_l, frame_r)
        else:
            # 通常のモノラルモード
            if not self.cap:
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return

            self.current_frame = frame
            self.current_frame_idx = frame_idx
            self._display_frame(frame)

        self._update_frame_info(frame_idx)
        self.frame_changed.emit(frame_idx)

    def _redraw_current(self):
        if self.is_stereo and self.current_frame_l is not None and self.current_frame_r is not None:
            self._display_frame_stereo(self.current_frame_l, self.current_frame_r)
        elif self.current_frame is not None:
            self._display_frame(self.current_frame)

    def _display_frame(self, frame: np.ndarray):
        """フレームを QPixmap に変換して表示"""
        label_w = self.frame_label.width() - 2
        label_h = self.frame_label.height() - 2
        h, w = frame.shape[:2]

        display = frame.copy()

        # 360° パンオフセット適用
        if self._is_equirectangular and self._pan_offset_x != 0:
            shift = self._pan_offset_x % w
            display = np.roll(display, shift, axis=1)

        # グリッドオーバーレイ
        if self.show_grid:
            display = self._draw_grid(display)

        # リサイズ
        scale = min(label_w / w, label_h / h, 1.0)
        tw, th = int(w * scale), int(h * scale)
        if scale < 0.95:
            display = cv2.resize(display, (tw, th), interpolation=cv2.INTER_AREA)

        # BGR→RGB
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        dh, dw, _ = rgb.shape
        qimg = QImage(rgb.data, dw, dh, 3 * dw, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # キーフレームインジケータをオーバーレイ
        if self.current_frame_idx in self._keyframe_set:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            # 背景矩形
            painter.setBrush(QColor(0, 180, 0, 180))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(dw - 140, 8, 130, 28, 6, 6)
            # テキスト
            painter.setPen(QPen(QColor(255, 255, 255)))
            font = QFont("sans-serif", 12, QFont.Bold)
            painter.setFont(font)
            painter.drawText(dw - 135, 10, 120, 24, Qt.AlignCenter, "✅ Keyframe")
            painter.end()

        self.frame_label.setPixmap(pixmap)

    def _display_frame_stereo(self, frame_l: np.ndarray, frame_r: np.ndarray):
        """
        ステレオフレームを表示

        Parameters
        ----------
        frame_l : np.ndarray
            左目フレーム
        frame_r : np.ndarray
            右目フレーム
        """
        # ステレオ表示モードに応じて処理
        mode_index = self.stereo_mode_combo.currentIndex()

        if mode_index == 0:  # Side-by-Side
            display = self._create_side_by_side(frame_l, frame_r)
        elif mode_index == 1:  # アナグリフ
            display = self._create_anaglyph(frame_l, frame_r)
        elif mode_index == 2:  # 左のみ
            display = frame_l.copy()
        elif mode_index == 3:  # 右のみ
            display = frame_r.copy()
        elif mode_index == 4:  # L/R切替（時間ベース）
            # 500ms ごとに切り替え
            import time
            toggle = int(time.time() * 2) % 2
            display = frame_l.copy() if toggle == 0 else frame_r.copy()
        else:
            display = frame_l.copy()

        # 360° パンオフセット適用
        if self._is_equirectangular and self._pan_offset_x != 0:
            h, w = display.shape[:2]
            shift = self._pan_offset_x % w
            display = np.roll(display, shift, axis=1)

        # グリッドオーバーレイ
        if self.show_grid:
            display = self._draw_grid(display)

        # 表示処理
        label_w = self.frame_label.width() - 2
        label_h = self.frame_label.height() - 2
        h, w = display.shape[:2]

        # リサイズ
        scale = min(label_w / w, label_h / h, 1.0)
        tw, th = int(w * scale), int(h * scale)
        if scale < 0.95:
            display = cv2.resize(display, (tw, th), interpolation=cv2.INTER_AREA)

        # BGR→RGB
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        dh, dw, _ = rgb.shape
        qimg = QImage(rgb.data, dw, dh, 3 * dw, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # キーフレームインジケータをオーバーレイ
        if self.current_frame_idx in self._keyframe_set:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(0, 180, 0, 180))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(dw - 140, 8, 130, 28, 6, 6)
            painter.setPen(QPen(QColor(255, 255, 255)))
            font = QFont("sans-serif", 12, QFont.Bold)
            painter.setFont(font)
            painter.drawText(dw - 135, 10, 120, 24, Qt.AlignCenter, "✅ Keyframe")
            painter.end()

        self.frame_label.setPixmap(pixmap)

    def _create_side_by_side(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """
        Side-by-Side 表示を作成

        Parameters
        ----------
        frame_l : np.ndarray
            左目フレーム
        frame_r : np.ndarray
            右目フレーム

        Returns
        -------
        np.ndarray
            並列表示されたフレーム
        """
        # 高さを揃える
        h_l, w_l = frame_l.shape[:2]
        h_r, w_r = frame_r.shape[:2]

        target_h = min(h_l, h_r)

        # 左をリサイズ
        if h_l != target_h:
            scale = target_h / h_l
            frame_l = cv2.resize(frame_l, (int(w_l * scale), target_h), interpolation=cv2.INTER_AREA)

        # 右をリサイズ
        if h_r != target_h:
            scale = target_h / h_r
            frame_r = cv2.resize(frame_r, (int(w_r * scale), target_h), interpolation=cv2.INTER_AREA)

        # 水平に結合
        combined = np.hstack([frame_l, frame_r])

        # 中央に境界線を描画
        mid_x = frame_l.shape[1]
        cv2.line(combined, (mid_x, 0), (mid_x, combined.shape[0]), (0, 255, 0), 2)

        # ラベルを追加
        cv2.putText(combined, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(combined, "RIGHT", (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)

        return combined

    def _create_anaglyph(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """
        アナグリフ（赤青メガネ用）表示を作成

        Parameters
        ----------
        frame_l : np.ndarray
            左目フレーム
        frame_r : np.ndarray
            右目フレーム

        Returns
        -------
        np.ndarray
            アナグリフ表示されたフレーム
        """
        # サイズを揃える
        h_l, w_l = frame_l.shape[:2]
        h_r, w_r = frame_r.shape[:2]

        if (h_l, w_l) != (h_r, w_r):
            # 右を左のサイズにリサイズ
            frame_r = cv2.resize(frame_r, (w_l, h_l), interpolation=cv2.INTER_AREA)

        # グレースケールに変換
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # アナグリフ作成
        # 赤チャンネルに左目、青・緑チャンネルに右目
        anaglyph = np.zeros_like(frame_l)
        anaglyph[:, :, 2] = gray_l  # Red = Left
        anaglyph[:, :, 0] = gray_r  # Blue = Right
        anaglyph[:, :, 1] = gray_r  # Green = Right

        return anaglyph

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Equirectangular グリッドを描画"""
        h, w = frame.shape[:2]
        out = frame.copy()
        # 緯度線
        for y in range(h // 8, h, h // 8):
            cv2.line(out, (0, y), (w, y), (0, 180, 100), 1)
        # 経度線
        for x in range(w // 16, w, w // 16):
            cv2.line(out, (x, 0), (x, h), (0, 180, 100), 1)
        # 中心十字
        cv2.line(out, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (100, 255, 100), 2)
        cv2.line(out, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (100, 255, 100), 2)
        return out

    def _update_frame_info(self, frame_idx: int):
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_idx)
        self.frame_spinbox.blockSignals(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)

        cur = frame_idx / self.fps if self.fps > 0 else 0
        total = self.total_frames / self.fps if self.fps > 0 else 0
        self.time_label.setText(f"{self._fmt(cur)} / {self._fmt(total)}")

    @staticmethod
    def _fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ==================================================================
    # 360° ドラッグ Pan
    # ==================================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_equirectangular:
            self._pan_drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_drag_start is not None and self._is_equirectangular:
            dx = event.pos().x() - self._pan_drag_start.x()
            self._pan_offset_x += dx
            self._pan_drag_start = event.pos()
            self._redraw_current()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pan_drag_start = None
        super().mouseReleaseEvent(event)

    # ==================================================================
    # スロット
    # ==================================================================

    def _on_playback_tick(self):
        if self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            self.play_button.setText("▶ 再生")
            self.play_timer.stop()
            return
        self.seek_to_frame(self.current_frame_idx + 1)

    def _on_slider_moved(self, value: int):
        self.seek_to_frame(value)

    def _on_spinbox_changed(self, value: int):
        self.seek_to_frame(value)

    def _on_speed_changed(self, text: str):
        try:
            self.playback_speed = float(text.rstrip('x'))
            self.playback_speed_changed.emit(self.playback_speed)
            if self.is_playing:
                self.play_timer.stop()
                interval = max(1, int(1000 / (self.fps * self.playback_speed)))
                self.play_timer.start(interval)
        except ValueError:
            pass

    def _on_grid_toggle(self, checked: bool):
        self.show_grid = checked
        self.grid_toggle.setText("Grid ON" if checked else "Grid OFF")
        self._redraw_current()

    def _on_stereo_mode_changed(self, index: int):
        """ステレオ表示モード変更時の処理"""
        if self.is_stereo:
            self._redraw_current()
            logger.debug(f"ステレオ表示モード変更: {self.stereo_mode_combo.currentText()}")

    # ==================================================================
    # クリーンアップ
    # ==================================================================

    def closeEvent(self, event):
        if self.play_timer:
            self.play_timer.stop()
        if self.cap:
            self.cap.release()
        if self.cap_l:
            self.cap_l.release()
        if self.cap_r:
            self.cap_r.release()
        super().closeEvent(event)
