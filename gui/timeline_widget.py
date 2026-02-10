"""
タイムラインウィジェット - 360Split GUI
キーフレームマーカー、品質スコア可視化、タイムライン操作
"""

from typing import List, Optional
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMenu
from PySide6.QtCore import Qt, QRect, QSize, Signal, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush, QAction

from utils.logger import get_logger
logger = get_logger(__name__)


class TimelineWidget(QWidget):
    """
    カスタムタイムラインウィジェット

    ビデオの全長を表示し、キーフレーム位置をマーカーで表示、
    各フレームの品質スコアを波形で可視化。
    クリック操作でシーク、右クリックメニューでキーフレーム管理。

    Signals:
    --------
    positionChanged : Signal(int)
        位置変更時シグナル（フレームインデックス）
    keyframeClicked : Signal(int)
        キーフレームがクリックされたときのシグナル
    keyframeRemoved : Signal(int)
        キーフレームが削除されたときのシグナル
    """

    positionChanged = Signal(int)
    keyframeClicked = Signal(int)
    keyframeRemoved = Signal(int)

    def __init__(self):
        """
        タイムラインウィジェットの初期化
        """
        super().__init__()

        # タイムライン設定
        self.total_frames = 0
        self.fps = 30.0
        self.current_position = 0
        self.pixels_per_frame = 1.0

        # キーフレーム
        self.keyframe_frames: List[int] = []
        self.keyframe_scores: List[float] = []

        # 品質スコアデータ
        self.quality_data: List[float] = []

        # マウスホバー情報
        self.hovered_frame = -1

        # UI設定
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3d3d3d;")

    def set_duration(self, frame_count: int, fps: float):
        """
        タイムライン範囲を設定

        Parameters:
        -----------
        frame_count : int
            総フレーム数
        fps : float
            フレームレート
        """
        self.total_frames = frame_count
        self.fps = fps
        self._calculate_scale()

    def set_position(self, frame_idx: int):
        """
        現在位置を設定

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        """
        if 0 <= frame_idx < self.total_frames:
            self.current_position = frame_idx
            self.update()

    def set_keyframes(self, frames: List[int], scores: List[float]):
        """
        キーフレーム情報を設定

        Parameters:
        -----------
        frames : List[int]
            キーフレームのフレームインデックスリスト
        scores : List[float]
            対応するスコアリスト（0-1）
        """
        self.keyframe_frames = frames
        self.keyframe_scores = scores
        self.update()

    def set_quality_data(self, scores: List[float]):
        """
        品質スコアデータを設定

        Parameters:
        -----------
        scores : List[float]
            各フレームの品質スコアリスト（0-1）
        """
        self.quality_data = scores
        self.update()

    def remove_keyframe(self, frame_idx: int):
        """
        指定フレームのキーフレームマークを削除

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        """
        if frame_idx in self.keyframe_frames:
            idx = self.keyframe_frames.index(frame_idx)
            self.keyframe_frames.pop(idx)
            self.keyframe_scores.pop(idx)
            self.update()
            self.keyframeRemoved.emit(frame_idx)

    def _calculate_scale(self):
        """
        フレームからピクセルへの変換スケールを計算
        """
        widget_width = self.width() - 20  # 左右マージン
        if self.total_frames > 0:
            self.pixels_per_frame = widget_width / self.total_frames
        else:
            self.pixels_per_frame = 1.0

    def paintEvent(self, event):
        """
        タイムラインのペイント処理
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        if self.total_frames == 0:
            painter.drawText(self.rect(), Qt.AlignCenter, "ビデオを読み込んでください")
            return

        # タイムラインエリア
        margin = 10
        timeline_y = 40
        timeline_height = 30
        timeline_rect = QRect(margin, timeline_y, self.width() - 2 * margin, timeline_height)

        # === 品質スコアの波形を描画 ===
        if self.quality_data:
            self._draw_quality_waveform(painter, timeline_rect)

        # === タイムラインバーを描画 ===
        self._draw_timeline_bar(painter, timeline_rect)

        # === キーフレームマーカーを描画 ===
        self._draw_keyframe_markers(painter, timeline_rect)

        # === 現在位置インジケータを描画 ===
        self._draw_position_indicator(painter, timeline_rect)

        # === 時刻ラベルを描画 ===
        self._draw_time_labels(painter, timeline_rect)

    def _draw_quality_waveform(self, painter: QPainter, timeline_rect: QRect):
        """
        品質スコアの波形を描画

        Parameters:
        -----------
        painter : QPainter
            ペインタオブジェクト
        timeline_rect : QRect
            タイムラインの描画領域
        """
        if not self.quality_data or len(self.quality_data) == 0:
            return

        painter.setOpacity(0.5)
        painter.setPen(QPen(QColor("#6090e0"), 1))
        painter.setBrush(QBrush(QColor("#5080d0")))

        # ポイントを計算
        points = []
        data_len = len(self.quality_data)
        step = max(1, data_len // (timeline_rect.width() // 2))

        for i in range(0, data_len, step):
            score = self.quality_data[i]
            x = timeline_rect.left() + (i / self.total_frames) * timeline_rect.width()
            y = timeline_rect.bottom() - score * timeline_rect.height()
            points.append((int(x), int(y)))

        # 波形をポリラインで描画
        if len(points) > 1:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                painter.drawLine(x1, y1, x2, y2)

        painter.setOpacity(1.0)

    def _draw_timeline_bar(self, painter: QPainter, timeline_rect: QRect):
        """
        タイムラインバーを描画

        Parameters:
        -----------
        painter : QPainter
            ペインタオブジェクト
        timeline_rect : QRect
            タイムラインの描画領域
        """
        # 背景バー
        painter.fillRect(timeline_rect, QColor("#3d3d3d"))

        # 枠線
        painter.setPen(QPen(QColor("#505050"), 1))
        painter.drawRect(timeline_rect)

        # 目盛り
        painter.setPen(QPen(QColor("#505050"), 1))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        # 10フレームごとに目盛りを描画
        tick_interval = max(1, self.total_frames // 20)
        for frame in range(0, self.total_frames, tick_interval):
            x = timeline_rect.left() + (frame / self.total_frames) * timeline_rect.width()
            painter.drawLine(int(x), timeline_rect.bottom(), int(x), timeline_rect.bottom() + 3)

            # 時刻を表示（60フレームごと）
            if frame % (tick_interval * 2) == 0:
                time_sec = frame / self.fps if self.fps > 0 else 0
                time_str = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
                painter.setPen(QPen(QColor("#808080"), 1))
                painter.drawText(int(x) - 15, timeline_rect.bottom() + 15, 30, 15, Qt.AlignCenter, time_str)

    def _draw_keyframe_markers(self, painter: QPainter, timeline_rect: QRect):
        """
        キーフレームマーカーを描画

        Parameters:
        -----------
        painter : QPainter
            ペインタオブジェクト
        timeline_rect : QRect
            タイムラインの描画領域
        """
        for frame_idx, score in zip(self.keyframe_frames, self.keyframe_scores):
            # マーカー位置
            x = timeline_rect.left() + (frame_idx / self.total_frames) * timeline_rect.width()

            # スコアに基づいて色分け（高スコア=緑、中=黄、低=赤）
            if score >= 0.75:
                color = QColor("#2aff2a")  # 緑
            elif score >= 0.5:
                color = QColor("#ffff00")  # 黄
            else:
                color = QColor("#ff4444")  # 赤

            # マーカー三角形を描画
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 1))

            marker_width = 10
            marker_height = 15
            points = [
                QPoint(int(x), timeline_rect.top() - marker_height),
                QPoint(int(x - marker_width // 2), timeline_rect.top()),
                QPoint(int(x + marker_width // 2), timeline_rect.top()),
            ]
            painter.drawPolygon(points)

    def _draw_position_indicator(self, painter: QPainter, timeline_rect: QRect):
        """
        現在位置インジケータを描画

        Parameters:
        -----------
        painter : QPainter
            ペインタオブジェクト
        timeline_rect : QRect
            タイムラインの描画領域
        """
        x = timeline_rect.left() + (self.current_position / self.total_frames) * timeline_rect.width()

        # 垂直線
        painter.setPen(QPen(QColor("#ffff00"), 2))
        painter.drawLine(int(x), timeline_rect.top(), int(x), timeline_rect.bottom())

        # 上部マーカー
        painter.setBrush(QBrush(QColor("#ffff00")))
        painter.drawEllipse(int(x) - 4, timeline_rect.top() - 12, 8, 8)

    def _draw_time_labels(self, painter: QPainter, timeline_rect: QRect):
        """
        時刻ラベルを描画

        Parameters:
        -----------
        painter : QPainter
            ペインタオブジェクト
        timeline_rect : QRect
            タイムラインの描画領域
        """
        painter.setPen(QPen(QColor("#a0a0a0"), 1))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        # 開始時刻
        painter.drawText(5, 20, 50, 15, Qt.AlignLeft, "00:00")

        # 終了時刻
        end_sec = self.total_frames / self.fps if self.fps > 0 else 0
        end_str = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
        painter.drawText(self.width() - 55, 20, 50, 15, Qt.AlignRight, end_str)

        # 現在時刻
        current_sec = self.current_position / self.fps if self.fps > 0 else 0
        current_str = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d}"
        painter.drawText(self.width() // 2 - 25, 20, 50, 15, Qt.AlignCenter, current_str)

    def mousePressEvent(self, event):
        """
        マウスプレスイベント処理
        """
        if event.button() == Qt.LeftButton:
            # 左クリック：位置変更
            frame_idx = self._pixel_to_frame(int(event.position().x()))
            self.set_position(frame_idx)
            self.positionChanged.emit(frame_idx)

        elif event.button() == Qt.RightButton:
            # 右クリック：コンテキストメニュー
            frame_idx = self._pixel_to_frame(int(event.position().x()))
            self._show_context_menu(frame_idx, event.globalPosition().toPoint())

    def mouseMoveEvent(self, event):
        """
        マウスムーブイベント処理
        """
        frame_idx = self._pixel_to_frame(int(event.position().x()))
        self.hovered_frame = frame_idx
        self.update()

    def _pixel_to_frame(self, pixel_x: int) -> int:
        """
        ピクセル座標をフレームインデックスに変換

        Parameters:
        -----------
        pixel_x : int
            ピクセルX座標

        Returns:
        --------
        int
            フレームインデックス
        """
        margin = 10
        timeline_width = self.width() - 2 * margin
        relative_x = max(0, min(pixel_x - margin, timeline_width))
        frame_idx = int((relative_x / timeline_width) * self.total_frames)
        return min(frame_idx, self.total_frames - 1)

    def _show_context_menu(self, frame_idx: int, global_pos: QPoint):
        """
        コンテキストメニューを表示

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        global_pos : QPoint
            グローバル座標
        """
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())

        # キーフレームが存在するかチェック
        is_keyframe = frame_idx in self.keyframe_frames

        if is_keyframe:
            remove_action = QAction("このフレームからキーフレームを削除", self)
            remove_action.triggered.connect(lambda: self.remove_keyframe(frame_idx))
            menu.addAction(remove_action)

        else:
            add_action = QAction("このフレームをキーフレームに追加", self)
            add_action.triggered.connect(lambda: self._add_keyframe(frame_idx))
            menu.addAction(add_action)

        menu.exec(global_pos)  # In PySide6, exec() is used instead of exec_()

    def _add_keyframe(self, frame_idx: int):
        """
        キーフレームを追加

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        """
        if frame_idx not in self.keyframe_frames:
            # スコアがない場合は0.5を使用
            score = self.quality_data[frame_idx] if 0 <= frame_idx < len(self.quality_data) else 0.5
            self.keyframe_frames.append(frame_idx)
            self.keyframe_scores.append(score)
            self.keyframe_frames.sort()
            idx = self.keyframe_frames.index(frame_idx)
            self.keyframe_scores = [self.keyframe_scores[i] for i in sorted(range(len(self.keyframe_frames)),
                                                                               key=lambda k: self.keyframe_frames[k])]
            self.update()
