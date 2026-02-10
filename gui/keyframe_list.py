"""
キーフレーム一覧ウィジェット - 360Split v2 GUI
抽出されたキーフレームのサムネイル一覧を表示。

機能:
  - サムネイルグリッド表示
  - フレーム順 / スコア順ソート
  - 選択 / 一括選択 / 削除
  - クリックでプレーヤーへシーク
  - ダブルクリックで詳細表示
"""

from typing import List, Optional, Set
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout,
    QPushButton, QLabel, QCheckBox, QComboBox, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QAction

from utils.logger import get_logger
logger = get_logger(__name__)


# ======================================================================
# 個別サムネイルウィジェット
# ======================================================================

class _KeyframeThumbnail(QWidget):
    """1つのキーフレームを表すサムネイルカード"""

    clicked = Signal(int)
    double_clicked = Signal(int)
    delete_requested = Signal(int)

    THUMB_W = 200
    THUMB_H = 130

    def __init__(self, frame_idx: int, image: np.ndarray,
                 score: float, parent=None):
        super().__init__(parent)
        self.frame_idx = frame_idx
        self.image = image
        self.score = score
        self.is_selected = False

        self.setFixedSize(self.THUMB_W + 10, self.THUMB_H + 50)
        self.setCursor(Qt.PointingHandCursor)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        # サムネイル画像
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.THUMB_W, self.THUMB_H,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self._img_label = QLabel()
        self._img_label.setPixmap(pixmap)
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("border: 2px solid #3d3d3d;")
        layout.addWidget(self._img_label)

        # 情報行
        info = QHBoxLayout()
        info.setSpacing(4)

        self._cb = QCheckBox()
        self._cb.toggled.connect(self._on_check)
        info.addWidget(self._cb)

        idx_label = QLabel(f"#{self.frame_idx}")
        idx_label.setStyleSheet("color: #ccc; font-size: 10px;")
        info.addWidget(idx_label)

        info.addStretch()

        score_color = self._score_color(self.score)
        score_label = QLabel(f"{self.score:.2f}")
        score_label.setStyleSheet(f"color: {score_color}; font-size: 10px; font-weight: bold;")
        info.addWidget(score_label)

        layout.addLayout(info)

    def set_selected(self, sel: bool):
        self.is_selected = sel
        self._cb.blockSignals(True)
        self._cb.setChecked(sel)
        self._cb.blockSignals(False)
        border = "3px solid #5080d0" if sel else "2px solid #3d3d3d"
        self._img_label.setStyleSheet(f"border: {border};")

    def _on_check(self, checked: bool):
        self.set_selected(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frame_idx)
        elif event.button() == Qt.RightButton:
            self._context_menu(event.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.frame_idx)

    def _context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #2d2d2d; color: #fff; } "
            "QMenu::item:selected { background: #404080; }"
        )
        del_action = QAction("削除", self)
        del_action.triggered.connect(lambda: self.delete_requested.emit(self.frame_idx))
        menu.addAction(del_action)
        menu.exec(pos)

    @staticmethod
    def _score_color(score: float) -> str:
        if score >= 0.75:
            return "#2aff2a"
        elif score >= 0.5:
            return "#ffff00"
        return "#ff4444"


# ======================================================================
# メインパネル
# ======================================================================

class KeyframeListWidget(QWidget):
    """
    キーフレームサムネイル一覧パネル

    Signals
    -------
    keyframe_selected : Signal(int)
        サムネイルクリック時
    keyframe_deleted : Signal(int)
        サムネイル削除時
    """

    keyframe_selected = Signal(int)
    keyframe_deleted = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.keyframe_frames: List[int] = []
        self.keyframe_scores: List[float] = []
        self.keyframe_images: dict = {}
        self.video_path: Optional[str] = None
        self.selected_frames: Set[int] = set()

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ツールバー
        tb = QHBoxLayout()
        tb.setSpacing(5)

        tb.addWidget(QLabel("ソート:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["フレーム順", "スコア順"])
        self._sort_combo.setFixedWidth(100)
        self._sort_combo.currentTextChanged.connect(lambda _: self._update_display())
        tb.addWidget(self._sort_combo)

        tb.addSpacing(10)

        self._btn_sel_all = QPushButton("全選択")
        self._btn_sel_all.setFixedWidth(60)
        self._btn_sel_all.clicked.connect(self.select_all)
        tb.addWidget(self._btn_sel_all)

        self._btn_desel = QPushButton("解除")
        self._btn_desel.setFixedWidth(60)
        self._btn_desel.clicked.connect(self.deselect_all)
        tb.addWidget(self._btn_desel)

        self._btn_del = QPushButton("削除")
        self._btn_del.setFixedWidth(60)
        self._btn_del.clicked.connect(self._delete_selected)
        tb.addWidget(self._btn_del)

        tb.addStretch()

        self._count_label = QLabel("0 件")
        self._count_label.setStyleSheet("color: #888;")
        tb.addWidget(self._count_label)

        layout.addLayout(tb)

        # スクロールエリア
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: #1e1e1e; border: 1px solid #3d3d3d; }")

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(5)
        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll, stretch=1)

    # ==================================================================
    # 公開API
    # ==================================================================

    def set_video_path(self, path: str):
        self.video_path = path

    def set_keyframes(self, frames: List[int], scores: List[float]):
        """キーフレーム情報を設定してサムネイルを読み込む"""
        self.keyframe_frames = list(frames)
        self.keyframe_scores = list(scores)
        self.selected_frames.clear()
        self._load_thumbnails()
        self._update_display()

    def get_selected_keyframes(self) -> List[int]:
        return sorted(self.selected_frames)

    def select_all(self):
        self.selected_frames = set(self.keyframe_frames)
        self._update_display()

    def deselect_all(self):
        self.selected_frames.clear()
        self._update_display()

    def clear(self):
        self.keyframe_frames.clear()
        self.keyframe_scores.clear()
        self.keyframe_images.clear()
        self.selected_frames.clear()
        self._update_display()

    # ==================================================================
    # 内部
    # ==================================================================

    def _load_thumbnails(self):
        """ビデオからサムネイル画像を読み込む"""
        if not self.video_path:
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            self.keyframe_images.clear()
            for idx in self.keyframe_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    tw = _KeyframeThumbnail.THUMB_W
                    th = int(h * tw / w)
                    self.keyframe_images[idx] = cv2.resize(frame, (tw, th))
            cap.release()
        except Exception as e:
            logger.exception(f"サムネイル読み込みエラー: {e}")

    def _update_display(self):
        """グリッドを再構築"""
        # クリア
        while self._grid_layout.count() > 0:
            item = self._grid_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        # ソート
        if self._sort_combo.currentText() == "スコア順":
            order = sorted(range(len(self.keyframe_frames)),
                           key=lambda i: self.keyframe_scores[i], reverse=True)
        else:
            order = list(range(len(self.keyframe_frames)))

        col, row = 0, 0
        cols = 2  # 2列
        for i in order:
            fidx = self.keyframe_frames[i]
            score = self.keyframe_scores[i]
            img = self.keyframe_images.get(fidx)
            if img is None:
                continue

            thumb = _KeyframeThumbnail(fidx, img, score, self)
            thumb.clicked.connect(self._on_click)
            thumb.double_clicked.connect(self._on_dbl_click)
            thumb.delete_requested.connect(self._on_delete_one)

            if fidx in self.selected_frames:
                thumb.set_selected(True)

            self._grid_layout.addWidget(thumb, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1

        self._grid_layout.setRowStretch(row + 1, 1)
        self._count_label.setText(f"{len(self.keyframe_frames)} 件")

    def _on_click(self, fidx: int):
        if fidx in self.selected_frames:
            self.selected_frames.discard(fidx)
        else:
            self.selected_frames.add(fidx)
        self._update_display()
        self.keyframe_selected.emit(fidx)

    def _on_dbl_click(self, fidx: int):
        self.keyframe_selected.emit(fidx)

    def _on_delete_one(self, fidx: int):
        if fidx in self.keyframe_frames:
            idx = self.keyframe_frames.index(fidx)
            self.keyframe_frames.pop(idx)
            if idx < len(self.keyframe_scores):
                self.keyframe_scores.pop(idx)
            self.keyframe_images.pop(fidx, None)
            self.selected_frames.discard(fidx)
            self._update_display()
            self.keyframe_deleted.emit(fidx)

    def _delete_selected(self):
        if not self.selected_frames:
            return
        reply = QMessageBox.question(
            self, "確認",
            f"{len(self.selected_frames)} 個のキーフレームを削除しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for fidx in list(self.selected_frames):
                self._on_delete_one(fidx)
