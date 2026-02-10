"""
キーフレームパネル - 360Split GUI
キーフレームサムネイル表示、選択、管理、詳細情報ダイアログ
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout,
    QPushButton, QLabel, QCheckBox, QComboBox, QDialog,
    QMessageBox, QMenu
)
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QIcon, QAction

from utils.logger import get_logger
logger = get_logger(__name__)


class KeyframeDetailDialog(QDialog):
    """
    キーフレーム詳細情報ダイアログ

    選択されたキーフレームの詳細な品質スコアを表示。
    - プレビュー画像
    - 各種スコア（鮮明度、露光、モーションブラー、深度）
    - 幾何学的スコア（GRIC、特徴点分散）
    - 適応スコア（SSIM、モーメンタム）
    - 複合スコア分解図
    """

    def __init__(self, frame_image: np.ndarray, frame_idx: int,
                 quality_dict: dict, parent=None):
        """
        詳細ダイアログの初期化

        Parameters:
        -----------
        frame_image : np.ndarray
            フレーム画像（BGR）
        frame_idx : int
            フレームインデックス
        quality_dict : dict
            品質スコア辞書
        parent : QWidget, optional
            親ウィジェット
        """
        super().__init__(parent)
        self.setWindowTitle(f"キーフレーム詳細 - フレーム {frame_idx}")
        self.setGeometry(100, 100, 600, 700)

        self.frame_image = frame_image
        self.frame_idx = frame_idx
        self.quality_dict = quality_dict

        self._setup_ui()

    def _setup_ui(self):
        """
        UIの構築
        """
        layout = QVBoxLayout(self)

        # === フレーム情報 ===
        info_label = QLabel(f"フレーム #{self.frame_idx}")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        info_label.setFont(font)
        layout.addWidget(info_label)

        # === プレビュー画像 ===
        frame_rgb = cv2.cvtColor(self.frame_image, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = 3 * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaledToWidth(400, Qt.SmoothTransformation)

        preview_label = QLabel()
        preview_label.setPixmap(pixmap)
        preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(preview_label)

        # === スコア詳細 ===
        scores_label = QLabel("品質スコア:")
        font.setPointSize(10)
        scores_label.setFont(font)
        layout.addWidget(scores_label)

        # スコアテーブル
        scores_layout = QVBoxLayout()
        scores_layout.setSpacing(5)

        score_items = [
            ("鮮明度 (Sharpness)", self.quality_dict.get('sharpness', 0.0)),
            ("露光 (Exposure)", self.quality_dict.get('exposure', 0.0)),
            ("モーションブラー (Motion Blur)", 1.0 - self.quality_dict.get('motion_blur', 0.0)),
            ("Softmax深度", self.quality_dict.get('softmax_depth', 0.0)),
        ]

        for label, score in score_items:
            self._add_score_row(scores_layout, label, score)

        layout.addLayout(scores_layout)

        # === 複合スコア ===
        combined_score = (
            self.quality_dict.get('sharpness', 0) * 0.30 +
            self.quality_dict.get('exposure', 0) * 0.15 +
            self.quality_dict.get('softmax_depth', 0) * 0.30 +
            (1.0 - self.quality_dict.get('motion_blur', 0)) * 0.25
        )

        combined_label = QLabel(f"複合スコア (Combined): {combined_score:.3f}")
        font.setPointSize(11)
        font.setBold(True)
        combined_label.setFont(font)
        combined_label.setStyleSheet(f"color: {self._get_score_color(combined_score)};")
        layout.addWidget(combined_label)

        layout.addStretch()

        # === クローズボタン ===
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def _add_score_row(self, layout: QVBoxLayout, label: str, score: float):
        """
        スコア行を追加

        Parameters:
        -----------
        layout : QVBoxLayout
            親レイアウト
        label : str
            ラベル
        score : float
            スコア値
        """
        row_layout = QHBoxLayout()

        label_widget = QLabel(f"{label}: {score:.3f}")
        row_layout.addWidget(label_widget)

        # プログレスバー風の表示
        bar_label = QLabel()
        bar_width = int(score * 200)
        bar_color = self._get_score_color(score)
        bar_label.setFixedSize(200, 20)
        bar_label.setStyleSheet(
            f"background: linear-gradient(to right, {bar_color} 0%, {bar_color} {bar_width}px, "
            f"#3d3d3d {bar_width}px, #3d3d3d 100%); "
            f"border: 1px solid #505050; border-radius: 3px;"
        )
        row_layout.addWidget(bar_label)

        layout.addLayout(row_layout)

    @staticmethod
    def _get_score_color(score: float) -> str:
        """
        スコアに基づいて色を返す

        Parameters:
        -----------
        score : float
            スコア値（0-1）

        Returns:
        --------
        str
            色のHEXコード
        """
        if score >= 0.75:
            return "#2aff2a"  # 緑
        elif score >= 0.5:
            return "#ffff00"  # 黄
        else:
            return "#ff4444"  # 赤


class KeyframeThumbnailWidget(QWidget):
    """
    キーフレームサムネイルウィジェット

    1つのキーフレーム表示とインタラクション処理
    """

    clicked = Signal(int)  # フレームインデックス
    double_clicked = Signal(int)
    delete_requested = Signal(int)

    def __init__(self, frame_idx: int, frame_image: np.ndarray,
                 score: float, video_path: str, parent=None):
        """
        サムネイルウィジェットの初期化

        Parameters:
        -----------
        frame_idx : int
            フレームインデックス
        frame_image : np.ndarray
            フレーム画像（BGR）
        score : float
            キーフレームスコア
        video_path : str
            ビデオファイルパス
        parent : QWidget, optional
            親ウィジェット
        """
        super().__init__(parent)
        self.frame_idx = frame_idx
        self.frame_image = frame_image
        self.score = score
        self.video_path = video_path
        self.is_selected = False

        self.setFixedSize(220, 180)

        self._setup_ui()

    def _setup_ui(self):
        """
        UIの構築
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        # === サムネイル画像 ===
        frame_rgb = cv2.cvtColor(self.frame_image, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = 3 * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaledToWidth(200, Qt.SmoothTransformation)

        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("border: 2px solid #3d3d3d;")
        layout.addWidget(self.image_label)

        # === フレーム情報 ===
        info_layout = QHBoxLayout()

        frame_label = QLabel(f"#{self.frame_idx}")
        font = QFont()
        font.setPointSize(8)
        frame_label.setFont(font)
        info_layout.addWidget(frame_label)

        score_color = self._get_score_color(self.score)
        score_label = QLabel(f"Score: {self.score:.2f}")
        score_label.setFont(font)
        score_label.setStyleSheet(f"color: {score_color};")
        info_layout.addWidget(score_label, alignment=Qt.AlignRight)

        layout.addLayout(info_layout)

        # === チェックボックス ===
        self.checkbox = QCheckBox("選択")
        self.checkbox.setFont(font)
        self.checkbox.toggled.connect(self._on_checkbox_toggled)
        layout.addWidget(self.checkbox)

    def set_selected(self, selected: bool):
        """
        選択状態を設定

        Parameters:
        -----------
        selected : bool
            選択状態
        """
        self.is_selected = selected
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(selected)
        self.checkbox.blockSignals(False)

        if selected:
            self.image_label.setStyleSheet("border: 3px solid #5080d0;")
        else:
            self.image_label.setStyleSheet("border: 2px solid #3d3d3d;")

    def _get_score_color(self, score: float) -> str:
        """
        スコアに基づいて色を返す
        """
        if score >= 0.75:
            return "#2aff2a"
        elif score >= 0.5:
            return "#ffff00"
        else:
            return "#ff4444"

    def mousePressEvent(self, event):
        """
        マウスプレスイベント
        """
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frame_idx)

        elif event.button() == Qt.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, event):
        """
        ダブルクリックイベント
        """
        self.double_clicked.emit(self.frame_idx)

    def _on_checkbox_toggled(self, checked: bool):
        """
        チェックボックストグル時のコールバック
        """
        self.set_selected(checked)

    def _show_context_menu(self, global_pos: QPoint):
        """
        コンテキストメニューを表示

        Parameters:
        -----------
        global_pos : QPoint
            グローバル座標
        """
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2d2d2d; color: #ffffff; }"
            "QMenu::item:selected { background-color: #404080; }"
        )

        delete_action = QAction("削除", self)
        delete_action.triggered.connect(lambda: self.delete_requested.emit(self.frame_idx))
        menu.addAction(delete_action)

        detail_action = QAction("詳細を表示", self)
        detail_action.triggered.connect(self._show_details)
        menu.addAction(detail_action)

        menu.exec(global_pos)  # In PySide6, exec() is used instead of exec_()

    def _show_details(self):
        """
        詳細ダイアログを表示
        """
        # 品質スコア情報を取得
        from core.quality_evaluator import QualityEvaluator
        quality_dict = QualityEvaluator.evaluate(self.frame_image)

        dialog = KeyframeDetailDialog(self.frame_image, self.frame_idx, quality_dict, self.parent())
        dialog.exec()  # In PySide6, exec() is used instead of exec_()


class KeyframePanel(QWidget):
    """
    キーフレーム管理パネル

    キーフレームサムネイルのスクロール表示、選択、ソート、
    管理機能を提供。

    Signals:
    --------
    keyframe_selected : Signal(int)
        キーフレームが選択されたときのシグナル
    keyframe_deleted : Signal(int)
        キーフレームが削除されたときのシグナル
    """

    keyframe_selected = Signal(int)
    keyframe_deleted = Signal(int)

    def __init__(self):
        """
        キーフレームパネルの初期化
        """
        super().__init__()

        self.keyframe_frames: List[int] = []
        self.keyframe_scores: List[float] = []
        self.keyframe_images: dict = {}  # frame_idx -> image
        self.video_path: Optional[str] = None
        self.selected_frames: set = set()

        self._setup_ui()

    def _setup_ui(self):
        """
        UIの構築
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # === ツールバー ===
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(5)

        # ソート選択
        sort_label = QLabel("ソート:")
        toolbar_layout.addWidget(sort_label)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["フレーム順", "スコア順", "タイプ別"])
        self.sort_combo.setMaximumWidth(120)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        toolbar_layout.addWidget(self.sort_combo)

        toolbar_layout.addSpacing(20)

        # すべて選択ボタン
        self.select_all_button = QPushButton("すべて選択")
        self.select_all_button.setMaximumWidth(100)
        self.select_all_button.clicked.connect(self.select_all)
        toolbar_layout.addWidget(self.select_all_button)

        # 選択解除ボタン
        self.deselect_all_button = QPushButton("選択解除")
        self.deselect_all_button.setMaximumWidth(100)
        self.deselect_all_button.clicked.connect(self.deselect_all)
        toolbar_layout.addWidget(self.deselect_all_button)

        toolbar_layout.addSpacing(20)

        # 削除ボタン
        self.delete_button = QPushButton("削除")
        self.delete_button.setMaximumWidth(80)
        self.delete_button.clicked.connect(self._on_delete_selected)
        toolbar_layout.addWidget(self.delete_button)

        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)

        # === キーフレームスクロール領域 ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { background-color: #1e1e1e; border: 1px solid #3d3d3d; }"
        )

        # グリッドレイアウト
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(5)
        scroll_area.setWidget(self.grid_widget)
        layout.addWidget(scroll_area, stretch=1)

    def set_keyframes(self, frames: List[int], scores: List[float]):
        """
        キーフレーム情報を設定

        Parameters:
        -----------
        frames : List[int]
            フレームインデックスリスト
        scores : List[float]
            スコアリスト
        """
        self.keyframe_frames = frames
        self.keyframe_scores = scores
        self._load_thumbnail_images()
        self._update_display()

    def set_video_path(self, video_path: str):
        """
        ビデオパスを設定

        Parameters:
        -----------
        video_path : str
            ビデオファイルパス
        """
        self.video_path = video_path

    def _load_thumbnail_images(self):
        """
        キーフレーム画像を読み込む
        """
        if not self.video_path:
            return

        try:
            cap = cv2.VideoCapture(self.video_path)
            self.keyframe_images.clear()

            for frame_idx in self.keyframe_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # 縮小（メモリ効率化）
                    h, w = frame.shape[:2]
                    scale = 200 / w
                    resized = cv2.resize(frame, (200, int(h * scale)))
                    self.keyframe_images[frame_idx] = resized

            cap.release()

        except Exception as e:
            logger.exception(f"サムネイル読み込みエラー: {e}")

    def _update_display(self):
        """
        グリッド表示を更新
        """
        # 既存ウィジェットをクリア
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        # ソート順序に従って表示
        if self.sort_combo.currentText() == "スコア順":
            indices = sorted(range(len(self.keyframe_frames)),
                           key=lambda i: self.keyframe_scores[i], reverse=True)
        else:
            indices = list(range(len(self.keyframe_frames)))

        # キーフレームウィジェットを追加
        col = 0
        row = 0
        for i in indices:
            frame_idx = self.keyframe_frames[i]
            score = self.keyframe_scores[i]
            frame_image = self.keyframe_images.get(frame_idx)

            if frame_image is None:
                continue

            widget = KeyframeThumbnailWidget(frame_idx, frame_image, score,
                                           self.video_path or "", self)
            widget.clicked.connect(self._on_thumbnail_clicked)
            widget.double_clicked.connect(self._on_thumbnail_double_clicked)
            widget.delete_requested.connect(self._on_thumbnail_delete_requested)

            # 選択状態を復元
            if frame_idx in self.selected_frames:
                widget.set_selected(True)

            self.grid_layout.addWidget(widget, row, col)

            col += 1
            if col >= 3:  # 3列
                col = 0
                row += 1

        # 最後の行にストレッチを設定
        self.grid_layout.setRowStretch(row + 1, 1)

    def get_selected_keyframes(self) -> List[int]:
        """
        選択されたキーフレームを取得

        Returns:
        --------
        List[int]
            選択されたフレームインデックスリスト
        """
        return list(self.selected_frames)

    def select_all(self):
        """
        すべてのキーフレームを選択
        """
        self.selected_frames = set(self.keyframe_frames)
        self._update_display()

    def deselect_all(self):
        """
        すべてのキーフレームの選択を解除
        """
        self.selected_frames.clear()
        self._update_display()

    def clear(self):
        """
        パネルをクリア
        """
        self.keyframe_frames.clear()
        self.keyframe_scores.clear()
        self.keyframe_images.clear()
        self.selected_frames.clear()
        self._update_display()

    # === スロット ===

    def _on_sort_changed(self, text: str):
        """
        ソート方法変更時のコールバック
        """
        self._update_display()

    def _on_thumbnail_clicked(self, frame_idx: int):
        """
        サムネイルがクリックされたときのコールバック
        """
        self.selected_frames.add(frame_idx)
        self._update_display()
        self.keyframe_selected.emit(frame_idx)

    def _on_thumbnail_double_clicked(self, frame_idx: int):
        """
        サムネイルがダブルクリックされたときのコールバック
        """
        self.keyframe_selected.emit(frame_idx)

    def _on_thumbnail_delete_requested(self, frame_idx: int):
        """
        削除が要求されたときのコールバック
        """
        if frame_idx in self.keyframe_frames:
            idx = self.keyframe_frames.index(frame_idx)
            self.keyframe_frames.pop(idx)
            self.keyframe_scores.pop(idx)
            if frame_idx in self.keyframe_images:
                del self.keyframe_images[frame_idx]
            if frame_idx in self.selected_frames:
                self.selected_frames.remove(frame_idx)

            self._update_display()
            self.keyframe_deleted.emit(frame_idx)

    def _on_delete_selected(self):
        """
        選択されたキーフレームを削除
        """
        if not self.selected_frames:
            return

        reply = QMessageBox.question(
            self,
            "確認",
            f"{len(self.selected_frames)}個のキーフレームを削除してもよろしいですか？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for frame_idx in list(self.selected_frames):
                self._on_thumbnail_delete_requested(frame_idx)
