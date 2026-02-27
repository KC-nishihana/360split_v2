"""
タイムラインウィジェット - 360Split v2 GUI
pyqtgraph ベースのスコアグラフ + 現在位置 + キーフレームマーカー

表示内容:
  - X軸: フレーム番号 / 時間
  - Y軸: 正規化スコア (0.0 - 1.0)
  - 折れ線: Sharpness(青), GRIC(赤), SSIM変化(緑)
  - マーカー: キーフレーム位置に縦線
  - 操作: クリック/ドラッグでシーク
"""

from typing import List, Optional
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QMenu
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from utils.logger import get_logger
logger = get_logger(__name__)


class TimelineWidget(QWidget):
    """
    pyqtgraph ベースのタイムラインウィジェット

    3種のスコアを折れ線グラフで重ね描き、
    キーフレーム位置に縦線マーカーを表示。
    クリック/ドラッグでビデオプレーヤーと同期シーク。

    Signals
    -------
    positionChanged : Signal(int)
        ユーザー操作によるフレーム位置変更
    keyframeClicked : Signal(int)
        キーフレームマーカーのクリック
    keyframeRemoved : Signal(int)
        キーフレーム削除
    """

    positionChanged = Signal(int)
    keyframeClicked = Signal(int)
    keyframeRemoved = Signal(int)
    keyframeAddRequested = Signal(int)
    keyframeRemoveRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # データ
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_position: int = 0

        # スコアデータ
        self._sharpness_data: np.ndarray = np.array([])
        self._gric_data: np.ndarray = np.array([])
        self._ssim_data: np.ndarray = np.array([])
        self._vo_conf_data: np.ndarray = np.array([])
        self._frame_indices: np.ndarray = np.array([])
        self._vo_conf_low_th: float = 0.35
        self._vo_conf_mid_th: float = 0.55

        # キーフレーム
        self.keyframe_frames: List[int] = []
        self.keyframe_scores: List[float] = []
        self.stationary_ranges: List[tuple[int, int]] = []

        self.setMinimumHeight(180)
        self.setMaximumHeight(250)

        self._plot_widget = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # --- 凡例チェックボックス ---
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(10)

        legend_label = QLabel("スコア表示:")
        legend_label.setStyleSheet("color: #aaa; font-size: 11px;")
        legend_layout.addWidget(legend_label)

        self._cb_sharpness = QCheckBox("Sharpness")
        self._cb_sharpness.setChecked(True)
        self._cb_sharpness.setStyleSheet("color: #4488ff;")
        self._cb_sharpness.toggled.connect(self._update_visibility)
        legend_layout.addWidget(self._cb_sharpness)

        self._cb_gric = QCheckBox("GRIC")
        self._cb_gric.setChecked(True)
        self._cb_gric.setStyleSheet("color: #ff4444;")
        self._cb_gric.toggled.connect(self._update_visibility)
        legend_layout.addWidget(self._cb_gric)

        self._cb_ssim = QCheckBox("SSIM変化")
        self._cb_ssim.setChecked(True)
        self._cb_ssim.setStyleSheet("color: #44cc44;")
        self._cb_ssim.toggled.connect(self._update_visibility)
        legend_layout.addWidget(self._cb_ssim)

        self._cb_vo_conf = QCheckBox("VO信頼度")
        self._cb_vo_conf.setChecked(True)
        self._cb_vo_conf.setStyleSheet("color: #ffaa33;")
        self._cb_vo_conf.toggled.connect(self._update_visibility)
        legend_layout.addWidget(self._cb_vo_conf)

        self._cb_stationary = QCheckBox("停止区間")
        self._cb_stationary.setChecked(True)
        self._cb_stationary.setStyleSheet("color: #aaaaaa;")
        self._cb_stationary.toggled.connect(self._update_visibility)
        legend_layout.addWidget(self._cb_stationary)

        legend_layout.addStretch()

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        legend_layout.addWidget(self._info_label)

        layout.addLayout(legend_layout)

        # --- pyqtgraph プロット ---
        if HAS_PYQTGRAPH:
            self._setup_pyqtgraph(layout)
        else:
            self._setup_fallback(layout)

    # ======================================================================
    # pyqtgraph セットアップ
    # ======================================================================

    def _setup_pyqtgraph(self, parent_layout: QVBoxLayout):
        """pyqtgraph の PlotWidget を構築"""
        pg.setConfigOptions(antialias=True, background='#1a1a2e', foreground='#cccccc')

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setYRange(0, 1.05, padding=0)
        self._plot_widget.setLabel('left', '正規化スコア')
        self._plot_widget.setLabel('bottom', 'フレーム')
        self._plot_widget.getViewBox().setLimits(yMin=-0.05, yMax=1.1)

        # 折れ線プロットアイテム
        self._curve_sharpness = self._plot_widget.plot(
            [], [], pen=pg.mkPen('#4488ff', width=1.5), name='Sharpness'
        )
        self._curve_gric = self._plot_widget.plot(
            [], [], pen=pg.mkPen('#ff4444', width=1.5), name='GRIC'
        )
        self._curve_ssim = self._plot_widget.plot(
            [], [], pen=pg.mkPen('#44cc44', width=1.5), name='SSIM変化'
        )
        self._curve_vo_conf = self._plot_widget.plot(
            [], [], pen=pg.mkPen('#ffaa33', width=1.6), name='VO信頼度'
        )

        # 現在位置の縦線
        self._position_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen('#ffdd00', width=2, style=Qt.SolidLine),
            movable=False
        )
        self._plot_widget.addItem(self._position_line)

        # キーフレームマーカー用リスト
        self._kf_lines: List[pg.InfiniteLine] = []
        self._stationary_regions: List[pg.LinearRegionItem] = []
        self._vo_conf_regions: List[pg.LinearRegionItem] = []

        # マウスイベント接続
        self._plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self._plot_widget.scene().sigMouseMoved.connect(self._on_plot_mouse_moved)

        parent_layout.addWidget(self._plot_widget)

    def _setup_fallback(self, parent_layout: QVBoxLayout):
        """pyqtgraph 無し時のフォールバック"""
        fallback = QLabel(
            "pyqtgraph が未インストールです。\n"
            "pip install pyqtgraph でインストールしてください。"
        )
        fallback.setAlignment(Qt.AlignCenter)
        fallback.setStyleSheet("color: #ff8800; background: #1e1e1e; padding: 20px;")
        parent_layout.addWidget(fallback)

    # ======================================================================
    # 公開API
    # ======================================================================

    def set_duration(self, frame_count: int, fps: float):
        """タイムライン範囲を設定"""
        self.total_frames = frame_count
        self.fps = fps
        if self._plot_widget and HAS_PYQTGRAPH:
            self._plot_widget.setXRange(0, frame_count, padding=0.01)

    def set_position(self, frame_idx: int):
        """現在位置を設定（ビデオプレーヤーから呼ばれる）"""
        if 0 <= frame_idx <= self.total_frames:
            self.current_position = frame_idx
            if self._plot_widget and HAS_PYQTGRAPH:
                self._position_line.setValue(frame_idx)
            self._update_info_label(frame_idx)

    def set_keyframes(self, frames: List[int], scores: List[float]):
        """キーフレーム情報を設定"""
        self.keyframe_frames = list(frames)
        self.keyframe_scores = list(scores)
        self._draw_keyframe_markers()

    def set_stationary_ranges(self, ranges: List[tuple[int, int]]):
        """停止区間レンジを設定"""
        self.stationary_ranges = [(int(s), int(e)) for s, e in ranges if int(e) >= int(s)]
        self._draw_stationary_regions()

    def set_score_data(self, frame_indices: List[int],
                       sharpness: List[float],
                       gric: Optional[List[float]] = None,
                       ssim_change: Optional[List[float]] = None,
                       vo_confidence: Optional[List[float]] = None,
                       vo_confidence_low_threshold: float = 0.35,
                       vo_confidence_mid_threshold: float = 0.55):
        """
        スコアデータを設定してプロットを更新

        Parameters
        ----------
        frame_indices : list[int]
        sharpness : list[float]  正規化済み 0-1
        gric : list[float] or None
        ssim_change : list[float] or None  (1-SSIM)
        """
        n = len(frame_indices)
        self._frame_indices = np.array(frame_indices, dtype=np.float64)
        self._sharpness_data = np.clip(np.array(sharpness[:n], dtype=np.float64), 0, 1)

        if gric is not None and len(gric) == n:
            self._gric_data = np.clip(np.array(gric, dtype=np.float64), 0, 1)
        else:
            self._gric_data = np.array([])

        if ssim_change is not None and len(ssim_change) == n:
            self._ssim_data = np.clip(np.array(ssim_change, dtype=np.float64), 0, 1)
        else:
            self._ssim_data = np.array([])

        if vo_confidence is not None and len(vo_confidence) == n:
            self._vo_conf_data = np.clip(np.array(vo_confidence, dtype=np.float64), 0, 1)
        else:
            self._vo_conf_data = np.array([])
        self._vo_conf_low_th = float(np.clip(vo_confidence_low_threshold, 0.0, 1.0))
        self._vo_conf_mid_th = float(np.clip(vo_confidence_mid_threshold, self._vo_conf_low_th, 1.0))

        self._redraw_curves()
        self._draw_vo_confidence_regions()

    def append_score_batch(self, frame_indices: List[int],
                           sharpness: List[float]):
        """Stage 1 のプログレッシブ結果を追記"""
        fi = np.array(frame_indices, dtype=np.float64)
        sh = np.clip(np.array(sharpness, dtype=np.float64), 0, 1)

        if len(self._frame_indices) == 0:
            self._frame_indices = fi
            self._sharpness_data = sh
        else:
            self._frame_indices = np.concatenate([self._frame_indices, fi])
            self._sharpness_data = np.concatenate([self._sharpness_data, sh])

        self._redraw_curves()

    def set_quality_data(self, scores: List[float]):
        """旧APIとの互換: 品質スコアデータを設定"""
        if not scores:
            return
        n = len(scores)
        self._frame_indices = np.arange(0, n, dtype=np.float64)
        self._sharpness_data = np.clip(np.array(scores, dtype=np.float64), 0, 1)
        self._vo_conf_data = np.array([])
        self._redraw_curves()

    def remove_keyframe(self, frame_idx: int):
        """キーフレームを削除"""
        if frame_idx in self.keyframe_frames:
            idx = self.keyframe_frames.index(frame_idx)
            self.keyframe_frames.pop(idx)
            if idx < len(self.keyframe_scores):
                self.keyframe_scores.pop(idx)
            self._draw_keyframe_markers()
            self.keyframeRemoved.emit(frame_idx)

    # ======================================================================
    # 内部描画
    # ======================================================================

    def _redraw_curves(self):
        """折れ線グラフを更新"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        if len(self._frame_indices) == 0:
            return

        # ダウンサンプリング（表示パフォーマンス）
        max_points = 2000
        step = max(1, len(self._frame_indices) // max_points)

        fi = self._frame_indices[::step]
        sh = self._sharpness_data[::step] if len(self._sharpness_data) > 0 else np.array([])

        if len(sh) == len(fi):
            self._curve_sharpness.setData(fi, sh)

        if len(self._gric_data) > 0:
            gd = self._gric_data[::step]
            if len(gd) == len(fi):
                self._curve_gric.setData(fi, gd)

        if len(self._ssim_data) > 0:
            sd = self._ssim_data[::step]
            if len(sd) == len(fi):
                self._curve_ssim.setData(fi, sd)
        else:
            self._curve_ssim.setData([], [])

        if len(self._vo_conf_data) > 0:
            vd = self._vo_conf_data[::step]
            if len(vd) == len(fi):
                self._curve_vo_conf.setData(fi, vd)
        else:
            self._curve_vo_conf.setData([], [])

    def _draw_keyframe_markers(self):
        """キーフレーム位置の縦線を描画"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return

        # 既存マーカーを削除
        for line in self._kf_lines:
            self._plot_widget.removeItem(line)
        self._kf_lines.clear()

        # 新しいマーカーを追加
        for i, frame_idx in enumerate(self.keyframe_frames):
            score = self.keyframe_scores[i] if i < len(self.keyframe_scores) else 0.5

            if score >= 0.75:
                color = '#44ff44'
            elif score >= 0.5:
                color = '#ffff00'
            else:
                color = '#ff4444'

            line = pg.InfiniteLine(
                pos=frame_idx, angle=90,
                pen=pg.mkPen(color, width=1, style=Qt.DashLine),
                movable=False
            )
            self._plot_widget.addItem(line)
            self._kf_lines.append(line)

    def _draw_stationary_regions(self):
        """停止区間の帯を描画"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return

        for region in self._stationary_regions:
            self._plot_widget.removeItem(region)
        self._stationary_regions.clear()

        for start, end in self.stationary_ranges:
            region = pg.LinearRegionItem(
                values=(float(start), float(end)),
                orientation='vertical',
                movable=False,
                brush=(170, 170, 170, 50),
                pen=pg.mkPen((180, 180, 180, 90), width=1),
            )
            region.setZValue(-50)
            self._plot_widget.addItem(region)
            self._stationary_regions.append(region)

    def _draw_vo_confidence_regions(self):
        """VO低信頼度区間を帯で描画"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        for region in self._vo_conf_regions:
            self._plot_widget.removeItem(region)
        self._vo_conf_regions.clear()
        if len(self._frame_indices) == 0 or len(self._vo_conf_data) != len(self._frame_indices):
            return

        def _append_regions(mask: np.ndarray, brush, pen) -> None:
            start = None
            for i, flag in enumerate(mask.tolist()):
                if flag and start is None:
                    start = i
                if (not flag) and start is not None:
                    s = float(self._frame_indices[start])
                    e = float(self._frame_indices[i - 1] + 1.0)
                    region = pg.LinearRegionItem(values=(s, e), orientation='vertical', movable=False, brush=brush, pen=pen)
                    region.setZValue(-60)
                    self._plot_widget.addItem(region)
                    self._vo_conf_regions.append(region)
                    start = None
            if start is not None:
                s = float(self._frame_indices[start])
                e = float(self._frame_indices[-1] + 1.0)
                region = pg.LinearRegionItem(values=(s, e), orientation='vertical', movable=False, brush=brush, pen=pen)
                region.setZValue(-60)
                self._plot_widget.addItem(region)
                self._vo_conf_regions.append(region)

        low_mask = self._vo_conf_data < self._vo_conf_low_th
        mid_mask = np.logical_and(self._vo_conf_data >= self._vo_conf_low_th, self._vo_conf_data < self._vo_conf_mid_th)
        _append_regions(low_mask, (220, 70, 70, 45), pg.mkPen((220, 70, 70, 80), width=1))
        _append_regions(mid_mask, (255, 170, 70, 35), pg.mkPen((255, 170, 70, 70), width=1))

    def _update_visibility(self):
        """チェックボックスに基づくカーブの表示/非表示"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        self._curve_sharpness.setVisible(self._cb_sharpness.isChecked())
        self._curve_gric.setVisible(self._cb_gric.isChecked())
        self._curve_ssim.setVisible(self._cb_ssim.isChecked())
        self._curve_vo_conf.setVisible(self._cb_vo_conf.isChecked())
        show_stationary = self._cb_stationary.isChecked()
        for region in self._stationary_regions:
            region.setVisible(show_stationary)
        show_vo_conf = self._cb_vo_conf.isChecked()
        for region in self._vo_conf_regions:
            region.setVisible(show_vo_conf)

    def _update_info_label(self, frame_idx: int):
        """情報ラベルを更新"""
        if self.fps <= 0:
            return
        sec = frame_idx / self.fps
        minutes = int(sec // 60)
        secs = sec % 60
        extra = ""
        if len(self._frame_indices) > 0 and len(self._vo_conf_data) == len(self._frame_indices):
            i = int(np.argmin(np.abs(self._frame_indices - float(frame_idx))))
            if 0 <= i < len(self._vo_conf_data):
                extra = f" | VO {float(self._vo_conf_data[i]):.2f}"
        self._info_label.setText(
            f"Frame {frame_idx}  |  {minutes:02d}:{secs:05.2f}{extra}"
        )

    # ======================================================================
    # マウスイベント
    # ======================================================================

    def _on_plot_clicked(self, event):
        """プロットエリアのクリック→シーク"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        try:
            pos = event.scenePos()
            mouse_point = self._plot_widget.getViewBox().mapSceneToView(pos)
            frame_idx = int(round(mouse_point.x()))
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))

            if event.button() == Qt.MouseButton.RightButton:
                global_pos = event.screenPos().toPoint() if hasattr(event, "screenPos") else pos.toPoint()
                self._show_context_menu(frame_idx, global_pos)
                return

            # キーフレーム近傍チェック（±3フレーム）
            for kf_idx in self.keyframe_frames:
                if abs(frame_idx - kf_idx) <= 3:
                    self.keyframeClicked.emit(kf_idx)
                    return

            self.set_position(frame_idx)
            self.positionChanged.emit(frame_idx)
        except Exception:
            pass

    def _on_plot_mouse_moved(self, pos):
        """マウス移動時のホバー情報"""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        try:
            mouse_point = self._plot_widget.getViewBox().mapSceneToView(pos)
            frame_idx = int(round(mouse_point.x()))
            if 0 <= frame_idx < self.total_frames:
                self._update_info_label(frame_idx)
        except Exception:
            pass

    def _show_context_menu(self, frame_idx: int, global_pos):
        menu = QMenu(self)
        near_kf = None
        for kf_idx in self.keyframe_frames:
            if abs(int(kf_idx) - int(frame_idx)) <= 2:
                near_kf = int(kf_idx)
                break
        if near_kf is None:
            add_action = QAction(f"キーフレーム追加 ({frame_idx})", self)
            add_action.triggered.connect(lambda: self.keyframeAddRequested.emit(int(frame_idx)))
            menu.addAction(add_action)
        else:
            remove_action = QAction(f"キーフレーム除外 ({near_kf})", self)
            remove_action.triggered.connect(lambda: self.keyframeRemoveRequested.emit(int(near_kf)))
            menu.addAction(remove_action)
        menu.exec(global_pos)
