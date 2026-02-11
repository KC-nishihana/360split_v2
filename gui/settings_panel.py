"""
設定パネル (サイドパネル) - 360Split v2 GUI
KeyframeConfig のフィールドをフォーム化し、
スライダー操作でリアルタイムに Live Preview を行う。

SettingsPanel は QWidget として右ドックに配置される（QDialog ではない）。
パラメータ変更時に setting_changed シグナルを発行し、
メインウィンドウ経由で既存スコアデータに対するキーフレーム判定を再実行する。
"""

import json
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QPushButton, QScrollArea,
    QComboBox, QFileDialog
)
from PySide6.QtCore import Qt, Signal

from config import KeyframeConfig, SelectionConfig, WeightsConfig, GRICConfig, Equirect360Config

from utils.logger import get_logger
logger = get_logger(__name__)


class _LinkedSliderSpin(QWidget):
    """スライダーとスピンボックスを連動させるヘルパーウィジェット"""

    valueChanged = Signal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, step: float = 0.01,
                 decimals: int = 2, parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QLabel(label)
        self._label.setMinimumWidth(140)
        self._label.setStyleSheet("color: #ccc;")
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, stretch=1)

        self._spin = QDoubleSpinBox()
        self._spin.setMinimum(min_val)
        self._spin.setMaximum(max_val)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(decimals)
        self._spin.setValue(default)
        self._spin.setFixedWidth(80)
        self._spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self._spin)

    def _on_slider(self, val: int):
        fval = val / self._scale
        self._spin.blockSignals(True)
        self._spin.setValue(fval)
        self._spin.blockSignals(False)
        self.valueChanged.emit(fval)

    def _on_spin(self, val: float):
        self._slider.blockSignals(True)
        self._slider.setValue(int(val * self._scale))
        self._slider.blockSignals(False)
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, val: float):
        self._spin.blockSignals(True)
        self._slider.blockSignals(True)
        self._spin.setValue(val)
        self._slider.setValue(int(val * self._scale))
        self._spin.blockSignals(False)
        self._slider.blockSignals(False)


class SettingsPanel(QWidget):
    """
    パラメータ調整用サイドパネル

    KeyframeConfig のフィールドを自動的にフォーム化。
    スライダー操作時は再解析を走らせず、既存スコアデータに基づき
    キーフレーム判定ロジックだけを再実行する（"Live Preview"）。

    Signals
    -------
    setting_changed : Signal(dict)
        設定が変更された時に発火。dict は変更後の設定全体。
    run_stage2_requested : Signal()
        「詳細解析」ボタンが押された時に発火。
    """

    setting_changed = Signal(dict)
    run_stage2_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = KeyframeConfig()
        self._load_settings()
        self._setup_ui()

    # ==================================================================
    # UI 構築
    # ==================================================================

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # スクロール可能領域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)

        # ---------- 環境プリセット ----------
        grp_preset = QGroupBox("環境プリセット")
        grp_preset.setStyleSheet(self._group_style())
        pl = QVBoxLayout(grp_preset)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("プリセット:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Custom (手動設定)",
            "Outdoor (屋外・高品質)",
            "Indoor (屋内・追跡重視)",
            "Mixed (混合・適応型)"
        ])
        self._preset_combo.setCurrentIndex(0)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, stretch=1)
        pl.addLayout(preset_row)

        # プリセット説明
        self._preset_desc = QLabel("")
        self._preset_desc.setWordWrap(True)
        self._preset_desc.setStyleSheet("color: #999; font-size: 11px; padding: 4px;")
        pl.addWidget(self._preset_desc)

        layout.addWidget(grp_preset)

        # ---------- 品質スコア重み ----------
        grp_weights = QGroupBox("品質スコア重み (α+β+γ+δ=1)")
        grp_weights.setStyleSheet(self._group_style())
        wl = QVBoxLayout(grp_weights)

        self._w_sharpness = _LinkedSliderSpin(
            "鮮明度 (α):", 0.0, 1.0, self._config.weights.alpha)
        self._w_sharpness.valueChanged.connect(self._on_live_change)
        wl.addWidget(self._w_sharpness)

        self._w_geometric = _LinkedSliderSpin(
            "幾何学 (β):", 0.0, 1.0, self._config.weights.beta)
        self._w_geometric.valueChanged.connect(self._on_live_change)
        wl.addWidget(self._w_geometric)

        self._w_content = _LinkedSliderSpin(
            "コンテンツ (γ):", 0.0, 1.0, self._config.weights.gamma)
        self._w_content.valueChanged.connect(self._on_live_change)
        wl.addWidget(self._w_content)

        self._w_exposure = _LinkedSliderSpin(
            "露光 (δ):", 0.0, 1.0, self._config.weights.delta)
        self._w_exposure.valueChanged.connect(self._on_live_change)
        wl.addWidget(self._w_exposure)

        layout.addWidget(grp_weights)

        # ---------- 選択パラメータ ----------
        grp_sel = QGroupBox("選択パラメータ")
        grp_sel.setStyleSheet(self._group_style())
        sl = QVBoxLayout(grp_sel)

        self._laplacian_th = _LinkedSliderSpin(
            "ラプラシアン閾値:", 0.0, 500.0,
            self._config.selection.laplacian_threshold, step=5.0, decimals=0)
        self._laplacian_th.valueChanged.connect(self._on_live_change)
        sl.addWidget(self._laplacian_th)

        self._ssim_th = _LinkedSliderSpin(
            "SSIM変化閾値:", 0.0, 1.0,
            self._config.selection.ssim_change_threshold)
        self._ssim_th.valueChanged.connect(self._on_live_change)
        sl.addWidget(self._ssim_th)

        self._motion_blur_th = _LinkedSliderSpin(
            "モーションブラー閾値:", 0.0, 1.0,
            self._config.selection.motion_blur_threshold)
        self._motion_blur_th.valueChanged.connect(self._on_live_change)
        sl.addWidget(self._motion_blur_th)

        # 最小/最大キーフレーム間隔 (整数)
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("KF間隔 (min/max):"))
        self._min_interval = QSpinBox()
        self._min_interval.setRange(1, 200)
        self._min_interval.setValue(self._config.selection.min_keyframe_interval)
        self._min_interval.valueChanged.connect(self._on_live_change)
        interval_row.addWidget(self._min_interval)

        self._max_interval = QSpinBox()
        self._max_interval.setRange(1, 600)
        self._max_interval.setValue(self._config.selection.max_keyframe_interval)
        self._max_interval.valueChanged.connect(self._on_live_change)
        interval_row.addWidget(self._max_interval)
        sl.addLayout(interval_row)

        layout.addWidget(grp_sel)

        # ---------- GRIC ----------
        grp_gric = QGroupBox("GRIC (幾何学的評価)")
        grp_gric.setStyleSheet(self._group_style())
        gl = QVBoxLayout(grp_gric)

        self._gric_ratio = _LinkedSliderSpin(
            "縮退判定閾値:", 0.5, 1.0,
            self._config.gric.degeneracy_threshold)
        self._gric_ratio.valueChanged.connect(self._on_live_change)
        gl.addWidget(self._gric_ratio)

        self._ransac_th = _LinkedSliderSpin(
            "RANSAC閾値:", 0.5, 10.0,
            self._config.gric.ransac_threshold, step=0.5, decimals=1)
        self._ransac_th.valueChanged.connect(self._on_live_change)
        gl.addWidget(self._ransac_th)

        layout.addWidget(grp_gric)

        # ---------- 360° ----------
        grp_360 = QGroupBox("360° Equirectangular")
        grp_360.setStyleSheet(self._group_style())
        el = QVBoxLayout(grp_360)

        self._use_mask = QCheckBox("天頂/天底ポーラーマスクを有効化")
        self._use_mask.setChecked(self._config.equirect360.enable_polar_mask)
        self._use_mask.toggled.connect(self._on_live_change)
        el.addWidget(self._use_mask)

        self._mask_ratio = _LinkedSliderSpin(
            "マスク比率:", 0.0, 0.30,
            self._config.equirect360.mask_polar_ratio, step=0.01)
        self._mask_ratio.valueChanged.connect(self._on_live_change)
        el.addWidget(self._mask_ratio)

        layout.addWidget(grp_360)

        # ---------- ボタン ----------
        btn_layout = QHBoxLayout()

        self._btn_stage2 = QPushButton("▶ 詳細解析 (Stage 2)")
        self._btn_stage2.setStyleSheet(
            "QPushButton { background: #cc4400; font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background: #ff5500; }"
        )
        self._btn_stage2.clicked.connect(self.run_stage2_requested.emit)
        btn_layout.addWidget(self._btn_stage2)

        self._btn_reset = QPushButton("デフォルトに戻す")
        self._btn_reset.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self._btn_reset)

        layout.addLayout(btn_layout)

        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ==================================================================
    # 設定の読み書き
    # ==================================================================

    def get_config(self) -> KeyframeConfig:
        """現在のパネル値から KeyframeConfig を構築して返す"""
        c = KeyframeConfig()
        c.weights.alpha = self._w_sharpness.value()
        c.weights.beta = self._w_geometric.value()
        c.weights.gamma = self._w_content.value()
        c.weights.delta = self._w_exposure.value()

        c.selection.laplacian_threshold = self._laplacian_th.value()
        c.selection.ssim_change_threshold = self._ssim_th.value()
        c.selection.motion_blur_threshold = self._motion_blur_th.value()
        c.selection.min_keyframe_interval = self._min_interval.value()
        c.selection.max_keyframe_interval = self._max_interval.value()

        c.gric.degeneracy_threshold = self._gric_ratio.value()
        c.gric.ransac_threshold = self._ransac_th.value()

        c.equirect360.enable_polar_mask = self._use_mask.isChecked()
        c.equirect360.mask_polar_ratio = self._mask_ratio.value()

        return c

    def get_selector_dict(self) -> dict:
        """KeyframeSelector 互換の dict を返す"""
        return self.get_config().to_selector_dict()

    def _load_settings(self):
        """~/.360split/settings.json から読み込み"""
        settings_file = Path.home() / ".360split" / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                self._config = KeyframeConfig.from_dict(d)
            except Exception as e:
                logger.warning(f"設定読み込みエラー: {e}")

    def save_settings(self):
        """
        ライブプレビュー専用パネルのため、永続的な保存は行いません。

        永続的な設定保存は settings_dialog (メニュー: 編集 → 設定...) から行ってください。
        このパネルはリアルタイムプレビュー用の一時的なパラメータ調整に使用されます。

        Note:
        -----
        このメソッドは後方互換性のために残されていますが、実際には保存を行いません。
        settings_dialog で保存された設定は reload_from_file() で読み込まれます。
        """
        logger.info("settings_panel は永続的な保存を行いません。settings_dialog を使用してください。")

    def reload_from_file(self):
        """
        settings_dialog で保存された設定を読み込んで UI を更新

        settings_dialog (モーダルダイアログ) で設定が保存された後、
        このメソッドを呼び出すことで settings_panel の UI を同期します。

        Note:
        -----
        このメソッドは main_window から settings_dialog が閉じた後に呼び出されます。
        """
        # 設定ファイルから再読み込み
        self._load_settings()

        # UI ウィジェットを更新（シグナルをブロックして一括更新）
        c = self._config

        # プリセットコンボをCustomにリセット（設定ファイルはCustomとして扱う）
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(0)  # Custom
        self._preset_combo.blockSignals(False)
        self._preset_desc.setText("手動で設定をカスタマイズします。各パラメータを自由に調整できます。")

        # 重み
        self._w_sharpness.setValue(c.weights.alpha)
        self._w_geometric.setValue(c.weights.beta)
        self._w_content.setValue(c.weights.gamma)
        self._w_exposure.setValue(c.weights.delta)

        # 選択パラメータ
        self._laplacian_th.setValue(c.selection.laplacian_threshold)
        self._ssim_th.setValue(c.selection.ssim_change_threshold)
        self._motion_blur_th.setValue(c.selection.motion_blur_threshold)
        self._min_interval.setValue(c.selection.min_keyframe_interval)
        self._max_interval.setValue(c.selection.max_keyframe_interval)

        # GRIC
        self._gric_ratio.setValue(c.gric.degeneracy_threshold)
        self._ransac_th.setValue(c.gric.ransac_threshold)

        # 360度設定
        self._use_mask.setChecked(c.equirect360.enable_polar_mask)
        self._mask_ratio.setValue(c.equirect360.mask_polar_ratio)

        logger.info("settings_dialog からの設定を読み込みました")

        # Live Previewをトリガーして反映
        self._on_live_change()

    # ==================================================================
    # プリセット管理
    # ==================================================================

    def _on_preset_changed(self, index: int):
        """
        プリセット選択変更時のコールバック

        Parameters:
        -----------
        index : int
            選択されたプリセットのインデックス
        """
        from core.config_loader import ConfigManager

        preset_map = {
            0: None,  # Custom
            1: "outdoor",
            2: "indoor",
            3: "mixed"
        }

        preset_id = preset_map.get(index)

        if preset_id is None:
            # Customモード
            self._preset_desc.setText(
                "手動で設定をカスタマイズします。各パラメータを自由に調整できます。"
            )
            return

        try:
            # ConfigManagerでプリセットをロード
            config_manager = ConfigManager()
            preset_info = config_manager.get_preset_info(
                f"{preset_id}_high_quality" if preset_id == "outdoor" else
                f"{preset_id}_robust_tracking" if preset_id == "indoor" else
                f"{preset_id}_adaptive"
            )

            if preset_info is None:
                logger.warning(f"プリセット '{preset_id}' が見つかりません")
                return

            # 説明を表示
            self._preset_desc.setText(preset_info.description)

            # UIの値を更新（シグナルをブロックして一気に更新）
            params = preset_info.parameters

            # 重み
            self._w_sharpness.setValue(params.get('weight_sharpness', 0.30))
            self._w_exposure.setValue(params.get('weight_exposure', 0.15))
            self._w_geometric.setValue(params.get('weight_geometric', 0.30))
            self._w_content.setValue(params.get('weight_content', 0.25))

            # 選択パラメータ
            self._laplacian_th.setValue(params.get('laplacian_threshold', 100.0))
            self._ssim_th.setValue(params.get('ssim_threshold', 0.85))
            self._motion_blur_th.setValue(params.get('motion_blur_threshold', 0.3))
            self._min_interval.setValue(params.get('min_keyframe_interval', 5))
            self._max_interval.setValue(params.get('max_keyframe_interval', 60))

            # GRIC
            self._gric_ratio.setValue(params.get('gric_degeneracy_threshold', 0.85))
            self._ransac_th.setValue(params.get('ransac_threshold', 3.0))

            # 360度設定
            self._use_mask.setChecked(params.get('enable_polar_mask', True))
            self._mask_ratio.setValue(params.get('mask_polar_ratio', 0.10))

            logger.info(f"プリセット '{preset_id}' ({preset_info.name}) を適用しました")

            # Live Previewをトリガー
            self._on_live_change()

        except Exception as e:
            logger.error(f"プリセット適用エラー: {e}")
            self._preset_desc.setText(f"エラー: {e}")

    # ==================================================================
    # Live Preview
    # ==================================================================

    def _on_live_change(self, *_args):
        """
        パラメータ変更時のコールバック。
        再解析は走らせず、setting_changed シグナルを発行して
        メインウィンドウに判定再実行を委譲する。
        """
        self.setting_changed.emit(self.get_selector_dict())

    def _reset_defaults(self):
        """デフォルト値にリセット"""
        d = KeyframeConfig()
        self._w_sharpness.setValue(d.weights.alpha)
        self._w_geometric.setValue(d.weights.beta)
        self._w_content.setValue(d.weights.gamma)
        self._w_exposure.setValue(d.weights.delta)
        self._laplacian_th.setValue(d.selection.laplacian_threshold)
        self._ssim_th.setValue(d.selection.ssim_change_threshold)
        self._motion_blur_th.setValue(d.selection.motion_blur_threshold)
        self._min_interval.setValue(d.selection.min_keyframe_interval)
        self._max_interval.setValue(d.selection.max_keyframe_interval)
        self._gric_ratio.setValue(d.gric.degeneracy_threshold)
        self._ransac_th.setValue(d.gric.ransac_threshold)
        self._use_mask.setChecked(d.equirect360.enable_polar_mask)
        self._mask_ratio.setValue(d.equirect360.mask_polar_ratio)

    # ==================================================================
    # ユーティリティ
    # ==================================================================

    @staticmethod
    def _group_style() -> str:
        return (
            "QGroupBox { "
            "  color: #ddd; font-weight: bold; "
            "  border: 1px solid #3d3d3d; border-radius: 4px; "
            "  margin-top: 8px; padding-top: 16px; "
            "} "
            "QGroupBox::title { "
            "  subcontrol-origin: margin; "
            "  left: 10px; padding: 0 4px; "
            "}"
        )
