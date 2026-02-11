"""
エクスポートダイアログ - 360Split v2

キーフレーム画像のエクスポート設定を行うモーダルダイアログ。
出力投影モード（Original / Cubemap / Perspective）、画像フォーマット、
360度処理、マスク処理の設定を一画面で行える。

出力ディレクトリ構造:
  <export_dir>/
  ├── keyframe_000123.png        # 元画像（常に出力）
  ├── cubemap/
  │   └── frame_000123/
  │       ├── front.png
  │       ├── back.png
  │       ├── left.png
  │       ├── right.png
  │       ├── up.png
  │       └── down.png
  └── perspective/
      └── frame_000123/
          ├── y+0_p+0.png
          ├── y+90_p+0.png
          └── ...
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox,
    QLineEdit, QFileDialog, QDialogButtonBox,
    QTabWidget, QWidget, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt

from utils.logger import get_logger
logger = get_logger(__name__)

# デフォルトの Perspective ヨー角プリセット
_PERSPECTIVE_PRESETS = {
    "4方向 (0°/90°/180°/270°)": ([0.0, 90.0, 180.0, -90.0], [0.0]),
    "6方向 (Cubemapと同等)": ([0.0, 90.0, 180.0, -90.0], [90.0, -90.0]),
    "8方向 (45°刻み)": ([0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0], [0.0]),
    "正面のみ": ([0.0], [0.0]),
    "カスタム": (None, None),
}


class ExportDialog(QDialog):
    """
    エクスポート設定ダイアログ

    キーフレーム画像の出力形式・投影モード・前処理を設定する。
    OK を押すと get_settings() で全設定を dict で取得できる。
    """

    def __init__(self, parent=None, num_keyframes: int = 0):
        super().__init__(parent)
        self.setWindowTitle("キーフレームエクスポート設定")
        self.setMinimumWidth(520)
        self.setMinimumHeight(600)
        self._num_keyframes = num_keyframes

        self._setup_ui()
        self._load_last_settings()
        self._update_ui_state()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ヘッダー情報
        info = QLabel(f"エクスポート対象: {self._num_keyframes} キーフレーム")
        info.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(info)

        # タブウィジェット
        tabs = QTabWidget()
        tabs.addTab(self._create_output_tab(), "📁 出力設定")
        tabs.addTab(self._create_projection_tab(), "🌐 投影変換")
        tabs.addTab(self._create_preprocess_tab(), "🔧 前処理")
        layout.addWidget(tabs)

        # ボタン
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # タブ 1: 出力設定
    # ------------------------------------------------------------------

    def _create_output_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 出力ディレクトリ
        dir_group = QGroupBox("出力先")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_edit = QLineEdit(str(Path.home() / "360split_output"))
        dir_layout.addWidget(self.dir_edit)
        browse_btn = QPushButton("参照...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(browse_btn)
        layout.addWidget(dir_group)

        # 画像フォーマット
        fmt_group = QGroupBox("画像設定")
        fmt_layout = QGridLayout(fmt_group)

        fmt_layout.addWidget(QLabel("フォーマット:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "tiff"])
        self.format_combo.currentTextChanged.connect(self._update_ui_state)
        fmt_layout.addWidget(self.format_combo, 0, 1)

        fmt_layout.addWidget(QLabel("JPEG品質:"), 1, 0)
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(1, 100)
        self.jpeg_quality_spin.setValue(95)
        fmt_layout.addWidget(self.jpeg_quality_spin, 1, 1)

        fmt_layout.addWidget(QLabel("ファイル名接頭辞:"), 2, 0)
        self.prefix_edit = QLineEdit("keyframe")
        fmt_layout.addWidget(self.prefix_edit, 2, 1)

        layout.addWidget(fmt_group)
        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # タブ 2: 投影変換
    # ------------------------------------------------------------------

    def _create_projection_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- Cubemap ---
        cubemap_group = QGroupBox("Cubemap 出力")
        cubemap_group.setCheckable(True)
        cubemap_group.setChecked(False)
        self.cubemap_group = cubemap_group
        cb_layout = QGridLayout(cubemap_group)

        cb_layout.addWidget(QLabel("面サイズ (px):"), 0, 0)
        self.cubemap_face_spin = QSpinBox()
        self.cubemap_face_spin.setRange(128, 8192)
        self.cubemap_face_spin.setSingleStep(128)
        self.cubemap_face_spin.setValue(1024)
        cb_layout.addWidget(self.cubemap_face_spin, 0, 1)

        cb_layout.addWidget(QLabel(
            "6面 (front/back/left/right/up/down) を\n"
            "フレームごとにサブフォルダへ出力します"
        ), 1, 0, 1, 2)

        layout.addWidget(cubemap_group)

        # --- Perspective ---
        persp_group = QGroupBox("Perspective（透視投影）出力")
        persp_group.setCheckable(True)
        persp_group.setChecked(False)
        self.persp_group = persp_group
        pe_layout = QGridLayout(persp_group)

        pe_layout.addWidget(QLabel("FOV (°):"), 0, 0)
        self.persp_fov_spin = QDoubleSpinBox()
        self.persp_fov_spin.setRange(30.0, 170.0)
        self.persp_fov_spin.setSingleStep(5.0)
        self.persp_fov_spin.setValue(90.0)
        pe_layout.addWidget(self.persp_fov_spin, 0, 1)

        pe_layout.addWidget(QLabel("出力サイズ (px):"), 1, 0)
        self.persp_size_spin = QSpinBox()
        self.persp_size_spin.setRange(128, 4096)
        self.persp_size_spin.setSingleStep(128)
        self.persp_size_spin.setValue(1024)
        pe_layout.addWidget(self.persp_size_spin, 1, 1)

        pe_layout.addWidget(QLabel("方向プリセット:"), 2, 0)
        self.persp_preset_combo = QComboBox()
        for name in _PERSPECTIVE_PRESETS.keys():
            self.persp_preset_combo.addItem(name)
        self.persp_preset_combo.currentTextChanged.connect(self._on_persp_preset)
        pe_layout.addWidget(self.persp_preset_combo, 2, 1)

        # カスタムヨー/ピッチ入力
        pe_layout.addWidget(QLabel("ヨー角 (°):"), 3, 0)
        self.persp_yaw_edit = QLineEdit("0, 90, 180, -90")
        self.persp_yaw_edit.setToolTip("カンマ区切りで角度を指定（例: 0, 90, 180, -90）")
        pe_layout.addWidget(self.persp_yaw_edit, 3, 1)

        pe_layout.addWidget(QLabel("ピッチ角 (°):"), 4, 0)
        self.persp_pitch_edit = QLineEdit("0")
        self.persp_pitch_edit.setToolTip("カンマ区切りで角度を指定（例: 0, 30, -30）")
        pe_layout.addWidget(self.persp_pitch_edit, 4, 1)

        # 出力概算
        self.persp_count_label = QLabel("")
        self.persp_count_label.setStyleSheet("color: #88aaff; font-style: italic;")
        pe_layout.addWidget(self.persp_count_label, 5, 0, 1, 2)

        # 入力変更時に概算を更新
        self.persp_yaw_edit.textChanged.connect(self._update_persp_count)
        self.persp_pitch_edit.textChanged.connect(self._update_persp_count)

        layout.addWidget(persp_group)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # タブ 3: 前処理
    # ------------------------------------------------------------------

    def _create_preprocess_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Equirectangular リサイズ
        equirect_group = QGroupBox("Equirectangular リサイズ")
        equirect_group.setCheckable(True)
        equirect_group.setChecked(False)
        self.equirect_group = equirect_group
        eq_layout = QGridLayout(equirect_group)

        eq_layout.addWidget(QLabel("幅 (px):"), 0, 0)
        self.equirect_w_spin = QSpinBox()
        self.equirect_w_spin.setRange(512, 16384)
        self.equirect_w_spin.setSingleStep(256)
        self.equirect_w_spin.setValue(4096)
        eq_layout.addWidget(self.equirect_w_spin, 0, 1)

        eq_layout.addWidget(QLabel("高さ (px):"), 1, 0)
        self.equirect_h_spin = QSpinBox()
        self.equirect_h_spin.setRange(256, 8192)
        self.equirect_h_spin.setSingleStep(128)
        self.equirect_h_spin.setValue(2048)
        eq_layout.addWidget(self.equirect_h_spin, 1, 1)

        layout.addWidget(equirect_group)

        # ポーラーマスク
        polar_group = QGroupBox("ポーラーマスク（天頂/天底黒塗り）")
        polar_group.setCheckable(True)
        polar_group.setChecked(False)
        self.polar_group = polar_group
        pl_layout = QGridLayout(polar_group)

        pl_layout.addWidget(QLabel("マスク比率:"), 0, 0)
        self.polar_ratio_spin = QDoubleSpinBox()
        self.polar_ratio_spin.setRange(0.01, 0.50)
        self.polar_ratio_spin.setSingleStep(0.01)
        self.polar_ratio_spin.setValue(0.10)
        pl_layout.addWidget(self.polar_ratio_spin, 0, 1)

        layout.addWidget(polar_group)

        # ナディアマスク
        nadir_group = QGroupBox("ナディアマスク（円形マスク）")
        nadir_group.setCheckable(True)
        nadir_group.setChecked(False)
        self.nadir_group = nadir_group
        nd_layout = QGridLayout(nadir_group)

        nd_layout.addWidget(QLabel("半径 (px):"), 0, 0)
        self.nadir_radius_spin = QSpinBox()
        self.nadir_radius_spin.setRange(10, 1000)
        self.nadir_radius_spin.setValue(100)
        nd_layout.addWidget(self.nadir_radius_spin, 0, 1)

        layout.addWidget(nadir_group)

        # 装備検出マスク
        equip_group = QGroupBox("装備検出マスク（下部領域）")
        equip_group.setCheckable(True)
        equip_group.setChecked(False)
        self.equip_group = equip_group
        ep_layout = QGridLayout(equip_group)

        ep_layout.addWidget(QLabel("膨張サイズ (px):"), 0, 0)
        self.equip_dilation_spin = QSpinBox()
        self.equip_dilation_spin.setRange(0, 100)
        self.equip_dilation_spin.setValue(15)
        ep_layout.addWidget(self.equip_dilation_spin, 0, 1)

        layout.addWidget(equip_group)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # イベントハンドラ
    # ------------------------------------------------------------------

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "出力先を選択", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _update_ui_state(self):
        is_jpg = self.format_combo.currentText() in ('jpg', 'jpeg')
        self.jpeg_quality_spin.setEnabled(is_jpg)
        self._update_persp_count()

    def _on_persp_preset(self, name: str):
        yaws, pitches = _PERSPECTIVE_PRESETS.get(name, (None, None))
        if yaws is not None:
            self.persp_yaw_edit.setText(", ".join(str(y) for y in yaws))
            self.persp_pitch_edit.setText(", ".join(str(p) for p in pitches))
            self.persp_yaw_edit.setEnabled(False)
            self.persp_pitch_edit.setEnabled(False)
        else:
            self.persp_yaw_edit.setEnabled(True)
            self.persp_pitch_edit.setEnabled(True)

    def _update_persp_count(self):
        yaws = self._parse_float_list(self.persp_yaw_edit.text())
        pitches = self._parse_float_list(self.persp_pitch_edit.text())
        n = len(yaws) * len(pitches) * self._num_keyframes
        self.persp_count_label.setText(
            f"→ {len(yaws)}×{len(pitches)} 方向 × "
            f"{self._num_keyframes} KF = {n} 画像"
        )

    # ------------------------------------------------------------------
    # 設定の取得 / 保存
    # ------------------------------------------------------------------

    def get_settings(self) -> Dict[str, Any]:
        """ダイアログの全設定を辞書で返す"""
        yaws = self._parse_float_list(self.persp_yaw_edit.text())
        pitches = self._parse_float_list(self.persp_pitch_edit.text())
        sz = self.persp_size_spin.value()

        return {
            # 出力先
            "output_dir": self.dir_edit.text(),
            "output_format": self.format_combo.currentText(),
            "jpeg_quality": self.jpeg_quality_spin.value(),
            "prefix": self.prefix_edit.text(),

            # Cubemap
            "enable_cubemap": self.cubemap_group.isChecked(),
            "cubemap_face_size": self.cubemap_face_spin.value(),

            # Perspective
            "enable_perspective": self.persp_group.isChecked(),
            "perspective_fov": self.persp_fov_spin.value(),
            "perspective_yaw_list": yaws,
            "perspective_pitch_list": pitches,
            "perspective_size": (sz, sz),

            # Equirectangular リサイズ
            "enable_equirect": self.equirect_group.isChecked(),
            "equirect_width": self.equirect_w_spin.value(),
            "equirect_height": self.equirect_h_spin.value(),

            # ポーラーマスク
            "enable_polar_mask": self.polar_group.isChecked(),
            "mask_polar_ratio": self.polar_ratio_spin.value(),

            # ナディアマスク
            "enable_nadir_mask": self.nadir_group.isChecked(),
            "nadir_mask_radius": self.nadir_radius_spin.value(),

            # 装備検出
            "enable_equipment_detection": self.equip_group.isChecked(),
            "mask_dilation_size": self.equip_dilation_spin.value(),
        }

    def _on_accept(self):
        """OK 時に設定を永続化"""
        self._save_last_settings()
        self.accept()

    def _save_last_settings(self):
        settings = self.get_settings()
        # perspective_size はタプルなので JSON 化のためリストに変換
        settings["perspective_size"] = list(settings["perspective_size"])
        save_dir = Path.home() / ".360split"
        save_dir.mkdir(exist_ok=True)
        path = save_dir / "export_settings.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"エクスポート設定保存失敗: {e}")

    def _load_last_settings(self):
        # 1) グローバル設定（settings_dialog の保存先）をベースに適用
        # 2) 直近エクスポート設定があれば上書き
        settings_dir = Path.home() / ".360split"
        global_path = settings_dir / "settings.json"
        export_path = settings_dir / "export_settings.json"

        try:
            if global_path.exists():
                with open(global_path, 'r', encoding='utf-8') as f:
                    g = json.load(f)

                projection_mode = str(g.get("projection_mode", "Equirectangular"))
                enable_cubemap = projection_mode == "Cubemap"
                enable_perspective = projection_mode == "Perspective"

                global_export_defaults = {
                    "output_dir": g.get("output_directory", self.dir_edit.text()),
                    "output_format": g.get("output_image_format", "png"),
                    "jpeg_quality": g.get("output_jpeg_quality", 95),
                    "prefix": g.get("naming_prefix", "keyframe"),
                    "enable_equirect": True,
                    "equirect_width": g.get("equirect_width", 4096),
                    "equirect_height": g.get("equirect_height", 2048),
                    "enable_polar_mask": g.get("enable_polar_mask", False),
                    "mask_polar_ratio": g.get("mask_polar_ratio", 0.10),
                    "enable_nadir_mask": g.get("enable_nadir_mask", False),
                    "nadir_mask_radius": g.get("nadir_mask_radius", 100),
                    "enable_equipment_detection": g.get("enable_equipment_detection", False),
                    "mask_dilation_size": g.get("mask_dilation_size", 15),
                    "enable_cubemap": enable_cubemap,
                    "enable_perspective": enable_perspective,
                    "perspective_fov": g.get("perspective_fov", 90.0),
                }
                self._apply_settings_dict(global_export_defaults)

            if export_path.exists():
                with open(export_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                self._apply_settings_dict(s)

        except Exception as e:
            logger.warning(f"エクスポート設定読み込み失敗: {e}")

    def _apply_settings_dict(self, s: Dict[str, Any]):
        """保存済み辞書を UI に反映する共通処理"""
        self.dir_edit.setText(s.get("output_dir", self.dir_edit.text()))
        idx = self.format_combo.findText(str(s.get("output_format", "png")).lower())
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.jpeg_quality_spin.setValue(int(s.get("jpeg_quality", 95)))
        self.prefix_edit.setText(s.get("prefix", "keyframe"))

        # Cubemap
        self.cubemap_group.setChecked(bool(s.get("enable_cubemap", False)))
        self.cubemap_face_spin.setValue(int(s.get("cubemap_face_size", 1024)))

        # Perspective
        self.persp_group.setChecked(bool(s.get("enable_perspective", False)))
        self.persp_fov_spin.setValue(float(s.get("perspective_fov", 90.0)))
        yaw_list = s.get("perspective_yaw_list", [0.0, 90.0, 180.0, -90.0])
        pitch_list = s.get("perspective_pitch_list", [0.0])
        self.persp_yaw_edit.setText(", ".join(str(y) for y in yaw_list))
        self.persp_pitch_edit.setText(", ".join(str(p) for p in pitch_list))
        sz = s.get("perspective_size", [1024, 1024])
        if isinstance(sz, (list, tuple)) and len(sz) >= 1:
            self.persp_size_spin.setValue(int(sz[0]))

        # Equirect
        self.equirect_group.setChecked(bool(s.get("enable_equirect", False)))
        self.equirect_w_spin.setValue(int(s.get("equirect_width", 4096)))
        self.equirect_h_spin.setValue(int(s.get("equirect_height", 2048)))

        # マスク
        self.polar_group.setChecked(bool(s.get("enable_polar_mask", False)))
        self.polar_ratio_spin.setValue(float(s.get("mask_polar_ratio", 0.10)))
        self.nadir_group.setChecked(bool(s.get("enable_nadir_mask", False)))
        self.nadir_radius_spin.setValue(int(s.get("nadir_mask_radius", 100)))
        self.equip_group.setChecked(bool(s.get("enable_equipment_detection", False)))
        self.equip_dilation_spin.setValue(int(s.get("mask_dilation_size", 15)))

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_float_list(text: str) -> List[float]:
        """カンマ区切り文字列を float リストに変換"""
        result = []
        for part in text.split(","):
            part = part.strip()
            if part:
                try:
                    result.append(float(part))
                except ValueError:
                    pass
        return result if result else [0.0]
