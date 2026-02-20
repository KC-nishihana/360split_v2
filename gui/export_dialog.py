"""
エクスポートダイアログ - 360Split v2

実行時に必要な設定のみを扱い、共通パラメータは settings_dialog で一元管理する。
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QDialogButtonBox, QTabWidget, QWidget
)

from gui.settings_dialog import SettingsDialog
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

    変更可能:
    - 出力先
    - Cubemap/Perspective 追加出力

    共通設定:
    - 形式/品質/命名
    - 前処理/マスク/対象検出
    これらは settings_dialog で一元管理する。
    """

    def __init__(self, parent=None, num_keyframes: int = 0):
        super().__init__(parent)
        self.setWindowTitle("キーフレームエクスポート設定")
        self.setMinimumWidth(680)
        self.setMinimumHeight(760)
        self._num_keyframes = num_keyframes
        self._global_settings: Dict[str, Any] = {}

        self._setup_ui()
        self._load_last_settings()
        self._load_global_settings()
        self._update_ui_state()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(f"エクスポート対象: {self._num_keyframes} キーフレーム")
        info.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(info)

        tabs = QTabWidget()
        tabs.addTab(self._create_output_tab(), "📁 実行設定")
        tabs.addTab(self._create_projection_tab(), "🌐 追加出力")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_output_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        dir_group = QGroupBox("出力先")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_edit = QLineEdit(str(Path.home() / "360split_output"))
        dir_layout.addWidget(self.dir_edit)
        browse_btn = QPushButton("参照...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(browse_btn)
        layout.addWidget(dir_group)

        naming_group = QGroupBox("ファイル名")
        naming_layout = QGridLayout(naming_group)
        naming_layout.addWidget(QLabel("接頭辞:"), 0, 0)
        self.prefix_edit = QLineEdit("keyframe")
        self.prefix_edit.setToolTip("例: keyframe -> keyframe_000123.png")
        naming_layout.addWidget(self.prefix_edit, 0, 1)
        layout.addWidget(naming_group)

        global_group = QGroupBox("共通設定（設定ダイアログで管理）")
        global_layout = QVBoxLayout(global_group)
        note = QLabel(
            "形式・品質・命名・前処理・マスク設定は、"
            "「編集 > 設定...」で変更してください。"
        )
        note.setWordWrap(True)
        global_layout.addWidget(note)

        self.global_summary_label = QLabel("")
        self.global_summary_label.setWordWrap(True)
        self.global_summary_label.setStyleSheet("color: #dddddd;")
        global_layout.addWidget(self.global_summary_label)

        open_settings_btn = QPushButton("設定ダイアログを開く...")
        open_settings_btn.clicked.connect(self._open_global_settings)
        global_layout.addWidget(open_settings_btn)
        layout.addWidget(global_group)

        stereo_group = QGroupBox("LR入力時の実行オプション")
        stereo_layout = QVBoxLayout(stereo_group)
        self.stereo_stitch_check = QCheckBox("L/R をステッチして1枚として出力する")
        self.stereo_stitch_check.setChecked(True)
        self.stereo_stitch_check.setToolTip("OFF時は L/ と R/ フォルダに分けて出力します")
        stereo_layout.addWidget(self.stereo_stitch_check)
        layout.addWidget(stereo_group)

        layout.addStretch()
        return widget

    def _create_projection_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

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

        pe_layout.addWidget(QLabel("ヨー角 (°):"), 3, 0)
        self.persp_yaw_edit = QLineEdit("0, 90, 180, -90")
        self.persp_yaw_edit.setToolTip("カンマ区切りで角度を指定（例: 0, 90, 180, -90）")
        pe_layout.addWidget(self.persp_yaw_edit, 3, 1)

        pe_layout.addWidget(QLabel("ピッチ角 (°):"), 4, 0)
        self.persp_pitch_edit = QLineEdit("0")
        self.persp_pitch_edit.setToolTip("カンマ区切りで角度を指定（例: 0, 30, -30）")
        pe_layout.addWidget(self.persp_pitch_edit, 4, 1)

        self.persp_count_label = QLabel("")
        self.persp_count_label.setStyleSheet("color: #88aaff; font-style: italic;")
        pe_layout.addWidget(self.persp_count_label, 5, 0, 1, 2)

        self.persp_yaw_edit.textChanged.connect(self._update_persp_count)
        self.persp_pitch_edit.textChanged.connect(self._update_persp_count)

        layout.addWidget(persp_group)
        layout.addStretch()
        return widget

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "出力先を選択", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _update_ui_state(self):
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

    def get_settings(self) -> Dict[str, Any]:
        """ダイアログの全設定を辞書で返す"""
        self._load_global_settings()

        yaws = self._parse_float_list(self.persp_yaw_edit.text())
        pitches = self._parse_float_list(self.persp_pitch_edit.text())
        sz = self.persp_size_spin.value()
        g = self._global_settings

        return {
            "output_dir": self.dir_edit.text(),
            "output_format": str(g.get("output_image_format", "png")).lower(),
            "jpeg_quality": int(g.get("output_jpeg_quality", 95)),
            "prefix": self.prefix_edit.text().strip() or str(g.get("naming_prefix", "keyframe")),
            "enable_cubemap": self.cubemap_group.isChecked(),
            "cubemap_face_size": self.cubemap_face_spin.value(),
            "enable_perspective": self.persp_group.isChecked(),
            "perspective_fov": self.persp_fov_spin.value(),
            "perspective_yaw_list": yaws,
            "perspective_pitch_list": pitches,
            "perspective_size": (sz, sz),
            "enable_equirect": True,
            "equirect_width": int(g.get("equirect_width", 4096)),
            "equirect_height": int(g.get("equirect_height", 2048)),
            "enable_stereo_stitch": self.stereo_stitch_check.isChecked(),
            "stitching_mode": str(g.get("stitching_mode", "Fast")),
            "enable_polar_mask": bool(g.get("enable_polar_mask", False)),
            "mask_polar_ratio": float(g.get("mask_polar_ratio", 0.10)),
            "enable_nadir_mask": bool(g.get("enable_nadir_mask", False)),
            "nadir_mask_radius": int(g.get("nadir_mask_radius", 100)),
            "enable_equipment_detection": bool(g.get("enable_equipment_detection", False)),
            "mask_dilation_size": int(g.get("mask_dilation_size", 15)),
            "enable_target_mask_generation": bool(g.get("enable_target_mask_generation", False)),
            "target_classes": list(g.get("target_classes", ["人物", "人", "自転車", "バイク", "車両", "動物"])),
            "enable_dynamic_mask_removal": bool(g.get("enable_dynamic_mask_removal", False)),
            "dynamic_mask_use_yolo_sam": bool(g.get("dynamic_mask_use_yolo_sam", True)),
            "dynamic_mask_target_classes": list(
                g.get("dynamic_mask_target_classes", g.get("target_classes", ["人物", "人", "自転車", "バイク", "車両", "動物"]))
            ),
            "yolo_model_path": str(g.get("yolo_model_path", "yolo26n-seg.pt")),
            "sam_model_path": str(g.get("sam_model_path", "sam3_t.pt")),
            "confidence_threshold": float(g.get("confidence_threshold", 0.25)),
            "detection_device": str(g.get("detection_device", "auto")),
            "mask_output_dirname": str(g.get("mask_output_dirname", "masks")),
            "mask_add_suffix": bool(g.get("mask_add_suffix", True)),
            "mask_suffix": str(g.get("mask_suffix", "_mask")),
            "mask_output_format": str(g.get("mask_output_format", "same")),
            "dynamic_mask_use_motion_diff": bool(g.get("dynamic_mask_use_motion_diff", True)),
            "dynamic_mask_motion_frames": int(g.get("dynamic_mask_motion_frames", 3)),
            "dynamic_mask_motion_threshold": int(g.get("dynamic_mask_motion_threshold", 30)),
            "dynamic_mask_dilation_size": int(g.get("dynamic_mask_dilation_size", 5)),
            "dynamic_mask_inpaint_enabled": bool(g.get("dynamic_mask_inpaint_enabled", False)),
            "dynamic_mask_inpaint_module": str(g.get("dynamic_mask_inpaint_module", "")),
        }

    def _on_accept(self):
        self._save_last_settings()
        self.accept()

    def _save_last_settings(self):
        """エクスポート画面固有の値のみ永続化"""
        settings = {
            "output_dir": self.dir_edit.text(),
            "prefix": self.prefix_edit.text().strip(),
            "enable_stereo_stitch": self.stereo_stitch_check.isChecked(),
            "enable_cubemap": self.cubemap_group.isChecked(),
            "cubemap_face_size": self.cubemap_face_spin.value(),
            "enable_perspective": self.persp_group.isChecked(),
            "perspective_fov": self.persp_fov_spin.value(),
            "perspective_yaw_list": self._parse_float_list(self.persp_yaw_edit.text()),
            "perspective_pitch_list": self._parse_float_list(self.persp_pitch_edit.text()),
            "perspective_size": [self.persp_size_spin.value(), self.persp_size_spin.value()],
        }
        save_dir = Path.home() / ".360split"
        save_dir.mkdir(exist_ok=True)
        path = save_dir / "export_settings.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"エクスポート設定保存失敗: {e}")

    def _load_last_settings(self):
        settings_dir = Path.home() / ".360split"
        global_path = settings_dir / "settings.json"
        export_path = settings_dir / "export_settings.json"

        try:
            if global_path.exists():
                with open(global_path, 'r', encoding='utf-8') as f:
                    g = json.load(f)

                projection_mode = str(g.get("projection_mode", "Equirectangular"))
                self.dir_edit.setText(g.get("output_directory", self.dir_edit.text()))
                self.prefix_edit.setText(str(g.get("naming_prefix", "keyframe")))
                self.stereo_stitch_check.setChecked(bool(g.get("enable_stereo_stitch", True)))
                self.cubemap_group.setChecked(projection_mode == "Cubemap")
                self.persp_group.setChecked(projection_mode == "Perspective")
                self.persp_fov_spin.setValue(float(g.get("perspective_fov", 90.0)))

            if export_path.exists():
                with open(export_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                self._apply_export_only_settings(s)

        except Exception as e:
            logger.warning(f"エクスポート設定読み込み失敗: {e}")

    def _apply_export_only_settings(self, s: Dict[str, Any]):
        self.dir_edit.setText(s.get("output_dir", self.dir_edit.text()))
        if "prefix" in s:
            self.prefix_edit.setText(str(s.get("prefix", "")))
        if "enable_stereo_stitch" in s:
            self.stereo_stitch_check.setChecked(bool(s.get("enable_stereo_stitch", True)))
        self.cubemap_group.setChecked(bool(s.get("enable_cubemap", False)))
        self.cubemap_face_spin.setValue(int(s.get("cubemap_face_size", 1024)))

        self.persp_group.setChecked(bool(s.get("enable_perspective", False)))
        self.persp_fov_spin.setValue(float(s.get("perspective_fov", 90.0)))
        yaw_list = s.get("perspective_yaw_list", [0.0, 90.0, 180.0, -90.0])
        pitch_list = s.get("perspective_pitch_list", [0.0])
        self.persp_yaw_edit.setText(", ".join(str(y) for y in yaw_list))
        self.persp_pitch_edit.setText(", ".join(str(p) for p in pitch_list))
        sz = s.get("perspective_size", [1024, 1024])
        if isinstance(sz, (list, tuple)) and len(sz) >= 1:
            self.persp_size_spin.setValue(int(sz[0]))

    def _load_global_settings(self):
        settings_file = Path.home() / ".360split" / "settings.json"
        g: Dict[str, Any] = {}
        try:
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    g = json.load(f)
        except Exception as e:
            logger.warning(f"共通設定の読み込み失敗: {e}")

        self._global_settings = g
        self._refresh_global_summary()

    def _refresh_global_summary(self):
        g = self._global_settings
        fmt = str(g.get("output_image_format", "png")).upper()
        jpg = int(g.get("output_jpeg_quality", 95))
        prefix = str(g.get("naming_prefix", "keyframe"))

        polar = "ON" if g.get("enable_polar_mask", False) else "OFF"
        nadir = "ON" if g.get("enable_nadir_mask", False) else "OFF"
        equip = "ON" if g.get("enable_equipment_detection", False) else "OFF"
        target = "ON" if g.get("enable_target_mask_generation", False) else "OFF"
        motion = "ON" if g.get("dynamic_mask_use_motion_diff", True) else "OFF"
        stitch_mode = str(g.get("stitching_mode", "Fast"))
        mask_dir = str(g.get("mask_output_dirname", "masks"))
        mask_fmt = str(g.get("mask_output_format", "same"))

        self.global_summary_label.setText(
            f"形式: {fmt} / JPEG品質: {jpg} / 接頭辞: {prefix}\n"
            f"ステッチモード(共通): {stitch_mode}\n"
            f"ポーラーマスク: {polar}, ナディア: {nadir}, 装備検出: {equip}\n"
            f"対象マスク生成: {target} (保存先: {mask_dir}, 形式: {mask_fmt}, MotionDiff: {motion})"
        )

    def _open_global_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            self._load_global_settings()

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
