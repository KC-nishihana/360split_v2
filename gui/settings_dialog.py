"""
設定ダイアログ - 360Split GUI
キーフレーム選択、360度処理、マスク、出力設定
"""

import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QSlider,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QSettings

from utils.logger import get_logger
logger = get_logger(__name__)


class SettingsDialog(QDialog):
    """
    設定ダイアログ

    4つのタブで構成：
    1. キーフレーム選択設定（品質重み、適応閾値）
    2. 360度処理設定（解像度、投影モード）
    3. マスク処理設定（ナディアマスク、装備検出）
    4. 出力設定（形式、品質、ディレクトリ）

    Attributes:
    -----------
    settings : dict
        現在の設定値を保持する辞書
    """

    def __init__(self, parent=None):
        """
        設定ダイアログの初期化

        Parameters:
        -----------
        parent : QWidget, optional
            親ウィジェット
        """
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setGeometry(200, 200, 700, 800)

        # 設定を読み込み
        self.settings = self._load_settings()

        self._setup_ui()

    def _setup_ui(self):
        """
        UIの構築
        """
        layout = QVBoxLayout(self)

        # === タブウィジェット ===
        tab_widget = QTabWidget()

        # Tab 1: キーフレーム選択
        keyframe_tab = self._create_keyframe_tab()
        tab_widget.addTab(keyframe_tab, "キーフレーム選択")

        # Tab 2: 360度処理
        processing_tab = self._create_processing_tab()
        tab_widget.addTab(processing_tab, "360度処理")

        # Tab 3: マスク処理
        mask_tab = self._create_mask_tab()
        tab_widget.addTab(mask_tab, "マスク処理")

        # Tab 4: 出力設定
        output_tab = self._create_output_tab()
        tab_widget.addTab(output_tab, "出力設定")

        layout.addWidget(tab_widget)

        # === ボタン ===
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("キャンセル")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        reset_button = QPushButton("デフォルトに戻す")
        reset_button.clicked.connect(self._on_reset)
        button_layout.addWidget(reset_button)

        button_layout.addStretch()

        layout.addLayout(button_layout)

    def _create_keyframe_tab(self) -> QWidget:
        """
        キーフレーム選択タブを作成

        Returns:
        --------
        QWidget
            タブウィジェット
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # === 品質スコア重み ===
        quality_group = QGroupBox("品質スコア重み")
        quality_layout = QGridLayout()

        # シャープネス重み
        quality_layout.addWidget(QLabel("鮮明度 (Sharpness):"), 0, 0)
        self.sharpness_weight_slider = QSlider(Qt.Horizontal)
        self.sharpness_weight_slider.setMinimum(0)
        self.sharpness_weight_slider.setMaximum(100)
        self.sharpness_weight_slider.setValue(
            int(self.settings.get('weight_sharpness', 0.30) * 100)
        )
        self.sharpness_weight_label = QLabel(
            f"{self.settings.get('weight_sharpness', 0.30):.2f}"
        )
        self.sharpness_weight_slider.valueChanged.connect(
            lambda v: self.sharpness_weight_label.setText(f"{v/100:.2f}")
        )
        quality_layout.addWidget(self.sharpness_weight_slider, 0, 1)
        quality_layout.addWidget(self.sharpness_weight_label, 0, 2)

        # 露光重み
        quality_layout.addWidget(QLabel("露光 (Exposure):"), 1, 0)
        self.exposure_weight_slider = QSlider(Qt.Horizontal)
        self.exposure_weight_slider.setMinimum(0)
        self.exposure_weight_slider.setMaximum(100)
        self.exposure_weight_slider.setValue(
            int(self.settings.get('weight_exposure', 0.15) * 100)
        )
        self.exposure_weight_label = QLabel(
            f"{self.settings.get('weight_exposure', 0.15):.2f}"
        )
        self.exposure_weight_slider.valueChanged.connect(
            lambda v: self.exposure_weight_label.setText(f"{v/100:.2f}")
        )
        quality_layout.addWidget(self.exposure_weight_slider, 1, 1)
        quality_layout.addWidget(self.exposure_weight_label, 1, 2)

        # 幾何学的スコア重み
        quality_layout.addWidget(QLabel("幾何学的スコア (Geometric):"), 2, 0)
        self.geometric_weight_slider = QSlider(Qt.Horizontal)
        self.geometric_weight_slider.setMinimum(0)
        self.geometric_weight_slider.setMaximum(100)
        self.geometric_weight_slider.setValue(
            int(self.settings.get('weight_geometric', 0.30) * 100)
        )
        self.geometric_weight_label = QLabel(
            f"{self.settings.get('weight_geometric', 0.30):.2f}"
        )
        self.geometric_weight_slider.valueChanged.connect(
            lambda v: self.geometric_weight_label.setText(f"{v/100:.2f}")
        )
        quality_layout.addWidget(self.geometric_weight_slider, 2, 1)
        quality_layout.addWidget(self.geometric_weight_label, 2, 2)

        # コンテンツ重み
        quality_layout.addWidget(QLabel("コンテンツ変化 (Content):"), 3, 0)
        self.content_weight_slider = QSlider(Qt.Horizontal)
        self.content_weight_slider.setMinimum(0)
        self.content_weight_slider.setMaximum(100)
        self.content_weight_slider.setValue(
            int(self.settings.get('weight_content', 0.25) * 100)
        )
        self.content_weight_label = QLabel(
            f"{self.settings.get('weight_content', 0.25):.2f}"
        )
        self.content_weight_slider.valueChanged.connect(
            lambda v: self.content_weight_label.setText(f"{v/100:.2f}")
        )
        quality_layout.addWidget(self.content_weight_slider, 3, 1)
        quality_layout.addWidget(self.content_weight_label, 3, 2)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # === 適応的選択閾値 ===
        adaptive_group = QGroupBox("適応的選択パラメータ")
        adaptive_layout = QGridLayout()

        # SSIM変化閾値
        adaptive_layout.addWidget(QLabel("SSIM変化閾値:"), 0, 0)
        self.ssim_threshold = QDoubleSpinBox()
        self.ssim_threshold.setMinimum(0.0)
        self.ssim_threshold.setMaximum(1.0)
        self.ssim_threshold.setSingleStep(0.05)
        self.ssim_threshold.setValue(self.settings.get('ssim_threshold', 0.85))
        adaptive_layout.addWidget(self.ssim_threshold, 0, 1)

        # 最小キーフレーム間隔
        adaptive_layout.addWidget(QLabel("最小キーフレーム間隔 (frames):"), 1, 0)
        self.min_keyframe_interval = QSpinBox()
        self.min_keyframe_interval.setMinimum(1)
        self.min_keyframe_interval.setMaximum(100)
        self.min_keyframe_interval.setValue(self.settings.get('min_keyframe_interval', 5))
        adaptive_layout.addWidget(self.min_keyframe_interval, 1, 1)

        # 最大キーフレーム間隔
        adaptive_layout.addWidget(QLabel("最大キーフレーム間隔 (frames):"), 2, 0)
        self.max_keyframe_interval = QSpinBox()
        self.max_keyframe_interval.setMinimum(1)
        self.max_keyframe_interval.setMaximum(300)
        self.max_keyframe_interval.setValue(self.settings.get('max_keyframe_interval', 60))
        adaptive_layout.addWidget(self.max_keyframe_interval, 2, 1)

        # Softmax温度β
        adaptive_layout.addWidget(QLabel("Softmax温度β:"), 3, 0)
        self.softmax_beta = QDoubleSpinBox()
        self.softmax_beta.setMinimum(0.1)
        self.softmax_beta.setMaximum(20.0)
        self.softmax_beta.setSingleStep(0.5)
        self.softmax_beta.setValue(self.settings.get('softmax_beta', 5.0))
        adaptive_layout.addWidget(self.softmax_beta, 3, 1)

        adaptive_group.setLayout(adaptive_layout)
        layout.addWidget(adaptive_group)

        # === GRIC パラメータ ===
        gric_group = QGroupBox("GRIC (幾何学的評価)")
        gric_layout = QGridLayout()

        # Lambda1
        gric_layout.addWidget(QLabel("Lambda1 (データ適合):"), 0, 0)
        self.gric_lambda1 = QDoubleSpinBox()
        self.gric_lambda1.setMinimum(0.1)
        self.gric_lambda1.setMaximum(10.0)
        self.gric_lambda1.setSingleStep(0.5)
        self.gric_lambda1.setValue(self.settings.get('gric_lambda1', 2.0))
        gric_layout.addWidget(self.gric_lambda1, 0, 1)

        # Lambda2
        gric_layout.addWidget(QLabel("Lambda2 (モデル複雑さ):"), 1, 0)
        self.gric_lambda2 = QDoubleSpinBox()
        self.gric_lambda2.setMinimum(0.1)
        self.gric_lambda2.setMaximum(20.0)
        self.gric_lambda2.setSingleStep(0.5)
        self.gric_lambda2.setValue(self.settings.get('gric_lambda2', 4.0))
        gric_layout.addWidget(self.gric_lambda2, 1, 1)

        # Sigma
        gric_layout.addWidget(QLabel("Sigma (残差標準偏差):"), 2, 0)
        self.gric_sigma = QDoubleSpinBox()
        self.gric_sigma.setMinimum(0.1)
        self.gric_sigma.setMaximum(10.0)
        self.gric_sigma.setSingleStep(0.1)
        self.gric_sigma.setValue(self.settings.get('gric_sigma', 1.0))
        gric_layout.addWidget(self.gric_sigma, 2, 1)

        # 縮退閾値
        gric_layout.addWidget(QLabel("縮退判定閾値 (H inlier率):"), 3, 0)
        self.gric_degeneracy_threshold = QDoubleSpinBox()
        self.gric_degeneracy_threshold.setMinimum(0.5)
        self.gric_degeneracy_threshold.setMaximum(1.0)
        self.gric_degeneracy_threshold.setSingleStep(0.05)
        self.gric_degeneracy_threshold.setValue(
            self.settings.get('gric_degeneracy_threshold', 0.85)
        )
        gric_layout.addWidget(self.gric_degeneracy_threshold, 3, 1)

        gric_group.setLayout(gric_layout)
        layout.addWidget(gric_group)

        layout.addStretch()
        return widget

    def _create_processing_tab(self) -> QWidget:
        """
        360度処理タブを作成

        Returns:
        --------
        QWidget
            タブウィジェット
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # === 出力解像度 ===
        resolution_group = QGroupBox("出力解像度")
        resolution_layout = QGridLayout()

        resolution_layout.addWidget(QLabel("幅 (Width):"), 0, 0)
        self.equirect_width = QSpinBox()
        self.equirect_width.setMinimum(512)
        self.equirect_width.setMaximum(8192)
        self.equirect_width.setSingleStep(256)
        self.equirect_width.setValue(self.settings.get('equirect_width', 4096))
        resolution_layout.addWidget(self.equirect_width, 0, 1)

        resolution_layout.addWidget(QLabel("高さ (Height):"), 1, 0)
        self.equirect_height = QSpinBox()
        self.equirect_height.setMinimum(256)
        self.equirect_height.setMaximum(4096)
        self.equirect_height.setSingleStep(128)
        self.equirect_height.setValue(self.settings.get('equirect_height', 2048))
        resolution_layout.addWidget(self.equirect_height, 1, 1)

        resolution_group.setLayout(resolution_layout)
        layout.addWidget(resolution_group)

        # === 投影モード ===
        projection_group = QGroupBox("投影モード")
        projection_layout = QGridLayout()

        projection_layout.addWidget(QLabel("投影方式:"), 0, 0)
        self.projection_mode = QComboBox()
        self.projection_mode.addItems(["Equirectangular", "Cubemap", "Perspective"])
        self.projection_mode.setCurrentText(self.settings.get('projection_mode', 'Equirectangular'))
        projection_layout.addWidget(self.projection_mode, 0, 1)

        projection_layout.addWidget(QLabel("視野角 (FOV):"), 1, 0)
        self.fov = QDoubleSpinBox()
        self.fov.setMinimum(20.0)
        self.fov.setMaximum(180.0)
        self.fov.setSingleStep(10.0)
        self.fov.setValue(self.settings.get('perspective_fov', 90.0))
        projection_layout.addWidget(self.fov, 1, 1)

        projection_group.setLayout(projection_layout)
        layout.addWidget(projection_group)

        # === 360°ポーラーマスク ===
        polar_group = QGroupBox("360°ポーラーマスク")
        polar_layout = QGridLayout()

        self.enable_polar_mask = QCheckBox("天頂/天底マスクを有効化")
        self.enable_polar_mask.setChecked(self.settings.get('enable_polar_mask', True))
        polar_layout.addWidget(self.enable_polar_mask, 0, 0, 1, 2)

        polar_layout.addWidget(QLabel("マスク比率 (上下%):"), 1, 0)
        self.mask_polar_ratio = QDoubleSpinBox()
        self.mask_polar_ratio.setMinimum(0.0)
        self.mask_polar_ratio.setMaximum(0.30)
        self.mask_polar_ratio.setSingleStep(0.01)
        self.mask_polar_ratio.setDecimals(2)
        self.mask_polar_ratio.setValue(self.settings.get('mask_polar_ratio', 0.10))
        polar_layout.addWidget(self.mask_polar_ratio, 1, 1)

        polar_group.setLayout(polar_layout)
        layout.addWidget(polar_group)

        # === ステッチングモード ===
        stitching_group = QGroupBox("ステッチングモード")
        stitching_layout = QVBoxLayout()

        self.stitching_mode = QComboBox()
        self.stitching_mode.addItems(["Fast", "High Quality (HQ)", "Depth-aware"])
        self.stitching_mode.setCurrentText(self.settings.get('stitching_mode', 'Fast'))
        stitching_layout.addWidget(self.stitching_mode)

        stitching_group.setLayout(stitching_layout)
        layout.addWidget(stitching_group)

        layout.addStretch()
        return widget

    def _create_mask_tab(self) -> QWidget:
        """
        マスク処理タブを作成

        Returns:
        --------
        QWidget
            タブウィジェット
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # === ナディアマスク ===
        nadir_group = QGroupBox("ナディアマスク処理")
        nadir_layout = QGridLayout()

        self.enable_nadir_mask = QCheckBox("ナディアマスクを有効化")
        self.enable_nadir_mask.setChecked(self.settings.get('enable_nadir_mask', False))
        nadir_layout.addWidget(self.enable_nadir_mask, 0, 0, 1, 2)

        nadir_layout.addWidget(QLabel("ナディアマスク半径:"), 1, 0)
        self.nadir_mask_radius = QSlider(Qt.Horizontal)
        self.nadir_mask_radius.setMinimum(0)
        self.nadir_mask_radius.setMaximum(500)
        self.nadir_mask_radius.setValue(
            int(self.settings.get('nadir_mask_radius', 100))
        )
        self.nadir_radius_label = QLabel(
            f"{self.settings.get('nadir_mask_radius', 100)}"
        )
        self.nadir_mask_radius.valueChanged.connect(
            lambda v: self.nadir_radius_label.setText(str(v))
        )
        nadir_layout.addWidget(self.nadir_mask_radius, 1, 1)
        nadir_layout.addWidget(self.nadir_radius_label, 1, 2)

        nadir_group.setLayout(nadir_layout)
        layout.addWidget(nadir_group)

        # === 装備検出 ===
        equipment_group = QGroupBox("装備検出")
        equipment_layout = QGridLayout()

        self.enable_equipment_detection = QCheckBox("装備検出を有効化")
        self.enable_equipment_detection.setChecked(
            self.settings.get('enable_equipment_detection', False)
        )
        equipment_layout.addWidget(self.enable_equipment_detection, 0, 0, 1, 2)

        equipment_layout.addWidget(QLabel("マスク膨張サイズ (Dilation):"), 1, 0)
        self.mask_dilation = QSpinBox()
        self.mask_dilation.setMinimum(0)
        self.mask_dilation.setMaximum(50)
        self.mask_dilation.setValue(self.settings.get('mask_dilation_size', 15))
        equipment_layout.addWidget(self.mask_dilation, 1, 1)

        equipment_group.setLayout(equipment_layout)
        layout.addWidget(equipment_group)

        layout.addStretch()
        return widget

    def _create_output_tab(self) -> QWidget:
        """
        出力設定タブを作成

        Returns:
        --------
        QWidget
            タブウィジェット
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # === 出力形式 ===
        format_group = QGroupBox("出力形式")
        format_layout = QGridLayout()

        format_layout.addWidget(QLabel("画像形式:"), 0, 0)
        self.output_format = QComboBox()
        self.output_format.addItems(["PNG", "JPEG", "TIFF"])
        self.output_format.setCurrentText(self.settings.get('output_image_format', 'png').upper())
        format_layout.addWidget(self.output_format, 0, 1)

        format_layout.addWidget(QLabel("JPEG品質:"), 1, 0)
        self.jpeg_quality = QSlider(Qt.Horizontal)
        self.jpeg_quality.setMinimum(50)
        self.jpeg_quality.setMaximum(100)
        self.jpeg_quality.setValue(self.settings.get('output_jpeg_quality', 95))
        self.jpeg_quality_label = QLabel(
            f"{self.settings.get('output_jpeg_quality', 95)}"
        )
        self.jpeg_quality.valueChanged.connect(
            lambda v: self.jpeg_quality_label.setText(str(v))
        )
        format_layout.addWidget(self.jpeg_quality, 1, 1)
        format_layout.addWidget(self.jpeg_quality_label, 1, 2)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # === 出力ディレクトリ ===
        directory_group = QGroupBox("出力ディレクトリ")
        directory_layout = QHBoxLayout()

        self.output_dir_label = QLabel(
            self.settings.get('output_directory', str(Path.home() / "360split_output"))
        )
        self.output_dir_label.setWordWrap(True)
        directory_layout.addWidget(self.output_dir_label, stretch=1)

        browse_button = QPushButton("参照...")
        browse_button.setMaximumWidth(80)
        browse_button.clicked.connect(self._on_browse_output_dir)
        directory_layout.addWidget(browse_button)

        directory_group.setLayout(directory_layout)
        layout.addWidget(directory_group)

        # === 命名規則 ===
        naming_group = QGroupBox("ファイル命名規則")
        naming_layout = QGridLayout()

        naming_layout.addWidget(QLabel("プレフィックス:"), 0, 0)
        self.naming_prefix = QComboBox()
        self.naming_prefix.addItems(["keyframe", "frame", "image"])
        self.naming_prefix.setCurrentText(self.settings.get('naming_prefix', 'keyframe'))
        naming_layout.addWidget(self.naming_prefix, 0, 1)

        naming_layout.addWidget(QLabel("例: keyframe_00001.png"))

        naming_group.setLayout(naming_layout)
        layout.addWidget(naming_group)

        layout.addStretch()
        return widget

    def _load_settings(self) -> dict:
        """
        設定ファイルから設定を読み込む

        Returns:
        --------
        dict
            設定辞書
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
            'gric_lambda1': 2.0,
            'gric_lambda2': 4.0,
            'gric_sigma': 1.0,
            'gric_degeneracy_threshold': 0.85,
            'equirect_width': 4096,
            'equirect_height': 2048,
            'projection_mode': 'Equirectangular',
            'perspective_fov': 90.0,
            'enable_polar_mask': True,
            'mask_polar_ratio': 0.10,
            'stitching_mode': 'Fast',
            'enable_nadir_mask': False,
            'nadir_mask_radius': 100,
            'enable_equipment_detection': False,
            'mask_dilation_size': 15,
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
        except Exception as e:
            logger.exception(f"設定ファイル読み込みエラー: {e}")

        return default_settings

    def _save_settings(self):
        """
        設定をファイルに保存
        """
        settings_dir = Path.home() / ".360split"
        settings_dir.mkdir(exist_ok=True)

        settings_file = settings_dir / "settings.json"

        # UI から設定値を取得
        settings_to_save = {
            'weight_sharpness': self.sharpness_weight_slider.value() / 100,
            'weight_exposure': self.exposure_weight_slider.value() / 100,
            'weight_geometric': self.geometric_weight_slider.value() / 100,
            'weight_content': self.content_weight_slider.value() / 100,
            'ssim_threshold': self.ssim_threshold.value(),
            'min_keyframe_interval': self.min_keyframe_interval.value(),
            'max_keyframe_interval': self.max_keyframe_interval.value(),
            'softmax_beta': self.softmax_beta.value(),
            'gric_lambda1': self.gric_lambda1.value(),
            'gric_lambda2': self.gric_lambda2.value(),
            'gric_sigma': self.gric_sigma.value(),
            'gric_degeneracy_threshold': self.gric_degeneracy_threshold.value(),
            'equirect_width': self.equirect_width.value(),
            'equirect_height': self.equirect_height.value(),
            'projection_mode': self.projection_mode.currentText(),
            'perspective_fov': self.fov.value(),
            'enable_polar_mask': self.enable_polar_mask.isChecked(),
            'mask_polar_ratio': self.mask_polar_ratio.value(),
            'stitching_mode': self.stitching_mode.currentText(),
            'enable_nadir_mask': self.enable_nadir_mask.isChecked(),
            'nadir_mask_radius': self.nadir_mask_radius.value(),
            'enable_equipment_detection': self.enable_equipment_detection.isChecked(),
            'mask_dilation_size': self.mask_dilation.value(),
            'output_image_format': self.output_format.currentText().lower(),
            'output_jpeg_quality': self.jpeg_quality.value(),
            'output_directory': self.output_dir_label.text(),
            'naming_prefix': self.naming_prefix.currentText(),
        }

        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"設定を保存しました: {settings_file}")

        except Exception as e:
            logger.exception(f"設定保存エラー: {e}")
            QMessageBox.critical(
                self,
                "エラー",
                f"設定の保存に失敗しました:\n{str(e)}"
            )

    # === スロット ===

    def _on_ok(self):
        """
        OKボタン押下時のコールバック
        """
        self._save_settings()
        self.accept()

    def _on_reset(self):
        """
        リセットボタン押下時のコールバック
        """
        reply = QMessageBox.question(
            self,
            "確認",
            "すべての設定をデフォルト値にリセットしますか？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # デフォルト値で再初期化
            settings_file = Path.home() / ".360split" / "settings.json"
            if settings_file.exists():
                settings_file.unlink()

            self.settings = self._load_settings()

            # UIを更新
            self.sharpness_weight_slider.setValue(int(0.30 * 100))
            self.exposure_weight_slider.setValue(int(0.15 * 100))
            self.geometric_weight_slider.setValue(int(0.30 * 100))
            self.content_weight_slider.setValue(int(0.25 * 100))
            self.ssim_threshold.setValue(0.85)
            self.min_keyframe_interval.setValue(5)
            self.max_keyframe_interval.setValue(60)
            self.softmax_beta.setValue(5.0)
            self.gric_lambda1.setValue(2.0)
            self.gric_lambda2.setValue(4.0)
            self.gric_sigma.setValue(1.0)
            self.gric_degeneracy_threshold.setValue(0.85)
            self.equirect_width.setValue(4096)
            self.equirect_height.setValue(2048)
            self.projection_mode.setCurrentText('Equirectangular')
            self.fov.setValue(90.0)
            self.enable_polar_mask.setChecked(True)
            self.mask_polar_ratio.setValue(0.10)
            self.stitching_mode.setCurrentText('Fast')
            self.enable_nadir_mask.setChecked(False)
            self.nadir_mask_radius.setValue(100)
            self.enable_equipment_detection.setChecked(False)
            self.mask_dilation.setValue(15)
            self.output_format.setCurrentText('PNG')
            self.jpeg_quality.setValue(95)
            self.output_dir_label.setText(str(Path.home() / "360split_output"))
            self.naming_prefix.setCurrentText('keyframe')

            QMessageBox.information(self, "完了", "設定をデフォルト値にリセットしました")

    def _on_browse_output_dir(self):
        """
        出力ディレクトリ参照ボタン押下時のコールバック
        """
        directory = QFileDialog.getExistingDirectory(
            self,
            "出力ディレクトリを選択",
            self.output_dir_label.text()
        )

        if directory:
            self.output_dir_label.setText(directory)
