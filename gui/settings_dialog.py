"""
設定ダイアログ - 360Split GUI
キーフレーム選択、360度処理、マスク、出力設定
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QSlider,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QScrollArea
    , QLineEdit
)
from PySide6.QtCore import Qt

from utils.logger import get_logger
logger = get_logger(__name__)

TARGET_CLASS_LABELS = ["人物", "人", "自転車", "バイク", "車両", "空", "動物", "その他"]


class SettingsDialog(QDialog):
    """
    設定ダイアログ

    6つのタブで構成：
    1. キーフレーム選択設定（品質重み、適応閾値）
    2. Stage0/Stage3設定（軌跡再評価）
    3. 360度処理設定（解像度、投影モード）
    4. マスク処理設定（ナディアマスク、装備検出）
    5. 出力設定（形式、品質、ディレクトリ）
    6. 対象マスク設定

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
        self.resize(860, 720)
        self.setMinimumSize(700, 520)

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
        tab_widget.setUsesScrollButtons(True)

        # Tab 1: キーフレーム選択
        keyframe_tab = self._create_keyframe_tab()
        tab_widget.addTab(self._wrap_scroll_tab(keyframe_tab), "キーフレーム選択")

        # Tab 2: 360度処理
        stage03_tab = self._create_stage03_tab()
        tab_widget.addTab(self._wrap_scroll_tab(stage03_tab), "Stage0/Stage3")

        # Tab 3: 360度処理
        processing_tab = self._create_processing_tab()
        tab_widget.addTab(self._wrap_scroll_tab(processing_tab), "360度処理")

        # Tab 4: マスク処理
        mask_tab = self._create_mask_tab()
        tab_widget.addTab(self._wrap_scroll_tab(mask_tab), "マスク処理")

        # Tab 5: 出力設定
        output_tab = self._create_output_tab()
        tab_widget.addTab(self._wrap_scroll_tab(output_tab), "出力設定")

        # Tab 6: 対象マスク
        target_mask_tab = self._create_target_mask_tab()
        tab_widget.addTab(self._wrap_scroll_tab(target_mask_tab), "対象マスク")

        layout.addWidget(tab_widget)
        if hasattr(self, "pose_backend"):
            self._on_pose_backend_changed(self.pose_backend.currentText())

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

    def _wrap_scroll_tab(self, content: QWidget) -> QWidget:
        """タブ内容をスクロール可能にして、小さい画面でも切れないようにする。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll

    def _create_keyframe_tab(self) -> QWidget:
        """
        キーフレーム選択タブを作成

        Returns:
        --------
        QWidget
            タブウィジェット
        """
        widget = QWidget()
        root_layout = QVBoxLayout(widget)
        root_layout.setSpacing(6)

        sub_tabs = QTabWidget()
        root_layout.addWidget(sub_tabs)

        basic_page = QWidget()
        basic_layout = QVBoxLayout(basic_page)
        basic_layout.setSpacing(10)
        sub_tabs.addTab(basic_page, "基本")

        threshold_page = QWidget()
        threshold_layout = QVBoxLayout(threshold_page)
        threshold_layout.setSpacing(10)
        sub_tabs.addTab(threshold_page, "選択閾値")

        advanced_page = QWidget()
        advanced_layout = QVBoxLayout(advanced_page)
        advanced_layout.setSpacing(10)
        sub_tabs.addTab(advanced_page, "GRIC/ログ")

        # === 環境プリセット選択 ===
        preset_group = QGroupBox("環境プリセット")
        preset_layout = QGridLayout()

        preset_layout.addWidget(QLabel("プリセット:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom (手動設定)",
            "Outdoor (屋外・高品質)",
            "Indoor (屋内・追跡重視)",
            "Mixed (混合・適応型)"
        ])
        self.preset_combo.setCurrentIndex(0)  # デフォルトはCustom
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo, 0, 1)

        # プリセット説明ラベル
        self.preset_description_label = QLabel("")
        self.preset_description_label.setWordWrap(True)
        self.preset_description_label.setStyleSheet("color: #666; font-size: 11px;")
        preset_layout.addWidget(self.preset_description_label, 1, 0, 1, 2)

        preset_group.setLayout(preset_layout)
        basic_layout.addWidget(preset_group)

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
        basic_layout.addWidget(quality_group)

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
        threshold_layout.addWidget(adaptive_group)

        # === 品質フィルタ（A案） ===
        quality_filter_group = QGroupBox("品質フィルタ（ROI + 正規化）")
        quality_filter_layout = QGridLayout()

        self.quality_filter_enabled = QCheckBox("品質フィルタを有効化")
        self.quality_filter_enabled.setChecked(bool(self.settings.get("quality_filter_enabled", True)))
        quality_filter_layout.addWidget(self.quality_filter_enabled, 0, 0, 1, 2)

        quality_filter_layout.addWidget(QLabel("品質しきい値 (0.0-1.0):"), 1, 0)
        self.quality_threshold = QDoubleSpinBox()
        self.quality_threshold.setMinimum(0.0)
        self.quality_threshold.setMaximum(1.0)
        self.quality_threshold.setSingleStep(0.05)
        self.quality_threshold.setDecimals(2)
        self.quality_threshold.setValue(float(self.settings.get("quality_threshold", 0.50)))
        quality_filter_layout.addWidget(self.quality_threshold, 1, 1)

        quality_filter_layout.addWidget(QLabel("Stage1 LR統合モード:"), 2, 0)
        self.stage1_lr_merge_mode = QComboBox()
        self.stage1_lr_merge_mode.addItems(["asymmetric_sky_v1", "strict_min"])
        self.stage1_lr_merge_mode.setCurrentText(str(self.settings.get("stage1_lr_merge_mode", "asymmetric_sky_v1")))
        quality_filter_layout.addWidget(self.stage1_lr_merge_mode, 2, 1)

        quality_filter_layout.addWidget(QLabel("LR弱レンズ下限:"), 3, 0)
        self.stage1_lr_asym_weak_floor = QDoubleSpinBox()
        self.stage1_lr_asym_weak_floor.setRange(0.0, 1.0)
        self.stage1_lr_asym_weak_floor.setSingleStep(0.01)
        self.stage1_lr_asym_weak_floor.setDecimals(2)
        self.stage1_lr_asym_weak_floor.setValue(float(self.settings.get("stage1_lr_asym_weak_floor", 0.35)))
        quality_filter_layout.addWidget(self.stage1_lr_asym_weak_floor, 3, 1)

        quality_filter_group.setLayout(quality_filter_layout)
        threshold_layout.addWidget(quality_filter_group)

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
        advanced_layout.addWidget(gric_group)

        # === Rerunログ ===
        rerun_group = QGroupBox("Rerunログ")
        rerun_layout = QGridLayout()
        self.enable_rerun_logging = QCheckBox("解析時にRerunログを有効化（GUI）")
        self.enable_rerun_logging.setChecked(self.settings.get('enable_rerun_logging', False))
        rerun_layout.addWidget(self.enable_rerun_logging, 0, 0, 1, 2)
        rerun_group.setLayout(rerun_layout)
        advanced_layout.addWidget(rerun_group)

        basic_layout.addStretch()
        threshold_layout.addStretch()
        advanced_layout.addStretch()
        return widget

    def _create_stage03_tab(self) -> QWidget:
        """Stage0/Stage3 設定タブを作成する。"""
        widget = QWidget()
        root_layout = QVBoxLayout(widget)
        root_layout.setSpacing(6)

        sub_tabs = QTabWidget()
        root_layout.addWidget(sub_tabs)

        stage_page = QWidget()
        stage_page_layout = QVBoxLayout(stage_page)
        stage_page_layout.setSpacing(10)
        sub_tabs.addTab(stage_page, "Stage0/Stage3")

        vo_page = QWidget()
        vo_page_layout = QVBoxLayout(vo_page)
        vo_page_layout.setSpacing(10)
        sub_tabs.addTab(vo_page, "VO/Calibration")

        perf_page = QWidget()
        perf_page_layout = QVBoxLayout(perf_page)
        perf_page_layout.setSpacing(10)
        sub_tabs.addTab(perf_page, "性能")

        stage03_group = QGroupBox("Stage0/Stage3 軌跡再評価")
        stage03_layout = QGridLayout()

        self.enable_stage0_scan = QCheckBox("Stage0軽量走査を有効化")
        self.enable_stage0_scan.setChecked(
            self.settings.get("enable_stage0_scan", True)
        )
        stage03_layout.addWidget(self.enable_stage0_scan, 0, 0, 1, 3)

        stage03_layout.addWidget(QLabel("Stage0サンプリング間隔:"), 1, 0)
        self.stage0_stride = QSpinBox()
        self.stage0_stride.setMinimum(1)
        self.stage0_stride.setMaximum(120)
        self.stage0_stride.setValue(int(self.settings.get("stage0_stride", 5)))
        stage03_layout.addWidget(self.stage0_stride, 1, 1)

        self.enable_stage3_refinement = QCheckBox("Stage3軌跡再評価を有効化")
        self.enable_stage3_refinement.setChecked(
            self.settings.get("enable_stage3_refinement", True)
        )
        stage03_layout.addWidget(self.enable_stage3_refinement, 2, 0, 1, 3)

        stage03_layout.addWidget(QLabel("Stage3 base重み:"), 3, 0)
        self.stage3_weight_base = QDoubleSpinBox()
        self.stage3_weight_base.setRange(0.0, 1.0)
        self.stage3_weight_base.setSingleStep(0.01)
        self.stage3_weight_base.setDecimals(2)
        self.stage3_weight_base.setValue(float(self.settings.get("stage3_weight_base", 0.70)))
        stage03_layout.addWidget(self.stage3_weight_base, 3, 1)

        stage03_layout.addWidget(QLabel("Stage3 trajectory重み:"), 4, 0)
        self.stage3_weight_trajectory = QDoubleSpinBox()
        self.stage3_weight_trajectory.setRange(0.0, 1.0)
        self.stage3_weight_trajectory.setSingleStep(0.01)
        self.stage3_weight_trajectory.setDecimals(2)
        self.stage3_weight_trajectory.setValue(float(self.settings.get("stage3_weight_trajectory", 0.25)))
        stage03_layout.addWidget(self.stage3_weight_trajectory, 4, 1)

        stage03_layout.addWidget(QLabel("Stage3 stage0-risk重み:"), 5, 0)
        self.stage3_weight_stage0_risk = QDoubleSpinBox()
        self.stage3_weight_stage0_risk.setRange(0.0, 1.0)
        self.stage3_weight_stage0_risk.setSingleStep(0.01)
        self.stage3_weight_stage0_risk.setDecimals(2)
        self.stage3_weight_stage0_risk.setValue(float(self.settings.get("stage3_weight_stage0_risk", 0.05)))
        stage03_layout.addWidget(self.stage3_weight_stage0_risk, 5, 1)

        stage03_group.setLayout(stage03_layout)
        stage_page_layout.addWidget(stage03_group)

        vo_group = QGroupBox("VO / Calibration")
        vo_layout = QGridLayout()

        vo_layout.addWidget(QLabel("VOプリセット:"), 0, 0)
        self.vo_preset_combo = QComboBox()
        self.vo_preset_combo.addItems(["Custom", "Quick", "Balanced", "Precise"])
        self.vo_preset_combo.setCurrentText("Custom")
        self.vo_preset_combo.currentTextChanged.connect(self._on_vo_preset_changed)
        vo_layout.addWidget(self.vo_preset_combo, 0, 1, 1, 2)

        self.vo_enabled = QCheckBox("VOを有効化")
        self.vo_enabled.setChecked(bool(self.settings.get("vo_enabled", True)))
        vo_layout.addWidget(self.vo_enabled, 1, 0, 1, 3)

        vo_layout.addWidget(QLabel("Calib XML:"), 2, 0)
        self.calib_xml = QLineEdit(str(self.settings.get("calib_xml", "")))
        vo_layout.addWidget(self.calib_xml, 2, 1)
        btn_calib = QPushButton("参照")
        btn_calib.clicked.connect(lambda: self._pick_file_for_line_edit(self.calib_xml))
        vo_layout.addWidget(btn_calib, 2, 2)

        vo_layout.addWidget(QLabel("Front Calib XML:"), 3, 0)
        self.front_calib_xml = QLineEdit(str(self.settings.get("front_calib_xml", "")))
        vo_layout.addWidget(self.front_calib_xml, 3, 1)
        btn_front_calib = QPushButton("参照")
        btn_front_calib.clicked.connect(lambda: self._pick_file_for_line_edit(self.front_calib_xml))
        vo_layout.addWidget(btn_front_calib, 3, 2)

        vo_layout.addWidget(QLabel("Rear Calib XML:"), 4, 0)
        self.rear_calib_xml = QLineEdit(str(self.settings.get("rear_calib_xml", "")))
        vo_layout.addWidget(self.rear_calib_xml, 4, 1)
        btn_rear_calib = QPushButton("参照")
        btn_rear_calib.clicked.connect(lambda: self._pick_file_for_line_edit(self.rear_calib_xml))
        vo_layout.addWidget(btn_rear_calib, 4, 2)

        vo_layout.addWidget(QLabel("Calib model:"), 5, 0)
        self.calib_model = QComboBox()
        self.calib_model.addItems(["auto", "opencv", "fisheye"])
        self.calib_model.setCurrentText(str(self.settings.get("calib_model", "auto")))
        vo_layout.addWidget(self.calib_model, 5, 1)

        vo_layout.addWidget(QLabel("VO中心ROI比率:"), 6, 0)
        self.vo_center_roi_ratio = QDoubleSpinBox()
        self.vo_center_roi_ratio.setRange(0.2, 1.0)
        self.vo_center_roi_ratio.setSingleStep(0.05)
        self.vo_center_roi_ratio.setDecimals(2)
        self.vo_center_roi_ratio.setValue(float(self.settings.get("vo_center_roi_ratio", 0.6)))
        vo_layout.addWidget(self.vo_center_roi_ratio, 6, 1)

        vo_layout.addWidget(QLabel("VO縮小長辺(px):"), 7, 0)
        self.vo_downscale_long_edge = QSpinBox()
        self.vo_downscale_long_edge.setRange(320, 2000)
        self.vo_downscale_long_edge.setValue(int(self.settings.get("vo_downscale_long_edge", 1000)))
        vo_layout.addWidget(self.vo_downscale_long_edge, 7, 1)

        vo_layout.addWidget(QLabel("VO最大特徴点数:"), 8, 0)
        self.vo_max_features = QSpinBox()
        self.vo_max_features.setRange(100, 2000)
        self.vo_max_features.setValue(int(self.settings.get("vo_max_features", 600)))
        vo_layout.addWidget(self.vo_max_features, 8, 1)

        vo_layout.addWidget(QLabel("VOサブサンプル間隔:"), 9, 0)
        self.vo_frame_subsample = QSpinBox()
        self.vo_frame_subsample.setRange(1, 12)
        self.vo_frame_subsample.setValue(int(self.settings.get("vo_frame_subsample", 1)))
        vo_layout.addWidget(self.vo_frame_subsample, 9, 1)

        vo_layout.addWidget(QLabel("Essential推定法:"), 10, 0)
        self.vo_essential_method = QComboBox()
        self.vo_essential_method.addItems(["auto", "ransac", "magsac"])
        self.vo_essential_method.setCurrentText(str(self.settings.get("vo_essential_method", "auto")))
        vo_layout.addWidget(self.vo_essential_method, 10, 1)

        self.vo_subpixel_refine = QCheckBox("サブピクセル補正を有効化")
        self.vo_subpixel_refine.setChecked(bool(self.settings.get("vo_subpixel_refine", True)))
        vo_layout.addWidget(self.vo_subpixel_refine, 11, 0, 1, 3)

        self.vo_adaptive_subsample = QCheckBox("動的サブサンプリングを有効化")
        self.vo_adaptive_subsample.setChecked(bool(self.settings.get("vo_adaptive_subsample", False)))
        vo_layout.addWidget(self.vo_adaptive_subsample, 12, 0, 1, 3)

        vo_layout.addWidget(QLabel("VO最小サブサンプル:"), 13, 0)
        self.vo_subsample_min = QSpinBox()
        self.vo_subsample_min.setRange(1, 10)
        self.vo_subsample_min.setValue(int(self.settings.get("vo_subsample_min", 1)))
        vo_layout.addWidget(self.vo_subsample_min, 13, 1)

        vo_layout.addWidget(QLabel("VO信頼度 low/mid:"), 14, 0)
        conf_row = QHBoxLayout()
        self.vo_confidence_low_threshold = QDoubleSpinBox()
        self.vo_confidence_low_threshold.setRange(0.0, 1.0)
        self.vo_confidence_low_threshold.setSingleStep(0.01)
        self.vo_confidence_low_threshold.setDecimals(2)
        self.vo_confidence_low_threshold.setValue(float(self.settings.get("vo_confidence_low_threshold", 0.35)))
        conf_row.addWidget(self.vo_confidence_low_threshold)
        self.vo_confidence_mid_threshold = QDoubleSpinBox()
        self.vo_confidence_mid_threshold.setRange(0.0, 1.0)
        self.vo_confidence_mid_threshold.setSingleStep(0.01)
        self.vo_confidence_mid_threshold.setDecimals(2)
        self.vo_confidence_mid_threshold.setValue(float(self.settings.get("vo_confidence_mid_threshold", 0.55)))
        conf_row.addWidget(self.vo_confidence_mid_threshold)
        vo_layout.addLayout(conf_row, 14, 1, 1, 2)

        self.calib_check_button = QPushButton("Calibration Check を実行")
        self.calib_check_button.clicked.connect(self._run_calibration_check_from_gui)
        vo_layout.addWidget(self.calib_check_button, 15, 0, 1, 3)

        vo_group.setLayout(vo_layout)
        vo_page_layout.addWidget(vo_group)

        pose_group = QGroupBox("Pose Backend")
        pose_layout = QGridLayout()

        pose_layout.addWidget(QLabel("Pose backend:"), 0, 0)
        self.pose_backend = QComboBox()
        self.pose_backend.addItems(["vo", "colmap"])
        self.pose_backend.setCurrentText(str(self.settings.get("pose_backend", "vo")))
        pose_layout.addWidget(self.pose_backend, 0, 1)

        pose_layout.addWidget(QLabel("Pose export format:"), 1, 0)
        self.pose_export_format = QComboBox()
        self.pose_export_format.addItems(["internal", "metashape"])
        self.pose_export_format.setCurrentText(str(self.settings.get("pose_export_format", "internal")))
        pose_layout.addWidget(self.pose_export_format, 1, 1)

        pose_layout.addWidget(QLabel("COLMAP path:"), 2, 0)
        self.colmap_path = QLineEdit(str(self.settings.get("colmap_path", "colmap")))
        pose_layout.addWidget(self.colmap_path, 2, 1)
        btn_colmap = QPushButton("参照")
        btn_colmap.clicked.connect(lambda: self._pick_file_for_line_edit(self.colmap_path))
        pose_layout.addWidget(btn_colmap, 2, 2)

        pose_layout.addWidget(QLabel("COLMAP workspace:"), 3, 0)
        self.colmap_workspace = QLineEdit(str(self.settings.get("colmap_workspace", "")))
        pose_layout.addWidget(self.colmap_workspace, 3, 1)
        btn_colmap_ws = QPushButton("参照")
        btn_colmap_ws.clicked.connect(lambda: self._pick_dir_for_line_edit(self.colmap_workspace))
        pose_layout.addWidget(btn_colmap_ws, 3, 2)

        pose_layout.addWidget(QLabel("COLMAP database.db:"), 4, 0)
        self.colmap_db_path = QLineEdit(str(self.settings.get("colmap_db_path", "")))
        pose_layout.addWidget(self.colmap_db_path, 4, 1)
        btn_colmap_db = QPushButton("参照")
        btn_colmap_db.clicked.connect(lambda: self._pick_file_for_line_edit(self.colmap_db_path))
        pose_layout.addWidget(btn_colmap_db, 4, 2)

        pose_layout.addWidget(QLabel("COLMAP pipeline mode:"), 5, 0)
        self.colmap_pipeline_mode = QComboBox()
        self.colmap_pipeline_mode.addItems(["minimal_v1", "legacy"])
        self.colmap_pipeline_mode.setCurrentText(str(self.settings.get("colmap_pipeline_mode", "minimal_v1")))
        pose_layout.addWidget(self.colmap_pipeline_mode, 5, 1)

        self.colmap_minimal_info_label = QLabel("")
        self.colmap_minimal_info_label.setWordWrap(True)
        self.colmap_minimal_info_label.setStyleSheet("color: #8b5e00; font-size: 11px;")
        pose_layout.addWidget(self.colmap_minimal_info_label, 6, 0, 1, 3)

        pose_layout.addWidget(QLabel("COLMAP keyframe policy:"), 7, 0)
        self.colmap_keyframe_policy = QComboBox()
        self.colmap_keyframe_policy.addItems(["legacy", "stage2_relaxed", "stage1_only"])
        self.colmap_keyframe_policy.setCurrentText(str(self.settings.get("colmap_keyframe_policy", "stage2_relaxed")))
        pose_layout.addWidget(self.colmap_keyframe_policy, 7, 1)

        pose_layout.addWidget(QLabel("COLMAP target mode:"), 8, 0)
        self.colmap_keyframe_target_mode = QComboBox()
        self.colmap_keyframe_target_mode.addItems(["auto", "fixed"])
        self.colmap_keyframe_target_mode.setCurrentText(str(self.settings.get("colmap_keyframe_target_mode", "auto")))
        pose_layout.addWidget(self.colmap_keyframe_target_mode, 8, 1)

        pose_layout.addWidget(QLabel("COLMAP target min/max:"), 9, 0)
        target_row = QHBoxLayout()
        self.colmap_keyframe_target_min = QSpinBox()
        self.colmap_keyframe_target_min.setRange(1, 100000)
        self.colmap_keyframe_target_min.setValue(int(self.settings.get("colmap_keyframe_target_min", 120)))
        target_row.addWidget(self.colmap_keyframe_target_min)
        self.colmap_keyframe_target_max = QSpinBox()
        self.colmap_keyframe_target_max.setRange(1, 100000)
        self.colmap_keyframe_target_max.setValue(int(self.settings.get("colmap_keyframe_target_max", 240)))
        target_row.addWidget(self.colmap_keyframe_target_max)
        pose_layout.addLayout(target_row, 9, 1, 1, 2)

        pose_layout.addWidget(QLabel("COLMAP NMS窓(sec):"), 10, 0)
        self.colmap_nms_window_sec = QDoubleSpinBox()
        self.colmap_nms_window_sec.setRange(0.01, 30.0)
        self.colmap_nms_window_sec.setSingleStep(0.05)
        self.colmap_nms_window_sec.setDecimals(2)
        self.colmap_nms_window_sec.setValue(float(self.settings.get("colmap_nms_window_sec", 0.35)))
        pose_layout.addWidget(self.colmap_nms_window_sec, 10, 1)

        self.colmap_enable_stage0 = QCheckBox("COLMAP時もStage0を有効化")
        self.colmap_enable_stage0.setChecked(bool(self.settings.get("colmap_enable_stage0", True)))
        pose_layout.addWidget(self.colmap_enable_stage0, 11, 0, 1, 3)

        self.colmap_motion_aware_selection = QCheckBox("COLMAP motion-aware選択を有効化")
        self.colmap_motion_aware_selection.setChecked(bool(self.settings.get("colmap_motion_aware_selection", True)))
        pose_layout.addWidget(self.colmap_motion_aware_selection, 12, 0, 1, 3)

        pose_layout.addWidget(QLabel("COLMAP motion窓倍率:"), 13, 0)
        self.colmap_nms_motion_window_ratio = QDoubleSpinBox()
        self.colmap_nms_motion_window_ratio.setRange(0.0, 10.0)
        self.colmap_nms_motion_window_ratio.setSingleStep(0.05)
        self.colmap_nms_motion_window_ratio.setDecimals(2)
        self.colmap_nms_motion_window_ratio.setValue(float(self.settings.get("colmap_nms_motion_window_ratio", 0.5)))
        pose_layout.addWidget(self.colmap_nms_motion_window_ratio, 13, 1)

        self.colmap_stage1_adaptive_threshold = QCheckBox("COLMAP Stage1適応しきい値を有効化")
        self.colmap_stage1_adaptive_threshold.setChecked(bool(self.settings.get("colmap_stage1_adaptive_threshold", True)))
        pose_layout.addWidget(self.colmap_stage1_adaptive_threshold, 14, 0, 1, 3)

        pose_layout.addWidget(QLabel("Stage1 bin下限:"), 15, 0)
        self.colmap_stage1_min_candidates_per_bin = QSpinBox()
        self.colmap_stage1_min_candidates_per_bin.setRange(0, 1000)
        self.colmap_stage1_min_candidates_per_bin.setValue(int(self.settings.get("colmap_stage1_min_candidates_per_bin", 3)))
        pose_layout.addWidget(self.colmap_stage1_min_candidates_per_bin, 15, 1)

        pose_layout.addWidget(QLabel("Stage1候補上限:"), 16, 0)
        self.colmap_stage1_max_candidates = QSpinBox()
        self.colmap_stage1_max_candidates.setRange(1, 100000)
        self.colmap_stage1_max_candidates.setValue(int(self.settings.get("colmap_stage1_max_candidates", 360)))
        pose_layout.addWidget(self.colmap_stage1_max_candidates, 16, 1)

        pose_layout.addWidget(QLabel("COLMAP rig policy:"), 17, 0)
        self.colmap_rig_policy = QComboBox()
        self.colmap_rig_policy.addItems(["lr_opk", "off"])
        self.colmap_rig_policy.setCurrentText(str(self.settings.get("colmap_rig_policy", "lr_opk")))
        pose_layout.addWidget(self.colmap_rig_policy, 17, 1)

        pose_layout.addWidget(QLabel("COLMAP rig seed OPK:"), 18, 0)
        seed = self.settings.get("colmap_rig_seed_opk_deg", [0.0, 0.0, 180.0])
        if isinstance(seed, (list, tuple)) and len(seed) == 3:
            seed_text = f"{float(seed[0]):.6g},{float(seed[1]):.6g},{float(seed[2]):.6g}"
        else:
            seed_text = "0,0,180"
        self.colmap_rig_seed_opk_deg = QLineEdit(seed_text)
        pose_layout.addWidget(self.colmap_rig_seed_opk_deg, 18, 1)

        pose_layout.addWidget(QLabel("COLMAP workspace scope:"), 19, 0)
        self.colmap_workspace_scope = QComboBox()
        self.colmap_workspace_scope.addItems(["run_scoped", "shared"])
        self.colmap_workspace_scope.setCurrentText(str(self.settings.get("colmap_workspace_scope", "run_scoped")))
        pose_layout.addWidget(self.colmap_workspace_scope, 19, 1)

        self.colmap_reuse_db = QCheckBox("COLMAP DBを再利用する")
        self.colmap_reuse_db.setChecked(bool(self.settings.get("colmap_reuse_db", False)))
        pose_layout.addWidget(self.colmap_reuse_db, 20, 0, 1, 3)

        pose_layout.addWidget(QLabel("COLMAP解析マスク:"), 21, 0)
        self.colmap_analysis_mask_profile = QComboBox()
        self.colmap_analysis_mask_profile.addItems(["colmap_safe", "legacy"])
        self.colmap_analysis_mask_profile.setCurrentText(str(self.settings.get("colmap_analysis_mask_profile", "colmap_safe")))
        pose_layout.addWidget(self.colmap_analysis_mask_profile, 21, 1)

        pose_layout.addWidget(QLabel("COLMAP selection profile:"), 28, 0)
        self.colmap_selection_profile = QComboBox()
        self.colmap_selection_profile.addItems(["no_vo_coverage", "legacy"])
        self.colmap_selection_profile.setCurrentText(str(self.settings.get("colmap_selection_profile", "no_vo_coverage")))
        pose_layout.addWidget(self.colmap_selection_profile, 28, 1)

        pose_layout.addWidget(QLabel("Stage2入口 budget / min_gap:"), 29, 0)
        stage15_row = QHBoxLayout()
        self.colmap_stage2_entry_budget = QSpinBox()
        self.colmap_stage2_entry_budget.setRange(1, 100000)
        self.colmap_stage2_entry_budget.setValue(int(self.settings.get("colmap_stage2_entry_budget", 180)))
        stage15_row.addWidget(self.colmap_stage2_entry_budget)
        self.colmap_stage2_entry_min_gap = QSpinBox()
        self.colmap_stage2_entry_min_gap.setRange(0, 300)
        self.colmap_stage2_entry_min_gap.setValue(int(self.settings.get("colmap_stage2_entry_min_gap", 3)))
        stage15_row.addWidget(self.colmap_stage2_entry_min_gap)
        pose_layout.addLayout(stage15_row, 29, 1, 1, 2)

        pose_layout.addWidget(QLabel("多様性SSIM / pHash:"), 30, 0)
        diversity_row = QHBoxLayout()
        self.colmap_diversity_ssim_threshold = QDoubleSpinBox()
        self.colmap_diversity_ssim_threshold.setRange(0.0, 1.0)
        self.colmap_diversity_ssim_threshold.setSingleStep(0.01)
        self.colmap_diversity_ssim_threshold.setDecimals(2)
        self.colmap_diversity_ssim_threshold.setValue(float(self.settings.get("colmap_diversity_ssim_threshold", 0.93)))
        diversity_row.addWidget(self.colmap_diversity_ssim_threshold)
        self.colmap_diversity_phash_hamming = QSpinBox()
        self.colmap_diversity_phash_hamming.setRange(0, 64)
        self.colmap_diversity_phash_hamming.setValue(int(self.settings.get("colmap_diversity_phash_hamming", 10)))
        diversity_row.addWidget(self.colmap_diversity_phash_hamming)
        pose_layout.addLayout(diversity_row, 30, 1, 1, 2)

        pose_layout.addWidget(QLabel("Final target policy:"), 31, 0)
        self.colmap_final_target_policy = QComboBox()
        self.colmap_final_target_policy.addItems(["soft_auto", "fixed"])
        self.colmap_final_target_policy.setCurrentText(str(self.settings.get("colmap_final_target_policy", "soft_auto")))
        pose_layout.addWidget(self.colmap_final_target_policy, 31, 1)

        pose_layout.addWidget(QLabel("Final soft min/max:"), 32, 0)
        final_soft_row = QHBoxLayout()
        self.colmap_final_soft_min = QSpinBox()
        self.colmap_final_soft_min.setRange(1, 100000)
        self.colmap_final_soft_min.setValue(int(self.settings.get("colmap_final_soft_min", 80)))
        final_soft_row.addWidget(self.colmap_final_soft_min)
        self.colmap_final_soft_max = QSpinBox()
        self.colmap_final_soft_max.setRange(1, 100000)
        self.colmap_final_soft_max.setValue(int(self.settings.get("colmap_final_soft_max", 220)))
        final_soft_row.addWidget(self.colmap_final_soft_max)
        pose_layout.addLayout(final_soft_row, 32, 1, 1, 2)

        self.colmap_no_supplement_on_low_quality = QCheckBox("低品質フレームでの補充を禁止")
        self.colmap_no_supplement_on_low_quality.setChecked(
            bool(self.settings.get("colmap_no_supplement_on_low_quality", True))
        )
        pose_layout.addWidget(self.colmap_no_supplement_on_low_quality, 33, 0, 1, 3)

        pose_layout.addWidget(QLabel("並進しきい値:"), 22, 0)
        self.pose_select_translation_threshold = QDoubleSpinBox()
        self.pose_select_translation_threshold.setRange(0.0, 50.0)
        self.pose_select_translation_threshold.setSingleStep(0.1)
        self.pose_select_translation_threshold.setDecimals(2)
        self.pose_select_translation_threshold.setValue(float(self.settings.get("pose_select_translation_threshold", 1.2)))
        pose_layout.addWidget(self.pose_select_translation_threshold, 22, 1)

        pose_layout.addWidget(QLabel("回転しきい値(deg):"), 23, 0)
        self.pose_select_rotation_threshold_deg = QDoubleSpinBox()
        self.pose_select_rotation_threshold_deg.setRange(0.0, 180.0)
        self.pose_select_rotation_threshold_deg.setSingleStep(0.5)
        self.pose_select_rotation_threshold_deg.setDecimals(2)
        self.pose_select_rotation_threshold_deg.setValue(float(self.settings.get("pose_select_rotation_threshold_deg", 5.0)))
        pose_layout.addWidget(self.pose_select_rotation_threshold_deg, 23, 1)

        pose_layout.addWidget(QLabel("最小観測点数:"), 24, 0)
        self.pose_select_min_observations = QSpinBox()
        self.pose_select_min_observations.setRange(0, 100000)
        self.pose_select_min_observations.setValue(int(self.settings.get("pose_select_min_observations", 30)))
        pose_layout.addWidget(self.pose_select_min_observations, 24, 1)

        self.pose_select_enable_translation = QCheckBox("並進条件を有効化")
        self.pose_select_enable_translation.setChecked(bool(self.settings.get("pose_select_enable_translation", True)))
        pose_layout.addWidget(self.pose_select_enable_translation, 25, 0, 1, 3)

        self.pose_select_enable_rotation = QCheckBox("回転条件を有効化")
        self.pose_select_enable_rotation.setChecked(bool(self.settings.get("pose_select_enable_rotation", True)))
        pose_layout.addWidget(self.pose_select_enable_rotation, 26, 0, 1, 3)

        self.pose_select_enable_observations = QCheckBox("観測点数条件を有効化")
        self.pose_select_enable_observations.setChecked(bool(self.settings.get("pose_select_enable_observations", False)))
        pose_layout.addWidget(self.pose_select_enable_observations, 27, 0, 1, 3)

        self.pose_backend.currentTextChanged.connect(self._on_pose_backend_changed)
        self.colmap_pipeline_mode.currentTextChanged.connect(
            lambda _text: self._on_pose_backend_changed(self.pose_backend.currentText())
        )
        self._on_pose_backend_changed(self.pose_backend.currentText())

        pose_group.setLayout(pose_layout)
        vo_page_layout.addWidget(pose_group)

        perf_group = QGroupBox("性能設定")
        perf_layout = QGridLayout()

        perf_layout.addWidget(QLabel("OpenCVスレッド数 (0=auto):"), 0, 0)
        self.opencv_thread_count = QSpinBox()
        self.opencv_thread_count.setRange(0, 128)
        self.opencv_thread_count.setValue(int(self.settings.get("opencv_thread_count", 0)))
        perf_layout.addWidget(self.opencv_thread_count, 0, 1)

        perf_layout.addWidget(QLabel("Stage1プロセス数 (0=auto):"), 1, 0)
        self.stage1_process_workers = QSpinBox()
        self.stage1_process_workers.setRange(0, 128)
        self.stage1_process_workers.setValue(int(self.settings.get("stage1_process_workers", 0)))
        perf_layout.addWidget(self.stage1_process_workers, 1, 1)

        perf_layout.addWidget(QLabel("Stage1プリフェッチサイズ:"), 2, 0)
        self.stage1_prefetch_size = QSpinBox()
        self.stage1_prefetch_size.setRange(1, 256)
        self.stage1_prefetch_size.setValue(int(self.settings.get("stage1_prefetch_size", 32)))
        perf_layout.addWidget(self.stage1_prefetch_size, 2, 1)

        perf_layout.addWidget(QLabel("Stage1品質バッチサイズ:"), 3, 0)
        self.stage1_metrics_batch_size = QSpinBox()
        self.stage1_metrics_batch_size.setRange(1, 512)
        self.stage1_metrics_batch_size.setValue(int(self.settings.get("stage1_metrics_batch_size", 64)))
        perf_layout.addWidget(self.stage1_metrics_batch_size, 3, 1)

        self.stage1_gpu_batch_enabled = QCheckBox("Stage1でGPUバッチ品質計算を有効化")
        self.stage1_gpu_batch_enabled.setChecked(bool(self.settings.get("stage1_gpu_batch_enabled", True)))
        perf_layout.addWidget(self.stage1_gpu_batch_enabled, 4, 0, 1, 3)

        perf_layout.addWidget(QLabel("macOS Capture Backend:"), 5, 0)
        self.darwin_capture_backend = QComboBox()
        self.darwin_capture_backend.addItems(["auto", "avfoundation", "ffmpeg"])
        self.darwin_capture_backend.setCurrentText(str(self.settings.get("darwin_capture_backend", "auto")))
        perf_layout.addWidget(self.darwin_capture_backend, 5, 1)

        perf_layout.addWidget(QLabel("MPS最小画素数:"), 6, 0)
        self.mps_min_pixels = QSpinBox()
        self.mps_min_pixels.setRange(1, 100000000)
        self.mps_min_pixels.setSingleStep(1024)
        self.mps_min_pixels.setValue(int(self.settings.get("mps_min_pixels", 256 * 256)))
        perf_layout.addWidget(self.mps_min_pixels, 6, 1)

        perf_group.setLayout(perf_layout)
        perf_page_layout.addWidget(perf_group)

        stage_page_layout.addStretch()
        vo_page_layout.addStretch()
        perf_page_layout.addStretch()
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

        # === 魚眼外周マスク ===
        fisheye_group = QGroupBox("魚眼外周マスク（OSV/前後魚眼）")
        fisheye_layout = QGridLayout()

        self.enable_fisheye_border_mask = QCheckBox("魚眼外周マスクを有効化")
        self.enable_fisheye_border_mask.setChecked(
            bool(self.settings.get('enable_fisheye_border_mask', True))
        )
        fisheye_layout.addWidget(self.enable_fisheye_border_mask, 0, 0, 1, 2)

        fisheye_layout.addWidget(QLabel("有効領域半径比 (0-1):"), 1, 0)
        self.fisheye_mask_radius_ratio = QDoubleSpinBox()
        self.fisheye_mask_radius_ratio.setMinimum(0.0)
        self.fisheye_mask_radius_ratio.setMaximum(1.0)
        self.fisheye_mask_radius_ratio.setSingleStep(0.01)
        self.fisheye_mask_radius_ratio.setDecimals(2)
        self.fisheye_mask_radius_ratio.setValue(
            float(self.settings.get('fisheye_mask_radius_ratio', 0.94))
        )
        fisheye_layout.addWidget(self.fisheye_mask_radius_ratio, 1, 1)

        fisheye_layout.addWidget(QLabel("中心オフセットX (px):"), 2, 0)
        self.fisheye_mask_center_offset_x = QSpinBox()
        self.fisheye_mask_center_offset_x.setMinimum(-2000)
        self.fisheye_mask_center_offset_x.setMaximum(2000)
        self.fisheye_mask_center_offset_x.setValue(
            int(self.settings.get('fisheye_mask_center_offset_x', 0))
        )
        fisheye_layout.addWidget(self.fisheye_mask_center_offset_x, 2, 1)

        fisheye_layout.addWidget(QLabel("中心オフセットY (px):"), 3, 0)
        self.fisheye_mask_center_offset_y = QSpinBox()
        self.fisheye_mask_center_offset_y.setMinimum(-2000)
        self.fisheye_mask_center_offset_y.setMaximum(2000)
        self.fisheye_mask_center_offset_y.setValue(
            int(self.settings.get('fisheye_mask_center_offset_y', 0))
        )
        fisheye_layout.addWidget(self.fisheye_mask_center_offset_y, 3, 1)

        fisheye_group.setLayout(fisheye_layout)
        layout.addWidget(fisheye_group)

        # === 魚眼分割出力（cross5） ===
        split_group = QGroupBox("魚眼分割出力（OSV cross5）")
        split_layout = QGridLayout()

        self.enable_split_views = QCheckBox("分割画像を出力する")
        self.enable_split_views.setChecked(bool(self.settings.get("enable_split_views", True)))
        split_layout.addWidget(self.enable_split_views, 0, 0, 1, 2)

        split_layout.addWidget(QLabel("分割サイズ (px):"), 1, 0)
        self.split_view_size = QSpinBox()
        self.split_view_size.setMinimum(128)
        self.split_view_size.setMaximum(8192)
        self.split_view_size.setValue(int(self.settings.get("split_view_size", 1600)))
        split_layout.addWidget(self.split_view_size, 1, 1)

        split_layout.addWidget(QLabel("HFOV (deg):"), 2, 0)
        self.split_view_hfov = QDoubleSpinBox()
        self.split_view_hfov.setMinimum(1.0)
        self.split_view_hfov.setMaximum(179.0)
        self.split_view_hfov.setSingleStep(1.0)
        self.split_view_hfov.setDecimals(1)
        self.split_view_hfov.setValue(float(self.settings.get("split_view_hfov", 80.0)))
        split_layout.addWidget(self.split_view_hfov, 2, 1)

        split_layout.addWidget(QLabel("VFOV (deg):"), 3, 0)
        self.split_view_vfov = QDoubleSpinBox()
        self.split_view_vfov.setMinimum(1.0)
        self.split_view_vfov.setMaximum(179.0)
        self.split_view_vfov.setSingleStep(1.0)
        self.split_view_vfov.setDecimals(1)
        self.split_view_vfov.setValue(float(self.settings.get("split_view_vfov", 80.0)))
        split_layout.addWidget(self.split_view_vfov, 3, 1)

        split_layout.addWidget(QLabel("cross yaw (deg):"), 4, 0)
        self.split_cross_yaw_deg = QDoubleSpinBox()
        self.split_cross_yaw_deg.setMinimum(0.0)
        self.split_cross_yaw_deg.setMaximum(180.0)
        self.split_cross_yaw_deg.setSingleStep(0.5)
        self.split_cross_yaw_deg.setDecimals(1)
        self.split_cross_yaw_deg.setValue(float(self.settings.get("split_cross_yaw_deg", 50.5)))
        split_layout.addWidget(self.split_cross_yaw_deg, 4, 1)

        split_layout.addWidget(QLabel("cross pitch (deg):"), 5, 0)
        self.split_cross_pitch_deg = QDoubleSpinBox()
        self.split_cross_pitch_deg.setMinimum(0.0)
        self.split_cross_pitch_deg.setMaximum(180.0)
        self.split_cross_pitch_deg.setSingleStep(0.5)
        self.split_cross_pitch_deg.setDecimals(1)
        self.split_cross_pitch_deg.setValue(float(self.settings.get("split_cross_pitch_deg", 50.5)))
        split_layout.addWidget(self.split_cross_pitch_deg, 5, 1)

        split_layout.addWidget(QLabel("cross inward (deg):"), 6, 0)
        self.split_cross_inward_deg = QDoubleSpinBox()
        self.split_cross_inward_deg.setMinimum(0.0)
        self.split_cross_inward_deg.setMaximum(90.0)
        self.split_cross_inward_deg.setSingleStep(1.0)
        self.split_cross_inward_deg.setDecimals(1)
        self.split_cross_inward_deg.setValue(float(self.settings.get("split_cross_inward_deg", 10.0)))
        split_layout.addWidget(self.split_cross_inward_deg, 6, 1)

        split_layout.addWidget(QLabel("inward up/down (deg):"), 7, 0)
        inward_ud_row = QHBoxLayout()
        self.split_inward_up_deg = QDoubleSpinBox()
        self.split_inward_up_deg.setMinimum(0.0)
        self.split_inward_up_deg.setMaximum(90.0)
        self.split_inward_up_deg.setSingleStep(1.0)
        self.split_inward_up_deg.setDecimals(1)
        self.split_inward_up_deg.setValue(float(self.settings.get("split_inward_up_deg", 25.0)))
        inward_ud_row.addWidget(self.split_inward_up_deg)
        self.split_inward_down_deg = QDoubleSpinBox()
        self.split_inward_down_deg.setMinimum(0.0)
        self.split_inward_down_deg.setMaximum(90.0)
        self.split_inward_down_deg.setSingleStep(1.0)
        self.split_inward_down_deg.setDecimals(1)
        self.split_inward_down_deg.setValue(float(self.settings.get("split_inward_down_deg", 25.0)))
        inward_ud_row.addWidget(self.split_inward_down_deg)
        split_layout.addLayout(inward_ud_row, 7, 1)

        split_layout.addWidget(QLabel("inward left/right (deg):"), 8, 0)
        inward_lr_row = QHBoxLayout()
        self.split_inward_left_deg = QDoubleSpinBox()
        self.split_inward_left_deg.setMinimum(0.0)
        self.split_inward_left_deg.setMaximum(90.0)
        self.split_inward_left_deg.setSingleStep(1.0)
        self.split_inward_left_deg.setDecimals(1)
        self.split_inward_left_deg.setValue(float(self.settings.get("split_inward_left_deg", 25.0)))
        inward_lr_row.addWidget(self.split_inward_left_deg)
        self.split_inward_right_deg = QDoubleSpinBox()
        self.split_inward_right_deg.setMinimum(0.0)
        self.split_inward_right_deg.setMaximum(90.0)
        self.split_inward_right_deg.setSingleStep(1.0)
        self.split_inward_right_deg.setDecimals(1)
        self.split_inward_right_deg.setValue(float(self.settings.get("split_inward_right_deg", 25.0)))
        inward_lr_row.addWidget(self.split_inward_right_deg)
        split_layout.addLayout(inward_lr_row, 8, 1)

        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

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

    def _create_target_mask_tab(self) -> QWidget:
        """
        対象マスクタブを作成

        Returns:
        --------
        QWidget
            タブウィジェット
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # === 対象検出 ===
        detection_group = QGroupBox("対象検出")
        detection_layout = QGridLayout()

        self.enable_target_mask_generation = QCheckBox("キーフレーム出力後に対象マスクを生成")
        self.enable_target_mask_generation.setChecked(
            self.settings.get("enable_target_mask_generation", False)
        )
        detection_layout.addWidget(self.enable_target_mask_generation, 0, 0, 1, 3)

        detection_layout.addWidget(QLabel("検出対象:"), 1, 0)
        self.target_class_checks = {}
        selected_targets = set(self.settings.get("target_classes", []))
        cls_col = 1
        cls_row = 1
        for label in TARGET_CLASS_LABELS:
            cb = QCheckBox(label)
            cb.setChecked(label in selected_targets)
            self.target_class_checks[label] = cb
            detection_layout.addWidget(cb, cls_row, cls_col)
            cls_col += 1
            if cls_col > 2:
                cls_col = 1
                cls_row += 1

        base_row = cls_row + 1
        detection_layout.addWidget(QLabel("YOLOモデル:"), base_row, 0)
        self.yolo_model_path = QComboBox()
        self.yolo_model_path.setEditable(True)
        self.yolo_model_path.addItems([
            "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"
        ])
        self.yolo_model_path.setCurrentText(self.settings.get("yolo_model_path", "yolo26n-seg.pt"))
        detection_layout.addWidget(self.yolo_model_path, base_row, 1, 1, 2)

        detection_layout.addWidget(QLabel("SAMモデル:"), base_row + 1, 0)
        self.sam_model_path = QComboBox()
        self.sam_model_path.setEditable(True)
        self.sam_model_path.addItems(["sam3_t.pt", "sam3_s.pt", "sam3_b.pt", "sam3_l.pt"])
        self.sam_model_path.setCurrentText(self.settings.get("sam_model_path", "sam3_t.pt"))
        detection_layout.addWidget(self.sam_model_path, base_row + 1, 1, 1, 2)

        detection_layout.addWidget(QLabel("信頼度閾値:"), base_row + 2, 0)
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setMinimum(0.01)
        self.confidence_threshold.setMaximum(1.0)
        self.confidence_threshold.setSingleStep(0.01)
        self.confidence_threshold.setDecimals(2)
        self.confidence_threshold.setValue(self.settings.get("confidence_threshold", 0.25))
        detection_layout.addWidget(self.confidence_threshold, base_row + 2, 1)

        detection_layout.addWidget(QLabel("推論デバイス:"), base_row + 3, 0)
        self.detection_device = QComboBox()
        self.detection_device.addItems(["auto", "cpu", "mps", "cuda", "0"])
        self.detection_device.setCurrentText(self.settings.get("detection_device", "auto"))
        detection_layout.addWidget(self.detection_device, base_row + 3, 1, 1, 2)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # === Stage2 動体除去 ===
        dynamic_group = QGroupBox("Stage2 動体除去（幾何評価）")
        dynamic_layout = QGridLayout()

        self.enable_dynamic_mask_removal = QCheckBox("Stage2で動体領域を除外する")
        self.enable_dynamic_mask_removal.setChecked(
            self.settings.get("enable_dynamic_mask_removal", False)
        )
        dynamic_layout.addWidget(self.enable_dynamic_mask_removal, 0, 0, 1, 3)

        self.dynamic_mask_use_yolo_sam = QCheckBox("YOLO/SAM検出領域を除外に使用")
        self.dynamic_mask_use_yolo_sam.setChecked(
            self.settings.get("dynamic_mask_use_yolo_sam", True)
        )
        dynamic_layout.addWidget(self.dynamic_mask_use_yolo_sam, 1, 0, 1, 3)

        self.dynamic_mask_use_motion_diff = QCheckBox("背景差分の動体領域を除外に使用")
        self.dynamic_mask_use_motion_diff.setChecked(
            self.settings.get("dynamic_mask_use_motion_diff", True)
        )
        dynamic_layout.addWidget(self.dynamic_mask_use_motion_diff, 2, 0, 1, 3)

        dynamic_layout.addWidget(QLabel("動体検出フレーム数:"), 3, 0)
        self.dynamic_mask_motion_frames = QSpinBox()
        self.dynamic_mask_motion_frames.setMinimum(2)
        self.dynamic_mask_motion_frames.setMaximum(12)
        self.dynamic_mask_motion_frames.setValue(
            int(self.settings.get("dynamic_mask_motion_frames", 3))
        )
        dynamic_layout.addWidget(self.dynamic_mask_motion_frames, 3, 1)

        dynamic_layout.addWidget(QLabel("差分しきい値:"), 4, 0)
        self.dynamic_mask_motion_threshold = QSpinBox()
        self.dynamic_mask_motion_threshold.setMinimum(1)
        self.dynamic_mask_motion_threshold.setMaximum(255)
        self.dynamic_mask_motion_threshold.setValue(
            int(self.settings.get("dynamic_mask_motion_threshold", 30))
        )
        dynamic_layout.addWidget(self.dynamic_mask_motion_threshold, 4, 1)

        dynamic_layout.addWidget(QLabel("マスク膨張サイズ:"), 5, 0)
        self.dynamic_mask_dilation_size = QSpinBox()
        self.dynamic_mask_dilation_size.setMinimum(0)
        self.dynamic_mask_dilation_size.setMaximum(51)
        self.dynamic_mask_dilation_size.setValue(
            int(self.settings.get("dynamic_mask_dilation_size", 5))
        )
        dynamic_layout.addWidget(self.dynamic_mask_dilation_size, 5, 1)

        self.dynamic_mask_inpaint_enabled = QCheckBox("インペイントフックを有効化")
        self.dynamic_mask_inpaint_enabled.setChecked(
            self.settings.get("dynamic_mask_inpaint_enabled", False)
        )
        dynamic_layout.addWidget(self.dynamic_mask_inpaint_enabled, 6, 0, 1, 3)

        dynamic_layout.addWidget(QLabel("インペイントモジュール:"), 7, 0)
        self.dynamic_mask_inpaint_module = QComboBox()
        self.dynamic_mask_inpaint_module.setEditable(True)
        self.dynamic_mask_inpaint_module.addItems(["", "processing.video_inpaint"])
        self.dynamic_mask_inpaint_module.setCurrentText(
            self.settings.get("dynamic_mask_inpaint_module", "")
        )
        dynamic_layout.addWidget(self.dynamic_mask_inpaint_module, 7, 1, 1, 2)

        dynamic_group.setLayout(dynamic_layout)
        layout.addWidget(dynamic_group)

        # === 対象マスク出力 ===
        mask_output_group = QGroupBox("対象マスク出力")
        mask_output_layout = QGridLayout()

        mask_output_layout.addWidget(QLabel("マスクフォルダ名:"), 0, 0)
        self.mask_output_dirname = QComboBox()
        self.mask_output_dirname.setEditable(True)
        self.mask_output_dirname.addItems(["masks", "mask"])
        self.mask_output_dirname.setCurrentText(self.settings.get("mask_output_dirname", "masks"))
        mask_output_layout.addWidget(self.mask_output_dirname, 0, 1)

        self.mask_add_suffix = QCheckBox("ファイル名に接尾辞を追加")
        self.mask_add_suffix.setChecked(self.settings.get("mask_add_suffix", True))
        mask_output_layout.addWidget(self.mask_add_suffix, 1, 0, 1, 2)

        mask_output_layout.addWidget(QLabel("接尾辞:"), 2, 0)
        self.mask_suffix = QComboBox()
        self.mask_suffix.setEditable(True)
        self.mask_suffix.addItems(["_mask", "_seg"])
        self.mask_suffix.setCurrentText(self.settings.get("mask_suffix", "_mask"))
        mask_output_layout.addWidget(self.mask_suffix, 2, 1)

        mask_output_layout.addWidget(QLabel("マスク形式:"), 3, 0)
        self.mask_output_format = QComboBox()
        self.mask_output_format.addItems(["same", "png", "jpg", "tiff"])
        self.mask_output_format.setCurrentText(self.settings.get("mask_output_format", "same"))
        mask_output_layout.addWidget(self.mask_output_format, 3, 1)

        mask_output_group.setLayout(mask_output_layout)
        layout.addWidget(mask_output_group)

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
        from core.config_loader import ConfigManager

        settings_file = Path.home() / ".360split" / "settings.json"
        default_settings = ConfigManager.default_config()
        default_settings.update({
            'projection_mode': 'Equirectangular',
            'enable_fisheye_border_mask': True,
            'fisheye_mask_radius_ratio': 0.94,
            'fisheye_mask_center_offset_x': 0,
            'fisheye_mask_center_offset_y': 0,
            'enable_split_views': True,
            'split_view_size': 1600,
            'split_view_hfov': 80.0,
            'split_view_vfov': 80.0,
            'split_cross_yaw_deg': 50.5,
            'split_cross_pitch_deg': 50.5,
            'split_cross_inward_deg': 10.0,
            'split_inward_up_deg': 25.0,
            'split_inward_down_deg': 25.0,
            'split_inward_left_deg': 25.0,
            'split_inward_right_deg': 25.0,
            'enable_nadir_mask': False,
            'nadir_mask_radius': 100,
            'enable_equipment_detection': False,
            'mask_dilation_size': 15,
            'output_directory': str(Path.home() / "360split_output"),
            'naming_prefix': 'keyframe',
            'enable_target_mask_generation': False,
            'target_classes': ["人物", "人", "自転車", "バイク", "車両", "動物"],
            'yolo_model_path': 'yolo26n-seg.pt',
            'sam_model_path': 'sam3_t.pt',
            'confidence_threshold': 0.25,
            'detection_device': 'auto',
            'mask_output_dirname': 'masks',
            'mask_add_suffix': True,
            'mask_suffix': '_mask',
            'mask_output_format': 'same',
            'enable_dynamic_mask_removal': False,
            'dynamic_mask_use_yolo_sam': True,
            'dynamic_mask_use_motion_diff': True,
            'dynamic_mask_motion_frames': 3,
            'dynamic_mask_motion_threshold': 30,
            'dynamic_mask_dilation_size': 5,
            'dynamic_mask_target_classes': ["人物", "人", "自転車", "バイク", "車両", "動物"],
            'dynamic_mask_inpaint_enabled': False,
            'dynamic_mask_inpaint_module': '',
            'stage1_grab_threshold': 30,
            'stage1_eval_scale': 0.5,
            'opencv_thread_count': 0,
            'stage1_process_workers': 0,
            'stage1_prefetch_size': 32,
            'stage1_metrics_batch_size': 64,
            'stage1_gpu_batch_enabled': True,
            'darwin_capture_backend': 'auto',
            'mps_min_pixels': 256 * 256,
            'quality_filter_enabled': True,
            'quality_threshold': 0.50,
            'quality_roi_mode': 'circle',
            'quality_roi_ratio': 0.40,
            'quality_abs_laplacian_min': 35.0,
            'quality_use_orb': True,
            'quality_weight_sharpness': 0.40,
            'quality_weight_tenengrad': 0.30,
            'quality_weight_exposure': 0.15,
            'quality_weight_keypoints': 0.15,
            'quality_norm_p_low': 10.0,
            'quality_norm_p_high': 90.0,
            'quality_debug': False,
            'stage1_lr_merge_mode': 'asymmetric_sky_v1',
            'stage1_lr_asym_weak_floor': 0.35,
            'stage1_lr_sky_ratio_threshold': 0.55,
            'stage1_lr_sky_ratio_diff_threshold': 0.20,
            'stage1_lr_quality_gap_threshold': 0.15,
            'stage1_lr_semantic_sky_enabled': True,
            'enable_stage0_scan': True,
            'stage0_stride': 5,
            'enable_stage3_refinement': True,
            'stage3_weight_base': 0.70,
            'stage3_weight_trajectory': 0.25,
            'stage3_weight_stage0_risk': 0.05,
            'keep_temp_on_success': True,
            'vo_enabled': True,
            'vo_center_roi_ratio': 0.6,
            'vo_downscale_long_edge': 1000,
            'vo_max_features': 600,
            'vo_frame_subsample': 1,
            'vo_essential_method': 'auto',
            'vo_subpixel_refine': True,
            'vo_adaptive_subsample': False,
            'vo_subsample_min': 1,
            'vo_confidence_low_threshold': 0.35,
            'vo_confidence_mid_threshold': 0.55,
            'calib_xml': '',
            'front_calib_xml': '',
            'rear_calib_xml': '',
            'calib_model': 'auto',
            'pose_backend': 'vo',
            'colmap_path': 'colmap',
            'colmap_workspace': '',
            'colmap_db_path': '',
            'colmap_pipeline_mode': 'minimal_v1',
            'colmap_keyframe_policy': 'stage2_relaxed',
            'colmap_keyframe_target_mode': 'auto',
            'colmap_keyframe_target_min': 120,
            'colmap_keyframe_target_max': 240,
            'colmap_nms_window_sec': 0.35,
            'colmap_enable_stage0': True,
            'colmap_motion_aware_selection': True,
            'colmap_nms_motion_window_ratio': 0.5,
            'colmap_stage1_adaptive_threshold': True,
            'colmap_stage1_min_candidates_per_bin': 3,
            'colmap_stage1_max_candidates': 360,
            'colmap_selection_profile': 'no_vo_coverage',
            'colmap_stage2_entry_budget': 180,
            'colmap_stage2_entry_min_gap': 3,
            'colmap_diversity_ssim_threshold': 0.93,
            'colmap_diversity_phash_hamming': 10,
            'colmap_final_target_policy': 'soft_auto',
            'colmap_final_soft_min': 80,
            'colmap_final_soft_max': 220,
            'colmap_no_supplement_on_low_quality': True,
            'colmap_rig_policy': 'lr_opk',
            'colmap_rig_seed_opk_deg': [0.0, 0.0, 180.0],
            'colmap_workspace_scope': 'run_scoped',
            'colmap_reuse_db': False,
            'colmap_analysis_mask_profile': 'colmap_safe',
            'pose_export_format': 'internal',
            'pose_select_translation_threshold': 1.2,
            'pose_select_rotation_threshold_deg': 5.0,
            'pose_select_min_observations': 30,
            'pose_select_enable_translation': True,
            'pose_select_enable_rotation': True,
            'pose_select_enable_observations': False,
        })

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

        seed_text = self.colmap_rig_seed_opk_deg.text().strip()
        seed_parts = [p.strip() for p in seed_text.split(",") if p.strip()]
        if len(seed_parts) == 3:
            try:
                rig_seed_opk = [float(seed_parts[0]), float(seed_parts[1]), float(seed_parts[2])]
            except ValueError:
                rig_seed_opk = [0.0, 0.0, 180.0]
        else:
            rig_seed_opk = [0.0, 0.0, 180.0]

        # UI から設定値を取得
        settings_to_save = {
            'weight_sharpness': self.sharpness_weight_slider.value() / 100,
            'weight_exposure': self.exposure_weight_slider.value() / 100,
            'weight_geometric': self.geometric_weight_slider.value() / 100,
            'weight_content': self.content_weight_slider.value() / 100,
            'ssim_threshold': self.ssim_threshold.value(),
            'quality_filter_enabled': self.quality_filter_enabled.isChecked(),
            'quality_threshold': self.quality_threshold.value(),
            'stage1_lr_merge_mode': self.stage1_lr_merge_mode.currentText(),
            'stage1_lr_asym_weak_floor': self.stage1_lr_asym_weak_floor.value(),
            'stage1_lr_sky_ratio_threshold': float(self.settings.get("stage1_lr_sky_ratio_threshold", 0.55)),
            'stage1_lr_sky_ratio_diff_threshold': float(self.settings.get("stage1_lr_sky_ratio_diff_threshold", 0.20)),
            'stage1_lr_quality_gap_threshold': float(self.settings.get("stage1_lr_quality_gap_threshold", 0.15)),
            'stage1_lr_semantic_sky_enabled': bool(self.settings.get("stage1_lr_semantic_sky_enabled", True)),
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
            'enable_fisheye_border_mask': self.enable_fisheye_border_mask.isChecked(),
            'fisheye_mask_radius_ratio': self.fisheye_mask_radius_ratio.value(),
            'fisheye_mask_center_offset_x': self.fisheye_mask_center_offset_x.value(),
            'fisheye_mask_center_offset_y': self.fisheye_mask_center_offset_y.value(),
            'enable_split_views': self.enable_split_views.isChecked(),
            'split_view_size': self.split_view_size.value(),
            'split_view_hfov': self.split_view_hfov.value(),
            'split_view_vfov': self.split_view_vfov.value(),
            'split_cross_yaw_deg': self.split_cross_yaw_deg.value(),
            'split_cross_pitch_deg': self.split_cross_pitch_deg.value(),
            'split_cross_inward_deg': self.split_cross_inward_deg.value(),
            'split_inward_up_deg': self.split_inward_up_deg.value(),
            'split_inward_down_deg': self.split_inward_down_deg.value(),
            'split_inward_left_deg': self.split_inward_left_deg.value(),
            'split_inward_right_deg': self.split_inward_right_deg.value(),
            'enable_nadir_mask': self.enable_nadir_mask.isChecked(),
            'nadir_mask_radius': self.nadir_mask_radius.value(),
            'enable_equipment_detection': self.enable_equipment_detection.isChecked(),
            'mask_dilation_size': self.mask_dilation.value(),
            'enable_rerun_logging': self.enable_rerun_logging.isChecked(),
            'enable_target_mask_generation': self.enable_target_mask_generation.isChecked(),
            'target_classes': [
                label for label, cb in self.target_class_checks.items() if cb.isChecked()
            ],
            'yolo_model_path': self.yolo_model_path.currentText().strip(),
            'sam_model_path': self.sam_model_path.currentText().strip(),
            'confidence_threshold': self.confidence_threshold.value(),
            'detection_device': self.detection_device.currentText(),
            'output_image_format': self.output_format.currentText().lower(),
            'output_jpeg_quality': self.jpeg_quality.value(),
            'output_directory': self.output_dir_label.text(),
            'naming_prefix': self.naming_prefix.currentText(),
            'mask_output_dirname': self.mask_output_dirname.currentText().strip(),
            'mask_add_suffix': self.mask_add_suffix.isChecked(),
            'mask_suffix': self.mask_suffix.currentText().strip(),
            'mask_output_format': self.mask_output_format.currentText(),
            'enable_dynamic_mask_removal': self.enable_dynamic_mask_removal.isChecked(),
            'dynamic_mask_use_yolo_sam': self.dynamic_mask_use_yolo_sam.isChecked(),
            'dynamic_mask_use_motion_diff': self.dynamic_mask_use_motion_diff.isChecked(),
            'dynamic_mask_motion_frames': self.dynamic_mask_motion_frames.value(),
            'dynamic_mask_motion_threshold': self.dynamic_mask_motion_threshold.value(),
            'dynamic_mask_dilation_size': self.dynamic_mask_dilation_size.value(),
            'dynamic_mask_target_classes': [
                label for label, cb in self.target_class_checks.items() if cb.isChecked()
            ],
            'dynamic_mask_inpaint_enabled': self.dynamic_mask_inpaint_enabled.isChecked(),
            'dynamic_mask_inpaint_module': self.dynamic_mask_inpaint_module.currentText().strip(),
            'stage1_grab_threshold': int(self.settings.get('stage1_grab_threshold', 30)),
            'stage1_eval_scale': float(self.settings.get('stage1_eval_scale', 0.5)),
            'opencv_thread_count': self.opencv_thread_count.value(),
            'stage1_process_workers': self.stage1_process_workers.value(),
            'stage1_prefetch_size': self.stage1_prefetch_size.value(),
            'stage1_metrics_batch_size': self.stage1_metrics_batch_size.value(),
            'stage1_gpu_batch_enabled': self.stage1_gpu_batch_enabled.isChecked(),
            'darwin_capture_backend': self.darwin_capture_backend.currentText(),
            'mps_min_pixels': self.mps_min_pixels.value(),
            'enable_stage0_scan': self.enable_stage0_scan.isChecked(),
            'stage0_stride': self.stage0_stride.value(),
            'enable_stage3_refinement': self.enable_stage3_refinement.isChecked(),
            'stage3_weight_base': self.stage3_weight_base.value(),
            'stage3_weight_trajectory': self.stage3_weight_trajectory.value(),
            'stage3_weight_stage0_risk': self.stage3_weight_stage0_risk.value(),
            'keep_temp_on_success': bool(self.settings.get('keep_temp_on_success', True)),
            'vo_enabled': self.vo_enabled.isChecked(),
            'vo_center_roi_ratio': self.vo_center_roi_ratio.value(),
            'vo_downscale_long_edge': self.vo_downscale_long_edge.value(),
            'vo_max_features': self.vo_max_features.value(),
            'vo_frame_subsample': self.vo_frame_subsample.value(),
            'vo_essential_method': self.vo_essential_method.currentText(),
            'vo_subpixel_refine': self.vo_subpixel_refine.isChecked(),
            'vo_adaptive_subsample': self.vo_adaptive_subsample.isChecked(),
            'vo_subsample_min': self.vo_subsample_min.value(),
            'vo_confidence_low_threshold': self.vo_confidence_low_threshold.value(),
            'vo_confidence_mid_threshold': self.vo_confidence_mid_threshold.value(),
            'calib_xml': self.calib_xml.text().strip(),
            'front_calib_xml': self.front_calib_xml.text().strip(),
            'rear_calib_xml': self.rear_calib_xml.text().strip(),
            'calib_model': self.calib_model.currentText(),
            'pose_backend': self.pose_backend.currentText(),
            'colmap_path': self.colmap_path.text().strip(),
            'colmap_workspace': self.colmap_workspace.text().strip(),
            'colmap_db_path': self.colmap_db_path.text().strip(),
            'colmap_pipeline_mode': self.colmap_pipeline_mode.currentText(),
            'colmap_keyframe_policy': self.colmap_keyframe_policy.currentText(),
            'colmap_keyframe_target_mode': self.colmap_keyframe_target_mode.currentText(),
            'colmap_keyframe_target_min': self.colmap_keyframe_target_min.value(),
            'colmap_keyframe_target_max': max(
                self.colmap_keyframe_target_min.value(),
                self.colmap_keyframe_target_max.value(),
            ),
            'colmap_nms_window_sec': self.colmap_nms_window_sec.value(),
            'colmap_enable_stage0': self.colmap_enable_stage0.isChecked(),
            'colmap_motion_aware_selection': self.colmap_motion_aware_selection.isChecked(),
            'colmap_nms_motion_window_ratio': self.colmap_nms_motion_window_ratio.value(),
            'colmap_stage1_adaptive_threshold': self.colmap_stage1_adaptive_threshold.isChecked(),
            'colmap_stage1_min_candidates_per_bin': self.colmap_stage1_min_candidates_per_bin.value(),
            'colmap_stage1_max_candidates': self.colmap_stage1_max_candidates.value(),
            'colmap_selection_profile': self.colmap_selection_profile.currentText(),
            'colmap_stage2_entry_budget': self.colmap_stage2_entry_budget.value(),
            'colmap_stage2_entry_min_gap': self.colmap_stage2_entry_min_gap.value(),
            'colmap_diversity_ssim_threshold': self.colmap_diversity_ssim_threshold.value(),
            'colmap_diversity_phash_hamming': self.colmap_diversity_phash_hamming.value(),
            'colmap_final_target_policy': self.colmap_final_target_policy.currentText(),
            'colmap_final_soft_min': self.colmap_final_soft_min.value(),
            'colmap_final_soft_max': max(self.colmap_final_soft_min.value(), self.colmap_final_soft_max.value()),
            'colmap_no_supplement_on_low_quality': self.colmap_no_supplement_on_low_quality.isChecked(),
            'colmap_rig_policy': self.colmap_rig_policy.currentText(),
            'colmap_rig_seed_opk_deg': rig_seed_opk,
            'colmap_workspace_scope': self.colmap_workspace_scope.currentText(),
            'colmap_reuse_db': self.colmap_reuse_db.isChecked(),
            'colmap_analysis_mask_profile': self.colmap_analysis_mask_profile.currentText(),
            'pose_export_format': self.pose_export_format.currentText(),
            'pose_select_translation_threshold': self.pose_select_translation_threshold.value(),
            'pose_select_rotation_threshold_deg': self.pose_select_rotation_threshold_deg.value(),
            'pose_select_min_observations': self.pose_select_min_observations.value(),
            'pose_select_enable_translation': self.pose_select_enable_translation.isChecked(),
            'pose_select_enable_rotation': self.pose_select_enable_rotation.isChecked(),
            'pose_select_enable_observations': self.pose_select_enable_observations.isChecked(),
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
            # Customモード（プリセット非適用）
            self.preset_description_label.setText(
                "手動で設定をカスタマイズします。各パラメータを自由に調整できます。"
            )
            return

        try:
            # ConfigManagerでプリセットをロード
            config_manager = ConfigManager()
            preset_info = config_manager.get_preset_info(
                config_manager.resolve_preset_id(preset_id)
            )

            if preset_info is None:
                logger.warning(f"プリセット '{preset_id}' が見つかりません")
                return

            # 説明を表示
            self.preset_description_label.setText(preset_info.description)

            # UIの値を更新
            params = preset_info.parameters

            # 重み
            self.sharpness_weight_slider.setValue(
                int(params.get('weight_sharpness', 0.30) * 100)
            )
            self.exposure_weight_slider.setValue(
                int(params.get('weight_exposure', 0.15) * 100)
            )
            self.geometric_weight_slider.setValue(
                int(params.get('weight_geometric', 0.30) * 100)
            )
            self.content_weight_slider.setValue(
                int(params.get('weight_content', 0.25) * 100)
            )

            # 適応的選択パラメータ
            self.ssim_threshold.setValue(
                params.get('ssim_threshold', 0.85)
            )
            self.quality_filter_enabled.setChecked(bool(params.get('quality_filter_enabled', True)))
            self.quality_threshold.setValue(float(params.get('quality_threshold', 0.50)))
            self.stage1_lr_merge_mode.setCurrentText(str(params.get('stage1_lr_merge_mode', 'asymmetric_sky_v1')))
            self.stage1_lr_asym_weak_floor.setValue(float(params.get('stage1_lr_asym_weak_floor', 0.35)))
            self.min_keyframe_interval.setValue(
                params.get('min_keyframe_interval', 5)
            )
            self.max_keyframe_interval.setValue(
                params.get('max_keyframe_interval', 60)
            )
            self.softmax_beta.setValue(
                params.get('softmax_beta', 5.0)
            )

            # GRICパラメータ
            self.gric_lambda1.setValue(
                params.get('gric_lambda1', 2.0)
            )
            self.gric_lambda2.setValue(
                params.get('gric_lambda2', 4.0)
            )
            self.gric_sigma.setValue(
                params.get('gric_sigma', 1.0)
            )
            self.gric_degeneracy_threshold.setValue(
                params.get('gric_degeneracy_threshold', 0.85)
            )

            # 360度設定
            self.enable_polar_mask.setChecked(
                params.get('enable_polar_mask', True)
            )
            self.mask_polar_ratio.setValue(
                params.get('mask_polar_ratio', 0.10)
            )
            self.enable_fisheye_border_mask.setChecked(
                params.get('enable_fisheye_border_mask', True)
            )
            self.fisheye_mask_radius_ratio.setValue(
                params.get('fisheye_mask_radius_ratio', 0.94)
            )
            self.fisheye_mask_center_offset_x.setValue(
                int(params.get('fisheye_mask_center_offset_x', 0))
            )
            self.fisheye_mask_center_offset_y.setValue(
                int(params.get('fisheye_mask_center_offset_y', 0))
            )
            self.enable_split_views.setChecked(bool(params.get('enable_split_views', True)))
            self.split_view_size.setValue(int(params.get('split_view_size', 1600)))
            self.split_view_hfov.setValue(float(params.get('split_view_hfov', 80.0)))
            self.split_view_vfov.setValue(float(params.get('split_view_vfov', 80.0)))
            self.split_cross_yaw_deg.setValue(float(params.get('split_cross_yaw_deg', 50.5)))
            self.split_cross_pitch_deg.setValue(float(params.get('split_cross_pitch_deg', 50.5)))
            self.split_cross_inward_deg.setValue(float(params.get('split_cross_inward_deg', 10.0)))
            self.split_inward_up_deg.setValue(float(params.get('split_inward_up_deg', 25.0)))
            self.split_inward_down_deg.setValue(float(params.get('split_inward_down_deg', 25.0)))
            self.split_inward_left_deg.setValue(float(params.get('split_inward_left_deg', 25.0)))
            self.split_inward_right_deg.setValue(float(params.get('split_inward_right_deg', 25.0)))
            self.enable_stage0_scan.setChecked(
                params.get('enable_stage0_scan', True)
            )
            self.stage0_stride.setValue(
                int(params.get('stage0_stride', 5))
            )
            self.enable_stage3_refinement.setChecked(
                params.get('enable_stage3_refinement', True)
            )
            self.stage3_weight_base.setValue(
                float(params.get('stage3_weight_base', 0.70))
            )
            self.stage3_weight_trajectory.setValue(
                float(params.get('stage3_weight_trajectory', 0.25))
            )
            self.stage3_weight_stage0_risk.setValue(
                float(params.get('stage3_weight_stage0_risk', 0.05))
            )
            self.vo_enabled.setChecked(bool(params.get('vo_enabled', True)))
            self.vo_center_roi_ratio.setValue(float(params.get('vo_center_roi_ratio', 0.6)))
            self.vo_downscale_long_edge.setValue(int(params.get('vo_downscale_long_edge', 1000)))
            self.vo_max_features.setValue(int(params.get('vo_max_features', 600)))
            self.vo_frame_subsample.setValue(int(params.get('vo_frame_subsample', 1)))
            self.vo_essential_method.setCurrentText(str(params.get('vo_essential_method', 'auto')))
            self.vo_subpixel_refine.setChecked(bool(params.get('vo_subpixel_refine', True)))
            self.vo_adaptive_subsample.setChecked(bool(params.get('vo_adaptive_subsample', False)))
            self.vo_subsample_min.setValue(int(params.get('vo_subsample_min', 1)))
            self.vo_confidence_low_threshold.setValue(float(params.get('vo_confidence_low_threshold', 0.35)))
            self.vo_confidence_mid_threshold.setValue(float(params.get('vo_confidence_mid_threshold', 0.55)))
            self.calib_xml.setText(str(params.get('calib_xml', '')))
            self.front_calib_xml.setText(str(params.get('front_calib_xml', '')))
            self.rear_calib_xml.setText(str(params.get('rear_calib_xml', '')))
            self.calib_model.setCurrentText(str(params.get('calib_model', 'auto')))
            self.pose_backend.setCurrentText(str(params.get('pose_backend', 'vo')))
            self.colmap_path.setText(str(params.get('colmap_path', 'colmap')))
            self.colmap_workspace.setText(str(params.get('colmap_workspace', '')))
            self.colmap_db_path.setText(str(params.get('colmap_db_path', '')))
            self.colmap_pipeline_mode.setCurrentText(str(params.get('colmap_pipeline_mode', 'minimal_v1')))
            self.colmap_keyframe_policy.setCurrentText(str(params.get('colmap_keyframe_policy', 'stage2_relaxed')))
            self.colmap_keyframe_target_mode.setCurrentText(str(params.get('colmap_keyframe_target_mode', 'auto')))
            self.colmap_keyframe_target_min.setValue(int(params.get('colmap_keyframe_target_min', 120)))
            self.colmap_keyframe_target_max.setValue(int(params.get('colmap_keyframe_target_max', 240)))
            self.colmap_nms_window_sec.setValue(float(params.get('colmap_nms_window_sec', 0.35)))
            self.colmap_enable_stage0.setChecked(bool(params.get('colmap_enable_stage0', True)))
            self.colmap_motion_aware_selection.setChecked(bool(params.get('colmap_motion_aware_selection', True)))
            self.colmap_nms_motion_window_ratio.setValue(float(params.get('colmap_nms_motion_window_ratio', 0.5)))
            self.colmap_stage1_adaptive_threshold.setChecked(bool(params.get('colmap_stage1_adaptive_threshold', True)))
            self.colmap_stage1_min_candidates_per_bin.setValue(int(params.get('colmap_stage1_min_candidates_per_bin', 3)))
            self.colmap_stage1_max_candidates.setValue(int(params.get('colmap_stage1_max_candidates', 360)))
            self.colmap_selection_profile.setCurrentText(str(params.get('colmap_selection_profile', 'no_vo_coverage')))
            self.colmap_stage2_entry_budget.setValue(int(params.get('colmap_stage2_entry_budget', 180)))
            self.colmap_stage2_entry_min_gap.setValue(int(params.get('colmap_stage2_entry_min_gap', 3)))
            self.colmap_diversity_ssim_threshold.setValue(float(params.get('colmap_diversity_ssim_threshold', 0.93)))
            self.colmap_diversity_phash_hamming.setValue(int(params.get('colmap_diversity_phash_hamming', 10)))
            self.colmap_final_target_policy.setCurrentText(str(params.get('colmap_final_target_policy', 'soft_auto')))
            self.colmap_final_soft_min.setValue(int(params.get('colmap_final_soft_min', 80)))
            self.colmap_final_soft_max.setValue(int(params.get('colmap_final_soft_max', 220)))
            self.colmap_no_supplement_on_low_quality.setChecked(bool(params.get('colmap_no_supplement_on_low_quality', True)))
            self.colmap_rig_policy.setCurrentText(str(params.get('colmap_rig_policy', 'lr_opk')))
            seed = params.get('colmap_rig_seed_opk_deg', [0.0, 0.0, 180.0])
            if isinstance(seed, (list, tuple)) and len(seed) == 3:
                self.colmap_rig_seed_opk_deg.setText(f"{float(seed[0]):.6g},{float(seed[1]):.6g},{float(seed[2]):.6g}")
            else:
                self.colmap_rig_seed_opk_deg.setText("0,0,180")
            self.colmap_workspace_scope.setCurrentText(str(params.get('colmap_workspace_scope', 'run_scoped')))
            self.colmap_reuse_db.setChecked(bool(params.get('colmap_reuse_db', False)))
            self.colmap_analysis_mask_profile.setCurrentText(str(params.get('colmap_analysis_mask_profile', 'colmap_safe')))
            self.pose_export_format.setCurrentText(str(params.get('pose_export_format', 'internal')))
            self.pose_select_translation_threshold.setValue(float(params.get('pose_select_translation_threshold', 1.2)))
            self.pose_select_rotation_threshold_deg.setValue(float(params.get('pose_select_rotation_threshold_deg', 5.0)))
            self.pose_select_min_observations.setValue(int(params.get('pose_select_min_observations', 30)))
            self.pose_select_enable_translation.setChecked(bool(params.get('pose_select_enable_translation', True)))
            self.pose_select_enable_rotation.setChecked(bool(params.get('pose_select_enable_rotation', True)))
            self.pose_select_enable_observations.setChecked(bool(params.get('pose_select_enable_observations', False)))
            self._on_pose_backend_changed(self.pose_backend.currentText())

            logger.info(f"プリセット '{preset_id}' ({preset_info.name}) を適用しました")

        except Exception as e:
            logger.error(f"プリセット適用エラー: {e}")
            QMessageBox.warning(
                self,
                "エラー",
                f"プリセットの適用に失敗しました:\n{e}"
            )

    def _on_vo_preset_changed(self, preset_name: str):
        preset = str(preset_name or "Custom").strip().lower()
        if preset == "quick":
            self.vo_frame_subsample.setValue(3) if hasattr(self, "vo_frame_subsample") else None
            self.vo_subsample_min.setValue(1)
            self.vo_max_features.setValue(300)
            self.vo_downscale_long_edge.setValue(640)
            self.vo_essential_method.setCurrentText("ransac")
            self.vo_subpixel_refine.setChecked(False)
            self.vo_adaptive_subsample.setChecked(True)
        elif preset == "balanced":
            self.vo_frame_subsample.setValue(1) if hasattr(self, "vo_frame_subsample") else None
            self.vo_subsample_min.setValue(1)
            self.vo_max_features.setValue(600)
            self.vo_downscale_long_edge.setValue(1000)
            self.vo_essential_method.setCurrentText("auto")
            self.vo_subpixel_refine.setChecked(True)
            self.vo_adaptive_subsample.setChecked(False)
        elif preset == "precise":
            self.vo_frame_subsample.setValue(1) if hasattr(self, "vo_frame_subsample") else None
            self.vo_subsample_min.setValue(1)
            self.vo_max_features.setValue(1000)
            self.vo_downscale_long_edge.setValue(1280)
            self.vo_essential_method.setCurrentText("magsac")
            self.vo_subpixel_refine.setChecked(True)
            self.vo_adaptive_subsample.setChecked(False)

    def _on_pose_backend_changed(self, backend_name: str):
        is_colmap = str(backend_name or "").strip().lower() == "colmap"
        pipeline_mode = str(self.colmap_pipeline_mode.currentText() if hasattr(self, "colmap_pipeline_mode") else "").strip().lower()
        if pipeline_mode not in {"legacy", "minimal_v1"}:
            pipeline_mode = "minimal_v1"
        is_minimal = bool(is_colmap and pipeline_mode == "minimal_v1")

        colmap_widgets = [
            w for w in [
                getattr(self, "colmap_pipeline_mode", None),
                getattr(self, "colmap_keyframe_policy", None),
                getattr(self, "colmap_keyframe_target_mode", None),
                getattr(self, "colmap_keyframe_target_min", None),
                getattr(self, "colmap_keyframe_target_max", None),
                getattr(self, "colmap_nms_window_sec", None),
                getattr(self, "colmap_enable_stage0", None),
                getattr(self, "colmap_motion_aware_selection", None),
                getattr(self, "colmap_nms_motion_window_ratio", None),
                getattr(self, "colmap_stage1_adaptive_threshold", None),
                getattr(self, "colmap_stage1_min_candidates_per_bin", None),
                getattr(self, "colmap_stage1_max_candidates", None),
                getattr(self, "colmap_selection_profile", None),
                getattr(self, "colmap_stage2_entry_budget", None),
                getattr(self, "colmap_stage2_entry_min_gap", None),
                getattr(self, "colmap_diversity_ssim_threshold", None),
                getattr(self, "colmap_diversity_phash_hamming", None),
                getattr(self, "colmap_final_target_policy", None),
                getattr(self, "colmap_final_soft_min", None),
                getattr(self, "colmap_final_soft_max", None),
                getattr(self, "colmap_no_supplement_on_low_quality", None),
                getattr(self, "colmap_rig_policy", None),
                getattr(self, "colmap_rig_seed_opk_deg", None),
                getattr(self, "colmap_workspace_scope", None),
                getattr(self, "colmap_reuse_db", None),
                getattr(self, "colmap_analysis_mask_profile", None),
            ] if w is not None
        ]
        for w in colmap_widgets:
            w.setEnabled(is_colmap)

        minimal_mode_colmap_ignored = [
            w for w in [
                getattr(self, "colmap_keyframe_policy", None),
                getattr(self, "colmap_keyframe_target_mode", None),
                getattr(self, "colmap_keyframe_target_min", None),
                getattr(self, "colmap_keyframe_target_max", None),
                getattr(self, "colmap_enable_stage0", None),
                getattr(self, "colmap_motion_aware_selection", None),
                getattr(self, "colmap_nms_motion_window_ratio", None),
                getattr(self, "colmap_selection_profile", None),
                getattr(self, "colmap_stage2_entry_budget", None),
                getattr(self, "colmap_stage2_entry_min_gap", None),
                getattr(self, "colmap_diversity_ssim_threshold", None),
                getattr(self, "colmap_diversity_phash_hamming", None),
                getattr(self, "colmap_final_target_policy", None),
                getattr(self, "colmap_final_soft_min", None),
                getattr(self, "colmap_final_soft_max", None),
                getattr(self, "colmap_no_supplement_on_low_quality", None),
            ] if w is not None
        ]
        for w in minimal_mode_colmap_ignored:
            w.setEnabled(bool(is_colmap and not is_minimal))

        minimal_mode_global_ignored = [
            w for w in [
                getattr(self, "enable_stage0_scan", None),
                getattr(self, "stage0_stride", None),
                getattr(self, "enable_stage3_refinement", None),
                getattr(self, "stage3_weight_base", None),
                getattr(self, "stage3_weight_trajectory", None),
                getattr(self, "stage3_weight_stage0_risk", None),
                getattr(self, "enable_dynamic_mask_removal", None),
                getattr(self, "dynamic_mask_use_yolo_sam", None),
                getattr(self, "dynamic_mask_use_motion_diff", None),
                getattr(self, "dynamic_mask_motion_frames", None),
                getattr(self, "dynamic_mask_motion_threshold", None),
                getattr(self, "dynamic_mask_dilation_size", None),
                getattr(self, "dynamic_mask_inpaint_enabled", None),
                getattr(self, "dynamic_mask_inpaint_module", None),
                getattr(self, "min_keyframe_interval", None),
            ] if w is not None
        ]
        for w in minimal_mode_global_ignored:
            w.setEnabled(not is_minimal)

        if not is_colmap:
            self.colmap_minimal_info_label.setText("")
        elif is_minimal:
            self.colmap_minimal_info_label.setText(
                "COLMAP Minimal v1 有効: Stage0/Stage1.5/Stage3/retarget/dynamic mask は無効化されます。"
            )
        else:
            self.colmap_minimal_info_label.setText(
                "COLMAP Legacy mode: 旧来のStage0/Stage1.5/Stage3/retarget設定が利用されます。"
            )

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
            self.quality_filter_enabled.setChecked(True)
            self.quality_threshold.setValue(0.50)
            self.stage1_lr_merge_mode.setCurrentText("asymmetric_sky_v1")
            self.stage1_lr_asym_weak_floor.setValue(0.35)
            self.min_keyframe_interval.setValue(5)
            self.max_keyframe_interval.setValue(60)
            self.softmax_beta.setValue(5.0)
            self.gric_lambda1.setValue(2.0)
            self.gric_lambda2.setValue(4.0)
            self.gric_sigma.setValue(1.0)
            self.gric_degeneracy_threshold.setValue(0.85)
            self.enable_rerun_logging.setChecked(False)
            self.equirect_width.setValue(4096)
            self.equirect_height.setValue(2048)
            self.projection_mode.setCurrentText('Equirectangular')
            self.fov.setValue(90.0)
            self.enable_polar_mask.setChecked(True)
            self.mask_polar_ratio.setValue(0.10)
            self.stitching_mode.setCurrentText('Fast')
            self.enable_fisheye_border_mask.setChecked(True)
            self.fisheye_mask_radius_ratio.setValue(0.94)
            self.fisheye_mask_center_offset_x.setValue(0)
            self.fisheye_mask_center_offset_y.setValue(0)
            self.enable_split_views.setChecked(True)
            self.split_view_size.setValue(1600)
            self.split_view_hfov.setValue(80.0)
            self.split_view_vfov.setValue(80.0)
            self.split_cross_yaw_deg.setValue(50.5)
            self.split_cross_pitch_deg.setValue(50.5)
            self.split_cross_inward_deg.setValue(10.0)
            self.split_inward_up_deg.setValue(25.0)
            self.split_inward_down_deg.setValue(25.0)
            self.split_inward_left_deg.setValue(25.0)
            self.split_inward_right_deg.setValue(25.0)
            self.enable_nadir_mask.setChecked(False)
            self.nadir_mask_radius.setValue(100)
            self.enable_equipment_detection.setChecked(False)
            self.mask_dilation.setValue(15)
            self.output_format.setCurrentText('PNG')
            self.jpeg_quality.setValue(95)
            self.output_dir_label.setText(str(Path.home() / "360split_output"))
            self.naming_prefix.setCurrentText('keyframe')
            self.enable_dynamic_mask_removal.setChecked(False)
            self.dynamic_mask_use_yolo_sam.setChecked(True)
            self.dynamic_mask_use_motion_diff.setChecked(True)
            self.dynamic_mask_motion_frames.setValue(3)
            self.dynamic_mask_motion_threshold.setValue(30)
            self.dynamic_mask_dilation_size.setValue(5)
            self.dynamic_mask_inpaint_enabled.setChecked(False)
            self.dynamic_mask_inpaint_module.setCurrentText('')
            self.enable_stage0_scan.setChecked(True)
            self.stage0_stride.setValue(5)
            self.enable_stage3_refinement.setChecked(True)
            self.stage3_weight_base.setValue(0.70)
            self.stage3_weight_trajectory.setValue(0.25)
            self.stage3_weight_stage0_risk.setValue(0.05)
            self.vo_enabled.setChecked(True)
            self.vo_center_roi_ratio.setValue(0.6)
            self.vo_downscale_long_edge.setValue(1000)
            self.vo_max_features.setValue(600)
            self.vo_frame_subsample.setValue(1)
            self.vo_essential_method.setCurrentText("auto")
            self.vo_subpixel_refine.setChecked(True)
            self.vo_adaptive_subsample.setChecked(False)
            self.vo_subsample_min.setValue(1)
            self.vo_confidence_low_threshold.setValue(0.35)
            self.vo_confidence_mid_threshold.setValue(0.55)
            self.opencv_thread_count.setValue(0)
            self.stage1_process_workers.setValue(0)
            self.stage1_prefetch_size.setValue(32)
            self.stage1_metrics_batch_size.setValue(64)
            self.stage1_gpu_batch_enabled.setChecked(True)
            self.darwin_capture_backend.setCurrentText("auto")
            self.mps_min_pixels.setValue(256 * 256)
            self.calib_xml.setText("")
            self.front_calib_xml.setText("")
            self.rear_calib_xml.setText("")
            self.calib_model.setCurrentText("auto")
            self.pose_backend.setCurrentText("vo")
            self.colmap_path.setText("colmap")
            self.colmap_workspace.setText("")
            self.colmap_db_path.setText("")
            self.colmap_pipeline_mode.setCurrentText("minimal_v1")
            self.colmap_keyframe_policy.setCurrentText("stage2_relaxed")
            self.colmap_keyframe_target_mode.setCurrentText("auto")
            self.colmap_keyframe_target_min.setValue(120)
            self.colmap_keyframe_target_max.setValue(240)
            self.colmap_nms_window_sec.setValue(0.35)
            self.colmap_enable_stage0.setChecked(True)
            self.colmap_motion_aware_selection.setChecked(True)
            self.colmap_nms_motion_window_ratio.setValue(0.5)
            self.colmap_stage1_adaptive_threshold.setChecked(True)
            self.colmap_stage1_min_candidates_per_bin.setValue(3)
            self.colmap_stage1_max_candidates.setValue(360)
            self.colmap_selection_profile.setCurrentText("no_vo_coverage")
            self.colmap_stage2_entry_budget.setValue(180)
            self.colmap_stage2_entry_min_gap.setValue(3)
            self.colmap_diversity_ssim_threshold.setValue(0.93)
            self.colmap_diversity_phash_hamming.setValue(10)
            self.colmap_final_target_policy.setCurrentText("soft_auto")
            self.colmap_final_soft_min.setValue(80)
            self.colmap_final_soft_max.setValue(220)
            self.colmap_no_supplement_on_low_quality.setChecked(True)
            self.colmap_rig_policy.setCurrentText("lr_opk")
            self.colmap_rig_seed_opk_deg.setText("0,0,180")
            self.colmap_workspace_scope.setCurrentText("run_scoped")
            self.colmap_reuse_db.setChecked(False)
            self.colmap_analysis_mask_profile.setCurrentText("colmap_safe")
            self.pose_export_format.setCurrentText("internal")
            self.pose_select_translation_threshold.setValue(1.2)
            self.pose_select_rotation_threshold_deg.setValue(5.0)
            self.pose_select_min_observations.setValue(30)
            self.pose_select_enable_translation.setChecked(True)
            self.pose_select_enable_rotation.setChecked(True)
            self.pose_select_enable_observations.setChecked(False)
            self._on_pose_backend_changed(self.pose_backend.currentText())

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

    def _pick_file_for_line_edit(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "キャリブレーションXMLを選択",
            str(Path.home()),
            "XML Files (*.xml);;All Files (*)",
        )
        if path:
            line_edit.setText(path)

    def _pick_dir_for_line_edit(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(
            self,
            "ディレクトリを選択",
            str(Path.home()),
        )
        if path:
            line_edit.setText(path)

    def _run_calibration_check_from_gui(self):
        out_dir = Path.home() / ".360split" / "calib_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "calibration_check.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--out",
            str(out_dir),
            "--calib-model",
            self.calib_model.currentText(),
            "--roi-ratio",
            str(self.vo_center_roi_ratio.value()),
        ]
        calib = self.calib_xml.text().strip()
        front_calib = self.front_calib_xml.text().strip()
        rear_calib = self.rear_calib_xml.text().strip()

        parent = self.parent()
        video_path = getattr(parent, "video_path", None) if parent is not None else None
        is_stereo = bool(getattr(parent, "is_stereo", False)) if parent is not None else False
        left_path = getattr(parent, "stereo_left_path", None) if parent is not None else None
        right_path = getattr(parent, "stereo_right_path", None) if parent is not None else None

        if is_stereo and left_path and right_path:
            cmd.extend(["--front-video", str(left_path), "--rear-video", str(right_path)])
            if front_calib:
                cmd.extend(["--front-calib-xml", front_calib])
            if rear_calib:
                cmd.extend(["--rear-calib-xml", rear_calib])
            if calib and not front_calib:
                cmd.extend(["--front-calib-xml", calib])
            if calib and not rear_calib:
                cmd.extend(["--rear-calib-xml", calib])
        elif video_path:
            cmd.extend(["--video", str(video_path)])
            if calib:
                cmd.extend(["--calib-xml", calib])
        else:
            QMessageBox.warning(self, "警告", "先に動画を読み込んでください。")
            return

        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env,
            )
            if result.returncode == 0:
                logger.info(f"Calibration Check complete: {out_dir}")
                QMessageBox.information(self, "完了", f"Calibration Check 完了: {out_dir}")
                if platform.system() == "Darwin":
                    subprocess.Popen(["open", str(out_dir)])
                elif platform.system() == "Windows":
                    os.startfile(str(out_dir))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(out_dir)])
            else:
                logger.warning(f"Calibration Check failed: {result.stderr}")
                QMessageBox.warning(self, "エラー", f"Calibration Check 失敗:\n{result.stderr or result.stdout}")
        except Exception as e:
            logger.exception(f"Calibration Check execution error: {e}")
            QMessageBox.warning(self, "エラー", f"Calibration Check 実行エラー:\n{e}")
