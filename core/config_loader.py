"""
360Split - プリセット管理システム
JSONベースの環境別プリセット設定のロード・マージ機能

撮影環境（屋外・屋内・混合）に応じた最適なパラメータを
プリセットファイルから読み込み、デフォルト設定とマージします。
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PresetInfo:
    """
    プリセット情報

    Attributes:
    -----------
    name : str
        プリセット表示名
    description : str
        プリセットの説明
    environment : str
        環境タイプ（outdoor/indoor/mixed）
    file_path : Path
        プリセットファイルパス
    parameters : dict
        パラメータ辞書
    notes : List[str]
        設定メモ
    """
    name: str
    description: str
    environment: str
    file_path: Path
    parameters: Dict
    notes: List[str]


class ConfigManager:
    """
    プリセット管理システム

    JSONプリセットファイルの読み込み、検証、マージを行います。
    デフォルト設定に対してプリセットをオーバーライドし、
    統一された設定辞書を返します。

    使用例:
    --------
    >>> manager = ConfigManager()
    >>> config = manager.load_preset('indoor')
    >>> print(config['laplacian_threshold'])
    50.0
    """

    PRESET_MAPPING = {
        'outdoor': 'outdoor_high_quality',
        'indoor': 'indoor_robust_tracking',
        'mixed': 'mixed_adaptive'
    }

    def __init__(self, presets_dir: Optional[Path] = None):
        """
        初期化

        Parameters:
        -----------
        presets_dir : Path, optional
            プリセットディレクトリパス。Noneの場合はプロジェクトルート/presets
        """
        if presets_dir is None:
            # プロジェクトルートからpresets/を探す
            project_root = Path(__file__).parent.parent
            self.presets_dir = project_root / "presets"
        else:
            self.presets_dir = Path(presets_dir)

        self._preset_cache: Dict[str, PresetInfo] = {}
        self._scan_presets()

    @staticmethod
    def default_config() -> Dict:
        """
        単一のデフォルト設定を返す。

        Notes:
        ------
        main.py / GUI / CLI のベース設定はここを唯一の定義源にする。
        """
        import config as default_config

        return {
            "laplacian_threshold": default_config.LAPLACIAN_THRESHOLD,
            "brightness_min": default_config.BRIGHTNESS_MIN,
            "brightness_max": default_config.BRIGHTNESS_MAX,
            "motion_blur_threshold": default_config.MOTION_BLUR_THRESHOLD,
            "exposure_threshold": 0.35,
            "quality_filter_enabled": True,
            "quality_threshold": 0.50,
            "quality_roi_mode": "circle",
            "quality_roi_ratio": 0.40,
            "quality_abs_laplacian_min": 35.0,
            "quality_use_orb": True,
            "quality_weight_sharpness": 0.40,
            "quality_weight_tenengrad": 0.30,
            "quality_weight_exposure": 0.15,
            "quality_weight_keypoints": 0.15,
            "quality_norm_p_low": 10.0,
            "quality_norm_p_high": 90.0,
            "quality_debug": False,
            "quality_tenengrad_scale": 1.0,
            "softmax_beta": default_config.SOFTMAX_BETA,
            "gric_ratio_threshold": default_config.GRIC_RATIO_THRESHOLD,
            "gric_degeneracy_threshold": default_config.GRIC_RATIO_THRESHOLD,
            "min_feature_matches": default_config.MIN_FEATURE_MATCHES,
            "ssim_change_threshold": default_config.SSIM_CHANGE_THRESHOLD,
            "ssim_threshold": default_config.SSIM_CHANGE_THRESHOLD,
            "min_keyframe_interval": default_config.MIN_KEYFRAME_INTERVAL,
            "max_keyframe_interval": default_config.MAX_KEYFRAME_INTERVAL,
            "nms_time_window": 1.0,
            "stationary_enable": True,
            "stationary_min_duration_sec": 0.7,
            "stationary_use_quantile_threshold": True,
            "stationary_quantile": 0.10,
            "stationary_translation_threshold": None,
            "stationary_rotation_threshold": None,
            "stationary_flow_threshold": None,
            "stationary_min_match_count_for_vo": default_config.MIN_FEATURE_MATCHES,
            "stationary_fallback_when_vo_unreliable": "not_stationary",
            "stationary_soft_penalty": True,
            "stationary_penalty": 0.7,
            "stationary_allow_boundary_frames": True,
            "stationary_boundary_grace_frames": 2,
            "stationary_hysteresis_exit_scale": 1.25,
            "momentum_boost_factor": default_config.MOMENTUM_BOOST_FACTOR,
            "weight_sharpness": default_config.WEIGHT_SHARPNESS,
            "weight_exposure": default_config.WEIGHT_EXPOSURE,
            "weight_geometric": default_config.WEIGHT_GEOMETRIC,
            "weight_content": default_config.WEIGHT_CONTENT,
            "pair_motion_aggregation": "max",
            "enable_rig_stitching": True,
            "rig_feature_method": "orb",
            "stitching_mode": "Fast",
            "enable_stereo_stitch": True,
            "enable_fisheye_border_mask": True,
            "fisheye_mask_radius_ratio": 0.94,
            "fisheye_mask_center_offset_x": 0,
            "fisheye_mask_center_offset_y": 0,
            "enable_split_views": True,
            "split_view_size": 1600,
            "split_view_hfov": 80.0,
            "split_view_vfov": 80.0,
            "split_cross_yaw_deg": 50.5,
            "split_cross_pitch_deg": 50.5,
            "split_cross_inward_deg": 10.0,
            "split_inward_up_deg": 25.0,
            "split_inward_down_deg": 25.0,
            "split_inward_left_deg": 25.0,
            "split_inward_right_deg": 25.0,
            "enable_rerun_logging": False,
            "sample_interval": 1,
            "stage1_batch_size": 32,
            "stage1_grab_threshold": 30,
            "stage1_eval_scale": 0.5,
            "opencv_thread_count": 0,
            "stage1_process_workers": 0,
            "stage1_prefetch_size": 32,
            "stage1_metrics_batch_size": 64,
            "stage1_gpu_batch_enabled": True,
            "darwin_capture_backend": "auto",
            "mps_min_pixels": 256 * 256,
            "equirect_width": default_config.EQUIRECT_WIDTH,
            "equirect_height": default_config.EQUIRECT_HEIGHT,
            "cubemap_face_size": default_config.CUBEMAP_FACE_SIZE,
            "perspective_fov": default_config.PERSPECTIVE_FOV,
            "mask_dilation_kernel": default_config.MASK_DILATION_KERNEL,
            "output_image_format": default_config.OUTPUT_IMAGE_FORMAT,
            "output_jpeg_quality": default_config.OUTPUT_JPEG_QUALITY,
            # 対象マスク生成
            "enable_target_mask_generation": False,
            "target_classes": ["人物", "人", "自転車", "バイク", "車両", "動物"],
            "yolo_model_path": "yolo26n-seg.pt",
            "sam_model_path": "sam3_t.pt",
            "confidence_threshold": 0.25,
            "detection_device": "auto",
            "mask_output_dirname": "masks",
            "mask_add_suffix": True,
            "mask_suffix": "_mask",
            "mask_output_format": "same",
            "colmap_format": False,
            # Stage2 動体除去
            "enable_dynamic_mask_removal": False,
            "dynamic_mask_use_yolo_sam": True,
            "dynamic_mask_use_motion_diff": True,
            "dynamic_mask_motion_frames": 3,
            "dynamic_mask_motion_threshold": 30,
            "dynamic_mask_dilation_size": 5,
            "enable_profile": False,
            "stage2_perf_profile": True,
            "stage2_mask_cache_ttl_frames": 0,
            "enable_stage0_scan": True,
            "stage0_stride": 5,
            "enable_stage3_refinement": True,
            "stage3_weight_base": 0.70,
            "stage3_weight_trajectory": 0.25,
            "stage3_weight_stage0_risk": 0.05,
            "stage3_disable_traj_when_vo_unreliable": True,
            "stage3_vo_valid_ratio_threshold": 0.50,
            "flow_downscale": 1.0,
            "resume_enabled": False,
            "keep_temp_on_success": False,
            "vo_enabled": True,
            "vo_center_roi_ratio": 0.6,
            "vo_max_features": 600,
            "vo_quality_level": 0.01,
            "vo_min_distance": 8,
            "vo_min_track_points": 24,
            "vo_ransac_threshold": 1.0,
            "vo_downscale_long_edge": 1000,
            "vo_match_norm_factor": 120.0,
            "vo_t_sign": 1.0,
            "vo_frame_subsample": 1,
            "vo_adaptive_roi_enable": True,
            "vo_adaptive_roi_min": 0.45,
            "vo_adaptive_roi_max": 0.70,
            "vo_fast_fail_inlier_ratio": 0.12,
            "vo_step_proxy_clip_px": 80.0,
            "vo_essential_method": "auto",
            "vo_subpixel_refine": True,
            "vo_adaptive_subsample": False,
            "vo_subsample_min": 1,
            "vo_confidence_low_threshold": 0.35,
            "vo_confidence_mid_threshold": 0.55,
            "calib_xml": "",
            "front_calib_xml": "",
            "rear_calib_xml": "",
            "calib_model": "auto",
            "pose_backend": "vo",
            "colmap_path": "colmap",
            "colmap_workspace": "",
            "colmap_db_path": "",
            "colmap_keyframe_policy": "",
            "colmap_keyframe_target_mode": "auto",
            "colmap_keyframe_target_min": 120,
            "colmap_keyframe_target_max": 240,
            "colmap_nms_window_sec": 0.35,
            "colmap_rig_policy": "lr_opk",
            "colmap_rig_seed_opk_deg": [0.0, 0.0, 180.0],
            "colmap_workspace_scope": "run_scoped",
            "colmap_reuse_db": False,
            "colmap_analysis_mask_profile": "colmap_safe",
            "pose_export_format": "internal",
            "pose_select_translation_threshold": 1.2,
            "pose_select_rotation_threshold_deg": 5.0,
            "pose_select_min_observations": 30,
            "pose_select_enable_translation": True,
            "pose_select_enable_rotation": True,
            "pose_select_enable_observations": False,
            "dynamic_mask_target_classes": ["人物", "人", "自転車", "バイク", "車両", "動物"],
            "dynamic_mask_inpaint_enabled": False,
            "dynamic_mask_inpaint_module": "",
        }

    @classmethod
    def resolve_preset_id(cls, preset_id: str) -> str:
        """短縮名(outdoor/indoor/mixed)を実ファイルIDに解決する。"""
        if not preset_id:
            return preset_id
        return cls.PRESET_MAPPING.get(preset_id, preset_id)

    def _scan_presets(self):
        """
        presetsディレクトリをスキャンして利用可能なプリセットを検出
        """
        if not self.presets_dir.exists():
            logger.warning(f"プリセットディレクトリが見つかりません: {self.presets_dir}")
            return

        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 必須フィールドの検証
                if 'parameters' not in data:
                    logger.warning(f"プリセット '{preset_file.name}' に 'parameters' フィールドがありません")
                    continue

                preset_id = preset_file.stem  # ファイル名（拡張子なし）
                preset_info = PresetInfo(
                    name=data.get('name', preset_id),
                    description=data.get('description', ''),
                    environment=data.get('environment', 'unknown'),
                    file_path=preset_file,
                    parameters=data['parameters'],
                    notes=data.get('notes', [])
                )

                self._preset_cache[preset_id] = preset_info
                logger.info(f"プリセット '{preset_id}' をロードしました: {preset_info.name}")

            except json.JSONDecodeError as e:
                logger.error(f"プリセット '{preset_file.name}' のJSON解析エラー: {e}")
            except Exception as e:
                logger.error(f"プリセット '{preset_file.name}' の読み込みエラー: {e}")

    def list_presets(self) -> List[str]:
        """
        利用可能なプリセットID一覧を取得

        Returns:
        --------
        List[str]
            プリセットIDリスト
        """
        return list(self._preset_cache.keys())

    def get_preset_info(self, preset_id: str) -> Optional[PresetInfo]:
        """
        プリセット情報を取得

        Parameters:
        -----------
        preset_id : str
            プリセットID（ファイル名、例: 'indoor_robust_tracking'）

        Returns:
        --------
        PresetInfo or None
            プリセット情報。存在しない場合はNone
        """
        return self._preset_cache.get(preset_id)

    def load_preset(self, preset_id: str, base_config: Optional[Dict] = None) -> Dict:
        """
        プリセットをロードし、ベース設定とマージ

        Parameters:
        -----------
        preset_id : str
            プリセットID（'outdoor', 'indoor', 'mixed' など）
            または短縮名（'outdoor_high_quality' の場合 'outdoor' でもOK）
        base_config : dict, optional
            ベース設定辞書。Noneの場合はプリセットのみ返す

        Returns:
        --------
        dict
            マージされた設定辞書

        Raises:
        -------
        FileNotFoundError
            プリセットが存在しない場合
        """
        preset_id = self.resolve_preset_id(preset_id)

        # プリセット取得
        preset_info = self.get_preset_info(preset_id)

        if preset_info is None:
            available = ', '.join(self.list_presets())
            raise FileNotFoundError(
                f"プリセット '{preset_id}' が見つかりません。\n"
                f"利用可能なプリセット: {available}"
            )

        logger.info(f"プリセット '{preset_id}' を適用: {preset_info.name}")
        logger.debug(f"説明: {preset_info.description}")

        # ベース設定がある場合はマージ、ない場合はプリセットのみ
        if base_config is not None:
            merged = base_config.copy()
            merged.update(preset_info.parameters)
            return merged
        else:
            return preset_info.parameters.copy()

    def merge_config(self, base_config: Dict, overrides: Dict) -> Dict:
        """
        設定辞書をマージ（ディープマージ）

        Parameters:
        -----------
        base_config : dict
            ベース設定
        overrides : dict
            上書き設定

        Returns:
        --------
        dict
            マージされた設定
        """
        result = base_config.copy()

        for key, value in overrides.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # 辞書の場合は再帰的にマージ
                result[key] = self.merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def validate_config(self, config: Dict) -> bool:
        """
        設定の妥当性検証

        Parameters:
        -----------
        config : dict
            検証する設定辞書

        Returns:
        --------
        bool
            妥当性チェック結果
        """
        required_keys = [
            'laplacian_threshold',
            'min_keyframe_interval',
            'max_keyframe_interval',
            'weight_sharpness',
            'weight_geometric',
            'weight_content',
            'weight_exposure'
        ]

        # 必須キーの存在確認
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logger.error(f"設定に必須キーが不足しています: {missing_keys}")
            return False

        # 重みの合計が1.0に近いかチェック
        weights_sum = (
            config['weight_sharpness'] +
            config['weight_geometric'] +
            config['weight_content'] +
            config['weight_exposure']
        )

        if not (0.99 <= weights_sum <= 1.01):
            logger.warning(
                f"重みの合計が1.0ではありません: {weights_sum:.3f} "
                f"(推奨: 1.0)"
            )

        # 閾値の範囲チェック
        if config['min_keyframe_interval'] >= config['max_keyframe_interval']:
            logger.error(
                f"min_keyframe_interval ({config['min_keyframe_interval']}) が "
                f"max_keyframe_interval ({config['max_keyframe_interval']}) 以上です"
            )
            return False

        return True

    def create_config_from_preset(self, preset_id: str,
                                   cli_overrides: Optional[Dict] = None) -> Dict:
        """
        プリセットとCLIオーバーライドから最終設定を作成

        デフォルト設定 → プリセット → CLIオーバーライド の順でマージ

        Parameters:
        -----------
        preset_id : str
            プリセットID
        cli_overrides : dict, optional
            CLIから渡された上書き設定

        Returns:
        --------
        dict
            最終的な設定辞書
        """
        base_config = self.default_config()

        # プリセット適用
        config = self.load_preset(preset_id, base_config)

        # CLIオーバーライド適用
        if cli_overrides:
            config = self.merge_config(config, cli_overrides)

        # 検証
        if not self.validate_config(config):
            logger.warning("設定の検証で警告が発生しましたが、処理を続行します")

        return config
