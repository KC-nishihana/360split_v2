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
        # 短縮名からフル名への変換マッピング
        preset_mapping = {
            'outdoor': 'outdoor_high_quality',
            'indoor': 'indoor_robust_tracking',
            'mixed': 'mixed_adaptive'
        }

        # 短縮名を試す
        if preset_id in preset_mapping:
            preset_id = preset_mapping[preset_id]

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
        # デフォルト設定をインポート
        import config as default_config

        base_config = {
            "laplacian_threshold": default_config.LAPLACIAN_THRESHOLD,
            "brightness_min": default_config.BRIGHTNESS_MIN,
            "brightness_max": default_config.BRIGHTNESS_MAX,
            "motion_blur_threshold": default_config.MOTION_BLUR_THRESHOLD,
            "exposure_threshold": 0.35,
            "softmax_beta": default_config.SOFTMAX_BETA,
            "gric_degeneracy_threshold": default_config.GRIC_RATIO_THRESHOLD,
            "min_feature_matches": default_config.MIN_FEATURE_MATCHES,
            "ssim_threshold": default_config.SSIM_CHANGE_THRESHOLD,
            "min_keyframe_interval": default_config.MIN_KEYFRAME_INTERVAL,
            "max_keyframe_interval": default_config.MAX_KEYFRAME_INTERVAL,
            "weight_sharpness": default_config.WEIGHT_SHARPNESS,
            "weight_exposure": default_config.WEIGHT_EXPOSURE,
            "weight_geometric": default_config.WEIGHT_GEOMETRIC,
            "weight_content": default_config.WEIGHT_CONTENT,
            "pair_motion_aggregation": "max",
            "enable_rig_stitching": True,
            "rig_feature_method": "orb",
            "output_image_format": default_config.OUTPUT_IMAGE_FORMAT,
            "output_jpeg_quality": default_config.OUTPUT_JPEG_QUALITY,
        }

        # プリセット適用
        config = self.load_preset(preset_id, base_config)

        # CLIオーバーライド適用
        if cli_overrides:
            config = self.merge_config(config, cli_overrides)

        # 検証
        if not self.validate_config(config):
            logger.warning("設定の検証で警告が発生しましたが、処理を続行します")

        return config
