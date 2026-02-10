"""
カスタム例外クラス - 360Split用

幾何学的評価における縮退検出やモデル推定失敗を
明示的に区別するための例外クラス群。
"""


class GeometricDegeneracyError(Exception):
    """
    幾何学的縮退エラー

    視差が不十分（回転のみの動き）、または平面シーンのため
    3D再構成に適さないフレームペアであることを示す。

    GRIC判定で GRIC_H < GRIC_F の場合（ホモグラフィが
    基礎行列より良くフィットする = 純回転/平面）に送出される。

    Attributes:
    -----------
    gric_h : float
        ホモグラフィのGRICスコア
    gric_f : float
        基礎行列のGRICスコア
    inlier_ratio_h : float
        ホモグラフィのインライア率
    message : str
        詳細メッセージ
    """

    def __init__(self, message: str = "視差不十分: 回転のみまたは平面シーン",
                 gric_h: float = 0.0, gric_f: float = 0.0,
                 inlier_ratio_h: float = 0.0):
        self.gric_h = gric_h
        self.gric_f = gric_f
        self.inlier_ratio_h = inlier_ratio_h
        super().__init__(f"{message} (GRIC_H={gric_h:.4f}, GRIC_F={gric_f:.4f})")


class EstimationFailureError(Exception):
    """
    行列推定失敗エラー

    特徴点不足、RANSAC収束失敗、行列推定の数値的不安定性
    などにより、幾何学的評価が完了できない場合に送出される。

    Attributes:
    -----------
    reason : str
        失敗の理由
    match_count : int
        検出されたマッチ数
    required_count : int
        必要な最小マッチ数
    """

    def __init__(self, reason: str = "行列推定に失敗",
                 match_count: int = 0, required_count: int = 8):
        self.reason = reason
        self.match_count = match_count
        self.required_count = required_count
        super().__init__(
            f"{reason} (マッチ数: {match_count}, 必要数: {required_count})"
        )


class InsufficientFeaturesError(EstimationFailureError):
    """
    特徴点不足エラー

    EstimationFailureErrorの特殊化。
    特徴点マッチングの結果が最小要件を満たさない場合に送出。
    """

    def __init__(self, match_count: int = 0, required_count: int = 8):
        super().__init__(
            reason="特徴点マッチ数が不足",
            match_count=match_count,
            required_count=required_count
        )
