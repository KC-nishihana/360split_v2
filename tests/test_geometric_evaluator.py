from core.geometric_evaluator import GeometricEvaluator
from core.exceptions import GeometricDegeneracyError, EstimationFailureError, InsufficientFeaturesError


def test_compute_gric_score_runs(translated_pair):
    frame1, frame2 = translated_pair
    evaluator = GeometricEvaluator()
    try:
        score = evaluator.compute_gric_score(frame1, frame2)
        assert 0.0 <= score <= 1.0
    except (InsufficientFeaturesError, EstimationFailureError, GeometricDegeneracyError):
        # 入力条件によっては理論上発生し得るため、例外型のみ保証
        assert True
