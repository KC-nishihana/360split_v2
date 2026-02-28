from core.pipeline.stage2_evaluator import run_stage2_evaluator


class _DummySelector:
    def __init__(self, parallel: bool):
        self.config = {"ENABLE_STAGE2_PIPELINE_PARALLEL": parallel}

    def _stage2_precise_evaluation(
        self,
        _video_loader,
        _metadata,
        stage1_candidates,
        _progress_callback,
        _frame_log_callback,
        _stage0_metrics,
    ):
        return list(stage1_candidates), []


def test_stage2_parallel_flag_keeps_same_output():
    candidates = [{"frame_idx": 1, "quality_scores": {"quality": 0.8}}]
    out_off = run_stage2_evaluator(_DummySelector(False), None, None, candidates, None, None, None)
    out_on = run_stage2_evaluator(_DummySelector(True), None, None, candidates, None, None, None)
    assert out_off == out_on
