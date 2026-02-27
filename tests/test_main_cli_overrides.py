from types import SimpleNamespace

from main import apply_cli_overrides


class _Args(SimpleNamespace):
    def __getattr__(self, _name):
        return None


def test_apply_cli_overrides_performance_knobs():
    config = {}
    args = _Args(
        opencv_threads=8,
        stage1_process_workers=6,
        stage1_prefetch_size=48,
        stage1_metrics_batch_size=96,
        stage1_gpu_batch=False,
        darwin_capture_backend="ffmpeg",
        mps_min_pixels=131072,
    )

    apply_cli_overrides(config, args)

    assert config["opencv_thread_count"] == 8
    assert config["stage1_process_workers"] == 6
    assert config["stage1_prefetch_size"] == 48
    assert config["stage1_metrics_batch_size"] == 96
    assert config["stage1_gpu_batch_enabled"] is False
    assert config["darwin_capture_backend"] == "ffmpeg"
    assert config["mps_min_pixels"] == 131072
