from types import SimpleNamespace

from main import apply_cli_overrides


class _Args(SimpleNamespace):
    def __getattr__(self, _name):
        return None


def test_apply_cli_overrides_performance_knobs():
    config = {}
    args = _Args(
        analysis_run_id="run-123",
        resume=True,
        keep_temp=True,
        colmap_format=True,
        opencv_threads=8,
        stage1_process_workers=6,
        stage1_prefetch_size=48,
        stage1_metrics_batch_size=96,
        stage1_gpu_batch=False,
        darwin_capture_backend="ffmpeg",
        mps_min_pixels=131072,
        pose_backend="colmap",
        colmap_path="/opt/homebrew/bin/colmap",
        colmap_keyframe_policy="stage2_relaxed",
        colmap_keyframe_target_mode="auto",
        colmap_keyframe_target_min=120,
        colmap_keyframe_target_max=240,
        colmap_nms_window_sec=0.35,
        colmap_enable_stage0=True,
        colmap_motion_aware_selection=True,
        colmap_nms_motion_window_ratio=0.5,
        colmap_stage1_adaptive_threshold=True,
        colmap_stage1_min_candidates_per_bin=3,
        colmap_stage1_max_candidates=360,
        colmap_rig_policy="lr_opk",
        colmap_rig_seed_opk=[0.0, 0.0, 180.0],
        colmap_workspace_scope="run_scoped",
        colmap_reuse_db=False,
        colmap_analysis_mask_profile="colmap_safe",
        pose_export_format="metashape",
    )

    apply_cli_overrides(config, args)

    assert config["analysis_run_id"] == "run-123"
    assert config["resume_enabled"] is True
    assert config["keep_temp_on_success"] is True
    assert config["colmap_format"] is True
    assert config["opencv_thread_count"] == 8
    assert config["stage1_process_workers"] == 6
    assert config["stage1_prefetch_size"] == 48
    assert config["stage1_metrics_batch_size"] == 96
    assert config["stage1_gpu_batch_enabled"] is False
    assert config["darwin_capture_backend"] == "ffmpeg"
    assert config["mps_min_pixels"] == 131072
    assert config["pose_backend"] == "colmap"
    assert config["colmap_path"] == "/opt/homebrew/bin/colmap"
    assert config["colmap_keyframe_policy"] == "stage2_relaxed"
    assert config["colmap_keyframe_target_mode"] == "auto"
    assert config["colmap_keyframe_target_min"] == 120
    assert config["colmap_keyframe_target_max"] == 240
    assert config["colmap_nms_window_sec"] == 0.35
    assert config["colmap_enable_stage0"] is True
    assert config["colmap_motion_aware_selection"] is True
    assert config["colmap_nms_motion_window_ratio"] == 0.5
    assert config["colmap_stage1_adaptive_threshold"] is True
    assert config["colmap_stage1_min_candidates_per_bin"] == 3
    assert config["colmap_stage1_max_candidates"] == 360
    assert config["colmap_rig_policy"] == "lr_opk"
    assert config["colmap_rig_seed_opk_deg"] == [0.0, 0.0, 180.0]
    assert config["colmap_workspace_scope"] == "run_scoped"
    assert config["colmap_reuse_db"] is False
    assert config["colmap_analysis_mask_profile"] == "colmap_safe"
    assert config["pose_export_format"] == "metashape"
