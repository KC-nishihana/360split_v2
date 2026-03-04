from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .colmap_backend import COLMAPPoseEstimator
from .exporters import export_selected_images, write_internal_pose_csv, write_metashape_csv
from .selector import PoseSelectionConfig, select_required_poses
from .types import PoseEstimationResult
from .vo_backend import VOPoseEstimator


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _cfg(cfg: Dict[str, Any], lower_key: str, upper_key: str, default: Any) -> Any:
    if lower_key in cfg:
        return cfg.get(lower_key)
    if upper_key in cfg:
        return cfg.get(upper_key)
    return default


def run_pose_pipeline(
    image_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    ctx = dict(context or {})

    backend = str(_cfg(cfg, "pose_backend", "POSE_BACKEND", "vo") or "vo").strip().lower()
    if backend not in {"vo", "colmap"}:
        backend = "vo"

    if backend == "colmap":
        reuse_db = bool(_cfg(cfg, "colmap_reuse_db", "COLMAP_REUSE_DB", False))
        estimator = COLMAPPoseEstimator(
            colmap_path=str(_cfg(cfg, "colmap_path", "COLMAP_PATH", "colmap") or "colmap"),
            workspace=str(_cfg(cfg, "colmap_workspace", "COLMAP_WORKSPACE", "") or "") or None,
            db_path=str(_cfg(cfg, "colmap_db_path", "COLMAP_DB_PATH", "") or "") or None,
            clear_db=bool(_cfg(cfg, "colmap_clear_db", "COLMAP_CLEAR_DB", False)) or (not reuse_db),
        )
    else:
        estimator = VOPoseEstimator()

    if backend == "colmap":
        ctx.setdefault("analysis_run_id", str(_cfg(cfg, "analysis_run_id", "ANALYSIS_RUN_ID", "") or ""))
        ctx.setdefault("colmap_workspace_scope", str(_cfg(cfg, "colmap_workspace_scope", "COLMAP_WORKSPACE_SCOPE", "run_scoped") or "run_scoped"))
        ctx.setdefault("colmap_reuse_db", bool(_cfg(cfg, "colmap_reuse_db", "COLMAP_REUSE_DB", False)))
        ctx.setdefault("colmap_rig_policy", str(_cfg(cfg, "colmap_rig_policy", "COLMAP_RIG_POLICY", "lr_opk") or "lr_opk"))
        ctx.setdefault("colmap_rig_seed_opk_deg", _cfg(cfg, "colmap_rig_seed_opk_deg", "COLMAP_RIG_SEED_OPK_DEG", [0.0, 0.0, 180.0]))
        ctx.setdefault(
            "colmap_sparse_model_pick_policy",
            str(_cfg(cfg, "colmap_sparse_model_pick_policy", "COLMAP_SPARSE_MODEL_PICK_POLICY", "registered_then_coverage") or "registered_then_coverage"),
        )
        ctx.setdefault(
            "colmap_input_subset_enabled",
            bool(_cfg(cfg, "colmap_input_subset_enabled", "COLMAP_INPUT_SUBSET_ENABLED", True)),
        )
        ctx.setdefault(
            "colmap_input_gate_method",
            str(_cfg(cfg, "colmap_input_gate_method", "COLMAP_INPUT_GATE_METHOD", "homography_degeneracy_v1") or "homography_degeneracy_v1"),
        )
        ctx.setdefault(
            "colmap_input_gate_strength",
            str(_cfg(cfg, "colmap_input_gate_strength", "COLMAP_INPUT_GATE_STRENGTH", "medium") or "medium"),
        )
        ctx.setdefault(
            "colmap_input_min_keep_ratio",
            float(_cfg(cfg, "colmap_input_min_keep_ratio", "COLMAP_INPUT_MIN_KEEP_RATIO", 0.20)),
        )
        ctx.setdefault(
            "colmap_input_max_gap_rescue_frames",
            int(_cfg(cfg, "colmap_input_max_gap_rescue_frames", "COLMAP_INPUT_MAX_GAP_RESCUE_FRAMES", 150)),
        )
        preview_idx_cfg = _cfg(cfg, "colmap_preview_frame_indices", "COLMAP_PREVIEW_FRAME_INDICES", [])
        if isinstance(preview_idx_cfg, (list, tuple)):
            preview_idx_norm = [int(v) for v in preview_idx_cfg if isinstance(v, (int, float)) and int(v) >= 0]
        else:
            preview_idx_norm = []
        ctx.setdefault("colmap_preview_frame_indices", preview_idx_norm)

    result: PoseEstimationResult = estimator.estimate(image_dir=image_dir, context=ctx)

    selector_cfg = PoseSelectionConfig(
        translation_threshold=float(_cfg(cfg, "pose_select_translation_threshold", "POSE_SELECT_TRANSLATION_THRESHOLD", 1.2)),
        rotation_threshold_deg=float(_cfg(cfg, "pose_select_rotation_threshold_deg", "POSE_SELECT_ROTATION_THRESHOLD_DEG", 5.0)),
        min_observations=int(_cfg(cfg, "pose_select_min_observations", "POSE_SELECT_MIN_OBSERVATIONS", 30)),
        enable_translation=_as_bool(_cfg(cfg, "pose_select_enable_translation", "POSE_SELECT_ENABLE_TRANSLATION", None), True),
        enable_rotation=_as_bool(_cfg(cfg, "pose_select_enable_rotation", "POSE_SELECT_ENABLE_ROTATION", None), True),
        enable_observations=_as_bool(_cfg(cfg, "pose_select_enable_observations", "POSE_SELECT_ENABLE_OBSERVATIONS", None), False),
        enable_spatial_post_filter=_as_bool(
            _cfg(cfg, "pose_select_spatial_post_filter", "POSE_SELECT_SPATIAL_POST_FILTER", None),
            backend == "colmap",
        ),
        spatial_cell_scale=float(
            _cfg(cfg, "pose_select_spatial_cell_scale", "POSE_SELECT_SPATIAL_CELL_SCALE", 3.0)
        ),
        spatial_min_distance_scale=float(
            _cfg(cfg, "pose_select_spatial_min_distance_scale", "POSE_SELECT_SPATIAL_MIN_DISTANCE_SCALE", 1.2)
        ),
        spatial_max_per_cell=int(
            _cfg(cfg, "pose_select_spatial_max_per_cell", "POSE_SELECT_SPATIAL_MAX_PER_CELL", 1)
        ),
        spatial_max_gap_frames=int(
            _cfg(cfg, "pose_select_spatial_max_gap_frames", "POSE_SELECT_SPATIAL_MAX_GAP_FRAMES", 150)
        ),
        spatial_min_keep_ratio=float(
            _cfg(cfg, "pose_select_spatial_min_keep_ratio", "POSE_SELECT_SPATIAL_MIN_KEEP_RATIO", 0.45)
        ),
        spatial_max_keep_ratio=float(
            _cfg(cfg, "pose_select_spatial_max_keep_ratio", "POSE_SELECT_SPATIAL_MAX_KEEP_RATIO", 0.80)
        ),
    )

    selected_payload = select_required_poses(result.poses, selector_cfg)
    selected_poses = selected_payload["selected"]
    selection_stats = dict(selected_payload["stats"])

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    internal_csv = write_internal_pose_csv(out_root / "pose_trajectory.csv", result.poses)

    export_format = str(_cfg(cfg, "pose_export_format", "POSE_EXPORT_FORMAT", "internal") or "internal").strip().lower()
    metashape_csv = None
    if export_format == "metashape":
        metashape_csv = write_metashape_csv(out_root / "metashape_import.csv", selected_poses)

    selected_images_dir = out_root / "selected_images"
    copied = export_selected_images(Path(image_dir), selected_poses, selected_images_dir)

    return {
        "backend": backend,
        "result": result,
        "trajectory_count": len(result.poses),
        "selected_count": len(selected_poses),
        "selection_stats": selection_stats,
        "pose_trajectory_csv": str(internal_csv),
        "metashape_csv": str(metashape_csv) if metashape_csv else None,
        "selected_images_dir": str(selected_images_dir),
        "selected_images_list": copied.get("selected_list_path"),
        "copied_count": int(copied.get("copied_count", 0)),
    }
