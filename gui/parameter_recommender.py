"""Rule-based parameter recommender for analysis dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _get_setting(settings: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in settings:
            return settings[key]
    for key in keys:
        up = key.upper()
        if up in settings:
            return settings[up]
    return None


def _recommend(
    out: List[Dict[str, Any]],
    *,
    stage: str,
    parameter: str,
    suggested_key: str,
    current: Any,
    suggested: Any,
    reason: str,
) -> None:
    if current == suggested:
        return
    out.append(
        {
            "stage": stage,
            "parameter": parameter,
            "setting_key": suggested_key,
            "current": current,
            "suggested": suggested,
            "reason": reason,
        }
    )


def _stage1_recommendations(payload: Dict[str, Any], settings: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    stage1 = payload.get("stage1") if isinstance(payload.get("stage1"), dict) else {}
    pass_rate = _to_float(stage1.get("pass_rate"))
    abs_guard_ratio = _to_float(stage1.get("abs_laplacian_guard_ratio"))
    if pass_rate is None:
        return

    q_th = _to_float(_get_setting(settings, "quality_threshold"))
    if q_th is not None:
        if pass_rate < 0.12:
            q_new = _clamp(q_th - 0.05, 0.30, 0.80)
            _recommend(
                out,
                stage="Stage1",
                parameter="quality_threshold",
                suggested_key="quality_threshold",
                current=round(q_th, 4),
                suggested=round(q_new, 4),
                reason=f"pass率 {pass_rate:.1%} が低いため閾値を緩和",
            )
        elif pass_rate > 0.40:
            q_new = _clamp(q_th + 0.05, 0.30, 0.80)
            _recommend(
                out,
                stage="Stage1",
                parameter="quality_threshold",
                suggested_key="quality_threshold",
                current=round(q_th, 4),
                suggested=round(q_new, 4),
                reason=f"pass率 {pass_rate:.1%} が高いため閾値を厳しく調整",
            )

    lap_min = _to_float(_get_setting(settings, "quality_abs_laplacian_min"))
    if lap_min is not None:
        if abs_guard_ratio is not None and abs_guard_ratio > 0.20 and pass_rate < 0.20:
            lap_new = _clamp(lap_min - 5.0, 20.0, 80.0)
            _recommend(
                out,
                stage="Stage1",
                parameter="quality_abs_laplacian_min",
                suggested_key="quality_abs_laplacian_min",
                current=round(lap_min, 2),
                suggested=round(lap_new, 2),
                reason=(
                    f"abs_laplacian_guard比率 {abs_guard_ratio:.1%} かつ pass率 {pass_rate:.1%} のため"
                    "最小ラプラシアンを緩和"
                ),
            )
        elif pass_rate > 0.50:
            lap_new = _clamp(lap_min + 5.0, 20.0, 80.0)
            _recommend(
                out,
                stage="Stage1",
                parameter="quality_abs_laplacian_min",
                suggested_key="quality_abs_laplacian_min",
                current=round(lap_min, 2),
                suggested=round(lap_new, 2),
                reason=f"pass率 {pass_rate:.1%} が高いため最小ラプラシアンを引き上げ",
            )


def _stage2_recommendations(payload: Dict[str, Any], settings: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    stage2 = payload.get("stage2") if isinstance(payload.get("stage2"), dict) else {}
    ssim_skip_ratio = _to_float(stage2.get("ssim_skip_ratio"))
    min_interval_ratio = _to_float(stage2.get("min_interval_ratio"))
    final_keyframes = _to_int(stage2.get("final_keyframes"))
    target_min = _to_int(stage2.get("target_min"))
    target_max = _to_int(stage2.get("target_max"))

    if (
        ssim_skip_ratio is None
        or min_interval_ratio is None
        or final_keyframes is None
        or target_min is None
        or target_max is None
        or target_min <= 0
        or target_max < target_min
    ):
        return

    ssim_th = _to_float(_get_setting(settings, "ssim_threshold", "ssim_change_threshold"))
    if ssim_th is not None:
        if ssim_skip_ratio > 0.35 and final_keyframes < target_min:
            new_val = _clamp(ssim_th - 0.03, 0.70, 0.97)
            _recommend(
                out,
                stage="Stage2",
                parameter="ssim_change_threshold",
                suggested_key="ssim_threshold",
                current=round(ssim_th, 4),
                suggested=round(new_val, 4),
                reason=(
                    f"ssim_skip比率 {ssim_skip_ratio:.1%} かつ final_keyframes={final_keyframes} < target_min={target_min}"
                ),
            )
        elif ssim_skip_ratio < 0.05 and final_keyframes > target_max:
            new_val = _clamp(ssim_th + 0.03, 0.70, 0.97)
            _recommend(
                out,
                stage="Stage2",
                parameter="ssim_change_threshold",
                suggested_key="ssim_threshold",
                current=round(ssim_th, 4),
                suggested=round(new_val, 4),
                reason=(
                    f"ssim_skip比率 {ssim_skip_ratio:.1%} かつ final_keyframes={final_keyframes} > target_max={target_max}"
                ),
            )

    min_interval = _to_int(_get_setting(settings, "min_keyframe_interval"))
    if min_interval is not None:
        if min_interval_ratio > 0.35 and final_keyframes < target_min:
            new_val = int(round(_clamp(float(min_interval - 1), 1.0, 30.0)))
            _recommend(
                out,
                stage="Stage2",
                parameter="min_keyframe_interval",
                suggested_key="min_keyframe_interval",
                current=int(min_interval),
                suggested=int(new_val),
                reason=(
                    f"min_interval比率 {min_interval_ratio:.1%} かつ final_keyframes={final_keyframes} < target_min={target_min}"
                ),
            )
        elif min_interval_ratio < 0.05 and final_keyframes > target_max:
            new_val = int(round(_clamp(float(min_interval + 1), 1.0, 30.0)))
            _recommend(
                out,
                stage="Stage2",
                parameter="min_keyframe_interval",
                suggested_key="min_keyframe_interval",
                current=int(min_interval),
                suggested=int(new_val),
                reason=(
                    f"min_interval比率 {min_interval_ratio:.1%} かつ final_keyframes={final_keyframes} > target_max={target_max}"
                ),
            )


def _stage3_recommendations(payload: Dict[str, Any], settings: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    stage3 = payload.get("stage3") if isinstance(payload.get("stage3"), dict) else {}
    diagnostics = stage3.get("diagnostics") if isinstance(stage3.get("diagnostics"), dict) else {}
    vo_valid_ratio = _to_float(stage3.get("vo_valid_ratio"))
    coverage_alert = bool(diagnostics.get("coverage_alert", False))
    novelty_alert = bool(diagnostics.get("novelty_alert", False))

    traj_w = _to_float(_get_setting(settings, "stage3_weight_trajectory"))
    base_w = _to_float(_get_setting(settings, "stage3_weight_base"))
    risk_w = _to_float(_get_setting(settings, "stage3_weight_stage0_risk"))

    traj_delta = 0.0
    if traj_w is not None and vo_valid_ratio is not None:
        if vo_valid_ratio < 0.50:
            new_traj = _clamp(traj_w - 0.10, 0.05, 0.45)
            traj_delta = new_traj - traj_w
            _recommend(
                out,
                stage="Stage3",
                parameter="stage3_weight_trajectory",
                suggested_key="stage3_weight_trajectory",
                current=round(traj_w, 4),
                suggested=round(new_traj, 4),
                reason=f"vo_valid_ratio {vo_valid_ratio:.1%} が閾値 50% 未満",
            )
        elif coverage_alert and vo_valid_ratio >= 0.50:
            new_traj = _clamp(traj_w + 0.05, 0.05, 0.45)
            traj_delta = new_traj - traj_w
            _recommend(
                out,
                stage="Stage3",
                parameter="stage3_weight_trajectory",
                suggested_key="stage3_weight_trajectory",
                current=round(traj_w, 4),
                suggested=round(new_traj, 4),
                reason="coverage_alert=true のため軌跡重みを増加",
            )

    if base_w is not None and abs(traj_delta) > 1e-9:
        if traj_delta < 0.0:
            base_new = _clamp(base_w + 0.05, 0.50, 0.90)
            _recommend(
                out,
                stage="Stage3",
                parameter="stage3_weight_base",
                suggested_key="stage3_weight_base",
                current=round(base_w, 4),
                suggested=round(base_new, 4),
                reason="trajectory重み低下に連動してbase重みを補正",
            )
        elif traj_delta > 0.0:
            base_new = _clamp(base_w - 0.03, 0.50, 0.90)
            _recommend(
                out,
                stage="Stage3",
                parameter="stage3_weight_base",
                suggested_key="stage3_weight_base",
                current=round(base_w, 4),
                suggested=round(base_new, 4),
                reason="trajectory重み増加に連動してbase重みを補正",
            )

    if risk_w is not None and novelty_alert:
        risk_new = _clamp(risk_w + 0.02, 0.00, 0.20)
        _recommend(
            out,
            stage="Stage3",
            parameter="stage3_weight_stage0_risk",
            suggested_key="stage3_weight_stage0_risk",
            current=round(risk_w, 4),
            suggested=round(risk_new, 4),
            reason="novelty_alert=true のため stage0 risk 重みを増加",
        )


def _colmap_target_shift(target_min: int, target_max: int, increase: bool) -> Tuple[int, int]:
    if increase:
        nmin = max(1, int(round(target_min * 1.15)))
        nmax = max(nmin, int(round(target_max * 1.15)))
        return nmin, nmax
    nmin = max(1, int(round(target_min * 0.90)))
    nmax = max(nmin, int(round(target_max * 0.90)))
    return nmin, nmax


def _colmap_recommendations(payload: Dict[str, Any], settings: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    colmap = payload.get("colmap") if isinstance(payload.get("colmap"), dict) else {}

    final_keyframes = _to_int(colmap.get("final_keyframes"))
    target_min = _to_int(colmap.get("target_min"))
    target_max = _to_int(colmap.get("target_max"))
    preview_count = _to_int(colmap.get("stage2_colmap_preview_count"))
    preview_max_gap = _to_float(colmap.get("stage2_colmap_preview_max_gap"))
    dense_neighbor_ratio = _to_float(colmap.get("dense_neighbor_low_novelty_ratio"))
    total_frames = _to_int(colmap.get("total_frames"))

    cur_target_min = _to_int(_get_setting(settings, "colmap_keyframe_target_min"))
    cur_target_max = _to_int(_get_setting(settings, "colmap_keyframe_target_max"))
    cur_budget = _to_int(_get_setting(settings, "colmap_stage2_entry_budget"))
    cur_min_gap = _to_int(_get_setting(settings, "colmap_stage2_entry_min_gap"))
    cur_div_ssim = _to_float(_get_setting(settings, "colmap_diversity_ssim_threshold"))

    # target min/max
    if (
        final_keyframes is not None
        and target_min is not None
        and target_max is not None
        and target_min > 0
        and target_max >= target_min
        and cur_target_min is not None
        and cur_target_max is not None
    ):
        if final_keyframes < target_min * 0.9:
            smin, smax = _colmap_target_shift(cur_target_min, cur_target_max, increase=True)
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_keyframe_target_min",
                suggested_key="colmap_keyframe_target_min",
                current=int(cur_target_min),
                suggested=int(smin),
                reason=(
                    f"final_keyframes={final_keyframes} < target_min*0.9 ({target_min * 0.9:.1f}) のため +15%"
                ),
            )
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_keyframe_target_max",
                suggested_key="colmap_keyframe_target_max",
                current=int(cur_target_max),
                suggested=int(smax),
                reason="target_minと連動して target_max も +15%",
            )
        elif final_keyframes > target_max * 1.15:
            smin, smax = _colmap_target_shift(cur_target_min, cur_target_max, increase=False)
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_keyframe_target_min",
                suggested_key="colmap_keyframe_target_min",
                current=int(cur_target_min),
                suggested=int(smin),
                reason=(
                    f"final_keyframes={final_keyframes} > target_max*1.15 ({target_max * 1.15:.1f}) のため -10%"
                ),
            )
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_keyframe_target_max",
                suggested_key="colmap_keyframe_target_max",
                current=int(cur_target_max),
                suggested=int(smax),
                reason="target_minと連動して target_max も -10%",
            )

    # stage2 entry budget
    if (
        preview_count is not None
        and target_min is not None
        and target_max is not None
        and cur_budget is not None
        and target_min > 0
        and target_max >= target_min
    ):
        if preview_count < target_min * 0.8:
            budget_new = max(1, int(round(cur_budget * 1.20)))
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_stage2_entry_budget",
                suggested_key="colmap_stage2_entry_budget",
                current=int(cur_budget),
                suggested=int(budget_new),
                reason=(
                    f"stage2_colmap_preview_count={preview_count} < target_min*0.8 ({target_min * 0.8:.1f}) のため +20%"
                ),
            )
        elif preview_count > target_max * 1.2:
            budget_new = max(1, int(round(cur_budget * 0.85)))
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_stage2_entry_budget",
                suggested_key="colmap_stage2_entry_budget",
                current=int(cur_budget),
                suggested=int(budget_new),
                reason=(
                    f"stage2_colmap_preview_count={preview_count} > target_max*1.2 ({target_max * 1.2:.1f}) のため -15%"
                ),
            )

    # stage2 entry min gap
    if (
        preview_max_gap is not None
        and cur_min_gap is not None
        and cur_budget is not None
        and total_frames is not None
        and total_frames > 0
        and cur_budget > 0
    ):
        expected_gap = float(total_frames) / float(max(cur_budget, 1))
        if preview_max_gap > expected_gap * 2.0:
            new_gap = int(round(_clamp(float(cur_min_gap - 1), 1.0, 10.0)))
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_stage2_entry_min_gap",
                suggested_key="colmap_stage2_entry_min_gap",
                current=int(cur_min_gap),
                suggested=int(new_gap),
                reason=(
                    f"preview_max_gap={preview_max_gap:.1f} > expected_gap*2 ({expected_gap * 2.0:.1f})"
                ),
            )
        elif preview_max_gap < expected_gap * 0.8:
            new_gap = int(round(_clamp(float(cur_min_gap + 1), 1.0, 10.0)))
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_stage2_entry_min_gap",
                suggested_key="colmap_stage2_entry_min_gap",
                current=int(cur_min_gap),
                suggested=int(new_gap),
                reason=(
                    f"preview_max_gap={preview_max_gap:.1f} < expected_gap*0.8 ({expected_gap * 0.8:.1f})"
                ),
            )

    # diversity ssim threshold
    preview_shortage = (
        preview_count is not None and target_min is not None and target_min > 0 and preview_count < target_min * 0.8
    )
    if cur_div_ssim is not None and dense_neighbor_ratio is not None:
        if dense_neighbor_ratio > 0.25:
            new_th = _clamp(cur_div_ssim - 0.01, 0.85, 0.99)
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_diversity_ssim_threshold",
                suggested_key="colmap_diversity_ssim_threshold",
                current=round(cur_div_ssim, 4),
                suggested=round(new_th, 4),
                reason=(
                    f"dense_neighbor_low_novelty比率 {dense_neighbor_ratio:.1%} が高いためしきい値を緩和"
                ),
            )
        elif dense_neighbor_ratio < 0.05 and preview_shortage:
            new_th = _clamp(cur_div_ssim + 0.01, 0.85, 0.99)
            _recommend(
                out,
                stage="COLMAP",
                parameter="colmap_diversity_ssim_threshold",
                suggested_key="colmap_diversity_ssim_threshold",
                current=round(cur_div_ssim, 4),
                suggested=round(new_th, 4),
                reason="dense_neighbor_low_novelty比率が低く preview不足のためしきい値を厳格化",
            )


def generate_parameter_recommendations(payload: Dict[str, Any], current_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate stage-wise recommendation rows from summarized artifacts."""
    if not isinstance(payload, dict) or not isinstance(current_settings, dict):
        return []

    out: List[Dict[str, Any]] = []
    _stage1_recommendations(payload, current_settings, out)
    _stage2_recommendations(payload, current_settings, out)
    _stage3_recommendations(payload, current_settings, out)
    _colmap_recommendations(payload, current_settings, out)
    return out
