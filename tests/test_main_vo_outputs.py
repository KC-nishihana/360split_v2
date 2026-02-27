import csv

from main import summarize_vo_diagnostics, write_vo_diagnostics


def test_vo_diagnostics_include_confidence_stats(tmp_path):
    records = [
        {
            "frame_index": 0,
            "t_xyz": [0.0, 0.0, 0.0],
            "q_wxyz": [1.0, 0.0, 0.0, 0.0],
            "metrics": {
                "vo_attempted": 1.0,
                "vo_valid": 1.0,
                "vo_pose_valid": 1.0,
                "vo_status_reason": "enabled",
                "vo_inlier_ratio": 0.8,
                "vo_step_proxy": 4.0,
                "vo_confidence": 0.75,
            },
        },
        {
            "frame_index": 1,
            "t_xyz": [1.0, 0.0, 0.0],
            "q_wxyz": [1.0, 0.0, 0.0, 0.0],
            "metrics": {
                "vo_attempted": 1.0,
                "vo_valid": 0.0,
                "vo_pose_valid": 0.0,
                "vo_status_reason": "estimate_failed_or_low_inlier",
                "vo_inlier_ratio": 0.2,
                "vo_step_proxy": 1.0,
                "vo_confidence": 0.20,
            },
        },
    ]

    summary = summarize_vo_diagnostics(records)
    assert "vo_confidence_mean" in summary
    assert 0.0 <= float(summary["vo_confidence_mean"]) <= 1.0

    _, csv_path, _ = write_vo_diagnostics(tmp_path, records)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(iter(reader))
        assert "vo_confidence" in row
