"""Acceptance-metric regressions for the loaded LW100 scan (no hardware)."""

from __future__ import annotations

import pytest

from apps.lw100_loaded_pd_scan import analyze_rows


def _row(index: int, *, tail_error_m: float = 0.0) -> dict[str, str]:
    t_s = index * 0.02
    center_m = 0.400
    return {
        "t_s": f"{t_s:.6f}",
        "x_tgt_m": f"{center_m:.9f}",
        "x_goal_eval_m": f"{center_m:.9f}",
        "x_ref_m": f"{center_m:.9f}",
        "x_meas_m": f"{center_m - tail_error_m:.9f}",
        "v_ref_m_s": "0.010000",
        "a_ref_m_s2": "0.000000",
        "v_goal_est_m_s": "0.010000",
        "v_meas_m_s": "0.010000",
        "v_cmd_m_s": "0.010000",
        "a_cmd_m_s2": "0.000000",
        "meas_rpm": "60",
        "hold_count": "0",
        "freeze_flag": "0",
    }


def test_full_run_instability_cannot_hide_outside_steady_score_window() -> None:
    rows = [_row(i, tail_error_m=0.040 if i >= 700 else 0.0) for i in range(750)]

    metrics = analyze_rows(rows, center_m=0.400, amp_m=0.040)

    assert metrics["p95_mm"] == 0.0
    assert metrics["all_p95_mm"] == pytest.approx(40.0)
    assert metrics["all_max_mm"] == pytest.approx(40.0)
    assert metrics["smooth_pass"] == 0


def test_extra_endpoint_reversal_fails_smoothness() -> None:
    rows = [_row(i) for i in range(750)]
    rows[400]["v_ref_m_s"] = "-0.002000"
    rows[400]["v_cmd_m_s"] = "-0.002000"

    metrics = analyze_rows(rows, center_m=0.400, amp_m=0.040)

    assert metrics["extra_ref_reversals"] == 2
    assert metrics["extra_cmd_reversals"] == 2
    assert metrics["smooth_pass"] == 0
