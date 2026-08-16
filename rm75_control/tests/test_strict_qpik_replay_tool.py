from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apps.joint_admittance_8dof.replay_strict_qpik import replay_csv
from rm75_control.control.joint_admittance_8dof.model import full_q_from_arm


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
Q0 = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _write_fixture(path: Path) -> None:
    fields = (
        ["t_wall_s", "dt_actual_s"]
        + [f"q_cmd_{i}" for i in range(8)]
        + [f"q_meas_{i}" for i in range(8)]
        + [f"v_cmd_{axis}" for axis in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + ["qdot_meas_0"]
        + ["qpik_final_sent_qdot_json"]
    )
    rows = []
    for index, rail_v in enumerate(("", "0.020", "0.030")):
        q_cmd = Q0 + (index + 1) * 1.0e-4
        qdot_sent = np.linspace(0.001, 0.008, 8) * (index + 1)
        row = {
            "t_wall_s": f"{index * 0.005:.6f}",
            "dt_actual_s": "0.005",
            "qdot_meas_0": rail_v,
            "qpik_final_sent_qdot_json": json.dumps(qdot_sent.tolist()),
        }
        row.update({f"q_cmd_{i}": f"{value:.9f}" for i, value in enumerate(q_cmd)})
        row.update({f"q_meas_{i}": f"{value:.9f}" for i, value in enumerate(Q0)})
        row.update(
            {
                "v_cmd_vx": "0.000",
                "v_cmd_vy": "0.004",
                "v_cmd_vz": "0.000",
                "v_cmd_wx": "0.000",
                "v_cmd_wy": "0.000",
                "v_cmd_wz": "0.000",
            }
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_replay_csv_writes_strict_qpik_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "small_wbc.csv"
    output = tmp_path / "strict_replay.csv"
    _write_fixture(source)

    result = replay_csv(
        source,
        CONFIG,
        output_csv=output,
        disable_cbf=True,
    )
    rows = result["rows"]
    summary = result["summary"]

    assert len(rows) == 3
    assert summary["mode"] == "velocity_level_log_replay"
    assert summary["hardware_execution_reconstructed"] is False
    assert summary["cbf_enabled"] is False
    assert summary["history_mode"] == {"logged_single_tick_snapshot": 3}
    assert summary["qp1_status_counts"] == {"solved": 3}
    assert summary["qp2_status_counts"] == {"solved": 3}
    assert summary["over_5ms_count"] == sum(row["timing_over_5ms"] for row in rows)
    assert rows[0]["rail_measurement_source"] == "initial_zero"
    assert rows[1]["rail_measurement_source"] == "qdot_meas_0"
    assert rows[1]["rail_measured_velocity_m_s"] == 0.02
    np.testing.assert_allclose(
        json.loads(rows[1]["snapshot_q_prev_json"]),
        Q0 + 1.0e-4,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        json.loads(rows[1]["snapshot_qdot_prev_json"]),
        np.linspace(0.001, 0.008, 8),
        atol=1.0e-12,
    )
    assert float(rows[1]["dt_s"]) == 0.005
    assert float(rows[1]["box_dt_s"]) == 0.005
    assert summary["history_fallback_counts"]["q_cmd_prev_from_current_q_meas"] == 1
    assert np.isfinite([row["tcp_residual_inf_m_s"] for row in rows]).all()
    assert all(len(json.loads(row["tcp_residual_json"])) == 6 for row in rows)
    assert all(len(json.loads(row["rail_measured_contribution_json"])) == 6 for row in rows)
    assert all(len(json.loads(row["arm_contribution_json"])) == 6 for row in rows)
    assert output.exists()
    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 3
    assert {
        "tcp_residual_inf_m_s",
        "qp1_status",
        "qp2_status",
        "rail_command_m_s",
        "rail_measured_contribution_json",
        "arm_contribution_json",
        "hard_active_constraint_ids_json",
        "qpik_total_ms",
        "box_dt_s",
        "history_mode",
        "snapshot_q_prev_json",
        "timing_over_5ms",
    }.issubset(written[0])


def test_replay_stride_and_max_rows_select_source_rows(tmp_path: Path) -> None:
    source = tmp_path / "small_wbc.csv"
    _write_fixture(source)
    result = replay_csv(source, CONFIG, stride=2, max_rows=2, disable_cbf=True)
    assert len(result["rows"]) == 2
    assert result["rows"][0]["source_row"] == 0
    assert result["rows"][1]["source_row"] == 2
    # Row 1 is intentionally unselected; it must still provide row 2's
    # adjacent logged command/velocity snapshot.
    np.testing.assert_allclose(
        json.loads(result["rows"][1]["snapshot_q_prev_json"]),
        Q0 + 2.0e-4,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        json.loads(result["rows"][1]["snapshot_qdot_prev_json"]),
        np.linspace(0.001, 0.008, 8) * 2.0,
        atol=1.0e-12,
    )
    assert result["summary"]["rows"] == 2


def test_replay_free_running_integrates_controller_state(tmp_path: Path) -> None:
    source = tmp_path / "small_wbc.csv"
    _write_fixture(source)
    result = replay_csv(
        source,
        CONFIG,
        disable_cbf=True,
        mode="free-running",
    )
    rows = result["rows"]
    summary = result["summary"]
    assert len(rows) == 3
    assert summary["replay_mode"] == "free-running"
    assert summary["mode"] == "free_running_log_replay"
    assert summary["history_mode"] == {"free_running": 3}
    assert all(row["history_mode"] == "free_running" for row in rows)
    assert all(row["rail_measurement_source"] == "controller_last_rail_command" for row in rows)
    assert summary["qp1_status_counts"] == {"solved": 3}
    assert np.isfinite([row["tcp_residual_inf_m_s"] for row in rows]).all()
    # The controller owns q; later ticks must not snap back to the logged q_cmd.
    q_last = json.loads(rows[-1]["snapshot_q_prev_json"])
    assert abs(float(q_last[0]) - float(Q0[0])) < 0.05
