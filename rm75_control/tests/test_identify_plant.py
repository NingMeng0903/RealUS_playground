"""Horizon error tubes from the plant-identification script."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from peirastic.apps.identify_plant import (
    analyze_csv,
    analyze_stop_reverse,
    _align_trigger_index,
    _event_metrics,
    _horizon_error_ub,
    _load_twist_csv,
    _open_loop_envelopes,
    _rollout_fopdt,
    _simulate_first_order,
    _step_dt,
)


def test_horizon_error_ub_has_nb_steps() -> None:
    dt = 0.005
    n = 240
    cmd = np.zeros(n)
    cmd[20:80] = 0.04
    cmd[140:200] = -0.04
    ach = _simulate_first_order(cmd, delay_steps=1, tp_s=0.040, dt=dt)
    ach = ach + 0.001 * np.sin(np.linspace(0.0, 6.0, n))
    ev, ex = _horizon_error_ub(
        cmd,
        ach,
        delay_steps=1,
        tp_s=0.040,
        dt=dt,
        horizon=40,
        margin=0.002,
    )
    assert len(ev) == 40
    assert len(ex) == 40
    assert all(e >= 0.001 for e in ev)
    assert ex[-1] > ex[0]
    assert all(ex[i] <= ex[i + 1] + 1e-12 for i in range(len(ex) - 1))


def test_analyze_csv_writes_shield_yaml(tmp_path: Path) -> None:
    dt = 0.005
    n = 300
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    cmd[30:90] = 0.04
    cmd[160:220] = -0.03
    ach = _simulate_first_order(cmd, delay_steps=2, tp_s=0.050, dt=dt)
    csv_path = tmp_path / "plant.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t_wall_s", "twist_vz", "twist_achieved_vz"]
        )
        writer.writeheader()
        for ti, u, v in zip(t, cmd, ach):
            writer.writerow(
                {
                    "t_wall_s": f"{ti:.6f}",
                    "twist_vz": f"{u:.6f}",
                    "twist_achieved_vz": f"{v:.6f}",
                }
            )
    yaml_path = tmp_path / "tube.yaml"
    assert analyze_csv(csv_path, horizon=20, write_yaml=yaml_path, margin=0.001) == 0
    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded["safety_shield"]["plant"]["horizon_steps"] == 20
    assert len(loaded["safety_shield"]["velocity_error_ub_m_s"]) == 20
    assert len(loaded["safety_shield"]["position_error_ub_m"]) == 20
    assert len(loaded["safety_shield"]["position_error_ub_plus_m"]) == 20


def test_analyze_stop_reverse_finds_stop_and_reverse(tmp_path: Path) -> None:
    dt = 0.005
    cmd = np.zeros(400)
    cmd[20:80] = 0.040
    cmd[80:140] = 0.0
    cmd[160:220] = 0.040
    cmd[220:280] = -0.040
    t = np.arange(cmd.size) * dt
    ach = _simulate_first_order(cmd, delay_steps=4, tp_s=0.020, dt=dt)
    csv_path = tmp_path / "stop.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t_wall_s", "twist_vz", "twist_achieved_vz"]
        )
        writer.writeheader()
        for ti, u, v in zip(t, cmd, ach):
            writer.writerow(
                {
                    "t_wall_s": f"{ti:.6f}",
                    "twist_vz": f"{u:.6f}",
                    "twist_achieved_vz": f"{v:.6f}",
                }
            )
    assert analyze_stop_reverse(csv_path, horizon=40) == 0


def test_event_metrics_press_dx_does_not_cancel_retract() -> None:
    dt = 0.005
    n = 80
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    cmd[:10] = 0.04
    ach = np.zeros(n)
    ach[:5] = 0.004
    ach[5:10] = -0.004
    met = _event_metrics(cmd, ach, t, start=0, settle_m_s=0.003, horizon=40)
    signed = float(np.sum(ach) * dt)
    assert abs(signed) < 1e-6
    assert met["dx_press"] == pytest.approx(5 * 0.004 * dt)
    assert met["dx_press"] > 0.0
    assert met["n_press"] == pytest.approx(5.0)


def test_event_metrics_nb_is_hold_not_first_crossing() -> None:
    dt = 0.005
    n = 80
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    ach = np.full(n, 0.020)
    ach[3] = 0.0
    ach[20:] = 0.0
    met = _event_metrics(
        cmd,
        ach,
        t,
        start=0,
        settle_m_s=0.003,
        horizon=40,
        hold_s=0.050,
        delay_s=0.0,
        u_clear_m_s=0.015,
        v_hold_m_s=0.015,
        a_hold_m_s2=50.0,
    )
    assert met["n_press"] == pytest.approx(20.0)
    assert math.isfinite(met["n_b"])
    assert met["n_b"] >= 20 + int(0.050 / dt) - 1
    assert met["n_b"] != pytest.approx(4.0)


def test_open_loop_envelopes_share_origins() -> None:
    dt = 0.005
    n = 120
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    cmd[10:50] = 0.04
    ach = _simulate_first_order(cmd, delay_steps=2, tp_s=0.020, dt=dt)
    dt_arr = _step_dt(t, dt)
    origins = [8, 16, 24]
    ev, ex, ex_plus, used = _open_loop_envelopes(
        cmd,
        ach,
        t,
        origins=origins,
        horizon=10,
        rollout_fn=lambda k: _rollout_fopdt(cmd, ach, k, 10, 2, 0.020, dt_arr),
    )
    assert used == 3
    assert len(ev) == 10
    assert len(ex) == 10
    assert len(ex_plus) == 10
    assert max(ev) < 1e-9


def test_event_metrics_does_not_force_nb_to_horizon() -> None:
    dt = 0.005
    n = 40
    t = np.arange(n) * dt
    cmd = np.full(n, 0.040)
    ach = np.full(n, 0.040)
    met = _event_metrics(cmd, ach, t, start=0, settle_m_s=0.003, horizon=40)
    assert met["reached_T"] == pytest.approx(0.0)
    assert not math.isfinite(met["n_b"])
    assert met["n_press"] == pytest.approx(40.0)


def test_step_dt_does_not_clip_large_gaps() -> None:
    t = np.array([0.0, 0.005, 0.010, 0.210])
    dt = _step_dt(t, 0.005)
    assert dt[-1] == pytest.approx(0.200)


def test_event_metrics_invalidates_timestamp_gap() -> None:
    dt = 0.005
    n = 80
    t = np.arange(n) * dt
    t[40:] += 0.20
    cmd = np.zeros(n)
    ach = np.full(n, 0.004)
    met = _event_metrics(cmd, ach, t, start=0, settle_m_s=0.003, horizon=40)
    assert met["gap_invalid"] == pytest.approx(1.0)
    assert not math.isfinite(met["dx_press"])


def test_load_twist_csv_defaults_to_twist_vz(tmp_path: Path) -> None:
    csv_path = tmp_path / "cols.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t_wall_s", "twist_vz", "u_sent", "twist_achieved_vz"]
        )
        writer.writeheader()
        for i in range(10):
            writer.writerow(
                {
                    "t_wall_s": f"{i * 0.005:.6f}",
                    "twist_vz": "0.040000",
                    "u_sent": "0.080000",
                    "twist_achieved_vz": "0.020000",
                }
            )
    cmd, ach, _t, twist, sent = _load_twist_csv(csv_path, input_column="twist_vz")
    assert float(np.mean(cmd)) == pytest.approx(0.04)
    assert float(np.mean(twist)) == pytest.approx(0.04)
    assert float(np.mean(sent)) == pytest.approx(0.08)
    cmd_s, *_rest = _load_twist_csv(csv_path, input_column="u_sent")
    assert float(np.mean(cmd_s)) == pytest.approx(0.08)


def test_analyze_stop_writes_uncertified_stop_dx_yaml(tmp_path: Path) -> None:
    dt = 0.005
    cmd = np.zeros(400)
    cmd[20:80] = 0.040
    cmd[80:140] = 0.0
    cmd[160:220] = 0.040
    cmd[220:280] = -0.040
    t = np.arange(cmd.size) * dt
    ach = _simulate_first_order(cmd, delay_steps=4, tp_s=0.020, dt=dt)
    csv_path = tmp_path / "stop.csv"
    yaml_path = tmp_path / "stop_dx.yaml"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t_wall_s", "twist_vz", "twist_achieved_vz"]
        )
        writer.writeheader()
        for ti, u, v in zip(t, cmd, ach):
            writer.writerow(
                {
                    "t_wall_s": f"{ti:.6f}",
                    "twist_vz": f"{u:.6f}",
                    "twist_achieved_vz": f"{v:.6f}",
                }
            )
    assert analyze_stop_reverse(csv_path, horizon=40, write_yaml=yaml_path) == 0
    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded["safety_shield"]["stop_dx_ub"]["certified"] is False
    bins = loaded["safety_shield"]["stop_dx_ub"]["bins"]
    assert bins
    assert "q_remain_m" in bins[0]
    assert "a0_m_s2" in bins[0]


def test_open_loop_ex_plus_does_not_cancel() -> None:
    dt = 0.005
    n = 40
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    ach = np.zeros(n)
    ach[11] = 0.010
    ach[12] = -0.010

    def pred(_k: int) -> np.ndarray:
        return np.zeros(8)

    _ev, ex, ex_plus, used = _open_loop_envelopes(
        cmd,
        ach,
        t,
        origins=[10],
        horizon=8,
        rollout_fn=pred,
    )
    assert used == 1
    assert ex[1] < 1e-9
    assert ex_plus[1] == pytest.approx(0.010 * dt)


def test_event_metrics_a0_is_press_positive_actual() -> None:
    dt = 0.005
    n = 40
    t = np.arange(n) * dt
    cmd = np.full(n, 0.040)
    ach = np.zeros(n)
    ach[9] = 0.010
    ach[10] = 0.016
    met = _event_metrics(cmd, ach, t, start=10, settle_m_s=0.003, horizon=20)
    assert met["a0"] == pytest.approx((0.016 - 0.010) / dt)
    cmd[5:10] = 0.040
    met2 = _event_metrics(
        cmd, ach, t, start=10, settle_m_s=0.003, horizon=20, delay_s=0.025
    )
    assert met2["q_remain_m"] > 0.0
    assert met2["a0"] >= 0.0


def test_align_trigger_uses_first_ub_not_velocity_edge() -> None:
    dt = 0.005
    n = 80
    t = np.arange(n) * dt
    cmd = np.full(n, 0.020)
    cmd[40] = 0.019
    cmd[41] = 0.017
    cmd[42] = 0.014
    cmd[50:] = 0.0
    idx = _align_trigger_index(
        cmd,
        t,
        {"v0_cmd": "0.020000", "u_b": "0.019000"},
        search_from=1,
    )
    assert idx == 40


def test_analyze_stop_aligns_event_log(tmp_path: Path) -> None:
    dt = 0.005
    n = 120
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    cmd[20:50] = 0.020
    cmd[50] = 0.019
    cmd[51] = 0.016
    cmd[52] = 0.012
    cmd[53:80] = 0.0
    ach = cmd.copy()
    csv_path = tmp_path / "win_a.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t_wall_s", "twist_vz", "twist_achieved_vz"]
        )
        writer.writeheader()
        for ti, u, v in zip(t, cmd, ach):
            writer.writerow(
                {
                    "t_wall_s": f"{ti:.6f}",
                    "twist_vz": f"{u:.6f}",
                    "twist_achieved_vz": f"{v:.6f}",
                }
            )
    log_path = tmp_path / "events.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["event_id", "trigger", "tick", "v0_cmd", "u_b"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "001_20_settled",
                "trigger": "backup_to_terminal",
                "tick": "0",
                "v0_cmd": "0.020000",
                "u_b": "0.019000",
            }
        )
    assert analyze_stop_reverse(csv_path, horizon=40, event_log=log_path) == 0

