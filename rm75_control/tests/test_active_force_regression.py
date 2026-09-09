"""Deterministic regression traces for the force controller's live paths.

The fixture in ``fixtures/active_force_baseline.npz`` was generated from the
controller before the cleanup of optional/retired layers.  It records the
emitted Cartesian command, motion-task decomposition, force state, contact
events, adaptive-Ke state, and the live flow/TDPA/shield observations.

Regenerate deliberately after an intentional controller change with:

    source rm75_control/env.sh
    python rm75_control/tests/test_active_force_regression.py --update-baseline

The traces use fixed samples and ``dt_actual`` values; no wall clock or robot
I/O is involved.  This is an output equivalence test, rather than a check for
class names or implementation details.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from peirastic.realman8dof.force.config import build_force_controller
from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)


DT = 0.005
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "active_force_baseline.npz"

_STATE_CODE = {"free": 0.0, "suspect_loss": 1.0, "contact": 2.0, "lost": 3.0}
_PRIVATE_TELEMETRY = {"recontact_timer_s": "_recontact_timer_s"}

# These are intentionally limited to live command/state contracts.  Retired
# DOB, tank, CDYOB, corridor, and surface-modulation telemetry is not pinned.
_TELEMETRY = (
    "v_force_cmd_z",
    "v_force_z",
    "v_r_z",
    "f_des_z_eff",
    "ke_est",
    "adaptive_bd",
    "mass_z_eff",
    "damping_z_eff",
    "instability_index",
    "force_pred_z",
    "cap_press_z",
    "cap_retract_z",
    "force_fast_z",
    "retract_guard_armed",
    "retract_fast_hold",
    "retract_fast_stop_count",
    "retract_fast_rearm_count",
    "force_reference_drive",
    "force_reference_scale_n",
    "force_reference_gate_scale",
    "force_reference_accel_m_s2",
    "force_reference_reversal_reset",
    "force_reference_fast_clear",
    "contact_present",
    "force_task_latched",
    "physical_contact_state",
    "physical_contact_acquire_event",
    "physical_contact_loss_event",
    "physical_contact_reacquire_event",
    "recontact_slow_latched",
    "recontact_detached_seen",
    "recontact_timer_s",
    "contact_episode_release_s",
    "force_point_z",
    "x_adm_z",
    "x_d_z",
    "x_tilde_z",
    "flow_alpha",
    "flow_tank_energy",
    "flow_fc",
    "flow_v_track",
    "flow_v_aux",
    "flow_retract_through",
    "flow_press",
    "flow_gamma_effective",
    "flow_feedback_stale",
    "flow_sign_verified",
    "tdpa_e_obs_j",
    "tdpa_alpha",
    "tdpa_clamped",
    "tdpa_passivity_holds",
    "shield_applied",
    "shield_feasible",
    "shield_f_ub_n",
    "shield_e_lb_j",
    "shield_w_lb_j",
    "shield_energy_margin_j",
    "shield_terminal_ok",
    "shield_recovery_latched",
    "shield_domain_ok",
    "shield_aj_ok",
    "shield_uncertified_brake",
    "shield_tube_violation",
)


def _step(
    f_z: float,
    desired_z: float,
    *,
    raw_z: float | None = None,
    in_contact: bool | None = None,
    pose: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    desired_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    vel_ff: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    dt_actual: float | None = None,
    v_tcp_z_actual: float | None = None,
    sensor_age_s: float | None = None,
) -> dict[str, Any]:
    return {
        "f_z": float(f_z),
        "desired_z": float(desired_z),
        "raw_z": None if raw_z is None else float(raw_z),
        "in_contact": in_contact,
        "pose": tuple(float(x) for x in pose),
        "desired_pose": tuple(float(x) for x in desired_pose),
        "vel_ff": tuple(float(x) for x in vel_ff),
        "dt_actual": dt_actual,
        "v_tcp_z_actual": v_tcp_z_actual,
        "sensor_age_s": sensor_age_s,
    }


def _scenario_steps(name: str) -> list[dict[str, Any]]:
    """Return a fixed input sequence for one live controller path."""

    if name == "force_seek_acquire_tracking":
        return (
            [_step(0.0, 4.0) for _ in range(12)]
            + [_step(f, 4.0, raw_z=f) for f in (0.2, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2)]
            + [_step(f, 4.0, raw_z=f) for f in (2.0, 3.0, 3.8, 4.0, 3.7, 4.2, 4.0)]
        )

    if name == "force_high_frequency":
        return [
            _step(
                4.0 + 0.7 * math.sin(2.0 * math.pi * 9.0 * i * DT),
                4.0,
                raw_z=4.0 + 0.7 * math.sin(2.0 * math.pi * 9.0 * i * DT)
                + 0.05 * math.cos(2.0 * math.pi * 9.0 * i * DT),
                in_contact=True,
                dt_actual=(0.004 if i % 3 == 0 else 0.005 if i % 3 == 1 else 0.006),
                v_tcp_z_actual=0.0,
                sensor_age_s=0.002 if i % 4 else 0.0,
            )
            for i in range(80)
        ]

    if name == "force_loss_recontact":
        # Automatic filtered contact acquisition, confirmed loss, and raw-only
        # recontact while the filtered channel is still low.
        return (
            [_step(1.1, 4.0, raw_z=1.1) for _ in range(8)]
            + [_step(0.2, 4.0, raw_z=0.2) for _ in range(26)]
            + [_step(0.2, 4.0, raw_z=1.1) for _ in range(5)]
        )

    if name == "force_retract":
        return (
            [_step(1.0, 4.0, raw_z=1.0) for _ in range(8)]
            + [_step(f, 4.0, raw_z=f, in_contact=True) for f in (6.0, 6.0, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5)]
        )

    if name == "force_path_tracking":
        pose = (0.010, -0.010, 0.002, 0.10, -0.10, 0.20)
        desired_pose = (0.012, -0.007, 0.002, 0.12, -0.05, 0.25)
        vel_ff = (0.020, -0.030, 0.0, 0.10, 0.05, -0.08)
        return [
            _step(
                3.2 + 0.1 * i,
                4.0,
                raw_z=3.2 + 0.1 * i,
                in_contact=True,
                pose=pose,
                desired_pose=desired_pose,
                vel_ff=vel_ff,
            )
            for i in range(8)
        ]

    if name == "force_payload_override":
        return (
            [_step(0.0, 2.5) for _ in range(6)]
            + [_step(f, 2.5, raw_z=f) for f in (0.4, 0.7, 0.9, 1.1, 1.4, 1.8, 2.2)]
            + [_step(2.5, 2.5, raw_z=2.5, in_contact=True) for _ in range(6)]
        )

    if name == "joint_proactive_force":
        return (
            [_step(0.9, 1.0, raw_z=0.9, in_contact=True) for _ in range(8)]
            + [_step(0.6, 1.0, raw_z=0.6, in_contact=True) for _ in range(24)]
            + [_step(1.4, 1.0, raw_z=1.4, in_contact=True) for _ in range(12)]
        )

    if name == "joint_8dof_ke_schedule":
        return (
            [_step(0.0, 1.0, raw_z=0.0) for _ in range(4)]
            + [_step(0.9, 1.0, raw_z=0.9) for _ in range(8)]
            + [_step(0.6, 1.0, raw_z=0.6, in_contact=True) for _ in range(18)]
            + [_step(1.4, 1.0, raw_z=1.4, in_contact=True) for _ in range(12)]
        )

    raise KeyError(name)


_CASES = (
    ("force_seek_acquire_tracking", "force", None),
    ("force_high_frequency", "force", None),
    ("force_loss_recontact", "force", None),
    ("force_retract", "force", None),
    ("force_path_tracking", "force", None),
    ("force_payload_override", "payload", {"desired_z": 2.5, "max_vz_tool_m_s": 0.04, "v_seek_free_m_s": 0.01}),
    ("joint_proactive_force", "joint", None),
    ("joint_8dof_ke_schedule", "joint_8dof", None),
)


def _controller(kind: str, payload: dict[str, Any] | None) -> AdmittanceController:
    # Baseline generation can run against the parent agent's read-only
    # snapshot while production files are being simplified.  Normal pytest
    # runs leave this unset and use the checked-in configs.
    config_root = Path(os.environ["ACTIVE_FORCE_CONFIG_ROOT"]) if os.environ.get("ACTIVE_FORCE_CONFIG_ROOT") else None
    force_path = (config_root / "peirastic-force.yaml") if config_root else (ROOT / "peirastic/configs/force.yaml")
    if kind == "force":
        raw = yaml.safe_load(force_path.read_text(encoding="utf-8"))
        return AdmittanceController(DT, AdmittanceConfig.from_dict(raw))
    if kind == "payload":
        ctrl, _raw, _fz = build_force_controller(
            DT,
            payload=payload,
            path=force_path,
        )
        return ctrl
    paths = {
        "joint": config_root / "joint_admittance.yaml" if config_root else ROOT / "rm75_control/configs/joint_admittance.yaml",
        "joint_8dof": config_root / "joint_admittance_8dof.yaml" if config_root else ROOT / "rm75_control/configs/joint_admittance_8dof.yaml",
    }
    raw = yaml.safe_load(paths[kind].read_text(encoding="utf-8"))
    return AdmittanceController(DT, AdmittanceConfig.from_dict(raw))


def _scalar(value: Any) -> float:
    if isinstance(value, str):
        return _STATE_CODE.get(value, float("nan"))
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if value is None:
        return float("nan")
    return float(value)


def _telemetry_value(ctrl: AdmittanceController, key: str) -> Any:
    attr = _PRIVATE_TELEMETRY.get(key, key)
    return getattr(ctrl, attr)


def _run_case(name: str, kind: str, payload: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    ctrl = _controller(kind, payload)
    commands: list[np.ndarray] = []
    motion: list[np.ndarray] = []
    telemetry: list[list[float]] = []
    for item in _scenario_steps(name):
        pose = np.asarray(item["pose"], dtype=float)
        desired_pose = np.asarray(item["desired_pose"], dtype=float)
        vel_ff = np.asarray(item["vel_ff"], dtype=float)
        f_ext = np.zeros(6, dtype=float)
        f_ext[2] = item["f_z"]
        f_des = np.zeros(6, dtype=float)
        f_des[2] = item["desired_z"]
        f_raw = None
        if item["raw_z"] is not None:
            f_raw = f_ext.copy()
            f_raw[2] = item["raw_z"]
        out = ctrl.compute_velocity_command(
            pose,
            desired_pose,
            vel_ff,
            f_ext,
            f_des,
            in_contact=item["in_contact"],
            f_ext_raw=f_raw,
            dt_actual=item["dt_actual"],
            v_tcp_z_actual=item["v_tcp_z_actual"],
            sensor_age_s=item["sensor_age_s"],
        )
        commands.append(np.asarray(out, dtype=float))
        motion.append(
            np.concatenate(
                (
                    np.asarray(ctrl.last_path_twist, dtype=float),
                    np.asarray(ctrl.last_feedback_twist, dtype=float),
                )
            )
        )
        telemetry.append([_scalar(_telemetry_value(ctrl, key)) for key in _TELEMETRY])
    return np.asarray(commands), np.column_stack((np.asarray(motion), np.asarray(telemetry)))


def _capture() -> dict[str, np.ndarray]:
    names = [name for name, _kind, _payload in _CASES]
    outputs: list[np.ndarray] = []
    offsets = [0]
    for name, kind, payload in _CASES:
        command, rest = _run_case(name, kind, payload)
        outputs.append(np.column_stack((command, rest)))
        offsets.append(offsets[-1] + len(command))
    return {
        "case_names": np.asarray(names),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "values": np.vstack(outputs),
        "telemetry_names": np.asarray(_TELEMETRY),
        "format_version": np.asarray([1], dtype=np.int64),
    }


def _load() -> dict[str, np.ndarray]:
    if not FIXTURE.exists():
        pytest.fail(
            f"missing baseline fixture {FIXTURE}; run "
            "test_active_force_regression.py --update-baseline first"
        )
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def test_active_force_controller_matches_captured_baseline() -> None:
    expected = _load()
    actual = _capture()
    assert actual["case_names"].tolist() == expected["case_names"].tolist()
    assert actual["offsets"].tolist() == expected["offsets"].tolist()
    assert actual["telemetry_names"].tolist() == expected["telemetry_names"].tolist()
    assert actual["values"].shape == expected["values"].shape
    np.testing.assert_allclose(
        actual["values"],
        expected["values"],
        rtol=2.0e-8,
        atol=5.0e-11,
        equal_nan=True,
    )


def _update_baseline() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **_capture())
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    if not args.update_baseline:
        parser.error("pass --update-baseline to write the captured fixture")
    _update_baseline()
