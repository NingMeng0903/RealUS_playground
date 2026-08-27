"""Free-running acceptance for the 232136 / 232307 failure modes."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONFIG = _ROOT / "configs" / "joint_admittance_8dof.yaml"
Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)
_LOG_CANDIDATES = (
    _ROOT / "apps" / "logs" / "gamepad_vcmd",
    Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/apps/logs/gamepad_vcmd"),
)


def _controller() -> JointIkController:
    collision = CollisionConfig(enabled=False)
    qp = QpConfig(
        backend="proxqp",
        collision=collision,
        smoothness_weight=np.r_[0.0, np.full(7, 0.15)],
    )
    cfg = JointIkConfig(control_frame="base", qp=qp, collision=collision)
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


def _find_log(name: str) -> Path | None:
    for folder in _LOG_CANDIDATES:
        path = folder / name
        if path.is_file():
            return path
    return None


def test_free_running_pure_minus_z_has_near_zero_residual() -> None:
    """232136 analogue: a feasible -Z twist must not be bought as slack."""
    controller = _controller()
    twist = np.array([0.0, 0.0, -0.12, 0.0, 0.0, 0.0])
    residuals = []
    for _ in range(200):
        step = controller.update(twist, q_meas=controller.q_cmd)
        residuals.append(float(np.max(np.abs(step.protected_residual))))
        assert step.fallback_level == "none"
        assert not step.solver_fault_latched
    # Accel/jerk boxes need time to open from rest; the cruise window is the
    # feasibility verdict, not the first 50 ms of the ramp.
    assert max(residuals[-40:]) <= 1.0e-4


def test_release_does_not_raise_or_reverse_rail_command() -> None:
    """232307 analogue: v_cmd=0 must brake, not coast faster or pull back."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config

    cfg = build_joint_ik_config(raw)
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    # This test exercises release braking, not the production 5 ms native
    # timing budget (which is covered by the dedicated overrun tests).
    cfg.qp.max_solve_ms = 50.0
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    plus_y = np.array([0.0, 0.12, 0.0, 0.0, 0.0, 0.0])
    cruise = np.zeros(8)
    cruise[0] = 0.08
    controller.core.sync_applied(cruise)
    moving = None
    for _ in range(80):
        moving = controller.update(plus_y, q_meas=controller.q_cmd, vel_ff=plus_y)
    assert moving is not None
    # Allocator + mid-ranging share +Y with the arm; the rail may take
    # only a few mm/s.  The contract under test is the release brake.
    assert float(moving.qdot[0]) > -0.02

    zero = np.zeros(6)
    rail_cmds = []
    v_reach = []
    for _ in range(250):
        step = controller.update(zero, q_meas=controller.q_cmd, vel_ff=zero)
        rail_cmds.append(float(step.qdot[0]))
        v_reach.append(float(step.v_reach))
        if step.fallback_level != "none":
            assert step.fallback_reason in {
                "box_infeasible",
                "final_certificate",
                "qp_failed",
            }
            assert np.allclose(step.qdot, 0.0)
            break

    rails = np.asarray(rail_cmds, dtype=float)
    assert float(np.max(np.abs(rails))) <= max(abs(float(rails[0])), 0.05) + 1.0e-4
    assert max(abs(v) for v in v_reach) <= 0.05 + 1.0e-6
    stopped = np.flatnonzero(np.abs(rails) <= 0.05 + 1.0e-6)
    if stopped.size:
        assert float(np.max(np.abs(rails[int(stopped[0]) :]))) <= 0.05 + 1.0e-6
    assert abs(float(rails[-1])) < 0.05


@pytest.mark.parametrize(
    "log_name,kind",
    [
        ("run_20260816_232136.csv", "minus_z"),
        ("run_20260816_232307.csv", "release"),
    ],
)
def test_logged_runs_free_running_acceptance(
    log_name: str, kind: str, tmp_path: Path
) -> None:
    path = _find_log(log_name)
    if path is None:
        pytest.skip(f"hardware log {log_name} is not in the workspace")

    from apps.joint_admittance_8dof.replay_strict_qpik import replay_csv

    if kind == "minus_z":
        with path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if not source_rows:
            pytest.skip("empty log")
        fields = list(source_rows[0].keys())

        def _finite(row: dict, name: str) -> float:
            try:
                return float(row.get(name) or "nan")
            except (TypeError, ValueError):
                return float("nan")

        mask = [
            _finite(row, "v_cmd_vz") < -0.08
            and abs(_finite(row, "v_cmd_vx")) < 0.01
            and abs(_finite(row, "v_cmd_vy")) < 0.01
            for row in source_rows
        ]
        if not any(mask):
            pytest.skip("log has no pure -Z segment")
        start = next(i for i, keep in enumerate(mask) if keep)
        end = start
        while end + 1 < len(mask) and mask[end + 1]:
            end += 1
        segment = source_rows[start : end + 1]
        if len(segment) < 120:
            pytest.skip("pure -Z window is shorter than the ramp")
        sliced = tmp_path / "minus_z_slice.csv"
        with sliced.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(segment)
        result = replay_csv(
            sliced, CONFIG, disable_cbf=True, mode="free-running", backend="python"
        )
        rows = result["rows"]
        assert result["summary"]["replay_mode"] == "free-running"
        residuals = np.array(
            [float(row["tcp_residual_inf_m_s"]) for row in rows[200:-20]]
        )
        if residuals.size < 20:
            residuals = np.array(
                [float(row["tcp_residual_inf_m_s"]) for row in rows[80:]]
            )
        assert float(np.nanpercentile(residuals, 95)) <= 2.0e-3
        assert float(np.nanmedian(residuals)) <= 1.0e-3
        return

    result = replay_csv(
        path, CONFIG, disable_cbf=True, mode="free-running", backend="python"
    )
    rows = result["rows"]
    assert rows
    assert result["summary"]["replay_mode"] == "free-running"

    seen_motion = False
    rail_after: list[float] = []
    v_reach_after: list[float] = []
    for row in rows:
        v_cmd = np.array(
            [float(row[f"v_cmd_{axis}"]) for axis in ("vx", "vy", "vz", "wx", "wy", "wz")]
        )
        moving = float(np.linalg.norm(v_cmd)) > 0.02
        if moving:
            seen_motion = True
            rail_after = []
            v_reach_after = []
            continue
        if seen_motion:
            rail_after.append(float(row["rail_command_m_s"]))
            v_reach_after.append(float(row["rail_v_reach_m_s"]))
    if len(rail_after) < 20:
        pytest.skip("log has no v_cmd=0 release segment after motion")
    rails = np.asarray(rail_after, dtype=float)
    assert float(np.max(np.abs(rails))) <= max(abs(float(rails[0])), 0.05) + 1.0e-4
    assert max(abs(v) for v in v_reach_after) <= 0.05 + 1.0e-6
    stopped = np.flatnonzero(np.abs(rails) <= 0.05 + 1.0e-6)
    if stopped.size:
        assert float(np.max(np.abs(rails[int(stopped[0]) :]))) <= 0.05 + 1.0e-6
