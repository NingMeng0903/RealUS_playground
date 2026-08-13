"""Cold-start regression for the longest 20260812 QPIK fallback regions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.generic_tasks import RobotState
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


SNAPSHOTS = (
    (
        np.array([0.569617, 1.408690, -1.198657, -2.124432, 1.970023, -0.767631, -0.096482, 0.983580]),
        np.array([0.569926, 1.368757, -1.136545, -2.066067, 1.994812, -0.801277, -0.051125, 1.058493]),
        np.array([0.0015351593, -0.0367668364, 0.0492158771, 0.0407801218, 0.0157581660, -0.1736972025, 0.0293456963, 0.1076268281]),
        np.array([-0.0800000000, -0.0013388132, -0.0066001638, 0.0085657782]),
        np.array([-0.0018427324, -0.0007478157]),
    ),
    (
        np.array([0.365299, 0.266355, -0.407307, -0.278642, 2.296871, -0.254294, 0.231483, 1.725310]),
        np.array([0.367012, 0.321352, -0.373274, -0.332006, 2.321093, -0.338320, 0.177248, 1.808686]),
        np.array([0.0010322598, 0.1685187155, 0.0970278920, -0.1537809977, -0.0000053347, -0.0751795322, -0.0779430503, 0.1000008253]),
        np.array([0.0465479657, -0.0021906908, -0.0059444232, 0.0122892989]),
        np.array([-0.0025580266, 0.0045983532]),
    ),
)


@pytest.mark.parametrize(("q_meas", "q_cmd", "qdot_prev", "protected", "scan"), SNAPSHOTS)
def test_old_max_iter_snapshot_is_single_call_hard_valid_fallback(
    q_meas: np.ndarray,
    q_cmd: np.ndarray,
    qdot_prev: np.ndarray,
    protected: np.ndarray,
    scan: np.ndarray,
) -> None:
    """The old CSV lacks path/feedback split; use the conservative beta channel."""

    raw = yaml.safe_load(CONFIG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.collision.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.core.solver.backend.cold_start()
    inner.core.sync_applied(qdot_prev)
    state = RobotState(
        q_meas=q_meas,
        q_cmd=q_cmd,
        qdot_applied_prev=qdot_prev,
        dt=cfg.dt,
        contact_active=True,
    )
    protected_twist = np.zeros(6)
    protected_twist[[2, 3, 4, 5]] = protected
    feedback_twist = np.zeros(6)
    feedback_twist[:2] = scan

    result = inner.core.solve(
        state,
        protected_twist_task=protected_twist,
        path_twist_task=np.zeros(6),
        feedback_twist_task=feedback_twist,
        rotation_base_task=kin.fk_placement(q_meas).rotation,
        resync_err=np.array([cfg.resync_err_rail_m] + [cfg.resync_err_rad] * 7),
        jacobian_base=kin.jacobian(q_meas),
    )

    assert result.solver.diagnostics.call_count == 1
    assert not result.solver.diagnostics.success
    assert result.solver.fallback
    assert result.solver.anchor_valid
    assert not result.solver.hard_failure
    assert result.solver.authority == 0.0
    assert result.solver.hard_residual_max <= cfg.generic_qpik.solver.feasibility_tolerance
    assert np.any(result.solver.protected_nominal_overflow > 0.0)
