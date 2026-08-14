"""IRD field as QPIK rail inverse-reachability (optional; needs the checkpoint)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.ird_adapter import (
    IrdConfig,
    try_load_ird,
)
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
)

_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


@pytest.fixture(scope="module")
def ird_handle():
    handle = try_load_ird(IrdConfig(enabled=True, device="cpu", allow_stale=True))
    if handle is None:
        pytest.skip("IRD checkpoint or ird_playground not available")
    return handle


def test_ird_scores_current_pose(ird_handle) -> None:
    kin = RobotKinematics()
    g = ird_handle.g(kin, _SEED_Q)
    assert np.isfinite(g)


def test_ird_rail_gradient_is_finite(ird_handle) -> None:
    kin = RobotKinematics()
    dg = ird_handle.dg_dy_rail(kin, _SEED_Q)
    assert np.isfinite(dg)


def test_ird_offset_query_picks_a_feasible_d(ird_handle) -> None:
    kin = RobotKinematics()
    y0 = float(kin.fk_placement(_SEED_Q).translation[1])
    y = np.linspace(y0 - 0.05, y0 + 0.05, 5)
    d_grid = np.linspace(-0.15, 0.15, 7)
    T = ird_handle.tcp_ird_from_q(kin, _SEED_Q)
    d_star = ird_handle.query_d_star(
        T,
        y_tcp0_m=y0,
        y_samples_m=y,
        d_samples_m=d_grid,
        rail_lo=0.02,
        rail_hi=0.76,
    )
    assert d_star is not None
    rails = y - float(d_star)
    assert np.all(rails >= 0.02 - 1e-9)
    assert np.all(rails <= 0.76 + 1e-9)


def test_plan_stroke_can_use_ird_without_replanning(ird_handle) -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin, PsiRetargetConfig(enabled=True, n_y=3, n_d=5, n_psi=5)
    )
    rt._ird = ird_handle
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    d0, _psi0 = rt.plan_stroke(
        _SEED_Q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.005, rail_hi=0.78
    )
    q2 = _SEED_Q.copy()
    q2[2] -= 0.2
    _psi1, d1 = rt.step(q2, 0.005, rail_lo=0.005, rail_hi=0.78)
    assert d1 == pytest.approx(d0, abs=1e-9)


def test_hot_path_goodness_stays_sigma_min_even_if_ird_loads() -> None:
    from rm75_control.control.joint_admittance_8dof.collision_model import (
        CollisionConfig,
    )
    from rm75_control.control.joint_admittance_8dof.loop import (
        JointIkConfig,
        JointIkController,
    )
    from rm75_control.control.joint_admittance_8dof.tasks.ird_adapter import (
        IrdRailGoodness,
    )
    from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
        CachedRailGoodness,
        SigmaMinGoodness,
    )

    kin = RobotKinematics()
    inner = JointIkController(
        kin,
        JointIkConfig(
            collision=CollisionConfig(enabled=False),
            ird=IrdConfig(enabled=True),
            psi_retarget=PsiRetargetConfig(enabled=False),
        ),
    )
    assert isinstance(inner._rail_goodness, CachedRailGoodness)
    assert isinstance(inner._rail_goodness.inner, SigmaMinGoodness)
    assert not isinstance(inner._rail_goodness.inner, IrdRailGoodness)


def test_default_ird_config_is_off_until_yaml_enables_it() -> None:
    assert IrdConfig().enabled is False
    assert Path(IrdConfig().checkpoint).name == "selected.pt"
