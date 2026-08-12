"""Hardware-free trajectory coverage for the generic two-level QPIK path.

These tests intentionally exercise the controller through task references rather
than through an application-specific scan helper.  The kinematics fixture is a
small deterministic velocity model, so the same checks run for a seven-DOF arm
and for an eight-DOF arm whose first variable is a continuous rail coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntime,
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    PostureGuide,
    ProtectedTask,
    RobotState,
    ScalableTask,
    ReferenceHorizon,
)
from rm75_control.control.joint_admittance_8dof.reference_governor import (
    ReferenceGovernor,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.task_adapter import (
    CartesianTaskProfile,
    build_cartesian_tasks,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class _TrajectoryKinematics:
    """Six-row velocity model with an explicit redundant rail column."""

    def __init__(self, n_dof: int) -> None:
        self.nv = int(n_dof)
        self._jacobian = np.zeros((6, self.nv), dtype=float)
        if self.nv == 7:
            self._jacobian[:, :6] = np.eye(6)
            self._jacobian[:, 6] = np.array([0.11, -0.07, 0.05, 0.03, -0.02, 0.04])
        elif self.nv == 8:
            # q[0] is a prismatic rail; q[1:7] form the arm and q[7] is
            # redundant.  Rail motion contributes to one Cartesian row but
            # remains an ordinary QP variable, never a post-solve command.
            self._jacobian[:, 1:7] = np.eye(6)
            self._jacobian[0, 0] = 0.45
            self._jacobian[:, 7] = np.array([0.11, -0.07, 0.05, 0.03, -0.02, 0.04])
        else:  # pragma: no cover - the tests intentionally cover 7 and 8
            raise ValueError("fixture supports seven or eight velocity variables")

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        # Returning a fresh array models a measured-state snapshot and prevents
        # a task producer from modifying the kinematics owned by the fixture.
        del q
        return self._jacobian.copy()


def _runtime(n_dof: int) -> tuple[GenericQpikRuntime, np.ndarray]:
    rail = n_dof == 8
    v_max = np.full(n_dof, 0.8, dtype=float)
    if rail:
        v_max[0] = 0.35
    limits = SafetyLimits(
        q_lower=np.full(n_dof, -5.0),
        q_upper=np.full(n_dof, 5.0),
        v_max=v_max,
        a_max=None,
        position_margin=np.zeros(n_dof),
    )
    config = GenericQpikRuntimeConfig(
        solver=TwoLevelQpikConfig(
            backend="scipy",
            qdot_lower=-v_max,
            qdot_upper=v_max,
            max_rows=64,
            max_scalable_groups=4,
            max_iter=100,
        ),
        rail_indices=(0,) if rail else (),
        wrist_indices=tuple(range(max(0, n_dof - 3), n_dof)),
    )
    runtime = GenericQpikRuntime(
        _TrajectoryKinematics(n_dof),
        limits,
        config,
        collision_config=CollisionConfig(enabled=False),
        damper_band=0.0,
    )
    return runtime, v_max


def _state(
    q: np.ndarray,
    qdot_prev: np.ndarray,
    *,
    dt: float = 0.01,
    timestamp: float = 0.0,
    contact_active: bool = False,
) -> RobotState:
    # Keep q_cmd deliberately one small increment ahead.  P0 must remain based
    # on q_meas while the lead is only a safety input, not a second FK state.
    return RobotState(
        q_meas=q,
        q_cmd=q + 0.01,
        qdot_applied_prev=qdot_prev,
        dt=dt,
        contact_active=contact_active,
        timestamp=timestamp,
    )


def _profile() -> CartesianTaskProfile:
    # Protected rows are deliberately not a fixed tool axis: one translation
    # row and two orientation rows are protected; two independent groups share
    # the remaining Cartesian rows.
    return CartesianTaskProfile.from_indices(
        protected=(1, 4, 5),
        scalable=(
            ("position", (0, 2)),
            ("orientation", (3,)),
        ),
        name="arbitrary_frame_profile",
    )


def _assert_result(result, v_max: np.ndarray) -> None:
    qdot = np.asarray(result.qdot, dtype=float)
    assert qdot.shape == v_max.shape
    assert np.isfinite(qdot).all()
    assert np.all(np.abs(qdot) <= v_max + 2.0e-5)
    for value in result.solver.group_alphas.values():
        assert 0.0 <= float(value) <= 1.0
    # QP2 may optimize posture and scalable rows, but it cannot rewrite the
    # output selected by QP1 for protected rows.
    np.testing.assert_allclose(
        result.solver.protected_achieved,
        result.solver.protected_locked_output,
        atol=8.0e-5,
    )
    assert result.p0.C.shape[1] == qdot.size
    assert np.all(result.p0.C @ qdot >= result.p0.lower - 2.0e-5)
    assert np.all(result.p0.C @ qdot <= result.p0.upper + 2.0e-5)


def _run_twists(
    n_dof: int,
    twists: list[np.ndarray],
    *,
    profile: CartesianTaskProfile | None = None,
    rotation: np.ndarray | None = None,
    contact_active: bool = False,
) -> list:
    runtime, v_max = _runtime(n_dof)
    R = (
        Rotation.from_rotvec(np.array([0.28, -0.19, 0.37])).as_matrix()
        if rotation is None
        else rotation
    )
    active_profile = _profile() if profile is None else profile
    q = np.zeros(n_dof, dtype=float)
    qdot_prev = np.zeros(n_dof, dtype=float)
    results = []
    for tick, twist in enumerate(twists):
        state = _state(
            q,
            qdot_prev,
            timestamp=0.01 * tick,
            contact_active=contact_active,
        )
        result = runtime.solve(
            state,
            twist_task=np.asarray(twist, dtype=float),
            rotation_base_task=R,
            profile=active_profile,
        )
        _assert_result(result, v_max)
        results.append(result)
        qdot_prev = result.qdot.copy()
        runtime.sync_applied(qdot_prev)
        q = q + state.dt * qdot_prev
    return results


def test_stationary_hold_has_bounded_zero_motion_in_any_task_frame() -> None:
    twists = [np.zeros(6) for _ in range(4)]
    results = _run_twists(7, twists)
    for result in results:
        np.testing.assert_allclose(result.qdot, np.zeros(7), atol=2.0e-5)


@pytest.mark.parametrize("n_dof", (7, 8))
def test_arbitrary_direction_line_is_continuous_for_arm_and_rail(n_dof: int) -> None:
    # The line is represented only by its sampled Cartesian twists.  Nothing
    # in the fixture or controller identifies a preferred scan direction.
    direction = np.array([0.23, -0.17, 0.12], dtype=float)
    twists = [np.r_[direction, [0.0, 0.0, 0.0]] for _ in range(10)]
    results = _run_twists(n_dof, twists)
    qdots = np.vstack([r.qdot for r in results])
    assert np.max(np.abs(np.diff(qdots, axis=0))) < 0.8
    if n_dof == 8:
        # Rail is part of the whole-body solve.  It is not post-pinned to zero.
        assert np.any(np.abs(qdots[:, 0]) > 1.0e-5)


def test_circle_and_three_dimensional_spline_like_references() -> None:
    angles = np.linspace(0.0, np.pi / 2.0, 8)
    circle = [
        np.r_[-0.16 * np.sin(a), 0.16 * np.cos(a), 0.04, 0.0, 0.0, 0.0]
        for a in angles
    ]
    spline = []
    for u in np.linspace(0.0, 1.0, 8):
        # Derivative of a bounded cubic curve in three independent directions.
        d = np.array(
            [0.12 * (1.0 - 2.0 * u), -0.10 + 0.20 * u, 0.08 * (1.0 - u) ** 2],
            dtype=float,
        )
        spline.append(np.r_[d, 0.04 * u, -0.03 * u, 0.02 * (1.0 - u)])
    results = _run_twists(8, circle + spline)
    assert len(results) == 16
    assert all(result.solver.qp1.success for result in results)
    assert all(result.solver.qp2.success for result in results)


def test_simultaneous_position_orientation_and_multiple_scalable_groups() -> None:
    profile = CartesianTaskProfile.from_indices(
        protected=(2, 4),
        scalable=(("position", (0, 1)), ("orientation", (3, 5))),
        name="mixed_pose",
    )
    twists = [
        np.array([0.12, -0.08, 0.03, 0.11, -0.07, 0.05], dtype=float),
        np.array([-0.05, 0.10, 0.02, -0.09, 0.04, -0.03], dtype=float),
    ]
    results = _run_twists(7, twists, profile=profile)
    for result in results:
        assert set(result.solver.group_alphas) == {"position", "orientation"}
        assert result.protected.n_rows == 2


def test_streaming_twist_without_future_horizon_does_not_accumulate_rejected_error() -> None:
    # A streaming caller supplies only the current twist.  A brief pause and a
    # reversal are explicit new commands; the controller has no hidden path or
    # period to extrapolate.
    twists = [
        np.array([0.18, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(6),
        np.zeros(6),
        np.array([-0.18, 0.04, 0.0, 0.0, 0.0, 0.0]),
        np.array([-0.18, 0.04, 0.0, 0.0, 0.0, 0.0]),
    ]
    results = _run_twists(8, twists, contact_active=True)
    assert np.linalg.norm(results[1].qdot) < 0.3
    assert np.linalg.norm(results[2].qdot) < 0.3
    assert np.all(np.isfinite(results[-1].qdot))


@dataclass(frozen=True)
class _TwistHorizon:
    values: tuple[np.ndarray, ...]

    def sample(
        self,
        t_s: float,
        horizon_s: float = 0.0,
        state: RobotState | None = None,
    ) -> tuple[np.ndarray, ...]:
        del t_s, horizon_s, state
        return tuple(value.copy() for value in self.values)


def test_reference_horizon_replacement_pause_and_reverse_are_finite() -> None:
    # Validate the generic horizon protocol and feed each replacement into the
    # same QPIK instance.  Replacing a horizon must not leave an old target
    # queued as a large catch-up command.
    first = _TwistHorizon(
        tuple(np.array([0.1, 0.02, 0.0, 0.0, 0.0, 0.0]) for _ in range(3))
    )
    second = _TwistHorizon(
        tuple(np.array([-0.08, -0.03, 0.01, 0.0, 0.0, 0.0]) for _ in range(3))
    )
    assert isinstance(first, ReferenceHorizon)
    assert len(first.sample(0.0, 0.03)) == 3
    runtime, v_max = _runtime(7)
    R = Rotation.from_euler("zyx", [0.3, -0.2, 0.5]).as_matrix()
    q = np.zeros(7)
    qdot_prev = np.zeros(7)
    all_results = []
    for horizon in (first, second):
        for twist in horizon.sample(0.0, 0.03):
            result = runtime.solve(
                _state(q, qdot_prev, timestamp=float(len(all_results)) * 0.01),
                twist_task=twist,
                rotation_base_task=R,
                profile=_profile(),
            )
            _assert_result(result, v_max)
            all_results.append(result)
            qdot_prev = result.qdot.copy()
            runtime.sync_applied(qdot_prev)
            q += 0.01 * qdot_prev
    # The first sample after replacement is bounded by the same velocity box;
    # no accumulated first-horizon error can produce a jump.
    assert np.max(np.abs(all_results[3].qdot - all_results[2].qdot)) < 0.8


def test_reference_governor_scales_absolute_pause_reverse_and_streaming_inputs() -> None:
    governor = ReferenceGovernor(["motion"], tau_s=0.0, residual_ok=0.05, residual_max=0.2)
    healthy = governor.update(
        0.01,
        absolute_reference=np.array([1.0, -0.5]),
        current=np.zeros(2),
        residuals={"motion": 0.0},
    )
    assert 0.0 <= healthy.alphas["motion"] <= 1.0
    paused = governor.update(
        0.01,
        streaming_twist=np.zeros(2),
        residuals={"motion": 0.0},
    )
    assert np.allclose(paused.twist, np.zeros(2))
    reversed_ref = governor.update(
        0.01,
        absolute_reference=np.array([-1.0, 0.5]),
        current=np.array([0.2, -0.1]),
        residuals={"motion": 0.0},
    )
    assert np.isfinite(np.asarray(reversed_ref.reference, dtype=float)).all()


def test_eight_dof_rail_is_a_continuous_qpik_variable() -> None:
    runtime, v_max = _runtime(8)
    protected = ProtectedTask(
        np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        np.array([0.04]),
        name="arm_protected",
    )
    rail = ScalableTask(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        np.array([0.12]),
        "rail",
        name="continuous_redundancy",
    )
    guide = PostureGuide(
        q_goal=np.zeros(8),
        qdot_guide=np.array([0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        valid_until=1.0,
        quality=1.0,
        planner_state="local",
    )
    result = runtime.solve_tasks(
        _state(np.zeros(8), np.zeros(8)),
        protected=protected,
        scalable=(rail,),
        posture_guide=guide,
    )
    _assert_result(result, v_max)
    assert abs(result.qdot[0]) > 1.0e-3
    assert result.solver.group_alphas["rail"] >= 0.0

