from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.generic_tasks import RobotState
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.posture_planner import PosturePlanningRequest
from rm75_control.control.joint_admittance_8dof.rm75_srs_planner import (
    Rm75SrsPlannerConfig,
    Rm75SrsPosturePlanner,
)


def test_no_horizon_uses_measured_pose_and_publishes_only_a_posture_guide() -> None:
    kin = RobotKinematics()
    q = full_q_from_arm(np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40]), rail_m=0.30)
    pose = kin.fk_pose(q)
    planner = Rm75SrsPosturePlanner(
        kinematics=kin,
        adapter_config=Rm75SrsPlannerConfig(
            rail_candidates=(0.30,),
            psi_candidates=(0.0,),
            branch_candidates=(0, 4, 2, 6),
            winding_offsets=(0,),
        ),
    )
    state = RobotState(q, q, np.zeros(8), 0.005, False, 1.0)
    try:
        guide = planner.plan_now(PosturePlanningRequest(state, {"pose_meas": pose}))
        np.testing.assert_allclose(guide.q_goal, q, atol=1e-9)
        np.testing.assert_allclose(guide.qdot_guide, np.zeros(8), atol=1e-9)
        assert guide.source == "Rm75SrsPosturePlanner"
        assert guide.metadata["branch"] == 0
    finally:
        planner.shutdown()


def test_horizon_samples_are_arbitrary_and_contact_keeps_discrete_state(monkeypatch) -> None:
    """The adapter calls SRS IK per candidate/sample, without scan assumptions."""

    import rm75_control.control.joint_admittance_8dof.rm75_srs_planner as adapter

    calls: list[tuple[float, float, int, float]] = []

    class FakeKin:
        q_lower = np.array([0.0] + [-4.0] * 7)
        q_upper = np.array([0.8] + [4.0] * 7)
        _R_link7_tcp = np.eye(3)
        _r_link7_tcp = np.array([0.0, 0.0, 0.22])

        def fk_pose(self, q):
            return np.zeros(6)

        def jacobian(self, q):
            return np.hstack((np.zeros((6, 1)), np.eye(6, 7)))

    def fake_psi(q):
        return 0.2

    def fake_branch(q):
        return 0

    def fake_srs(pose, psi, branch, y_rail=0.0, **kwargs):
        calls.append((float(pose[0]), float(psi), int(branch), float(y_rail)))
        # A valid, deterministic arm sample.  The adapter still performs the
        # graph's intermediate edge and velocity/acceleration checks.
        return np.array([0.1 + 0.1 * float(psi), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    monkeypatch.setattr(adapter, "psi_from_q", fake_psi)
    monkeypatch.setattr(adapter, "branch_from_q", fake_branch)
    monkeypatch.setattr(adapter, "srs_ik", fake_srs)

    q = np.array([0.20, 0.12, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    planner = Rm75SrsPosturePlanner(
        kinematics=FakeKin(),
        adapter_config=Rm75SrsPlannerConfig(
            rail_candidates=(0.20,),
            psi_candidates=(0.2, 0.3),
            branch_candidates=(0, 1),
            winding_offsets=(-1, 0, 1),
            horizon_dt_s=0.1,
        ),
    )
    state = RobotState(q, q, np.zeros(8), 0.005, True, 1.0)
    horizon = [{"pose": np.array([0.1, 0, 0, 0, 0, 0])}, {"pose": np.array([0.2, 0, 0, 0, 0, 0])}]
    try:
        guide = planner.plan_now(
            PosturePlanningRequest(state, {"pose_meas": np.zeros(6)}, horizon)
        )
        assert guide.metadata["branch"] == 0
        assert guide.metadata["winding"] == 0
        # Both arbitrary horizon poses and more than one rail/psi/branch
        # candidate were passed through the injected SRS IK function.
        assert {round(c[0], 3) for c in calls} >= {0.1, 0.2}
        assert len(calls) > 4
    finally:
        planner.shutdown()


def test_shadow_submit_does_not_require_command_side_effects() -> None:
    kin = RobotKinematics()
    q = full_q_from_arm(np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40]), rail_m=0.30)
    pose = kin.fk_pose(q)
    planner = Rm75SrsPosturePlanner(
        kinematics=kin,
        adapter_config=Rm75SrsPlannerConfig(
            rail_candidates=(0.30,),
            psi_candidates=(0.0,),
            branch_candidates=(0, 4, 2, 6),
            winding_offsets=(0,),
        ),
    )
    state = RobotState(q, q, np.zeros(8), 0.005, False, 1.0)
    try:
        seq = planner.submit(
            robot_state=state,
            current_task_reference={"pose_meas": pose},
            timestamp_s=1.0,
        )
        assert planner.wait_for(seq, timeout_s=2.0)
        snapshot = planner.latest(now_s=1.0)
        assert snapshot.value is not None
        np.testing.assert_allclose(snapshot.value.q_goal, q, atol=1e-9)
    finally:
        planner.shutdown()


def test_committed_corridor_is_sampled_at_servo_rate_without_hold_drift() -> None:
    kin = RobotKinematics()
    q = full_q_from_arm(
        np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40]),
        rail_m=0.30,
    )
    pose = kin.fk_pose(q)
    planner = Rm75SrsPosturePlanner(
        kinematics=kin,
        adapter_config=Rm75SrsPlannerConfig(
            rail_candidates=(0.30,),
            psi_candidates=(0.0,),
            branch_candidates=(0, 4, 2, 6),
            winding_offsets=(0,),
        ),
    )
    state = RobotState(q, q, np.zeros(8), 0.005, False, 1.0)
    try:
        planner.plan_now(PosturePlanningRequest(state, {"pose_meas": pose}))
        first = planner.sample_guide(
            robot_state=state,
            current_task_reference={"pose_meas": pose},
            now_s=1.0,
        )
        second = planner.sample_guide(
            robot_state=RobotState(q, q, np.zeros(8), 0.005, False, 1.005),
            current_task_reference={"pose_meas": pose},
            now_s=1.005,
        )
        assert first is not None and second is not None
        np.testing.assert_allclose(first.q_goal, q, atol=1e-12)
        np.testing.assert_allclose(second.q_goal, q, atol=1e-12)
        np.testing.assert_allclose(second.qdot_guide, np.zeros(8), atol=0.0)
    finally:
        planner.shutdown()
