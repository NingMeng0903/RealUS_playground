"""Contact-gated HFPC references and phase wiring."""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as Rsc

from peirastic.configs import DEFAULT_CONTROLLER_YAML
from peirastic.realman8dof.modes.contact_reference import ContactGatedReference
from peirastic.realman8dof.modes.track import (
    HybridTffOuter,
    build_track_hybrid_phase,
)
from rm75_control.control.admittance_common.reference import MotionReference
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


class AbsoluteReference:
    """Small absolute trajectory source with an optional old-style origin hook."""

    def __init__(self, pose0: np.ndarray, *, with_time: bool = False) -> None:
        self.pose0 = np.asarray(pose0, dtype=float).copy()
        self.samples: list[float] = []
        self.origins: list[tuple[np.ndarray, float | None]] = []
        self.with_time = bool(with_time)

    def set_origin(self, pose0: np.ndarray, **kwargs) -> None:
        # ``**kwargs`` exercises the wrapper's t_s=0 compatibility path while
        # retaining the absolute world trajectory.
        self.origins.append(
            (np.asarray(pose0, dtype=float).copy(), kwargs.get("t_s"))
        )

    def sample(self, t_s: float) -> MotionReference:
        t = float(t_s)
        self.samples.append(t)
        pose = self.pose0.copy()
        pose[0] += 0.02 * t
        vel = np.zeros(6, dtype=float)
        vel[0] = 0.02
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t)


class OldStyleReference(AbsoluteReference):
    def set_origin(self, pose0: np.ndarray) -> None:
        self.origins.append((np.asarray(pose0, dtype=float).copy(), None))


def _ctx() -> CompileContext:
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_contact_gate_{os.getpid()}"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(_SEED)
    return CompileContext(
        kin=kin,
        inner=inner,
        euler_order=cfg.euler_order,
        control_frame=cfg.control_frame,
        v_scale=cfg.v_scale,
    )


def test_gate_holds_first_pose_and_anchors_nonzero_input_clock() -> None:
    pose0 = np.array([0.4, -0.2, 0.3, 0.1, -0.2, 0.3])
    source = OldStyleReference(pose0)
    present = False
    gate = ContactGatedReference(source, lambda: present)

    gate.set_origin(pose0, t_s=17.0)
    held = gate.sample(10_000.0)
    assert not gate.started
    assert gate.elapsed_s == pytest.approx(0.0)
    assert np.array_equal(held.pose_d, pose0)
    assert np.array_equal(held.vel_ff, np.zeros(6))
    assert source.origins[0][1] is None
    assert source.samples == [0.0]

    present = True
    first = gate.sample(10_001.0)
    assert gate.started
    assert gate.elapsed_s == pytest.approx(0.0)
    assert first.t_ref == pytest.approx(0.0)
    assert first.pose_d[0] == pytest.approx(pose0[0])

    second = gate.sample(10_001.25)
    assert gate.elapsed_s == pytest.approx(0.25)
    assert second.t_ref == pytest.approx(0.25)
    assert second.pose_d[0] == pytest.approx(pose0[0] + 0.005)
    frozen = gate.sample(10_001.25)
    assert gate.elapsed_s == pytest.approx(0.25)
    assert frozen.t_ref == pytest.approx(0.25)

    # A loss is not a new episode and cannot rewind the accumulated clock.
    present = False
    third = gate.sample(10_001.50)
    assert gate.started
    assert gate.elapsed_s == pytest.approx(0.50)
    assert third.t_ref == pytest.approx(0.50)


def test_set_origin_clears_gate_and_passes_zero_time_to_new_style_source() -> None:
    pose0 = np.zeros(6)
    source = AbsoluteReference(pose0)
    present = True
    gate = ContactGatedReference(source, lambda: present)
    gate.sample(4.0)
    gate.sample(4.2)
    assert gate.started and gate.elapsed_s == pytest.approx(0.2)

    pose1 = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    gate.set_origin(pose1, t_s=99.0)
    assert not gate.started
    assert gate.elapsed_s == pytest.approx(0.0)
    assert source.origins[-1][1] == pytest.approx(0.0)
    present = False
    held = gate.sample(1_000.0)
    assert held.pose_d[0] == pytest.approx(pose0[0])
    assert held.vel_ff[0] == pytest.approx(0.0)


@pytest.mark.parametrize("use_tff_split", [False, True])
def test_track_hybrid_binds_gate_to_compiled_controller(
    use_tff_split: bool,
) -> None:
    ctx = _ctx()
    pose0 = ctx.kin.fk_pose(_SEED)
    source = AbsoluteReference(pose0)
    phase = build_track_hybrid_phase(
        ctx,
        source,
        duration_s=0.50,
        dt=0.005,
        use_tff_split=use_tff_split,
        payload={
            "wait_for_contact": True,
            "desired_z": 4.0,
            "enter_confirm_s": 0.005,
            "contact_enter_n": 0.8,
            "hard_enter_n": 0.8,
        },
    )
    gate = phase.outer.contact_gate
    assert isinstance(gate, ContactGatedReference)
    assert gate.reference is source
    assert phase.duration_s == pytest.approx(0.50)
    assert phase.wait_until is not None
    assert len(inspect.signature(phase.wait_until).parameters) == 1
    assert not phase.wait_until(pose0)
    controller = phase.outer.controller
    assert controller is not None
    assert hasattr(controller, "contact_present")
    assert phase.outer.contact_gate is gate

    phase.outer.set_origin(pose0)
    # Negative normal force cannot release the trajectory gate.
    phase.outer.sample(0.0, pose0, np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0]))
    assert not gate.started
    assert gate.elapsed_s == pytest.approx(0.0)
    assert np.allclose(gate.reference.samples, [0.0])

    # The controller confirms contact on this sample; the gate observes that
    # compiled-controller state on the following sample and starts at t=0.
    phase.outer.sample(0.005, pose0, np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]))
    assert bool(controller.contact_present)
    assert not gate.started
    phase.outer.sample(0.010, pose0, np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]))
    assert gate.started
    assert gate.elapsed_s == pytest.approx(0.0)
    assert phase.wait_until(pose0) is False

    # The gate never resets or replaces the controller after contact loss.
    controller_id = id(controller)
    controller.contact_present = False
    phase.outer.sample(0.110, pose0, np.zeros(6))
    assert id(phase.outer.controller) == controller_id
    assert gate.started
    assert gate.elapsed_s == pytest.approx(0.10)


def test_scanner_force_frame_gate_and_tangent_tracking() -> None:
    ctx = _ctx()
    # The scanner uses a tool pointing down: tool +Z maps to base -Z.
    pose0 = np.array([0.45, -0.30, 0.35, np.pi, 0.0, 0.0])
    source = AbsoluteReference(pose0)
    phase = build_track_hybrid_phase(
        ctx,
        source,
        duration_s=1.0,
        dt=0.005,
        use_tff_split=True,
        payload={
            "wait_for_contact": True,
            "desired_z": 4.0,
            "max_vz_tool_m_s": 0.012,
            "v_seek_free_m_s": 0.012,
            "contact_enter_n": 0.8,
            "enter_confirm_s": 0.05,
            "force_axes": [0, 0, 1, 0, 0, 0],
            "hybrid_motion": {
                "force_barrier": {"v_seek_free_m_s": 0.012},
            },
        },
    )
    assert isinstance(phase.outer, HybridTffOuter)
    phase.outer.set_origin(pose0)
    controller = phase.outer.controller
    assert controller is not None
    assert np.allclose(
        Rsc.from_euler("xyz", pose0[3:]).as_matrix()[:, 2],
        [0.0, 0.0, -1.0],
    )
    assert np.array_equal(controller.cfg.force_axes, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    rotation = Rsc.from_euler("xyz", pose0[3:]).as_matrix()
    # XY error is expressed in the tool frame; -tool-Z is 40 mm outward.
    offset_tool = np.array([0.006, -0.004, -0.040])
    pose_start = pose0.copy()
    pose_start[:3] += rotation @ offset_tool

    f_negative = np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0])
    for i in range(20):
        phase.outer.sample(i * 0.005, pose_start, f_negative)
    # A negative Fz cannot satisfy the existing positive-force contact
    # tracker, even after many governor ticks.
    assert not phase.outer.contact_gate.started
    assert phase.outer.contact_gate.elapsed_s == pytest.approx(0.0)
    assert np.allclose(source.samples, [0.0])
    assert np.allclose(phase.outer.position.last_pose_d, pose0)
    assert np.allclose(phase.outer.position.last_vel_ff, np.zeros(6))

    f_contact = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    controller_id = id(controller)
    started_at = None
    for i in range(1, 40):
        t_s = 0.1 + i * 0.005
        phase.outer.sample(t_s, pose_start, f_contact)
        if phase.outer.contact_gate.started:
            started_at = t_s
            break
    assert started_at is not None
    assert bool(controller.contact_present)
    assert phase.outer.contact_gate.elapsed_s == pytest.approx(0.0)
    # Contact starts the absolute source at its own first pose.  The 40 mm
    # approach offset is never added to the world trajectory.
    assert np.allclose(phase.outer.position.last_pose_d, pose0)
    assert np.allclose(source.samples, [0.0])

    # Cartesian XY feedback is in tool coordinates and points back toward the
    # target.  The normal coordinate is force-owned, so it has no position
    # spring despite the 40 mm normal error.
    feedback_tool = np.asarray(phase.outer.position.last_feedback_twist, dtype=float)
    assert float(np.dot(feedback_tool[:2], -offset_tool[:2])) > 1.0e-8
    assert phase.outer.position.last_feedback_twist[2] == pytest.approx(0.0)

    # Governor freeze (repeated input time) does not advance the gate; a later
    # tick resumes from the same elapsed reference time.
    phase.outer.sample(float(started_at), pose_start, f_contact)
    assert phase.outer.contact_gate.elapsed_s == pytest.approx(0.0)

    # One second of +10 N contact should command a bounded negative tool-Z
    # retract.  The first tick also proves a frozen governor timestamp does
    # not advance the gate, while the next tick advances it by one period.
    f_over = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    for i in range(1, 201):
        phase.outer.sample(float(started_at) + i * 0.005, pose_start, f_over)
        if i == 1:
            assert phase.outer.contact_gate.elapsed_s == pytest.approx(0.005)
            assert source.samples[-1] == pytest.approx(0.005)
            assert phase.outer.position.last_pose_d[0] == pytest.approx(
                pose0[0] + 0.02 * 0.005
            )
    assert id(phase.outer.controller) == controller_id
    assert phase.outer.contact_gate.started
    assert phase.outer.contact_gate.elapsed_s == pytest.approx(1.0)
    assert controller.v_force_cmd_z < 0.0
    assert abs(controller.v_force_cmd_z) <= 0.012 + 1.0e-12


def test_wait_for_contact_rejects_force_law_without_contact_state() -> None:
    ctx = _ctx()
    pose0 = ctx.kin.fk_pose(_SEED)
    with pytest.raises(ValueError, match="contact_present"):
        build_track_hybrid_phase(
            ctx,
            AbsoluteReference(pose0),
            use_tff_split=True,
            payload={"wait_for_contact": True, "law": "fce"},
        )


def test_default_hybrid_phase_does_not_wrap_reference() -> None:
    ctx = _ctx()
    pose0 = ctx.kin.fk_pose(_SEED)
    source = AbsoluteReference(pose0)
    phase = build_track_hybrid_phase(ctx, source, payload={"desired_z": 0.0})
    assert not hasattr(phase.outer, "contact_gate")
    if hasattr(phase.outer, "reference"):
        assert phase.outer.reference is source
