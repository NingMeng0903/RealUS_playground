"""Real IK startup smoke for the new pad chain, using recorded robot poses.

The transport is simulated. No robot, rail, controller IPC or image sockets
are opened; successful commands feed an idealized next kinematic state.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import yaml

from peirastic.configs import DEFAULT_CONTROLLER_YAML
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.session import CompileContext, compile_request
from peirastic.realman8dof.modes.pad_visual import pad_hybrid_payload
from rm75_control.control.joint_admittance_8dof.command_lifecycle import CommandLifecycleTransaction
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


CASES = json.loads((Path(__file__).parent / 'data/pad_install_20260921.json').read_text())


@pytest.mark.parametrize('backend', ['python', 'native'])
@pytest.mark.parametrize('scenario', ['seek', 'contact_repair'])
@pytest.mark.parametrize('case', CASES, ids=lambda row: row['source'])
def test_new_scan_pad_can_publish_from_recorded_startup_pose(case, backend, scenario):
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = backend
    cfg.ird.enabled = False
    cfg.native_shm_prefix = 'pad20_startup_test_' + str(os.getpid())
    kin = RobotKinematics()
    kin.apply_link7_to_tcp_offset(case['tcp_offset_pose'])
    inner = JointIkController(kin, cfg)
    phase = None
    try:
        if inner._native:
            inner._native.timeout_s = .5
        q = np.asarray(case['q_meas'], dtype=float)
        inner.reset(q)
        ctx = CompileContext(kin=kin, inner=inner, euler_order=cfg.euler_order,
                             control_frame=cfg.control_frame, v_scale=cfg.v_scale)
        stick = np.zeros(6)
        if scenario == 'contact_repair':
            stick[1] = .02
        phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, pad_hybrid_payload('ultrapoc')),
                                raw=raw, twist_read=lambda: stick)
        outer = phase.outer
        outer.pad_visual._receiver = SimpleNamespace(observation=None)
        outer.pad_visual._owns_receiver = False
        lifecycle = getattr(outer, 'command_lifecycle', None)
        if lifecycle is None:
            from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config
            from peirastic.realman8dof.modes.pad_scan_lifecycle import PadScanLifecycle
            lifecycle = PadScanLifecycle(outer, load_pad_scan_config())
            outer.command_lifecycle = lifecycle
        if scenario == 'contact_repair':
            # Isolate real IK/allocation/publication here; raw-image direction
            # confirmation is exercised separately by the evidence suite.
            guide = outer.pad_visual
            def confirmed_image_request(**kwargs):
                guide.evidence_valid = True
                guide.last_alpha = .5
                guide.omega_target = .05 if index < 6 else -.05
                guide.last_visual_omega = guide.omega_target
                guide.last_facts = dict(visual_request_rad_s=guide.omega_target,
                                       visual_reason='unilateral_missing_echo')
            guide.preview = confirmed_image_request
        clock = [time.monotonic()]
        lifecycle.clock = lambda: clock[0]
        outer.set_origin(kin.fk_pose(q))
        transaction = CommandLifecycleTransaction(lifecycle)
        rail = 0.
        guarded_conflict = scenario == 'guarded_conflicting_task'
        intervals = [.005] if guarded_conflict else [.005, .007, .004, .006] * 3
        for index, dt in enumerate(intervals):
            clock[0] += dt
            pose = kin.fk_pose(q)
            transaction.begin_candidate()
            contacting = scenario == 'contact_repair'
            fz, my = (4., .04) if contacting else (.4, -.1)
            command = outer.sample(clock[0], pose, [0., 0., fz, 0., my, 0.], contact=contacting,
                                   dt_actual=dt, sensor_age_s=.001, feedback_age_s=.001,
                                   feedback_fresh_tick=True, v_tcp_z_actual=0.)
            assert abs(command[2]) <= .0149+1e-8
            constraints = lifecycle.command_constraints()
            ik_command = command.copy()
            if guarded_conflict:
                assert lifecycle._pending['guarded']
                assert lifecycle._pending['nominal'][4] == 0.
                assert constraints['command_twist_lower'][1] == 0.
                assert constraints['command_twist_upper'][1] == 0.
                # Structural reproduction of the admission mismatch, not a
                # captured user failure pose: a soft IK objective asks for
                # forbidden rotation, and only rocking tier four admits zero.
                ik_command[4] = .009
                constraints['rocking_bounds'] = np.tile([.008, .010], 3)
            def solve_inner(admission):
                return inner.update(ik_command, .005, q_meas=q, vel_ff=outer.last_vel_ff,
                    pose_d=outer.last_pose_d, f_ext_z=fz, f_des_z=4., contact_active=contacting,
                    path_twist=outer.last_path_twist, feedback_twist=outer.last_feedback_twist,
                    v_force_z=outer.controller.v_force_z, rail_exec_vel_m_s=rail,
                    dt_wall_s=dt, auto_commit=False, commit_history=False, **admission)
            if guarded_conflict:
                old_admission = dict(constraints,
                    command_twist_lower=np.array([-.015, -.28]),
                    command_twist_upper=np.array([.015, .28]))
                unsafe = solve_inner(old_admission)
                assert not unsafe.fallback_reason
                assert not transaction.review(unsafe, published_rail_m=float(unsafe.q_send[0]),
                                              dt_s=.005, now_s=clock[0])
                assert lifecycle.last_review_reason == 'final_mechanical_guard'
                assert abs(transaction.final_tool[4]) > lifecycle.mechanical_guard_tolerance_rad_s
                inner.reset(q)
            step = solve_inner(constraints)
            assert not step.fallback_reason, (index, step.fallback_reason)
            if guarded_conflict:
                assert step.rocking_policy_tier == 4
            clock[0] += .001
            assert transaction.review(step, published_rail_m=float(step.q_send[0]),
                                      dt_s=.005, now_s=clock[0]), (index, lifecycle.last_review_reason,
                                      command.tolist(), transaction.final_tool.tolist())
            assert transaction.started()
            assert transaction.arm_dispatch(now_s=clock[0])
            final = transaction.final_tool.copy()
            if guarded_conflict:
                assert abs(final[4]) <= lifecycle.mechanical_guard_tolerance_rad_s
            inner.commit_publication(step.qdot)
            transaction.commit(now_s=clock[0]+.0001, rail_enabled=True)
            np.testing.assert_allclose(lifecycle._last_final, final)
            # Numerical feasibility tolerance; the configured physical caps
            # remain 15 mm/s and 0.28 rad/s.
            assert abs(final[2]) <= .015+1e-6 and abs(final[4]) <= .28+1e-5
            if not contacting:
                assert not np.any(outer.last_path_twist)
            else:
                assert np.linalg.norm(outer.last_path_twist[:2]) <= .01+1e-8
            assert lifecycle._pending is None
            q = step.q_send.copy()
            rail = .3*step.qdot[0]+.7*rail
    finally:
        if phase is not None and phase.on_exit is not None:
            phase.on_exit()
        if inner._native:
            inner._native.shutdown()


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_guarded_zero_is_enforced_inside_ik_even_at_rocking_tier_four(backend):
    test_new_scan_pad_can_publish_from_recorded_startup_pose(
        CASES[0], backend, 'guarded_conflicting_task')
