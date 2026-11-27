"""Independent closed-loop gallbladder skill pilot; legacy sources are read-only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'simulator' / 'active_observation_benchmark'
sys.path.insert(0, str(OLD.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(OLD / 'gallbladder_diagnostics'))
import psdp_scene as legacy_scene
import psdp_control as legacy_control
from psdp_scene import Scene
from psdp_control import Controller
from gallbladder_prescreen import generate, initialize as initialize_reference
from gallbladder_diagnostics.liver_guided_collection import sweep_motion
from servo import ACCEL, FRAME_STEPS, Motion
import gallbladder_geometry as legacy_geometry
from landmark_repair import install as install_landmark_repair

LANDMARK_AUDIT = None


def initialize():
    from gallbladder_runtime import ANATOMY_ARCHIVE, ATLAS_ARCHIVE, SMPLX_ARCHIVE

    global LANDMARK_AUDIT
    for variable, path in (('REALUS_GB_ANATOMY', ANATOMY_ARCHIVE),
                           ('REALUS_GB_SMPLX', SMPLX_ARCHIVE),
                           ('REALUS_GB_ATLAS', ATLAS_ARCHIVE)):
        if not path.is_file():
            raise FileNotFoundError(f'Set {variable} to the simulator input file. Expected {path}.')
    if LANDMARK_AUDIT is None:
        LANDMARK_AUDIT = install_landmark_repair(legacy_geometry)
    initialize_reference()


MAX_TICKS = 260
CENTER_X_MM = 12.0
MIN_SPAN_FRACTION = .55
MAX_COLLINEAR_ERROR_MM = 12.0
MAX_REPEAT_TICKS = 6
MAX_REPEAT_MARK_DELTA_MM = 16.0


def finite(x):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(x).ravel()]


def public_confirm(state, scene):
    """Image-only entry rule shared by both policies."""
    _, _, expected_length = scene.estimate(state)
    for full in reversed(state.captures):
        if state.tick - full.tick > MAX_REPEAT_TICKS:
            break
        if np.max(np.abs(state.v)) > .8 or state.tick - full.tick < 4:
            continue
        marks = full.marks
        if not np.isfinite(marks).all():
            continue
        axis = marks[0] - marks[2]
        span = float(np.linalg.norm(axis))
        if span < MIN_SPAN_FRACTION * expected_length:
            continue
        if not np.isfinite(full.reading[0,1]) or abs(float(full.reading[0,1])) > CENTER_X_MM:
            continue
        frac = float((marks[1] - marks[2]) @ axis / max(span**2, 1.e-9))
        off = float(np.linalg.norm(marks[1] - marks[2] - frac * axis))
        if not (-.08 <= frac <= 1.08 and off <= MAX_COLLINEAR_ERROR_MM):
            continue
        for adjacent in state.recent:
            if full.tick == adjacent.tick or np.max(np.abs(full.q-adjacent.q)) > 4.5:
                continue
            visible = np.isfinite(adjacent.marks[:, 0])
            if visible.sum() < 2:
                continue
            if np.max(np.linalg.norm(full.marks[visible]-adjacent.marks[visible], axis=1)) <= MAX_REPEAT_MARK_DELTA_MM:
                if np.max(np.abs(full.q-state.q)) > 4.5:
                    continue
                return full
    return None


def public_single(state, scene):
    """A plausible single-frame image that merits braking and confirmation dwell."""
    _, _, length = scene.estimate(state)
    for packet in reversed(state.captures):
        if state.tick-packet.tick > MAX_REPEAT_TICKS:
            break
        marks = packet.marks
        if (not np.isfinite(marks).all() or not np.isfinite(packet.reading[0,1])
                or abs(float(packet.reading[0,1])) > CENTER_X_MM):
            continue
        if packet.reading[0,0] < 75.:
            continue
        span = float(np.linalg.norm(marks[0]-marks[2]))
        if span >= MIN_SPAN_FRACTION*length:
            return packet
    return None


def configure(seed, condition, stream):
    geom = generate(seed)
    scene = Scene(seed, condition, geometry=geom)
    # Test the original sampled contact rather than the legacy forced subcostal start.
    scene.start = scene.case.start.copy()
    ctrl = Controller(scene)
    tape = scene.tape(stream)
    raw_observe = scene.observe
    scene._offline_observe = raw_observe
    scene._evaluation_tape = tape

    def observe_public(q, tick, tape_obj):
        packet = raw_observe(q, tick, tape_obj)
        # Keep only the image and noisy readout in the policy-visible packet.
        return replace(packet, qualified=False, genuine=False)

    scene.observe = observe_public
    legacy_control.review = lambda state: public_single(state, scene)
    legacy_scene.review = legacy_control.review
    base_advance = ctrl.advance

    def checked_advance(state, motion, tape_obj, *, trace=None, phase='probe'):
        pe = float(np.max(np.abs(motion.x[0]-state.q)))
        ve = float(np.max(np.abs(motion.v[0]-state.v)))
        ctrl._max_exec_join_position = max(getattr(ctrl, '_max_exec_join_position', 0.), pe)
        ctrl._max_exec_join_velocity = max(getattr(ctrl, '_max_exec_join_velocity', 0.), ve)
        if pe > 1.e-7 or ve > 1.e-7:
            raise RuntimeError('discontinuous ProDMP block at advance boundary')
        start_tick = int(state.tick)
        q0, v0 = state.q.copy(), state.v.copy()
        path = float(np.linalg.norm(np.diff(scene.body.surface(motion.x), axis=0), axis=1).sum())
        new_state = base_advance(state, motion, tape_obj, trace=trace, phase=phase)
        ctrl._motion_trace.append({"start_tick": start_tick, "end_tick": int(new_state.tick),
            "phase": phase, "q_start": q0.tolist(), "v_start": v0.tolist(),
            "q_end": new_state.q.tolist(), "v_end": new_state.v.tolist(),
            "path_mm": path, "join_position_error": pe, "join_velocity_error": ve})
        return new_state

    ctrl.advance = checked_advance
    ctrl._base_advance = base_advance
    ctrl._accepted_frame = None
    ctrl._stop_complete = False
    ctrl._stop_reason = None
    ctrl._execution_failure = None
    ctrl._motion_trace = []
    return scene, ctrl, tape


def shared_candidate_stabilize(scene, ctrl, state, tape, trace, deadline):
    """Shared policy stop, continuous brake, and optional image-confirmation dwell."""
    if getattr(ctrl, '_stop_complete', False):
        return state
    accepted = public_confirm(state, scene)
    if accepted is not None:
        ctrl._accepted_frame = accepted
        if np.max(np.abs(state.v)) < 1.e-6:
            ctrl._stop_reason = 'confirmed'
            ctrl._stop_complete = True
            return state
    candidate = public_single(state, scene) is not None
    base_advance = ctrl._base_advance
    brake = None
    speed = np.asarray(state.v, float)
    braking_distance = speed*np.abs(speed)/(2.*ACCEL)
    for factor in (1.0, 1.25, 1.5, 2.0):
        target = np.clip(state.q + factor*braking_distance, *legacy_control.CONTACT)
        brake = scene.motor.replan(state.q, state.v, target)
        if brake is not None:
            break
    if brake is None:
        ctrl._execution_failure = 'no_feasible_brake'
        ctrl._stop_reason = 'execution_failure'
        ctrl._stop_complete = True
        return state
    if (np.max(np.abs(brake.x[0]-state.q)) > 1.e-7 or
            np.max(np.abs(brake.v[0]-state.v)) > 1.e-7):
        raise RuntimeError('candidate brake did not preserve ProDMP position/velocity')
    for k in range(FRAME_STEPS, len(brake.x), FRAME_STEPS):
        if state.tick >= deadline:
            ctrl._execution_failure = 'deadline_during_brake'
            ctrl._stop_reason = 'execution_failure'
            ctrl._stop_complete = True
            return state
        block = Motion(brake.x[k-FRAME_STEPS:k+1], brake.v[k-FRAME_STEPS:k+1])
        if np.max(np.abs(block.x[0]-state.q)) > 1.e-7 or np.max(np.abs(block.v[0]-state.v)) > 1.e-7:
            raise RuntimeError('discontinuous brake sub-block')
        state = base_advance(state, block, tape, trace=trace, phase='SHARED_CONFIRM_BRAKE')
        accepted = public_confirm(state, scene)
        if accepted is not None:
            ctrl._accepted_frame = accepted
    if np.max(np.abs(state.v)) > .8:
        ctrl._execution_failure = 'brake_ended_above_speed_tolerance'
        ctrl._stop_reason = 'execution_failure'
        ctrl._stop_complete = True
        return state
    if candidate and ctrl._accepted_frame is None:
        idle_x = np.broadcast_to(state.q, (6*FRAME_STEPS+1, 4)).copy()
        idle = Motion(idle_x, np.zeros_like(idle_x))
        for k in range(FRAME_STEPS, len(idle.x), FRAME_STEPS):
            if state.tick >= deadline:
                break
            block = Motion(idle.x[k-FRAME_STEPS:k+1], idle.v[k-FRAME_STEPS:k+1])
            if np.max(np.abs(block.x[0]-state.q)) > 1.e-7 or np.max(np.abs(block.v[0]-state.v)) > 1.e-7:
                raise RuntimeError('discontinuous confirmation dwell')
            state = base_advance(state, block, tape, trace=trace, phase='SHARED_CONFIRM_DWELL')
            accepted = public_confirm(state, scene)
            if accepted is not None:
                ctrl._accepted_frame = accepted
                break
    if np.max(np.abs(state.v)) >= 1.e-6:
        ctrl._execution_failure = 'brake_final_velocity_nonzero'
        ctrl._stop_reason = 'execution_failure'
        ctrl._stop_complete = True
        return state
    if ctrl._accepted_frame is not None:
        ctrl._stop_reason = 'confirmed'
    elif candidate:
        ctrl._stop_reason = 'candidate_unconfirmed'
    else:
        ctrl._stop_reason = 'abstain_no_candidate'
    ctrl._stop_complete = True
    return state


def public_prefix_replay(scene, tape, prefix, rows):
    """Assimilate the shared scan's image marks as well as its liver readouts."""
    state = scene.at_start()
    for row in rows:
        tick = int(round(float(row[0])/legacy_scene.FRAME_DT))
        q = np.asarray(row[1:5], dtype=float)
        packet = scene._offline_observe(q, tick, tape)
        old_readout = np.asarray([np.nan if value is None else value for value in row[17:27]],float)
        if not np.allclose(packet.reading.ravel(), old_readout, rtol=1.e-8, atol=1.e-6, equal_nan=True):
            raise RuntimeError('shared sweep replay changed its time-indexed noisy readout')
        packet = replace(packet, qualified=False, genuine=False)
        state = scene.assimilate(state, packet, liver_override=prefix.liver)
    return replace(state, q=prefix.q.copy(), v=prefix.v.copy(), tick=prefix.tick,
                   travel=prefix.travel, turn=prefix.turn)


def prescribed_sweep_log(scene, sweep_index, observed_rows):
    """Recover exact q/v samples of the shared fixed ProDMP prefix and verify its trace."""
    final_theta, offset = legacy_scene.SWEEPS[sweep_index]
    motion = sweep_motion(scene.motor, scene.envelope, scene.start, final_theta, offset)
    surface = scene.body.surface(motion.x)
    blocks=[]
    qobs=np.asarray([row[1:5] for row in observed_rows],float)
    expected=motion.x[FRAME_STEPS::FRAME_STEPS]
    if len(qobs)!=len(expected) or np.max(np.abs(qobs-expected))>1.e-8:
        raise RuntimeError('shared sweep planned q sequence does not match executed observation trace')
    for j,k in enumerate(range(FRAME_STEPS,len(motion.x),FRAME_STEPS)):
        blocks.append({"start_tick":j,"end_tick":j+1,"phase":"liver_sweep",
            "q_start":motion.x[k-FRAME_STEPS].tolist(),"v_start":motion.v[k-FRAME_STEPS].tolist(),
            "q_end":motion.x[k].tolist(),"v_end":motion.v[k].tolist(),
            "path_mm":float(np.linalg.norm(np.diff(surface[k-FRAME_STEPS:k+1],axis=0),axis=1).sum()),
            "join_position_error":float(np.max(np.abs(motion.x[k-FRAME_STEPS]-motion.x[k-FRAME_STEPS]))),
            "join_velocity_error":float(np.max(np.abs(motion.v[k-FRAME_STEPS]-motion.v[k-FRAME_STEPS])))})
    return {"sweep_index":int(sweep_index),"path_mm":float(np.linalg.norm(np.diff(surface,axis=0),axis=1).sum()),
            "frames":len(blocks),"progress_mm":np.cumsum([x['path_mm'] for x in blocks]).tolist(),
            "motion_trace":blocks,"verified_against_observed_q":True}


def step_motion(scene, ctrl, state, motion, tape, phase, trace, deadline):
    body = scene.body
    pts = body.surface(motion.x)
    dist = np.r_[0., np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()]
    for k in range(FRAME_STEPS, len(motion.x), FRAME_STEPS):
        if state.tick >= deadline:
            break
        block = Motion(motion.x[k-FRAME_STEPS:k+1], motion.v[k-FRAME_STEPS:k+1])
        state = ctrl.advance(state, block, tape, trace=trace, phase=phase)
        state = replace(state, travel=state.travel + 0.)
        if np.max(np.abs(state.q-motion.x[k])) > 1.e-6:
            break
        if public_confirm(state, scene) is not None:
            ctrl._accepted_frame = public_confirm(state, scene)
            break
        if public_single(state, scene) is not None:
            final_deadline = getattr(scene, '_episode_deadline', deadline)
            state = shared_candidate_stabilize(scene, ctrl, state, tape, trace, final_deadline)
            break
    return state


def go_to(scene, ctrl, state, target, tape, phase, trace, deadline):
    if public_confirm(state, scene) is not None:
        ctrl._accepted_frame = public_confirm(state, scene)
        return state
    if public_single(state, scene) is not None:
        return shared_candidate_stabilize(scene, ctrl, state, tape, trace,
                                          getattr(scene, '_episode_deadline', deadline))
    motion = scene.motor.replan(state.q, state.v, np.clip(target, *legacy_control.CONTACT))
    if motion is None:
        return state
    return step_motion(scene, ctrl, state, motion, tape, phase, trace, deadline)


def align_target(scene, state):
    """Choose a local view using only the current public MAP anatomy hypothesis."""
    weights = scene.gall_weights(state)
    map_index = int(np.argmax(weights))
    predicted = scene.body.anatomy(scene.parameters[map_index])
    q = state.q.copy()

    def score(candidate):
        view = scene.body.view(candidate)
        section = scene.body.measure(view, predicted)
        centroid = float(section.gall[1])
        axis_error = float(np.rad2deg(np.arcsin(np.clip(abs(predicted.axis @ view.normal),0.,1.))))
        # Lexicographic priorities: same-lumen support, centered visible lumen,
        # then long-axis plane inclusion. Final terms retain local contact continuity.
        return (int(section.qualified), -abs(centroid), -axis_error,
                float(section.coverage), float(section.gall[0]),
                -float(np.linalg.norm(candidate-q)))

    best = q
    steps = ((-12.,0.,0.,0.),(-6.,0.,0.,0.),(6.,0.,0.,0.),(12.,0.,0.,0.),
             (0.,-6.,0.,0.),(0.,-3.,0.,0.),(0.,3.,0.,0.),(0.,6.,0.,0.),
             (0.,0.,-20.,0.),(0.,0.,-10.,0.),(0.,0.,10.,0.),(0.,0.,20.,0.),
             (0.,0.,0.,-16.),(0.,0.,0.,-8.),(0.,0.,0.,8.),(0.,0.,0.,16.),
             (-6.,-3.,0.,0.),(6.,3.,0.,0.))
    for _ in range(2):
        candidates = [best]
        candidates.extend(np.clip(best+np.asarray(delta), *legacy_control.CONTACT) for delta in steps)
        candidates.append(np.clip(scene.body.aim(best[:2], predicted.center, predicted.axis),
                                  *legacy_control.CONTACT))
        best = max(candidates, key=score)
    return best


def center_servo_target(scene, state, lateral_error):
    """Use the image centroid and model projection Jacobian to adjust contact."""
    q = state.q.copy()
    weights = scene.gall_weights(state)

    def predicted_lateral(chart):
        pts, _, _ = scene.project(chart)
        return float(weights @ pts[:, :, 0].mean(axis=1))

    j0, jt = q.copy(), q.copy()
    j0[0] += 1.0
    jt[1] += .5
    base = predicted_lateral(q)
    jac = np.array([predicted_lateral(j0)-base,
                    (predicted_lateral(jt)-base)/.5])
    delta = -float(lateral_error) * jac / max(float(jac@jac), .05)
    q[0] += float(np.clip(delta[0], -8., 8.))
    q[1] += float(np.clip(delta[1], -3., 3.))
    return np.clip(q, *legacy_control.CONTACT)


def local_public_targets(scene, state):
    """Small fan around current contact, ranked only with public belief forecasts."""
    q = state.q
    offsets = ([(-5.,0.,0.,0.), (5.,0.,0.,0.), (0.,-3.,0.,0.), (0.,3.,0.,0.),
                (0.,0.,-10.,0.), (0.,0.,10.,0.), (0.,0.,0.,-8.), (0.,0.,0.,8.)])
    weights = scene.gall_weights(state)
    options = []
    for off in offsets:
        target = np.clip(q+off, *legacy_control.CONTACT)
        _, support, quality = scene.project(target)
        score = float(quality @ weights + .05 * support.mean())
        options.append((score, target))
    return [target for _, target in sorted(options, key=lambda x:x[0], reverse=True)]


def skill_run(scene, ctrl, tape, *, deadline, trace=None, initial=None, shared_sweep=False):
    state = scene.at_start() if initial is None else initial
    body = scene.body
    phases = []
    if public_single(state, scene) is not None:
        phases.append({"phase":"SHARED_CANDIDATE_AT_ENTRY","tick":state.tick})
        return state, phases

    def run_sweep(final_theta, offset, label):
        nonlocal state
        motion = sweep_motion(scene.motor, scene.envelope, state.q, final_theta, offset)
        phases.append({"phase": label, "start_tick": state.tick, "path_mm": float(np.linalg.norm(np.diff(body.surface(motion.x), axis=0), axis=1).sum()), "frames": len(motion.x)//FRAME_STEPS})
        state = step_motion(scene, ctrl, state, motion, tape, label, trace, deadline)

    if shared_sweep:
        phases.append({"phase": "SWEEP_SHARED", "start_tick": 0, "end_tick": state.tick,
                       "path_mm": float(state.travel), "progress_fraction": 1.0,
                       "readout_frames": len(state.recent), "liver_posterior_updated": True})
    else:
        run_sweep(-55., 27., "SWEEP")
    image_found = any(np.isfinite(p.marks[:, 0]).sum() >= 2 for p in state.recent)
    if not image_found and state.tick < deadline:
        # Bounded recovery: return to a subcostal guide, then use a lateral window.
        run_sweep(-42., 38., "RECOVER_COSTAL")
        q = state.q.copy()
        q[1] = -55.
        q[0] = float(scene.envelope(q[1]) - 28.)
        q = scene.body.aim(q[:2], scene.estimate(state)[0], scene.estimate(state)[1])
        state = go_to(scene, ctrl, state, q, tape, "RECOVER_LATERAL", trace, deadline)
        phases.append({"phase": "LOST_TO_RECOVER", "tick": state.tick, "image_found": any(np.isfinite(p.marks[:,0]).sum()>=2 for p in state.recent)})

    segment_guard = any(np.isfinite(p.reading[0,0]) and np.isfinite(p.marks[:,0]).sum() >= 2
                        for p in state.recent)
    if (state.tick < deadline and segment_guard and public_single(state, scene) is None
            and public_confirm(state, scene) is None):
        center, axis, length = scene.estimate(state)
        marks = [p.marks for p in state.recent if np.isfinite(p.marks[:, 0]).sum() >= 2]
        lateral_rows = [float(p.reading[0,1]) for p in state.recent
                        if np.isfinite(p.reading[0,1])]
        lateral = float(np.median(lateral_rows)) if lateral_rows else 0.
        target = center_servo_target(scene, state, lateral)
        # Hold roll/rock through centering; ALIGN handles plane orientation.
        phases.append({"phase": "CENTER", "tick": state.tick, "lateral_servo_mm": lateral, "target_lateral_chart": float(target[0])})
        state = go_to(scene, ctrl, state, target, tape, "CENTER", trace, deadline)
    elif state.tick < deadline:
        phases.append({"phase": "CENTER_GUARD_FAILED", "tick": state.tick,
                       "guard": "recent gall readout and at least two image landmarks"})
    if (state.tick < deadline and segment_guard and public_single(state, scene) is None
            and public_confirm(state, scene) is None):
        center, axis, _ = scene.estimate(state)
        target = align_target(scene, state)
        phases.append({"phase": "ALIGN", "tick": state.tick, "target_chart": target.tolist()})
        state = go_to(scene, ctrl, state, target, tape, "ALIGN", trace, deadline)
    # Continue with small public-image servo corrections; never inspect evaluator flags.
    while state.tick < deadline and public_confirm(state, scene) is None and public_single(state, scene) is None:
        if not state.recent:
            break
        last = state.recent[-1]
        _, _, _ = scene.estimate(state)
        target = align_target(scene, state)
        if np.isfinite(last.marks[:, 0]).sum() >= 2:
            dx = float(last.reading[0,1]) if np.isfinite(last.reading[0,1]) else None
            if dx is not None:
                centered = center_servo_target(scene, state, dx)
                target[:2] = centered[:2]
        current = scene.motor.replan(state.q, state.v, target)
        if current is None or np.max(np.abs(target-state.q)) < .4:
            moved = False
            for fallback in local_public_targets(scene, state):
                current = scene.motor.replan(state.q, state.v, fallback)
                if current is not None and np.max(np.abs(fallback-state.q)) >= .4:
                    target = fallback
                    moved = True
                    break
            if not moved:
                break
        state = step_motion(scene, ctrl, state, current, tape, "CONFIRM", trace, deadline)
    phases.append({"phase": "CONFIRM", "tick": state.tick, "public_candidate": public_confirm(state, scene) is not None})
    return state, phases


def score_offline(scene, state, trace_rows, accepted_override=None):
    """All truth access stays in this offline evaluator, after policy execution."""
    accepted = accepted_override if accepted_override is not None else public_confirm(state, scene)
    anatomy = scene.truth

    def inspect(q, tick):
        view = scene.body.view(q, scene.case.rib)
        section, _, _ = scene.truth_packet(q)
        observed = scene._offline_observe(q, tick, scene._evaluation_tape)
        center_x = float(section.gall[1])
        plane_error = float(np.rad2deg(np.arcsin(np.clip(abs(anatomy.axis @ view.normal), 0., 1.))))
        strict_geometry = bool(section.qualified and abs(center_x) <= 12. and plane_error <= 15.)
        return strict_geometry, bool(section.qualified and observed.genuine), center_x, plane_error, bool(section.qualified)

    per_tick = {}
    for row in trace_rows:
        tick = int(round(float(row[0])/legacy_scene.FRAME_DT))
        if tick not in per_tick:
            per_tick[tick] = inspect(np.asarray(row[1:5], dtype=float), tick)
    strict_ticks = [tick for tick, value in per_tick.items()
                    if value[0] and scene._offline_observe(np.asarray(next(row[1:5] for row in trace_rows
                        if int(round(float(row[0])/legacy_scene.FRAME_DT)) == tick)), tick,
                        scene._evaluation_tape).genuine]
    loose_ticks = [tick for tick, value in per_tick.items() if value[1]]
    accepted_data = None if accepted is None else inspect(accepted.q, accepted.tick)
    accepted_genuine = bool(accepted is not None and
                            scene._offline_observe(accepted.q, accepted.tick, scene._evaluation_tape).genuine)
    strict = bool(accepted is not None and accepted_data[0] and accepted_genuine)
    legacy = bool(accepted is not None and accepted_data[4] and accepted_genuine)
    return {"strict_success": strict, "legacy_correct_at_delivery": legacy,
            "wrong_delivery": bool(accepted is not None and not strict),
            "ever_qualified_genuine": bool(loose_ticks), "ever_strict_genuine": bool(strict_ticks),
            "accepted_tick": None if accepted is None else int(accepted.tick),
            "offline_center_x_mm": None if accepted_data is None else accepted_data[2],
            "offline_axis_plane_error_deg": None if accepted_data is None else accepted_data[3],
            "offline_body_measure_qualified": None if accepted_data is None else accepted_data[4],
            "quality_tick_count": len(loose_ticks), "strict_tick_count": len(strict_ticks)}


def enforce_execution_success(score, state, ctrl):
    """A truth-qualified image is not a completed acquisition if execution failed."""
    out = dict(score)
    image_strict = bool(out['strict_success'])
    stopped = bool(np.max(np.abs(state.v)) < 1.e-6 and ctrl._execution_failure is None)
    out['image_strict_at_delivery'] = image_strict
    out['strict_success'] = bool(image_strict and stopped and ctrl._accepted_frame is not None)
    out['wrong_delivery'] = bool(ctrl._accepted_frame is not None and not out['strict_success'])
    out['final_stop_verified'] = stopped
    out['stop_reason'] = ctrl._stop_reason
    out['execution_failure'] = ctrl._execution_failure
    out['final_velocity_max_abs'] = float(np.max(np.abs(state.v)))
    return out


def run_case(seed, condition, stream):
    scene, ctrl, tape = configure(seed, condition, stream)
    # Equal initial, image-driven subcostal sweep for both policies.
    prefix_trace = []
    prefix = scene.start_prefix(seed % 3, tape, trace=prefix_trace)
    sweep_log = prescribed_sweep_log(scene, seed % 3, prefix_trace)
    prefix = public_prefix_replay(scene, tape, prefix, prefix_trace)
    deadline = prefix.tick + MAX_TICKS
    work_deadline = deadline - 12
    scene._episode_deadline = deadline
    baseline_trace = []
    baseline = ctrl.mpc(prefix, tape, work_deadline, trace=baseline_trace)
    baseline = shared_candidate_stabilize(scene, ctrl, baseline, tape, baseline_trace, deadline)
    baseline_score = enforce_execution_success(
        score_offline(scene, baseline, prefix_trace+baseline_trace, ctrl._accepted_frame), baseline, ctrl)
    # New scene/tape stream for candidate; same time-indexed noise condition, seed.
    skill_scene, skill_ctrl, skill_tape = configure(seed, condition, stream)
    trace = []
    skill_prefix_trace = []
    skill_prefix = skill_scene.start_prefix(seed % 3, skill_tape, trace=skill_prefix_trace)
    candidate_sweep_log = prescribed_sweep_log(skill_scene, seed % 3, skill_prefix_trace)
    skill_prefix = public_prefix_replay(skill_scene, skill_tape, skill_prefix, skill_prefix_trace)
    skill_scene._episode_deadline = deadline
    candidate, phases = skill_run(skill_scene, skill_ctrl, skill_tape, deadline=work_deadline, trace=trace,
                                  initial=skill_prefix, shared_sweep=True)
    candidate = shared_candidate_stabilize(skill_scene, skill_ctrl, candidate, skill_tape, trace, deadline)
    candidate_score = enforce_execution_success(
        score_offline(skill_scene, candidate, skill_prefix_trace+trace, skill_ctrl._accepted_frame), candidate, skill_ctrl)
    return {"seed": int(seed), "condition": condition,
            "source_landmark_version": LANDMARK_AUDIT["version"],
            "visible_centroid_definition": "noisy mask centroid readout[0,1]; offline section.gall[1]",
            "registered_body_beta_known": True,
            "online_truth_access": False, "offline_truth_scoring": True,
            "motion_model": "fitted contact ProDMP; local poses use MAP posterior Body.measure",
            "strict_acceptance": {"center_x_mm": 12., "axis_plane_error_deg": 15., "same_frame_qualified": True},
            "public_confirmation": {"center_x_mm": CENTER_X_MM, "expected_span_fraction": MIN_SPAN_FRACTION,
                                    "middle_collinear_mm": MAX_COLLINEAR_ERROR_MM},
            "baseline_old_mpc_shared_confirmation": {**baseline_score, "travel_mm": baseline.travel,
                "turn_deg": baseline.turn, "time_s": baseline.tick*legacy_scene.FRAME_DT,
                "trace": prefix_trace+baseline_trace, "stop_reason": ctrl._stop_reason,
                "execution_failure": ctrl._execution_failure,
                "final_velocity_max_abs": float(np.max(np.abs(baseline.v))),
                "final_chart": baseline.q.tolist(), "final_velocity": baseline.v.tolist(),
                "motion_trace": sweep_log['motion_trace']+ctrl._motion_trace,
                "locked_accepted_tick": None if ctrl._accepted_frame is None else int(ctrl._accepted_frame.tick)},
            "ordered_skill_shared_confirmation": {**candidate_score, "travel_mm": candidate.travel,
                "turn_deg": candidate.turn, "time_s": candidate.tick*legacy_scene.FRAME_DT,
                "phases": phases, "trace": skill_prefix_trace+trace,
                "stop_reason": skill_ctrl._stop_reason, "execution_failure": skill_ctrl._execution_failure,
                "final_velocity_max_abs": float(np.max(np.abs(candidate.v))),
                "final_chart": candidate.q.tolist(), "final_velocity": candidate.v.tolist(),
                "motion_trace": candidate_sweep_log['motion_trace']+skill_ctrl._motion_trace,
                "locked_accepted_tick": None if skill_ctrl._accepted_frame is None else int(skill_ctrl._accepted_frame.tick)},
            "baseline_frames": baseline.tick, "candidate_frames": candidate.tick,
            "shared_sweep": sweep_log, "candidate_shared_sweep": candidate_sweep_log,
            "shared_prefix_end_tick": prefix.tick, "absolute_deadline_tick": deadline,
            "decision_work_deadline_tick": work_deadline,
            "confirmed_frame_locked": {"baseline": ctrl._accepted_frame is not None,
                                       "candidate": skill_ctrl._accepted_frame is not None},
            "max_join_position_error": max(ctrl.join_position, skill_ctrl.join_position,
                                             ctrl._max_exec_join_position, skill_ctrl._max_exec_join_position),
            "max_join_velocity_error": max(ctrl.join_velocity, skill_ctrl.join_velocity,
                                            ctrl._max_exec_join_velocity, skill_ctrl._max_exec_join_velocity)}


def raw_start_audit(seed, condition, stream):
    """Independent difficult-start audit; it does not enter paired main results."""
    scene, ctrl, tape = configure(seed, condition, stream)
    # Offline cohort construction deliberately selects a rib-shadowed window.
    # The controller itself only receives the subsequent noisy image packets.
    theta_grid = (-58., -54., -50., -46., -42., -38., -34.)
    offset_grid = (16., 24., 32., 40., 46.)
    xx, dd = np.meshgrid(np.linspace(-65., 65., 35), np.linspace(20., 195., 40))
    fan = ((np.abs(xx) <= dd*np.tan(np.deg2rad(28.))) & (np.hypot(xx,dd) <= 210.))
    xx, dd = xx[fan], dd[fan]
    starts = []
    for theta in theta_grid:
        for offset in offset_grid:
            q = np.array([float(scene.envelope(theta)-offset), theta, 0., 0.])
            if np.any(q < legacy_control.CONTACT[0]) or np.any(q > legacy_control.CONTACT[1]):
                continue
            view = scene.body.view(q, scene.case.rib)
            shadow = float(np.mean(view.beam_blocked(xx, dd)))
            starts.append((shadow, q))
    if not starts:
        raise RuntimeError('could not construct a contact-valid shadowed start')
    shadow_fraction, shadow_start = max(starts, key=lambda x:x[0])
    scene.start = shadow_start.copy()
    initial = scene.at_start()
    packet = scene.observe(initial.q, 0, tape)
    initial = scene.assimilate(initial, packet)
    trace = [scene.trace_row(initial, packet, "RAW_START")]
    actions = ctrl.propose(initial)
    feasible = 0
    checked = 0
    for target in actions.values():
        if target is None:
            continue
        checked += 1
        feasible += scene.motor.replan(initial.q, initial.v, target) is not None
    final_deadline = MAX_TICKS
    scene._episode_deadline = final_deadline
    state, phases = skill_run(scene, ctrl, tape, deadline=final_deadline-12, trace=trace, initial=initial)
    state = shared_candidate_stabilize(scene, ctrl, state, tape, trace, final_deadline)
    scored = enforce_execution_success(score_offline(scene, state, trace, ctrl._accepted_frame), state, ctrl)
    return {"seed": int(seed), "condition": condition, "raw_start_chart": initial.q.tolist(),
            "source_landmark_version": LANDMARK_AUDIT["version"],
            "visible_centroid_definition": "noisy mask centroid readout[0,1]; offline section.gall[1]",
            "online_truth_access": False, "offline_truth_scoring": True,
            "initial_public_image": bool(np.isfinite(packet.reading[0,0]) or np.isfinite(packet.reading[1,0])),
            "initial_gall_readout": bool(np.isfinite(packet.reading[0,0])),
            "initial_liver_readout": bool(np.isfinite(packet.reading[1,0])),
            "offline_rib_shadow_fraction": float(shadow_fraction),
            "shadow_start_construction": "offline body rib ray fraction over fan grid; no anatomy flags used online",
            "raw_mpc_request_feasible_count": int(feasible), "raw_mpc_request_count": int(checked),
            "costal_sweep_completed": any(x['phase']=='SWEEP' for x in phases),
            "phase_events": phases, **scored, "travel_mm": float(state.travel),
            "time_s": state.tick*legacy_scene.FRAME_DT, "stop_reason": ctrl._stop_reason,
            "execution_failure": ctrl._execution_failure,
            "final_velocity_max_abs": float(np.max(np.abs(state.v))), "trace": trace}


def summarize(rows):
    summary = {"n_paired": len(rows), "conditions": {}}
    for condition in sorted(set(r['condition'] for r in rows)):
        group = [r for r in rows if r['condition'] == condition]
        b = [r['baseline_old_mpc_shared_confirmation'] for r in group]
        c = [r['ordered_skill_shared_confirmation'] for r in group]
        summary['conditions'][condition] = {
            "baseline_strict_success": sum(x['strict_success'] for x in b),
            "candidate_strict_success": sum(x['strict_success'] for x in c),
            "baseline_wrong_delivery": sum(x['wrong_delivery'] for x in b),
            "candidate_wrong_delivery": sum(x['wrong_delivery'] for x in c),
            "baseline_ever_qualified": sum(x['ever_qualified_genuine'] for x in b),
            "candidate_ever_qualified": sum(x['ever_qualified_genuine'] for x in c),
            "baseline_ever_strict_acquired": sum(x['ever_strict_genuine'] for x in b),
            "candidate_ever_strict_acquired": sum(x['ever_strict_genuine'] for x in c),
            "baseline_strict_acquired_not_delivered": sum(x['ever_strict_genuine'] and not x['strict_success'] for x in b),
            "candidate_strict_acquired_not_delivered": sum(x['ever_strict_genuine'] and not x['strict_success'] for x in c),
            "baseline_mean_travel_mm": float(np.mean([x['travel_mm'] for x in b])),
            "candidate_mean_travel_mm": float(np.mean([x['travel_mm'] for x in c])),
            "baseline_mean_time_s": float(np.mean([x['time_s'] for x in b])),
            "candidate_mean_time_s": float(np.mean([x['time_s'] for x in c])),
            "paired_strict_delta": int(sum(x['strict_success'] for x in c)-sum(x['strict_success'] for x in b))}
    return summary


def _run_case_task(task):
    return run_case(*task)


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['dev', 'eval'], default='dev')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    if args.stage == 'eval' and args.n < 32:
        parser.error('evaluation is frozen at no fewer than 32 new anatomies')
    initialize()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'eval':
        freeze_path = args.out/'freeze.json'
        if not freeze_path.exists():
            parser.error('freeze.json must be written and reviewed before evaluation')
        freeze = json.loads(freeze_path.read_text())
        for name, expected in freeze['frozen_code_sha256'].items():
            if _sha256(HERE/name) != expected:
                parser.error(f'frozen source changed after evaluation freeze: {name}')
        for name, expected in freeze['source_sha256'].items():
            if _sha256(name) != expected:
                parser.error(f'legacy source or asset changed after freeze: {name}')
    start = 58_649_000 if args.stage == 'dev' else 58_650_000
    rows = []
    raw_rows = []
    if args.stage == 'eval' and args.workers > 1:
        tasks = [(start+j, condition, i) for j in range(args.n)
                 for i, condition in enumerate(('wide', 'stress'))]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
            for index, row in enumerate(pool.map(_run_case_task, tasks, chunksize=1), 1):
                rows.append(row)
                if index % 8 == 0 or index == len(tasks):
                    print(f"eval: {index}/{len(tasks)} paired trajectories", flush=True)
    else:
        for j in range(args.n):
            seed = start + j
            for i, condition in enumerate(('wide', 'stress')):
                row = run_case(seed, condition, i)
                rows.append(row)
            print(f"{args.stage}: {j+1}/{args.n} anatomies", flush=True)
            if args.stage == 'dev' and j < 4:
                for i, condition in enumerate(('wide', 'stress')):
                    raw_rows.append(raw_start_audit(seed, condition, i))
    prefix = 'development' if args.stage == 'dev' else 'frozen_evaluation'
    with gzip.open(args.out/f'{prefix}.jsonl.gz', 'wt', compresslevel=4) as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    summary = {"stage": args.stage, "n_anatomies": args.n,
               "seed_range": [start, start+args.n-1], "noise_conditions": ['wide','stress'],
               "policy_comparison": "old MPC plus shared image-only confirmation vs ordered skill plus shared confirmation",
               "data_mode": "closed-loop procedure; observations and noisy readouts drive each subsequent action",
               "dird": False, "metrics": summarize(rows)}
    (args.out/f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if raw_rows:
        with gzip.open(args.out/'raw_start_audit.jsonl.gz', 'wt', compresslevel=4) as f:
            for row in raw_rows:
                f.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        summary['raw_start_audit'] = {
            'n': len(raw_rows), 'anatomies': min(4, args.n),
            'initial_public_image_count': sum(r['initial_public_image'] for r in raw_rows),
            'direct_plan_feasible_total': sum(r['raw_mpc_request_feasible_count'] for r in raw_rows),
            'direct_plan_request_total': sum(r['raw_mpc_request_count'] for r in raw_rows),
            'costal_sweep_completed': sum(r['costal_sweep_completed'] for r in raw_rows),
            'strict_delivery': sum(r['strict_success'] for r in raw_rows),
            'strict_acquired': sum(r['ever_strict_genuine'] for r in raw_rows),
            'wrong_delivery': sum(r['wrong_delivery'] for r in raw_rows)}
        (args.out/f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary['metrics'], indent=2))


if __name__ == '__main__':
    main()
