"""Training-only, physically executed near-success motion checkpoints.

The witness is an offline sampling hint.  It is used only to place chart-space
perturbations; checkpoint history contains only packets produced by the public
controller.  Runtime policy code must not import this module.
"""
from __future__ import annotations

import numpy as np

from view_geometry import CONTACT, FRAME_STEPS, Motion, prior_run, settled_plan


def _run(scene, ctrl, tape, state, history, motion, phase):
    """Assimilate each original-frame block through the ordinary controller."""
    for end in range(FRAME_STEPS, len(motion.x), FRAME_STEPS):
        block = Motion(motion.x[end-FRAME_STEPS:end+1],
                       motion.v[end-FRAME_STEPS:end+1])
        state = ctrl.advance(state, block, tape, phase=phase)
        history.append(state.recent[-1])
    return state


def _plan(motor, q0, q1, frames_left):
    return settled_plan(motor, np.asarray(q0), np.asarray(q1), int(frames_left))


def reverse_checkpoints(scene, ctrl, tape, prefix, prefix_history, witness_q,
                        variant=0):
    """Yield moving checkpoints on near-miss and standard approach paths.

    Near-miss remainders continue past the good view and settle at an imperfect
    chart offset, so they do not encode a hindsight-positive action label. The
    nominal remainder settles at the supplied witness. ``variant`` mirrors the
    path directions deterministically. Stages skip safely if a legal leg or
    the confirmation reserve is unavailable.
    """
    witness = np.clip(np.asarray(witness_q, dtype=float), *CONTACT)
    if witness.shape != (4,) or not np.isfinite(witness).all():
        return
    deadline = int(prefix.tick) + int(prior_run.MAX_TICKS)
    # Approach from signed chart offsets, then cross the witness or settle there.
    sign = -1. if int(variant) % 2 else 1.
    specs = (
        ('near_overshoot', sign*np.array([7., 2.5, 3., -2.]),
         -sign*np.array([7., 2.5, 3., -2.])),
        ('wide_near_miss', sign*np.array([13., 5., 5., -4.]),
         -sign*np.array([10., 3., 3., -2.])),
        ('nominal_approach', sign*np.array([7., 2.5, 3., -2.]),
         np.zeros(4)),
    )
    for stage, delta, terminal_delta in specs:
        history = list(prefix_history)
        state = prefix
        offset = np.clip(witness + delta, *CONTACT)
        # Reserve at least 24 ticks for confirmation and downstream recovery.
        first = _plan(scene.motor, state.q, offset, deadline-state.tick-24)
        if first is None:
            continue
        try:
            state = _run(scene, ctrl, tape, state, history, first,
                         'training_curriculum_' + stage + '_offset')
        except (ValueError, RuntimeError):
            continue
        endpoint = np.clip(witness + terminal_delta, *CONTACT)
        second = _plan(scene.motor, state.q, endpoint, deadline-state.tick-24)
        if second is None or len(second.x) <= FRAME_STEPS * 2:
            continue
        # Capture while the actual path passes the witness neighborhood. The
        # near-miss continuations proceed beyond it to a distinct endpoint.
        nblocks = (len(second.x)-1)//FRAME_STEPS
        fraction = .5 if stage == 'near_overshoot' else (
            .565 if stage == 'wide_near_miss' else .72)
        capture = max(1, min(nblocks-1, int(round(fraction*nblocks))))
        for block_id in range(capture):
            lo, hi = block_id*FRAME_STEPS, (block_id+1)*FRAME_STEPS
            block = Motion(second.x[lo:hi+1], second.v[lo:hi+1])
            state = ctrl.advance(state, block, tape,
                                 phase='training_curriculum_' + stage + '_approach')
            history.append(state.recent[-1])
        lo = capture*FRAME_STEPS
        remainder = Motion(second.x[lo:], second.v[lo:])
        # Explicitly enforce the checkpoint join and legal terminal state.
        if (np.max(np.abs(remainder.x[0]-state.q)) > 1.e-7 or
                np.max(np.abs(remainder.v[0]-state.v)) > 1.e-7 or
                np.max(np.abs(remainder.v[-1])) > 1.e-7 or state.tick >= deadline-24):
            continue
        yield stage, state, history, remainder
