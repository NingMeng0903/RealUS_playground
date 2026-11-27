"""Public, feasible out-and-back search proposals in the existing ProDMP space.

The two proposals use the current measured chart state and the same motion
limits as the recovery benchmark.  They do not use a hidden vessel location.
Both the learned scorer and the live MPC must receive the same augmented menu.
"""

from __future__ import annotations

import numpy as np


def add_search_sweeps(ctrl, state, alias, choices, *, amplitudes_mm=(1.5, 2.5)):
    """Return a copy of ``choices`` with feasible one-second sweep proposals.

    Each sweep has the same endpoint as the public wait proposal but briefly
    changes the lateral acoustic window.  The shape is projected into the
    endpoint-nullspace of the existing integrated ProDMP; no teleport or
    trajectory outside the primitive is introduced.
    """

    from controller import Candidate, KINDS, MAX_TRAVEL, PHASE  # existing benchmark
    from primitive import admissible

    result = dict(choices)
    wait = result.get("wait")
    if wait is None:
        return result
    base = np.asarray(wait.lateral.x, dtype=float)
    base_v = np.asarray(wait.lateral.v, dtype=float)
    pulse = np.sin(np.pi * PHASE) ** 2
    for amplitude in amplitudes_mm:
        for sign, side in ((-1.0, "left"), (1.0, "right")):
            name = f"sweep_{side}_{abs(amplitude):g}"
            KINDS.setdefault(name, "side")
            desired_x = base + sign * float(amplitude) * pulse
            desired_v = base_v + sign * float(amplitude) * np.pi * np.sin(2.0 * np.pi * PHASE)
            shape = ctrl.primitive.fit(desired_x, desired_v)
            plan = ctrl.primitive.condition(0, len(base) - 1, state.x, state.vx,
                                            wait.lateral.goal, 0.0, shape)
            if not admissible(plan):
                continue
            distance, turn = ctrl.geometry.cost(wait.s, plan.x)
            if state.travel + distance > MAX_TRAVEL:
                continue
            center = ctrl.mean(state, float(wait.s[-1]), alias)[0]
            result[name] = Candidate(name, wait.s, wait.vs, plan, distance,
                                     turn, float(plan.goal - center))
    return result


__all__ = ["add_search_sweeps"]
