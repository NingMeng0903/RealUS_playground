"""One shared rollout contract for one- and two-segment scan decisions."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "ab_teacher_b_training_20261118"
sys.path.insert(0, str(PREVIOUS))
import experiment as base

from adaptive_controller import FRAMES_PER_BLOCK, MAX_TRAVEL, final_metrics
from adaptive_teacher import posterior_worlds
from learned_recovery import OBJECTIVE, predict_deltas
from run_adaptive_recovery import score_outcome


HANDOFF = "handoff"
TARGETS = base.VISIBLE_TARGETS
MODEL_BANK = {}


def deadline(root):
    return root.frame + 2 * FRAMES_PER_BLOCK


def copy_state(state):
    return replace(state, belief=state.belief.copy())


def run_motion(controller, state, record, scene, action):
    """Execute a checked trajectory from its own actual starting state."""
    next_state, next_record = copy_state(state), record.copy()
    if action is not None and controller.can_continue(next_state):
        controller.execute(scene, next_state, next_record, action)
    return next_state, next_record


def run_tail(controller, state, record, scene, open_until):
    """Grant MPC the same recovery rights at the same absolute scan frame."""
    state, record = copy_state(state), record.copy()
    frames_left = max(0, open_until - state.frame)
    open_blocks = (frames_left + FRAMES_PER_BLOCK - 1) // FRAMES_PER_BLOCK
    controller.tail(scene, state, record, expanded=True, open_blocks=open_blocks)
    return state, record


def simulation_outcome(controller, state, record, scene, root, open_until):
    state, record = run_tail(controller, state, record, scene, open_until)
    result = final_metrics(state, record, sampled_latent=scene.latent, root_frame=root.frame)
    result["turn_degs"] = sum(step["executed_turn_deg"] for step in record.decisions)
    return np.array([result[field] for field in TARGETS], dtype=float)


def actual_outcome(controller, state, record, scene, root, open_until):
    state, record = run_tail(controller, state, record, scene, open_until)
    result = score_outcome(state, record, root)
    final = {event["bin"]: event for event in record.certificates}
    revoked = {event["bin"]: event["frame"] for event in record.retractions}
    retained = {j for j, event in final.items() if event["identity_ok"] and event["frame"] > root.frame
                and event["accepted_at"] >= revoked.get(j, -1)}
    result["new_query_image_retained"] = root.query_bin() in retained
    result["new_image_and_complete"] = float(bool(result["corrected_complete"]) and bool(retained))
    if (result["max_handoff_error"] > 1.e-7 or result["tcp_mm"] > MAX_TRAVEL + 1.e-7
            or result["seconds"] > 8. + 1.e-7):
        raise ValueError("executed scan exceeded its common limit")
    return {field: result[field] for field in base.FIELDS}


def starting_record(controller, state):
    return controller.virtual_record(state)


def value_after_action(controller, state, scene, action, open_until):
    next_state, next_record = run_motion(controller, state, starting_record(controller, state), scene, action)
    return simulation_outcome(controller, next_state, next_record, scene, state, open_until)


def action_bank(state, packets, seed, count=base.LABEL_TAPES):
    return list(base.label_worlds(state, packets, seed))[:count]


def second_menu(controller, state, open_until):
    if not controller.can_continue(state):
        return {}, None
    anchor = controller.choose_mpc(state, expanded=True, open_decision=state.frame < open_until)
    if anchor is None:
        return {}, None
    menu = {anchor.name: anchor}
    for name in base.NEW_REQUESTS:
        candidate = base.generator.request_action(controller, state, name)
        if candidate is not None and name not in menu:
            menu[name] = candidate
    return menu, anchor.name


def scored_choice(inputs, models):
    scores = predict_deltas(models, inputs["features"])
    return base.choice_from_predictions(inputs, scores)[0]


def choose_first(controller, root, prefix, menu, anchor, models):
    inputs = base.policy_inputs(controller, root, prefix, menu, anchor)
    choice = scored_choice(inputs, models)
    return HANDOFF if choice == anchor else choice


def second_inputs(controller, state, record, open_until):
    menu, anchor = second_menu(controller, state, open_until)
    if not menu or len(menu) == 1:
        return menu, anchor, None
    return menu, anchor, base.policy_inputs(controller, state, record, menu, anchor)


def choose_second(menu, anchor, inputs, models):
    if inputs is None:
        return HANDOFF, None
    choice = scored_choice(inputs, models)
    return (HANDOFF, None) if choice == anchor else (choice, menu[choice])


def root_evidence(controller, state, menu, pool, seed):
    predictions = posterior_worlds(state, base.generator.PROBE_SAMPLES, seed)
    scores = {name: base.generator.evidence_score(controller, state, menu[name], predictions)["score"]
              for name in pool}
    return sorted(pool, key=lambda name: (-round(scores[name], 12), menu[name].distance, name))


def comparison_pool(menu, aliases, anchor):
    preferred = sorted(set(aliases.values()) - {anchor})
    return preferred if len(preferred) >= base.PROPOSALS else sorted(set(menu) - {anchor})


def rating(vector):
    return float(vector[0]), float(vector @ OBJECTIVE)


def fixed_action(controller, state, name):
    if name == HANDOFF:
        return None
    if name in base.NEW_REQUESTS:
        return base.generator.request_action(controller, state, name)
    return controller.expanded_menu(state)[0].get(name)


def simulate_precommit(controller, root, first, worlds, open_until):
    """Score each fixed second command before the first actual probe returns."""
    states = [run_motion(controller, root, starting_record(controller, root), world, first) for world in worlds]
    results = {}
    for name in (HANDOFF, *base.NEW_REQUESTS):
        trials = []
        for (state, record), world in zip(states, worlds, strict=True):
            action = fixed_action(controller, state, name) if controller.can_continue(state) else None
            done, log = run_motion(controller, state, record, world, action)
            trials.append(simulation_outcome(controller, done, log, world, root, open_until))
        results[name] = np.mean(trials, axis=0)
    return max(results, key=lambda name: (rating(results[name]), name == HANDOFF))
