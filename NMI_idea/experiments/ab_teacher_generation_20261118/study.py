"""Connect the finite probe teachers to the existing streaming ultrasound executor."""

from dataclasses import replace
from itertools import product
from pathlib import Path
import sys
from time import process_time

import joblib
import numpy as np

from teacher_scores import acquisition_payoff, decision_value

EXPERIMENT = Path(__file__).resolve().parent.parent / "ab_search_embedding_a_20261117"
sys.path.insert(0, str(EXPERIMENT))
import benchmark_arbitrary_start as scans
from adaptive_controller import FRAME_STRIDE, FRAMES_PER_BLOCK, MAX_TRAVEL, station_bin
from adaptive_teacher import evaluate, expected, posterior_worlds, utility
from benchmark_window_encoding import choice_from_predictions, decision_inputs
from learned_recovery import predict_deltas
from run_adaptive_recovery import public_root, real_trial
from run_learned_recovery import CONTROLLERS, FAMILIES, MATERIAL_SURFACE, fingerprint
from train_startup_b import load_controllers


SOURCE = EXPERIMENT / "results" / "startup_b_new_library_v1"
EXISTING_B = EXPERIMENT / "results" / "identity_recovery_data_v1" / "models.joblib"
SOURCES = ("random", "direct", "precommit", "evidence")
CELLS = tuple(product(FAMILIES, ("natural", "burst")))
PROPOSAL_COUNT = 4
DISCOVERY_REPEATS = 4
DEV_REPEATS = 3
TEST_REPEATS = 8
PUBLIC_REPEATS = 8
PROBE_TICKS = (10, 25, 40)
PROBE_SAMPLES = 8
REQUESTS = {f"grid_{i:02d}": (ds, dx, 0.) for i, (ds, dx) in enumerate(
    product((-4., 0., 2.5, 5., 7.), (-6., -3., 0., 3., 6.)))}
FOLLOW_UP = ((0., 0.), (0., -5.), (0., 5.), (5., -5.), (5., 0.), (5., 5.))
STOCK_FIXED = ("advance", "wait", "recover", "left", "right", "side_left", "side_right",
               "back_left", "back_right", "back_center", "forward_left", "forward_right",
               "look_left", "look_right", *(f"cem_{i}" for i in range(6)))
FIELDS = ("corrected_complete", "final_wrong", "new_identity_mistake", "tcp_mm", "turn_degs",
          "seconds", "frames", "first_block_true_image", "fresh_query_confirmed", "target_reappeared",
          "centered_complete", "feedback_revisions", "max_handoff_error", "new_image_and_complete",
          "new_query_image_retained")
MODELS = []


def configure():
    load_controllers(SOURCE)
    MODELS[:] = joblib.load(EXISTING_B)["unconditioned_b"]
    used = (SOURCE / "library.json", EXISTING_B, SOURCE / "prodmp_fit.npy", MATERIAL_SURFACE)
    return {str(path.relative_to(EXPERIMENT)): fingerprint(path) for path in used}


def scenario(cell, seed):
    family, context = cell
    rng = np.random.default_rng([seed, 18280])
    blocks = int(rng.integers(1, 3))
    row = scans.make_row(family, "noise" if context == "natural" else "noise_and_loss", "rolling", blocks, seed)
    return row | {"context": context}


def checkpoint(row):
    controller, scene, state, prefix = scans.start_case(row, "in_progress")
    if state.frame != row["handoff_frame"] or not controller.can_continue(state):
        raise ValueError(f"missing feasible decision state for case {row['seed']}")
    return controller, scene, state, prefix


def request_action(controller, state, name):
    return controller.free_candidate(state, name, *REQUESTS[name], renderer="fit")


def novel_actions(controller, state, names, stock):
    actions, aliases = {}, {}
    for name in names:
        action = request_action(controller, state, name)
        if action is None:
            continue
        equivalent = next((key for key, a in (stock | actions).items()
                           if key != "wait" and np.max(abs(a.s - action.s)) < .05
                           and np.max(abs(a.lateral.x - action.lateral.x)) < .05), None)
        aliases[name] = name if equivalent is None else equivalent
        if equivalent is None:
            actions[name] = action
    return actions, aliases


def request_visibility(controller, state, action, predictor, query):
    """Peak predicted target visibility in the requested region; a nomination score."""
    ticks = [i for i in range(FRAME_STRIDE, len(action.s), FRAME_STRIDE) if station_bin(action.s[i]) == query]
    if not ticks:
        return np.zeros_like(state.belief)
    return np.max([predictor(action.s[i], action.lateral.x[i])[:, 0] for i in ticks], axis=0)


def direct_score(controller, scene, state, action):
    def privileged_visibility(s, x):
        return controller.world.imaging(
            s, x - scene.lateral_jitter, state.angle, width=scene.shadow_width_factor,
            lane_shift=scene.shadow_shift, depth_shift=scene.depth_jitter,
        )[1]

    return float(request_visibility(controller, state, action, privileged_visibility, state.query_bin())[scene.latent])


def evidence_score(controller, state, action, samples):
    query = state.query_bin()

    def predict(s, x):
        return controller.world.predict(s, x, state.angle)[1]

    prior = state.belief
    first_view = request_visibility(controller, state, action, predict, query)
    exit_state = replace(state, s=float(action.s[-1]), x=float(action.lateral.x[-1]),
                         vs=float(action.vs[-1]), vx=float(action.lateral.v[-1]),
                         travel=state.travel + action.distance, frame=state.frame + FRAMES_PER_BLOCK)
    follow_up = [np.zeros_like(prior)]
    if controller.can_continue(exit_state):
        for ds, dx in FOLLOW_UP:
            candidate = controller.free_candidate(exit_state, "observe", ds, dx, renderer="fit")
            if candidate is not None:
                follow_up.append(request_visibility(controller, state, candidate, predict, query))
    payoff = acquisition_payoff(first_view, np.column_stack(follow_up))
    if state.image_candidates is not None and state.ledger[query] is None:
        _, witness_identity = state.image_candidates[query]
        payoff = np.column_stack((payoff, witness_identity.T))
    posteriors = []
    for scenario_sample in samples:
        posterior = prior.copy()
        for tick in PROBE_TICKS:
            frame = state.frame + tick // FRAME_STRIDE
            s, x = action.s[tick], action.lateral.x[tick]
            packet, _, _ = controller.world.observe(scenario_sample, frame, s, x)
            mass, _ = controller.world.likelihood(packet, s, x, state.angle)
            posterior *= mass
            posterior /= posterior.sum()
        posteriors.append(posterior)
    value = decision_value(prior, np.asarray(posteriors), payoff)
    return {"score": value.score, "precommitted": value.precommitted, "information": value.information,
            "perfect_information": value.perfect_information, "first_view": float(prior @ first_view),
            "follow_up_options": int(payoff.shape[1])}


def trial(controller, state, prefix, future, action, *, demonstration=False):
    metrics, record = real_trial(controller, state, prefix, future, [action])
    if metrics["max_handoff_error"] > 1.e-7 or metrics["tcp_mm"] > MAX_TRAVEL + 1.e-7:
        raise ValueError("executed motion exceeded the common continuity or distance constraint")
    final_acceptances = {event["bin"]: event for event in record.certificates}
    last_revocations = {event["bin"]: event["frame"] for event in record.retractions}
    new_witness_bins = {j for j, event in final_acceptances.items()
                        if event["identity_ok"] and event["frame"] > state.frame
                        and event["accepted_at"] >= last_revocations.get(j, -1)}
    metrics["new_image_and_complete"] = float(bool(metrics["corrected_complete"]) and bool(new_witness_bins))
    metrics["new_query_image_retained"] = state.query_bin() in new_witness_bins
    output = {field: metrics[field] for field in FIELDS}
    if demonstration:
        offset = len(prefix.motion) - 1
        last = len(action.s) + offset
        output["motion"] = record.motion[offset:last]
        output["first_ref"] = np.column_stack((action.s, action.lateral.x, action.vs, action.lateral.v))
    return output


def mpc_action(controller, state, menu):
    return controller.mpc(state, menu, False, active=True, informative=True)


def discover(row):
    controller, truth, state, prefix = checkpoint(row)
    stock = controller.options_shared(state, renderer="fit")
    base = mpc_action(controller, state, stock)
    actions, _ = novel_actions(controller, state, REQUESTS, stock)
    names = list(actions)
    sample = posterior_worlds(state, PROBE_SAMPLES, 18284)
    started = process_time()
    direct = {name: direct_score(controller, truth, state, action) for name, action in actions.items()}
    direct_cpu = process_time() - started
    started = process_time()
    evidence = {name: evidence_score(controller, state, action, sample) for name, action in actions.items()}
    evidence_cpu = process_time() - started
    rng = np.random.default_rng([row["seed"], 18285])
    proposals = {
        "random": rng.permutation(names)[:PROPOSAL_COUNT].tolist(),
        "direct": sorted(names, key=lambda n: (-direct[n], actions[n].distance, n))[:PROPOSAL_COUNT],
        "precommit": sorted(names, key=lambda n: (-evidence[n]["precommitted"],
                                                 actions[n].distance, n))[:PROPOSAL_COUNT],
        "evidence": sorted(names, key=lambda n: (-evidence[n]["score"], -evidence[n]["information"],
                                                actions[n].distance, n))[:PROPOSAL_COUNT],
    }
    bank = scans.paired_futures(truth, row, DISCOVERY_REPEATS, 18281)
    qualify_actions = {"mpc": base} | {name: actions[name] for name in sorted(set().union(*proposals.values()))}
    qualification = {
        name: [trial(controller, state, prefix, future, action, demonstration=(name != "mpc" and i == 0))
               for i, future in enumerate(bank)]
        for name, action in qualify_actions.items()
    }
    return row | {"pool": names, "proposals": proposals, "qualification": qualification, "incumbent": "mpc",
                  "public_root": public_root(state), "first_packet_empty": not bool(prefix.packets[-1]["y"]),
                  "forecast": {"direct": direct, "evidence": evidence},
                  "screen_cpu_seconds": {"direct": direct_cpu, "evidence": evidence_cpu}}


def combined_menu(controller, state, library):
    stock = controller.options_shared(state, renderer="fit")
    available = sorted(set().union(*library.values()))
    generated, aliases = novel_actions(controller, state, available, stock)
    return stock, generated, aliases


def qualify(row, library):
    controller, scene, state, prefix = checkpoint(row)
    stock, generated, aliases = combined_menu(controller, state, library)
    menu = stock | generated
    mpc = mpc_action(controller, state, menu)
    bank = scans.paired_futures(scene, row, DEV_REPEATS, 18282)
    tests = {"mpc": mpc} | {k: stock.get(k, mpc) for k in STOCK_FIXED}
    tests |= {k: menu.get(aliases.get(k), mpc) for k in sorted(set().union(*library.values()))}
    unique = {a.name: a for a in tests.values()}
    cached = {name: [trial(controller, state, prefix, s, action) for s in bank]
              for name, action in unique.items()}
    available = sorted((set(stock) | set(aliases)) & tests.keys())
    return row | {"incumbent": "mpc", "available": available,
                  "qualification": {k: cached[a.name] for k, a in tests.items()}}


def public_selection(controller, state, stock, generated, aliases, library, incumbent):
    started = process_time()
    worlds = posterior_worlds(state, PUBLIC_REPEATS, 18286)
    menu = stock | generated
    forecast = {}
    cpu = {}
    for name, action in menu.items():
        action_started = process_time()
        forecast[name] = expected(evaluate(controller, state, action, worlds, open_blocks=1)[0])
        cpu[name] = process_time() - action_started
    overhead = process_time() - started - sum(cpu.values())
    selections = {}
    for source, request_ids in (("public_stock", ()), *library.items()):
        options = tuple(dict.fromkeys((*stock, *(aliases[name] for name in request_ids if name in aliases))))
        winner = max(options, key=lambda name: (utility(forecast[name]), name == incumbent))
        selections[source] = (menu[winner], overhead + sum(cpu[name] for name in options), len(options))
    return selections, forecast


def confirm(row, library, fixed):
    controller, scene, state, prefix = checkpoint(row)
    started = process_time()
    stock, generated, aliases = combined_menu(controller, state, library)
    menu = stock | generated
    menu_cpu = process_time() - started
    started = process_time()
    stock_mpc = mpc_action(controller, state, stock)
    stock_mpc_cpu = process_time() - started
    started = process_time()
    mpc = mpc_action(controller, state, menu)
    mpc_cpu = process_time() - started
    selections, forecast = public_selection(controller, state, stock, generated, aliases, library, stock_mpc.name)
    started = process_time()
    old_menu, inputs, old_mpc_cpu, _ = decision_inputs(controller, state, prefix, row, scans.CALIBRATION)
    inputs["first_required_bin"] = state.first_required_bin
    existing = choice_from_predictions(inputs, predict_deltas(MODELS, scans.retrained_inputs(inputs)))[0]
    b_cpu = process_time() - started - old_mpc_cpu
    fixed_name = fixed[row["family"]]
    fixed_resolved = aliases.get(fixed_name, fixed_name)
    arms = {
        "mpc": (mpc, mpc_cpu, len(menu)), "stock_mpc": (stock_mpc, stock_mpc_cpu, len(stock)),
        "best_fixed": (menu.get(fixed_resolved, mpc), None, 1),
        "existing_b": (old_menu[existing], b_cpu, len(stock)), **selections,
    }
    bank = scans.paired_futures(scene, row, TEST_REPEATS, 18283)
    cached = {name: [trial(controller, state, prefix, s, a) for s in bank] for name, a in
              {a.name: a for a, _, _ in arms.values()}.items()}
    return row | {
        "root": {"station": state.s, "frame": state.frame, "missed": state.missed,
                 "query": state.query_bin(), "association": prefix.packets[-1]["association"]},
        "arms": {arm: {"action": a.name, "outcomes": cached[a.name], "cpu_seconds": elapsed,
                       "candidates": count} for arm, (a, elapsed, count) in arms.items()},
        "fixed_fallback": fixed_resolved not in menu,
        "common_menu_cpu_seconds": menu_cpu,
        "aliases": aliases,
        "forecast": forecast,
    }
