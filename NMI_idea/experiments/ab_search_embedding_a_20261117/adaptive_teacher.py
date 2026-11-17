"""Offline motion discovery and a separate public-history decision rule."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from adaptive_controller import CENTER_MM, FreeCandidate, final_metrics
from recovery_world import Scene, noise_tape


def utility(outcome):
    return (outcome["corrected_complete"] - 1.25 * outcome["final_wrong"]
            - .002 * outcome["tcp_mm"] - .02 * outcome["seconds"])


def evaluate(controller, root, action, worlds, *, open_blocks=0, detailed=False):
    """Counterfactual outcomes. The starting state contains only public evidence."""
    outcomes, traces = [], []
    for scene in worlds:
        state = replace(root, belief=root.belief.copy())
        record = controller.virtual_record(state, detailed=detailed)
        if action is not None:
            controller.execute(scene, state, record, action)
        controller.tail(scene, state, record, open_blocks=open_blocks)
        outcome = final_metrics(state, record, sampled_latent=scene.latent, root_frame=root.frame)
        outcome["turn_degs"] = sum(step["executed_turn_deg"] for step in record.decisions)
        outcome["new_centered_mistake"] = any(
            row["accepted_slot"] >= 0 and row["roles_truth"][row["accepted_slot"]] != 0
            and abs(row["y"][row["accepted_slot"]][0]) <= CENTER_MM for row in record.packets)
        outcomes.append(outcome)
        if detailed:
            traces.append(record)
    return outcomes, traces


def expected(outcomes):
    return {field: float(np.mean([row[field] for row in outcomes]))
            for field in ("corrected_complete", "final_wrong", "new_centered_mistake",
                          "tcp_mm", "turn_degs", "seconds")}


def posterior_worlds(state, count, seed):
    """Stratify draws from p(static geometry, shadow, clutter | public history)."""
    rng = np.random.default_rng([seed, state.frame])
    mass = np.cumsum(state.belief)
    mass[-1] = 1.
    index = np.searchsorted(mass, (np.arange(count) + rng.random(count)) / count, side="right")
    return [Scene(int(j), noise_tape(rng)) for j in index]


def public_scores(controller, state, *, samples, remaining_open):
    """Choose again from the observed state at each of two decision times.

    Candidate scores use the complete shared MPC tail. This search performs
    successive decisions, with one explicit intervention in each forecast.
    """
    menu = controller.options_shared(state)
    if not menu:
        return menu, None, []
    incumbent = controller.mpc(state, menu, False, active=True, informative=True)
    # Draw once per decision; every candidate sees precisely the same bank.
    worlds = posterior_worlds(state, samples, 751_239)
    scores = []
    for name, action in menu.items():
        outcomes, _ = evaluate(controller, state, action, worlds, open_blocks=remaining_open)
        metrics = expected(outcomes)
        scores.append({"action": name, "expected": metrics, "utility": utility(metrics)})
    return menu, incumbent.name, scores


def choose_public(controller, state, *, samples, remaining_open):
    started = perf_counter()
    menu, incumbent, scores = public_scores(controller, state, samples=samples, remaining_open=remaining_open)
    if incumbent is None:
        return None, {"frame": state.frame, "candidates": [], "cpu_wall_s": perf_counter() - started}
    winner = max(scores, key=lambda row: (row["utility"], row["action"] == incumbent))
    return menu[winner["action"]], {"frame": state.frame, "picked": winner["action"],
                                    "mpc": incumbent, "samples": samples, "candidates": scores,
                                    "cpu_wall_s": perf_counter() - started}


def fixed_worlds(latent, count, seed, purpose):
    return [Scene(latent, noise_tape(np.random.default_rng([seed, purpose, i]))) for i in range(count)]


def cem_discover(controller, state, latent, seed, *, population, iterations, samples):
    """Privileged search of a spline endpoint and its interior waypoint."""
    limits = np.array([[-6., -8.5, -3.7], [7.8, 8.5, 3.7]])
    worlds = fixed_worlds(latent, samples, seed, 471)
    rng = np.random.default_rng([seed, 853])
    nominal = controller.choose_mpc(state, expanded=True, open_decision=True)
    base, _ = evaluate(controller, state, nominal, worlds, open_blocks=1)
    baseline = expected(base)
    rows, feasible = [], []
    for anchor in ([-3.5, -3., -1.], [-3.5, 3., 1.], [4.5, 0., 0.]):
        mean, scale = np.asarray(anchor), np.array([2.8, 3.1, 1.7])
        for iteration in range(iterations):
            draws = np.clip(rng.normal(mean, scale, (population, 3)), limits[0], limits[1])
            ranked = []
            for ds, dx, bow in draws:
                action = controller.free_candidate(state, "cem", ds, dx, bow)
                requested = [float(ds), float(dx), float(bow)]
                if action is None:
                    rows.append({"request": requested, "valid": False})
                    continue
                duplicate = next((i for i, (other, _) in enumerate(feasible)
                                  if np.max(abs(other.s - action.s)) < .05
                                  and np.max(abs(other.lateral.x - action.lateral.x)) < .05), None)
                if duplicate is not None:
                    row = feasible[duplicate][1]
                    rows.append({"request": requested, "valid": True, "duplicate_path_of": duplicate,
                                 "actual_scored_request": row["request"]})
                else:
                    outcomes, _ = evaluate(controller, state, action, worlds, open_blocks=1)
                    summary = expected(outcomes)
                    row = {"request": requested, "valid": True, "expected": summary,
                           "actual_goal": [float(action.s[-1]), action.lateral.goal],
                           "utility": utility(summary), "scored_path": len(feasible), "replicas": outcomes}
                    rows.append(row)
                    feasible.append((action, row))
                ranked.append((row["utility"], requested))
            if ranked:
                elite = np.asarray([x[1] for x in sorted(ranked, reverse=True)[:max(2, population // 4)]])
                mean = .3 * mean + .7 * elite.mean(axis=0)
                scale = .3 * scale + .7 * (elite.std(axis=0) + .25)
    ranked = sorted(feasible, key=lambda item: item[1]["utility"], reverse=True)
    return nominal, baseline, ranked, rows


def fit_nominal_primitive(primitive, examples):
    """Fit the open-loop proposal visible when the decision was taken."""
    features, coefficients = [], []
    for example in examples:
        nominal = np.asarray(example["nominal_x_vx"])
        x, v = nominal.T
        # Every demonstration begins at a block boundary, with zero lateral
        # speed. The analytic basis still enforces the actual initial speed.
        features.append([x[-1] - x[0], example["request"][2]])
        coefficients.append(primitive.fit(x, v))
    if len(features) < 2 or np.linalg.matrix_rank(features) < 2:
        return None
    return np.linalg.lstsq(features, coefficients, rcond=None)[0]


def replace_shape(controller, state, action):
    if not isinstance(action, FreeCandidate):
        return action, False
    substitute = controller.free_candidate(state, action.name, *action.request, renderer="fit")
    return (substitute, True) if substitute is not None else (action, False)
