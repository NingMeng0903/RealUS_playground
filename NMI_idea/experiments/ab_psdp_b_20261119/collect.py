"""Collect real successor histories, then pair the two continuations at the same exit."""

import numpy as np

from policy import MODELS, p, scores


LABEL_TAPES = 8
CHILD_TAPES = 8
ROLL_INS = 4


def scenario_rows(start, per_cell, *, setting="main"):
    return [p.base.scenario(cell, start + index * 10_000 + j) | {"setting": setting}
            for index, cell in enumerate(p.base.CELLS) for j in range(per_cell)]


def collect_successors(row):
    c, truth, root, prefix, menu, aliases, anchor, _ = p.base.at_handoff(row)
    gate = p.deadline(root)
    info = p.base.policy_inputs(c, root, prefix, menu, anchor)
    ranked = [p.scored_choice(info, MODELS[key]) for key in ("contrast", "legacy_program")]
    pool = p.comparison_pool(menu, aliases, anchor)
    evidence = p.root_evidence(c, root, menu, pool, row["seed"] + 24101)[0]
    rng = np.random.default_rng([row["seed"], 24102])
    # Include previously deployed first choices and feasible exploratory motions.
    names = list(dict.fromkeys([*ranked, evidence, *rng.permutation(list(menu)).tolist()]))[:ROLL_INS]
    truth_tapes = p.base.scans.paired_futures(truth, row, 2, 24103)
    children = []
    for i, name in enumerate(names):
        # Generate the successor by physical execution in the original hidden
        # world's simulator. The controller receives only its emitted observations.
        state, record = p.run_motion(c, root, prefix, truth_tapes[i % len(truth_tapes)], menu[name])
        options, incumbent, public = p.second_inputs(c, state, record, gate)
        if public is None:
            continue
        seed = int(np.random.SeedSequence([row["seed"], 24104, i]).generate_state(1)[0])
        bank = p.action_bank(state, record.packets, seed, count=CHILD_TAPES)
        outcomes = np.asarray([[p.value_after_action(c, state, w, a, gate) for w in bank]
                               for a in options.values()])
        means = outcomes.mean(axis=1)
        children.append({
            "first": name, "frame": state.frame, "s": state.s, "x": state.x,
            "vs": state.vs, "vx": state.vx, "remaining_mm": p.MAX_TRAVEL - state.travel,
            "bayes": state.belief, "roll_in_tape": i % len(truth_tapes),
            "post_observations": [packet["y"] for packet in record.packets if packet["frame"] > root.frame],
            "names": list(options), "anchor": incumbent, "features": public["features"],
            "outcomes": outcomes, "targets": means - means[public["names"].index(incumbent)],
        })
    return row | {"roll_in": names, "children": children}


def nominate(c, root, prefix, menu, aliases, anchor, info, seed, gate):
    pool = p.comparison_pool(menu, aliases, anchor)
    bank = p.action_bank(root, prefix.packets, seed + 24105, count=4)
    screen = {name: np.mean([p.value_after_action(c, root, w, menu[name], gate) for w in bank], axis=0)
              for name in pool}
    best = max(pool, key=lambda name: (p.rating(screen[name]), name))
    weaker = [name for name in pool if screen[name][0] < screen[best][0] - 1.e-9]
    close = sorted(weaker, key=lambda name: (abs(menu[name].distance - menu[best].distance), name))[:3]
    pair = min(close, key=lambda name: (p.rating(screen[name]), name)) if close else None
    evidence = p.root_evidence(c, root, menu, pool, seed + 24106)
    contrast = list(dict.fromkeys([best, *([pair] if pair is not None else []), *evidence]))[:4]
    incumbents = [p.scored_choice(info, MODELS[key]) for key in ("contrast", "legacy_program")]
    predicted = scores(info, "contrast")
    indices = sorted(range(len(predicted)), key=lambda i: p.rating(predicted[i]), reverse=True)
    plausible = [info["names"][i] for i in indices[:3]]
    rng = np.random.default_rng([seed, 24107])
    selected = list(dict.fromkeys([anchor, *contrast, *incumbents, *plausible, *rng.permutation(list(menu)).tolist()]))[:9]
    return selected, {"best": best, "contrast_partner": pair, "contrast_four": contrast,
                      "screen_replays": len(pool) * len(bank)}


def split_continuations(c, root, after, record, world, gate):
    direct = p.simulation_outcome(c, after, record, world, root, gate)
    options, incumbent, public = p.second_inputs(c, after, record, gate)
    name, action = p.choose_second(options, incumbent, public, MODELS["child"])
    if action is None:
        return direct, direct, name
    end, log = p.run_motion(c, after, record, world, action)
    return direct, p.simulation_outcome(c, end, log, world, root, gate), name


def collect_pairs(row):
    c, _, root, prefix, menu, aliases, anchor, _ = p.base.at_handoff(row)
    gate = p.deadline(root)
    info = p.base.policy_inputs(c, root, prefix, menu, anchor)
    selected, selection = nominate(c, root, prefix, menu, aliases, anchor, info, row["seed"], gate)
    bank = p.action_bank(root, prefix.packets, row["seed"] + 812001, count=LABEL_TAPES)
    indices = [info["names"].index(name) for name in selected]
    baseline, direct, adaptive, traces = [], [], [], {}
    for world in bank:
        baseline.append(p.simulation_outcome(c, root, p.starting_record(c, root), world, root, gate))
    for name in selected:
        one, two, chosen, exits = [], [], [], []
        for world in bank:
            after, record = p.run_motion(c, root, p.starting_record(c, root), world, menu[name])
            value_direct, value_child, branch = split_continuations(c, root, after, record, world, gate)
            one.append(value_direct)
            two.append(value_child)
            chosen.append(branch)
            exits.append([after.s, after.x, after.vs, after.vx, after.frame, after.travel])
        direct.append(one)
        adaptive.append(two)
        traces[name] = {"branch": chosen, "shared_exit": exits}
    # The motion chosen by MPC then resumed at the next frame is the same as
    # handing control to complete MPC now. A child continuation of that motion
    # remains a separate candidate and its label is not overwritten.
    np.testing.assert_allclose(direct[selected.index(anchor)], baseline, atol=1.e-10, rtol=0.)
    return row | {
        "names": selected, "anchor": anchor, "features": info["features"][indices],
        "baseline": baseline, "direct": direct, "adaptive": adaptive, "traces": traces,
        "selection": selection,
    }
