"""Matched first choices and next-decision labels from simulated probe histories."""

import numpy as np

import probe as p


def paired_deltas(values, anchor_index):
    means = np.mean(values, axis=1)
    return means - means[anchor_index]


def collect_first(archived):
    c, _, root, prefix, menu, aliases, anchor, _ = p.base.at_handoff(archived)
    if archived["incumbent"] != anchor:
        raise ValueError("the archived and current MPC anchors differ")
    gate = p.deadline(root)
    pool = p.comparison_pool(menu, aliases, anchor)
    evidence = archived["nominated"]
    rng = np.random.default_rng([archived["seed"], 18411])
    random = rng.permutation(pool)[:p.base.PROPOSALS].tolist()
    # Discovery comparisons and final training labels use separate stochastic outcomes.
    screening = p.action_bank(root, prefix.packets, archived["seed"] + 1_000_000, count=4)
    screened = {name: np.mean([p.value_after_action(c, root, world, menu[name], gate)
                              for world in screening], axis=0) for name in pool}
    lead = max(pool, key=lambda name: (p.rating(screened[name]), name))
    worse = [name for name in pool if screened[name][0] + 1.e-9 < screened[lead][0]]
    nearby = sorted(worse, key=lambda name: (abs(menu[name].distance - menu[lead].distance), name))[:3]
    contrast = sorted(nearby, key=lambda name: (p.rating(screened[name]), name))[:1]
    selected = list(dict.fromkeys([lead, *contrast, *evidence, *pool]))[:p.base.PROPOSALS]
    nominations = {"random": random, "evidence": evidence, "contrast": selected}
    names = sorted({anchor} | set().union(*nominations.values()))
    bank = p.action_bank(root, prefix.packets, archived["seed"])
    outcomes = {}
    for name in names:
        if name in archived["outcomes"]:
            outcomes[name] = [[trial[key] for key in p.TARGETS] for trial in archived["outcomes"][name]]
        else:
            outcomes[name] = [p.value_after_action(c, root, world, menu[name], gate) for world in bank]
    info = p.base.policy_inputs(c, root, prefix, menu, anchor)
    inputs = info["features"][[info["names"].index(name) for name in names]]
    values = np.asarray([outcomes[name] for name in names])
    return {key: archived[key] for key in ("seed", "family", "context")} | {
        "names": names, "anchor": anchor, "nominations": nominations, "features": inputs,
        "rollouts": values, "targets": paired_deltas(values, names.index(anchor)),
        "screened": screened, "contrast_partner": contrast[0] if contrast else None,
    }


def collect_children(row):
    c, _, root, prefix, menu, aliases, anchor, _ = p.base.at_handoff(row)
    gate = p.deadline(root)
    pool = p.comparison_pool(menu, aliases, anchor)
    root_choice = p.choose_first(c, root, prefix, menu, anchor, p.MODEL_BANK["contrast"])
    if root_choice == p.HANDOFF:
        root_choice = anchor
    evidence = p.root_evidence(c, root, menu, pool, row["seed"] + 18420)
    rng = np.random.default_rng([row["seed"], 18421])
    first = list(dict.fromkeys([root_choice, *evidence[:1], *rng.permutation(pool).tolist()]))[:3]
    outer_bank = p.action_bank(root, prefix.packets, row["seed"] + 18422, count=len(first))
    children = []
    for index, (name, world) in enumerate(zip(first, outer_bank, strict=True)):
        # Public inputs require its actual packets, while truth labels are recomputed below
        # on newly drawn posterior worlds at this later observation history.
        state, record = p.run_motion(c, root, prefix, world, menu[name])
        next_menu, next_anchor, info = p.second_inputs(c, state, record, gate)
        if info is None:
            continue
        bank = p.action_bank(state, record.packets, row["seed"] + 1_300_000 + index * 100_000, count=6)
        values = np.asarray([[p.value_after_action(c, state, sample, a, gate) for sample in bank]
                             for a in next_menu.values()])
        children.append({
            "first": name, "names": list(next_menu), "anchor": next_anchor, "features": info["features"],
            "rollouts": values, "targets": paired_deltas(values, info["names"].index(next_anchor)),
            "frame": state.frame, "s": state.s, "x": state.x, "remaining_mm": p.MAX_TRAVEL - state.travel,
            "empty_packets": sum(not packet["y"] for packet in record.packets if packet["frame"] > root.frame),
        })
    return row | {"children": children}


def collect_program(job):
    row, first_labels = job
    c, _, root, prefix, menu, _, anchor, _ = p.base.at_handoff(row)
    gate = p.deadline(root)
    bank = p.action_bank(root, prefix.packets, row["seed"])
    names = [anchor, *first_labels["nominations"]["contrast"]]
    features = np.asarray(first_labels["features"])[[first_labels["names"].index(name) for name in names]]
    outcomes = [first_labels["rollouts"][first_labels["names"].index(anchor)]]
    traces = {}
    for first in names[1:]:
        trials, chosen = [], []
        for world in bank:
            state, record = p.run_motion(c, root, p.starting_record(c, root), world, menu[first])
            next_menu, next_anchor, info = p.second_inputs(c, state, record, gate)
            decision, a = p.choose_second(next_menu, next_anchor, info, p.MODEL_BANK["child"])
            done, log = p.run_motion(c, state, record, world, a)
            trials.append(p.simulation_outcome(c, done, log, world, root, gate))
            chosen.append(decision)
        outcomes.append(trials)
        traces[first] = chosen
    return row | {"anchor": anchor, "names": names, "features": features, "rollouts": outcomes,
                  "targets": paired_deltas(outcomes, 0), "second_choices": traces}
