"""Paired closed-loop evaluation with equal feedback and execution limits."""

from collections import Counter
from dataclasses import replace
from time import process_time

import numpy as np

import probe as p


ONE_PROBE = ("old", "random", "evidence", "contrast")
ARMS = ("mpc", "fixed_single", "fixed", *ONE_PROBE, "repeat", "child", "program",
        "same_first_fixed", "shuffled_branch", "rollout_one", "rollout_program")
PRIMARY = "new_image_and_complete"


class DriftingScene(p.base.PerturbedScene):
    """Same anatomy and noise tapes, with an unmodelled smooth physical displacement."""

    def shifted_at(self, frame):
        phase = float(frame) * .1 * (2. * np.pi / 3.0)
        return replace(self, lateral_jitter=self.lateral_jitter + 1.8 * np.sin(phase),
                       depth_jitter=self.depth_jitter + .65 * np.sin(phase + .5))


class DriftWorld(p.base.StartupWorld):
    def observe(self, scene, frame, s, x):
        if isinstance(scene, DriftingScene):
            scene = scene.shifted_at(frame)
        # Use StartupWorld's actual field-of-view and acoustic masks.
        return super().observe(scene, frame, s, x)


def checkpoint(row):
    if row.get("setting", "main") != "drift":
        return p.base.at_handoff(row)
    # Give every policy the same streamed prefix in a moving world. The public
    # filter stays the same static-hypothesis filter used in training.
    from copy import copy
    controller = copy(p.base.scans.CONTROLLERS[row["family"]])
    controller.world = DriftWorld(controller.world.family, controller.world.reference)
    from benchmark_arbitrary_start import draw_scene, stress_tape
    raw = draw_scene(controller, row["seed"])
    scene = DriftingScene(**vars(replace(raw, tape=stress_tape(raw.tape, row))))
    root, prefix = controller.initial_scan(scene, start_s=row["start_s"], start_x=row["start_x"])
    for _ in range(row["prefix_blocks"]):
        action = controller.choose_mpc(root, expanded=False)
        if action is None:
            raise ValueError("drift initial scan lost its feasible incumbent")
        controller.execute(scene, root, prefix, action)
    if root.frame != row["handoff_frame"] or not controller.can_continue(root):
        raise ValueError("drift case is missing its scan checkpoint")
    started = process_time()
    menu, aliases = controller.expanded_menu(root)
    anchor = p.base.generator.mpc_action(controller, root, menu).name
    return controller, scene, root, prefix, menu, aliases, anchor, process_time() - started


def request_menu(menu, aliases):
    names = set(p.base.STOCK_REQUESTS) | set(p.base.NEW_REQUESTS)
    return {name: menu[aliases.get(name, name)] for name in sorted(names) if aliases.get(name, name) in menu}


def develop_fixed(row):
    """Exhaust the same first and second command families on development scenes only."""
    c, truth, root, prefix, root_menu, aliases, _, _ = p.base.at_handoff(row)
    gate = p.deadline(root)
    bank = p.base.scans.paired_futures(truth, row, 3, 18431)
    first_menu = request_menu(root_menu, aliases)
    outcomes = {}
    for future in bank:
        cache = {}
        for first in (p.HANDOFF, *sorted(set(p.base.STOCK_REQUESTS) | set(p.base.NEW_REQUESTS))):
            allowed_seconds = (p.HANDOFF,) if first == p.HANDOFF else (p.HANDOFF, *p.base.NEW_REQUESTS)
            action = first_menu.get(first)
            key = action.name if action is not None else p.HANDOFF
            if key not in cache:
                after, log = p.run_motion(c, root, prefix, future, action)
                cache[key] = {}
                for second in (p.HANDOFF, *p.base.NEW_REQUESTS):
                    follow = p.fixed_action(c, after, second) if action is not None and c.can_continue(after) else None
                    if follow is None and p.HANDOFF in cache[key]:
                        result = cache[key][p.HANDOFF]
                    else:
                        end, log_end = p.run_motion(c, after, log, future, follow)
                        result = p.actual_outcome(c, end, log_end, future, root, gate)
                    cache[key][second] = result
            for second in allowed_seconds:
                result = cache[key][second]
                outcomes.setdefault(f"{first}/{second}", []).append(
                    [result[k] for k in (PRIMARY, "final_wrong", "tcp_mm", "seconds", "turn_degs")])
    return row | {"outcomes": outcomes, "available_first": list(first_menu)}


def choose_fixed(rows):
    def select(table, choices):
        return max(choices, key=lambda key: (table[key][0], *(-table[key][1:]), key))

    selected = {}
    for family in sorted({row["family"] for row in rows}):
        values = [row["outcomes"] for row in rows if row["family"] == family]
        table = {key: np.mean([replay for case in values for replay in case[key]], axis=0)
                 for key in sorted(values[0])}
        single = [key for key in table if key.endswith(f"/{p.HANDOFF}")]
        single_key = select(table, single)
        all_key = select(table, table)
        selected[family] = {
            "single": single_key.split("/"), "program": all_key.split("/"),
            "single_development_means": table[single_key].tolist(),
            "program_development_means": table[all_key].tolist(), "sequences": len(table),
        }
    return selected


def rollout_choices(c, root, prefix, menu, aliases, anchor, gate, seed):
    """Matched open-loop first requests, with identical posterior worlds for both horizons."""
    candidates = p.comparison_pool(menu, aliases, anchor)
    bank = p.action_bank(root, prefix.packets, seed, count=8)
    t0 = process_time()
    single = {p.HANDOFF: np.mean([
        p.value_after_action(c, root, world, menu[anchor], gate) for world in bank
    ], axis=0)}
    afters = {}
    for name in candidates:
        afters[name] = [p.run_motion(c, root, p.starting_record(c, root), world, menu[name]) for world in bank]
        single[name] = np.mean([p.simulation_outcome(c, state, log, world, root, gate)
                                for (state, log), world in zip(afters[name], bank, strict=True)], axis=0)
    first_one = max(single, key=lambda name: (p.rating(single[name]), name == p.HANDOFF))
    single_cpu = process_time() - t0
    conditional = {p.HANDOFF: single[p.HANDOFF]}
    for name in candidates:
        trials = []
        for (state, record), world in zip(afters[name], bank, strict=True):
            options, incumbent, inputs = p.second_inputs(c, state, record, gate)
            _, a = p.choose_second(options, incumbent, inputs, p.MODEL_BANK["child"])
            end, log = p.run_motion(c, state, record, world, a)
            trials.append(p.simulation_outcome(c, end, log, world, root, gate))
        conditional[name] = np.mean(trials, axis=0)
    first_two = max(conditional, key=lambda name: (p.rating(conditional[name]), name == p.HANDOFF))
    return (first_one, first_two), (single_cpu, process_time() - t0)


def first_actions(c, root, prefix, menu, aliases, anchor, fixed, baseline_cpu):
    started = process_time()
    inputs = p.base.policy_inputs(c, root, prefix, menu, anchor)
    input_time = process_time() - started
    expected = {}
    cpu = {"mpc": baseline_cpu}
    for arm in (*ONE_PROBE, "program"):
        started = process_time()
        choice = p.scored_choice(inputs, p.MODEL_BANK[arm])
        expected[arm] = p.HANDOFF if choice == anchor else choice
        cpu[arm] = baseline_cpu + input_time + process_time() - started
    first_menu = request_menu(menu, aliases)
    for arm, selection in (("fixed_single", fixed["single"]), ("fixed", fixed["program"])):
        name = selection[0]
        expected[arm] = first_menu[name].name if name in first_menu else p.HANDOFF
        cpu[arm] = None
    expected["mpc"] = p.HANDOFF
    expected["same_first_fixed"] = expected["shuffled_branch"] = expected["program"]
    expected["repeat"] = expected["child"] = expected["contrast"]
    return expected, cpu


def subsequent(c, root, prefix, menu, first, scene, gate):
    states = {}
    for name in sorted(set(first.values())):
        a = menu.get(name) if name != p.HANDOFF else None
        reached, record = p.run_motion(c, root, prefix, scene, a)
        states[name] = reached, record
    choices, seconds, alternatives = {}, {}, {}
    cache = {}
    for arm, trained in (("repeat", "contrast"), ("child", "child"), ("program", "child"),
                         ("rollout_program", "child")):
        name = first[arm]
        if name == p.HANDOFF:
            choices[arm], seconds[arm], alternatives[arm] = (p.HANDOFF, None), 0., 0
            continue
        reached, record = states[name]
        if name not in cache:
            t0 = process_time()
            menu_after, anchor, inputs = p.second_inputs(c, reached, record, gate)
            cache[name] = menu_after, anchor, inputs, process_time() - t0
        menu_after, anchor, inputs, setup = cache[name]
        t0 = process_time()
        choices[arm] = p.choose_second(menu_after, anchor, inputs, p.MODEL_BANK[trained])
        seconds[arm] = setup + process_time() - t0
        alternatives[arm] = max(0, len(menu_after) - 1)
    return states, choices, seconds, alternatives


def test_case(job):
    row, fixed = job
    c, truth, root, prefix, menu, aliases, anchor, baseline_cpu = checkpoint(row)
    gate = p.deadline(root)
    first, first_cpu = first_actions(c, root, prefix, menu, aliases, anchor, fixed, baseline_cpu)
    (look_one, look_two), (cpu_one, cpu_two) = rollout_choices(
        c, root, prefix, menu, aliases, anchor, gate, row["seed"] + 400_000)
    first.update(rollout_one=look_one, rollout_program=look_two)
    first_cpu.update(rollout_one=baseline_cpu + cpu_one, rollout_program=baseline_cpu + cpu_two)
    precommitted = p.HANDOFF
    precommit_seconds = 0.
    if first["program"] != p.HANDOFF:
        t0 = process_time()
        precommit_worlds = p.action_bank(root, prefix.packets, row["seed"] + 18433, count=8)
        precommitted = p.simulate_precommit(c, root, menu[first["program"]], precommit_worlds, gate)
        precommit_seconds = process_time() - t0
    source = p.base.scans.paired_futures(truth, row, 8, 18432)
    realized = [subsequent(c, root, prefix, menu, first, scene, gate) for scene in source]
    tape_order = np.random.default_rng([row["seed"], 18434]).permutation(len(source))
    partners = dict(zip(tape_order, np.roll(tape_order, 1), strict=True))
    outcomes = {name: [] for name in ARMS}
    for index, (future, (states, branches, branch_cpu, alternatives)) in enumerate(zip(source, realized, strict=True)):
        tails = {}
        for name in ARMS:
            lead = first[name]
            reached, record = states[lead]
            if name in branches:
                second_id, a = branches[name]
                requested = second_id
            else:
                requested = {"fixed_single": fixed["single"][1], "fixed": fixed["program"][1],
                             "same_first_fixed": precommitted,
                             "shuffled_branch": realized[partners[index]][1]["program"][0]}.get(name, p.HANDOFF)
                a = p.fixed_action(c, reached, requested) if lead != p.HANDOFF and c.can_continue(reached) else None
                second_id = requested if a is not None else p.HANDOFF
            key = lead, second_id
            if key not in tails:
                end, log = p.run_motion(c, reached, record, future, a)
                result = p.actual_outcome(c, end, log, future, root, gate)
                suffix = [packet for packet in record.packets if packet["frame"] > root.frame]
                tails[key] = result | {
                    "first": lead, "second": second_id, "probes": int(lead != p.HANDOFF) + int(a is not None),
                    "first_exit_frame": reached.frame, "handoff_frame": end.frame,
                    "s_after_first": reached.s, "x_after_first": reached.x,
                    "vs_after_first": reached.vs, "vx_after_first": reached.vx,
                    "remaining_mm_after_first": p.MAX_TRAVEL - reached.travel,
                    "empty_after_first": sum(not packet["y"] for packet in suffix),
                    "seen_after_first": sum(bool(packet["y"]) for packet in suffix),
                    "posterior_tv_after_first": float(np.abs(root.belief - reached.belief).sum() / 2),
                    "first_probe_render": getattr(menu.get(lead), "renderer", None),
                    "second_probe_render": getattr(a, "renderer", None),
                }
            if name in ("fixed", "fixed_single", "shuffled_branch"):
                decision_cpu = None  # Fixed commands and observation ablation are diagnostic controls.
            elif name in ("repeat", "child"):
                decision_cpu = first_cpu["contrast"] + branch_cpu[name]
            elif name == "same_first_fixed":
                decision_cpu = first_cpu["program"] + precommit_seconds
            else:
                decision_cpu = first_cpu[name] + branch_cpu.get(name, 0.)
            outcomes[name].append(tails[key] | {
                "decision_cpu_seconds": decision_cpu, "requested_second": requested,
                "unavailable_second": requested != p.HANDOFF and second_id == p.HANDOFF,
                "feasible_second_alternatives": alternatives.get(name),
            })
    return row | {
        "root": {"s": root.s, "x": root.x, "frame": root.frame, "budget_mm": p.MAX_TRAVEL - root.travel,
                 "detected": bool(prefix.packets[-1]["y"]), "identity_slot": root.slot,
                 "prior_agreement": float(max(root.belief)), "query": root.query_bin()},
        "precommit_second": precommitted, "outcomes": outcomes,
    }


def branch_counts(rows, arm):
    counts = Counter(trial["probes"] for row in rows for trial in row["outcomes"][arm])
    return {"counts": {str(j): counts[j] for j in (0, 1, 2)},
            "roots_with_different_second_choices": sum(
                len({trial["second"] for trial in row["outcomes"][arm]}) > 1 for row in rows)}
