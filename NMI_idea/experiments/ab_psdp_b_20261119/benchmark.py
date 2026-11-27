"""Execute each frozen policy on shared roots and time-indexed disturbance tapes."""

from hashlib import sha256
import json
from time import process_time

from policy import (
    DIRECT, OBSERVE, STOP, MODELS, full_choice, gated_reference, legacy_eval, p, scores,
    select_direct, select_modes,
)


ARMS = (
    "mpc", "fixed_single", "random_data", "evidence_data", "published_evidence", "contrast_data",
    "legacy_program", "legacy_precommit", "contrast_child", "matched_q1", "matched_q1_child",
    "matched_q2_child", "matched_mixed_child", "psdp_mixed", "psdp", "psdp_direct", "psdp_precommit",
)
PRIMARY = legacy_eval.PRIMARY
TEST_TAPES = 8
TEST_PURPOSE = 24125


def make_plans(c, root, prefix, menu, aliases, anchor, fixed, baseline_cpu):
    t0 = process_time()
    info = p.base.policy_inputs(c, root, prefix, menu, anchor)
    feature_cpu = process_time() - t0
    direct = {"random_data": "random", "evidence_data": "evidence", "contrast_data": "contrast",
              "matched_q1": "q1", "published_evidence": "published_evidence"}
    mandatory = {"matched_q1_child": "q1", "matched_q2_child": "q2", "matched_mixed_child": "mixed_q2"}
    moves = legacy_eval.request_menu(menu, aliases)
    plans = {"mpc": STOP, "fixed_single": (moves[fixed].name, DIRECT) if fixed in moves else STOP}
    timings = {"mpc": baseline_cpu, "fixed_single": None}
    base_cost = baseline_cpu + feature_cpu
    for tag, model in direct.items():
        started = process_time()
        plans[tag] = select_direct(info, scores(info, model))
        timings[tag] = base_cost + process_time() - started
    lead, _ = plans["contrast_data"]
    plans["contrast_child"] = (lead, OBSERVE) if lead != p.HANDOFF else STOP
    timings["contrast_child"] = timings["contrast_data"]
    for tag, model in mandatory.items():
        started = process_time()
        values = scores(info, model)
        if model == "q1":
            values[info["names"].index(anchor)] = 0.
        plans[tag] = select_modes(info, None, values)
        timings[tag] = base_cost + process_time() - started
    started = process_time()
    plans["legacy_program"] = gated_reference(info, "legacy_program")
    timings["legacy_program"] = base_cost + process_time() - started
    for tag, model in (("psdp", "residual"), ("psdp_mixed", "mixed_residual")):
        started = process_time()
        plans[tag] = full_choice(info, residual=model)
        timings[tag] = base_cost + process_time() - started
    plans["legacy_precommit"] = plans["legacy_program"]
    plans["psdp_precommit"] = plans["psdp"]
    plans["psdp_direct"] = (plans["psdp"][0], DIRECT)
    timings["legacy_precommit"] = timings["legacy_program"]
    timings["psdp_precommit"] = timings["psdp_direct"] = timings["psdp"]
    return plans, timings


def precommit(c, root, prefix, menu, plan, gate, seed):
    lead, adapt = plan
    if not adapt or lead == p.HANDOFF:
        return p.HANDOFF, 0.
    start = process_time()
    worlds = p.action_bank(root, prefix.packets, seed)
    chosen = p.simulate_precommit(c, root, menu[lead], worlds, gate)
    return chosen, process_time() - start


def exit_and_history(state, record, root):
    public_packets = [{field: packet[field] for field in ("frame", "s", "x", "q", "y", "association")}
                      for packet in record.packets if packet["frame"] > root.frame]
    kinematics = [state.s, state.x, state.vs, state.vx, state.frame, state.travel]
    digest = sha256(json.dumps([kinematics, public_packets], separators=(",", ":")).encode()).hexdigest()
    return {"exit": kinematics, "history_sha256": digest, "empty_frames": sum(not v["y"] for v in public_packets)}


def evaluate_case(job):
    row, fixed = job
    c, truth, root, prefix, menu, aliases, anchor, cpu = legacy_eval.checkpoint(row)
    gate = p.deadline(root)
    plans, decision_time = make_plans(c, root, prefix, menu, aliases, anchor, fixed, cpu)
    # Equal planning samples for the old and new fixed-conditional controls.
    # Reuse exactly the same precommit result if the chosen first motion matches.
    forecast_cache, prescribed = {}, {}
    for tag in ("legacy_precommit", "psdp_precommit"):
        first, adaptive = plans[tag]
        if (first, adaptive) not in forecast_cache:
            forecast_cache[first, adaptive] = precommit(
                c, root, prefix, menu, plans[tag], gate, row["seed"] + 913001)
        prescribed[tag], elapsed = forecast_cache[first, adaptive]
        decision_time[tag] += elapsed
    trajectories = {arm: [] for arm in ARMS}
    leading = sorted({name for name, _ in plans.values()})
    for scene in p.base.scans.paired_futures(truth, row, TEST_TAPES, TEST_PURPOSE):
        exits = {name: p.run_motion(c, root, prefix, scene, menu.get(name)) for name in leading}
        logs = {name: exit_and_history(state, record, root) for name, (state, record) in exits.items()}
        public, decisions, evaluated = {}, {}, {}
        for arm in ARMS:
            first, adaptive = plans[arm]
            state, record = exits[first]
            request, action, extra, alternatives = p.HANDOFF, None, 0., None
            if adaptive:
                if arm in prescribed:
                    request = prescribed[arm]
                    if c.can_continue(state):
                        action = p.fixed_action(c, state, request)
                else:
                    child = "legacy_child" if arm == "legacy_program" else "child"
                    if first not in public:
                        start = process_time()
                        public[first] = p.second_inputs(c, state, record, gate), process_time() - start
                    tree, prep = public[first]
                    alternatives = max(0, len(tree[0]) - 1)
                    key = first, child
                    if key not in decisions:
                        start = process_time()
                        decisions[key] = p.choose_second(*tree, MODELS[child]), process_time() - start
                    (request, action), score_time = decisions[key]
                    extra = prep + score_time
            second = p.HANDOFF if action is None else action.name
            if (first, second) not in evaluated:
                end, log = p.run_motion(c, state, record, scene, action)
                evaluated[first, second] = p.actual_outcome(c, end, log, scene, root, gate)
            timing = decision_time[arm]
            trajectories[arm].append(evaluated[first, second] | logs[first] | {
                "first": first, "observe_mode": adaptive, "second": second, "requested_second": request,
                "second_unavailable": request != p.HANDOFF and action is None,
                "child_alternatives": alternatives,
                "probes": int(first != p.HANDOFF) + int(second != p.HANDOFF),
                "decision_cpu_seconds": None if timing is None else timing + extra,
            })
    # Sharing a cache is an implementation detail. The trace records physical
    # equality for the paired same-first controls so it can be re-audited.
    for control, target in (("psdp_direct", "psdp"), ("psdp_precommit", "psdp"),
                            ("legacy_precommit", "legacy_program")):
        for a, b in zip(trajectories[control], trajectories[target], strict=True):
            assert a["history_sha256"] == b["history_sha256"]
    return row | {"root_frame": root.frame, "deadline_frame": gate, "root_mm": root.travel,
                  "root_position": [root.s, root.x], "incumbent": anchor, "planned": plans,
                  "precommitted": prescribed, "outcomes": trajectories}
