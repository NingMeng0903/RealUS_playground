"""Independently regenerate continuation experiments against the archived MPC."""

import argparse
from collections import Counter
import hashlib
import json
import lzma
from pathlib import Path
import sys

import numba
import numpy as np


def without_cache(*args, **kwargs):
    kwargs.pop("cache", None)
    return original_njit(*args, **kwargs)


original_njit = numba.njit


def load_shared(repo, motion, settings):
    global acoustics, control, shared
    numba.njit = without_cache
    sys.path.insert(0, str(repo / "NMI_idea/experiments/ab_full_b_rollout_20261113"))
    import acoustics
    import control
    import run as shared

    assert hashlib.sha256(motion.read_bytes()).hexdigest() == settings["bank_sha256"]
    shared.initialize(repo, motion, settings["model_size"], 6, "joint")


def load_rows(folder):
    path = folder / "cases.jsonl"
    source = path.open() if path.exists() else lzma.open(path.with_suffix(".jsonl.xz"), "rt")
    with source:
        return [json.loads(line) for line in source]


def packet(detect, cdf, tick, address, tape, counts):
    innovation = tape[tick]
    present = np.flatnonzero(innovation[:3] < detect[tick, address, 0])
    counts["independent_packet_regenerations"] += 1
    if not len(present):
        return 0, -1
    pixels = np.array([
        min(int(np.searchsorted(cdf[tick, address, 0, kind], innovation[kind + 3])), 48)
        for kind in present
    ])
    distance = (acoustics.X_REP[pixels // 7] / 4)**2 + (acoustics.Z_REP[pixels % 7] / 3)**2
    tied = np.flatnonzero(np.abs(distance - min(distance)) <= 1e-12)
    selected = tied[int(innovation[6] * len(tied))]
    return int(1 + pixels[selected] + 49 * (len(present) > 1)), int(present[selected])


def witness_exists(scene, node, progress, reserve, slots):
    if progress == len(scene.targets):
        return True
    if slots < len(scene.targets) - progress:
        return False
    for goal, expense in enumerate(scene.edge_costs[node]):
        if np.all(expense <= reserve + 1e-8) and witness_exists(
            scene, goal, progress + int(scene.site[goal] == scene.targets[progress]), reserve - expense, slots - 1,
        ):
            return True
    return False


def mpc(scene, state):
    belief, good, reserve, node, slot, tick, progress, _ = state
    return control.mpc_action(scene.public(), belief, good, node, slot, tick, progress, reserve)


def state_hash(state):
    belief, good, reserve, node, slot, tick, progress, _ = state
    return shared.encode_state(belief, good, node, slot, tick, progress, reserve)


def replay(scene, detect, cdf, tape, initial, stop, forced, counts):
    belief, good, reserve, node, slot, tick, progress, wrong = initial
    trace = []
    handoff = None
    while slot < stop and progress < len(scene.targets):
        owns = len(trace) < len(forced) and (not trace or progress == initial[6])
        if owns:
            goal = forced[len(trace)]
        else:
            state = belief, good, reserve, node, slot, tick, progress, wrong
            if handoff is None:
                handoff = {"slot": slot, "public_state_sha256": state_hash(state)}
            goal = mpc(scene, state)
        if goal < 0:
            break
        expense = scene.edge_costs[node, goal]
        assert np.all(expense <= reserve + 1e-8)
        tick += int(expense[3])
        address = scene.to_address[goal]
        obs, label = packet(detect, cdf, tick, address, tape, counts)
        previous = belief
        belief, confidence = control.posterior(previous, scene.likelihood, scene.correct, tick, address, obs)
        claim = int(obs > 0 and scene.site[goal] == scene.targets[progress] and confidence >= .87)
        good, _ = control.advance_joint(previous, good, scene.likelihood, scene.correct, tick, address, obs, bool(claim))
        progress += claim
        wrong = wrong or bool(claim and label != 0)
        trace.append([node, goal, obs, confidence, claim, label, *expense[:3], tick, slot, int(owns)])
        reserve = reserve - expense
        node, slot = goal, slot + 1
    final = belief, good, reserve, node, slot, tick, progress, wrong
    if handoff is None:
        handoff = {"slot": slot, "public_state_sha256": state_hash(final)}
    return final, trace, handoff


def check_candidates(scene, state, depth, library, actions, counts):
    _, _, reserve, node, slot, _, progress, _ = state
    baseline = mpc(scene, state) if slot == depth else -1
    expected = {"MPC"}
    if slot == depth and slot < 6 and progress < len(scene.targets):
        target = scene.targets[progress]
        for goal, expense in enumerate(scene.edge_costs[node]):
            if not np.all(expense <= reserve + 1e-8) or not witness_exists(
                scene, goal, progress + int(scene.site[goal] == target), reserve - expense, 5 - slot,
            ):
                continue
            if goal != baseline:
                expected.add(str(goal))
            if library == "pairs" and slot < 5:
                for second, next_expense in enumerate(scene.edge_costs[goal]):
                    if np.all(next_expense <= reserve - expense + 1e-8) and witness_exists(
                        scene, second, progress + int(scene.site[second] == target), reserve - expense - next_expense, 4 - slot,
                    ):
                        expected.add(f"{goal}>{second}")
    assert expected == set(actions)
    for key in expected - {"MPC"}:
        prefix = actions[key]["prefix"]
        assert prefix == [int(part) for part in key.split(">")]
        witness = actions[key]["witness"]
        assert witness[:len(prefix)] == prefix and len(witness) <= 6 - slot
        at, stage, remaining = node, progress, reserve.copy()
        for j, goal in enumerate(witness):
            expense = scene.edge_costs[at, goal]
            assert np.all(expense <= remaining + 1e-8)
            remaining = remaining - expense
            if len(prefix) == 1 or j != 0:
                stage += int(scene.site[goal] == scene.targets[stage])
            at = goal
        assert stage == len(scene.targets)
        counts["joint_witnesses"] += 1
    return baseline


def measure(scene, final):
    _, _, remaining, _, slots, tick, claims, any_wrong = final
    used = scene.reserve - remaining
    return [int(claims == len(scene.targets) and not any_wrong), int(any_wrong),
            used[0], used[1], tick * scene.clock_unit, used[2], slots]


def archived_parity(scene, detect, cdf, tape, prior, prefix, tail, raw, depth):
    option = control.empty_option() if not prefix else np.array([prefix[0], 1, control.RULE_ONE, -1, -1], dtype=np.int64)
    success, wrong, steps, _, _, _, _, cost, trace = control.simulate(
        scene.public(), detect, cdf, 0, tape, prior, scene.start, scene.reserve, option,
    )
    assert raw[:2] == [success, wrong] and raw[-1] == steps
    np.testing.assert_allclose(raw[2:6], cost[[0, 1, 3, 2]] * [1, 1, scene.clock_unit, 1], atol=1e-10, rtol=0)
    if steps:
        np.testing.assert_allclose(np.asarray(tail)[:, :10], trace[:steps, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]],
                                   atol=1e-12, rtol=0)


def check_record(row, settings, counts, repeats):
    seed = acoustics.seed_for(settings["namespace"], row["index"])
    assert seed == row["case_seed"]
    data = shared.DATA[row["task"]]
    scene, geometry = data["scene"], data["geometry"]
    case = acoustics.make_scenario(geometry.world, seed)
    _, _, raw_detect, raw_cdf = acoustics.probabilities(geometry, case, data["clocks"], "joint")
    detect, cdf = np.ascontiguousarray(raw_detect[:, :, None]), np.ascontiguousarray(raw_cdf[:, :, None])
    original_tape = acoustics.tape(seed, len(scene.likelihood))
    anchor, _ = packet(detect, cdf, 0, scene.to_address[scene.start], original_tape, counts)
    assert row["public"]["anchor"] == anchor
    h = scene.likelihood.shape[-1]
    prior, _ = control.posterior(np.full(h, 1 / h), scene.likelihood, scene.correct, 0, scene.to_address[scene.start], anchor)
    initial = prior, prior.copy(), scene.reserve.copy(), scene.start, 0, 0, 0, False
    depth = row["depth"]
    state, prefix_trace, _ = replay(scene, detect, cdf, original_tape, initial, depth, (), counts)
    np.testing.assert_allclose(prefix_trace, row["prefix_trace"], atol=1e-12, rtol=0)
    assert row["public"]["history"] == [[int(s[1]), int(s[2])] for s in prefix_trace]
    assert row["public"]["public_state_sha256"] == state_hash(state)
    assert row["public"]["clock"] == state[5] and row["public"]["progress"] == state[6]
    baseline = check_candidates(scene, state, depth, settings["library"], row["actions"], counts)
    assert row["public"]["baseline"] == baseline
    if "arms" in row:
        assert hashlib.sha256(original_tape.tobytes()).hexdigest() == row["tape_sha256"]
        trials = [(original_tape, arm["chosen"], arm["measured"], arm) for arm in row["arms"].values()]
    else:
        trials = []
        for rep, record in enumerate(row["tapes"][:repeats]):
            tape_seed = acoustics.seed_for(seed, 200000 + rep)
            assert tape_seed == record["seed"]
            tape = acoustics.tape(tape_seed, len(scene.likelihood))
            tape[:state[5] + 1] = original_tape[:state[5] + 1]
            assert hashlib.sha256(tape.tobytes()).hexdigest() == record["sha256"]
            trials.extend((tape, key, outcomes[rep], None) for key, outcomes in row["outcomes"].items())
    for tape, key, expected, arm in trials:
        forced = row["actions"][key]["prefix"]
        final, tail, handoff = replay(scene, detect, cdf, tape, state, 6, forced, counts)
        raw = measure(scene, final)
        np.testing.assert_allclose(raw, expected[:7], atol=1e-10, rtol=0)
        assert expected[7] >= raw[4]
        if arm is not None:
            assert arm["continuation"] == forced
            np.testing.assert_allclose(tail, arm["tail"], atol=1e-12, rtol=0)
            assert arm["terminal_public_state_sha256"] == state_hash(final)
            assert arm["handoff"] == handoff
        if len(forced) == 2:
            owned = sum(step[-1] for step in tail)
            assert owned == (1 if tail[0][4] else 2)
            counts["pairs_early_handoff" if owned == 1 else "pairs_execute_second"] += 1
        if not forced or (depth == 0 and len(forced) == 1):
            archived_parity(scene, detect, cdf, tape, prior, forced, prefix_trace + tail, raw, depth)
            counts["archived_full_trace_comparisons"] += 1
        counts["paired_terminal_paths"] += 1
    counts["public_states"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("pilot", "extended_oracle", "policy_pilot", "confirmation"))
    parser.add_argument("--study", type=Path, default=Path("work/continuation_b"))
    parser.add_argument("--repo", type=Path, default=Path("work/RealUS_playground"))
    parser.add_argument("--motion", type=Path, default=Path("work/full_b/motion/BANK_MAIN.json"))
    parser.add_argument("--first-cases", type=int, default=6)
    parser.add_argument("--extra-cases", type=int, nargs="*", default=[])
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    folder = args.study / "data" / args.split
    settings = json.loads((folder / "settings.json").read_text())
    load_shared(args.repo, args.motion, settings)
    selected = set(range(min(args.first_cases, settings["n"]))) | set(args.extra_cases)
    assert selected and all(0 <= index < settings["n"] for index in selected)
    rows = [r for r in load_rows(folder) if r["index"] in selected]
    assert len(rows) == len(selected) * len(settings["tasks"]) * len(settings["depths"])
    counts = Counter()
    for row in rows:
        check_record(row, settings, counts, args.repeats)
    print(json.dumps({"split": args.split, "case_indices": sorted(selected), **counts}, indent=2))


if __name__ == "__main__":
    main()
