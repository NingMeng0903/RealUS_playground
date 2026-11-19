"""Replay checked first continuations, with exact shared MPC after the fork."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import lzma
from multiprocessing import get_context
from pathlib import Path
import sys
import time

from numba import njit
import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = "ab_full_b_rollout_20261113"
REPO = None
BASE = "MPC"
EMPTY = np.empty(0, dtype=np.int64)
NAMESPACES = dict(oracle=717001, train=717002, development=717003, confirmation=717004,
                  pilot=717000, extended_oracle=717005)
FIELDS = ("correct_completion", "any_wrong", "tcp_mm", "turn_deg", "motion_s", "rail_mm",
          "captures", "combined_s")


def load_shared(repo):
    global REPO, acoustics, control, shared
    REPO = Path(repo).resolve()
    sys.path.insert(0, str(REPO / "NMI_idea" / "experiments" / PREVIOUS))
    import acoustics
    import control
    import run as shared


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def initialize(args):
    load_shared(args.repo)
    if args.motion is None:
        archive = REPO / "NMI_idea" / "experiments" / PREVIOUS / "motion" / "BANK_MAIN.json.xz"
        motion = args.out / "BANK_MAIN.json"
        motion.write_bytes(lzma.decompress(archive.read_bytes()))
    else:
        motion = args.motion
    bank_hash = hashlib.sha256(motion.read_bytes()).hexdigest()
    frozen_path = REPO / "NMI_idea" / "experiments" / PREVIOUS / "FREEZE.json"
    frozen = json.loads(frozen_path.read_text())["sha256"]
    assert bank_hash == frozen["motion/BANK_MAIN.json"], bank_hash
    for name in ("control.py", "run.py", "acoustics.py"):
        assert hashlib.sha256((frozen_path.parent / name).read_bytes()).hexdigest() == frozen[name], name
    shared.initialize(REPO, motion, args.model_size, shared.CAPTURES, "joint")
    return bank_hash, {name: frozen[name] for name in ("control.py", "run.py", "acoustics.py")}


@njit(cache=True)
def execute(model, true_detect, true_cdf, innovations, state, prefix, stop_slot):
    """Carry public filtering state and use MPC whenever the prefix is exhausted."""
    like, correct, addresses, sites, _, _, costs, targets, potential, _, _, _ = model
    belief, good, reserve, at, slot, clock, progress, wrong = state
    trace = np.empty((len(potential) - 1 - slot, 12), dtype=np.float64)
    used = 0
    initial_progress = progress
    while slot < stop_slot and slot < len(potential) - 1 and progress < len(targets):
        owns_action = used < len(prefix) and (used == 0 or progress == initial_progress)
        goal = prefix[used] if owns_action else control.mpc_action(
            model, belief, good, at, slot, clock, progress, reserve)
        if goal < 0:
            break
        if not control.allowed(costs, at, goal, reserve):
            raise ValueError("continuation uses an unchecked or over-budget edge")
        expense = costs[at, goal]
        clock += int(expense[3])
        obs, kind = control.capture_draw(true_detect, true_cdf, clock, addresses[goal], 0, innovations)
        previous = belief
        belief, confidence = control.posterior(previous, like, correct, clock, addresses[goal], obs)
        claim = obs > 0 and sites[goal] == targets[progress] and confidence >= control.CONFIDENCE
        good, _ = control.advance_joint(previous, good, like, correct, clock, addresses[goal], obs, claim)
        if claim:
            progress += 1
            wrong = wrong or kind != 0
        trace[used, 0], trace[used, 1], trace[used, 2] = at, goal, obs
        trace[used, 3], trace[used, 4], trace[used, 5] = confidence, int(claim), kind
        trace[used, 6], trace[used, 7], trace[used, 8] = expense[0], expense[1], expense[2]
        trace[used, 9], trace[used, 10], trace[used, 11] = clock, slot, int(owns_action)
        reserve = reserve - expense
        at = goal
        used += 1
        slot += 1
    return (belief, good, reserve, at, slot, clock, progress, wrong), trace[:used]


def independent_case(task, seed):
    data = shared.DATA[task]
    scene, geometry = data["scene"], data["geometry"]
    scenario = acoustics.make_scenario(geometry.world, seed)
    _, _, detect, cdf = acoustics.probabilities(geometry, scenario, data["clocks"], "joint")
    return scene, np.ascontiguousarray(detect[:, :, None]), np.ascontiguousarray(cdf[:, :, None])


def begin(task, seed, detect, cdf, depth):
    data = shared.DATA[task]
    scene = data["scene"]
    tape = acoustics.tape(seed, len(scene.likelihood))
    belief, obs, _ = shared.initial_prior(data, detect, cdf, tape, "anchor")
    state = belief, belief.copy(), scene.reserve.copy(), scene.start, 0, 0, 0, False
    return (*execute(scene.public(), detect, cdf, tape, state, EMPTY, depth), obs, tape)


def mpc_command(scene, state):
    belief, good, reserve, at, slot, clock, progress, _ = state
    return control.mpc_action(scene.public(), belief, good, at, slot, clock, progress, reserve)


def full_witness(scene, at, slots, progress, reserve):
    """One joint feasible witness under ideal future claims, if it exists."""
    if progress == len(scene.targets):
        return ()
    if slots < len(scene.targets) - progress:
        return None
    if np.any(scene.needs[slots, at, progress] > reserve + 1.e-8):
        return None
    for goal, cost in enumerate(scene.edge_costs[at]):
        # Non-advancing visits can be mechanically necessary between sites.
        if goal == at and scene.site[goal] != scene.targets[progress]:
            continue
        if np.all(cost <= reserve + 1.e-8):
            rest = full_witness(scene, goal, slots - 1,
                                progress + int(scene.site[goal] == scene.targets[progress]), reserve - cost)
            if rest is not None:
                return (goal, *rest)
    return None


def witness_for(scene, at, slots, progress, reserve, prefix):
    executed = []
    for j, goal in enumerate(prefix):
        if progress == len(scene.targets):
            return tuple(executed)
        edge = scene.edge_costs[at, goal]
        if slots <= 0 or not np.all(edge <= reserve + 1.e-8):
            return None
        executed.append(goal)
        reserve = reserve - edge
        # Verify a full joint-budget witness on the unresolved branch.
        if len(prefix) == 1 or j != 0:
            progress += int(scene.site[goal] == scene.targets[progress])
        at, slots = goal, slots - 1
    rest = full_witness(scene, at, slots, progress, reserve)
    return None if rest is None else (*executed, *rest)


def candidate_library(scene, state, depth, library):
    _, _, reserve, at, slot, _, progress, _ = state
    baseline = mpc_command(scene, state) if slot == depth else -1
    candidates = {BASE: ((), None)}
    # Include resumes after a voluntary MPC stop if its budget still permits an
    # executable finish. Any episode already complete stays in the denominator.
    if slot != depth or progress == len(scene.targets) or slot >= len(scene.potential) - 1:
        return baseline, candidates
    slots = len(scene.potential) - 1 - slot
    for goal, cost in enumerate(scene.edge_costs[at]):
        if not np.all(cost <= reserve + 1.e-8):
            continue
        witness = witness_for(scene, at, slots, progress, reserve, (goal,))
        if witness is None:
            continue
        # Forcing exactly MPC's first command followed by MPC is an exact
        # duplicate and is represented by the incumbent.
        if goal != baseline:
            candidates[str(goal)] = ((goal,), witness)
        if library == "pairs" and slots >= 2:
            for second, next_cost in enumerate(scene.edge_costs[goal]):
                if not np.all(next_cost <= reserve - cost + 1.e-8):
                    continue
                witness = witness_for(scene, at, slots, progress, reserve, (goal, second))
                if witness is not None:
                    candidates[f"{goal}>{second}"] = ((goal, second), witness)
    return baseline, candidates


def state_hash(state):
    belief, good, reserve, at, slot, clock, progress, _ = state
    return shared.encode_state(belief, good, at, slot, clock, progress, reserve)


def trace_public(trace):
    return [[int(row[1]), int(row[2])] for row in trace]


def state_public(scene, state, trace, anchor, baseline, depth):
    belief, good, reserve, at, slot, clock, progress, _ = state
    last = int(trace[-1, 2]) if len(trace) else anchor
    if last > 0:
        pixel = (last - 1) % acoustics.PIXELS
        x, z = float(acoustics.X_REP[pixel // acoustics.NZ]), float(acoustics.Z_REP[pixel % acoustics.NZ])
    else:
        x = z = 0.
    p = float(belief[scene.material_coordinate >= 0].sum())
    return dict(depth=depth, anchor=anchor, history=trace_public(trace), public_state_sha256=state_hash(state),
                at=int(at), site=int(scene.site[at]), tilt=int(scene.tilt[at]), rho=int(scene.rho[at]),
                slots=int(len(scene.potential) - 1 - slot), clock=int(clock), progress=int(progress),
                target=int(scene.targets[progress]) if progress < len(scene.targets) else -1,
                reserve=reserve.tolist(), prior_good=float(good.sum()), ess=float(1. / (belief @ belief)),
                displacement_mean=float(belief @ scene.material_coordinate), displacement_positive=p,
                last_detected=int(last > 0), last_multiple=int(last > acoustics.PIXELS), last_x=x, last_z=z,
                baseline=int(baseline))


def describe_action(scene, state, prefix):
    belief, good, reserve, at, slot, clock, progress, _ = state
    result = []
    for j, goal in enumerate(prefix):
        cost = scene.edge_costs[at, goal]
        reserve = reserve - cost
        clock += int(cost[3])
        address = scene.to_address[goal]
        yes, risk = control.claim_under_mass(belief, good, scene.likelihood, scene.correct, clock, address)
        target = scene.targets[progress]
        future = control.remaining_value(scene.potential, scene.needs,
                                          len(scene.potential) - 2 - slot - j, goal, progress, reserve)
        result.append(dict(goal=int(goal), site=int(scene.site[goal]), tilt=int(scene.tilt[goal]),
                           rho=int(scene.rho[goal]), cost=cost.tolist(), future=float(future),
                           target=int(scene.site[goal] == target), yes=float(yes), risk=float(risk),
                           miss=float(belief @ scene.likelihood[clock, address, 0]),
                           expected_correct_return=float(belief @ scene.correct[clock, address].sum(axis=0))))
        at = goal
    return result


def metrics(scene, state, elapsed):
    _, _, reserve, _, slot, clock, progress, wrong = state
    used = scene.reserve - reserve
    seconds = float(clock) * scene.clock_unit
    return [int(progress == len(scene.targets) and not wrong), int(wrong), *map(float, used[:2]),
            seconds, float(used[2]), int(slot), seconds + elapsed]


def finish(scene, detect, cdf, tape, state, prefix, prefix_cpu=0., prediction_cpu=0.):
    start = time.process_time()
    final, tail = execute(scene.public(), detect, cdf, tape, state, np.asarray(prefix, dtype=np.int64),
                          len(scene.potential) - 1)
    elapsed = time.process_time() - start + prefix_cpu + prediction_cpu
    return metrics(scene, final, elapsed), tail, state_hash(final)


def compare_shared(scene, detect, cdf, tape, state, prefix_trace, tail, final_metrics):
    """Ensure resuming direct MPC matches the archived implementation end to end."""
    result = control.simulate(scene.public(), detect, cdf, 0, tape, state[0], scene.start,
                              scene.reserve, control.empty_option())
    success, wrong, steps, _, _, _, _, cost, trace = result
    trace = trace[:steps]
    joined = np.concatenate((prefix_trace, tail))
    assert final_metrics[:2] == [success, wrong] and len(joined) == steps
    expected = trace[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]]
    np.testing.assert_allclose(joined[:, :10], expected, atol=1.e-12, rtol=0.)
    np.testing.assert_allclose(final_metrics[2:6], [cost[0], cost[1], cost[3] * scene.clock_unit, cost[2]],
                               atol=1.e-10, rtol=0.)


def public_actions(scene, state, prefix_trace, anchor, depth, library):
    baseline, candidates = candidate_library(scene, state, depth, library)
    public = state_public(scene, state, prefix_trace, anchor, baseline, depth)
    base_prefix = (baseline,) if baseline >= 0 else ()
    public["baseline_features"] = describe_action(scene, state, base_prefix)
    # Drop hidden prior-claim correctness and evaluator labels from every input
    # to the learned score. The good vector is belief mass, not true correctness.
    actions = {key: dict(prefix=list(prefix), witness=witness,
                         features=describe_action(scene, state, prefix) if key != BASE else public["baseline_features"])
               for key, (prefix, witness) in candidates.items()}
    return public, actions


def episode_context(task, seed, depth, scene, detect, cdf, library):
    started = time.process_time()
    state, prefix_trace, anchor, prefix_tape = begin(task, seed, detect, cdf, depth)
    prefix_cpu = time.process_time() - started
    public, actions = public_actions(scene, state, prefix_trace, anchor, depth, library)
    return state, prefix_trace, prefix_tape, prefix_cpu, public, actions


def collect_one(job):
    index, args = job
    seed = acoustics.seed_for(NAMESPACES[args.split], index)
    rows = []
    for task in args.tasks:
        scene, detect, cdf = independent_case(task, seed)
        start_state, _, _, _ = begin(task, seed, detect, cdf, 0)
        for depth in args.depths:
            state, prefix_trace, prefix_tape, prefix_cpu, public, actions = episode_context(
                task, seed, depth, scene, detect, cdf, args.library)
            results = {key: [] for key in actions}
            tapes = []
            for rep in range(args.repeats):
                tape_seed = acoustics.seed_for(seed, 200000 + rep)
                tape = acoustics.tape(tape_seed, len(scene.likelihood))
                tape[:state[5] + 1] = prefix_tape[:state[5] + 1]
                tapes.append(dict(seed=tape_seed, sha256=digest(tape)))
                for key, action in actions.items():
                    val, trace, _ = finish(scene, detect, cdf, tape, state, action["prefix"], prefix_cpu)
                    if index < 2 and rep < 2 and key == BASE:
                        compare_shared(scene, detect, cdf, tape, start_state, prefix_trace, trace, val)
                    results[key].append(val)
            rows.append(dict(index=index, case_seed=seed, task=task, depth=depth, public=public,
                             prefix_trace=prefix_trace.tolist(), tapes=tapes, actions=actions, outcomes=results))
    return rows


def warm():
    for task in shared.TASKS:
        scene, detect, cdf = independent_case(task, 10)
        state, prefix, _, tape = begin(task, 10, detect, cdf, 0)
        value, tail, _ = finish(scene, detect, cdf, tape, state, ())
        compare_shared(scene, detect, cdf, tape, state, prefix, tail, value)
        episode_context(task, 10, 0, scene, detect, cdf, "single")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--split", choices=tuple(NAMESPACES), required=True)
    parser.add_argument("--model-size", type=int, default=256)
    parser.add_argument("--tasks", nargs="+", default=["daughter", "trunk"])
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--library", choices=("single", "pairs"), default="single")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.split == "confirmation":
        parser.error("the confirmation split is reserved for the frozen policy runner")
    if min(args.depths) < 0 or max(args.depths) >= 6 or args.repeats < 1 or args.n < 1:
        parser.error("invalid depth, case count or repeat count")
    args.out.mkdir(parents=True, exist_ok=True)
    bank, sources = initialize(args)
    warm()
    settings = {k: v for k, v in vars(args).items() if k not in ("repo", "out", "motion", "workers")}
    settings.update(bank_sha256=bank, sources=sources, namespace=NAMESPACES[args.split], fields=FIELDS,
                    caps=[shared.TRAVEL_LIMIT, shared.TURN_LIMIT, shared.RAIL_LIMIT, shared.CAPTURES],
                    action_index="position in main-bank nodes sorted by integer id", first_capture="shared anchor at t=0",
                    two_step_rule="second command only when the first did not advance public task progress")
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    with ((args.out / "cases.jsonl").open("w") as file,
          ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool):
        jobs = ((index, args) for index in range(args.n))
        for index, rows in enumerate(pool.map(collect_one, jobs)):
            for row in rows:
                file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            if index % 12 == 0 or index + 1 == args.n:
                file.flush()
                print(f"{args.split}: {index + 1}/{args.n} independent cases", flush=True)


if __name__ == "__main__":
    main()
