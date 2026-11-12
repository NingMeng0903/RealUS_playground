"""Run paired, full-episode MPC and conditional-prefix comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

import acoustics
import control


HERE = Path(__file__).resolve().parent
TASKS = {"daughter": ([4, 3], 1), "trunk": ([6, 7], 0)}
ARMS = ("MPC", "B", "one", "fixed", "status", "single_realization", "unguarded")
SPLIT = {"smoke": 716901, "development": 716902, "confirmation": 716903, "headroom": 716904}
CAPTURES = 6
TRAVEL_LIMIT = 65.
TURN_LIMIT = 90.
RAIL_LIMIT = 250.
CONFIG = DATA = BANK = None
PLAN_CACHE = {}


def initialize(repo, motion, size, captures, mode, rate=1.,
               resource_caps=(TRAVEL_LIMIT, TURN_LIMIT, RAIL_LIMIT)):
    global DATA, BANK
    BANK = json.loads(Path(motion).read_text())
    world = acoustics.setup_world(repo)
    nodes = sorted(BANK["nodes"], key=lambda n: int(n["id"]))
    ids = {int(node["id"]): j for j, node in enumerate(nodes)}
    site = np.array([node["site"] for node in nodes], dtype=np.int64)
    views = np.array([node["view"] for node in nodes], dtype=np.int64)
    tilt = np.where(views == 7, -1, np.where(views == 8, 1, 0))
    realizations = {label: i for i, label in enumerate(sorted({str(n["realization"]) for n in nodes}))}
    rho = np.array([realizations[str(n["realization"])] for n in nodes], dtype=np.int64)
    durations = {int(round(float(e["duration_s"]) * 1000)) for e in BANK["edges"]}
    ms = math.gcd(*durations)
    clock_unit = ms / 1000.
    time_cap = captures * max(durations) // ms
    clocks = np.arange(time_cap + 1) * clock_unit
    DATA = {}
    for task, (goal, label) in TASKS.items():
        geometry = acoustics.Geometry(world, nodes, label)
        edges = np.full((len(nodes), len(nodes), 4), np.inf)
        for edge in BANK["edges"]:
            a, b = ids[int(edge["start"])], ids[int(edge["goal"])]
            if site[a] not in (5, *goal) or site[b] not in (5, *goal):
                continue
            if edge["max_joint_rate_fraction"] > rate + 1.e-10:
                continue
            ticks = int(round(float(edge["duration_s"]) * 1000)) // ms
            edges[a, b] = (edge["travel_mm"], edge["turn_deg"], edge["rail_mm"], ticks)
        start = ids[int(BANK["task_starts"][task])]
        targets = np.asarray(goal, dtype=np.int64)
        if len(np.flatnonzero(np.isfinite(edges[start, :, 0]))) == 0:
            raise RuntimeError(f"no checked motion from {task} start")
        likelihood, correct, detect, cdf = acoustics.model_tables(geometry, clocks, size, mode)
        hit = .45 if task == "daughter" else .68
        reserve = np.array([*resource_caps, time_cap])
        potential, needed = control.nominal_potential(site, edges, targets, captures, hit)
        material = np.array([acoustics.make_scenario(world, acoustics.seed_for(acoustics.PRIOR_SEED, j),
                                                    mode).body.w_subject_mm[5] for j in range(size)])
        scene = control.Scene(likelihood, correct, detect, cdf,
                              geometry.node_to_address, site, tilt, rho, edges, targets, start,
                              reserve, potential, needed, material, 5, clock_unit)
        DATA[task] = dict(scene=scene, geometry=geometry, likelihood0=likelihood[:1], correct0=correct[:1],
                          clocks=clocks, clock_unit=clock_unit, mode=mode, nodes=nodes)
    return clock_unit, float(clocks[-1]), sorted(v / 1000. for v in durations)


def mechanical_support(scene):
    """Enumerate finite legal routes with ideal fresh acceptance, before data."""
    cap = len(scene.potential) - 1
    edges = scene.edge_costs
    at_start = scene.start
    routes = []

    def visit(at, reserve, path, done):
        if done == len(scene.targets):
            routes.append(path)
            return
        if len(path) >= cap:
            return
        if np.any(scene.needs[cap - len(path), at, done] > reserve + 1.e-8):
            return
        for node, cost in enumerate(edges[at]):
            if np.all(cost <= reserve + 1.e-8):
                visit(node, reserve - cost, (*path, node), done + int(scene.site[node] == scene.targets[done]))

    visit(at_start, scene.reserve, (), 0)
    opposite = [r for r in routes if len(r) > 1 and any(
        scene.tilt[a] and scene.tilt[a] == -scene.tilt[b] for a, b in zip(r, r[1:]))]
    switched = [r for r in routes if any(scene.rho[n] != scene.rho[at_start] for n in r)]
    # Compare immediate acoustic successors of each equal-pose pair. A rail
    # difference alone need not change the sound-view library.
    equivalent = []
    for i in range(len(scene.site)):
        for j in range(i + 1, len(scene.site)):
            if scene.to_address[i] != scene.to_address[j]:
                continue
            x = {int(scene.to_address[k]) for k in range(len(scene.site))
                 if np.all(edges[i, k] <= scene.reserve + 1.e-8)}
            y = {int(scene.to_address[k]) for k in range(len(scene.site))
                 if np.all(edges[j, k] <= scene.reserve + 1.e-8)}
            if x or y:
                equivalent.append(dict(nodes=[i, j], first_only=sorted(x - y), second_only=sorted(y - x)))
    return dict(complete_routes=len(routes), with_opposite_tilts=len(opposite),
                with_other_realization=len(switched), root_nodes=sorted({r[0] for r in routes}),
                equivalent_support=equivalent,
                example_opposite=list(opposite[0]) if opposite else None,
                example_realization=list(switched[0]) if switched else None)


def warmup():
    for data in DATA.values():
        scene = data["scene"]
        prior = np.full(scene.likelihood.shape[-1], 1. / scene.likelihood.shape[-1])
        noise = acoustics.tape(2, len(scene.likelihood))
        control.simulate(scene.public(), scene.detect, scene.cdf, 0, noise, prior, scene.start,
                         scene.reserve, control.empty_option())
        indices = np.array([0], dtype=np.int64)
        control.rollout(scene.public(), scene.detect, scene.cdf, indices, noise[None], prior,
                        scene.start, scene.reserve, control.empty_option())
        control.library(scene, prior)


def pair_plan(task, prior, prefix_obs, args):
    key = task, prefix_obs, args.screen, args.validate, tuple(args.arms)
    if key in PLAN_CACHE:
        return PLAN_CACHE[key]
    scene = DATA[task]["scene"]
    model = scene.public()
    cap = len(scene.likelihood)
    base = control.empty_option()
    # Model inference scenarios are keyed only by observable task and anchor
    # packet, independently of the actual evaluation case and tape.
    task_id = list(TASKS).index(task)
    screen_seed = acoustics.seed_for(34017298 + task_id, prefix_obs + 1)
    validate_seed = acoustics.seed_for(34017298 + task_id, prefix_obs + 10001)
    screen = control.plan_random(screen_seed, prior, cap, args.screen)
    check = control.plan_random(validate_seed, prior, cap, args.validate)
    began = time.process_time()
    roots = control.library(scene, prior)
    base_screen = control.rollout(model, scene.detect, scene.cdf, *screen, prior,
                                  scene.start, scene.reserve, base)
    base_check = control.rollout(model, scene.detect, scene.cdf, *check, prior,
                                 scene.start, scene.reserve, base)
    common_s = time.process_time() - began
    outcomes = {"MPC": (base, {"reason": "baseline"}, 0.)}
    screen_cache, validate_cache = {}, {}
    for arm in args.arms:
        if arm == "MPC":
            continue
        # Charge each arm its own option search; sharing cached model rollouts
        # across algorithms would obscure its independent compute use.
        began = time.process_time()
        option, info = control.choose_option(
            scene, prior, "B" if arm == "unguarded" else arm,
            roots, screen, check, base_screen, base_check, None, None,
            unguarded=arm == "unguarded")
        outcomes[arm] = (option, info, common_s + time.process_time() - began)
    if len(PLAN_CACHE) > 128:
        PLAN_CACHE.clear()
    baseline = {"screen": base_screen.mean(axis=0).tolist(),
                "validation": base_check.mean(axis=0).tolist()}
    result = outcomes, roots, baseline
    PLAN_CACHE[key] = result
    return result


def initial_prior(data, detect, cdf, noise, initialization):
    scene = data["scene"]
    h = scene.likelihood.shape[-1]
    prior = np.full(h, 1. / h)
    if initialization == "coarse":
        return prior, -1, -1
    obs, kind = control.capture_draw(detect, cdf, 0, scene.to_address[scene.start], 0, noise)
    post, _ = control.posterior(prior, data["likelihood0"], data["correct0"], 0,
                                scene.to_address[scene.start], obs)
    return post, obs, kind


def encode_state(prior, good, node, slot, clock, progress, reserve):
    payload = b"".join((np.asarray([node, slot, clock, progress], dtype="<i8").tobytes(),
                        np.asarray(reserve, dtype="<f8").tobytes(), np.asarray(prior, dtype="<f8").tobytes(),
                        np.asarray(good, dtype="<f8").tobytes()))
    return hashlib.sha256(payload).hexdigest()


def actual(scene, data, prior, option, detect, cdf, innovations, planning_s):
    model = scene.public()
    started = time.process_time()
    success, wrong, steps, owned, handoff, progress, ess, cost, trace = control.simulate(
        model, detect, cdf, 0, innovations, prior, scene.start, scene.reserve, option)
    execution_cpu_s = time.process_time() - started
    sampled_motion_s = cost[3] * data["clock_unit"]
    rows = []
    belief = prior.copy()
    good = prior.copy()
    progress_here, at, clock = 0, scene.start, 0
    remaining = scene.reserve.copy()
    handoff_state = None
    for index, entry in enumerate(trace[:steps]):
        origin, goal, phase, obs, conf, claim, kind, travel, turn, rail, arrival = entry.tolist()
        start, end, phase, obs, kind = map(int, (origin, goal, phase, obs, kind))
        if start != at:
            raise ValueError("motion history is discontinuous")
        if index == handoff:
            requested = control.mpc_action(model, belief, good, at, index, clock, progress_here, remaining)
            if requested != end:
                raise ValueError("handoff did not run the shared MPC on the actual exit state")
            handoff_state = encode_state(belief, good, at, index, clock, progress_here, remaining)
        clock += int(scene.edge_costs[at, end, 3])
        if clock != int(arrival):
            raise ValueError("sampled motion and capture time disagree")
        previous = belief
        belief, reconstructed = control.posterior(previous, scene.likelihood, scene.correct,
                                                   clock, scene.to_address[end], obs)
        good, _ = control.advance_joint(previous, good, scene.likelihood, scene.correct,
                                        clock, scene.to_address[end], obs, bool(claim))
        if abs(reconstructed - conf) > 1.e-12:
            raise ValueError("replay confidence mismatch")
        progress_here += int(claim)
        remaining -= scene.edge_costs[at, end]
        at = end
        rows.append(dict(slot=index, t_s=float(data["clocks"][clock]), start=start, goal=end, phase=phase,
                         site=int(scene.site[end]), signed_tilt=int(scene.tilt[end]),
                         realization=int(scene.rho[end]), observation=obs, confidence=conf,
                         claim=bool(claim), eval_only_kind=kind, travel_mm=travel, turn_deg=turn,
                         rail_mm=rail))
    if handoff == steps:
        if control.mpc_action(model, belief, good, at, steps, clock, progress_here, remaining) != -1:
            raise ValueError("missing MPC action at terminal handoff")
        handoff_state = encode_state(belief, good, at, steps, clock, progress_here, remaining)
    return dict(success=bool(success), wrong=bool(wrong), progress=progress, steps=steps,
                option_steps=owned, handoff_at=None if handoff < 0 else handoff,
                handoff_state=handoff_state, min_ess=float(ess), path_mm=float(cost[0]),
                turn_deg=float(cost[1]), rail_mm=float(cost[2]), motion_s=sampled_motion_s,
                option_planning_cpu_s=planning_s, online_loop_cpu_s=execution_cpu_s,
                measured_combined_s=sampled_motion_s + planning_s + execution_cpu_s,
                option=option.tolist(), trace=rows)


def one(job):
    idx, args = job
    seed = acoustics.seed_for(SPLIT[args.split], idx)
    records, plan_records = [], {}
    for task in args.tasks:
        data = DATA[task]
        scene, geometry = data["scene"], data["geometry"]
        scenario = acoustics.make_scenario(geometry.world, seed, args.mode)
        _, _, case_detect, case_cdf = acoustics.probabilities(geometry, scenario, data["clocks"], args.mode)
        case_detect = np.ascontiguousarray(case_detect[:, :, None])
        case_cdf = np.ascontiguousarray(case_cdf[:, :, None])
        tape = acoustics.tape(seed, len(data["clocks"]))
        for initialization in args.initial:
            prior, prefix_obs, prefix_kind = initial_prior(data, case_detect, case_cdf, tape, initialization)
            selections, roots, forecast = pair_plan(task, prior, prefix_obs, args)
            initial = encode_state(prior, prior, scene.start, 0, 0, 0, scene.reserve)
            plan_key = f"{task}/{prefix_obs}"
            plan_records[plan_key] = dict(roots=roots, baseline=forecast,
                                          arms={k: dict(option=v[0].tolist(), **v[1])
                                                for k, v in selections.items()})
            for arm in args.arms:
                option, _, cpu_s = selections[arm]
                result = actual(scene, data, prior, option, case_detect, case_cdf, tape, cpu_s)
                records.append(dict(result, index=idx, case_seed=seed, arm=arm, task=task,
                                    initial=initialization, plan_key=plan_key, start_state=initial,
                                    start_observation=prefix_obs, start_eval_only_kind=prefix_kind,
                                    perturbation_hash=hashlib.sha256(tape.tobytes()).hexdigest()))
            if args.headroom_replicates and initialization == args.initial[0]:
                # Diagnostics can force public options on additional noisy
                # copies of a fixed case. Outcome labels never select a live
                # policy or alter the main cohort.
                choices = [control.empty_option(), *control.programs(roots, "B", scene, prior)]
                outcome = []
                for rep in range(args.headroom_replicates):
                    innovation = acoustics.tape(acoustics.seed_for(seed, 4000 + rep), len(scene.likelihood))
                    for choice in choices:
                        s, w, steps, owned, handoff, p, _, cost, trace = control.simulate(
                            scene.public(), case_detect, case_cdf, 0, innovation, prior,
                            scene.start, scene.reserve, choice)
                        outcome.append(dict(rep=rep, option=choice.tolist(), success=int(s),
                                            wrong=int(w), steps=steps, option_steps=owned,
                                            handoff=handoff, progress=p, costs=cost.tolist(),
                                            phases=[int(k) for k in trace[:steps, 2]]))
                records.append(dict(index=idx, case_seed=seed, arm="evaluator_headroom", task=task,
                                    initial=initialization, results=outcome))
    return records, plan_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parent / "RealUS_playground")
    parser.add_argument("--motion", type=Path, default=HERE / "motion" / "BANK.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT, default="smoke")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-size", type=int, default=256)
    parser.add_argument("--captures", type=int, default=CAPTURES)
    parser.add_argument("--screen", type=int, default=20)
    parser.add_argument("--validate", type=int, default=40)
    parser.add_argument("--mode", choices=("joint", "zero_depth_bias"), default="joint")
    parser.add_argument("--rate-limit", type=float, default=1.)
    parser.add_argument("--travel-limit", type=float, default=TRAVEL_LIMIT)
    parser.add_argument("--turn-limit", type=float, default=TURN_LIMIT)
    parser.add_argument("--rail-limit", type=float, default=RAIL_LIMIT)
    parser.add_argument("--mechanical-only", action="store_true")
    parser.add_argument("--initial", nargs="+", choices=("coarse", "anchor"), default=["coarse", "anchor"])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["MPC", "B", "one", "fixed", "status"])
    parser.add_argument("--headroom-replicates", type=int, default=0)
    args = parser.parse_args()
    if args.split == "confirmation" and args.headroom_replicates:
        raise ValueError("no forced-action diagnostic in the frozen evaluation")
    args.out.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    caps = args.travel_limit, args.turn_limit, args.rail_limit
    unit, limit, durations = initialize(args.repo, args.motion, 4 if args.mechanical_only else args.model_size,
                                        args.captures, args.mode, args.rate_limit, caps)
    support = {task: mechanical_support(DATA[task]["scene"]) for task in args.tasks}
    (args.out / "mechanical_support.json").write_text(json.dumps(support, indent=2) + "\n")
    if args.mechanical_only:
        print(json.dumps({k: {f: v for f, v in s.items() if f != "equivalent_support"}
                          for k, s in support.items()}, indent=2), flush=True)
        return
    if any(not s["complete_routes"] or not s["with_opposite_tilts"] for s in support.values()):
        raise RuntimeError("The sampled graph must support both completion and opposite-tilt probing")
    config = vars(args) | {"clock_unit_s": unit, "durations_s": durations, "seed_namespace": SPLIT[args.split],
                           "reserves": [*caps, limit],
                           "sites_and_labels": TASKS, "mpc_mask_quantiles": control.MPC_SCENARIOS,
                           "source_sha256": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                                             for n in ("run.py", "control.py", "acoustics.py")},
                           "motion_bank_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest()}
    (args.out / "settings.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    warmup()
    print(f"Model and controller ready in {time.perf_counter() - start:.2f}s", flush=True)
    jobs = [(j, args) for j in range(args.n)]
    forecast = {}
    count, false, total = Counter(), Counter(), Counter()
    with (args.out / "cases.jsonl").open("w") as out:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
            for idx, (rows, plans) in enumerate(pool.map(one, jobs), 1):
                forecast.update(plans)
                for row in rows:
                    out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    if row["arm"] != "evaluator_headroom":
                        key = "/".join(row[k] for k in ("task", "initial", "arm"))
                        count[key] += int(row["success"])
                        false[key] += int(row["wrong"])
                        total[key] += 1
                out.flush()
                if idx == 1 or idx % 8 == 0 or idx == len(jobs):
                    print(f"{idx}/{len(jobs)} cases in {time.perf_counter() - start:.1f}s", flush=True)
    summary = {k: dict(n=total[k], correct=count[k], wrong=false[k]) for k in total}
    (args.out / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "plans.json").write_text(json.dumps(forecast, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
