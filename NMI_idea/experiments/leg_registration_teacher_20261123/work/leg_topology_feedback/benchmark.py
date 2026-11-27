"""Paired scanning-only experiment on the named 730 mm atlas route."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TEACHER = HERE / "stage1" if (HERE / "stage1").is_dir() else HERE.parent / "leg_teacher_accuracy"
sys.path.insert(0, os.environ.get("LEG_STAGE1_CODE", str(DEFAULT_TEACHER)))
import teacher_benchmark as reference
from environment import spec_for
from history import History, select
from anchor import execute, find_bridge, find_prefix

ROUTE = "groin_to_distal_calf"


def choose_cases(group, size):
    seeds = {"diagnostic": reference.run.DEV_SEED, "verification": 395_780_013,
             "verification_2": 284_611_973, "fresh": 918_027_451, "fresh_2": 314_022_891}
    counts = Counter()
    cases = []
    for case in reference.run.run_cases(size * 48, "development", seeds[group]):
        if case.task != ROUTE or case.source != 2 or counts[case.profile] >= size // 2:
            continue
        cases.append(replace(case, challenge=False))
        counts[case.profile] += 1
        if len(cases) == size:
            return cases
    raise RuntimeError("Could not construct both profile groups")


def finish(env, node, spec, trial, limits, *, unobserved=0):
    while node.station < spec[2]:
        env.route.advance(node, spec, trial, limits)
    if (node.tick > limits.ticks or node.distance > limits.mm + 1e-6
            or len(node.reader.frames) + unobserved != node.tick or node.station != spec[2]):
        raise RuntimeError("A branch violated the common terminal or measurement budget")
    return node


def evaluate_frames(frames, choices):
    metrics = reference.selection_metrics(frames, choices)
    metrics["chosen_labels"] = [None if c is None else int(frames[c[0]][3][c[1]]) for c in choices]
    metrics["chosen_ticks"] = [None if c is None else int(frames[c[0]][8]) for c in choices]
    return metrics


def summary(node, history, threshold):
    frames = node.reader.frames
    history.append(frames)
    filtered = evaluate_frames(frames, select(history.samples, history.online, threshold))
    smoothed = evaluate_frames(frames, select(history.samples, history.smoothed(), threshold))
    return {"filter": filtered, "history": smoothed,
            "capture_all": int(len({f[5] for f in frames if 0 in f[3]}) == 6),
            "ticks": node.tick, "distance": node.distance, "complete": True}


def run_branch(arm, env, spec, node, trial, limits, plan):
    if arm != "fixed":
        execute(env, spec, node, trial, limits, plan, repeat=arm != "anchor", mute=arm == "idle")
    return finish(env, node, spec, trial, limits, unobserved=plan.ticks if arm == "idle" else 0)


def teacher_choice(env, spec, prefix, history, limits, plan, threshold, trials, trial):
    arms = ("fixed", "repeat", "anchor")
    utilities = {arm: [] for arm in arms}
    for draw in range(trials):
        # The simulated choice consumes different future innovations from all actual branches.
        seed = int(np.random.SeedSequence([trial.world_seed, prefix.tick, draw, 98813]).generate_state(
            1, dtype=np.uint64)[0])
        alternative = reference.Trial(trial.source, seed, trial.profile, world_seed=trial.world_seed, split=trial.split)
        alternative.layout, alternative.window = trial.layout, trial.window
        for arm in arms:
            node = run_branch(arm, env, spec, prefix.fork(), alternative, limits, plan)
            result = summary(node, history.fork(), threshold)["history"]
            score = 2.4 * result["pass"] + .48 * result["correct_regions"] - .77 * result["wrong_regions"]
            score -= .30 * result["wrong_delivery"]
            utilities[arm].append(score)
    # Compare all choices with the same budget and the same future tapes.
    # Ties follow the common controller.
    winner = max(arms, key=lambda arm: np.mean(utilities[arm]))
    return winner, utilities


def simulate(job):
    case, pack, threshold, tapes, config = job
    env = reference.load_environment(str(pack), case.shape)
    outputs = []
    for index in range(tapes):
        sensor_case = replace(case, tape_seed=case.tape_seed + index * 1_002_003_007)
        spec = spec_for(env.scanner, sensor_case)
        node, trial, limits = env.start(sensor_case, spec)
        forecast = reference.GeometryForecast(env, trial)
        output = {"case": case.identifier, "shape": case.shape, "profile": case.profile,
                  "layout": trial.layout.identifier, "tape_seed": int(sensor_case.tape_seed),
                  "limits": [limits.ticks, limits.mm]}
        history = History(env, trial, forecast, correlation_scale=config["corr"])
        plan = None
        if config["action"]:
            bridge = config["bridge"]
            planning_trial = reference.Trial(case.source, 0, case.profile, n=1)
            prior_forecast = reference.GeometryForecast(env, planning_trial)
            node, plan, trigger = find_prefix(env, spec, node, trial, limits, history, prior_forecast, bridge,
                                              threshold, allow_local=config["action"] == "both",
                                              trigger=config["trigger"])
            output["trigger"] = trigger
        if plan is not None:
            output["branch"] = {"anchor_station": float(env.scanner.stations[plan.anchor_station]),
                                "anchor_sector": plan.anchor_sector, "kind": plan.kind,
                                "alternate_return": plan.alternate_return, "extra_ticks": plan.ticks,
                                "extra_distance_mm": plan.distance,
                                "entry_station": float(env.scanner.stations[int(node.station)]),
                                "entry_sector": int(node.sector), "prefix_frames": len(node.reader.frames),
                                "round_trip_stations": [float(x) for x in plan.poses[:, 0]],
                                "round_trip_sectors": [float(x % 24.) for x in plan.poses[:, 1]]}
            for arm in ("idle", "repeat", "anchor"):
                branch = run_branch(arm, env, spec, node.fork(), trial, limits, plan)
                output[arm] = summary(branch, history.fork(), threshold)
                if arm == "anchor":
                    frames = [frame for frame in branch.reader.frames
                              if frame[8] <= node.tick or frame[8] > node.tick + plan.ticks]
                    posthoc = history.fork()
                    posthoc.append(frames)
                    output["anchor"]["without_probe_readout"] = evaluate_frames(
                        frames, select(posthoc.samples, posthoc.smoothed(), threshold))
            if config["screen"]:
                choice, utilities = teacher_choice(env, spec, node, history, limits, plan, threshold,
                                                   config["screen"], trial)
                output["teacher_choice"] = {"choice": choice, "utilities": utilities}
        node = finish(env, node, spec, trial, limits)
        original = reference.teacher_selections(node.reader.frames,
                                                 lambda f: forecast.joint_marginal(f, np.empty(0)), threshold)
        output["fixed"] = summary(node, history, threshold)
        output["fixed"]["independent"] = evaluate_frames(node.reader.frames, original)
        if config["atlas"]:
            atlas = History(env, trial, forecast, oracle_geometry=False, correlation_scale=config["corr"])
            output["atlas_only"] = summary(node, atlas, threshold)
        outputs.append(output)
    return outputs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["diagnostic", "verification", "verification_2", "fresh", "fresh_2"],
                        default="diagnostic")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--tapes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--corr", type=float, default=1.)
    parser.add_argument("--atlas", action="store_true")
    parser.add_argument("--action", choices=["junction", "both"])
    parser.add_argument("--screen", type=int, default=0)
    parser.add_argument("--trigger", choices=["local", "coverage"], default="local")
    args = parser.parse_args()
    if args.cases <= 0 or args.cases % 2 or not 0 <= args.corr <= 1 or args.tapes <= 0:
        parser.error("Expected positive even --cases, positive --tapes and correlation scale in [0, 1]")
    pack = reference.V2 / "atlas_routes.npz"
    manifest = json.loads((reference.V2 / "atlas_routes_manifest.json").read_text())
    if reference.run.digest(pack) != manifest["pack_sha256"]:
        raise RuntimeError("Named atlas pack hash changed")
    threshold = json.loads((reference.LEGACY / "smoke/verification.json").read_text())["threshold"]
    if args.screen < 0 or (args.screen and not args.action):
        parser.error("--screen requires an anchor action and a positive trial count")
    cases = choose_cases(args.group, args.cases)
    ref = reference.motion._load_pack(pack)["neutral_grid"]
    route_manifest = manifest["routes"][ROUTE]
    indices = np.flatnonzero((ref <= route_manifest["neutral_start_mm"])
                             & (ref >= route_manifest["neutral_end_mm"]))
    bridge = find_bridge(manifest, (0, int(indices[0]), int(indices[-1])), ref)
    config = {"corr": args.corr, "atlas": args.atlas, "action": args.action, "screen": args.screen,
              "bridge": bridge, "trigger": args.trigger}
    jobs = [(case, pack, threshold, args.tapes, config) for case in cases]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                             initializer=reference.run.protect_holdout) as pool:
        rows = [row for chunk in pool.map(simulate, jobs) for row in chunk]
    print(json.dumps({"group": args.group, "config": config, "threshold": threshold,
                      "atlas_sha256": manifest["pack_sha256"], "rows": rows},
                     default=lambda scalar: scalar.item(), indent=2))


if __name__ == "__main__":
    main()
