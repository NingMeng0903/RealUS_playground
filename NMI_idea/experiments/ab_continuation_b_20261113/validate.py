"""Prospective paired execution of the frozen public continuation predictor."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

import continuations as replay
from opportunity import cohorts, read_rows
from paired_metrics import summarize
from predictor import ARMS, BASE, Predictor


MODEL = None
TRACE_FIELDS = ("start", "goal", "packet", "confidence", "claim", "eval_only_kind",
                "tcp_mm", "turn_deg", "rail_mm", "clock_tick", "capture_index", "selected_prefix")


def checked_freeze(path):
    freeze = json.loads(path.read_text())
    for name, expected in freeze["files"].items():
        digest = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"frozen source changed: {name}")
    return freeze


def estimate_actions(task, depth, scene, state, prefix_trace, anchor, library):
    began = time.process_time()
    public, actions = replay.public_actions(scene, state, prefix_trace, anchor, depth, library)
    forecast = MODEL.forecast(task, public, actions)
    return public, actions, forecast, time.process_time() - began


def fixed_decision(task, depth, scene, state):
    key = MODEL.fixed[f"{task}/{depth}"]["choice"]
    if key == BASE or state[4] != depth or state[6] == len(scene.targets):
        return BASE
    planned = tuple(map(int, key.split(">")))
    if len(planned) == 1 and planned[0] == replay.mpc_command(scene, state):
        return BASE
    _, _, reserve, at, slot, _, progress, _ = state
    slots = len(scene.potential) - 1 - slot
    single = replay.witness_for(scene, at, slots, progress, reserve, planned[:1])
    return key if single is not None and replay.witness_for(scene, at, slots, progress, reserve, planned) is not None else BASE


def handoff_check(scene, state, tail, selected):
    belief, good, reserve, at, slot, clock, progress, wrong = state
    initial_progress = progress
    handoff = None
    for row in tail:
        start, goal, packet, claim, label, owns = (int(row[i]) for i in (0, 1, 2, 4, 5, 11))
        assert start == at and int(row[10]) == slot
        expected_own = slot - state[4] < len(selected) and (slot == state[4] or progress == initial_progress)
        assert bool(owns) == expected_own
        if owns:
            assert handoff is None and goal == selected[slot - state[4]]
        else:
            context = belief, good, reserve, at, slot, clock, progress, wrong
            if handoff is None:
                handoff = dict(slot=slot, public_state_sha256=replay.state_hash(context))
            assert replay.mpc_command(scene, context) == goal
        edge = scene.edge_costs[at, goal]
        assert replay.control.allowed(scene.edge_costs, at, goal, reserve)
        clock += int(edge[3])
        assert clock == int(row[9])
        previous = belief
        belief, confidence = replay.control.posterior(previous, scene.likelihood, scene.correct,
                                                       clock, scene.to_address[goal], packet)
        assert abs(confidence - row[3]) < 1.e-12
        assert bool(claim) == (packet > 0 and scene.site[goal] == scene.targets[progress]
                               and confidence >= replay.control.CONFIDENCE)
        good, _ = replay.control.advance_joint(previous, good, scene.likelihood, scene.correct,
                                                clock, scene.to_address[goal], packet, bool(claim))
        progress += claim
        wrong = wrong or bool(claim and label != 0)
        reserve, at, slot = reserve - edge, goal, slot + 1
    final = belief, good, reserve, at, slot, clock, progress, wrong
    if handoff is None:
        handoff = dict(slot=slot, public_state_sha256=replay.state_hash(final))
    assert replay.mpc_command(scene, final) == -1
    return handoff, replay.state_hash(final)


def validate_one(job):
    index, args = job
    seed = replay.acoustics.seed_for(replay.NAMESPACES[args.split], index)
    rows = []
    for task in args.tasks:
        scene, detect, cdf = replay.independent_case(task, seed)
        initial_state, _, _, _ = replay.begin(task, seed, detect, cdf, 0)
        for depth in args.depths:
            began = time.process_time()
            state, prefix_trace, anchor, tape = replay.begin(task, seed, detect, cdf, depth)
            prefix_cpu = time.process_time() - began
            public, actions, forecasts, forecast_cpu = estimate_actions(task, depth, scene, state, prefix_trace,
                                                                         anchor, args.library)
            results = {}
            for arm in ARMS:
                began = time.process_time()
                if arm == BASE:
                    chosen = BASE
                elif arm == "fixed":
                    chosen = fixed_decision(task, depth, scene, state)
                else:
                    chosen = MODEL.choose(task, depth, forecasts, arm)
                if arm == "fixed":
                    assert chosen == MODEL.choose(task, depth, forecasts, arm)
                planning = 0. if arm == BASE else time.process_time() - began
                if arm.startswith("B_"):
                    planning += forecast_cpu
                path = actions[chosen]["prefix"]
                value, tail, final = replay.finish(scene, detect, cdf, tape, state, path, prefix_cpu, planning)
                if index < 3 and arm == BASE:
                    replay.compare_shared(scene, detect, cdf, tape, initial_state, prefix_trace, tail, value)
                handoff, replayed_hash = handoff_check(scene, state, tail, path)
                assert final == replayed_hash
                results[arm] = dict(chosen=chosen, continuation=path, measured=value, prediction_cpu_s=planning,
                                    handoff=handoff, terminal_public_state_sha256=final, tail=tail.tolist())
            rows.append(dict(index=index, case_seed=seed, task=task, depth=depth, public=public,
                             tape_sha256=replay.digest(tape), prefix_trace=prefix_trace.tolist(),
                             actions=actions, forecasts={key: v.tolist() for key, v in forecasts.items()}, arms=results))
    return rows


def event_counts(rows, arm):
    counts, routes, wrong = Counter(), Counter(), Counter()
    for row in rows:
        run = row["arms"][arm]
        mpc = row["arms"][BASE]
        tail = run["tail"]
        selected = run["continuation"]
        owned = sum(step[11] for step in tail)
        counts["selected_non_mpc"] += int(run["chosen"] != BASE)
        counts["selected_two_step"] += int(len(selected) == 2)
        counts["executed_second_step"] += int(owned == 2)
        counts["handoff_after_first_claim"] += int(len(selected) == 2 and owned == 1 and tail[0][4] == 1.)
        counts["first_changed"] += int([step[1] for step in tail[:1]] != [step[1] for step in mpc["tail"][:1]])
        counts["route_changed"] += int([step[1] for step in tail] != [step[1] for step in mpc["tail"]])
        if selected:
            routes[run["chosen"]] += 1
            first = row["actions"][run["chosen"]]["features"][0]
            counts["first_alternate_realization"] += int(first["rho"] != row["public"]["rho"])
            counts["first_equal_pose_realization_change"] += int(first["rho"] != row["public"]["rho"]
                and first["site"] == row["public"]["site"] and first["tilt"] == row["public"]["tilt"])
            if owned == 2:
                second = row["actions"][run["chosen"]]["features"][1]
                counts["opposing_same_site_pair"] += int(first["site"] == second["site"] and first["tilt"] != 0
                                                        and first["tilt"] == -second["tilt"])
        bad = {int(step[5]) for step in row["prefix_trace"] + tail if step[4] and step[5] != 0.}
        for label in bad:
            wrong[{1: "other_lumen", 2: "persistent_or_local_false_structure"}[label]] += 1
    return dict(counts=counts, selected_routes=routes, wrong_cases_by_return_kind=wrong)


def report(rows):
    result = {}
    for (task, depth), group in cohorts(rows).items():
        data = {arm: [r["arms"][arm]["measured"] for r in group] for arm in ARMS}
        base = data[BASE]
        effects = {arm: summarize(values, base) for arm, values in data.items() if arm != BASE}
        for arm, summary in effects.items():
            summary["predicted_delta_mean"] = np.mean([r["forecasts"][r["arms"][arm]["chosen"]] for r in group], axis=0).tolist()
            summary["combined_time_equivalent"] = all(abs(v) <= .10 for v in summary["delta_ci95_case_clustered"]["combined_s"])
        result[f"{task}/{depth}"] = dict(n_cases=len(group),
            endpoint_counts={arm: dict(correct=sum(r[0] for r in values), wrong=sum(r[1] for r in values))
                             for arm, values in data.items()},
            mean_measured={arm: dict(zip(replay.FIELDS, np.asarray(values).mean(axis=0).tolist()))
                           for arm, values in data.items()},
            vs_mpc=effects, cohort_vs_fixed=summarize(data["B_cohort"], data["fixed"]),
            mechanism={arm: event_counts(group, arm) for arm in ARMS},
            mean_prediction_cpu_s={arm: float(np.mean([r["arms"][arm]["prediction_cpu_s"] for r in group]))
                                   for arm in ARMS})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    global MODEL
    if not args.report_only:
        if args.freeze is None or args.repo is None:
            parser.error("--freeze and --repo are required for execution")
        freeze = checked_freeze(args.freeze)
        for k, v in freeze["confirmation"].items():
            setattr(args, k, v)
        args.split = "pilot" if args.pilot else "confirmation"
        args.n = 4 if args.pilot else args.n
        args.out.mkdir(parents=True, exist_ok=True)
        bank, sources = replay.initialize(args)
        assert bank == freeze["shared_bank_sha256"]
        assert replay.NAMESPACES["confirmation"] == freeze["confirmation_namespace"]
        MODEL = Predictor(json.loads((args.freeze.parent / "model/predictor.json").read_text()))
        replay.warm()
        settings = dict(n=args.n, tasks=args.tasks, depths=args.depths, bank_sha256=bank, sources=sources,
                        namespace=replay.NAMESPACES[args.split], library=args.library, arms=ARMS,
                        model_size=args.model_size, trace_fields=TRACE_FIELDS, metrics=replay.FIELDS)
        (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        with ((args.out / "cases.jsonl").open("w") as stream,
              ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool):
            for index, rows in enumerate(pool.map(validate_one, ((i, args) for i in range(args.n)))):
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                if index % 32 == 0 or index + 1 == args.n:
                    stream.flush()
                    print(f"{args.split}: {index + 1}/{args.n}", flush=True)
    result = report(read_rows(args.out))
    (args.out / "paired.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for group, row in result.items():
        print(group, row["endpoint_counts"],
              {arm: (round(stats["delta"]["correct_completion"], 4), stats["all_costs_equivalent"])
               for arm, stats in row["vs_mpc"].items()})


if __name__ == "__main__":
    main()
