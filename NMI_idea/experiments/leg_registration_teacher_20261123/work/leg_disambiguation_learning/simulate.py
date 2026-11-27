"""Same-prefix action values and closed-loop outcomes on the 730 mm route."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("LEG_DIAGNOSTICS", HERE.parent))
sys.path[:0] = [str(WORK / "leg_topology_feedback"), str(WORK / "leg_teacher_accuracy")]

import benchmark as ref
from anchor import build_plan, execute as execute_anchor, find_bridge
from environment import Arm, end_reward, spec_for
from history import History, select
from scan_kernel import Trial

MOTION = ref.reference.motion
ATLAS = ref.reference.V2 / "atlas_routes.npz"
ARM = Arm("disambiguation", dird=False)
CHECKS_MM = (930, 1085, 1210)
THRESHOLD = .82
TAPES = {"train": 2, "validate": 2, "test": 3}
SPLITS = {"train": (790_125_223, (0, 1, 2, 5)),
          "validate": (714_030_649, (0, 1, 2, 5)),
          "test": (760_239_167, (3, 4, 6))}


@dataclass(frozen=True)
class Action:
    name: str
    group: str
    ticks: int = 0
    distance: float = 0.
    plan: object = None
    goal: tuple = (0., 0.)
    controls: tuple = (0., 0., 0.)


CONTINUE = Action("continue", "continue")


def rng_seed(*values):
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint64)[0])


def cases(split, size, seed_override=None):
    seed, shapes = SPLITS[split]
    if seed_override is not None:
        seed = int(seed_override)
    shape_rng = np.random.default_rng(seed + 1)
    counts = Counter()
    for case in ref.reference.run.run_cases(size * 96, "development", seed):
        if case.source != 2 or case.task != ref.ROUTE or counts[case.profile] >= size // 2:
            continue
        counts[case.profile] += 1
        yield replace(case, shape=int(shape_rng.choice(shapes)), challenge=False)
        if sum(counts.values()) == size:
            return
    raise RuntimeError("Couldn't assemble both scan profiles")


def public_context(env):
    # Even the simulator's noise profile and shadow realization are private.
    nominal = Trial(2, 0, "stress", n=1)
    forecast = ref.reference.GeometryForecast(env, nominal)
    return History(env, nominal, forecast, oracle_geometry=False), forecast


def echo_state(history, post, region):
    """Summaries of acquired echoes only. Each class includes a missing outcome."""
    indices = [i for i, sample in enumerate(history.samples) if sample.region == region]
    recent = indices[-12:]
    station_evidence = {}
    for i in indices:
        station = int(round(history.samples[i].station))
        score = float(post[i][0].max(initial=0.))
        station_evidence[station] = max(score, station_evidence.get(station, 0.))
    if not recent:
        return np.zeros(18)
    masses = np.zeros(8)
    ambiguity, likelihood, margin, counts = [], [], [], []
    for i in recent:
        probabilities = post[i]
        counts.append(probabilities.shape[1])
        if not probabilities.shape[1]:
            likelihood.append(0.)
            continue
        order = np.argsort(probabilities[0])
        roles = probabilities[:, order[-1]]
        masses += roles
        likelihood.append(float(roles[0]))
        margin.append(float(roles[0] - max(roles[1:])))
        ambiguity.append(float(-(roles * np.log(np.maximum(roles, 1e-12))).sum() / np.log(8.)))
    peaks = sorted(station_evidence.values(), reverse=True)
    return np.r_[peaks[0], peaks[1] if len(peaks) > 1 else 0.,
                 np.mean(likelihood), np.mean(margin) if margin else 0.,
                 np.mean(ambiguity) if ambiguity else 0.,
                 (np.array(peaks) >= THRESHOLD).mean(), np.mean(counts) / 7.,
                 np.mean(np.asarray(counts) == 0), len(peaks) / 25.,
                 (masses / len(recent)), np.argmax(masses[1:6]) / 4.]


def prefix_features(env, node, spec, limits, history):
    region = MOTION.region_of(spec, node.station)
    angle = node.sector * MOTION.ANGLE_STEP
    footprint = np.r_[(node.station - spec[1]) / (spec[2] - spec[1]), region / 5.,
                      np.sin(angle), np.cos(angle), node.tick / limits.ticks,
                      node.distance / limits.mm, np.asarray(limits.probe(node, spec)) / [30., 86.]]
    post = history.smoothed()
    echo = echo_state(history, post, region)
    if len(echo) != 18:
        raise RuntimeError("Unexpected echo-state length")
    previous = echo_state(history, post, region - 1)
    picks = select(history.samples, post, THRESHOLD)
    public_coverage = np.array([0. if pick is None else pick[2] for pick in picks])
    behavior = env.features(node, spec, limits, limits.probe(node, spec), ARM)[:31]
    return footprint.tolist(), np.r_[echo, previous, public_coverage, behavior].tolist()


def atlas_view(env, station, sector):
    points, shade = env.scanner._anatomy.truth(env.scanner.surface, 2, float(station), float(sector % 24))
    visible = (np.abs(points[:, 0]) <= 25.) & (points[:, 1] >= 3.) & (points[:, 1] <= 80.)
    target = float(visible[0]) * (1. - .96 * shade[0])
    rivals = np.flatnonzero(visible[1:]) + 1
    if not visible[0] or not len(rivals):
        separation = 1. if visible[0] else 0.
    else:
        gap = (points[rivals] - points[0]) / [4.7, 4.7]
        separation = 1. - float(np.max(np.exp(-.5 * np.square(gap).sum(axis=1))))
    rival_profile = np.zeros(5)
    for index in rivals:
        if np.isfinite(points[0]).all():
            rival_profile[index - 1] = np.clip(np.linalg.norm(points[index] - points[0]) / 20., 0., 1.)
    return np.r_[target, separation, rival_profile, len(rivals) / 5.]


def bank(env, spec, node, limits, forecast, trail, junction):
    station, sector = int(node.station), int(node.sector)
    caps = limits.probe(node, spec)
    actions = [CONTINUE]
    if caps[0] < 4:
        return actions
    actions.append(Action("dwell", "resample", min(18, caps[0]), goal=(station, sector)))
    for name, group, skill, controls in (
        ("backup", "resample", 2, (0., .65, 0.)),
        ("window_left", "disambiguate", 1, (-.78, .60, -.30)),
        ("window_right", "disambiguate", 1, (.78, -.60, .30)),
        ("trace_left", "disambiguate", 2, (-.65, 1., .25)),
        ("trace_right", "disambiguate", 2, (.65, 1., -.25)),
    ):
        plan = env.plan(node, spec, skill, np.asarray(controls), caps, ARM)
        if plan is None:
            continue
        # A planned turn may be contracted by the common reachability checks.
        delta = abs(MOTION._sector_delta(sector, plan.goal[1]))
        if group == "disambiguate" and delta < .55:
            continue
        actions.append(Action(name, group, plan.ticks, plan.distance, plan, tuple(plan.goal), controls))
    if 3 <= station - junction <= 22:
        link = build_plan(env, spec, node, limits, [(junction, "junction")], forecast, trail, allow_local=False)
        if link is not None:
            actions.append(Action("junction", "disambiguate", link.ticks, link.distance,
                                  link, (link.anchor_station, link.anchor_sector)))
    return actions


def descriptors(env, node, action):
    goal = action.goal if action != CONTINUE else (node.station, node.sector)
    here = atlas_view(env, node.station, node.sector)
    view = atlas_view(env, *goal)
    return np.r_[goal[0] / 160., (goal[0] - node.station) / 25.,
                 MOTION._sector_delta(goal[1], node.sector) / 6.,
                 action.ticks / 70., action.distance / 120., action.controls,
                 view, view - here].tolist()


def get_outcome(frames, history, threshold=THRESHOLD):
    history.append(frames)
    posterior = history.smoothed()
    picks = select(history.samples, posterior, threshold)
    result = ref.evaluate_frames(frames, picks)
    true, rival = np.zeros(6), np.zeros(6)
    capture = np.zeros(6, bool)
    for frame, post in zip(frames, posterior, strict=True):
        region, labels = frame[5], frame[3]
        target = labels == 0
        capture[region] |= bool(target.any())
        true[region] = max(true[region], float(post[0, target].max(initial=0.)))
        rival[region] = max(rival[region], float(post[0, ~target].max(initial=0.)))
    result["captured_regions"] = int(capture.sum())
    result["soft_margin"] = float((true - rival).mean())
    # Smooth supervision is present even on rejected echoes and failed scans.
    result["score"] = float(end_reward(result) + .30 * result["soft_margin"])
    return result


def run_action(env, spec, prefix, trial, limits, action, stop=None):
    node = prefix.fork()
    caps = limits.probe(node, spec)
    if action.name in ("dwell", "matched_dwell"):
        for _ in range(action.ticks):
            node.tick += 1
            env.scanner._observe(node, spec, trial, node.station, node.sector, node.tick)
    elif action.name == "junction":
        execute_anchor(env, spec, node, trial, limits, action.plan)
    elif action.plan is not None:
        event = env.material.run_probe(node, spec, trial, action.plan, *caps, feedback=True, stop=stop,
                                       use_dird=False, copy_node=False)
        node = event.node
        if not event.legal or (node.station, node.sector) != (prefix.station, prefix.sector):
            raise RuntimeError(f"Invalid executed motion {action.name}: {event.reason}")
    return node


def branch(env, spec, prefix, trial, limits, action, history, check_images=False, stop=None):
    node = run_action(env, spec, prefix, trial, limits, action, stop)
    action_ticks = node.tick - prefix.tick
    action_mm = node.distance - prefix.distance
    last_probe_frame = len(node.reader.frames)
    ref.finish(env, node, spec, trial, limits)
    result = get_outcome(node.reader.frames, history.fork())
    result.update(ticks=node.tick, distance=node.distance, action_ticks=action_ticks, action_mm=action_mm)
    if check_images and action_ticks:
        frames = node.reader.frames[:len(prefix.reader.frames)] + node.reader.frames[last_probe_frame:]
        hidden = get_outcome(frames, history.fork())
        result["without_probe_images"] = {k: hidden[k] for k in
                                          ("region_outcomes", "correct_regions", "wrong_regions", "pass", "score")}
    return result


def reference_path(env, spec, node, trial, limits, stations):
    prefixes, trail = {}, {}
    while True:
        station = int(node.station)
        trail[station] = int(node.sector)
        if station in stations:
            prefixes[station] = (node.fork(), trail.copy())
        if station == spec[2]:
            return node, prefixes
        env.route.advance(node, spec, trial, limits)


def one_world(job):
    case, split, junction, draws = job
    env = ref.reference.load_environment(str(ATLAS), case.shape)
    spec = spec_for(env.scanner, case)
    node, trial, limits = env.start(case, spec)
    stations = [int(np.abs(env.scanner.stations + mm).argmin()) for mm in CHECKS_MM]
    final, prefixes = reference_path(env, spec, node, trial, limits, stations)
    history, forecast = public_context(env)
    base = get_outcome(final.reader.frames, history.fork())
    base.update(ticks=final.tick, distance=final.distance, action_ticks=0, action_mm=0.)
    records = []
    compare_actual = split != "train"
    for station in stations:
        prefix, trail = prefixes[station]
        history.append(prefix.reader.frames)
        position, echoes = prefix_features(env, prefix, spec, limits, history)
        actions = bank(env, spec, prefix, limits, forecast, trail, junction)
        rolls = [trial.branch(prefix.tick, rng_seed(case.world_seed, station, i, 2084)) for i in range(draws)]
        probe_rows = []
        timing_controls = {}
        for action in actions:
            # No row is filtered by eventual correctness, score or feasibility of winning.
            expected = [branch(env, spec, prefix, tape, limits, action, history) for tape in rolls]
            row = dict(name=action.name, group=action.group, descriptors=descriptors(env, prefix, action),
                       rollout_scores=[round(e["score"], 6) for e in expected])
            if compare_actual:
                actual = (base if action.name == "continue" else
                          branch(env, spec, prefix, trial, limits, action, history, check_images=split == "test"))
                row["actual"] = actual
                if split == "test" and action.group == "disambiguate":
                    delay = actual["action_ticks"]
                    if delay not in timing_controls:
                        timing_controls[delay] = branch(
                            env, spec, prefix, trial, limits, Action("matched_dwell", "resample", delay), history)
                    row["matched_dwell"] = timing_controls[delay]
            probe_rows.append(row)
        records.append(dict(station=station, nominal_mm=-float(env.scanner.stations[station]),
                            region=MOTION.region_of(spec, station), position=position, echoes=echoes,
                            actions=probe_rows))
    world_fingerprint = sha256(np.nan_to_num(trial.layout.centers).tobytes() + bytes([case.shape])).hexdigest()
    return dict(case=case.identifier, layout=world_fingerprint, world_seed=case.world_seed,
                tape_seed=case.tape_seed, profile=case.profile, shape=case.shape, base=base, prefixes=records)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cases <= 0 or args.cases % 2:
        parser.error("--cases must be positive and even")
    manifest = json.loads((ref.reference.V2 / "atlas_routes_manifest.json").read_text())
    if ref.reference.run.digest(ATLAS) != manifest["pack_sha256"]:
        raise RuntimeError("Atlas bytes differ from the registered route")
    grid = ref.reference.motion._load_pack(ATLAS)["neutral_grid"]
    junction = find_bridge(manifest, (0, 9, len(grid) - 1), grid)
    draws = args.draws or TAPES[args.split]
    jobs = [(case, args.split, junction, draws) for case in cases(args.split, args.cases, args.seed)]
    with ProcessPoolExecutor(args.workers, initializer=ref.reference.run.protect_holdout) as pool:
        with args.out.open("w") as output:
            for index, row in enumerate(pool.map(one_world, jobs), start=1):
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                if index % 4 == 0 or index == len(jobs):
                    print(f"{args.split}: {index}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
