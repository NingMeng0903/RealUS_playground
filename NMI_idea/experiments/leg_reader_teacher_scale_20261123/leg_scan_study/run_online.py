"""Train with actual rollouts, freeze on atlas/CT validation, and evaluate Blender."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import time

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
import torch

import motion
from online_reader import PublicHistoryReader
from policy import ScanPolicy, update_ppo
from teacher import fit_teacher, review_utility
from soft_identity import region_potential, update_identity


TASKS = {
    "saphenofemoral": (-525, -590, (0, 1, 2, 3)),
    "femoral": (-595, -790, (0, 1, 2, 3)),
    "popliteal": (-835, -875, (0, 1, 2, 3)),
    "calf": (-980, -1255, (0, 1, 2, 3)),
    "groin_to_distal_calf": (-525, -1255, (0, 1, 2, 3, 4)),
}
PROFILES = ("ordinary", "stress")
PROBE_TICKS, PROBE_MM = 104, 270.
ACTION_TICKS, ACTION_MM = 32, 85.
MAX_PROBES = 5
WIDTH = 30.
GATE_THRESHOLDS = (.5, .62, .74)
TEST_ARMS = ("mpc", "active_mpc", "prodmp_rule", "prodmp_random",
             "prodmp_matched", "ppo_ungated", "ppo_gated")
RESOURCE_FEATURES = (
    "probe_tick_fraction", "probe_distance_fraction", "hard_time_fraction",
    "return_time_fraction", "accepted_probe_fraction", "captures_in_region",
)
FEATURE_NAMES = (*motion.FEATURE_NAMES, *RESOURCE_FEATURES)
HISTORY_CONFIDENCE = FEATURE_NAMES.index("history_confidence")
HISTORY_SUPPORT = FEATURE_NAMES.index("history_support")
DISAGREEMENT = FEATURE_NAMES.index("disagreement")


@dataclass(frozen=True)
class Case:
    identifier: str
    partition: str
    source: int
    shape: int
    task: str
    profile: str
    tape_seed: int
    action_seed: int


@dataclass(frozen=True)
class Job:
    case: Case
    arms: tuple
    version: str = ""
    weights: dict | None = None
    ungated_threshold: float = .5
    gated_threshold: float | None = .5
    full_log: bool = True
    teacher: bool = False


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=json_value, allow_nan=False) + "\n")


def line(stream, data):
    stream.write(json.dumps(data, separators=(",", ":"), default=json_value, allow_nan=False) + "\n")


def region_of(spec, station):
    index = min(int(round(station)) - spec[1], spec[2] - spec[1])
    return int(spec[4][max(0, index)])


def make_spec(scanner, case):
    if case.partition == "blender":
        return motion.holdout_specs(scanner)[case.task]
    if case.source == 3:
        lo, hi = map(int, scanner.ct_bounds)
        return motion.definition(0, lo, hi)
    hi, lo, roles = TASKS[case.task]
    stations = np.asarray(scanner.stations)
    route = np.flatnonzero((stations <= hi) & (stations >= lo))
    return motion.definition(0, int(route[0]), int(route[-1]), roles)


def init_worker(pack, legacy):
    global CONFIG, WORKER_POLICY, WORKER_VERSION
    CONFIG = Path(pack), legacy if legacy is None else Path(legacy)
    WORKER_POLICY, WORKER_VERSION = None, None
    torch.set_num_threads(1)
    scanners.cache_clear()
    motion.Reader = PublicHistoryReader


@lru_cache(maxsize=8)
def scanners(shape, holdout):
    pack, legacy = CONFIG
    if holdout:
        if legacy is None:
            raise ValueError("Blender geometry is unavailable outside the evaluation stage")
        return motion.Scanner.holdout(pack, legacy)
    return motion.Scanner.training(pack, shape=shape, target=0)


def weights_of(policy):
    return {name: tensor.detach().cpu().numpy().copy()
            for name, tensor in policy.state_dict().items()}


def install_policy(job):
    global WORKER_POLICY, WORKER_VERSION
    if job.weights is None:
        return None
    if WORKER_VERSION != job.version:
        WORKER_POLICY = ScanPolicy(len(FEATURE_NAMES))
        WORKER_POLICY.load_state_dict({name: torch.as_tensor(value)
                                        for name, value in job.weights.items()})
        WORKER_POLICY.eval()
        WORKER_VERSION = job.version
    return WORKER_POLICY


def states_for_region(reader, region):
    witness = reader.select(region)
    if witness is None:
        return -1
    frame, slot = witness[:2]
    return int(reader.frames[frame][3][slot] == 0)


def coverage(reader):
    states = [states_for_region(reader, region) for region in range(6)]
    return {
        "pass": int(all(value == 1 for value in states)),
        "correct_regions": states.count(1),
        "wrong_regions": states.count(0),
        "no_result_regions": states.count(-1),
        "wrong_delivery": int(0 in states),
        "region_outcomes": states,
    }


def reward_term(metrics):
    return (2.5 * metrics["pass"] + .45 * metrics["correct_regions"]
            - .80 * metrics["wrong_regions"] - .40 * metrics["wrong_delivery"])


def qualified_probe(features):
    history_agrees = features[DISAGREEMENT] < .5
    confident = (features[HISTORY_CONFIDENCE] >= .77
                 and features[HISTORY_SUPPORT] >= .30 and history_agrees)
    return not confident


def opportunity(spec, station):
    if station == spec[2]:
        return True
    this_region = region_of(spec, station)
    return any(region_of(spec, station + k) != this_region
               for k in range(1, min(3, spec[2] - station) + 1))


def context(env, node, spec, bank_ticks, bank_mm, hard_ticks, reserve, probes):
    ticks, _ = reserve
    remaining = float(ticks[int(node.station) - spec[1]])
    raw = np.asarray(motion.public_features(node, spec), dtype=np.float32)
    raw[HISTORY_SUPPORT] = node.reader.details(region_of(spec, node.station))[1]
    if raw.shape != (len(motion.FEATURE_NAMES),):
        raise ValueError(f"Unexpected observation shape {raw.shape}")
    acquired = sum(frame[5] == region_of(spec, node.station) for frame in node.reader.frames)
    extras = [bank_ticks / PROBE_TICKS, bank_mm / PROBE_MM,
              (hard_ticks - node.tick) / hard_ticks, remaining / hard_ticks,
              probes / MAX_PROBES, min(acquired / 12., 1.)]
    return np.nan_to_num(np.r_[raw, extras].astype(np.float32), nan=0., posinf=2., neginf=-2.)


def action_grid():
    return np.array(list(itertools.product((-.8, -.4, .4, .8), (-.7, 0., .7), (-.6, .6))))


def predictive_action(scanner, node, spec, rng=None, *, optional=False):
    controls = action_grid()
    if rng is not None:
        return controls[int(rng.integers(len(controls)))]
    s, a = int(node.station), int(node.sector)
    reader = node.reader
    priority = np.full(len(controls), -np.inf)
    for i, (angle, axial, shape) in enumerate(controls):
        dst_s = int(np.clip(s + round(motion.AXIAL_RANGE * axial), spec[1], spec[2]))
        dst_a = (a + round(motion.ANGLE_RANGE * angle)) % 24
        if not scanner.valid[dst_s, dst_a]:
            continue
        q = reader.model.feature(dst_s, dst_a, reader.w)
        detect, sep, _, shadow, window, merge = q
        merit = detect + .19 * sep + .12 * window - .15 * shadow - .13 * merge
        priority[i] = merit - .04 * abs(angle) - .035 * abs(axial) - .005 * shape
    best = int(np.argmax(priority))
    if optional:
        now = reader.model.feature(s, a, reader.w)
        current = now[0] + .19 * now[1] + .12 * now[4] - .15 * now[3] - .13 * now[5]
        ambiguity = .08 * (1. - max(reader.frames[-1][6], default=0.))
        if priority[best] + ambiguity < current + .04:
            return None
    return controls[best]


def arm_decision(arm, scanner, node, spec, features, eligible, policy, rng, job):
    if arm in ("train", "ppo_ungated", "ppo_gated", "ppo_no_teacher", "prodmp_matched"):
        threshold = job.ungated_threshold if arm != "ppo_gated" else job.gated_threshold
        forced_mpc = threshold is None
        decision = policy.choose(features, eligible and not forced_mpc,
                                 rng if arm == "train" else None,
                                 .5 if forced_mpc else threshold)
        action = (predictive_action(scanner, node, spec)
                  if arm == "prodmp_matched" and decision.recheck else decision.action)
        return decision.recheck, action, decision
    active = eligible and arm in ("active_mpc", "prodmp_rule", "prodmp_random")
    action = (predictive_action(scanner, node, spec, rng if arm == "prodmp_random" else None,
                                optional=arm in ("active_mpc", "prodmp_rule"))
              if active else None)
    return action is not None, np.zeros(3) if action is None else action, None


def frame_logs(reader, first, last, case_id, arm, step):
    observed, labels = [], []
    for index in range(first, last):
        station, sector, z, truth, q, region, probability, unknown, tick = reader.frames[index]
        observed.append({"episode": case_id, "arm": arm, "step": step, "frame": index,
                         "station_index": station, "sector_index": sector, "tick": tick,
                         "region": region, "echoes_mm": z, "frame_quality": q,
                         "target_posterior": probability, "unknown_posterior": unknown})
        labels.append({"episode": case_id, "arm": arm, "frame": index, "labels": truth})
    return observed, labels



def teacher_annotation(scanner, spec, trial, case, actual, prefixes):
    row = {"episode": case.identifier, "source": case.source, "shape": case.shape,
           "profile": case.profile, "student_tape_seed": case.tape_seed,
           "eligible_prefix": bool(prefixes), "branches": []}
    if not prefixes:
        return row
    rng = np.random.default_rng(np.random.SeedSequence([case.action_seed, 3801]))
    want_success = bool(rng.random() < .20)
    regions = sorted(prefixes)
    matching = [r for r in regions if (actual["region_outcomes"][r] == 1) == want_success]
    region = int(rng.choice(matching or regions))
    prefix, features, tick_cap, mm_cap, student_control = prefixes[region]
    independent_seed = case.action_seed + 1_700_000_000
    tapes = (trial, motion.Trial(case.source, independent_seed, case.profile))
    initial_potential = region_potential(prefix.reader, region)
    horizon = min(int(prefix.station) + 3, spec[2])
    actions = [("continue", None), ("public_predictive", predictive_action(scanner, prefix, spec))]
    if student_control is not None:
        actions.append(("student_action", student_control))
    for i, angle in enumerate(rng.choice((-.60, -.32, .32, .60), size=3, replace=False)):
        control = np.array([angle, rng.uniform(-.60, .45), rng.uniform(-.75, .75)])
        actions.append((f"sample_{i}", control))

    for name, control in actions:
        branch = {"candidate": name, "controls": control, "replays": []}
        for tape in tapes:
            node = prefix.fork()
            reason, legal, feedback = "continue", True, 0
            motion_ticks, motion_distance = 0, 0.
            if control is not None:
                act = motion.ContinuousAction(*(float(v) for v in control))
                observed = scanner.prodmp(node, spec, tape, act, max_ticks=tick_cap, max_mm=mm_cap)
                node, reason, legal = observed.node, observed.reason, observed.legal
                feedback, motion_ticks, motion_distance = (observed.feedback_updates,
                                                            observed.ticks, observed.distance)
            while node.station < horizon:
                node = scanner.advance(node, spec, tape)
            outcome = review_utility(node, region, len(prefix.reader.frames), prefix.tick, prefix.distance, initial_potential)
            branch["replays"].append(outcome | {
                "legal": legal, "reason": reason, "feedback_updates": feedback,
                "action_ticks": motion_ticks, "action_distance_mm": motion_distance,
                "new_frames": len(node.reader.frames) - len(prefix.reader.frames),
            })
        row["branches"].append(branch)
    baseline = row["branches"][0]["replays"]
    robust = []
    for branch in row["branches"][1:]:
        cases = branch["replays"]
        gains = [a["utility"] - b["utility"] for a, b in zip(cases, baseline)]
        strict = [a["correct"] - b["correct"] + b["wrong"] - a["wrong"]
                  for a, b in zip(cases, baseline)]
        branch["paired_gain"] = gains
        branch["strict_pair_changes"] = strict
        branch["robust_positive"] = (all(a["legal"] for a in cases)
                                     and min(gains) >= .035 and min(strict) >= 0)
        if branch["robust_positive"]:
            robust.append(branch)
    best = max(robust, key=lambda b: (min(b["paired_gain"]), np.mean(b["paired_gain"])), default=None)
    requested_full_tail = bool(rng.random() < .25)
    tail_check, best_before_tail = None, None if best is None else best["candidate"]
    if requested_full_tail and best is not None:
        tail_check = verify_teacher_tail(scanner, spec, prefix, tapes, best["controls"], tick_cap, mm_cap)
        if not tail_check["accepted"]:
            best = None
    return row | {
        "full_tail_requested": requested_full_tail, "full_tail_check": tail_check,
        "best_local_candidate": best_before_tail, "selected_region": region, "student_outcome": actual["region_outcomes"][region],
        "success_control_requested": want_success, "prefix_station": prefix.station,
        "prefix_sector": prefix.sector, "prefix_tick": prefix.tick,
        "prefix_frames": len(prefix.reader.frames), "future_horizon_station": horizon,
        "student_prefix_action": student_control, "independent_tape_seed": independent_seed,
        "features": features, "target_skill": int(best is not None),
        "positive_branch_count": len(robust),
        "target_controls": best["controls"] if best is not None else np.zeros(3),
        "target_candidate": best["candidate"] if best is not None else "continue",
    }


def verify_teacher_tail(scanner, spec, prefix, tapes, controls, tick_cap, mm_cap):
    replays = []
    for tape in tapes:
        entries = []
        for use_action in (False, True):
            node = prefix.fork()
            if use_action:
                command = motion.ContinuousAction(*(float(v) for v in controls))
                result = scanner.prodmp(node, spec, tape, command, max_ticks=tick_cap, max_mm=mm_cap)
                if not result.legal:
                    return {"accepted": False, "reason": result.reason, "replays": replays}
                node = result.node
            while node.station < spec[2]:
                node = scanner.advance(node, spec, tape)
            terminal = coverage(node.reader)
            score = reward_term(terminal) + sum(region_potential(node.reader, r) for r in range(6))
            score -= .006 * (node.tick - prefix.tick) + .001 * (node.distance - prefix.distance)
            entries.append({"utility": score, "strict": terminal, "end_tick": node.tick})
        base, active = entries
        replays.append({"continue": base, "active": active,
                        "utility_difference": active["utility"] - base["utility"],
                        "wrong_increase": active["strict"]["wrong_regions"] - base["strict"]["wrong_regions"],
                        "pass_change": active["strict"]["pass"] - base["strict"]["pass"]})
    accepted = all(r["utility_difference"] >= -.05 and r["wrong_increase"] <= 0
                   and r["pass_change"] >= 0 for r in replays)
    return {"accepted": accepted, "replays": replays}

def run_arm(scanner, spec, trial, model, policy, job, arm):
    case = job.case
    rng = np.random.default_rng(case.action_seed)
    node = scanner.start(spec, trial, model)
    if node is None or not node.reader.frames:
        raise RuntimeError(f"No initial observation for {case.identifier}")
    n_start = len(node.reader.frames)
    reserve = tuple(np.asarray(part) for part in scanner.route_reserve(spec))
    reserve_ticks, reserve_mm = reserve
    hard_ticks = int(node.tick + reserve_ticks[0] + PROBE_TICKS)
    hard_mm = float(node.distance + reserve_mm[0] + PROBE_MM)
    if hard_ticks >= len(trial.tape.normal):
        raise ValueError(f"Budget {hard_ticks} exceeds the common tape for {case.identifier}")
    bank_ticks, bank_mm = PROBE_TICKS, PROBE_MM
    captured = set()
    potentials = np.zeros(6)
    update_identity(node.reader, spec, 0, n_start, potentials)
    steps, obs_rows, label_rows = [], [], []
    proposed, accepted, physically_rejected = 0, 0, 0
    visited_prefixes = {}
    totals = {"probe_ticks": 0, "probe_distance_mm": 0., "opportunities": 0, "eligible": 0,
              "main_route_advances": 0, "probe_frames": 0, "initial_frames": n_start}
    if job.full_log:
        a, b = frame_logs(node.reader, 0, n_start, case.identifier, arm, -1)
        obs_rows.extend(a)
        label_rows.extend(b)
    for frame in node.reader.frames:
        if 0 in frame[3]:
            captured.add(int(frame[5]))

    done = False
    while not done:
        idx = int(node.station) - spec[1]
        start_station, start_sector, first = node.station, node.sector, len(node.reader.frames)
        tick, distance = node.tick, node.distance
        initial_count = len(captured)
        features = context(scanner, node, spec, bank_ticks, bank_mm, hard_ticks, reserve, accepted)
        checkpoint = opportunity(spec, int(node.station))
        available_ticks = int(min(bank_ticks, ACTION_TICKS, hard_ticks - tick - reserve_ticks[idx]))
        available_mm = float(min(bank_mm, ACTION_MM, hard_mm - distance - reserve_mm[idx]))
        eligible = (checkpoint and accepted < MAX_PROBES and available_ticks >= 5
                    and available_mm >= 6. and qualified_probe(features))
        totals["opportunities"] += int(checkpoint)
        totals["eligible"] += int(eligible)
        selected, controls, decision = arm_decision(
            arm, scanner, node, spec, features, eligible, policy, rng, job,
        )
        if job.teacher and arm == "train" and eligible:
            region = region_of(spec, node.station)
            visited_prefixes[region] = (node.fork(), features.copy(), available_ticks, available_mm,
                                        controls.copy() if selected else None)
        event = None
        if selected:
            proposed += 1
            controls = [float(x) for x in controls]
            action = motion.ContinuousAction(*controls)
            executor = scanner.mpc if arm == "active_mpc" else scanner.prodmp
            event = executor(node, spec, trial, action, max_ticks=available_ticks, max_mm=available_mm)
            if event.legal:
                if event.node is node or event.node.station != start_station or event.node.sector != start_sector:
                    raise AssertionError("A probe must return to its original main-route pose")
                node = event.node
                accepted += 1
                bank_ticks -= event.ticks
                bank_mm -= event.distance
                totals["probe_ticks"] += event.ticks
                totals["probe_distance_mm"] += event.distance
                totals["probe_frames"] += len(node.reader.frames) - first
            else:
                if event.node is not node or node.tick != tick or len(node.reader.frames) != first:
                    raise AssertionError("A rejected action may not consume observations or time")
                physically_rejected += 1
        probe_after = len(node.reader.frames)
        if node.station == spec[2]:
            done = True
        else:
            node = scanner.advance(node, spec, trial)
            totals["main_route_advances"] += 1
            if node.station != start_station + 1:
                raise AssertionError("The common continuation must advance one station")
        if node.tick > hard_ticks or node.distance > hard_mm + 1e-5:
            raise AssertionError(f"Shared hard budget exceeded for {case.identifier}")

        last = len(node.reader.frames)
        for frame in node.reader.frames[first:last]:
            if 0 in frame[3]:
                captured.add(int(frame[5]))
        observable = context(scanner, node, spec, bank_ticks, bank_mm, hard_ticks, reserve, accepted)
        elapsed, travelled = node.tick - tick, node.distance - distance
        soft_gain = update_identity(node.reader, spec, first, last, potentials)
        capture_gain = .075 * (len(captured) - initial_count)
        movement_cost = .006 * elapsed + .001 * travelled
        intervention_cost = (.023 if event.legal else .047) if selected else 0.
        terminal_reward = 0.
        if done:
            metrics = coverage(node.reader)
            terminal_reward = reward_term(metrics)
        reward = soft_gain + capture_gain - movement_cost - intervention_cost + terminal_reward
        step = {
            "episode": case.identifier, "arm": arm, "step": len(steps),
            "station": start_station, "sector": start_sector, "tick": tick,
            "checkpoint": checkpoint, "eligible": eligible,
            "probe_tick_cap": available_ticks, "probe_distance_cap_mm": available_mm,
            "features": features, "selected_skill": "recheck" if selected else "continue",
            "controls": controls if selected else None,
            "latent": decision.latent if decision else np.zeros(3),
            "skill_probability": decision.probability if decision else None,
            "log_probability": decision.log_probability if decision else 0.,
            "value": decision.value if decision else 0.,
            "action_accepted": event is not None and event.legal,
            "action_reason": event.reason if event is not None else None,
            "feedback_updates": event.feedback_updates if event is not None else 0,
            "control_fraction": getattr(event, "applied_fraction", None) if event is not None else None,
            "frame_start": first, "frame_after_probe": probe_after, "frame_stop": last,
            "next_features": observable, "next_tick": node.tick, "next_station": node.station,
            "reward": reward, "soft_identity_gain": soft_gain, "capture_reward": capture_gain,
            "movement_cost": movement_cost, "intervention_cost": intervention_cost,
            "terminal_identity_reward": terminal_reward, "done": done,
        }
        if job.full_log:
            if event is not None and event.legal:
                step["probe_trajectory"] = event.trajectory
            a, b = frame_logs(node.reader, first, last, case.identifier, arm, len(steps))
            obs_rows.extend(a)
            label_rows.extend(b)
        steps.append(step)
    metrics |= {
        "captured_regions": len(captured), "frames": len(node.reader.frames), "reward": sum(s["reward"] for s in steps),
        "soft_identity_potential": float(potentials.sum()),
        "soft_identity_gain": sum(s["soft_identity_gain"] for s in steps),
        "time_s": node.tick / 10., "travel_mm": node.distance, "hard_time_s": hard_ticks / 10.,
        "hard_distance_mm": hard_mm, "recheck_proposals": proposed, "rechecks": accepted,
        "rejected_rechecks": physically_rejected, **totals,
    }
    instruction = None
    if job.teacher and arm == "train":
        instruction = teacher_annotation(scanner, spec, trial, case, metrics, visited_prefixes)
    return (metrics, steps if (job.full_log or case.partition == "train") else [],
            obs_rows, label_rows, instruction)


def run_job(job):
    case = job.case
    scanner = scanners(case.shape, case.partition == "blender")
    spec = make_spec(scanner, case)
    trial = motion.Trial(case.source, case.tape_seed, case.profile)
    model = scanner.model(WIDTH)
    policy = install_policy(job)
    return case, {arm: run_arm(scanner, spec, trial, model, policy, job, arm)
                  for arm in job.arms}


def development_cases(seed, batches, batch_size, validation_cases, shape_count):
    if shape_count < 3 or batch_size % 4 or validation_cases % 4:
        raise ValueError("Use at least three shapes and development counts divisible by four")
    routing = tuple(TASKS)
    training_shapes = tuple(x for x in range(shape_count) if x not in (3, 4, 6))
    validation_shapes = tuple(x for x in range(shape_count) if x not in training_shapes)
    if not validation_shapes:
        training_shapes, validation_shapes = tuple(range(shape_count - 1)), (shape_count - 1,)

    def cases(count, partition, base, shapes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, base]))
        numbers = rng.choice(1_000_000_000, count * 2, replace=False)
        result = []
        orders = [rng.permutation(4) for _ in range(count // 4)]
        atlas_routes = [rng.permutation(len(routing)) for _ in range((count + 4) // 5)]
        for i in range(count):
            source = int(orders[i // 4][i % 4])
            shape = int(shapes[(i + i // 4) % len(shapes)])
            task = "medseg_labelled_trunk" if source == 3 else routing[atlas_routes[i // 5][i % 5]]
            profile = PROFILES[(i // 4 + source) % 2]
            result.append(Case(f"{partition}_{i:05d}", partition, source, shape, task, profile,
                               base + int(numbers[i]), base + int(numbers[count + i])))
        return result

    return (cases(batches * batch_size, "train", 6_000_000_000, training_shapes),
            cases(validation_cases, "validation", 8_000_000_000, validation_shapes))


def test_cases(repetitions):
    return [
        Case(f"blender_{task}_{profile}_{tape:03d}", "blender", 4, 0, task, profile,
             884_400 + 1000 * task_num + tape, 16_700_000 + 1000 * task_num + 50 * prof_num + tape)
        for task_num, task in enumerate(TASKS)
        for prof_num, profile in enumerate(PROFILES)
        for tape in range(repetitions)
    ]


class Logs:
    def __init__(self, out, name, full):
        self.out, self.name, self.full = out, name, full

    def __enter__(self):
        kinds = ("episodes", "steps", "frames", "evaluator_labels") if self.full else ("episodes",)
        self.streams = {
            kind: gzip.open(self.out / f"{self.name}_{kind}.jsonl.gz", "wt", compresslevel=3)
            for kind in kinds
        }
        return self

    def row(self, case, arm, product, **metadata):
        metrics, steps, frames, labels, _instruction = product
        line(self.streams["episodes"], asdict(case) | {"arm": arm, **metadata, **metrics})
        if self.full:
            for kind, rows in (("steps", steps), ("frames", frames), ("evaluator_labels", labels)):
                for row in rows:
                    line(self.streams[kind], row | metadata)

    def __exit__(self, *_):
        for stream in self.streams.values():
            stream.close()


def summary(rows):
    fields = ("pass", "correct_regions", "wrong_regions", "wrong_delivery", "no_result_regions",
              "captured_regions", "time_s", "travel_mm", "frames", "rechecks", "recheck_proposals",
              "rejected_rechecks", "probe_ticks", "probe_frames", "opportunities", "eligible", "reward",
              "soft_identity_potential", "soft_identity_gain")
    if not rows:
        return {}
    return {"cases": len(rows), **{key: float(np.mean([r[key] for r in rows])) for key in fields},
            "active_cases": sum(r["rechecks"] > 0 for r in rows),
            "proposal_cases": sum(r["recheck_proposals"] > 0 for r in rows)}


def evaluate_validation(executor, policy, cases, log, update, also_mpc):
    jobs = []
    if also_mpc:
        jobs.extend(Job(case, ("mpc",), full_log=False) for case in cases)
    weights = weights_of(policy)
    for threshold in GATE_THRESHOLDS:
        arm = "ppo_ungated" if threshold == .5 else "ppo_gated"
        jobs.extend(Job(case, (arm,), f"validation-{update}", weights,
                        ungated_threshold=threshold, gated_threshold=threshold, full_log=False)
                    for case in cases)
    scores = {str(threshold): [] for threshold in GATE_THRESHOLDS}
    if also_mpc:
        scores["mpc"] = []
    for i, (case, result) in enumerate(executor.map(run_job, jobs, chunksize=1)):
        arm, product = next(iter(result.items()))
        threshold = "mpc" if arm == "mpc" else str(jobs[i].ungated_threshold)
        log.row(case, arm, product, update=update, threshold=threshold)
        scores[threshold].append(product[0])
    return {threshold: summary(metrics) for threshold, metrics in scores.items()}


def learning_curve(out, rows):
    import csv
    columns = ("update", "total_episodes", "total_steps", "training_reward", "training_pass",
               "training_correct_regions", "training_wrong_delivery", "training_rechecks",
               "validation_reward", "validation_correct_regions", "validation_wrong_delivery",
               "validation_rechecks", "validation_mpc_reward")
    with (out / "learning_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    valid = [row for row in rows if row.get("validation_reward") is not None]
    for ax, metric, ylabel in zip(
        axes.flat, ("reward", "correct_regions", "wrong_delivery", "rechecks"),
        ("Episode return", "Correct regions per route", "Wrong delivery fraction", "Rechecks per route"),
    ):
        train = [row for row in rows if row.get(f"training_{metric}") is not None]
        ax.plot([r["total_episodes"] for r in train], [r[f"training_{metric}"] for r in train],
                marker=".", label="On-policy rollouts")
        ax.plot([r["total_episodes"] for r in valid], [r[f"validation_{metric}"] for r in valid],
                marker="o", label="Deterministic validation")
        if metric == "reward" and valid:
            ax.axhline(valid[0]["validation_mpc_reward"], color=".4", ls="--", label="Validation MPC")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training episodes")
        ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=170)
    plt.close(fig)


def select_deployment(checkpoints, mpc):
    # Candidate ordering and margins are fixed in the code and recorded before
    # any evaluation with Blender is permitted.
    best_ungated = max((c for c in checkpoints if c["update"] > 0), key=lambda c: c["validation"]["0.5"]["reward"])
    qualified = []
    for checkpoint in checkpoints:
        if checkpoint["update"] == 0:
            continue
        for threshold in GATE_THRESHOLDS:
            result = checkpoint["validation"][str(threshold)]
            if (result["reward"] >= mpc["reward"] + .02
                    and result["pass"] >= mpc["pass"]
                    and result["wrong_delivery"] <= mpc["wrong_delivery"]
                    and result["active_cases"] >= max(1, np.ceil(.10 * result["cases"]))):
                qualified.append((result["reward"], checkpoint["update"], float(threshold)))
    if qualified:
        _, update, threshold = max(qualified, key=lambda x: (x[0], x[2], -x[1]))
    else:
        update, threshold = best_ungated["update"], None
    return best_ungated["update"], update, threshold


def train(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "freeze.json").exists():
        raise FileExistsError(f"A frozen model already exists in {args.out}")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.train_pack, allow_pickle=False) as pack:
        shape_count = len(pack["ys"])
        bounds = pack["neutral_grid"][pack["ct_bounds"]].tolist()
    train_set, validation_set = development_cases(args.seed, args.updates, args.batch_size,
                                                  args.validation_cases, shape_count)
    manifest = {
        "seed": args.seed, "train_pack": str(args.train_pack), "train_pack_sha256": digest(args.train_pack),
        "source_names": ["BodyParts3D right", "BodyParts3D left mirrored", "Open3DModel right", "MedSeg"],
        "training_domain_ct_neutral_stations_mm": bounds,
        "training_cases": list(map(asdict, train_set)), "validation_cases": list(map(asdict, validation_set)),
        "feature_names": FEATURE_NAMES, "model": "hierarchical Bernoulli + tanh diagonal Gaussian PPO",
        "policy_prior_width_mm": WIDTH, "probe_bank_ticks": PROBE_TICKS, "probe_bank_mm": PROBE_MM,
        "probe_caps_ticks": ACTION_TICKS, "probe_caps_mm": ACTION_MM, "max_rechecks": MAX_PROBES,
        "gate_thresholds": GATE_THRESHOLDS,
        "reward": "Continuous target witness quality minus squared distractor confidence risk; potential change plus capture, terminal strict identity and movement/intervention costs",
        "batch_size": args.batch_size,
        "updates": args.updates, "ppo_epochs": args.epochs, "workers": args.workers,
        "validation_every": args.validate_every, "blender_geometry_opened": False,
        "teacher_per_batch": args.teacher_per_batch,
        "teacher_protocol": "Student low-evidence prefixes; paired actual actuator replays on original and independent tapes; complete current region through local future; conservative skill and motor distillation",
        "code_sha256": code_hashes(),
    }
    write_json(args.out / "development_protocol.json", manifest)
    policy = ScanPolicy(len(FEATURE_NAMES))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    checkpoints, curve = [], []
    all_steps = 0
    start_clock = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.train_pack, None),
                             mp_context=multiprocessing.get_context("fork")) as executor, \
            Logs(args.out, "training", True) as training, Logs(args.out, "validation", False) as validation, \
            (args.out / "updates.jsonl").open("w") as updates, \
            gzip.open(args.out / "teacher_branches.jsonl.gz", "wt", compresslevel=3) as teacher_log:
        scores = evaluate_validation(executor, policy, validation_set, validation, 0, True)
        mpc = scores.pop("mpc")
        checkpoints.append({"update": 0, "validation": scores})
        torch.save(policy.state_dict(), args.out / "checkpoint_000.pt")
        curve.append({"update": 0, "total_episodes": 0, "total_steps": 0,
                      "validation_mpc_reward": mpc["reward"], **validation_curve(scores)})
        print(f"validation 0/{args.updates}: ppo return {scores['0.5']['reward']:.3f}, "
              f"MPC {mpc['reward']:.3f} ({time.monotonic()-start_clock:.1f}s)", flush=True)
        for update in range(1, args.updates + 1):
            subset = train_set[(update - 1) * args.batch_size:update * args.batch_size]
            weights = weights_of(policy)
            indices = np.random.default_rng(np.random.SeedSequence([args.seed, update, 175])).permutation(len(subset))
            taught = set(indices[:args.teacher_per_batch])
            jobs = (Job(case, ("train",), f"train-{update-1}", weights, teacher=i in taught)
                    for i, case in enumerate(subset))
            transitions, metrics, annotations = [], [], []
            for case, result in executor.map(run_job, jobs, chunksize=1):
                product = result["train"]
                metrics.append(product[0])
                transitions.append(product[1])
                training.row(case, "train", product, behavior_update=update - 1)
                if product[4] is not None:
                    annotation = product[4]
                    line(teacher_log, annotation | {"behavior_update": update - 1})
                    annotations.append(annotation)
            report = update_ppo(policy, optimizer, transitions, rng, epochs=args.epochs)
            report.update(fit_teacher(policy, optimizer, [a for a in annotations if a["eligible_prefix"]]))
            report.update(teacher_calls=len(annotations),
                          teacher_branches=sum(len(a.get("branches", ())) for a in annotations),
                          teacher_independent_replays=sum(int(a["eligible_prefix"]) * len(a.get("branches", ()))
                                                          for a in annotations),
                          teacher_failed_student_prefixes=sum(a.get("student_outcome", 1) != 1 for a in annotations),
                          teacher_full_tail_checks=sum(a.get("full_tail_check") is not None for a in annotations),
                          teacher_full_tail_accepted=sum(bool((a.get("full_tail_check") or {}).get("accepted"))
                                                         for a in annotations))
            all_steps += report["steps"]
            sampled = summary(metrics)
            row = {"update": update, "total_episodes": update * args.batch_size, "total_steps": all_steps,
                   "validation_mpc_reward": mpc["reward"],
                   **{f"training_{key}": sampled[key] for key in
                      ("reward", "pass", "correct_regions", "wrong_delivery", "rechecks")}}
            if update % args.validate_every == 0 or update == args.updates:
                scores = evaluate_validation(executor, policy, validation_set, validation, update, False)
                checkpoints.append({"update": update, "validation": scores})
                row.update(validation_curve(scores))
                torch.save(policy.state_dict(), args.out / f"checkpoint_{update:03d}.pt")
            line(updates, report | row | {"elapsed_seconds": time.monotonic() - start_clock})
            updates.flush()
            curve.append(row)
            print(f"update {update}/{args.updates}: {report['steps']} steps, {sampled['rechecks']:.2f} "
                  f"rechecks, return {sampled['reward']:.3f}, "
                  f"{time.monotonic()-start_clock:.1f}s elapsed", flush=True)

    if code_hashes() != manifest["code_sha256"]:
        raise RuntimeError("Source files changed during training. The policy cannot be frozen.")
    ungated_update, gated_update, gate = select_deployment(checkpoints, mpc)
    ungated_path = args.out / f"checkpoint_{ungated_update:03d}.pt"
    gated_path = args.out / f"checkpoint_{gated_update:03d}.pt"
    learning_curve(args.out, curve)
    frozen = {
        "training_protocol_sha256": digest(args.out / "development_protocol.json"),
        "code_sha256": code_hashes(), "train_pack_sha256": manifest["train_pack_sha256"],
        "feature_names": FEATURE_NAMES, "blender_geometry_opened": False,
        "completed_episodes": len(train_set), "completed_steps": all_steps, "updates": args.updates,
        "wall_time_seconds": time.monotonic() - start_clock,
        "validation_mpc": mpc, "validation_checkpoints": checkpoints,
        "ungated": {"update": ungated_update, "threshold": .5, "file": ungated_path.name,
                    "sha256": digest(ungated_path)},
        "gated": {"update": gated_update, "threshold": gate, "file": gated_path.name,
                  "sha256": digest(gated_path), "mpc_fallback": gate is None},
        "deployment_rule": {"validation_reward_margin": .02, "must_match_mpc_pass": True,
                            "must_match_mpc_wrong_delivery": True,
                            "min_validation_fraction_accepted_recheck": .10},
    }
    write_json(args.out / "freeze.json", frozen)
    print(f"frozen {len(train_set)} episodes, {all_steps} steps, "
          f"ungated update {ungated_update}, gated threshold {gate}", flush=True)


def validation_curve(scores):
    return {f"validation_{key}": scores["0.5"][key]
            for key in ("reward", "correct_regions", "wrong_delivery", "rechecks")}


def code_hashes():
    paths = (Path(__file__), Path(motion.__file__), Path(__file__).with_name("policy.py"), Path(__file__).with_name("online_reader.py"), Path(__file__).with_name("teacher.py"),
             Path(__file__).with_name("soft_identity.py"))
    legacy = (Path(motion.legacy_scan.__file__), Path(motion.legacy_geometry.__file__),
              Path(motion.ARCHIVED_PRODMP.__file__))
    return {path.name: digest(path) for path in (*paths, *legacy)}


def load_frozen(out, spec):
    path = out / spec["file"]
    if digest(path) != spec["sha256"]:
        raise ValueError(f"Changed frozen policy {path}")
    policy = ScanPolicy(len(FEATURE_NAMES))
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return weights_of(policy)


def paired_summary(rows, arm, baseline):
    rng = np.random.default_rng(37_541)
    strata = {}
    metrics = ("pass", "correct_regions", "wrong_delivery", "time_s", "rechecks", "reward")
    for row in rows:
        a, b = row["arms"][arm], row["arms"][baseline]
        delta = [a[key] - b[key] for key in metrics]
        cluster = strata.setdefault(row["task"], {})
        cluster.setdefault(row["tape_seed"], []).append(delta)
    boot = np.zeros((2000, len(metrics)))
    measure = np.zeros(len(metrics))
    for clusters in strata.values():
        matrix = np.stack([np.sum(cluster, axis=0) for cluster in clusters.values()])
        sample = rng.integers(len(matrix), size=(len(boot), len(matrix)))
        boot += matrix[sample].sum(axis=1)
        measure += matrix.sum(axis=0)
    boot /= len(rows)
    measure /= len(rows)
    bounds = np.quantile(boot, (.025, .975), axis=0)
    answer = {metric: {"difference": float(measure[j]),
                       "ci95": bounds[:, j].tolist()} for j, metric in enumerate(metrics)}
    for metric in ("pass", "wrong_delivery"):
        answer[metric]["arm_only"] = sum(row["arms"][arm][metric] > row["arms"][baseline][metric] for row in rows)
        answer[metric]["baseline_only"] = sum(row["arms"][arm][metric] < row["arms"][baseline][metric] for row in rows)
    return answer


def evaluate(args):
    torch.set_num_threads(1)
    if args.legacy_geometry is None:
        raise ValueError("Evaluation requires --legacy-geometry")
    frozen_path = args.out / "freeze.json"
    frozen = json.loads(frozen_path.read_text())
    if not frozen["blender_geometry_opened"] is False or frozen["code_sha256"] != code_hashes():
        raise ValueError("Frozen training protocol or source code changed")
    if digest(args.train_pack) != frozen["train_pack_sha256"]:
        raise ValueError("The approved training geometry changed")
    if digest(args.out / "development_protocol.json") != frozen["training_protocol_sha256"]:
        raise ValueError("The recorded development split changed")
    ungated = load_frozen(args.out, frozen["ungated"])
    gated = load_frozen(args.out, frozen["gated"])
    tests = test_cases(args.reps)
    arms = TEST_ARMS
    if args.control_out:
        control = json.loads((args.control_out / "freeze.json").read_text())
        control_protocol = json.loads((args.control_out / "development_protocol.json").read_text())
        experimental_protocol = json.loads((args.out / "development_protocol.json").read_text())
        common = ("seed", "training_cases", "validation_cases", "updates", "batch_size", "ppo_epochs",
                  "train_pack_sha256", "code_sha256")
        if (control_protocol["teacher_per_batch"] != 0 or control["code_sha256"] != code_hashes()
                or any(control_protocol[k] != experimental_protocol[k] for k in common)):
            raise ValueError("No-teacher comparator does not match the frozen development protocol")
        control_weights = load_frozen(args.control_out, control["ungated"])
        arms = (*arms, "ppo_no_teacher")
    write_json(args.out / "blender_test_protocol.json", {
        "one_blender_anatomy": True, "freeze_sha256": digest(frozen_path),
        "test_cases": list(map(asdict, tests)), "arms": arms,
        "test_geometry": str(args.legacy_geometry), "test_geometry_sha256": digest(args.legacy_geometry),
        "all_arms_share": ["tape", "task", "reader", "cadence", "main route planner", "hard budget"],
    })
    t0 = time.monotonic()
    alternate_weights = frozen["gated"]["sha256"] != frozen["ungated"]["sha256"]
    composite_arms = tuple(arm for arm in TEST_ARMS if arm != "ppo_gated" or not alternate_weights)
    jobs = [
        Job(case, composite_arms, "frozen-ungated", ungated, .5, frozen["gated"]["threshold"])
        for case in tests
    ]
    if alternate_weights:
        jobs.extend(Job(case, ("ppo_gated",), "frozen-gated", gated, .5, frozen["gated"]["threshold"])
                    for case in tests)
    if args.control_out:
        jobs.extend(Job(case, ("ppo_no_teacher",), "frozen-no-teacher", control_weights)
                    for case in tests)
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.train_pack, args.legacy_geometry),
                             mp_context=multiprocessing.get_context("fork")) as executor, \
            Logs(args.out, "blender", True) as audit:
        for i, (case, products) in enumerate(executor.map(run_job, jobs, chunksize=1), start=1):
            row = rows.setdefault(case.identifier, asdict(case) | {"arms": {}})
            for arm, product in products.items():
                row["arms"][arm] = product[0]
                audit.row(case, arm, product)
            if i % 10 == 0 or i == len(jobs):
                print(f"Blender jobs {i}/{len(jobs)}, {time.monotonic()-t0:.1f}s elapsed", flush=True)
    rows = list(rows.values())
    with gzip.open(args.out / "blender_paired_cases.jsonl.gz", "wt", compresslevel=3) as stream:
        for row in rows:
            line(stream, row)
    pairs = list(itertools.product(("ppo_ungated", "ppo_gated"),
                                   ("mpc", "active_mpc", "prodmp_rule", "prodmp_random", "prodmp_matched")))
    if args.control_out:
        pairs.extend((("ppo_no_teacher", "mpc"), ("ppo_ungated", "ppo_no_teacher")))
    outcome = {
        "cases": len(rows), "anatomies": 1, "full_route_length_mm": 730.,
        "freeze_sha256": digest(frozen_path), "gated_mpc_fallback": frozen["gated"]["mpc_fallback"],
        "workers": args.workers, "wall_time_seconds": time.monotonic() - t0,
        "overall": {arm: summary([r["arms"][arm] for r in rows]) for arm in arms},
        "by_task_and_profile": {
            f"{task}/{profile}": {
                arm: summary([r["arms"][arm] for r in rows
                              if r["task"] == task and r["profile"] == profile])
                for arm in arms
            }
            for task, profile in itertools.product(TASKS, PROFILES)
        },
        "paired": {f"{arm}_vs_{ref}": paired_summary(rows, arm, ref) for arm, ref in pairs},
        "analysis_units": "Paired disturbance-seed clusters within task; ordinary and stress stay together. One anatomy.",
    }
    write_json(args.out / "blender_summary.json", outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--train-pack", type=Path, required=True)
    parser.add_argument("--legacy-geometry", type=Path)
    parser.add_argument("--control-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-cases", type=int, default=16)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-per-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=14311)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reps", type=int, default=16)
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.validation_cases, args.validate_every,
           args.epochs, args.workers, args.reps) < 1:
        parser.error("Run counts must be positive")
    if not 0 <= args.teacher_per_batch <= args.batch_size:
        parser.error("Teacher quota must be between zero and the batch size")
    if args.stage == "train" and args.legacy_geometry is not None:
        parser.error("Blender geometry must be inaccessible in train")
    (train if args.stage == "train" else evaluate)(args)
