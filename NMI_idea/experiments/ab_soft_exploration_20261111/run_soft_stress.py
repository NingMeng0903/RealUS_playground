"""Paired mechanism test for a two-bin scan with a temporarily hidden target."""
from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from program_value import accepted, observation_tables, program_scores, update
from toy_world import Fixture, latent_index, legal_actions, prior, sample_step


ARMS = ("A_task", "hard_nominal", "soft_nominal", "hard_exit", "soft_exit")
PLANNING_DRAWS = 128
MAX_PACKETS = 32


@lru_cache(maxsize=None)
def route(start, goal, fixture):
    """Shortest legal route, with a deterministic tie order."""
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for action in sorted(legal_actions(node, fixture), key=lambda x: x.name):
            if action.kind == "move" and action.dest not in seen:
                seen.add(action.dest)
                queue.append((action.dest, path + (action,)))
    return None


def scan_at(node, fixture):
    return next(a for a in legal_actions(node, fixture) if a.kind == "scan")


@lru_cache(maxsize=None)
def programs(node, fixture, condition_on_exit):
    rows = []
    for first in sorted(legal_actions(node, fixture), key=lambda x: x.name):
        for side in ("L", "R"):
            terminal = "T" + side
            start = first.dest if condition_on_exit else "C"
            onward = route(start, terminal, fixture)
            if onward is not None:
                scan = scan_at(terminal, fixture)
                # The arrival frame is the first capture at this terminal.
                rows.append((side, (first, *onward, scan)))
    return tuple(rows)


def fallback(node, fixture, belief, pending):
    choices = []
    for side in ("L", "R"):
        onward = route(node, "T" + side, fixture)
        if onward is None:
            continue
        possible = sum(belief[latent_index(side, False, present)] for present in (False, True))
        first = onward[0] if onward else scan_at(node, fixture)
        choices.append((float(possible), side, first))
    if not choices:
        return None, None
    _, side, first = min(choices, key=lambda x: (-x[0], x[1] != pending, x[1]))
    return first, side


def choose(arm, node, fixture, belief, pending, start_draws, packet_draws):
    if arm == "A_task":
        # Strong immediate-completion policy with legal task-oriented routing
        # whenever there is no eligible first-frame completion.
        actions = sorted(legal_actions(node, fixture), key=lambda x: x.name)
        correct_mass = []
        for action in actions:
            _, p_target = update(belief, fixture, action, 4)
            transition, _, target, _, _ = observation_tables(action, fixture)
            probability = float((belief @ transition) @ target[:, 4])
            correct_mass.append(probability if accepted(action, 4, p_target) else 0.)
        if max(correct_mass, default=0.) > 0.:
            index = min(range(len(actions)), key=lambda i: (
                -correct_mass[i], actions[i].side != pending, actions[i].name))
            return actions[index], actions[index].side or pending, False, None, None
    else:
        rows = programs(node, fixture, arm.endswith("_exit"))
        if rows:
            scores = program_scores(
                belief, fixture, [p for _, p in rows], mode=arm.split("_")[0],
                start_draws=start_draws, packet_draws=packet_draws)
            index = min(range(len(rows)), key=lambda i: (
                -scores[i], rows[i][0] != pending, rows[i][1][0].name, rows[i][0]))
            if scores[index] > 0.:
                side, program = rows[index]
                previous_score = max((scores[i] for i, row in enumerate(rows)
                                      if row[0] == pending), default=None)
                return (program[0], side, False, float(scores[index]),
                        None if previous_score is None else float(previous_score))
    first, side = fallback(node, fixture, belief, pending)
    return first, side, True, None, None


def fixtures():
    for side, shadow, narrow in itertools.product(("L", "R"), ("L", "R"), (False, True)):
        yield side, False, Fixture(shadow, narrow), "narrow" if narrow else "wide"
    for side in ("L", "R"):
        yield side, False, Fixture(None, False), "open_control"
    for side in ("L", "R"):
        yield side, True, Fixture(None, True), "permanent_occlusion"


def run(arm, fixture, true_side, permanent, actual_draws, planning_draws):
    state = latent_index(true_side, permanent, False)
    belief = prior()
    node = "EL"
    pending = physical_side = "L"
    visits = [physical_side]
    previous_missing = False
    events = []
    metric = dict(switches=0, switches_after_missing=0, retreats=0,
                  returns=0, no_information_or_progress_after_missing=0,
                  missing=0, clutter=0, fallback=0)
    outcome = "unresolved"
    first_target_frame = None
    true_goal = "T" + true_side
    for step, uniforms in enumerate(actual_draws):
        initial, packets = planning_draws[step]
        action, selected_side, used_fallback, value, pending_value = choose(
            arm, node, fixture, belief, pending, initial, packets)
        if action is None:
            break
        if action not in legal_actions(node, fixture):
            raise AssertionError("controller selected an illegal graph edge")
        next_node = action.dest
        if previous_missing:
            _, emission, _, _, _ = observation_tables(action, fixture)
            no_information = np.max(np.abs(emission - emission[0])) <= 1.e-12
            before = route(node, true_goal, fixture)
            after = route(next_node, true_goal, fixture)
            no_progress = before is not None and after is not None and len(after) >= len(before)
            metric["no_information_or_progress_after_missing"] += int(no_information and no_progress)
        metric["retreats"] += int(node != "C" and next_node == "C")
        if next_node != "C" and next_node[-1] != physical_side:
            physical_side = next_node[-1]
            metric["switches"] += 1
            metric["switches_after_missing"] += int(previous_missing)
            metric["returns"] += int(len(visits) >= 2 and visits[-2] == physical_side)
            visits.append(physical_side)
        state, obs, true_target = sample_step(state, action, fixture, *uniforms)
        belief, target_probability = update(belief, fixture, action, obs)
        stop = accepted(action, obs, target_probability)
        if true_target and first_target_frame is None:
            first_target_frame = step + 1
        previous_missing = obs == 0
        metric["missing"] += int(previous_missing)
        metric["clutter"] += int(obs != 0 and not true_target)
        metric["fallback"] += int(used_fallback)
        previous_pending = pending
        pending = None if previous_missing and arm.startswith("hard_") else selected_side
        remaining = route(next_node, "T" + pending, fixture) if pending else ()
        events.append(dict(action=action.name, entry=node, exit=next_node,
                           observation=obs, p_target=target_probability, stop=stop,
                           pending_before=previous_pending, pending_goal=pending,
                           pending_waypoints=[a.dest for a in remaining] if remaining else [],
                           program_value=value, previous_goal_value=pending_value))
        node = next_node
        if stop:
            outcome = "correct" if true_target else "wrong_stop"
            break
    return dict(arm=arm, outcome=outcome, frames=len(events), bin0_preserved=True,
                first_target_frame=first_target_frame,
                continuous_branch_completion=outcome == "correct" and metric["switches"] == 0
                and metric["retreats"] == 0, events=events, **metric)


def summary(records):
    output = {}
    for arm in ARMS:
        groups = {}
        for group in ("recoverable", "narrow", "wide", "open_control", "permanent_occlusion"):
            rows = [r for r in records if r["arm"] == arm and
                    (r["stratum"] == group or group == "recoverable" and r["oracle_possible"])]
            groups[group] = dict(
                n=len(rows), correct=sum(r["outcome"] == "correct" for r in rows),
                wrong_stop=sum(r["outcome"] == "wrong_stop" for r in rows),
                unresolved=sum(r["outcome"] == "unresolved" for r in rows),
                **{k: sum(r[k] for r in rows) for k in (
                    "switches", "switches_after_missing", "retreats", "returns",
                    "no_information_or_progress_after_missing", "continuous_branch_completion",
                    "frames", "missing", "clutter")})
        output[arm] = groups
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    planning_draws = []
    for step in range(MAX_PACKETS):
        rng = np.random.default_rng(np.random.SeedSequence([721603, step]))
        planning_draws.append((rng.random(PLANNING_DRAWS),
                               rng.random((PLANNING_DRAWS, 8, 3))))
    rows = []
    with (args.output / "EPISODES.jsonl").open("w", encoding="utf-8") as handle:
        for i, (side, permanent, fixture, stratum) in enumerate(fixtures()):
            oracle_possible = not permanent and route("EL", "T" + side, fixture) is not None
            for repeat in range(args.replicates):
                rng = np.random.default_rng(np.random.SeedSequence([721602, 721601+i, repeat]))
                packets = rng.random((MAX_PACKETS, 3))
                for arm in ARMS:
                    row = run(arm, fixture, side, permanent, packets, planning_draws)
                    row.update(case=i, repeat=repeat, stratum=stratum, true_side=side,
                               shadow=fixture.shadow, narrow=fixture.narrow,
                               oracle_possible=oracle_possible)
                    rows.append(row)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
    here = Path(__file__).resolve().parent
    result = dict(
        scope="synthetic graph mechanism study, no native motion or acoustics validation",
        paired_by="fixture, repeat, fresh packet index and random channel",
        no_hard_time_limit=True, administrative_packet_checkpoints=[8, 16, MAX_PACKETS],
        replicates_per_fixture=args.replicates, planning_draws=PLANNING_DRAWS,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (here / "run_soft_stress.py", here / "program_value.py",
                                 here / "toy_world.py")},
        groups=summary(rows))
    (args.output / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({a: g["recoverable"] for a, g in result["groups"].items()}))


if __name__ == "__main__":
    main()
