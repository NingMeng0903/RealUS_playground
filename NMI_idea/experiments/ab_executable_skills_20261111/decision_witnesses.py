"""Compare plans on identical public histories and record activated decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from skill_value import (
    FIXED_MAPS, RESPONSE_MAPS, SIDES, TOL, _feedback_skill, _stats, accepted,
    actual_statistics, observation_tables, path_statistics, plan, terminal_path, update,
)
from toy_world import Fixture, legal_actions


def signature(skill, belief, fixture):
    first = skill.path[0]
    transition, emission, _, _, _ = observation_tables(first, fixture)
    probabilities = (belief @ transition) @ emission
    live = []
    for packet, probability in enumerate(probabilities):
        if probability <= TOL:
            continue
        _, p_target = update(belief, fixture, first, packet)
        if accepted(first, packet, p_target):
            continue
        if skill.goals_by_packet:
            side = skill.goals_by_packet[packet]
            suffix = terminal_path(first.dest, side, fixture, first.dest == "T" + side)
        else:
            suffix = skill.path[1:]
        live.append((packet, tuple(a.name for a in suffix)))
    return first.name, tuple(live)


def table(arm, node, fixture, belief, pending):
    candidates = []
    for side in SIDES:
        path = terminal_path(node, side, fixture)
        stats = _stats(path_statistics(path, fixture, belief, belief))
        candidates.append(dict(path=[a.name for a in path], goal=side, **stats.as_dict()))
    if arm != "A_task":
        for first in legal_actions(node, fixture):
            skill = _feedback_skill(first, fixture, belief, belief,
                                    FIXED_MAPS if arm == "A_fixed_goal" else RESPONSE_MAPS,
                                    pending, nominal_exit=arm == "B_nominal")
            candidates.append(dict(model_score=skill.as_dict(),
                                   actual_score=actual_statistics(skill, fixture, belief).as_dict()))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recoverable-only", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.episodes.open()]
    others = ("A_task", "A_fixed_goal", "B_nominal", "B_MAP")
    contrasts = {arm: Counter() for arm in others}
    best = {}
    unique = set()
    feedback_count = feedback_two_goals = 0
    for row in rows:
        if row["arm"] != "B_actual" or (args.recoverable_only and not row["oracle_possible"]):
            continue
        fixture = Fixture(**row["fixture"])
        for index, record in enumerate(row["programs"]):
            node, pending = record["entry"], record["pending"]
            belief_key = tuple(record["belief"])
            belief = np.asarray(belief_key, dtype=float)
            primary = plan("B_actual", node, fixture, belief_key, pending)
            if primary.goals_by_packet:
                feedback_count += 1
                first = primary.path[0]
                transition, emission, _, _, _ = observation_tables(first, fixture)
                probabilities = (belief @ transition) @ emission
                live_goals = {primary.goals_by_packet[y] for y, p in enumerate(probabilities)
                              if p > TOL and not accepted(
                                  first, y, update(belief, fixture, first, y)[1])}
                feedback_two_goals += int(len(live_goals) > 1)
            key = (fixture, node, pending, belief_key)
            if key in unique:
                continue
            unique.add(key)
            a = signature(primary, belief, fixture)
            for arm in others:
                other = plan(arm, node, fixture, belief_key, pending)
                b = signature(other, belief, fixture)
                correct_exit = actual_statistics(other, fixture, belief)
                gain = primary.statistics.value - correct_exit.value
                counter = contrasts[arm]
                counter["public_states"] += 1
                counter["different_program"] += int(a != b)
                counter["different_first_action"] += int(a[0] != b[0])
                counter["positive_exact_actual_value_delta"] += int(gain > TOL)
                if a != b and (arm not in best or gain > best[arm]["value_delta"] + TOL):
                    best[arm] = dict(
                        case_id=row["case_id"], replicate=row["replicate"], fixture=row["fixture"],
                        frame=record["frame"], node=node, pending=pending, belief=record["belief"],
                        value_delta=gain, primary=primary.as_dict(), rival=other.as_dict(),
                        rival_value_at_actual_exit=correct_exit.as_dict(),
                        primary_signature=a, rival_signature=b,
                        realized_program=[e for e in row["events"] if e["program"] == index],
                    )
    for arm, witness in best.items():
        fixture = Fixture(**witness["fixture"])
        belief = np.asarray(witness["belief"], dtype=float)
        witness["primary_candidates"] = table("B_actual", witness["node"], fixture, belief, witness["pending"])
        if arm != "B_MAP":
            witness["rival_candidates"] = table(arm, witness["node"], fixture, belief, witness["pending"])
    result = dict(
        scope=("distinct recorded recoverable B_actual public decision states" if args.recoverable_only
               else "all distinct recorded B_actual public decision states"),
        valuation="exact conditional model, not population inference",
        feedback_programs_chosen=feedback_count,
        feedback_programs_with_multiple_live_goal_branches=feedback_two_goals,
        contrasts=contrasts, largest_value_gap_examples=best,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(contrasts=contrasts, feedback=feedback_count,
                          two_live_goals=feedback_two_goals)))


if __name__ == "__main__":
    main()
