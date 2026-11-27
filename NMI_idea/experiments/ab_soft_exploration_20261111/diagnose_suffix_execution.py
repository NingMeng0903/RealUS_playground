"""Diagnostic control: execute the suffix evaluated by the same soft policy."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

import run_soft_stress as runner
from toy_world import legal_actions


class ExecuteEvaluatedSuffix:
    def __init__(self, base_choose):
        self.base_choose = base_choose
        self.remaining = deque()
        self.goal = None

    def __call__(self, arm, node, fixture, belief, pending, initial, packets):
        if self.remaining and self.remaining[0] in legal_actions(node, fixture):
            return self.remaining.popleft(), self.goal, False, None, None
        self.remaining.clear()
        decision = self.base_choose(arm, node, fixture, belief, pending, initial, packets)
        action, side, used_fallback, value, _ = decision
        if action is not None and not used_fallback and value is not None:
            selected = next(program for target, program in runner.programs(node, fixture, True)
                            if target == side and program[0] == action)
            self.remaining.extend(selected[1:])
            self.goal = side
        return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.base / "SUMMARY.json").read_text())
    replicates = config["replicates_per_fixture"]
    old_choose = runner.choose
    planning_draws = []
    for step in range(runner.MAX_PACKETS):
        rng = np.random.default_rng(np.random.SeedSequence([721603, step]))
        planning_draws.append((rng.random(runner.PLANNING_DRAWS),
                               rng.random((runner.PLANNING_DRAWS, 8, 3))))
    records = []
    try:
        for case, (side, permanent, fixture, stratum) in enumerate(runner.fixtures()):
            for repeat in range(replicates):
                rng = np.random.default_rng(np.random.SeedSequence([721602, 721601+case, repeat]))
                packets = rng.random((runner.MAX_PACKETS, 3))
                runner.choose = ExecuteEvaluatedSuffix(old_choose)
                row = runner.run("soft_exit", fixture, side, permanent, packets, planning_draws)
                row.update(case=case, repeat=repeat, stratum=stratum,
                           oracle_possible=not permanent)
                records.append(row)
    finally:
        runner.choose = old_choose
    base_records = {}
    for line in (args.base / "EPISODES.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["arm"] == "soft_exit":
            base_records[row["case"], row["repeat"]] = row
    assert len(base_records) == len(records)
    for row in records:
        base = base_records[row["case"], row["repeat"]]
        assert row["events"][0]["action"] == base["events"][0]["action"]
        assert row["events"][0]["observation"] == base["events"][0]["observation"]
    group_stats = runner.summary(records)["soft_exit"]
    paired = [(r, base_records[r["case"], r["repeat"]])
              for r in records if r["oracle_possible"]]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "EPISODES.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    result = dict(
        scope="post-result mechanism diagnostic on the same fixed synthetic cohort",
        change="execute selected legal suffix until public stop or suffix exhausted",
        base=str(args.base.resolve()),
        same_first_action_and_packet=len(records), groups=group_stats,
        paired_correct_only_with_suffix=sum(
            r["outcome"] == "correct" and b["outcome"] != "correct" for r, b in paired),
        paired_correct_only_without_suffix=sum(
            r["outcome"] != "correct" and b["outcome"] == "correct" for r, b in paired),
    )
    (args.output / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
