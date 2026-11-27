"""Paired supplementary test of evidence-dependent release of a saved suffix."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from run_experiment import ARMS, Episode, cases, packet_draws, source_manifest
from skill_value import HERE, TOL, Statistics, path_statistics, terminal_path


SEED = 721622
REPLICATES = 16
CHECKPOINTS = (8, 16, 32, 64)


@dataclass
class ReleasedEpisode(Episode):
    releases: list[dict] = field(default_factory=list)

    def evaluate_remaining_suffix(self) -> None:
        if self.arm == "A_legacy" or self.outcome != "unresolved" or not self.actions:
            return
        event = self.events[-1]
        if "selected_goal" in event:
            # The first feedback packet just selected a branch. Execute at
            # least one of its suffix actions before considering release.
            return
        program = event["program"]
        used = Counter(e["exit"] for e in self.events if e["program"] == program)
        other_side = "R" if self.pending == "L" else "L"
        if used["T" + other_side] >= 2:
            return
        current_path = tuple(self.actions)
        proposed_path = terminal_path(
            self.node, other_side, self.case.fixture,
            already_captured_at_terminal=used["T" + other_side] == 1)
        fixture, belief = self.case.fixture, self.belief
        keep = Statistics(*map(float, path_statistics(current_path, fixture, belief, belief)))
        switch = Statistics(*map(float, path_statistics(proposed_path, fixture, belief, belief)))
        if switch.value <= keep.value + TOL or switch.wrong > keep.wrong + TOL:
            return
        record = dict(
            frame=len(self.events), program=program, packet=event["packet"],
            actual_node=self.node, posterior=belief.tolist(),
            terminal_attempts={side: used["T" + side] for side in ("L", "R")},
            old_goal=self.pending, new_goal=other_side,
            old_suffix=[a.name for a in current_path], new_suffix=[a.name for a in proposed_path],
            old=keep.as_dict(), new=switch.as_dict(),
        )
        self.releases.append(record)
        self.pending = other_side
        self.actions = deque(proposed_path)
        event["release"] = record
        event["remaining_actions"] = record["new_suffix"]

    def advance(self, until: int) -> None:
        if until > len(self.draws):
            raise ValueError("insufficient paired observation draws")
        while len(self.events) < until and self.outcome == "unresolved":
            super().advance(len(self.events) + 1)
            self.checkpoints.pop()
            self.evaluate_remaining_suffix()
        self.checkpoints.append(dict(
            requested=until, frames=len(self.events), outcome=self.outcome,
            actual_node=self.node, posterior=self.belief.tolist(),
            pending=self.pending, remaining_actions=[a.name for a in self.actions],
        ))

    def record(self, replicate: int) -> dict:
        return dict(super().record(replicate), releases=self.releases)


def outcomes(rows: list[dict]) -> dict:
    return dict(
        n=len(rows),
        **{key: sum(r["outcome"] == key for r in rows)
           for key in ("correct", "wrong_stop", "unresolved")},
        **{key: sum(r[key] for r in rows)
           for key in ("frames", "returns", "retreats", "switches")},
        releases=sum(len(r.get("releases", ())) for r in rows),
        episodes_with_release=sum(bool(r.get("releases")) for r in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        seed=SEED, replicates=REPLICATES, checkpoints=CHECKPOINTS, arms=ARMS,
        scope="supplementary paired synthetic sensitivity, separate from primary study",
        rule="after one suffix primitive, same posterior, higher C/N and no higher wrong-stop probability",
        terminal_attempts_preserved=True,
        source_sha256={**source_manifest(), str(HERE / "release_diagnostic.py"):
                       hashlib.sha256((HERE / "release_diagnostic.py").read_bytes()).hexdigest()},
    )
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    groups = {}
    for name, episode_class in (("retained", Episode), ("released", ReleasedEpisode)):
        records = []
        with (args.output / f"{name}.jsonl").open("x") as stream:
            for i, case in enumerate(cases()):
                for replicate in range(REPLICATES):
                    draws = packet_draws(SEED, i, replicate, CHECKPOINTS[-1])
                    for arm in ARMS:
                        episode = episode_class(case, arm, draws)
                        for boundary in CHECKPOINTS:
                            episode.advance(boundary)
                            if episode.outcome != "unresolved":
                                break
                        record = episode.record(replicate)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        records.append(record)
        groups[name] = {
            arm: {stratum: outcomes([r for r in records if r["arm"] == arm and
                                     r["oracle_possible"] == (stratum == "recoverable")])
                  for stratum in ("recoverable", "permanent")}
            for arm in ARMS
        }
        print(f"{name}: {len(records)} complete", flush=True)
    (args.output / "SUMMARY.json").write_text(json.dumps(groups, indent=2) + "\n")


if __name__ == "__main__":
    main()
