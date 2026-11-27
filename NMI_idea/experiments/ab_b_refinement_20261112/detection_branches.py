"""Compare discrete public outcome branches with full posterior-conditioned tails."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

import refine_rechecks as study
from predictive import predictive_sets


class DetectionBook(study.EvidenceBook):
    def assimilate(self, belief, pose, action, values, labels=None):
        self.observations_present = len(values) > 0
        return super().assimilate(belief, pose, action, values, labels)


class DetectionPlanner(study.Planner):
    def plan(self, book, *args, **kwargs):
        self.public_book = book
        return super().plan(book, *args, **kwargs)

    def _outcomes(self, book, pose, index, belief, counts):
        action = tuple(self.actions[pose][index])
        samples = predictive_sets(
            belief, self.model.model_cache[(pose, *action)],
            half_width=study.WIDTH, p_detect=study.PD,
            noise_mixture=study.STRESS.noise_mixture,
            clutter_rate=study.STRESS.clutter_rate, count=self.n_predictive,
        )
        self.outcome_groups = []
        for values, weight in samples:
            simulated = book.copy()
            posterior = simulated.assimilate(belief, pose, action, values)
            updated, _, ready, _ = simulated.status(posterior)
            progress = bool(np.any(updated > counts))
            state = posterior, updated, progress, bool(np.all(ready)), weight
            self.outcome_groups.append((bool(values), state))
            yield state

    def _plan_suffix(self, book, belief, counts, pose, index, next_pose, arm, remaining):
        plan, expected = super()._plan_suffix(
            book, belief, counts, pose, index, next_pose, arm, remaining,
        )
        if plan is None:
            return plan, expected
        exit_state = (pose, *self.actions[pose][index])
        grouped = {}
        for present, (posterior, updated, progress, ready, weight) in self.outcome_groups:
            scores = np.zeros(len(plan.options)) if ready else self.score(
                next_pose, exit_state, posterior, updated, arm, plan.options, exact=True,
            )
            key = present, progress
            grouped[key] = grouped.get(key, np.zeros(len(plan.options))) + weight * scores
        # Keep the original common head objective so this contrast isolates
        # execution of the same candidates after more detailed public flags.
        plan.locked_by_flags = {
            key: int(plan.options[np.argmax(values)]) for key, values in grouped.items()
        }
        return plan, expected

    def choose(self, program, pose, previous, belief, before, after, arm, switched):
        progress = bool(np.any(after > before))
        flags = self.public_book.observations_present, progress
        locked = program.locked.copy()
        locked[progress] = program.locked_by_flags.get(flags, locked[progress])
        return super().choose(replace(program, locked=locked), pose, previous, belief,
                              before, after, arm, switched)


def main():
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--out", type=Path, required=True)
    options, _ = output_parser.parse_known_args()
    study.ARMS["locked_seen"] = study.ARMS["locked"]
    study.ARMS["B_seen"] = study.ARMS["B"]
    if "--arms" not in sys.argv:
        sys.argv.extend(["--arms", "locked_seen", "B_seen"])
    study.Planner = DetectionPlanner
    study.EvidenceBook = DetectionBook
    study.main()
    config_path = options.out / "CONFIG.json"
    config = json.loads(config_path.read_text())
    config["source_sha256"][Path(__file__).name] = study.sha(Path(__file__))
    config["shared_observation_flags"] = ["any centroid", "witness count increased"]
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
