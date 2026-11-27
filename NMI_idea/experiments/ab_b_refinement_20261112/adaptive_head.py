"""Ablate selecting the first acquisition by contingent continuation value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import refine_rechecks as study


class AdaptiveHeadPlanner(study.Planner):
    def _plan_suffix(self, book, belief, counts, pose, index, next_pose, arm, remaining):
        plan, value = super()._plan_suffix(
            book, belief, counts, pose, index, next_pose, arm, remaining,
        )
        if plan is not None:
            value += plan.predicted_gain
        return plan, value


def main():
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--out", type=Path, required=True)
    options, _ = output_parser.parse_known_args()
    study.ARMS["B_joint"] = study.ARMS["B"]
    study.ARMS["B_joint_no_fisher"] = study.ARMS["B_no_fisher"]
    if "--arms" not in sys.argv:
        sys.argv.extend(["--arms", "B_joint"])
    study.Planner = AdaptiveHeadPlanner
    study.main()
    config_path = options.out / "CONFIG.json"
    config = json.loads(config_path.read_text())
    config["source_sha256"][Path(__file__).name] = study.sha(Path(__file__))
    config["head_objective"] = "shared base score plus expected best suffix after each predictive public outcome"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
