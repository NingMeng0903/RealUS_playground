"""Require larger evidence gains for cost-increasing suffix changes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import refine_rechecks as study


class GuardedPlanner(study.Planner):
    def choose(self, program, pose, previous, belief, before, after, arm, switched):
        choice = super().choose(program, pose, previous, belief, before, after, arm, switched)
        if choice.index == choice.locked_index:
            return choice
        options = self.actions[pose]
        candidate = self.motion(previous, pose, tuple(options[choice.index]), arm.learned_primitive)
        locked = self.motion(previous, pose, tuple(options[choice.locked_index]), arm.learned_primitive)
        extra_travel = max(candidate.length_m - locked.length_m, 0.)
        extra_margin = self.travel_weight * arm.travel_scale * extra_travel
        if choice.gain <= self.switch_margin + extra_margin:
            return replace(choice, index=choice.locked_index)
        return choice


def main():
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--out", type=Path, required=True)
    options, _ = output_parser.parse_known_args()
    study.ARMS["B_guarded"] = study.ARMS["B"]
    study.ARMS["B_once_guarded"] = study.ARMS["B_once"]
    if "--arms" not in sys.argv:
        sys.argv.extend(["--arms", "B_guarded"])
    study.Planner = GuardedPlanner
    study.main()
    config_path = options.out / "CONFIG.json"
    config = json.loads(config_path.read_text())
    config["source_sha256"][Path(__file__).name] = study.sha(Path(__file__))
    config["switch_rule"] = "add travel_weight times positive excess path to the existing switch margin"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
