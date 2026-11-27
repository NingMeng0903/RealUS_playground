"""Run original and same-components greedy controls on the refinement cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import refine_rechecks as study


class OriginalGreedyRecheck(study.Recheck):
    def plan(self, pose, next_pose, previous, mean, covariance, counts, arm):
        # The old one-step implementation in the common runner normally
        # dispatches to A_fisher. Select the archived ordinary A instead.
        strategy = "A" if arm == "A_fisher" else arm
        return super().plan(pose, next_pose, previous, mean, covariance, counts, strategy)


def main():
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--out", type=Path, required=True)
    options, _ = output_parser.parse_known_args()
    arms = {
        "basic_A": study.Arm("legacy", "legacy_greedy"),
        "basic_A_predictive_gate": study.Arm("predictive", "legacy_greedy"),
        "greedy_no_fisher": study.Arm("predictive", "greedy", fisher=0.),
    }
    study.Recheck = OriginalGreedyRecheck
    study.ARMS.update(arms)
    if "--arms" not in sys.argv:
        sys.argv.extend(["--arms", *arms])
    study.main()
    config_path = options.out / "CONFIG.json"
    config = json.loads(config_path.read_text())
    config["source_sha256"][Path(__file__).name] = study.sha(Path(__file__))
    config["controls"] = {
        "basic_A": "Original certification.Recheck A, legacy witness gate",
        "basic_A_predictive_gate": "Original certification.Recheck A, same predictive witness gate as refined B",
        "greedy_no_fisher": "Refined one-step objective and predictive witness gate, task-direction information term zero",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
