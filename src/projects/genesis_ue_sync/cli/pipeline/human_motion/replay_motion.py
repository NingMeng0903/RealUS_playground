from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.human_motion.validation import motion_quality_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a HumanMotionSequence before Genesis/UE replay.")
    p.add_argument("--sequence-npz", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = motion_quality_report(HumanMotionSequence.load(args.sequence_npz))
    text = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
