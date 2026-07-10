from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.human_motion.validation import motion_quality_report, write_motion_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated/refit human motion and optionally write a manifest.")
    p.add_argument("--sequence-npz", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-manifest", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seq = HumanMotionSequence.load(args.sequence_npz)
    report = motion_quality_report(seq)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.output_manifest is not None:
        write_motion_manifest(sequence_npz_path=args.sequence_npz, output_manifest_path=args.output_manifest, tags=("validated",))
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
