from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.human_motion.generation import MotionDiffuseAdapter, PlaceholderMotionGenerator
from projects.genesis_ue_sync.sim_platform.human_motion.planning import RuleBasedActionParser
from projects.genesis_ue_sync.sim_platform.human_motion.validation import write_motion_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a HumanMotionSequence from a prompt.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=("placeholder", "motiondiffuse"), default="placeholder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    blocks = RuleBasedActionParser().parse(args.prompt)
    if args.backend == "motiondiffuse":
        generator = MotionDiffuseAdapter()
        seq = generator.generate(prompt=args.prompt, action_blocks=blocks, seed=args.seed)
    else:
        generator = PlaceholderMotionGenerator(fps=float(args.fps))
        seq = generator.generate(prompt=args.prompt, action_blocks=blocks, seed=int(args.seed))
    output_npz = seq.save(args.output_npz)
    summary = {"sequence_npz_path": str(output_npz), "frame_count": int(seq.frame_count), "fps": float(seq.fps)}
    if args.output_manifest is not None:
        write_motion_manifest(
            sequence_npz_path=output_npz,
            output_manifest_path=args.output_manifest,
            prompt=args.prompt,
            action_blocks=blocks,
            tags=("generated", args.backend),
        )
        summary["manifest_path"] = str(args.output_manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
