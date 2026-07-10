from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import load_sync_scene_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a scene spec copy that points at a generated/refit human motion NPZ.")
    p.add_argument("--base-scene-spec", type=Path, required=True)
    p.add_argument("--sequence-npz", type=Path, required=True)
    p.add_argument("--output-scene-spec", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_sync_scene_payload(args.base_scene_spec)
    seq = HumanMotionSequence.load(args.sequence_npz)
    motion = dict(payload.get("motion", {}))
    motion["source_id"] = f"generated/{seq.sequence_name}"
    motion["source_path"] = ""
    motion["sequence_npz_path"] = str(args.sequence_npz)
    motion["fps"] = float(seq.fps)
    motion["frame_count"] = int(seq.frame_count)
    motion["start_frame"] = 0
    motion["frame_step"] = 1
    payload["motion"] = motion
    render = dict(payload.get("render", {}))
    render["fps"] = float(seq.fps)
    render["frame_limit"] = int(seq.frame_count)
    render["ue_frame_count"] = int(seq.frame_count)
    render["ue_frame_step"] = 1
    payload["render"] = render
    args.output_scene_spec.parent.mkdir(parents=True, exist_ok=True)
    args.output_scene_spec.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_scene_spec": str(args.output_scene_spec), "sequence_npz_path": str(args.sequence_npz)}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
