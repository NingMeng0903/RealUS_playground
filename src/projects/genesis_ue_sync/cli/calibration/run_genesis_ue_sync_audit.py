#!/usr/bin/env python3
"""Emit Genesis vs UE alignment audit JSON (bed, cameras, SMPL placement samples, robot URDF FK)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.scenes import resolve_scene_spec_with_augmentation
from projects.genesis_ue_sync.sim_platform.scenes.human_scene_placement import HumanScenePlacement
from projects.genesis_ue_sync.sim_platform.scenes.sync_alignment_audit import build_genesis_ue_sync_audit_report, write_sync_audit_json


def _parse_basis_rpy_deg_env() -> tuple[float, float, float]:
    raw = str(os.environ.get("AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG", "0 0 0")).strip()
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return (0.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-spec", type=Path, default=paths.default_scene_spec_path)
    p.add_argument("--augmentation-spec", type=Path, default=None)
    p.add_argument("--sequence-npz", type=Path, default=None, help="HumanMotionSequence npz (optional; enables SMPL placement)")
    p.add_argument("--human-placement-json", type=Path, default=None, help="Use precomputed HumanScenePlacement JSON")
    p.add_argument("--placement-sample-frames", type=int, default=11)
    p.add_argument("--torch-device", type=str, default=None, help="cpu/cuda for SMPL eval")
    p.add_argument("--output-json", type=Path, default=paths.tmp_root / "genesis_ue_sync_audit.json")
    p.add_argument("--robot-visual-basis-rpy-deg", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_spec, _aug = resolve_scene_spec_with_augmentation(args.scene_spec, args.augmentation_spec)

    seq_path = None if args.sequence_npz is None else args.sequence_npz.expanduser().resolve()
    sequence: HumanMotionSequence | None = None
    if seq_path is not None:
        sequence = HumanMotionSequence.load(seq_path)

    placement: HumanScenePlacement | None = None
    if args.human_placement_json is not None:
        placement = HumanScenePlacement.load(args.human_placement_json.expanduser().resolve())

    basis = tuple(float(v) for v in args.robot_visual_basis_rpy_deg) if args.robot_visual_basis_rpy_deg is not None else _parse_basis_rpy_deg_env()

    report = build_genesis_ue_sync_audit_report(
        scene_spec,
        sequence=sequence,
        sequence_npz_path=str(seq_path) if seq_path is not None else None,
        device=args.torch_device,
        placement_sample_frames=int(args.placement_sample_frames),
        human_placement=placement,
        robot_visual_basis_rpy_deg=basis,
    )

    write_sync_audit_json(report, args.output_json.expanduser().resolve())
    print(json.dumps({"written": str(args.output_json), "scene": scene_spec.name}, indent=2))


if __name__ == "__main__":
    main()
