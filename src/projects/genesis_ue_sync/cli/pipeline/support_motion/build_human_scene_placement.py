#!/usr/bin/env python3
"""Offline GT bed-fit: write HumanScenePlacement JSON for Genesis + UE scene_init."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.project import project_paths

from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence, load_amass_sequence
from projects.genesis_ue_sync.sim_platform.human_refit.placement_resolver import (
    preferred_placement_output_path,
    resolve_or_compute_placement_for_amass,
)
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import prepare_smpl_capsule_runtime_asset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass-npz", type=Path, required=True)
    p.add_argument(
        "--scene-spec",
        type=Path,
        default=Path("configs/scenes/amass_lie_sync_scene.yaml"),
    )
    p.add_argument("--fit-samples", type=int, default=11)
    p.add_argument(
        "--human-center-mode",
        type=str,
        default="bed_center",
        choices=("bed_center", "scene_anchor"),
    )
    p.add_argument(
        "--root-projection-bed-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After bed-fit, shift XY so frame-0 root lies on bed center.",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Override placement JSON path.")
    p.add_argument("--force", action="store_true", help="Recompute even if sidecar exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = project_paths(__file__).root
    scene_path = args.scene_spec if args.scene_spec.is_absolute() else repo / args.scene_spec
    npz_path = args.amass_npz if args.amass_npz.is_absolute() else repo / args.amass_npz
    scene_spec = load_sync_scene_spec(scene_path)
    seq = load_amass_sequence(npz_path)
    capsule_asset = prepare_smpl_capsule_runtime_asset(
        seq,
        cache_dir=repo / "outputs" / "genesis_capsule_urdf_cache",
        device="cpu",
        genesis_proxy="mjcf",
    )
    placement, sidecar = resolve_or_compute_placement_for_amass(
        scene_spec,
        seq,
        amass_npz_path=npz_path,
        repo_root=repo,
        proxy_geometry=capsule_asset.proxy_geometry,
        placement_sample_frames=int(args.fit_samples),
        device="cpu",
        force_recompute=bool(args.force),
        human_center_mode=str(args.human_center_mode),
        root_projection_bed_center=bool(args.root_projection_bed_center),
        fit_samples=int(args.fit_samples),
    )
    out_path = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else preferred_placement_output_path(scene_spec, npz_path, repo_root=repo)
    )
    if out_path != sidecar:
        placement.save(out_path)
    print(f"human_scene_placement: {out_path}")
    print(f"world_offset_m={list(placement.world_offset_m)} support_plane_z_m={placement.support_plane_z_m:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
