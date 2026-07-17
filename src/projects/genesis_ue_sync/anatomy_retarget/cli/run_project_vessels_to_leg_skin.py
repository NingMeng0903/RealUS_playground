#!/usr/bin/env python3
"""Project T-pose vessel centerlines to the baked SMPL leg skin d=0."""

from __future__ import annotations

import argparse
from pathlib import Path

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates import (
    load_leg_volume_atlas,
    project_vessel_centerlines_to_skin,
    remap_vessel_projection_to_skin,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_canonical_smpl
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.paths import (
    layered_atlas_path,
    leg_volume_production_figures_dir,
    leg_volume_production_vessels_dir,
    production_atlas_path,
    production_vessel_material_path,
    resolve_repo_path,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.visualize import draw_vessel_projection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas-left", type=Path, default=None)
    p.add_argument("--atlas-right", type=Path, default=None)
    p.add_argument("--canonical-dir", type=Path, default=Path("outputs/anatomy_retarget/latest_canonical"))
    p.add_argument(
        "--vessel-centerlines",
        type=Path,
        default=Path("outputs/anatomy_retarget/limb_vessel_planning/centerlines/vessel_centerlines_rest.obj"),
    )
    p.add_argument(
        "--source-projection-npz",
        type=Path,
        default=None,
        help="Existing baked vessel xi_skin to remap onto the current atlas surface. Falls back to reprojection if missing.",
    )
    p.add_argument("--force-reproject", action="store_true", help="Ignore --source-projection-npz and recompute projection.")
    p.add_argument("--bake-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bake_dir = resolve_repo_path(args.bake_dir) if args.bake_dir is not None else leg_volume_production_vessels_dir()
    figs = leg_volume_production_figures_dir()
    bake_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    default_left = production_atlas_path("left") if production_atlas_path("left").is_file() else layered_atlas_path("left")
    default_right = production_atlas_path("right") if production_atlas_path("right").is_file() else layered_atlas_path("right")
    left = resolve_repo_path(args.atlas_left) if args.atlas_left is not None else default_left
    right = resolve_repo_path(args.atlas_right) if args.atlas_right is not None else default_right
    atlases = {
        "left": load_leg_volume_atlas(left),
        "right": load_leg_volume_atlas(right),
    }
    if args.source_projection_npz is not None:
        source_projection = resolve_repo_path(args.source_projection_npz)
    else:
        legacy_source = resolve_repo_path("outputs/anatomy_retarget/leg_volume_coordinates/vessel_skin_projection.npz")
        source_projection = production_vessel_material_path() if production_vessel_material_path().is_file() else legacy_source
    if source_projection.is_file() and not bool(args.force_reproject):
        _projection, projected_lines = remap_vessel_projection_to_skin(
            source_projection,
            atlases,
            output_obj=bake_dir / "vessel_centerlines_skin_projected.obj",
            output_npz=bake_dir / "vessel_skin_projection.npz",
        )
        print(f"INFO remapped baked vessel coordinates from {source_projection}")
    else:
        _projection, projected_lines = project_vessel_centerlines_to_skin(
            args.vessel_centerlines,
            atlases,
            output_obj=bake_dir / "vessel_centerlines_skin_projected.obj",
            output_npz=bake_dir / "vessel_skin_projection.npz",
        )
        print("INFO recomputed vessel projection from centerlines")
    smpl_vertices, _smpl_faces, _skeleton = load_canonical_smpl(args.canonical_dir)
    draw_vessel_projection(
        figs / "vessel_projection_d0.png",
        args.vessel_centerlines,
        projected_lines,
        atlases=atlases,
        smpl_vertices=smpl_vertices,
    )
    print(f"INFO vessel projection exported -> {bake_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
