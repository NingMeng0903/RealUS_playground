#!/usr/bin/env python3
"""Bake canonical SMPL left/right leg volume coordinates and diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates import (
    LegVolumeConfig,
    bake_leg_volume_atlases,
    project_vessel_centerlines_to_skin,
    save_leg_volume_atlas,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_canonical_smpl
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.visualize import (
    draw_d_slice_contours,
    draw_leg_volume_fields_3d,
    draw_vessel_projection,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-dir", type=Path, default=Path("outputs/anatomy_retarget/latest_canonical"))
    p.add_argument("--output-dir", type=Path, default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas"))
    p.add_argument(
        "--vessel-centerlines",
        type=Path,
        default=Path("outputs/anatomy_retarget/limb_vessel_planning/centerlines/vessel_centerlines_rest.obj"),
    )
    p.add_argument("--station-count", type=int, default=48)
    p.add_argument("--skin-sample-stride", type=int, default=1)
    p.add_argument("--no-vessel-projection", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    cfg = LegVolumeConfig(
        station_count=int(args.station_count),
        skin_sample_stride=int(args.skin_sample_stride),
    )
    atlases = bake_leg_volume_atlases(args.canonical_dir, config=cfg)
    save_leg_volume_atlas(out / "atlas_left.npz", atlases["left"])
    save_leg_volume_atlas(out / "atlas_right.npz", atlases["right"])
    draw_leg_volume_fields_3d(figs / "leg_volume_fields_3d.png", atlases)
    draw_d_slice_contours(figs / "left_leg_d_slice_contours.png", atlases["left"])

    if not args.no_vessel_projection and Path(args.vessel_centerlines).is_file():
        _projection, projected_lines = project_vessel_centerlines_to_skin(
            args.vessel_centerlines,
            atlases,
            output_obj=out / "vessel_centerlines_skin_projected.obj",
            output_npz=out / "vessel_skin_projection.npz",
        )
        smpl_vertices, _smpl_faces, _skeleton = load_canonical_smpl(args.canonical_dir)
        draw_vessel_projection(
            figs / "vessel_projection_d0.png",
            args.vessel_centerlines,
            projected_lines,
            atlases=atlases,
            smpl_vertices=smpl_vertices,
        )
    print(f"INFO leg volume coordinates exported -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
