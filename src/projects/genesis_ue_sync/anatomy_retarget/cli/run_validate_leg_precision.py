#!/usr/bin/env python3
"""Validate high-precision canonical leg coordinate refiners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.paths import atlas_path, resolve_repo_path
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.surface_refine import (
    SurfaceAtlasRefiner,
    wrap_angle_delta,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas", type=Path, default=atlas_path("left"), help="Leg atlas .npz path.")
    p.add_argument("--samples", type=int, default=32, help="Random volume xi samples.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-mm", type=float, default=1.0, help="Maximum accepted volume roundtrip error in mm.")
    p.add_argument("--surface-max-mm", type=float, default=0.01, help="Maximum accepted surface vertex error in mm.")
    p.add_argument("--theta-max-rad", type=float, default=1.0e-2, help="Maximum accepted theta roundtrip error.")
    return p.parse_args()


def _stats_mm(errors_mm: np.ndarray) -> dict[str, float]:
    return {
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors_mm)))),
        "p95_mm": float(np.percentile(errors_mm, 95.0)),
        "p99_mm": float(np.percentile(errors_mm, 99.0)),
        "max_mm": float(np.max(errors_mm)),
    }


def main() -> int:
    args = parse_args()
    atlas = load_leg_volume_atlas(resolve_repo_path(args.atlas))
    surface = SurfaceAtlasRefiner.from_atlas(atlas)
    volume = VolumeTetRefiner.from_atlas(
        atlas,
        candidate_k=12,
        newton_steps=0,
    )
    if volume is None:
        raise SystemExit("Atlas does not contain harmonic tetrahedral fields.")

    skin_xi = np.stack([atlas.skin_theta, atlas.skin_h, np.zeros_like(atlas.skin_h)], axis=1).astype(np.float32)
    skin_pred = surface.xi_to_p(skin_xi, reference_points=atlas.skin_vertices)
    surface_err_mm = np.linalg.norm(skin_pred - atlas.skin_vertices, axis=1) * 1000.0

    rng = np.random.default_rng(int(args.seed))
    n = min(max(1, int(args.samples)), int(atlas.volume_points.shape[0]))
    idx = rng.choice(atlas.volume_points.shape[0], size=n, replace=False)
    volume_p = atlas.volume_points[idx].astype(np.float32)
    volume_xi = atlas.volume_xi[idx].astype(np.float32)
    volume_xi_back = volume.p_to_xi(volume_p)
    theta_err = np.abs(wrap_angle_delta(volume_xi_back[:, 0], volume_xi[:, 0]))
    h_err = np.abs(volume_xi_back[:, 1] - volume_xi[:, 1])
    d_err = np.abs(volume_xi_back[:, 2] - volume_xi[:, 2])
    coord_err = np.sqrt(np.square(theta_err) + np.square(h_err) + np.square(d_err))

    metrics = {
        "atlas": str(resolve_repo_path(args.atlas)),
        "sample_count": int(volume_xi.shape[0]),
        "surface_vertex": _stats_mm(surface_err_mm),
        "volume_p_to_xi_coord": {
            "rmse": float(np.sqrt(np.mean(np.square(coord_err)))),
            "p95": float(np.percentile(coord_err, 95.0)),
            "p99": float(np.percentile(coord_err, 99.0)),
            "max": float(np.max(coord_err)),
        },
        "volume_theta_p95_rad": float(np.percentile(theta_err, 95.0)),
        "volume_theta_max_rad": float(np.max(theta_err)),
        "volume_h_p95": float(np.percentile(h_err, 95.0)),
        "volume_h_max": float(np.max(h_err)),
        "volume_d_p95": float(np.percentile(d_err, 95.0)),
        "volume_d_max": float(np.max(d_err)),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    ok = (
        metrics["surface_vertex"]["max_mm"] <= float(args.surface_max_mm)
        and metrics["volume_p_to_xi_coord"]["max"] <= float(args.theta_max_rad)
    )
    print("ACCEPT" if ok else "REJECT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
