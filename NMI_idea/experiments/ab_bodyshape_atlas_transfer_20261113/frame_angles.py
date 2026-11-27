"""Measure physical frame rotation at fixed addresses under shape changes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline

from transport import ShapeTransport


def tet_edges(vertices, cells):
    points = vertices[cells]
    return np.moveaxis(points[:, 1:] - points[:, :1], 1, 2)


def unit(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def gradients_to_frame(gradients):
    """Use the slide's h/d frame, with depth projected off the h axis."""
    h = unit(gradients[:, :, 0])
    depth = gradients[:, :, 1]
    d = unit(depth - h * np.sum(h * depth, axis=1, keepdims=True))
    return np.stack((np.cross(h, d), h, d), axis=2)


def rotation_angles(reference, frame):
    cosine = (np.einsum("nij,nij->n", reference, frame) - 1.) * .5
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def centerline_frames(asset, vertices, joints, parallel_frames):
    """Rebuild geometric axes on the transported centerline at fixed stations."""
    rings = vertices.reshape(asset.grid_shape + (3,))
    outer, inner = rings[:, -1], rings[:, 0]
    fraction = asset.radial_fractions[0]
    center = ((inner - fraction * outer) / (1. - fraction)).mean(axis=1)
    stations = asset.h.reshape(asset.grid_shape)[:, 0, 0]
    tangent = CubicSpline(stations, center)(stations, 1)
    hip = 1 if asset.side == "left" else 2
    frame = parallel_frames(tangent, joints[0] - joints[hip])
    uv = (outer - center[:, None]) @ frame[:, :, :2]
    azimuth = np.arctan2(uv[..., 1], uv[..., 0])
    return frame, azimuth


def compare(model, report, parallel_frames):
    rows = []
    for side, data in report["sides"].items():
        asset = model.sides[side]
        cells = asset.tets
        differences = np.stack((asset.h[cells], asset.depth[cells]), axis=-1)
        differences = differences[:, 1:] - differences[:, :1]
        # Use all cells, and hold knee angle fixed between each beta pair.
        references = {}
        for angle in report["poses_deg"]:
            v0, _, _, joints0 = model.deform(side, (0., 0.), angle)
            E0 = tet_edges(v0.astype(np.float32).astype(float), cells)
            grad0 = np.linalg.solve(E0.transpose(0, 2, 1), differences)
            geom0, theta0 = centerline_frames(asset, v0, joints0, parallel_frames)
            references[angle] = np.linalg.inv(E0), grad0, gradients_to_frame(grad0), geom0, theta0
        for case in data["cases"]:
            if not case.get("material_check_pass") or case["beta_delta"] == [0., 0.]:
                continue
            angle = case["knee_deg"]
            v1, _, _, joints1 = model.deform(side, case["beta_delta"], angle)
            E1 = tet_edges(v1.astype(np.float32).astype(float), cells)
            grad1 = np.linalg.solve(E1.transpose(0, 2, 1), differences)
            inv_E0, grad0, R0, geom0, theta0 = references[angle]
            A = E1 @ inv_E0
            transported_grad1 = np.linalg.solve(A.transpose(0, 2, 1), grad0)
            R1 = gradients_to_frame(grad1)
            R1_transport = gradients_to_frame(transported_grad1)
            changes = np.quantile(rotation_angles(R0, R1), [.5, .95, 1.])
            geom1, theta1 = centerline_frames(asset, v1, joints1, parallel_frames)
            geom_angles = rotation_angles(geom0, geom1)
            azimuth_drift = np.rad2deg(np.abs(np.angle(np.exp(1j * (theta1 - theta0)))))
            rows.append(dict(side=side, beta_delta=case["beta_delta"], knee_deg=angle,
                             sample_tets=len(cells), angle_median_deg=float(changes[0]),
                             angle_p95_deg=float(changes[1]), angle_max_deg=float(changes[2]),
                             centerline_frame_max_deg=float(geom_angles.max()),
                             remeasured_azimuth_max_drift_deg=float(azimuth_drift.max()),
                             gradient_transport_frame_max_abs_error=float(np.abs(R1 - R1_transport).max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.among_us.resolve() / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import (
        parallel_transport_frames,
    )

    model = ShapeTransport(args.among_us, args.repair)
    output = {"scope": "Paired body shapes at the same knee pose, with admitted geometry only. "
                       "Compare h/d frames, geometric centerline frames and outer-point azimuths.",
              "grids": {}}
    for file in args.geometry:
        source = file.read_bytes()
        report = json.loads(source)
        rows = compare(model, report, parallel_transport_frames)
        output["grids"][report["grid"]] = {
            "geometry_input_sha256": hashlib.sha256(source).hexdigest(),
            "max_case_p95_deg": max(row["angle_p95_deg"] for row in rows),
            "max_deg": max(row["angle_max_deg"] for row in rows),
            "centerline_frame_max_deg": max(row["centerline_frame_max_deg"] for row in rows),
            "remeasured_azimuth_max_drift_deg": max(row["remeasured_azimuth_max_drift_deg"] for row in rows),
            "frame_transport_max_abs_error": max(row["gradient_transport_frame_max_abs_error"] for row in rows),
            "cases": rows,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
