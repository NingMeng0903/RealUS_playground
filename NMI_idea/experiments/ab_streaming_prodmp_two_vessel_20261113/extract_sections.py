"""Extract dense center-view sections between dynamic geometry stations.

The extractor follows the registered two-vessel section rule used by
``ab_static_two_vessel_b_20261113/extract_geometry.py``.  Only the compact
section table is written; the registered mesh and frozen geometry remain
inputs to the command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


POSE_DEG = 15.0
VIEW = 3
STATIONS = (3, 4, 5)
SUBSEGMENTS = 32
MAX_SECTION_DISTANCE_M = 0.045
VESSEL_NAMES = ("Artery", "Vein")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one input file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_stats(poly: np.ndarray) -> tuple[np.ndarray, float]:
    """Return centroid and absolute area for an open polygon vertex list."""

    next_point = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * next_point[:, 1] - poly[:, 1] * next_point[:, 0]
    area = cross.sum() / 2.0
    if abs(area) < 1.0e-12:
        raise ValueError("degenerate vessel section")
    centroid = ((poly + next_point) * cross[:, None]).sum(axis=0) / (6.0 * area)
    return centroid, abs(area)


def interpolate_stations(
    station_values: np.ndarray,
    station_s_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate stations 3->4 and 4->5, retaining exact station endpoints.

    ``station_values`` has leading station dimension and is interpolated
    linearly.  The output has 65 entries: 33 from 3->4 followed by 32 new
    entries from 4->5.  ``Slerp`` is intentionally handled by the caller for
    orientation-valued frames.
    """

    pieces = []
    s_pieces = []
    for left, right in zip(STATIONS[:-1], STATIONS[1:], strict=True):
        fractions = np.linspace(0.0, 1.0, SUBSEGMENTS + 1)
        values = (1.0 - fractions.reshape((-1,) + (1,) * (station_values.ndim - 1))) * station_values[left]
        values += fractions.reshape((-1,) + (1,) * (station_values.ndim - 1)) * station_values[right]
        positions = station_s_mm[left] + fractions * (station_s_mm[right] - station_s_mm[left])
        if pieces:
            values = values[1:]
            positions = positions[1:]
        pieces.append(values)
        s_pieces.append(positions)
    return np.concatenate(pieces, axis=0), np.concatenate(s_pieces, axis=0) - station_s_mm[STATIONS[0]]


def interpolate_frames(frames: np.ndarray, station_s_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate frame matrices over both station intervals by SO(3) Slerp."""

    pieces = []
    s_pieces = []
    for left, right in zip(STATIONS[:-1], STATIONS[1:], strict=True):
        fractions = np.linspace(0.0, 1.0, SUBSEGMENTS + 1)
        key_s = np.array([station_s_mm[left], station_s_mm[right]], dtype=float)
        key_rotations = Rotation.from_matrix(frames[[left, right]])
        interpolator = Slerp(key_s, key_rotations)
        values = interpolator(station_s_mm[left] + fractions * (station_s_mm[right] - station_s_mm[left])).as_matrix()
        positions = station_s_mm[left] + fractions * (station_s_mm[right] - station_s_mm[left])
        if pieces:
            values = values[1:]
            positions = positions[1:]
        pieces.append(values)
        s_pieces.append(positions)
    return np.concatenate(pieces, axis=0), np.concatenate(s_pieces, axis=0) - station_s_mm[STATIONS[0]]


def retarget_meshes(asset: Path, transforms: np.ndarray):
    """Decode the registered Artery and Vein meshes at the selected pose."""

    import trimesh

    with np.load(asset, allow_pickle=True) as vessel:
        points_h = np.column_stack((vessel["vertices_rest"], np.ones(len(vessel["vertices_rest"]))))
        # This is the same registered LBS decode as the static extractor.
        transform = (vessel["lbs_weights"] @ transforms[:, :3].reshape(len(transforms), 12)).reshape(-1, 3, 4)
        vertices = np.einsum("nij,nj->ni", transform, points_h)
        faces = vessel["faces"]
        meshes = {}
        for name, (lo, hi) in zip(vessel["source_mesh_names"], vessel["source_vertex_ranges"], strict=True):
            if name in VESSEL_NAMES:
                valid = np.all((faces >= lo) & (faces < hi), axis=1)
                meshes[name] = trimesh.Trimesh(vertices=vertices[lo:hi], faces=faces[valid] - lo, process=False)
    if set(meshes) != set(VESSEL_NAMES):
        raise ValueError(f"registered asset must contain {VESSEL_NAMES}, found {tuple(meshes)}")
    return meshes


def section_one(
    mesh,
    origin_m: np.ndarray,
    axes: np.ndarray,
    prior_center_m: np.ndarray,
    shift_m: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Section one mesh and return projected centroid/radius in millimetres."""

    if shift_m is None:
        shift_m = np.zeros(3, dtype=float)
    # Moving both tubes by d is equivalent to moving the section plane by -d.
    section = mesh.section(plane_normal=axes[:, 0], plane_origin=origin_m - shift_m)
    if section is None:
        raise ValueError("mesh has no section at interpolated slice")
    curves = [
        line
        for line in section.discrete
        if len(line) >= 4
        and np.linalg.norm(line[0] - line[-1]) < 1.0e-8
        and np.linalg.norm(line[:-1].mean(axis=0) - prior_center_m) < MAX_SECTION_DISTANCE_M
    ]
    if len(curves) != 1:
        raise ValueError(f"expected one local closed contour, found {len(curves)}")
    # The closed point is repeated at the end; omit it for polygon statistics.
    poly = (curves[0][:-1] + shift_m - origin_m) @ axes[:, 1:] * 1000.0
    center, area = polygon_stats(poly)
    return center, float(np.sqrt(area / np.pi))


def extract(asset: Path, dynamic: Path, nominal: Path) -> dict:
    """Build the 65-slice center-view geometry table."""

    with np.load(dynamic, allow_pickle=False) as geometry:
        pose_matches = np.flatnonzero(geometry["pose_knee_deg"] == POSE_DEG)
        if len(pose_matches) != 1:
            raise ValueError(f"expected one {POSE_DEG:g}-degree pose, found {pose_matches.tolist()}")
        pose = int(pose_matches[0])
        station_s_mm = np.asarray(geometry["s_mm"], dtype=float)
        origins_station_m = np.asarray(geometry["posed_views_m"][pose, :, VIEW], dtype=float)
        frames_station = np.asarray(geometry["posed_frames"][pose, :, VIEW], dtype=float)
        prior_station_m = np.asarray(geometry["posed_vessel_m"][pose], dtype=float)
        transforms = np.asarray(geometry["joint_transforms"][pose], dtype=float)

    for index in STATIONS:
        if index < 0 or index >= len(station_s_mm):
            raise IndexError(f"station index {index} outside frozen table")
    if not np.all(np.diff(station_s_mm) > 0):
        raise ValueError("frozen station s_mm must be strictly increasing")

    origins_m, s_mm = interpolate_stations(origins_station_m, station_s_mm)
    prior_centers_m, prior_s_mm = interpolate_stations(prior_station_m, station_s_mm)
    frames, frame_s_mm = interpolate_frames(frames_station, station_s_mm)
    if not np.array_equal(s_mm, prior_s_mm) or not np.array_equal(s_mm, frame_s_mm):
        raise AssertionError("interpolated station grids disagree")
    if len(s_mm) != 2 * SUBSEGMENTS + 1 or not np.all(np.diff(s_mm) > 0) or s_mm[0] != 0.0:
        raise AssertionError("invalid dense station grid")

    meshes = retarget_meshes(asset, transforms)
    centers_mm = np.empty((len(s_mm), len(VESSEL_NAMES), 2), dtype=float)
    radii_mm = np.empty((len(s_mm), len(VESSEL_NAMES)), dtype=float)
    for i, (origin, axes, prior) in enumerate(zip(origins_m, frames, prior_centers_m, strict=True)):
        for vessel_index, name in enumerate(VESSEL_NAMES):
            centers_mm[i, vessel_index], radii_mm[i, vessel_index] = section_one(
                meshes[name], origin, axes, prior
            )

    result = {
        "s_mm": s_mm.tolist(),
        "centers_mm": centers_mm.tolist(),
        "radii_mm": radii_mm.tolist(),
        "origin_atlas_mm": (origins_m * 1000.0).tolist(),
        "axes_atlas": frames.tolist(),
        "source_sha256": {
            "dynamic_geometry_frozen_npz": sha256(dynamic),
            "local_registered_artery_vein_export": sha256(asset),
            "nominal_static_geometry_json": sha256(nominal),
        },
        "metadata": {
            "schema": "ab_streaming_prodmp_two_vessel_sections_v1",
            "pose_knee_deg": POSE_DEG,
            "view": VIEW,
            "station_indices": list(STATIONS),
            "subsegments_per_interval": SUBSEGMENTS,
            "mesh_names": list(VESSEL_NAMES),
            "frame_convention": "columns: normal, lateral, depth",
            "section_rule": {
                "contour": "one closed local contour, len >= 4",
                "prior_center": "linear interpolation of posed_vessel_m at the selected stations",
                "max_distance_mm": MAX_SECTION_DISTANCE_M * 1000.0,
                "projection": "(contour - origin) @ axes[:, 1:3] in millimetres",
            },
        },
    }

    # Compare the exact endpoint sections to the zero-shift center-view rows in
    # the static artifact.  This check is kept in the JSON so the compact output
    # carries its own endpoint audit.
    with nominal.open() as stream:
        static = json.load(stream)
    static_centers = np.asarray(static["centers_mm"], dtype=float)
    static_radii = np.asarray(static["section_radii_mm"], dtype=float)
    # Static pose order is (site 3 view 3), (site 3 view 7), (site 3 view 8),
    # then site 4's triplet and site 5's triplet.  The zero-shift hypothesis is
    # the middle entry of the 25-shift bank.
    static_pose_indices = np.array([0, 3, 6], dtype=int)
    endpoint_indices = np.array([0, SUBSEGMENTS, 2 * SUBSEGMENTS], dtype=int)
    endpoint_center_error = np.linalg.norm(
        centers_mm[endpoint_indices] - static_centers[static_pose_indices, 12], axis=-1
    )
    endpoint_radius_error = np.abs(radii_mm[endpoint_indices] - static_radii[static_pose_indices, 12])
    result["metadata"]["endpoint_static_check"] = {
        "nominal_shift_index": 12,
        "static_pose_indices": static_pose_indices.tolist(),
        "max_center_discrepancy_mm": float(endpoint_center_error.max()),
        "max_radius_discrepancy_mm": float(endpoint_radius_error.max()),
        "center_discrepancies_mm": endpoint_center_error.tolist(),
        "radius_discrepancies_mm": endpoint_radius_error.tolist(),
        "centers_range_mm": {
            "min": centers_mm.min(axis=(0, 1)).tolist(),
            "max": centers_mm.max(axis=(0, 1)).tolist(),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path, help="registered vessels_local NPZ")
    parser.add_argument("--dynamic", required=True, type=Path, help="frozen dynamic geometry NPZ")
    parser.add_argument("--nominal", required=True, type=Path, help="nominal static GEOMETRY.json")
    parser.add_argument("--out", required=True, type=Path, help="output SECTIONS.json")
    args = parser.parse_args()
    result = extract(args.asset, args.dynamic, args.nominal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    check = result["metadata"]["endpoint_static_check"]
    print(
        json.dumps(
            {
                "slices": len(result["s_mm"]),
                "max_endpoint_center_discrepancy_mm": check["max_center_discrepancy_mm"],
                "max_endpoint_radius_discrepancy_mm": check["max_radius_discrepancy_mm"],
                "centers_range_mm": check["centers_range_mm"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
