"""Extract only the local section statistics used by the static benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
from scipy.special import ndtr


SITES = (3, 4, 5)
VIEWS = (3, 7, 8)
OFFSET_UNITS = np.arange(-2, 3, dtype=float)
SHIFT_SD_MM = np.array([2.3, 1.7])
POSE_DEG = 15.
MAX_SECTION_DISTANCE_M = .045


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_banks(experiments):
    paths = (
        experiments / "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz",
        experiments / "ab_inverse_acquisition_b_20261113/motion/BANK.json.xz",
    )
    banks = []
    for path in paths:
        with lzma.open(path, "rt") as stream:
            banks.append(json.load(stream))
    selected = [row for row in banks[0]["nodes"] if row["realization"] == "dird_top"
                and row["site"] in SITES and row["view"] in VIEWS]
    selected.sort(key=lambda row: (row["site"], row["view"]))
    assert [(row["site"], row["view"]) for row in selected] == [(s, v) for s in SITES for v in VIEWS]
    index = {row["id"]: i for i, row in enumerate(selected)}
    cost = np.full((len(selected), len(selected), 4), np.inf)
    source = [[None for _ in selected] for _ in selected]
    for bank, path in zip(banks, paths, strict=True):
        for edge in bank["edges"]:
            if edge["start"] not in index or edge["goal"] not in index:
                continue
            i, j = index[edge["start"]], index[edge["goal"]]
            debit = np.array([edge["travel_mm"], edge["turn_deg"], edge["rail_mm"], edge["duration_s"]])
            if np.sum(debit[:3]) < np.sum(cost[i, j, :3]):
                cost[i, j] = debit
                source[i][j] = path.parent.parent.name
    return selected, cost, source, paths


def polygon_stats(poly):
    next_point = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * next_point[:, 1] - poly[:, 1] * next_point[:, 0]
    area = cross.sum() / 2.
    if abs(area) < 1.e-12:
        raise ValueError("degenerate vessel section")
    return ((poly + next_point) * cross[:, None]).sum(axis=0) / (6. * area), abs(area)


def extract(asset, archive):
    import trimesh

    selected, costs, edge_sources, banks = load_banks(archive)
    geometry_path = archive / "ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(geometry_path, allow_pickle=False) as geometry:
        pose = int(np.flatnonzero(geometry["pose_knee_deg"] == POSE_DEG).item())
        transforms = geometry["joint_transforms"][pose]
        points = geometry["posed_views_m"][pose]
        frames = geometry["posed_frames"][pose]
        nominal = geometry["posed_vessel_m"][pose]
        to_robot = geometry["robot_from_atlas"]
    # Only this locally generated, project-owned retarget output has object labels.
    with np.load(asset, allow_pickle=True) as vessel:
        points_h = np.column_stack((vessel["vertices_rest"], np.ones(len(vessel["vertices_rest"]))))
        transform = (vessel["lbs_weights"] @ transforms[:, :3].reshape(len(transforms), 12)).reshape(-1, 3, 4)
        vertices = np.einsum("nij,nj->ni", transform, points_h)
        faces = vessel["faces"]
        meshes = {}
        for name, (lo, hi) in zip(vessel["source_mesh_names"], vessel["source_vertex_ranges"], strict=True):
            if name in ("Artery", "Vein"):
                valid = np.all((faces >= lo) & (faces < hi), axis=1)
                meshes[name] = trimesh.Trimesh(vertices=vertices[lo:hi], faces=faces[valid] - lo, process=False)
    assert set(meshes) == {"Artery", "Vein"}

    basis = frames[SITES[0], VIEWS[0], :, 1:]
    offset = np.stack(np.meshgrid(OFFSET_UNITS, OFFSET_UNITS, indexing="ij"), axis=-1).reshape(-1, 2)
    shifts = ((offset * SHIFT_SD_MM) @ basis.T) / 1000.
    masses = np.diff(ndtr([-np.inf, -1.5, -.5, .5, 1.5, np.inf]))
    prior = np.outer(masses, masses).reshape(-1)
    centers = np.empty((len(selected), len(shifts), len(meshes), 2))
    radii = np.empty(centers.shape[:-1])
    position_error = []
    for i, row in enumerate(selected):
        s, v = row["site"], row["view"]
        origin, axes = points[s, v], frames[s, v]
        tip = to_robot[:3, :3] @ origin + to_robot[:3, 3]
        position_error.append(float(np.linalg.norm(tip * 1000. - np.asarray(row["tip_xyz_mm"]))))
        for h, shift in enumerate(shifts):
            for k, name in enumerate(("Artery", "Vein")):
                # Moving both tubes by d is equivalent to moving the section plane by -d.
                section = meshes[name].section(plane_normal=axes[:, 0], plane_origin=origin - shift)
                if section is None:
                    raise ValueError(f"Missing {name} section at pose {i}, shift {h}")
                curves = [line for line in section.discrete if len(line) >= 4
                          and np.linalg.norm(line[0] - line[-1]) < 1.e-8
                          and np.linalg.norm(line[:-1].mean(axis=0) - nominal[s]) < MAX_SECTION_DISTANCE_M]
                if len(curves) != 1:
                    raise ValueError(f"Expected one local {name} contour at pose {i}, shift {h}, found {len(curves)}")
                poly = (curves[0][:-1] + shift - origin) @ axes[:, 1:] * 1000.
                centers[i, h, k], area = polygon_stats(poly)
                radii[i, h, k] = np.sqrt(area / np.pi)
    if max(position_error) >= .01:
        raise ValueError("probe plane origins do not match the checked bank")
    return {
        "scope": "Section centroids/radii, one registered anatomical template; no body mesh or image data",
        "pose_knee_deg": POSE_DEG,
        "source_sha256": {
            "joint_transforms": sha256(geometry_path),
            "local_registered_artery_vein_export": sha256(asset),
            **{path.parent.parent.name: sha256(path) for path in banks},
        },
        "source_mesh_names": ["Artery", "Vein"],
        "poses": [{key: row[key] for key in ("id", "site", "view", "realization")} for row in selected],
        "positions_error_mm": position_error,
        "edge_resources": ["tcp_mm", "turn_deg", "rail_mm", "motion_s"],
        "edge_sources": edge_sources,
        "costs": [[row.tolist() if np.all(np.isfinite(row)) else None for row in rows] for rows in costs],
        "shift_mm": (shifts * 1000.).tolist(),
        "prior": prior.tolist(),
        "centers_mm": centers.tolist(),
        "section_radii_mm": radii.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset", "repo", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = extract(args.asset, args.repo / "NMI_idea/experiments")
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"poses": len(result["poses"]), "hypotheses": len(result["prior"]),
                      "max_bank_tip_error_mm": max(result["positions_error_mm"])}))


if __name__ == "__main__":
    main()
