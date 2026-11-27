"""Known-vessel topology and shape-adjusted data for continuous target scans."""

from pathlib import Path
import json

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
import trimesh

from geometry import (
    ANGLES, ATLAS_PACK, BLENDER, BP, CLOUD, NORMALS, OPEN, PREVIOUS, TANGENTS,
    bone_prior, neutral_lines, profiles, read_mesh_pack, training_meshes,
)
from shape_training import BETAS, CT_KEYS, SMPL, fixed_profiles, guides, move_section


STATIONS = np.arange(-480., -1280.1, -5.)
STEP = float(abs(STATIONS[1] - STATIONS[0]))
ANGULAR_STEP = 2. * np.pi / len(ANGLES)
MISSING, MEASURED, BRIDGED = 0, 1, 2
MAIN = 0
CT = 3


def intervals(mask):
    locations = np.flatnonzero(mask)
    return [part for part in np.split(locations, np.flatnonzero(np.diff(locations) > 1) + 1)
            if part.size]


def fill_track(points, origin, radii, occupancy, allowed, identity):
    """Bridge internal supported gaps while preserving measured samples."""
    status = np.where(np.isfinite(points).all(axis=-1), MEASURED, MISSING).astype(np.uint8)
    joins = []
    good = np.flatnonzero(status)
    for first, last in zip(good[:-1], good[1:], strict=True):
        if last == first + 1:
            continue
        names = [allowed[first], allowed[last]]
        length = (last - first) * STEP
        event = {
            "parts": names, "from_mm": float(STATIONS[first]), "to_mm": float(STATIONS[last]),
            "samples_missing": int(last - first - 1),
        }
        if names[0] != names[1] and tuple(names) not in identity:
            joins.append(event | {"status": "different_vessels"})
            continue
        before = good[np.maximum(np.searchsorted(good, first) - 1, 0)]
        after = good[np.minimum(np.searchsorted(good, last) + 1, len(good) - 1)]
        support = np.unique([before, first, last, after])
        # For a long jump the ends must extrapolate in the same direction.
        shift = points[last] - points[first]
        incoming = (points[first] - points[before]) / max((first - before) * STEP, STEP)
        outgoing = (points[after] - points[last]) / max((after - last) * STEP, STEP)
        residual = np.linalg.norm(shift - length * (incoming + outgoing) * .5)
        if residual > max(12., .8 * length) or length > 80.:
            joins.append(event | {"status": "direction_or_distance", "residual_mm": float(residual)})
            continue
        sample = np.arange(first + 1, last)
        fitted = PchipInterpolator(support, points[support], axis=0)(sample)
        # The fitted segment is checked densely, independent of the output knots.
        dense = np.linspace(first, last, 2 * (last - first) + 1)[1:-1]
        path = PchipInterpolator(support, points[support], axis=0)(dense)
        center = np.column_stack([np.interp(dense, np.arange(len(STATIONS)), p) for p in origin.T])
        theta = np.mod(np.arctan2(*(path - center)[:, ::-1].T), 2. * np.pi) / ANGULAR_STEP
        lower = np.floor(dense).astype(int)
        lo = np.floor(theta).astype(int) % len(ANGLES)
        hi = (lo + 1) % len(ANGLES)
        wx, wy = theta % 1., dense % 1.
        radius = (1. - wy) * ((1. - wx) * radii[lower, lo] + wx * radii[lower, hi])
        radius += wy * ((1. - wx) * radii[lower + 1, lo] + wx * radii[lower + 1, hi])
        occ = occupancy(np.column_stack((path[:, 0], STATIONS[0] - STEP * dense, path[:, 1])))
        if np.any(np.linalg.norm(path - center, axis=-1) >= radius - 1.) or np.any(occ > .55):
            joins.append(event | {"status": "skin_or_bone"})
            continue
        points[sample] = fitted
        status[sample] = BRIDGED
        joins.append(event | {"status": "bridged", "residual_mm": float(residual)})
    return status, joins


def topology(meshes, groups, mode, origin, radii, *, continuations):
    occupancy = bone_prior()
    layers, support, graphs = [], [], []
    for idx, names in enumerate(groups):
        points = np.full((len(STATIONS), 2), np.nan)
        identity = [""] * len(STATIONS)
        for name in names:
            profile, _ = profiles(meshes, ((name,),), mode, STATIONS)
            selectors = np.isfinite(profile[:, 0]).all(axis=1)
            points[selectors] = profile[selectors, 0]
            for s in np.flatnonzero(selectors):
                identity[s] = name
        status, bridges = fill_track(points, origin, radii, occupancy, identity,
                                     continuations.get(idx, ()))
        # Keep separate nodes for each named vessel, including neighbouring
        # but independent vessels carried in a common nuisance role.
        ids = np.flatnonzero(status)
        connected = []
        for a, b in zip(ids[:-1], ids[1:], strict=True):
            legal = identity[a] == identity[b] or (identity[a], identity[b]) in continuations.get(idx, ())
            inferred = status[a] == BRIDGED or status[b] == BRIDGED
            if b == a + 1 and (legal or inferred):
                connected.append([int(a), int(b), "interpolated" if inferred else "supported"])
        graph = {
            "identity": list(names),
            "nodes": [
                {"station": int(s), "xyz_mm": [float(points[s, 0]), float(STATIONS[s]), float(points[s, 1])],
                 "name": identity[s] or None, "provenance": "sampled" if status[s] == MEASURED else "interpolated"}
                for s in ids
            ],
            "edges": connected,
            "gaps": bridges,
            "declared_continuations": list(continuations.get(idx, ())),
        }
        layers.append(points)
        support.append(status)
        graphs.append(graph)
    return np.stack(layers, axis=1), np.stack(support, axis=1), graphs


def verified_branches(graph, declared):
    """Attach only junctions for which the source provides a joint location."""
    records = []
    for parent, child, point, provenance in declared:
        a, b = (graph[i]["nodes"] for i in (parent, child))
        if a and b:
            a, b = (min(g, key=lambda n: np.linalg.norm(np.asarray(n["xyz_mm"]) - point)) for g in (a, b))
            a_dist = float(np.linalg.norm(np.asarray(a["xyz_mm"]) - point))
            b_dist = float(np.linalg.norm(np.asarray(b["xyz_mm"]) - point))
            if max(a_dist, b_dist) <= 2. * STEP:
                records.append({"xyz_mm": list(map(float, point)), "members": [[parent, a["station"]],
                                [child, b["station"]]], "source": provenance})
    return records


def paired_continuations(groups):
    return {i: tuple(zip(groups[i][:-1], groups[i][1:], strict=True))
            for i in (0, 1)}


def make_training(out):
    out.mkdir(exist_ok=True, parents=True)
    with np.load(SMPL, allow_pickle=False) as data:
        skin = data["v_template"].astype(float) * 1000.
        shapes = data["shapedirs"][..., :3].astype(float) * 1000.
        regressor, faces = data["J_regressor"], data["f"]
    base_guides = guides(skin, regressor @ skin)
    base_origin, base_radii = fixed_profiles(skin, faces, base_guides, STATIONS)
    archive = read_mesh_pack(ATLAS_PACK)
    medseg = {key: trimesh.Trimesh(*archive[f"MedSeg|{key}"], process=False)
              for role in CT_KEYS for key in role}
    sources = (*training_meshes(), medseg)
    groups = (BP, BP, OPEN, CT_KEYS)
    names = ("BodyParts3D right", "BodyParts3D mirrored left", "Open3DModel right", "MedSeg")
    rows = [topology(meshes, terms, "mesh", base_origin, base_radii,
                     continuations=paired_continuations(terms))
            for meshes, terms in zip(sources, groups, strict=True)]
    centers, status = (np.stack([r[k] for r in rows]) for k in (0, 1))
    ct_target = intervals(status[CT, :, MAIN] == MEASURED)
    if len(ct_target) != 1:
        raise ValueError("The MedSeg route must have one uninterrupted verified target span")
    lo, hi = [int(ct_target[0][i]) for i in (0, -1)]
    all_shapes = []
    for beta in BETAS:
        body = skin + np.einsum("vij,j->vi", shapes, beta)
        marks = guides(body, regressor @ body)
        y = np.interp(STATIONS, base_guides[:, 0, 1], marks[:, 0, 1])
        origin, radii = fixed_profiles(body, faces, marks, y)
        moved = np.stack([move_section(centers[:, s], base_origin, base_radii, origin, radii, s)
                          for s in range(len(STATIONS))], axis=1)
        all_shapes.append((y, origin, radii, moved))
    ys, origins, radii, anatomy = map(np.stack, zip(*all_shapes, strict=True))
    np.savez_compressed(out / "ct_geometry.npz", ys=ys, origins=origins, radii=radii, centers=anatomy,
                        status=status, ct_bounds=np.array([lo, hi]), neutral_grid=STATIONS)
    graph = {"coordinates": "neutral SMPL-X mm", "sources": dict(zip(names, [r[2] for r in rows], strict=True)),
             "target_role": MAIN, "training_bounds_mm": [float(-STATIONS[lo]), float(-STATIONS[hi])]}
    (out / "ct_reference_topology.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
    print("CT material stations", -STATIONS[lo], -STATIONS[hi], "shapes", len(ys), flush=True)
    for name, (_, labels, graphs) in zip(names, rows, strict=True):
        print(name, "bridged samples", (labels == BRIDGED).sum(axis=0).tolist(),
              "unresolved", sum(g["status"] != "bridged" for role in graphs for g in role["gaps"]), flush=True)


def blender_truth(out, origin, radii):
    arteries = json.loads((PREVIOUS / "evidence/arterial_paths.json").read_text())
    curves = neutral_lines() | {k: np.asarray(v["neutral"]) for k, v in arteries.items() if k.startswith("R_")}
    joined = paired_continuations(BLENDER)
    joined[1] = joined[1][:-1]  # R_CALF_M is an arterial candidate with uncertain junction.
    centers, status, graph = topology(curves, BLENDER, "line", origin, radii, continuations=joined)
    meeting = curves["R_SAPH_V"][0]
    near, _ = cKDTree(curves["R_COM_FEM_V"]).query(meeting)
    branches = verified_branches(graph, [(0, 2, meeting, "shared registered saphenofemoral endpoint")]) if near < 1.e-6 else []
    (out / "blender_topology.json").write_text(json.dumps({"coordinates": "neutral SMPL-X mm",
        "vessels": graph, "junctions": branches}, ensure_ascii=False, indent=2) + "\n")
    return centers, status


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    make_training(parser.parse_args().out)
