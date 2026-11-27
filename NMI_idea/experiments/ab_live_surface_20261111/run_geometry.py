"""Offline comparison of nominal and measured probe frames on production skin anchors."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2"
SEED = 721311
POSES = range(0, 91, 15)
STEP_MM = np.array([4.0, 0.0, 0.0])
PATCH = np.stack(np.meshgrid(np.linspace(-10.0, 10.0, 9), np.linspace(-10.0, 10.0, 9)), -1).reshape(-1, 2)
POSITION_CONTRACT_MM = 1e-5
ROTATION_CONTRACT = 1e-8


def unit(vector):
    length = np.linalg.norm(vector)
    if length <= 1e-10:
        raise ValueError("The direction has zero length.")
    return vector / length


def frame(normal, direction):
    tangent = unit(direction - normal * (direction @ normal))
    return np.column_stack((tangent, np.cross(normal, tangent), normal))


def incident_normals(vertices, faces, face_ids):
    triangles = vertices[faces[face_ids]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return unit(normals.sum(axis=0))


def skin_anchors(side, rng):
    with np.load(ASSETS / f"atlas_{side}.npz") as atlas:
        skin = atlas["skin_vertices"].astype(float)
        faces = atlas["skin_faces"]
        h, theta = atlas["skin_h"], atlas["skin_theta"]
        outward = atlas["skin_normals"]
        distances, mapping = cKDTree(atlas["harmonic_vertices"]).query(skin)
        assert np.max(distances) == 0 and len(np.unique(mapping)) == len(skin)
        assert np.all(atlas["harmonic_d"][mapping] == 0)
    eligible = np.flatnonzero((h >= 0.20) & (h <= 0.88))
    ids = rng.choice(eligible, size=32, replace=False)
    incident = [np.flatnonzero((faces == index).any(axis=1)) for index in ids]
    signs = [
        np.sign(incident_normals(skin, faces, face_ids) @ outward[index])
        for index, face_ids in zip(ids, incident)
    ]
    bias = rng.uniform(-6.0, 6.0, size=len(ids))
    tilt = np.deg2rad(rng.uniform(0.0, 15.0, size=len(ids)))
    azimuth = rng.uniform(-np.pi, np.pi, size=len(ids))
    slopes = np.tan(tilt[:, None]) * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
    addresses = [
        {"side": side, "skin_vertex_id": int(index), "h": float(h[index]), "theta_rad": float(theta[index]), "depth": 0}
        for index in ids
    ]
    for angle in POSES:
        with np.load(ASSETS / f"{side}_pose_{angle:02}.npz") as data:
            posed = data["vertices"][mapping].astype(float) * 1000.0
        for k, (index, face_ids) in enumerate(zip(ids, incident)):
            assert abs(signs[k]) == 1
            normal = signs[k] * incident_normals(posed, faces, face_ids)
            neighbors = np.unique(faces[face_ids])
            delta = posed[neighbors] - posed[index]
            delta -= (delta @ normal)[:, None] * normal
            direction = np.linalg.lstsq(delta, h[neighbors] - h[index], rcond=1e-8)[0]
            yield addresses[k], angle, posed[index], frame(normal, direction), bias[k], slopes[k]


def estimate_contact(points, ray_origin, ray_direction, task_direction):
    """Fit the observed plane and intersect it with the public approach line."""
    if len(points) < 25:
        return "point_count", None
    center = points.mean(axis=0)
    residual = points - center
    spectrum, axes = np.linalg.eigh((residual.T @ residual) / len(points))
    thickness, support, _ = np.sqrt(np.maximum(spectrum, 0.0))
    if support < 2.0:
        return "tangent_support", None
    if thickness > 0.7 or spectrum[1] / max(spectrum[0], 1e-12) < 25.0:
        return "plane_fit", None
    normal = axes[:, 0]
    if normal @ ray_direction < 0:
        normal = -normal
    denom = normal @ ray_direction
    if denom < 0.5:
        return "approach_incidence", None
    projected = task_direction - normal * (task_direction @ normal)
    if np.linalg.norm(projected) < 0.25 * np.linalg.norm(task_direction):
        return "tangent_direction", None
    contact = ray_origin + ((center - ray_origin) @ normal / denom) * ray_direction
    return "accepted", (contact, frame(normal, task_direction))


def height(xy, bias, slope, curvature):
    return bias + xy @ slope + 0.5 * curvature * np.sum(xy**2, axis=-1)


def normal_at(xy, slope, curvature):
    return unit(np.r_[-(slope + curvature * xy), 1.0])


def angle_deg(a, b):
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))


def action_error(contact_frame, true_frame, origin, rotation, bias, slope, curvature):
    contact, basis = contact_frame
    true_contact, true_basis = true_frame
    command = contact + basis @ STEP_MM
    target_local = rotation.T @ (command - origin)
    xy = target_local[:2]
    # This is height along the nominal approach axis, not the exact distance to a curved surface.
    gap = abs(target_local[2] - height(xy, bias, slope, curvature))
    normal_error = angle_deg(basis[:, 2], rotation @ normal_at(xy, slope, curvature))
    return {
        "origin_mm": float(np.linalg.norm(contact - true_contact)),
        "contact_normal_deg": angle_deg(basis[:, 2], true_basis[:, 2]),
        "command_from_oracle_mm": float(np.linalg.norm(command - (true_contact + true_basis @ STEP_MM))),
        "command_height_gap_mm": float(gap),
        "command_normal_deg": normal_error,
        "passes_geometry": bool(gap <= 1.0 and normal_error <= 5.0),
    }


def check_equivariance(points, origin, normal, direction, reference, expected, rng, contract):
    transform = Rotation.random(random_state=rng).as_matrix()
    translation = rng.uniform(-1500.0, 1500.0, size=3)
    status, decoded = estimate_contact(
        points @ transform.T + translation, transform @ origin + translation,
        transform @ normal, transform @ direction,
    )
    expected_status, expected_frame = expected
    assert status == expected_status, (status, expected_status)
    contract["gate_checks"] += 1
    # Check both paths. Passing this contract does not imply accurate surface geometry.
    transformed_nominal = frame(transform @ reference[:, 2], transform @ reference[:, 0])
    nominal_error = np.linalg.norm(transformed_nominal - transform @ reference)
    contract["nominal_rotation_fro_max"] = max(contract["nominal_rotation_fro_max"], float(nominal_error))
    expected_nominal_position = transform @ (origin + reference @ STEP_MM) + translation
    nominal_position = transform @ origin + translation + transformed_nominal @ STEP_MM
    contract["nominal_command_mm_max"] = max(
        contract["nominal_command_mm_max"], float(np.linalg.norm(nominal_position - expected_nominal_position)),
    )
    if expected_frame is None:
        return
    contact, basis = decoded
    expected_contact, expected_basis = expected_frame
    command = contact + basis @ STEP_MM
    expected_command = transform @ (expected_contact + expected_basis @ STEP_MM) + translation
    contract["accepted_frames_checked"] += 1
    contract["live_contact_mm_max"] = max(
        contract["live_contact_mm_max"], float(np.linalg.norm(contact - (transform @ expected_contact + translation))),
    )
    contract["live_command_mm_max"] = max(contract["live_command_mm_max"], float(np.linalg.norm(command - expected_command)))
    contract["live_rotation_fro_max"] = max(
        contract["live_rotation_fro_max"], float(np.linalg.norm(basis - transform @ expected_basis)),
    )


def replay(case, origin, basis, bias, slope, rng, transform_rng, contract):
    curvature = 0.01 if case == "curved" else 0.0
    if case == "aligned":
        bias, slope = 0.0, np.zeros(2)
    xy = PATCH.copy()
    if case == "narrow_strip":
        xy[:, 1] *= 0.01
    normal, tangent = basis[:, 2], basis[:, 0]
    true_normal = basis @ normal_at(np.zeros(2), slope, curvature)
    true_basis = frame(true_normal, tangent)
    true_contact = origin + bias * normal
    points = np.column_stack((xy, height(xy, bias, slope, curvature))) @ basis.T + origin
    if case == "mixed_surface":
        indices = rng.choice(len(points), size=20, replace=False)
        points[indices] += 5.0 * true_normal
    points += rng.normal(scale=0.25, size=points.shape)
    task_direction = true_normal + 0.01 * true_basis[:, 0] if case == "near_normal_direction" else tangent

    fit = estimate_contact(points, origin, normal, task_direction)
    check_equivariance(points, origin, normal, task_direction, basis, fit, transform_rng, contract)
    status, estimate = fit
    row = {"gate": status}
    if case in ("aligned", "offset_plane", "curved"):
        true_frame = true_contact, true_basis
        for method, candidate in (("nominal", (origin, basis)), ("live", estimate), ("oracle", true_frame)):
            if candidate is not None:
                row[method] = action_error(candidate, true_frame, origin, basis, bias, slope, curvature)
    return row


def summarize(rows):
    count = len(rows)
    statuses = sorted({r["gate"] for r in rows})
    result = {"replays": count, "gate": {s: sum(r["gate"] == s for r in rows) for s in statuses}}
    for method in ("nominal", "live", "oracle"):
        available = [r[method] for r in rows if method in r]
        if not available:
            continue
        passes = sum(x["passes_geometry"] for x in available)
        result[method] = {"issued": len(available), "passes_all_replays": passes, "pass_rate": passes / count}
        for key in available[0]:
            if key != "passes_geometry":
                values = [x[key] for x in available]
                result[method][key] = {
                    "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
                    "max": max(values),
                }
    return result


def main():
    rng = np.random.default_rng(SEED)
    transform_rng = np.random.default_rng(SEED + 1)
    cases = ("aligned", "offset_plane", "curved", "narrow_strip", "mixed_surface", "near_normal_direction")
    records = []
    contract = {
        "gate_checks": 0, "accepted_frames_checked": 0, "nominal_rotation_fro_max": 0.0,
        "nominal_command_mm_max": 0.0, "live_contact_mm_max": 0.0, "live_command_mm_max": 0.0,
        "live_rotation_fro_max": 0.0,
    }
    for side in ("left", "right"):
        for address, pose, origin, basis, bias, slope in skin_anchors(side, rng):
            records.append({
                "address": address, "knee_pose_deg": pose, "offset_mm": float(bias), "slope": slope.tolist(),
                "cases": {case: replay(case, origin, basis, bias, slope, rng, transform_rng, contract) for case in cases},
            })
    assert contract["live_contact_mm_max"] <= POSITION_CONTRACT_MM
    assert contract["live_command_mm_max"] <= POSITION_CONTRACT_MM
    assert contract["nominal_command_mm_max"] <= POSITION_CONTRACT_MM
    assert contract["live_rotation_fro_max"] <= ROTATION_CONTRACT
    assert contract["nominal_rotation_fro_max"] <= ROTATION_CONTRACT
    summary = {case: summarize([r["cases"][case] for r in records]) for case in cases}
    files = sorted(list(ASSETS.glob("atlas_*.npz")) + list(ASSETS.glob("*_pose_*.npz")))
    payload = {
        "scope": "Production skin anchors with synthetic contact patches and noisy synthetic points",
        "seed": SEED, "unique_material_addresses": 64, "poses_per_address": len(POSES),
        "position_contract_mm": POSITION_CONTRACT_MM, "rotation_contract_fro": ROTATION_CONTRACT,
        "assets_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "equivariance": contract, "summary": summary, "records": records,
    }
    (HERE / "GEOMETRY_RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"equivariance": contract, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

