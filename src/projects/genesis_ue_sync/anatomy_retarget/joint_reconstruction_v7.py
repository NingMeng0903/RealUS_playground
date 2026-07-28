"""Offline V7 reconstruction from immutable V71 joint material domains.

The functions here are used while baking beta response samples, never in the
pose hot path.  Every probe is a fixed V71 vertex id.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import with_source_driver_coupling
from .joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
    fit_sphere_fixed_radius_v7,
    fit_sphere_v7,
)
from .rigged_asset import AnatomyRiggedAsset


_SIDES = ("left", "right")
_SIDE_SUFFIX = {"left": "L", "right": "R"}
_SMPLX_HIP = {"left": 1, "right": 2}
_SMPLX_KNEE = {"left": 4, "right": 5}
_SMPLX_ANKLE = {"left": 7, "right": 8}
_PATELLA_GAIN = {"left": 0.2275, "right": 0.2107}
_LEG_HINGE_BLEND_LO_DEG = 5.0
_LEG_HINGE_BLEND_HI_DEG = 15.0


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bone_index(asset: AnatomyRiggedAsset, name: str) -> int:
    try:
        return list(asset.source_bone_names or []).index(name)
    except ValueError as exc:
        raise ValueError(f"V7 reconstruction requires source bone {name!r}") from exc


def _mesh_index(asset: AnatomyRiggedAsset, name: str) -> int:
    try:
        return list(asset.source_mesh_names).index(name)
    except ValueError as exc:
        raise ValueError(f"V7 reconstruction requires source mesh {name!r}") from exc


def _mesh_vertices(asset: AnatomyRiggedAsset, name: str) -> np.ndarray:
    if asset.source_vertex_ranges is None:
        raise ValueError("V7 reconstruction requires source_vertex_ranges")
    start, stop = np.asarray(
        asset.source_vertex_ranges[_mesh_index(asset, name)], dtype=np.int64
    )
    return np.arange(int(start), int(stop), dtype=np.int64)


def _nearest_distances(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(target).query(points, k=1)[0], dtype=np.float64)
    except Exception:
        squared = np.sum(
            (points[:, None, :] - target[None, :, :]) ** 2, axis=2
        )
        return np.sqrt(np.min(squared, axis=1))


def _optimize_knee_hinge_pivot_v7(
    *,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    side: str,
    hinge_world: np.ndarray,
    centroid_pivot: np.ndarray,
    sweep_maximum_deg: float = 120.0,
    sample_count: int = 13,
    minimum_gap_m: float = 0.0002,
    search_radius_m: float = 0.020,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place the hinge line so both tibiofemoral compartments stay in contact.

    A femoral condyle is not a circular arc, so a single-axis hinge through an
    arbitrary point makes one compartment lift as the knee flexes: through the
    domain centroid the medial gap reached 4.0 mm at 100 degrees, outside the
    3 mm contact corridor.  The pivot is a free parameter of the authored hinge
    (only the axis direction is inherited from V71), so it is solved here once,
    offline, against the frozen domains over the authorized flexion range.
    Only the two components perpendicular to the axis matter: sliding the point
    along the axis leaves the hinge line, and therefore the motion, unchanged.
    """
    axis = np.asarray(hinge_world, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    basis_seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, basis_seed)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second = np.cross(axis, first)
    base = np.asarray(centroid_pivot, dtype=np.float64).reshape(3)
    compartments = [
        (
            vertices[domains.require(f"{side}/tibial_plateau_{name}")] - base,
            vertices[domains.require(f"{side}/femoral_condyle_{name}")],
        )
        for name in ("medial", "lateral")
    ]
    angles = np.radians(
        np.linspace(0.0, float(sweep_maximum_deg), int(sample_count), dtype=np.float64)
    )
    rotations = []
    for angle in angles:
        cross = np.asarray(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            ),
            dtype=np.float64,
        )
        rotations.append(
            np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        )

    def sweep_extrema(offset: np.ndarray) -> tuple[float, float]:
        worst = 0.0
        tightest = float("inf")
        for rotation in rotations:
            sample = 0.0
            for plateau, condyle in compartments:
                swept = (rotation @ (plateau - offset).T).T + offset + base
                sample = max(sample, float(np.min(_nearest_distances(swept, condyle))))
            worst = max(worst, sample)
            tightest = min(tightest, sample)
        return worst, tightest

    best_offset = np.zeros(3, dtype=np.float64)
    best_worst, best_tightest = sweep_extrema(best_offset)
    centroid_worst = best_worst
    coordinates = np.zeros(2, dtype=np.float64)
    radius = float(search_radius_m)
    while radius >= 0.0005:
        grid = np.linspace(-radius, radius, 9, dtype=np.float64)
        for first_step in grid + coordinates[0]:
            for second_step in grid + coordinates[1]:
                offset = first_step * first + second_step * second
                worst, tightest = sweep_extrema(offset)
                # A smaller maximum gap is worthless if the surfaces start
                # crossing, so a candidate that closes the corridor below the
                # floor is rejected outright rather than scored.
                if tightest < float(minimum_gap_m) or worst >= best_worst:
                    continue
                best_worst, best_tightest = worst, tightest
                best_offset = offset
                coordinates = np.asarray([first_step, second_step], dtype=np.float64)
        radius /= 4.0
    report = {
        "source": "hinge_line_contact_optimized",
        "centroid_worst_gap_m": centroid_worst,
        "optimized_worst_gap_m": best_worst,
        "optimized_tightest_gap_m": best_tightest,
        "pivot_shift_m": float(np.linalg.norm(best_offset)),
        "sample_count": int(sample_count),
        "sweep_maximum_deg": float(sweep_maximum_deg),
    }
    return base + best_offset, report


def _contact_band_translation(
    *,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    side: str,
    target_gap_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the minimum proximal-tibia translation for both condyles."""

    def measurements(translation: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for compartment in ("medial", "lateral"):
            plateau = domains.require(f"{side}/tibial_plateau_{compartment}")
            condyle = domains.require(f"{side}/femoral_condyle_{compartment}")
            distance = _nearest_distances(
                vertices[plateau] + translation[None, :],
                vertices[condyle],
            )
            values.append(float(np.quantile(distance, 0.05)))
        return np.asarray(values, dtype=np.float64)

    try:
        from scipy.optimize import least_squares

        solved = least_squares(
            lambda translation: np.concatenate(
                (
                    (measurements(translation) - float(target_gap_m)) / 0.001,
                    np.asarray(translation, dtype=np.float64) / 0.02 * 0.04,
                )
            ),
            np.zeros(3, dtype=np.float64),
            bounds=(-0.02, 0.02),
            max_nfev=256,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
        translation = np.asarray(solved.x, dtype=np.float64)
    except Exception:
        shifts = []
        for compartment in ("medial", "lateral"):
            plateau = domains.require(f"{side}/tibial_plateau_{compartment}")
            condyle = domains.require(f"{side}/femoral_condyle_{compartment}")
            shifts.append(
                np.mean(vertices[condyle], axis=0)
                - np.mean(vertices[plateau], axis=0)
            )
        translation = np.clip(
            0.25 * np.mean(np.stack(shifts), axis=0), -0.02, 0.02
        )
    return translation, {
        "translation_m": translation.tolist(),
        "q05_before_m": measurements(np.zeros(3)).tolist(),
        "q05_after_m": measurements(translation).tolist(),
        "target_gap_m": float(target_gap_m),
    }


def _force_rigid_mesh_driver(
    asset: AnatomyRiggedAsset,
    *,
    mesh_name: str,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
) -> None:
    mesh_index = _mesh_index(asset, mesh_name)
    if asset.source_mesh_controller_bones is None:
        raise ValueError("V7 reconstruction requires explicit mesh controllers")
    controller = int(asset.source_mesh_controller_bones[mesh_index])
    indices = _mesh_vertices(asset, mesh_name)
    driver_indices[indices] = controller
    driver_weights[indices] = 0.0
    driver_weights[indices, 0] = 1.0


def _socket_neighbourhood_displacement(
    *,
    faces: np.ndarray,
    mesh_indices: np.ndarray,
    core_indices: np.ndarray,
    core_displacement: np.ndarray,
    ring_count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse a fixed socket correction over a small immutable pelvis ring."""
    mesh = np.asarray(mesh_indices, dtype=np.int64)
    core = np.asarray(core_indices, dtype=np.int64)
    local_by_global = np.full(int(np.max(mesh)) + 1, -1, dtype=np.int64)
    local_by_global[mesh] = np.arange(len(mesh), dtype=np.int64)
    triangles = np.asarray(faces, dtype=np.int64)
    triangles = triangles[np.all(np.isin(triangles, mesh), axis=1)]
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges = local_by_global[edges]
    adjacency: list[set[int]] = [set() for _ in range(len(mesh))]
    for first, second in edges.tolist():
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    displacement = np.zeros((len(mesh), 3), dtype=np.float64)
    assigned = np.zeros(len(mesh), dtype=bool)
    core_local = local_by_global[core]
    displacement[core_local] = np.asarray(core_displacement, dtype=np.float64)
    assigned[core_local] = True
    frontier = set(int(value) for value in core_local.tolist())
    for ring in range(1, int(ring_count) + 1):
        next_frontier: set[int] = set()
        for vertex in frontier:
            next_frontier.update(
                neighbour
                for neighbour in adjacency[vertex]
                if not assigned[neighbour]
            )
        if not next_frontier:
            break
        weight = float((ring_count + 1 - ring) / (ring_count + 1))
        for vertex in next_frontier:
            previous = [
                neighbour
                for neighbour in adjacency[vertex]
                if assigned[neighbour]
            ]
            if previous:
                displacement[vertex] = (
                    weight * np.mean(displacement[previous], axis=0)
                )
        assigned[list(next_frontier)] = True
        frontier = next_frontier
    return mesh[assigned], displacement[assigned]


def _restore_socket_template_v7(
    *,
    asset: AnatomyRiggedAsset,
    source_socket_points: np.ndarray,
    source_head_radius_m: float,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    side: str,
) -> dict[str, Any]:
    """Restore V71 socket shape at the current beta/head scale and pelvis pose."""
    suffix = _SIDE_SUFFIX[side]
    head = domains.require(f"{side}/femoral_head")
    socket = domains.require(f"{side}/acetabulum")
    source_points = np.asarray(source_socket_points, dtype=np.float64)
    if source_points.shape != (len(socket), 3) or not np.all(
        np.isfinite(source_points)
    ):
        raise ValueError(f"{side} V71 socket template has an invalid shape")
    source_radius = float(source_head_radius_m)
    if not np.isfinite(source_radius) or source_radius <= 0.0:
        raise ValueError(f"{side} V71 femoral-head radius is invalid")
    source_socket = fit_sphere_fixed_radius_v7(
        source_points, radius_m=source_radius
    )
    subject_head = fit_sphere_v7(vertices[head])
    subject_socket = fit_sphere_fixed_radius_v7(
        vertices[socket], radius_m=float(subject_head["radius_m"])
    )
    if not all(
        item["available"]
        for item in (source_socket, subject_head, subject_socket)
    ):
        raise ValueError(f"{side} source socket template sphere fit failed")
    source_centered = (
        source_points
        - np.asarray(source_socket["center"], dtype=np.float64)
    )
    subject_centered = (
        vertices[socket]
        - np.asarray(subject_socket["center"], dtype=np.float64)
    )
    u, _singular, vt = np.linalg.svd(source_centered.T @ subject_centered)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    radius_scale = float(subject_head["radius_m"]) / max(source_radius, 1.0e-12)
    target = (
        np.asarray(subject_socket["center"], dtype=np.float64)
        + radius_scale * source_centered @ rotation.T
    )
    core_displacement = target - vertices[socket]
    ilium = _mesh_vertices(asset, f"Ilium_{suffix}")
    affected, displacement = _socket_neighbourhood_displacement(
        faces=asset.faces,
        mesh_indices=ilium,
        core_indices=socket,
        core_displacement=core_displacement,
    )
    vertices[affected] += displacement
    final_socket = fit_sphere_fixed_radius_v7(
        vertices[socket], radius_m=float(subject_head["radius_m"])
    )
    return {
        "source": "true_v71_fixed_material_socket",
        "radius_scale": radius_scale,
        "core_vertex_count": int(len(socket)),
        "transition_vertex_count": int(len(affected) - len(socket)),
        "maximum_displacement_m": float(
            np.max(np.linalg.norm(displacement, axis=1))
        ),
        "rms_displacement_m": float(
            np.sqrt(np.mean(np.sum(displacement * displacement, axis=1)))
        ),
        "sphere_residual_before_m": float(subject_socket["rms_residual_m"]),
        "sphere_residual_after_m": float(final_socket["rms_residual_m"]),
        "whole_pelvis_scaled": False,
    }


def _apply_frozen_patella_law_v7(
    *,
    asset: AnatomyRiggedAsset,
    law: Any,
    domains: FrozenJointMaterialDomainsV7,
    side: str,
    vertices: np.ndarray,
    bind_global: np.ndarray,
    bone_head: np.ndarray,
    bone_tail: np.ndarray,
    corrective_gain: np.ndarray,
    patella_bone: int,
    patella_responses: dict[str, Any],
    hinge_local: np.ndarray,
    knot_degrees: np.ndarray,
) -> dict[str, Any]:
    """Bake the frozen V71 patella response instead of a candidate-fitted spline.

    The patella keeps its authored parent-local translation and receives a
    rotation-only driver in Tibia_Bone space, exactly as the V71 Action does.
    Only the residual trochlear-corridor translation is solved here, from
    frozen material domains, and it is bounded by the frozen law.
    """
    from .patella_oracle_v7 import (
        solve_patella_contact_corrections_v7,
    )

    patella = domains.require(f"{side}/patella")
    femur_surface = domains.require(f"{side}/femur")
    rest_gap = float(
        np.min(_nearest_distances(vertices[patella], vertices[femur_surface]))
    )
    target = float(law.corridor_target_m)
    lower = float(law.corridor_min_m) + 0.0002
    upper = float(law.corridor_max_m) - 0.001
    rest_translation = np.zeros(3, dtype=np.float64)
    if not (lower <= rest_gap <= upper):
        distances = _nearest_distances(vertices[patella], vertices[femur_surface])
        nearest = int(np.argmin(distances))
        femur_distances = _nearest_distances(
            vertices[femur_surface], vertices[patella][nearest : nearest + 1]
        )
        contact = vertices[femur_surface][int(np.argmin(femur_distances))]
        direction = contact - vertices[patella][nearest]
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 1.0e-9:
            rest_translation = direction * (
                (magnitude - target) / magnitude
            )
            bound = float(law.max_contact_translation_m)
            norm = float(np.linalg.norm(rest_translation))
            if norm > bound:
                rest_translation *= bound / norm
            vertices[patella] += rest_translation
            bind_global[patella_bone, :3, 3] += rest_translation
            bone_head[patella_bone] += rest_translation
            bone_tail[patella_bone] += rest_translation
    # The old femur-referenced corrective gain is retired by this path.
    corrective_gain[patella_bone] = 0.0
    translations, contact_report = solve_patella_contact_corrections_v7(
        law,
        vertices=vertices,
        faces=asset.faces,
        domains=domains,
        asset=replace(asset, target_rest_global=bind_global.astype(np.float32)),
        side=side,
        knots_deg=np.asarray(knot_degrees, dtype=np.float64),
        knee_axis_local=np.asarray(hinge_local, dtype=np.float64),
    )
    response_deg = np.degrees(
        np.asarray(
            [
                law.response_rad(side, float(np.radians(knot)))
                for knot in np.asarray(knot_degrees, dtype=np.float64)
            ],
            dtype=np.float64,
        )
    )
    patella_responses[str(patella_bone)] = {
        "side": side,
        "smplx_joint": int(_SMPLX_KNEE[side]),
        "axis_local": np.asarray(
            law.axis_patella_local[side], dtype=np.float64
        ).tolist(),
        "knots_deg": np.asarray(knot_degrees, dtype=np.float64).tolist(),
        "response_deg": response_deg.tolist(),
        "translation_parent_local_m": np.asarray(translations).tolist(),
        "maximum_translation_m": float(law.max_contact_translation_m),
        "oracle_digest": str(law.content_digest()),
    }
    return {
        "source": "frozen_v71_patella_oracle",
        "oracle_digest": str(law.content_digest()),
        "response_slope": float(law.response_slope[side]),
        "rest_gap_before_m": rest_gap,
        "rest_translation_m": rest_translation.tolist(),
        "contact": contact_report,
    }


def reconstruct_articular_subject_v7(
    asset: AnatomyRiggedAsset,
    *,
    domains: FrozenJointMaterialDomainsV7,
    source_reference: AnatomyRiggedAsset | None = None,
    source_socket_templates: Mapping[str, np.ndarray] | None = None,
    knee_target_gap_m: float = 0.0010,
    patella_target_gap_m: float = 0.002,
    patella_law: Any | None = None,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Correct hip/knee/patella rest geometry without scaling articular ends."""
    asset.validate()
    domains.validate_topology(asset.vertices_rest, asset.faces)
    if (
        asset.source_bone_names is None
        or asset.target_bind_global is None
        or asset.target_bind_local is None
        or asset.runtime_inverse_bind is None
        or asset.driver_indices is None
        or asset.driver_weights is None
    ):
        raise ValueError("V7 reconstruction requires a schema-v6 source rig")

    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    bind_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    bone_head = np.asarray(
        asset.target_bone_head
        if asset.target_bone_head is not None
        else asset.source_bone_head,
        dtype=np.float64,
    ).copy()
    bone_tail = np.asarray(
        asset.target_bone_tail
        if asset.target_bone_tail is not None
        else asset.source_bone_tail,
        dtype=np.float64,
    ).copy()
    driver_rest = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else asset.rest_joints,
        dtype=np.float64,
    ).copy()
    driver_indices = np.asarray(asset.driver_indices, dtype=np.int32).copy()
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64).copy()
    corrective_gain = np.asarray(
        asset.source_bone_corrective_gain
        if asset.source_bone_corrective_gain is not None
        else np.zeros(len(asset.source_bone_names), dtype=np.float32),
        dtype=np.float64,
    ).copy()
    driver_types = list(asset.source_bone_driver_types or [])
    if len(driver_types) != len(asset.source_bone_names):
        raise ValueError("V7 reconstruction requires explicit source driver modes")

    side_reports: dict[str, Any] = {}
    local_fk: set[int] = set()
    knee_hinges: dict[str, Any] = {}
    tibia_glides: dict[str, Any] = {}
    patella_splines: dict[str, Any] = {}
    patella_responses: dict[str, Any] = {}
    patella_reports: dict[str, Any] = {}
    socket_reports: dict[str, Any] = {}
    leg_hinge_solves: dict[str, Any] = {}
    for side in _SIDES:
        suffix = _SIDE_SUFFIX[side]
        femur_bone = _bone_index(asset, f"Femur_Rot_{suffix}")
        knee_bone = _bone_index(asset, f"Knee_Rotate_{suffix}")
        tibia_bone = _bone_index(asset, f"Tibia_Bone_{suffix}")
        tibia_twist = _bone_index(asset, f"Tibia_Twist_{suffix}")
        patella_bone = _bone_index(asset, f"Patella_Rotate_{suffix}")

        if source_reference is not None:
            if (
                len(source_reference.vertices_rest) != len(asset.vertices_rest)
                or not np.array_equal(source_reference.faces, asset.faces)
                or list(source_reference.source_mesh_names)
                != list(asset.source_mesh_names)
            ):
                raise ValueError(
                    "V7 source socket template topology does not match subject"
                )
            reference_head = fit_sphere_v7(
                np.asarray(source_reference.vertices_rest)[
                    domains.require(f"{side}/femoral_head")
                ]
            )
            if not reference_head["available"]:
                raise ValueError(f"{side} V71 femoral-head template fit failed")
            socket_reports[side] = _restore_socket_template_v7(
                asset=asset,
                source_socket_points=np.asarray(source_reference.vertices_rest)[
                    domains.require(f"{side}/acetabulum")
                ],
                source_head_radius_m=float(reference_head["radius_m"]),
                vertices=vertices,
                domains=domains,
                side=side,
            )
        elif source_socket_templates is not None:
            points_key = f"{side}/socket_points_m"
            radius_key = f"{side}/femoral_head_radius_m"
            if points_key not in source_socket_templates or radius_key not in source_socket_templates:
                raise ValueError(f"{side} V71 socket template coefficients are missing")
            socket_reports[side] = _restore_socket_template_v7(
                asset=asset,
                source_socket_points=source_socket_templates[points_key],
                source_head_radius_m=float(
                    np.asarray(source_socket_templates[radius_key]).reshape(-1)[0]
                ),
                vertices=vertices,
                domains=domains,
                side=side,
            )
        else:
            socket_reports[side] = {
                "source": "subject_existing_socket",
                "available": False,
            }

        head_indices = domains.require(f"{side}/femoral_head")
        socket_indices = domains.require(f"{side}/acetabulum")
        head_fit = fit_sphere_v7(vertices[head_indices])
        if not head_fit["available"]:
            raise ValueError(f"{side} femoral-head sphere fit failed")
        socket_fit = fit_sphere_fixed_radius_v7(
            vertices[socket_indices], radius_m=float(head_fit["radius_m"])
        )
        if not socket_fit["available"]:
            raise ValueError(f"{side} acetabulum sphere fit failed")
        head_center = np.asarray(head_fit["center"], dtype=np.float64)
        socket_center = np.asarray(socket_fit["center"], dtype=np.float64)
        hip_delta = socket_center - head_center

        femur = domains.require(f"{side}/femur")
        condyles = np.concatenate(
            (
                domains.require(f"{side}/femoral_condyle_medial"),
                domains.require(f"{side}/femoral_condyle_lateral"),
            )
        )
        distal_center = np.mean(vertices[condyles], axis=0)
        shaft = distal_center - head_center
        shaft_length = float(np.linalg.norm(shaft))
        if shaft_length <= 0.1:
            raise ValueError(f"{side} femur shaft is degenerate")
        shaft_axis = shaft / shaft_length
        station = ((vertices[femur] - head_center) @ shaft_axis) / shaft_length
        proximal_weight = 1.0 - _smoothstep((station - 0.18) / 0.62)
        vertices[femur] += proximal_weight[:, None] * hip_delta[None, :]

        bind_global[femur_bone, :3, 3] = socket_center
        bone_head[femur_bone] = socket_center
        driver_rest[_SMPLX_HIP[side]] = socket_center

        tibia = domains.require(f"{side}/tibia")
        plateau = np.concatenate(
            (
                domains.require(f"{side}/tibial_plateau_medial"),
                domains.require(f"{side}/tibial_plateau_lateral"),
            )
        )
        knee_translation, knee_report = _contact_band_translation(
            vertices=vertices,
            domains=domains,
            side=side,
            target_gap_m=float(knee_target_gap_m),
        )
        plateau_center = np.mean(vertices[plateau], axis=0)
        tibia_points = vertices[tibia]
        distal = tibia_points[
            int(np.argmax(np.linalg.norm(tibia_points - plateau_center, axis=1)))
        ]
        tibia_axis = distal - plateau_center
        tibia_length = float(np.linalg.norm(tibia_axis))
        if tibia_length <= 0.1:
            raise ValueError(f"{side} tibia shaft is degenerate")
        tibia_axis /= tibia_length
        tibia_station = (
            (vertices[tibia] - plateau_center) @ tibia_axis
        ) / tibia_length
        plateau_weight = 1.0 - _smoothstep((tibia_station - 0.12) / 0.58)
        vertices[tibia] += plateau_weight[:, None] * knee_translation[None, :]
        centroid_pivot = np.mean(
            np.concatenate(
                (
                    vertices[condyles],
                    vertices[plateau],
                ),
                axis=0,
            ),
            axis=0,
        )
        medial_center = np.mean(
            vertices[domains.require(f"{side}/femoral_condyle_medial")],
            axis=0,
        )
        lateral_center = np.mean(
            vertices[domains.require(f"{side}/femoral_condyle_lateral")],
            axis=0,
        )
        epicondylar_world = lateral_center - medial_center
        epicondylar_world /= max(float(np.linalg.norm(epicondylar_world)), 1.0e-12)
        if patella_law is not None:
            # The condyle-centroid axis is a geometric estimate that sits about
            # 19 degrees away from the hinge the V71 animator actually keyed, and
            # flexing about it drove the patella out of the trochlear corridor.
            hinge_local = np.asarray(
                patella_law.axis_knee_local[side], dtype=np.float64
            ).reshape(3)
            hinge_local /= max(float(np.linalg.norm(hinge_local)), 1.0e-12)
            hinge_world = bind_global[knee_bone, :3, :3] @ hinge_local
        else:
            hinge_world = epicondylar_world
            hinge_local = bind_global[knee_bone, :3, :3].T @ hinge_world
            hinge_local /= max(float(np.linalg.norm(hinge_local)), 1.0e-12)
        knee_pivot, pivot_report = _optimize_knee_hinge_pivot_v7(
            vertices=vertices,
            domains=domains,
            side=side,
            hinge_world=hinge_world,
            centroid_pivot=centroid_pivot,
        )
        bind_global[knee_bone, :3, 3] = knee_pivot
        bone_tail[femur_bone] = knee_pivot
        bone_head[knee_bone] = knee_pivot
        # Five-degree samples are still tiny (<1 KiB for both knees) and avoid
        # interpolating across a narrow condyle contact transition.
        knot_degrees = np.linspace(0.0, 120.0, 25, dtype=np.float64)
        # Keep the authored Femur->Knee_Rotate translation exact.  An earlier
        # prototype put a noisy nearest-contact translation (up to 12.7 mm)
        # on the hinge itself.  That made the contact diagnostic pass by
        # physically disconnecting the knee pivot from the femur.  The compact
        # V7 hinge therefore contains rotation only; the 1 mm rest corridor
        # supplies enough margin for the accepted rolling contact sweep.
        translation_local = np.zeros((len(knot_degrees), 3), dtype=np.float64)
        knee_hinges[str(knee_bone)] = {
            "side": side,
            "smplx_joint": int(_SMPLX_KNEE[side]),
            "axis_local": hinge_local.tolist(),
            "input_mode": "axis_angle_norm",
            "knots_deg": knot_degrees.tolist(),
            "response_deg": knot_degrees.tolist(),
            "translation_local_m": translation_local.tolist(),
            "pivot_translation_locked": True,
        }
        ankle_bone = _bone_index(asset, f"Ankle_Rot_{suffix}")
        # Orient the hinge so a positive rotation is anatomical flexion: the
        # shank must swing posteriorly, away from the patella.
        shank_bind = bind_global[ankle_bone, :3, 3] - knee_pivot
        shank_bind = shank_bind / max(float(np.linalg.norm(shank_bind)), 1.0e-12)
        anterior = np.mean(vertices[domains.require(f"{side}/patella")], axis=0) - knee_pivot
        anterior = anterior - float(anterior @ shank_bind) * shank_bind
        anterior_norm = float(np.linalg.norm(anterior))
        if anterior_norm <= 1.0e-9:
            raise ValueError(f"{side} patella gives no anterior reference for the knee hinge")
        anterior /= anterior_norm
        hinge_axis_sign = (
            -1 if float(np.cross(hinge_world, shank_bind) @ anterior) > 0.0 else 1
        )
        if hinge_axis_sign < 0:
            hinge_world = -hinge_world
            hinge_local = -hinge_local
            knee_hinges[str(knee_bone)]["axis_local"] = hinge_local.tolist()
        hinge_femur_local = bind_global[femur_bone, :3, :3].T @ hinge_world
        hinge_femur_local = hinge_femur_local / max(
            float(np.linalg.norm(hinge_femur_local)), 1.0e-12
        )
        # The proximal shift above seated the femoral head on the acetabulum;
        # refit it so the runtime rotates the femur about the seated centre
        # rather than about whatever the bone origin happens to be.
        seated_head_fit = fit_sphere_v7(vertices[head_indices])
        if not seated_head_fit["available"]:
            raise ValueError(f"{side} seated femoral-head sphere fit failed")
        head_femur_local = bind_global[femur_bone, :3, :3].T @ (
            np.asarray(seated_head_fit["center"], dtype=np.float64)
            - bind_global[femur_bone, :3, 3]
        )
        leg_hinge_solves[side] = {
            "femur_bone": int(femur_bone),
            "knee_bone": int(knee_bone),
            "ankle_bone": int(ankle_bone),
            "smplx_hip": int(_SMPLX_HIP[side]),
            "smplx_knee": int(_SMPLX_KNEE[side]),
            "smplx_ankle": int(_SMPLX_ANKLE[side]),
            "hinge_axis_femur_local": hinge_femur_local.tolist(),
            "femoral_head_femur_local": head_femur_local.tolist(),
            "femoral_head_vertex_indices": np.asarray(
                head_indices, dtype=np.int64
            ).tolist(),
            "hinge_axis_sign": int(hinge_axis_sign),
            "blend_lo_deg": float(_LEG_HINGE_BLEND_LO_DEG),
            "blend_hi_deg": float(_LEG_HINGE_BLEND_HI_DEG),
        }
        glide_local = [np.zeros(3, dtype=np.float64)]
        for angle_deg in knot_degrees[1:]:
            angle = float(np.radians(angle_deg))
            cross = np.asarray(
                (
                    (0.0, -hinge_world[2], hinge_world[1]),
                    (hinge_world[2], 0.0, -hinge_world[0]),
                    (-hinge_world[1], hinge_world[0], 0.0),
                ),
                dtype=np.float64,
            )
            rotation = (
                np.eye(3)
                + np.sin(angle) * cross
                + (1.0 - np.cos(angle)) * (cross @ cross)
            )
            swept = vertices.copy()
            swept[plateau] = (
                (rotation @ (vertices[plateau] - knee_pivot).T).T
                + knee_pivot
            )
            correction_world, _sweep_report = _contact_band_translation(
                vertices=swept,
                domains=domains,
                side=side,
                target_gap_m=float(knee_target_gap_m),
            )
            magnitude = float(np.linalg.norm(correction_world))
            if magnitude > 0.001:
                correction_world *= 0.001 / magnitude
            posed_knee_rotation = (
                rotation @ bind_global[knee_bone, :3, :3]
            )
            glide_local.append(
                posed_knee_rotation.T @ correction_world
            )
        glide_local_array = np.asarray(glide_local, dtype=np.float64)
        # Remove nearest-surface knot jitter without enlarging the strict
        # 0.5 mm local-FK translation budget.
        if len(glide_local_array) > 2:
            padded = np.pad(glide_local_array, ((1, 1), (0, 0)), mode="edge")
            glide_local_array = (
                padded[:-2] + 2.0 * padded[1:-1] + padded[2:]
            ) / 4.0
            glide_local_array[0] = 0.0
            norm = np.linalg.norm(glide_local_array, axis=1)
            active = norm > 0.001
            glide_local_array[active] *= (
                0.001 / norm[active]
            )[:, None]
        tibia_glides[str(tibia_bone)] = {
            "side": side,
            "smplx_joint": int(_SMPLX_KNEE[side]),
            "knots_deg": knot_degrees.tolist(),
            "translation_parent_local_m": glide_local_array.tolist(),
            "maximum_translation_m": 0.001,
        }

        patella = domains.require(f"{side}/patella")
        patella_articular = domains.require(f"{side}/patella_articular")
        trochlea = domains.require(f"{side}/trochlea")
        if patella_law is not None:
            patella_reports[side] = _apply_frozen_patella_law_v7(
                asset=asset,
                law=patella_law,
                domains=domains,
                side=side,
                vertices=vertices,
                bind_global=bind_global,
                bone_head=bone_head,
                bone_tail=bone_tail,
                corrective_gain=corrective_gain,
                patella_bone=patella_bone,
                patella_responses=patella_responses,
                hinge_local=hinge_local,
                knot_degrees=knot_degrees,
            )
            for mesh_name in (
                f"Femur_{suffix}",
                f"Tibia_{suffix}",
                f"Patella_{suffix}",
                f"Humerus_{suffix}",
                f"Ulna_{suffix}",
                f"Radius_{suffix}",
            ):
                _force_rigid_mesh_driver(
                    asset,
                    mesh_name=mesh_name,
                    driver_indices=driver_indices,
                    driver_weights=driver_weights,
                )
            local_fk.update((femur_bone, knee_bone))
            driver_types[tibia_bone] = "bind_follow"
            driver_types[tibia_twist] = "bind_follow"
            driver_rest[_SMPLX_KNEE[side]] = knee_pivot
            final_head = fit_sphere_v7(vertices[head_indices])
            final_socket = fit_sphere_fixed_radius_v7(
                vertices[socket_indices], radius_m=float(final_head["radius_m"])
            )
            side_reports[side] = {
                "hip_center_before_m": float(np.linalg.norm(hip_delta)),
                "hip_center_after_m": float(
                    np.linalg.norm(
                        np.asarray(final_head["center"])
                        - np.asarray(final_socket["center"])
                    )
                ),
                "femoral_head_radius_change_m": abs(
                    float(final_head["radius_m"]) - float(head_fit["radius_m"])
                ),
                "knee": knee_report,
                "knee_pivot_m": knee_pivot.tolist(),
                "knee_pivot_optimization": pivot_report,
                "knee_hinge_source": "v71_authored_local_axis",
                "knee_epicondylar_axis_error_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                abs(float(hinge_world @ epicondylar_world)),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
                "patella": patella_reports[side],
            }
            continue
        distance_matrix = np.linalg.norm(
            vertices[patella_articular, None, :]
            - vertices[trochlea][None, :, :],
            axis=2,
        )
        flat_index = int(np.argmin(distance_matrix))
        patella_index, trochlea_index = np.unravel_index(
            flat_index, distance_matrix.shape
        )
        direction = (
            vertices[trochlea[trochlea_index]]
            - vertices[patella_articular[patella_index]]
        )
        distance = float(np.linalg.norm(direction))
        patella_translation = (
            direction * max(0.0, distance - float(patella_target_gap_m))
            / max(distance, 1.0e-12)
        )
        vertices[patella] += patella_translation
        bind_global[patella_bone, :3, 3] += patella_translation
        bone_head[patella_bone] += patella_translation
        bone_tail[patella_bone] += patella_translation
        corrective_gain[patella_bone] = _PATELLA_GAIN[side]
        patella_translation_local = [np.zeros(3, dtype=np.float64)]
        for angle_deg in knot_degrees[1:]:
            angle = float(np.radians(angle_deg) * _PATELLA_GAIN[side])
            cross = np.asarray(
                (
                    (0.0, -hinge_world[2], hinge_world[1]),
                    (hinge_world[2], 0.0, -hinge_world[0]),
                    (-hinge_world[1], hinge_world[0], 0.0),
                ),
                dtype=np.float64,
            )
            rotation = (
                np.eye(3)
                + np.sin(angle) * cross
                + (1.0 - np.cos(angle)) * (cross @ cross)
            )
            swept_articular = (
                (
                    rotation
                    @ (vertices[patella_articular] - knee_pivot).T
                ).T
                + knee_pivot
            )
            distance_matrix = np.linalg.norm(
                swept_articular[:, None, :]
                - vertices[trochlea][None, :, :],
                axis=2,
            )
            flat_index = int(np.argmin(distance_matrix))
            articular_index, trochlea_index = np.unravel_index(
                flat_index, distance_matrix.shape
            )
            direction = (
                vertices[trochlea[trochlea_index]]
                - swept_articular[articular_index]
            )
            distance = float(np.linalg.norm(direction))
            correction_world = (
                direction
                * (distance - float(patella_target_gap_m))
                / max(distance, 1.0e-12)
            )
            patella_translation_local.append(
                bind_global[femur_bone, :3, :3].T @ correction_world
            )
        pivot_local_h = np.concatenate((knee_pivot, np.ones(1)))
        pivot_reference = (
            np.linalg.inv(bind_global[femur_bone]) @ pivot_local_h
        )[:3]
        patella_splines[str(patella_bone)] = {
            "side": side,
            "reference_bone": int(femur_bone),
            "smplx_joint": int(_SMPLX_KNEE[side]),
            "axis_reference_local": (
                bind_global[femur_bone, :3, :3].T @ hinge_world
            ).tolist(),
            "pivot_reference_local_m": pivot_reference.tolist(),
            "knots_deg": knot_degrees.tolist(),
            "response_deg": (
                knot_degrees * float(_PATELLA_GAIN[side])
            ).tolist(),
            "translation_reference_local_m": np.asarray(
                patella_translation_local
            ).tolist(),
        }

        # Femur follows the pelvis/socket through parent-local FK.
        # Knee_Rotate receives the SMPL-X knee action once.  Tibia_Bone and
        # Tibia_Twist then retain their authored child-local offsets; mapping
        # both independently to the same SMPL-X knee is the double-rotation
        # bug that made the femur spear the patella in deep flexion.
        local_fk.update((femur_bone, knee_bone))
        driver_types[tibia_bone] = "bind_follow"
        driver_types[tibia_twist] = "bind_follow"
        driver_rest[_SMPLX_KNEE[side]] = knee_pivot
        # Each of these is one rigid bone, but the authored weights blend the
        # shaft and twist bones across it, so posing stretched the mesh.  The
        # forearm pair follows the same rule as the tibia: the ulna rides the
        # shaft bone and the radius rides the twist bone it rotates with.
        for mesh_name in (
            f"Femur_{suffix}",
            f"Tibia_{suffix}",
            f"Patella_{suffix}",
            f"Humerus_{suffix}",
            f"Ulna_{suffix}",
            f"Radius_{suffix}",
        ):
            _force_rigid_mesh_driver(
                asset,
                mesh_name=mesh_name,
                driver_indices=driver_indices,
                driver_weights=driver_weights,
            )
        final_head = fit_sphere_v7(vertices[head_indices])
        final_socket = fit_sphere_fixed_radius_v7(
            vertices[socket_indices], radius_m=float(final_head["radius_m"])
        )
        side_reports[side] = {
            "hip_center_before_m": float(np.linalg.norm(hip_delta)),
            "hip_center_after_m": float(
                np.linalg.norm(
                    np.asarray(final_head["center"])
                    - np.asarray(final_socket["center"])
                )
            ),
            "femoral_head_radius_change_m": abs(
                float(final_head["radius_m"]) - float(head_fit["radius_m"])
            ),
            "knee": knee_report,
            "knee_pivot_m": knee_pivot.tolist(),
            "patella_translation_m": patella_translation.tolist(),
            "patella_gain": float(_PATELLA_GAIN[side]),
        }

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    bind_local = bind_global.copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            bind_local[bone] = (
                np.linalg.inv(bind_global[int(parent)]) @ bind_global[bone]
            )
    # The legacy refit also left target_bone_head/tail several centimetres
    # away from the fitted bind origins (notably Tibia_Twist->Ankle).  Station
    # material coordinates use these endpoints, while FK uses target bind
    # matrices, so the two representations must describe the same skeleton.
    bind_origins = bind_global[:, :3, 3]
    endpoint_shift = bind_origins - bone_head
    bone_head = bind_origins.copy()
    bone_tail += endpoint_shift
    metadata = dict(asset.metadata or {})
    # V71's useful invariant was not a particular body shape.  It was that
    # every source child retained its authored parent-local translation while
    # the controller supplied only the desired orientation.  Later refits
    # kept independently mapping ankle/wrist/hand descendants to SMPL-X
    # globals; two neighbouring vessel vertices could therefore be driven by
    # frames that disagreed by tens of centimetres at the same joint.  Rebuild
    # the complete fitted hierarchy here.  ``target_bind_local`` was recomputed
    # above from this beta's corrected global bind, so this inherits linkage,
    # not V71's body proportions.
    direct_driver_bones: set[int] = set()
    metadata.update(
        {
            "artifact_schema": 7,
            "source_full_local_fk_v2": True,
            "source_connected_local_fk_v3": False,
            "source_local_fk_bones_v3": [],
            "source_direct_driver_bones_v1": [],
            "source_full_local_fk_provenance_v7": {
                "source_prior": "true_v71_blender_parent_local_bind",
                "bind_space": "beta_corrected_target_bind_local",
                "independent_global_child_anchors": False,
            },
            "source_anatomical_pivots_v7": True,
            "source_knee_hinge_splines_v7": knee_hinges,
            "source_tibia_glide_splines_v7": tibia_glides,
            "source_patella_splines_v7": patella_splines,
            "source_patella_v71_response_v8": patella_responses,
            "source_leg_hinge_solve_v1": leg_hinge_solves,
            "source_patella_response_v7": {
                side: {
                    "knots_deg": [0.0, 30.0, 60.0, 90.0, 120.0],
                    "gain": float(_PATELLA_GAIN[side]),
                }
                for side in _SIDES
            },
            "v7_no_scale_to_pass": True,
        }
    )
    corrected = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        driver_indices=driver_indices,
        driver_weights=driver_weights.astype(np.float32),
        source_driver_rest_joints=driver_rest.astype(np.float32),
        source_bone_driver_types=driver_types,
        source_bone_corrective_gain=corrective_gain.astype(np.float32),
        target_rest_global=bind_global.astype(np.float32),
        target_rest_local=bind_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(bind_global).astype(np.float32),
        target_bone_head=bone_head.astype(np.float32),
        target_bone_tail=bone_tail.astype(np.float32),
        source_driver_coupling=None,
        pose_cache_vertices=None,
        pose_cache_hash="",
        metadata=metadata,
    )
    corrected = with_source_driver_coupling(corrected)
    corrected.validate()
    passed = all(
        report["hip_center_after_m"] <= 0.002
        and report["femoral_head_radius_change_m"] <= 0.001
        for report in side_reports.values()
    )
    return corrected, {
        "schema_version": 7,
        "method": "v71_fixed_domains_minimum_articular_correction",
        "patella_source": (
            "frozen_v71_patella_oracle"
            if patella_law is not None
            else "candidate_fitted_spline_v7"
        ),
        "patella_oracle_digest": (
            str(patella_law.content_digest()) if patella_law is not None else ""
        ),
        "sides": side_reports,
        "socket_template": socket_reports,
        "selected_local_fk_bones": sorted(local_fk),
        "scaled_structures": [],
        "passed": bool(passed),
    }


__all__ = ["reconstruct_articular_subject_v7"]
