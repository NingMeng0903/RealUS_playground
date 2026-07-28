from __future__ import annotations

import json

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
    REQUIRED_CONTROLLER_JOINTS,
    REQUIRED_LOCAL_FK_LINKS,
    build_joint_material_domains_v7,
    diagnose_joint_contact_v7,
    rigid_edge_metrics_v7,
)


_OCTA_FACES = np.asarray(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ],
    dtype=np.int64,
)


def _octahedron(center: tuple[float, float, float], radius: float) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64)
    return center_array + radius * np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )


def _diagnostic_fixture():
    chunks: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    memberships: dict[str, np.ndarray] = {}

    def add(name: str, center: tuple[float, float, float], radius: float) -> np.ndarray:
        start = sum(len(chunk) for chunk in chunks)
        chunks.append(_octahedron(center, radius))
        faces.append(_OCTA_FACES + start)
        indices = np.arange(start, start + 6, dtype=np.int64)
        memberships[name] = indices
        return indices

    for side, x in (("left", -0.10), ("right", 0.10)):
        head = add(f"{side}/femoral_head", (x, 0.0, 0.0), 0.010)
        add(f"{side}/acetabulum", (x, 0.0, 0.0), 0.012)
        medial_condyle = add(
            f"{side}/femoral_condyle_medial",
            (x + (-0.006 if side == "left" else 0.006), -0.40, 0.0),
            0.004,
        )
        lateral_condyle = add(
            f"{side}/femoral_condyle_lateral",
            (x + (0.006 if side == "left" else -0.006), -0.40, 0.0),
            0.004,
        )
        medial_plateau = add(
            f"{side}/tibial_plateau_medial",
            (x + (-0.006 if side == "left" else 0.006), -0.410, 0.0),
            0.004,
        )
        lateral_plateau = add(
            f"{side}/tibial_plateau_lateral",
            (x + (0.006 if side == "left" else -0.006), -0.410, 0.0),
            0.004,
        )
        trochlea = add(f"{side}/trochlea", (x, -0.40, 0.003), 0.003)
        patella = add(f"{side}/patella", (x, -0.40, 0.010), 0.003)
        memberships[f"{side}/patella_articular"] = patella.copy()
        memberships[f"{side}/pelvis"] = memberships[f"{side}/acetabulum"].copy()
        memberships[f"{side}/femur"] = np.concatenate(
            (head, medial_condyle, lateral_condyle, trochlea)
        )
        memberships[f"{side}/tibia"] = np.concatenate(
            (medial_plateau, lateral_plateau)
        )

    vertices = np.concatenate(chunks, axis=0)
    triangles = np.concatenate(faces, axis=0)
    domains = FrozenJointMaterialDomainsV7.freeze(
        source_bind_vertices=vertices,
        faces=triangles,
        domains=memberships,
    )
    controller = {
        name: {"translation_error_m": 0.0002, "rotation_error_deg": 0.2}
        for name in REQUIRED_CONTROLLER_JOINTS
    }
    local_fk = {
        name: {"translation_error_m": 0.0002, "rotation_error_deg": 0.2}
        for name in REQUIRED_LOCAL_FK_LINKS
    }
    trajectory = np.repeat(vertices[None, :, :], 3, axis=0)
    for pose, delta in enumerate((0.0, 0.001, 0.002)):
        for side in ("left", "right"):
            trajectory[pose, domains.require(f"{side}/patella"), 1] += delta
    return vertices, triangles, domains, controller, local_fk, trajectory


def _good_report():
    vertices, faces, domains, controller, local_fk, trajectory = (
        _diagnostic_fixture()
    )
    report = diagnose_joint_contact_v7(
        domains,
        reference_vertices=vertices,
        final_vertices=vertices,
        faces=faces,
        controller_observations=controller,
        local_fk_observations=local_fk,
        trajectory_vertices=trajectory,
        oracle_trajectory_vertices=trajectory,
    )
    return report, vertices, faces, domains, controller, local_fk, trajectory


def test_fixed_domains_round_trip_and_reject_changed_topology(tmp_path) -> None:
    _report, vertices, faces, domains, *_rest = _good_report()
    path = tmp_path / "joint_domains.json"
    domains.save_json(path)
    loaded = FrozenJointMaterialDomainsV7.load_json(path)

    assert json.loads(path.read_text())["schema_version"] == 7
    assert loaded.topology_digest == domains.topology_digest
    assert np.array_equal(
        loaded.require("left/patella"), domains.require("left/patella")
    )
    assert not np.shares_memory(
        loaded.require("left/patella"), domains.require("left/patella")
    )
    assert not np.intersect1d(
        loaded.require("left/patella"), loaded.require("left/tibia")
    ).size
    with pytest.raises(ValueError, match="faces do not match"):
        loaded.validate_topology(vertices, faces[::-1])


def test_three_gates_are_strict_and_ignore_incoming_pass_flags() -> None:
    report, vertices, faces, domains, _controller, _local_fk, trajectory = (
        _good_report()
    )
    assert report["passed"]
    assert all(report["gates"][gate]["pass"] for gate in report["pass_requires"])

    # This is the old false-positive shape: upstream code says "pass" but
    # supplies no independently measured residuals.  V7 ignores those flags.
    claimed = diagnose_joint_contact_v7(
        domains,
        reference_vertices=vertices,
        final_vertices=vertices,
        faces=faces,
        controller_observations={
            name: {"pass": True} for name in REQUIRED_CONTROLLER_JOINTS
        },
        local_fk_observations={
            name: {"pass": True} for name in REQUIRED_LOCAL_FK_LINKS
        },
        trajectory_vertices=trajectory,
        oracle_trajectory_vertices=trajectory,
    )
    assert claimed["gates"]["geometry"]["pass"]
    assert not claimed["gates"]["controller"]["pass"]
    assert not claimed["gates"]["local_fk"]["pass"]
    assert not claimed["passed"]


def test_final_vertices_override_no_metric_and_bad_hip_cannot_self_certify() -> None:
    _report, vertices, faces, domains, controller, local_fk, trajectory = (
        _good_report()
    )
    bad = vertices.copy()
    bad[domains.require("left/femoral_head")] += np.array([0.010, 0.0, 0.0])

    report = diagnose_joint_contact_v7(
        domains,
        reference_vertices=vertices,
        final_vertices=bad,
        faces=faces,
        controller_observations=controller,
        local_fk_observations=local_fk,
        trajectory_vertices=trajectory,
        oracle_trajectory_vertices=trajectory,
    )
    hip = report["gates"]["geometry"]["hips"]["left"]
    assert hip["center_error_m"] > 0.009
    assert not hip["pass"]
    assert not report["gates"]["geometry"]["pass"]
    assert not report["passed"]


def test_rigidly_rotated_aspherical_head_passes_the_hip_gate() -> None:
    """Articulation redistributes clearance; only deforming the bone avoids it."""
    _report, vertices, faces, domains, controller, local_fk, trajectory = (
        _good_report()
    )
    head = domains.require("left/femoral_head")
    center = vertices[domains.require("left/acetabulum")].mean(axis=0)
    angle = np.radians(20.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated = vertices.copy()
    rotated[head] = (rotation @ (vertices[head] - center).T).T + center

    report = diagnose_joint_contact_v7(
        domains,
        reference_vertices=vertices,
        final_vertices=rotated,
        faces=faces,
        controller_observations=controller,
        local_fk_observations=local_fk,
        trajectory_vertices=trajectory,
        oracle_trajectory_vertices=trajectory,
    )
    hip = report["gates"]["geometry"]["hips"]["left"]
    # Larger than the 1 mm median change the gate used to reject.
    assert hip["clearance_median_change_m"] > 0.002
    assert hip["center_error_m"] < 1.0e-9
    assert hip["radius_change_m"] < 1.0e-9
    assert hip["clearance_min_drop_m"] == pytest.approx(0.0)
    assert hip["pass"]


def test_head_crashing_into_the_socket_still_fails_the_hip_gate() -> None:
    _report, vertices, faces, domains, controller, local_fk, trajectory = (
        _good_report()
    )
    head = domains.require("left/femoral_head")
    crashed = vertices.copy()
    # Drive one head vertex out toward the socket wall. The least-squares sphere
    # centre and radius barely move, so only the clearance collapse is left to
    # catch it.
    crashed[head[2], 1] += 0.0018

    report = diagnose_joint_contact_v7(
        domains,
        reference_vertices=vertices,
        final_vertices=crashed,
        faces=faces,
        controller_observations=controller,
        local_fk_observations=local_fk,
        trajectory_vertices=trajectory,
        oracle_trajectory_vertices=trajectory,
    )
    hip = report["gates"]["geometry"]["hips"]["left"]
    assert hip["clearance_min_drop_m"] > 0.001
    assert not hip["pass"]
    assert not report["passed"]


def test_patella_is_not_masked_by_a_good_tibia() -> None:
    _report, vertices, faces, domains, controller, local_fk, trajectory = (
        _good_report()
    )
    bad_trajectory = trajectory.copy()
    bad_trajectory[1:, domains.require("left/patella"), 2] += 0.020

    report = diagnose_joint_contact_v7(
        domains,
        reference_vertices=vertices,
        final_vertices=vertices,
        faces=faces,
        controller_observations=controller,
        local_fk_observations=local_fk,
        trajectory_vertices=bad_trajectory,
        oracle_trajectory_vertices=trajectory,
    )
    geometry = report["gates"]["geometry"]
    assert geometry["knees"]["left"]["pass"]
    assert geometry["rigidity"]["left/tibia"]["pass"]
    assert not geometry["patellofemoral"]["left"]["pass"]
    assert not report["passed"]


def test_rigid_edge_gate_detects_length_scaling() -> None:
    _report, vertices, faces, domains, *_rest = _good_report()
    femur = domains.require("left/femur")
    bad = vertices.copy()
    center = np.mean(bad[femur], axis=0)
    bad[femur] = center + 1.05 * (bad[femur] - center)

    metrics = rigid_edge_metrics_v7(
        reference_vertices=vertices,
        final_vertices=bad,
        faces=faces,
        indices=femur,
    )
    assert metrics["ratio_q99"] == pytest.approx(1.05)
    assert not metrics["pass"]


def _cube(center: tuple[float, float, float], half: float = 0.02) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    return c + np.asarray(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ]
    )


_CUBE_FACES = np.asarray(
    [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ],
    dtype=np.int64,
)


def test_builder_freezes_separate_leg_surfaces_from_source_topology() -> None:
    chunks: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    names: list[str] = []
    ranges: list[tuple[int, int]] = []

    def add_object(name: str, centers: list[tuple[float, float, float]]) -> None:
        start = sum(len(chunk) for chunk in chunks)
        for center in centers:
            offset = sum(len(chunk) for chunk in chunks)
            chunks.append(_cube(center))
            all_faces.append(_CUBE_FACES + offset)
        stop = sum(len(chunk) for chunk in chunks)
        names.append(name)
        ranges.append((start, stop))

    for side, x in (("L", -0.15), ("R", 0.15)):
        add_object(f"Pelvis_{side}", [(x, 0.0, 0.0)])
        add_object(
            f"Femur_{side}",
            [(x, -0.05, 0.0), (x, -0.25, 0.0), (x, -0.45, 0.0)],
        )
        add_object(
            f"Tibia_{side}",
            [(x, -0.47, 0.0), (x, -0.70, 0.0), (x, -0.90, 0.0)],
        )
        add_object(f"Patella_{side}", [(x, -0.44, 0.05)])

    vertices = np.concatenate(chunks, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    registration = vertices.copy()
    built = build_joint_material_domains_v7(
        source_bind_vertices=vertices,
        registration_vertices=registration,
        faces=faces,
        source_mesh_names=names,
        source_vertex_ranges=np.asarray(ranges),
        source_tissues=["bone"] * len(names),
    )
    frozen_left_patella = built.require("left/patella").copy()
    registration[:] = 123.0

    assert np.array_equal(built.require("left/patella"), frozen_left_patella)
    assert not np.intersect1d(
        built.require("left/patella"), built.require("left/tibia")
    ).size
    assert not np.intersect1d(
        built.require("right/patella"), built.require("right/tibia")
    ).size
    assert len(built.require("left/femoral_head")) >= 4
    assert len(built.require("left/acetabulum")) >= 4
