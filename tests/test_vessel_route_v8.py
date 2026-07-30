from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget import vessel_route_v8
from projects.genesis_ue_sync.anatomy_retarget.vessel_route_v8 import (
    CollisionSurfaceV8,
    VesselComponentV8,
    route_vessel_vertices_v8,
)


def _tube(
    *,
    x_min: float,
    x_max: float,
    center_y: float,
    radius: float,
    stations: int = 25,
    ring: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, stations)
    angle = np.linspace(0.0, 2.0 * np.pi, ring, endpoint=False)
    vertices = np.asarray(
        [
            (value, center_y + radius * np.cos(theta), radius * np.sin(theta))
            for value in x
            for theta in angle
        ],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for station in range(stations - 1):
        for side in range(ring):
            a = station * ring + side
            b = station * ring + (side + 1) % ring
            c = (station + 1) * ring + side
            d = (station + 1) * ring + (side + 1) % ring
            faces.extend(((a, c, b), (b, c, d)))
    return vertices, np.asarray(faces, dtype=np.int32)


def _sphere_marker(radius: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (radius, 0.0, 0.0),
            (-radius, 0.0, 0.0),
            (0.0, radius, 0.0),
            (0.0, 0.0, radius),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 3), (2, 1, 3), (1, 0, 3), (0, 1, 2)), dtype=np.int32
    )
    return vertices, faces


def _install_analytic_sphere_distance(monkeypatch: object) -> None:
    def analytic(
        points: np.ndarray,
        surface_vertices: np.ndarray,
        _surface_faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius = float(np.max(np.linalg.norm(surface_vertices, axis=1)))
        norm = np.linalg.norm(points, axis=1)
        normals = np.asarray(points, dtype=np.float64) / np.maximum(
            norm[:, None], 1.0e-12
        )
        closest = radius * normals
        return norm - radius, closest, normals

    monkeypatch.setattr(vessel_route_v8, "signed_distance", analytic)


def test_connected_route_reduces_bone_penetration_without_point_clipping(
    monkeypatch: object,
) -> None:
    _install_analytic_sphere_distance(monkeypatch)
    vertices, faces = _tube(
        x_min=-0.65,
        x_max=0.65,
        center_y=0.34,
        radius=0.025,
    )
    body_vertices, body_faces = _sphere_marker(2.0)
    bone_vertices, bone_faces = _sphere_marker(0.35)
    component = VesselComponentV8(
        mesh_name="Artery",
        vertex_ids=np.arange(len(vertices), dtype=np.int32),
        local_faces=faces,
    )
    initial_signed, _closest, _normals = vessel_route_v8.signed_distance(
        vertices, bone_vertices, bone_faces
    )
    routed, report = route_vessel_vertices_v8(
        vertices,
        [component],
        skin_vertices=body_vertices,
        skin_faces=body_faces,
        collision_surfaces=[
            CollisionSurfaceV8("Femur_L", bone_vertices, bone_faces)
        ],
        max_iterations=12,
        bone_clearance_m=0.001,
        smooth_weight=20.0,
        maximum_component_displacement_m=0.06,
    )
    final_signed, _closest, _normals = vessel_route_v8.signed_distance(
        routed, bone_vertices, bone_faces
    )
    assert float(np.min(final_signed)) > float(np.min(initial_signed)) + 0.02
    assert report["bone_maximum_penetration_m"] < -float(np.min(initial_signed))
    # Unconstrained end rings move as part of the connected field.  An
    # independent closest-point clip would leave them exactly unchanged.
    displacement = np.linalg.norm(routed - vertices, axis=1)
    assert float(np.max(displacement[:8])) > 0.0
    assert report["topology_preserved"] is True
    assert report["runtime_collision_solve"] is False


def test_route_reports_fail_closed_when_constraints_conflict(
    monkeypatch: object,
) -> None:
    _install_analytic_sphere_distance(monkeypatch)
    vertices, faces = _tube(
        x_min=-0.10,
        x_max=0.10,
        center_y=0.0,
        radius=0.01,
        stations=7,
    )
    # The collision solid is larger than the containing skin, so no route can
    # satisfy both.  The bake must return evidence rather than claim success.
    skin_vertices, skin_faces = _sphere_marker(0.20)
    bone_vertices, bone_faces = _sphere_marker(0.30)
    component = VesselComponentV8(
        mesh_name="Vein",
        vertex_ids=np.arange(len(vertices), dtype=np.int32),
        local_faces=faces,
    )
    _routed, report = route_vessel_vertices_v8(
        vertices,
        [component],
        skin_vertices=skin_vertices,
        skin_faces=skin_faces,
        collision_surfaces=[
            CollisionSurfaceV8("ImpossibleBone", bone_vertices, bone_faces)
        ],
        max_iterations=4,
        maximum_component_displacement_m=0.05,
    )
    assert report["passed"] is False
    assert report["publishable"] is False
    assert (
        report["skin_outside_count"] > 0
        or report["bone_maximum_penetration_m"] > 0.001
    )


def test_route_fails_when_inside_physical_skin_but_inside_margin(
    monkeypatch: object,
) -> None:
    _install_analytic_sphere_distance(monkeypatch)
    skin_vertices, skin_faces = _sphere_marker(2.0)
    # This point is physically inside the skin but only 0.1 mm from it.  The
    # zero displacement cap leaves the violated contracted-shell constraint
    # observable in the final report.
    vertices = np.asarray(((1.9999, 0.0, 0.0),), dtype=np.float64)
    component = VesselComponentV8(
        mesh_name="NearSkinNerve",
        vertex_ids=np.asarray((0,), dtype=np.int32),
        local_faces=np.empty((0, 3), dtype=np.int32),
    )

    _routed, report = route_vessel_vertices_v8(
        vertices,
        [component],
        skin_vertices=skin_vertices,
        skin_faces=skin_faces,
        collision_surfaces=[],
        max_iterations=1,
        skin_margin_m=0.00025,
        maximum_component_displacement_m=0.0,
    )

    assert report["skin_outside_count"] == 0
    assert report["skin_clearance_violation_count"] == 1
    assert report["skin_maximum_clearance_violation_m"] > 0.0
    assert report["passed"] is False
    assert report["publishable"] is False
