"""Source-surface contact preservation for Stage-1 rigid anatomy."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .material_fit import shaft_preserving_segment_map


def _mesh_indices(asset: Any, *, token: str, side: str) -> np.ndarray:
    suffix = "_l" if side == "left" else "_r"
    chunks: list[np.ndarray] = []
    for name, (start, stop), tissue in zip(
        asset.source_mesh_names,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
        strict=True,
    ):
        lower = str(name).lower()
        if (
            str(tissue).lower() == "bone"
            and token in lower
            and (lower.endswith(suffix) or f"{suffix}_" in lower)
        ):
            chunks.append(np.arange(int(start), int(stop), dtype=np.int64))
    if not chunks:
        raise ValueError(f"missing {side} bone mesh containing {token!r}")
    return np.concatenate(chunks)


def _closest_pair(
    first: np.ndarray,
    second: np.ndarray,
    vertices: np.ndarray,
) -> tuple[int, int, float]:
    distance, nearest = cKDTree(vertices[second]).query(vertices[first], k=1)
    local = int(np.argmin(distance))
    return (
        int(first[local]),
        int(second[int(nearest[local])]),
        float(distance[local]),
    )


def _endpoint_caps(
    indices: np.ndarray,
    vertices: np.ndarray,
    *,
    fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    points = vertices[indices]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _u, _singular, vt = np.linalg.svd(centered, full_matrices=False)
    parameter = centered @ vt[0]
    low, high = np.quantile(parameter, (fraction, 1.0 - fraction))
    return indices[parameter <= low], indices[parameter >= high]


def _orient_caps(
    caps: tuple[np.ndarray, np.ndarray],
    *,
    proximal_neighbour: np.ndarray,
    distal_neighbour: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first, second = caps
    direct = _closest_pair(first, proximal_neighbour, vertices)[2] + _closest_pair(
        second, distal_neighbour, vertices
    )[2]
    reversed_cost = _closest_pair(second, proximal_neighbour, vertices)[2] + _closest_pair(
        first, distal_neighbour, vertices
    )[2]
    return (first, second) if direct <= reversed_cost else (second, first)


def _minimum_displacement_contact_target(
    *,
    mapped: np.ndarray,
    bone_cap: np.ndarray,
    neighbour: np.ndarray,
    source_clearance: float,
) -> tuple[int, int, np.ndarray, float]:
    """Close a mapped gap along its shortest path, retaining source clearance."""
    bone_vertex, neighbour_vertex, mapped_gap = _closest_pair(
        bone_cap, neighbour, mapped
    )
    delta = mapped[bone_vertex] - mapped[neighbour_vertex]
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-10:
        raise ValueError("mapped source contact has a degenerate clearance direction")
    target = (
        mapped[neighbour_vertex]
        + delta / length * float(source_clearance)
    )
    return bone_vertex, neighbour_vertex, target, mapped_gap


def fit_lower_leg_source_contacts(
    asset: Any,
    *,
    source_vertices: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit tibia/fibula shafts while preserving authored endpoint clearances.

    Endpoint-cap membership and clearance magnitudes are frozen on the Blender
    source mesh.  Each subject-beta target follows the shortest mapped contact
    direction, which gives the minimum endpoint displacement and avoids
    transporting a source-space lateral offset outside a narrow target limb.
    Only the shaft length changes; both epiphyses undergo rigid motion through
    ``shaft_preserving_segment_map``.
    """
    source = np.asarray(
        source_vertices
        if source_vertices is not None
        else (
            asset.source_bind_vertices
            if getattr(asset, "source_bind_vertices", None) is not None
            else asset.registration_reference
        ),
        dtype=np.float64,
    )
    mapped = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    if source.shape != mapped.shape:
        raise ValueError("source contact reference must match mapped anatomy")

    report: dict[str, Any] = {}
    for side in ("left", "right"):
        femur = _mesh_indices(asset, token="femur", side=side)
        tibia = _mesh_indices(asset, token="tibia", side=side)
        fibula = _mesh_indices(asset, token="fibula", side=side)
        talus = _mesh_indices(asset, token="talus", side=side)

        tibia_proximal_cap, tibia_distal_cap = _orient_caps(
            _endpoint_caps(tibia, source),
            proximal_neighbour=femur,
            distal_neighbour=talus,
            vertices=source,
        )

        _source_tibia_proximal, _source_femur_contact, source_tibia_knee = _closest_pair(
            tibia_proximal_cap, femur, source
        )
        _source_tibia_distal, _source_talus_contact, source_tibia_ankle = _closest_pair(
            tibia_distal_cap, talus, source
        )
        tibia_proximal, femur_contact, target_a, mapped_tibia_knee_before = (
            _minimum_displacement_contact_target(
                mapped=mapped,
                bone_cap=tibia_proximal_cap,
                neighbour=femur,
                source_clearance=source_tibia_knee,
            )
        )
        tibia_distal, talus_contact, target_b, mapped_tibia_ankle_before = (
            _minimum_displacement_contact_target(
                mapped=mapped,
                bone_cap=tibia_distal_cap,
                neighbour=talus,
                source_clearance=source_tibia_ankle,
            )
        )
        tibia_before = mapped[tibia].copy()
        mapped[tibia] = shaft_preserving_segment_map(
            mapped[tibia],
            source_a=mapped[tibia_proximal],
            source_b=mapped[tibia_distal],
            target_a=target_a,
            target_b=target_b,
        )

        fibula_proximal_cap, fibula_distal_cap = _orient_caps(
            _endpoint_caps(fibula, source),
            proximal_neighbour=tibia_proximal_cap,
            distal_neighbour=talus,
            vertices=source,
        )
        _source_fibula_proximal, _source_tibia_contact, source_fibula_knee = _closest_pair(
            fibula_proximal_cap, tibia_proximal_cap, source
        )
        _source_fibula_distal, _source_fibula_talus_contact, source_fibula_ankle = _closest_pair(
            fibula_distal_cap, talus, source
        )
        fibula_proximal, tibia_contact, target_a, mapped_fibula_knee_before = (
            _minimum_displacement_contact_target(
                mapped=mapped,
                bone_cap=fibula_proximal_cap,
                neighbour=tibia_proximal_cap,
                source_clearance=source_fibula_knee,
            )
        )
        fibula_distal, fibula_talus_contact, target_b, mapped_fibula_ankle_before = (
            _minimum_displacement_contact_target(
                mapped=mapped,
                bone_cap=fibula_distal_cap,
                neighbour=talus,
                source_clearance=source_fibula_ankle,
            )
        )
        fibula_before = mapped[fibula].copy()
        mapped[fibula] = shaft_preserving_segment_map(
            mapped[fibula],
            source_a=mapped[fibula_proximal],
            source_b=mapped[fibula_distal],
            target_a=target_a,
            target_b=target_b,
        )

        report[side] = {
            "policy": "minimum_displacement_source_clearance_shaft_fit_v2",
            "source_clearance_m": {
                "tibia_knee": source_tibia_knee,
                "tibia_ankle": source_tibia_ankle,
                "fibula_knee": source_fibula_knee,
                "fibula_ankle": source_fibula_ankle,
            },
            "mapped_clearance_m": {
                "tibia_knee": _closest_pair(tibia, femur, mapped)[2],
                "tibia_ankle": _closest_pair(tibia, talus, mapped)[2],
                "fibula_knee": _closest_pair(fibula, tibia, mapped)[2],
                "fibula_ankle": _closest_pair(fibula, talus, mapped)[2],
            },
            "mapped_clearance_before_m": {
                "tibia_knee": mapped_tibia_knee_before,
                "tibia_ankle": mapped_tibia_ankle_before,
                "fibula_knee": mapped_fibula_knee_before,
                "fibula_ankle": mapped_fibula_ankle_before,
            },
            "maximum_tibia_displacement_m": float(
                np.max(np.linalg.norm(mapped[tibia] - tibia_before, axis=1))
            ),
            "maximum_fibula_displacement_m": float(
                np.max(np.linalg.norm(mapped[fibula] - fibula_before, axis=1))
            ),
        }

    metadata = dict(asset.metadata or {})
    metadata["source_surface_contact_fit"] = report
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": mapped.astype(np.float32),
            "metadata": metadata,
        }
    )
    return result, report
