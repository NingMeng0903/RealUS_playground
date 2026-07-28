"""Fixed-material joint contact diagnostics for the V7 anatomy operator.

The older joint report selected landmarks again after subject fitting and
allowed values copied from ``material_fit`` metadata to replace measurements.
That makes a poor fit capable of redefining its own probes.  This module binds
all anatomical surface domains once, on the immutable authored topology, and
only indexes those persisted vertex ids when evaluating a subject or pose.

The module deliberately has no dependency on the fitting pipeline or its
metadata.  Controller, local-FK, and final-geometry evidence are evaluated as
three independent gates and the public result is their logical AND.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 7
SIDES = ("left", "right")

REQUIRED_CONTROLLER_JOINTS = (
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
)
REQUIRED_LOCAL_FK_LINKS = (
    "left/Femur_Rot>Knee_Rotate",
    "left/Knee_Rotate>Tibia_Bone",
    "left/Tibia_Bone>Patella_Rotate",
    "right/Femur_Rot>Knee_Rotate",
    "right/Knee_Rotate>Tibia_Bone",
    "right/Tibia_Bone>Patella_Rotate",
)


@dataclass(frozen=True)
class JointContactThresholdsV7:
    """Fail-closed acceptance limits, in metres and degrees."""

    controller_translation_error_m: float = 0.001
    controller_rotation_error_deg: float = 1.0
    local_fk_translation_error_m: float = 0.001
    local_fk_rotation_error_deg: float = 1.0
    hip_center_error_m: float = 0.002
    hip_center_drift_m: float = 0.001
    hip_radius_relative_change: float = 0.02
    hip_radius_absolute_change_m: float = 0.001
    hip_clearance_median_change_m: float = 0.001
    hip_clearance_q95_change_m: float = 0.002
    hip_max_separation_m: float = 0.003
    knee_gap_min_m: float = 0.0
    knee_gap_max_m: float = 0.003
    knee_gap_change_m: float = 0.002
    knee_axis_error_deg: float = 2.0
    knee_pivot_drift_m: float = 0.0015
    femur_length_change_m: float = 0.0005
    patellofemoral_gap_min_m: float = 0.0
    patellofemoral_gap_max_m: float = 0.004
    patellofemoral_gap_drift_m: float = 0.002
    patella_trajectory_rms_m: float = 0.002
    patella_trajectory_max_m: float = 0.003
    patella_trajectory_direction_deg: float = 2.0
    rigid_q01_ratio: float = 0.99
    rigid_q99_ratio: float = 1.01
    rigid_min_ratio: float = 0.98
    rigid_max_ratio: float = 1.02

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JointContactThresholdsV7":
        known = {field: value[field] for field in asdict(cls()) if field in value}
        return cls(**known)


def _array_digest(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def _topology_digest(vertex_count: int, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"vertices:{int(vertex_count)}".encode("ascii"))
    _array_digest(digest, "faces", np.asarray(faces, dtype=np.int64))
    return digest.hexdigest()


def _geometry_digest(vertices: np.ndarray) -> str:
    digest = hashlib.sha256()
    _array_digest(digest, "source_bind_vertices", np.asarray(vertices, dtype=np.float64))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenJointMaterialDomainsV7:
    """Vertex memberships selected once from the authored source topology."""

    vertex_count: int
    face_count: int
    topology_digest: str
    source_bind_digest: str
    domains: Mapping[str, np.ndarray]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        frozen: dict[str, np.ndarray] = {}
        for name, value in self.domains.items():
            indices = np.unique(np.asarray(value, dtype=np.int64).reshape(-1))
            if not len(indices):
                raise ValueError(f"material domain {name!r} is empty")
            if np.any(indices < 0) or np.any(indices >= int(self.vertex_count)):
                raise ValueError(f"material domain {name!r} contains an invalid vertex")
            indices.setflags(write=False)
            frozen[str(name)] = indices
        object.__setattr__(self, "domains", MappingProxyType(frozen))
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ValueError(
                f"joint material domain schema must be {SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )

    @classmethod
    def freeze(
        cls,
        *,
        source_bind_vertices: np.ndarray,
        faces: np.ndarray,
        domains: Mapping[str, np.ndarray],
    ) -> "FrozenJointMaterialDomainsV7":
        vertices = _vertices(source_bind_vertices, "source_bind_vertices")
        triangles = _faces(faces, len(vertices))
        return cls(
            vertex_count=len(vertices),
            face_count=len(triangles),
            topology_digest=_topology_digest(len(vertices), triangles),
            source_bind_digest=_geometry_digest(vertices),
            domains=domains,
        )

    def validate_topology(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        points = _vertices(vertices, "vertices")
        triangles = _faces(faces, len(points))
        if len(points) != int(self.vertex_count) or len(triangles) != int(self.face_count):
            raise ValueError("candidate topology does not match frozen material domains")
        if _topology_digest(len(points), triangles) != self.topology_digest:
            raise ValueError("candidate faces do not match frozen material domains")

    def require(self, name: str) -> np.ndarray:
        try:
            return self.domains[name]
        except KeyError as exc:
            raise ValueError(f"required fixed material domain {name!r} is missing") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "vertex_count": int(self.vertex_count),
            "face_count": int(self.face_count),
            "topology_digest": self.topology_digest,
            "source_bind_digest": self.source_bind_digest,
            "domains": {
                name: np.asarray(indices, dtype=np.int64).tolist()
                for name, indices in sorted(self.domains.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenJointMaterialDomainsV7":
        return cls(
            schema_version=int(value["schema_version"]),
            vertex_count=int(value["vertex_count"]),
            face_count=int(value["face_count"]),
            topology_digest=str(value["topology_digest"]),
            source_bind_digest=str(value["source_bind_digest"]),
            domains={
                str(name): np.asarray(indices, dtype=np.int64)
                for name, indices in dict(value["domains"]).items()
            },
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "FrozenJointMaterialDomainsV7":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _vertices(value: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{label} must be a non-empty [N,3] array")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return points


def _faces(value: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(value, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be [F,3]")
    if faces.size and (np.any(faces < 0) or np.any(faces >= int(vertex_count))):
        raise ValueError("faces reference an invalid vertex")
    return faces


def _normalise_side(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"l", "left"}:
        return "left"
    if text in {"r", "right"}:
        return "right"
    return ""


def _name_side(name: str) -> str:
    lower = name.lower()
    if (
        lower.endswith(("_l", ".l", "-l", "_left"))
        or "_l_" in lower
        or "left" in lower
    ):
        return "left"
    if (
        lower.endswith(("_r", ".r", "-r", "_right"))
        or "_r_" in lower
        or "right" in lower
    ):
        return "right"
    return ""


def _mesh_indices(
    *,
    mesh_names: Sequence[str],
    vertex_ranges: np.ndarray,
    tissues: Sequence[str] | None,
    sides: Sequence[str] | None,
    tokens: Sequence[str],
    side: str | None,
    allow_midline: bool = False,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    lowered_tokens = tuple(token.lower() for token in tokens)
    for mesh_index, (name, limits) in enumerate(zip(mesh_names, vertex_ranges)):
        lower = str(name).lower()
        if not any(token in lower for token in lowered_tokens):
            continue
        if tissues is not None and str(tissues[mesh_index]).lower() != "bone":
            continue
        declared = (
            _normalise_side(sides[mesh_index])
            if sides is not None and mesh_index < len(sides)
            else ""
        )
        actual_side = declared or _name_side(lower)
        if side is not None and actual_side != side and not (
            allow_midline and actual_side == ""
        ):
            continue
        start, stop = (int(limits[0]), int(limits[1]))
        if stop > start:
            selected.append(np.arange(start, stop, dtype=np.int64))
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(selected))


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vt[0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError("cannot derive a principal axis from coincident points")
    return axis / norm


def _nearest_distance(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(points) or not len(target):
        return np.full((len(points),), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distance, _indices = cKDTree(target).query(points, k=1)
        return np.asarray(distance, dtype=np.float64)
    except Exception:
        result = np.empty((len(points),), dtype=np.float64)
        batch = max(1, min(2048, int(8_000_000 / max(1, len(target)))))
        for start in range(0, len(points), batch):
            chunk = points[start : start + batch]
            squared = np.sum((chunk[:, None] - target[None, :]) ** 2, axis=2)
            result[start : start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
        return result


def _endpoint_caps(
    indices: np.ndarray,
    vertices: np.ndarray,
    *,
    fraction: float = 0.22,
) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 8:
        raise ValueError("an endpoint material domain requires at least eight vertices")
    points = vertices[indices]
    parameter = (points - np.mean(points, axis=0)) @ _principal_axis(points)
    low, high = np.quantile(parameter, (fraction, 1.0 - fraction))
    return indices[parameter <= low], indices[parameter >= high]


def _choose_near(
    first: np.ndarray,
    second: np.ndarray,
    vertices: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first_distance = float(np.median(_nearest_distance(vertices[first], target)))
    second_distance = float(np.median(_nearest_distance(vertices[second], target)))
    return (first, second) if first_distance <= second_distance else (second, first)


def _split_medial_lateral(
    indices: np.ndarray,
    vertices: np.ndarray,
    *,
    side: str,
    lateral_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parameter = vertices[indices] @ lateral_axis
    middle = float(np.median(parameter))
    # ``lateral_axis`` points left -> right.  A left medial surface points
    # right, and a right medial surface points left.
    medial_mask = parameter >= middle if side == "left" else parameter <= middle
    medial = indices[medial_mask]
    lateral = indices[~medial_mask]
    if len(medial) < 3 or len(lateral) < 3:
        order = np.argsort(parameter)
        half = max(3, len(order) // 2)
        lower, upper = indices[order[:half]], indices[order[half:]]
        medial, lateral = (upper, lower) if side == "left" else (lower, upper)
    return medial, lateral


def fit_sphere_v7(points: np.ndarray) -> dict[str, Any]:
    """Least-squares sphere fit with explicit degeneracy and residual output."""
    samples = _vertices(points, "sphere points")
    if len(samples) < 4:
        return {"available": False, "reason": "sphere fit needs at least four points"}
    matrix = np.column_stack((2.0 * samples, np.ones(len(samples))))
    rhs = np.sum(samples * samples, axis=1)
    solution, _residuals, rank, _singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    if int(rank) < 4:
        return {"available": False, "reason": "sphere points are degenerate"}
    center = solution[:3]
    radius_squared = float(solution[3] + np.dot(center, center))
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        return {"available": False, "reason": "sphere fit produced an invalid radius"}
    radius = float(np.sqrt(radius_squared))
    radial_residual = np.abs(np.linalg.norm(samples - center, axis=1) - radius)
    return {
        "available": True,
        "center": center,
        "radius_m": radius,
        "rms_residual_m": float(np.sqrt(np.mean(radial_residual**2))),
        "max_residual_m": float(np.max(radial_residual)),
    }


def fit_sphere_fixed_radius_v7(
    points: np.ndarray,
    *,
    radius_m: float,
    initial_center: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a sphere cap while holding its anatomical radius fixed.

    An acetabulum is only a shallow concave cap, so an unconstrained four
    parameter sphere fit is ill-conditioned and can move its centre by
    centimetres while retaining a small residual.  The radius of the opposing
    fixed femoral-head material domain supplies the missing constraint.  The
    centre is still solved only from socket vertices; the femoral-head centre
    is never used by this fit.
    """
    samples = _vertices(points, "fixed-radius sphere points")
    radius = float(radius_m)
    if len(samples) < 4 or not np.isfinite(radius) or radius <= 0.0:
        return {
            "available": False,
            "reason": "fixed-radius sphere fit needs four points and a positive radius",
        }
    initial_fit = fit_sphere_v7(samples)
    if initial_center is not None:
        initial = np.asarray(initial_center, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(initial)):
            return {
                "available": False,
                "reason": "fixed-radius sphere initial centre is non-finite",
            }
    else:
        initial = (
            np.asarray(initial_fit["center"], dtype=np.float64)
            if initial_fit["available"]
            else np.mean(samples, axis=0)
        )
    try:
        from scipy.optimize import least_squares

        solved = least_squares(
            lambda center: np.linalg.norm(samples - center, axis=1) - radius,
            initial,
            loss="soft_l1",
            f_scale=5.0e-4,
            max_nfev=256,
        )
        center = np.asarray(solved.x, dtype=np.float64)
    except Exception:
        # Small deterministic Gauss-Newton fallback for Blender/minimal
        # environments where scipy is deliberately unavailable.
        center = initial.copy()
        for _iteration in range(64):
            delta = center[None, :] - samples
            distance = np.linalg.norm(delta, axis=1)
            valid = distance > 1.0e-10
            if np.count_nonzero(valid) < 3:
                break
            jacobian = delta[valid] / distance[valid, None]
            residual = distance[valid] - radius
            step, _residuals, _rank, _singular = np.linalg.lstsq(
                jacobian, -residual, rcond=None
            )
            center += step
            if float(np.linalg.norm(step)) <= 1.0e-10:
                break
    radial_residual = np.abs(np.linalg.norm(samples - center, axis=1) - radius)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(radial_residual)):
        return {"available": False, "reason": "fixed-radius sphere fit diverged"}
    return {
        "available": True,
        "center": center,
        "radius_m": radius,
        "rms_residual_m": float(np.sqrt(np.mean(radial_residual**2))),
        "max_residual_m": float(np.max(radial_residual)),
        "radius_constrained": True,
    }


def _refine_spherical_cap(
    indices: np.ndarray,
    vertices: np.ndarray,
    *,
    minimum: int = 8,
) -> np.ndarray:
    if len(indices) <= minimum:
        return indices
    fit = fit_sphere_v7(vertices[indices])
    if not fit["available"]:
        return indices
    residual = np.abs(
        np.linalg.norm(vertices[indices] - fit["center"], axis=1)
        - float(fit["radius_m"])
    )
    cutoff = float(np.quantile(residual, 0.80))
    selected = indices[residual <= cutoff]
    return selected if len(selected) >= minimum else indices


def _connected_subset(
    indices: np.ndarray,
    faces: np.ndarray,
    *,
    seed: int | None = None,
    minimum: int = 3,
) -> np.ndarray:
    """Keep a topology-connected material patch instead of a nearest-point cloud."""
    candidates = np.unique(np.asarray(indices, dtype=np.int64))
    if len(candidates) <= minimum:
        return candidates
    membership = set(int(value) for value in candidates)
    adjacency: dict[int, set[int]] = {value: set() for value in membership}
    for triangle in np.asarray(faces, dtype=np.int64):
        inside = [int(value) for value in triangle if int(value) in membership]
        for first in inside:
            adjacency[first].update(value for value in inside if value != first)
    components: list[list[int]] = []
    unseen = set(membership)
    while unseen:
        start = unseen.pop()
        component = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            neighbours = adjacency[current].intersection(unseen)
            unseen.difference_update(neighbours)
            component.extend(neighbours)
            stack.extend(neighbours)
        components.append(component)
    selected: list[int] | None = None
    if seed is not None:
        selected = next(
            (component for component in components if int(seed) in component),
            None,
        )
    if selected is None or len(selected) < minimum:
        selected = max(components, key=len)
    return (
        np.asarray(sorted(selected), dtype=np.int64)
        if len(selected) >= minimum
        else candidates
    )


def _closest_subset(
    indices: np.ndarray,
    vertices: np.ndarray,
    target: np.ndarray,
    *,
    fraction: float,
    minimum: int,
    maximum: int = 512,
) -> np.ndarray:
    if not len(indices):
        return indices
    count = min(len(indices), maximum, max(minimum, int(np.ceil(fraction * len(indices)))))
    distance = _nearest_distance(vertices[indices], target)
    order = np.argpartition(distance, count - 1)[:count]
    return indices[order]


def build_joint_material_domains_v7(
    *,
    source_bind_vertices: np.ndarray,
    registration_vertices: np.ndarray | None,
    faces: np.ndarray,
    source_mesh_names: Sequence[str],
    source_vertex_ranges: np.ndarray,
    source_tissues: Sequence[str] | None = None,
    source_sides: Sequence[str] | None = None,
    joint_hints: Mapping[str, np.ndarray] | None = None,
    source_bone_names: Sequence[str] | None = None,
    source_bone_head: np.ndarray | None = None,
) -> FrozenJointMaterialDomainsV7:
    """Freeze hip/knee/patellofemoral domains on immutable source vertex ids.

    ``registration_vertices`` may supply the pre-armature geometry used for
    selection, but it must have exactly the source-bind topology.  The returned
    ids remain valid for every beta and pose derived from that topology.
    """
    source = _vertices(source_bind_vertices, "source_bind_vertices")
    reference = (
        source
        if registration_vertices is None
        else _vertices(registration_vertices, "registration_vertices")
    )
    if reference.shape != source.shape:
        raise ValueError("source bind and registration vertices must share topology")
    triangles = _faces(faces, len(source))
    ranges = np.asarray(source_vertex_ranges, dtype=np.int64)
    if ranges.shape != (len(source_mesh_names), 2):
        raise ValueError("source_vertex_ranges must be [mesh_count,2]")
    if source_tissues is not None and len(source_tissues) != len(source_mesh_names):
        raise ValueError("source_tissues length does not match source_mesh_names")
    if source_sides is not None and len(source_sides) != len(source_mesh_names):
        raise ValueError("source_sides length does not match source_mesh_names")
    hints = {
        str(name): np.asarray(value, dtype=np.float64).reshape(3)
        for name, value in dict(joint_hints or {}).items()
    }
    if source_bone_names is not None or source_bone_head is not None:
        if source_bone_names is None or source_bone_head is None:
            raise ValueError(
                "source_bone_names and source_bone_head must be provided together"
            )
        bone_heads = np.asarray(source_bone_head, dtype=np.float64)
        if bone_heads.shape != (len(source_bone_names), 3):
            raise ValueError("source_bone_head must be [source_bone_count,3]")
        bone_lookup = {
            str(name).strip().lower(): bone_heads[index]
            for index, name in enumerate(source_bone_names)
        }
        for side, suffix in (("left", "l"), ("right", "r")):
            pivot = bone_lookup.get(f"femur_rot_{suffix}")
            if pivot is not None:
                hints.setdefault(f"{side}_femoral_pivot", pivot)

    pelvis = _mesh_indices(
        mesh_names=source_mesh_names,
        vertex_ranges=ranges,
        tissues=source_tissues,
        sides=source_sides,
        tokens=("ilium", "ischium", "pubis", "acetabul", "pelvis"),
        side=None,
        allow_midline=True,
    )
    if len(pelvis) < 8:
        raise ValueError("cannot build hip domains without a pelvis surface")

    if "left_hip" in hints and "right_hip" in hints:
        lateral_axis = hints["right_hip"] - hints["left_hip"]
        lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1.0e-12)
    else:
        lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    domains: dict[str, np.ndarray] = {}
    for side in SIDES:
        femur = _mesh_indices(
            mesh_names=source_mesh_names,
            vertex_ranges=ranges,
            tissues=source_tissues,
            sides=source_sides,
            tokens=("femur",),
            side=side,
        )
        tibia = _mesh_indices(
            mesh_names=source_mesh_names,
            vertex_ranges=ranges,
            tissues=source_tissues,
            sides=source_sides,
            tokens=("tibia",),
            side=side,
        )
        patella = _mesh_indices(
            mesh_names=source_mesh_names,
            vertex_ranges=ranges,
            tissues=source_tissues,
            sides=source_sides,
            tokens=("patella",),
            side=side,
        )
        if min(len(femur), len(tibia), len(patella)) < 8:
            raise ValueError(
                f"cannot build {side} leg domains without separate femur, "
                "tibia, and patella surfaces"
            )

        femur_low, femur_high = _endpoint_caps(femur, reference)
        hip_target = (
            hints.get(f"{side}_hip", reference[pelvis])
            if f"{side}_hip" in hints
            else reference[pelvis]
        )
        hip_target = np.asarray(hip_target, dtype=np.float64).reshape(-1, 3)
        head_cap, condyle_cap = _choose_near(
            femur_low, femur_high, reference, hip_target
        )
        # V71's authored Femur_Rot head is a topology/pivot prior, not an
        # acceptance measurement.  Select a radial material shell around that
        # immutable prior, then fit its centre from surface vertices.  This
        # avoids the old "top 15% of the fitted femur" probe whose neck-heavy
        # cap moved the reported centre by 12--15 mm.
        femoral_pivot = hints.get(f"{side}_femoral_pivot")
        if femoral_pivot is not None:
            radial = np.linalg.norm(
                reference[femur] - np.asarray(femoral_pivot)[None, :], axis=1
            )
            cutoff = float(np.quantile(radial, 0.15))
            femoral_head = femur[radial <= cutoff]
        else:
            femoral_head = _refine_spherical_cap(head_cap, reference)
        head_seed = int(
            femoral_head[
                int(np.argmin(_nearest_distance(reference[femoral_head], hip_target)))
            ]
        )
        femoral_head = _connected_subset(
            femoral_head, triangles, seed=head_seed, minimum=4
        )
        head_fit = fit_sphere_v7(reference[femoral_head])
        head_center = (
            np.asarray(head_fit["center"], dtype=np.float64)
            if head_fit["available"]
            else np.mean(reference[femoral_head], axis=0)
        )
        head_radius = (
            float(head_fit["radius_m"])
            if head_fit["available"]
            else float(np.median(np.linalg.norm(reference[femoral_head] - head_center, axis=1)))
        )

        pelvis_distance_to_center = np.linalg.norm(
            reference[pelvis] - head_center, axis=1
        )
        socket_score = np.abs(pelvis_distance_to_center - head_radius)
        # Freeze only the articular radial shell.  A broad percentage of the
        # whole ilium includes non-socket surfaces and makes a shallow-cap
        # sphere fit underdetermined.  The adaptive count keeps the domain
        # usable across source resolutions without reselecting after refit.
        socket_count = min(
            len(pelvis),
            128,
            max(24, int(np.ceil(0.02 * len(pelvis)))),
        )
        acetabulum = pelvis[
            np.argpartition(socket_score, socket_count - 1)[:socket_count]
        ]
        socket_seed = int(pelvis[int(np.argmin(socket_score))])
        acetabulum = _connected_subset(
            acetabulum, triangles, seed=socket_seed, minimum=4
        )

        condyle_cap = _refine_spherical_cap(condyle_cap, reference)
        medial_condyle, lateral_condyle = _split_medial_lateral(
            condyle_cap,
            reference,
            side=side,
            lateral_axis=lateral_axis,
        )
        condyle_points = reference[
            np.concatenate((medial_condyle, lateral_condyle))
        ]
        tibia_low, tibia_high = _endpoint_caps(tibia, reference, fraction=0.18)
        plateau_cap, _tibia_distal = _choose_near(
            tibia_low, tibia_high, reference, condyle_points
        )
        medial_plateau, lateral_plateau = _split_medial_lateral(
            plateau_cap,
            reference,
            side=side,
            lateral_axis=lateral_axis,
        )

        patella_articular = _closest_subset(
            patella,
            reference,
            condyle_points,
            fraction=0.50,
            minimum=4,
            maximum=256,
        )
        trochlea = _closest_subset(
            condyle_cap,
            reference,
            reference[patella_articular],
            fraction=0.50,
            minimum=4,
            maximum=256,
        )
        side_pelvis = _closest_subset(
            pelvis,
            reference,
            reference[acetabulum],
            fraction=0.25,
            minimum=len(acetabulum),
            maximum=max(512, len(acetabulum)),
        )

        domains.update(
            {
                f"{side}/pelvis": side_pelvis,
                f"{side}/femur": femur,
                f"{side}/femoral_head": femoral_head,
                f"{side}/acetabulum": acetabulum,
                f"{side}/femoral_condyle_medial": medial_condyle,
                f"{side}/femoral_condyle_lateral": lateral_condyle,
                f"{side}/tibia": tibia,
                f"{side}/tibial_plateau_medial": medial_plateau,
                f"{side}/tibial_plateau_lateral": lateral_plateau,
                # Patella is intentionally independent from tibia/fibula.
                f"{side}/patella": patella,
                f"{side}/patella_articular": patella_articular,
                f"{side}/trochlea": trochlea,
            }
        )

    return FrozenJointMaterialDomainsV7.freeze(
        source_bind_vertices=source,
        faces=triangles,
        domains=domains,
    )


def _serialise_sphere(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if result.get("center") is not None:
        result["center"] = np.asarray(result["center"], dtype=np.float64).tolist()
    return result


def _clearance_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    distances = np.concatenate((_nearest_distance(a, b), _nearest_distance(b, a)))
    return {
        "min_m": float(np.min(distances)),
        "median_m": float(np.median(distances)),
        "q95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(np.max(distances)),
    }


def _domain_edges(faces: np.ndarray, indices: np.ndarray) -> np.ndarray:
    largest_index = max(
        int(np.max(faces)) if faces.size else -1,
        int(np.max(indices)) if len(indices) else -1,
    )
    membership = np.zeros(largest_index + 1, dtype=bool)
    if not len(membership):
        return np.empty((0, 2), dtype=np.int64)
    membership[indices] = True
    selected = faces[np.all(membership[faces], axis=1)]
    if not len(selected):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate(
        (selected[:, (0, 1)], selected[:, (1, 2)], selected[:, (2, 0)]),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def rigid_edge_metrics_v7(
    *,
    reference_vertices: np.ndarray,
    final_vertices: np.ndarray,
    faces: np.ndarray,
    indices: np.ndarray,
    thresholds: JointContactThresholdsV7 | None = None,
) -> dict[str, Any]:
    limits = thresholds or JointContactThresholdsV7()
    reference = _vertices(reference_vertices, "reference_vertices")
    final = _vertices(final_vertices, "final_vertices")
    if reference.shape != final.shape:
        raise ValueError("reference and final vertices must share topology")
    triangles = _faces(faces, len(reference))
    edges = _domain_edges(triangles, np.asarray(indices, dtype=np.int64))
    if not len(edges):
        return {"available": False, "reason": "material domain contains no complete faces", "pass": False}
    rest_length = np.linalg.norm(reference[edges[:, 1]] - reference[edges[:, 0]], axis=1)
    valid = rest_length > 1.0e-10
    if not np.any(valid):
        return {"available": False, "reason": "material domain edges are degenerate", "pass": False}
    edges = edges[valid]
    rest_length = rest_length[valid]
    final_length = np.linalg.norm(final[edges[:, 1]] - final[edges[:, 0]], axis=1)
    ratio = final_length / rest_length
    q01, q99 = np.quantile(ratio, (0.01, 0.99))
    minimum, maximum = float(np.min(ratio)), float(np.max(ratio))
    passed = bool(
        q01 >= limits.rigid_q01_ratio
        and q99 <= limits.rigid_q99_ratio
        and minimum >= limits.rigid_min_ratio
        and maximum <= limits.rigid_max_ratio
    )
    return {
        "available": True,
        "edge_count": int(len(edges)),
        "ratio_min": minimum,
        "ratio_q01": float(q01),
        "ratio_median": float(np.median(ratio)),
        "ratio_q99": float(q99),
        "ratio_max": maximum,
        "pass": passed,
    }


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(3)
    b = np.asarray(second, dtype=np.float64).reshape(3)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1.0e-12 and norm_b <= 1.0e-12:
        return 0.0
    denominator = norm_a * norm_b
    if denominator <= 1.0e-12:
        return float("inf")
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def patellofemoral_trajectory_metrics_v7(
    domains: FrozenJointMaterialDomainsV7,
    *,
    posed_vertices: np.ndarray,
    oracle_vertices: np.ndarray,
    faces: np.ndarray,
    side: str,
    thresholds: JointContactThresholdsV7 | None = None,
) -> dict[str, Any]:
    """Compare patella motion relative to the trochlea over an entire sweep."""
    limits = thresholds or JointContactThresholdsV7()
    posed = np.asarray(posed_vertices, dtype=np.float64)
    oracle = np.asarray(oracle_vertices, dtype=np.float64)
    if posed.ndim != 3 or posed.shape[2] != 3 or posed.shape != oracle.shape:
        raise ValueError("posed_vertices and oracle_vertices must be matching [P,N,3]")
    if len(posed) < 2:
        return {"available": False, "reason": "trajectory needs at least two poses", "pass": False}
    domains.validate_topology(posed[0], faces)
    patella = domains.require(f"{side}/patella")
    trochlea = domains.require(f"{side}/trochlea")

    def offsets(sequence: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.mean(frame[patella], axis=0) - np.mean(frame[trochlea], axis=0)
                for frame in sequence
            ],
            axis=0,
        )

    posed_offset = offsets(posed)
    oracle_offset = offsets(oracle)
    # Compare motion, not beta-dependent absolute rest offset.
    posed_motion = posed_offset - posed_offset[0]
    oracle_motion = oracle_offset - oracle_offset[0]
    error = np.linalg.norm(posed_motion - oracle_motion, axis=1)
    direction_error = _angle_deg(posed_motion[-1], oracle_motion[-1])
    gaps = np.asarray(
        [
            _clearance_summary(frame[patella], frame[trochlea])["min_m"]
            for frame in posed
        ],
        dtype=np.float64,
    )
    gap_drift = float(np.max(np.abs(gaps - gaps[0])))
    rms = float(np.sqrt(np.mean(error**2)))
    maximum = float(np.max(error))
    passed = bool(
        np.all(gaps >= limits.patellofemoral_gap_min_m - 1.0e-12)
        and np.all(gaps <= limits.patellofemoral_gap_max_m)
        and gap_drift <= limits.patellofemoral_gap_drift_m
        and rms <= limits.patella_trajectory_rms_m
        and maximum <= limits.patella_trajectory_max_m
        and direction_error <= limits.patella_trajectory_direction_deg
    )
    return {
        "available": True,
        "pose_count": int(len(posed)),
        "gap_min_m": float(np.min(gaps)),
        "gap_max_m": float(np.max(gaps)),
        "gap_drift_m": gap_drift,
        "trajectory_rms_m": rms,
        "trajectory_max_m": maximum,
        "trajectory_direction_error_deg": direction_error,
        "pass": passed,
    }


def _observation_metric(
    observation: Mapping[str, Any],
    aliases: Sequence[str],
) -> float | None:
    for name in aliases:
        if name not in observation:
            continue
        try:
            value = float(observation[name])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None
    return None


def evaluate_controller_gate_v7(
    observations: Mapping[str, Mapping[str, Any]] | None,
    *,
    thresholds: JointContactThresholdsV7 | None = None,
    required: Sequence[str] = REQUIRED_CONTROLLER_JOINTS,
) -> dict[str, Any]:
    """Evaluate raw controller errors; an incoming ``pass`` flag is ignored."""
    limits = thresholds or JointContactThresholdsV7()
    observations = observations or {}
    items: dict[str, Any] = {}
    failures: list[str] = []
    for name in required:
        observation = observations.get(name, {})
        translation = _observation_metric(
            observation,
            ("translation_error_m", "pivot_error_m", "anchor_error_m"),
        )
        rotation = _observation_metric(
            observation,
            ("rotation_error_deg", "axis_error_deg"),
        )
        passed = bool(
            translation is not None
            and rotation is not None
            and translation <= limits.controller_translation_error_m
            and rotation <= limits.controller_rotation_error_deg
        )
        items[name] = {
            "translation_error_m": translation,
            "rotation_error_deg": rotation,
            "pass": passed,
        }
        if not passed:
            failures.append(name)
    return {"items": items, "failures": failures, "pass": not failures}


def evaluate_local_fk_gate_v7(
    observations: Mapping[str, Mapping[str, Any]] | None,
    *,
    thresholds: JointContactThresholdsV7 | None = None,
    required: Sequence[str] = REQUIRED_LOCAL_FK_LINKS,
) -> dict[str, Any]:
    """Evaluate measured child-local frame errors, never a precomputed flag."""
    limits = thresholds or JointContactThresholdsV7()
    observations = observations or {}
    items: dict[str, Any] = {}
    failures: list[str] = []
    for name in required:
        observation = observations.get(name, {})
        translation = _observation_metric(
            observation,
            ("translation_error_m", "local_translation_error_m"),
        )
        rotation = _observation_metric(
            observation,
            ("rotation_error_deg", "local_rotation_error_deg"),
        )
        passed = bool(
            translation is not None
            and rotation is not None
            and translation <= limits.local_fk_translation_error_m
            and rotation <= limits.local_fk_rotation_error_deg
        )
        items[name] = {
            "translation_error_m": translation,
            "rotation_error_deg": rotation,
            "pass": passed,
        }
        if not passed:
            failures.append(name)
    return {"items": items, "failures": failures, "pass": not failures}


def _hip_metrics(
    domains: FrozenJointMaterialDomainsV7,
    *,
    reference: np.ndarray,
    final: np.ndarray,
    side: str,
    limits: JointContactThresholdsV7,
) -> dict[str, Any]:
    head = domains.require(f"{side}/femoral_head")
    socket = domains.require(f"{side}/acetabulum")
    rest_head = fit_sphere_v7(reference[head])
    rest_socket = fit_sphere_fixed_radius_v7(
        reference[socket],
        radius_m=float(rest_head["radius_m"])
        if rest_head["available"]
        else float("nan"),
    )
    final_head = fit_sphere_v7(final[head])
    pelvis = domains.require(f"{side}/pelvis")
    reference_pelvis = reference[pelvis]
    final_pelvis = final[pelvis]
    reference_center = np.mean(reference_pelvis, axis=0)
    final_center = np.mean(final_pelvis, axis=0)
    covariance = (reference_pelvis - reference_center).T @ (
        final_pelvis - final_center
    )
    u, _singular, vt = np.linalg.svd(covariance)
    pelvis_rotation = vt.T @ u.T
    if float(np.linalg.det(pelvis_rotation)) < 0.0:
        vt[-1] *= -1.0
        pelvis_rotation = vt.T @ u.T
    predicted_socket_center = (
        pelvis_rotation
        @ (
            np.asarray(rest_socket["center"], dtype=np.float64)
            - reference_center
        )
        + final_center
        if rest_socket["available"]
        else None
    )
    final_socket = fit_sphere_fixed_radius_v7(
        final[socket],
        radius_m=float(final_head["radius_m"])
        if final_head["available"]
        else float("nan"),
        initial_center=predicted_socket_center,
    )
    fits_available = all(
        value["available"]
        for value in (rest_head, rest_socket, final_head, final_socket)
    )
    if not fits_available:
        return {
            "available": False,
            "reason": "a fixed femoral-head or acetabulum sphere fit failed",
            "sphere_fits": {
                "reference_head": _serialise_sphere(rest_head),
                "reference_socket": _serialise_sphere(rest_socket),
                "final_head": _serialise_sphere(final_head),
                "final_socket": _serialise_sphere(final_socket),
            },
            "pass": False,
        }
    rest_delta = np.asarray(rest_head["center"]) - np.asarray(rest_socket["center"])
    final_delta = np.asarray(final_head["center"]) - np.asarray(final_socket["center"])
    center_error = float(np.linalg.norm(final_delta))
    center_drift = abs(float(np.linalg.norm(final_delta)) - float(np.linalg.norm(rest_delta)))
    rest_radius = float(rest_head["radius_m"])
    final_radius = float(final_head["radius_m"])
    radius_change = abs(final_radius - rest_radius)
    radius_relative_change = radius_change / max(rest_radius, 1.0e-12)
    rest_clearance = _clearance_summary(reference[head], reference[socket])
    final_clearance = _clearance_summary(final[head], final[socket])
    median_change = abs(final_clearance["median_m"] - rest_clearance["median_m"])
    q95_change = abs(final_clearance["q95_m"] - rest_clearance["q95_m"])
    passed = bool(
        center_error <= limits.hip_center_error_m
        and center_drift <= limits.hip_center_drift_m
        and (
            radius_relative_change <= limits.hip_radius_relative_change
            or radius_change <= limits.hip_radius_absolute_change_m
        )
        and median_change <= limits.hip_clearance_median_change_m
        and q95_change <= limits.hip_clearance_q95_change_m
        and final_clearance["max_m"] <= (
            rest_clearance["max_m"] + limits.hip_max_separation_m
        )
    )
    return {
        "available": True,
        "sphere_fits": {
            "reference_head": _serialise_sphere(rest_head),
            "reference_socket": _serialise_sphere(rest_socket),
            "final_head": _serialise_sphere(final_head),
            "final_socket": _serialise_sphere(final_socket),
        },
        "center_error_m": center_error,
        "center_drift_m": center_drift,
        "radius_change_m": radius_change,
        "radius_relative_change": radius_relative_change,
        "reference_clearance": rest_clearance,
        "final_clearance": final_clearance,
        "clearance_median_change_m": median_change,
        "clearance_q95_change_m": q95_change,
        "pass": passed,
    }


def _knee_metrics(
    domains: FrozenJointMaterialDomainsV7,
    *,
    reference: np.ndarray,
    final: np.ndarray,
    side: str,
    limits: JointContactThresholdsV7,
) -> dict[str, Any]:
    interfaces: dict[str, Any] = {}
    interface_pass = True
    for compartment in ("medial", "lateral"):
        condyle = domains.require(f"{side}/femoral_condyle_{compartment}")
        plateau = domains.require(f"{side}/tibial_plateau_{compartment}")
        rest = _clearance_summary(reference[condyle], reference[plateau])
        posed = _clearance_summary(final[condyle], final[plateau])
        change = abs(posed["min_m"] - rest["min_m"])
        passed = bool(
            limits.knee_gap_min_m - 1.0e-12
            <= posed["min_m"]
            <= limits.knee_gap_max_m
            and change <= limits.knee_gap_change_m
        )
        interfaces[compartment] = {
            "reference_clearance": rest,
            "final_clearance": posed,
            "gap_change_m": change,
            "pass": passed,
        }
        interface_pass = interface_pass and passed

    head_rest = fit_sphere_v7(reference[domains.require(f"{side}/femoral_head")])
    head_final = fit_sphere_v7(final[domains.require(f"{side}/femoral_head")])
    condyle_indices = np.concatenate(
        (
            domains.require(f"{side}/femoral_condyle_medial"),
            domains.require(f"{side}/femoral_condyle_lateral"),
        )
    )
    if head_rest["available"] and head_final["available"]:
        rest_length = float(
            np.linalg.norm(np.asarray(head_rest["center"]) - np.mean(reference[condyle_indices], axis=0))
        )
        final_length = float(
            np.linalg.norm(np.asarray(head_final["center"]) - np.mean(final[condyle_indices], axis=0))
        )
        length_change = abs(final_length - rest_length)
    else:
        rest_length = final_length = length_change = float("inf")
    passed = bool(interface_pass and length_change <= limits.femur_length_change_m)
    return {
        "available": bool(head_rest["available"] and head_final["available"]),
        "interfaces": interfaces,
        "reference_femur_length_m": rest_length,
        "final_femur_length_m": final_length,
        "femur_length_change_m": length_change,
        "pass": passed,
    }


def diagnose_joint_contact_geometry_v7(
    domains: FrozenJointMaterialDomainsV7,
    *,
    reference_vertices: np.ndarray,
    final_vertices: np.ndarray,
    faces: np.ndarray,
    trajectory_vertices: np.ndarray | None,
    oracle_trajectory_vertices: np.ndarray | None,
    thresholds: JointContactThresholdsV7 | None = None,
) -> dict[str, Any]:
    """Recompute every geometry metric from final vertices and fixed ids."""
    limits = thresholds or JointContactThresholdsV7()
    reference = _vertices(reference_vertices, "reference_vertices")
    final = _vertices(final_vertices, "final_vertices")
    if reference.shape != final.shape:
        raise ValueError("reference and final vertices must share topology")
    triangles = _faces(faces, len(reference))
    domains.validate_topology(final, triangles)

    hips: dict[str, Any] = {}
    knees: dict[str, Any] = {}
    patellofemoral: dict[str, Any] = {}
    rigidity: dict[str, Any] = {}
    failures: list[str] = []
    for side in SIDES:
        hips[side] = _hip_metrics(
            domains,
            reference=reference,
            final=final,
            side=side,
            limits=limits,
        )
        knees[side] = _knee_metrics(
            domains,
            reference=reference,
            final=final,
            side=side,
            limits=limits,
        )
        if trajectory_vertices is None or oracle_trajectory_vertices is None:
            patellofemoral[side] = {
                "available": False,
                "reason": "candidate and oracle trajectory sweeps are required",
                "pass": False,
            }
        else:
            patellofemoral[side] = patellofemoral_trajectory_metrics_v7(
                domains,
                posed_vertices=trajectory_vertices,
                oracle_vertices=oracle_trajectory_vertices,
                faces=triangles,
                side=side,
                thresholds=limits,
            )
        for structure in ("femur", "tibia", "patella"):
            key = f"{side}/{structure}"
            rigidity[key] = rigid_edge_metrics_v7(
                reference_vertices=reference,
                final_vertices=final,
                faces=triangles,
                indices=domains.require(key),
                thresholds=limits,
            )
        for label, item in (
            (f"hip/{side}", hips[side]),
            (f"knee/{side}", knees[side]),
            (f"patellofemoral/{side}", patellofemoral[side]),
        ):
            if not item["pass"]:
                failures.append(label)
        for structure in ("femur", "tibia", "patella"):
            if not rigidity[f"{side}/{structure}"]["pass"]:
                failures.append(f"rigidity/{side}/{structure}")
    return {
        "hips": hips,
        "knees": knees,
        "patellofemoral": patellofemoral,
        "rigidity": rigidity,
        "thresholds": limits.to_dict(),
        "failures": failures,
        "pass": not failures,
    }


def diagnose_joint_contact_v7(
    domains: FrozenJointMaterialDomainsV7,
    *,
    reference_vertices: np.ndarray,
    final_vertices: np.ndarray,
    faces: np.ndarray,
    controller_observations: Mapping[str, Mapping[str, Any]] | None,
    local_fk_observations: Mapping[str, Mapping[str, Any]] | None,
    trajectory_vertices: np.ndarray | None,
    oracle_trajectory_vertices: np.ndarray | None,
    thresholds: JointContactThresholdsV7 | None = None,
) -> dict[str, Any]:
    """Return the strict controller AND local-FK AND geometry result."""
    limits = thresholds or JointContactThresholdsV7()
    controller = evaluate_controller_gate_v7(
        controller_observations, thresholds=limits
    )
    local_fk = evaluate_local_fk_gate_v7(local_fk_observations, thresholds=limits)
    geometry = diagnose_joint_contact_geometry_v7(
        domains,
        reference_vertices=reference_vertices,
        final_vertices=final_vertices,
        faces=faces,
        trajectory_vertices=trajectory_vertices,
        oracle_trajectory_vertices=oracle_trajectory_vertices,
        thresholds=limits,
    )
    gates = {
        "controller": controller,
        "local_fk": local_fk,
        "geometry": geometry,
    }
    failures = [name for name, gate in gates.items() if not gate["pass"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "domain_topology_digest": domains.topology_digest,
        "gates": gates,
        "pass_requires": ["controller", "local_fk", "geometry"],
        "failures": failures,
        "passed": not failures,
    }


__all__ = [
    "FrozenJointMaterialDomainsV7",
    "JointContactThresholdsV7",
    "REQUIRED_CONTROLLER_JOINTS",
    "REQUIRED_LOCAL_FK_LINKS",
    "build_joint_material_domains_v7",
    "diagnose_joint_contact_geometry_v7",
    "diagnose_joint_contact_v7",
    "evaluate_controller_gate_v7",
    "evaluate_local_fk_gate_v7",
    "fit_sphere_fixed_radius_v7",
    "fit_sphere_v7",
    "patellofemoral_trajectory_metrics_v7",
    "rigid_edge_metrics_v7",
]
