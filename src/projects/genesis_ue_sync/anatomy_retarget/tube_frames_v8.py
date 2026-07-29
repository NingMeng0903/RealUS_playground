"""Frozen V71-compatible matrix-LBS coupling for vessels and nerves.

V8 deliberately starts with the narrow mechanism that can be verified against
the authored Blender rig: the original fourteen sparse Armature influences and
the 235 source-bone skinning matrices.  It does not claim to implement
centreline stations, cross-section reconstruction, or parallel transport.

The pack is built per subject rest asset.  Runtime recomputes the tube vertices
from the frozen rest coordinates and weights, then overwrites any earlier LBS,
soft-follow, or other provisional result for exactly the frozen material
domain.  No pose-time topology query or geometry solve is performed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


TUBE_COUPLING_BACKEND_V8 = "strict_matrix_lbs_14slot_v8"
TUBE_TISSUES_V8 = frozenset(("vessel", "nerve"))
SOURCE_BONE_COUNT_V8 = 235
INFLUENCE_SLOTS_V8 = 14
_RUNTIME_PREFIX_V8 = "tube_coupling_v8."
_RUNTIME_FIELD_NAMES_V8 = frozenset(
    (
        "schema_version",
        "artifact_kind",
        "vertex_count",
        "vertex_ids",
        "rest_vertices_m",
        "driver_indices",
        "driver_weights",
        "material_edges",
        "mesh_indices",
        "mesh_vertex_ranges",
        "topology_digest",
        "domain_digest",
        "rest_digest",
        "weight_digest",
        "source_bone_count",
        "influence_slots",
        "backend",
        "content_digest",
    )
)


def _sha256_arrays(namespace: bytes, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(namespace + b"\0")
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def tube_topology_digest_v8(vertex_count: int, faces: np.ndarray) -> str:
    """Digest the immutable global triangle topology."""
    triangles = np.ascontiguousarray(
        np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    )
    return _sha256_arrays(
        b"anatomy-tube-topology-v8",
        np.asarray((int(vertex_count),), dtype=np.int64),
        triangles,
    )


def _has_v7_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "tube_frame_v7" in str(key).lower() or _has_v7_marker(child):
                return True
        return False
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind in {"U", "S"}:
            return any(
                "tube_frame_v7" in str(item).lower()
                for item in array.reshape(-1).tolist()
            )
        if array.dtype.kind == "O":
            return any(_has_v7_marker(item) for item in array.reshape(-1))
        return False
    if isinstance(value, (tuple, list)):
        return any(_has_v7_marker(child) for child in value)
    return isinstance(value, str) and "tube_frame_v7" in value.lower()


def reject_v7_tube_fields_v8(
    asset: AnatomyRiggedAsset,
    runtime_fields: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed if a V7 material-frame override is still reachable."""
    if _has_v7_marker(asset.metadata or {}) or _has_v7_marker(runtime_fields or {}):
        raise ValueError("V8 rejects all tube_frame_v7 fields")


def _require_full_local_fk(asset: AnatomyRiggedAsset) -> None:
    if (asset.metadata or {}).get("source_full_local_fk_v2") is not True:
        raise ValueError("V8 tube coupling requires source_full_local_fk_v2=true")


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int32)
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0).astype(np.int32)


@dataclass(frozen=True)
class _TubeDomainV8:
    vertex_ids: np.ndarray
    material_edges: np.ndarray
    mesh_indices: np.ndarray
    mesh_vertex_ranges: np.ndarray
    digest: str


def _tube_domain_v8(asset: AnatomyRiggedAsset) -> _TubeDomainV8:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = list(asset.source_tissues or [])
    mesh_names = list(asset.source_mesh_names)
    if ranges.shape != (len(mesh_names), 2) or len(tissues) != len(mesh_names):
        raise ValueError("V8 tube coupling requires complete source mesh semantics")

    vertex_count = int(len(asset.vertices_rest))
    cursor = 0
    for start, stop in ranges.tolist():
        if int(start) != cursor or int(stop) < int(start):
            raise ValueError("source vertex ranges must be ordered and contiguous")
        cursor = int(stop)
    if cursor != vertex_count:
        raise ValueError("source vertex ranges must cover vertices_rest")

    selected_meshes: list[int] = []
    selected_vertices: list[np.ndarray] = []
    selected_edges: list[np.ndarray] = []
    selected_ranges: list[tuple[int, int]] = []
    tube_cursor = 0
    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    for mesh_index, ((start, stop), tissue) in enumerate(zip(ranges, tissues)):
        if str(tissue).strip().lower() not in TUBE_TISSUES_V8:
            continue
        start_i, stop_i = int(start), int(stop)
        if stop_i <= start_i:
            raise ValueError("vessel/nerve source mesh may not be empty")
        ids = np.arange(start_i, stop_i, dtype=np.int32)
        local_faces = faces[
            np.all((faces >= start_i) & (faces < stop_i), axis=1)
        ] - start_i
        edges = _unique_edges(local_faces)
        if len(edges):
            selected_edges.append(edges + tube_cursor)
        selected_meshes.append(mesh_index)
        selected_vertices.append(ids)
        selected_ranges.append((tube_cursor, tube_cursor + len(ids)))
        tube_cursor += len(ids)

    if not selected_vertices:
        raise ValueError("asset contains no vessel or nerve material")
    vertex_ids = np.concatenate(selected_vertices).astype(np.int32)
    material_edges = (
        np.concatenate(selected_edges).astype(np.int32)
        if selected_edges
        else np.empty((0, 2), dtype=np.int32)
    )
    mesh_indices = np.asarray(selected_meshes, dtype=np.int32)
    mesh_ranges = np.asarray(selected_ranges, dtype=np.int32)
    semantic_bytes = "\0".join(
        f"{mesh_names[index]}\0{str(tissues[index]).strip().lower()}"
        for index in selected_meshes
    ).encode("utf-8")
    digest = hashlib.sha256(b"anatomy-tube-domain-v8\0")
    digest.update(semantic_bytes)
    for value in (ranges, mesh_indices, mesh_ranges, vertex_ids, material_edges):
        array = np.ascontiguousarray(value)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return _TubeDomainV8(
        vertex_ids=vertex_ids,
        material_edges=material_edges,
        mesh_indices=mesh_indices,
        mesh_vertex_ranges=mesh_ranges,
        digest=digest.hexdigest(),
    )


def _weight_digest(
    vertex_ids: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> str:
    return _sha256_arrays(
        b"anatomy-tube-authored-weights-v8", vertex_ids, indices, weights
    )


def _rest_digest(vertex_ids: np.ndarray, vertices: np.ndarray) -> str:
    return _sha256_arrays(
        b"anatomy-tube-subject-rest-v8", vertex_ids, vertices
    )


def _readonly_copy(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TubeCouplingPackV8:
    """Content-bound runtime data for one subject's vessel/nerve domain."""

    vertex_count: int
    vertex_ids: np.ndarray
    rest_vertices_m: np.ndarray
    driver_indices: np.ndarray
    driver_weights: np.ndarray
    material_edges: np.ndarray
    mesh_indices: np.ndarray
    mesh_vertex_ranges: np.ndarray
    topology_digest: str
    domain_digest: str
    rest_digest: str
    weight_digest: str
    source_bone_count: int = SOURCE_BONE_COUNT_V8
    influence_slots: int = INFLUENCE_SLOTS_V8
    backend: str = TUBE_COUPLING_BACKEND_V8

    def __post_init__(self) -> None:
        for name, dtype in (
            ("vertex_ids", np.int32),
            ("rest_vertices_m", np.float32),
            ("driver_indices", np.int16),
            ("driver_weights", np.float32),
            ("material_edges", np.int32),
            ("mesh_indices", np.int32),
            ("mesh_vertex_ranges", np.int32),
        ):
            object.__setattr__(
                self, name, _readonly_copy(getattr(self, name), dtype)
            )
        self.validate()

    def validate(self) -> None:
        ids = np.asarray(self.vertex_ids, dtype=np.int64).reshape(-1)
        rest = np.asarray(self.rest_vertices_m)
        indices = np.asarray(self.driver_indices)
        weights = np.asarray(self.driver_weights)
        edges = np.asarray(self.material_edges)
        mesh_indices = np.asarray(self.mesh_indices)
        mesh_ranges = np.asarray(self.mesh_vertex_ranges)
        if int(self.vertex_count) <= 0:
            raise ValueError("tube pack requires a positive global vertex count")
        if (
            not len(ids)
            or np.any(ids < 0)
            or np.any(ids >= int(self.vertex_count))
            or not np.array_equal(ids, np.unique(ids))
        ):
            raise ValueError("tube pack vertex_ids must be unique and ordered")
        if rest.shape != (len(ids), 3) or not np.all(np.isfinite(rest)):
            raise ValueError("tube pack rest vertices are invalid")
        expected = (len(ids), INFLUENCE_SLOTS_V8)
        if indices.shape != expected or weights.shape != expected:
            raise ValueError("V8 requires exactly 14 sparse Armature slots")
        if int(self.influence_slots) != INFLUENCE_SLOTS_V8:
            raise ValueError("V8 influence slot count is immutable at 14")
        if int(self.source_bone_count) != SOURCE_BONE_COUNT_V8:
            raise ValueError("V8 source bone count is immutable at 235")
        if (
            np.any(indices < 0)
            or np.any(indices >= SOURCE_BONE_COUNT_V8)
            or np.any(weights < 0.0)
            or not np.all(np.isfinite(weights))
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("tube pack contains invalid authored weights")
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise ValueError("tube pack material_edges must be [E,2]")
        if edges.size and (np.any(edges < 0) or np.any(edges >= len(ids))):
            raise ValueError("tube pack material edge is outside the frozen domain")
        if (
            mesh_indices.ndim != 1
            or mesh_ranges.shape != (len(mesh_indices), 2)
            or not len(mesh_indices)
            or int(mesh_ranges[0, 0]) != 0
            or int(mesh_ranges[-1, 1]) != len(ids)
            or np.any(mesh_ranges[:, 1] <= mesh_ranges[:, 0])
            or (
                len(mesh_ranges) > 1
                and not np.array_equal(mesh_ranges[:-1, 1], mesh_ranges[1:, 0])
            )
        ):
            raise ValueError("tube pack mesh ranges must partition the frozen domain")
        if self.backend != TUBE_COUPLING_BACKEND_V8:
            raise ValueError("unknown V8 tube coupling backend")
        for name in (
            "topology_digest",
            "domain_digest",
            "rest_digest",
            "weight_digest",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if _weight_digest(
            self.vertex_ids, self.driver_indices, self.driver_weights
        ) != str(self.weight_digest):
            raise ValueError("tube pack weight digest mismatch")
        if _rest_digest(
            self.vertex_ids, self.rest_vertices_m
        ) != str(self.rest_digest):
            raise ValueError("tube pack rest digest mismatch")

    def content_digest(self) -> str:
        digest = hashlib.sha256(b"anatomy-tube-coupling-pack-v8\0")
        digest.update(
            np.asarray(
                (
                    int(self.vertex_count),
                    int(self.source_bone_count),
                    int(self.influence_slots),
                ),
                dtype=np.int64,
            ).tobytes()
        )
        for value in (
            self.topology_digest,
            self.domain_digest,
            self.rest_digest,
            self.weight_digest,
            self.backend,
        ):
            digest.update(str(value).encode("ascii"))
            digest.update(b"\0")
        for value in (
            self.vertex_ids,
            self.rest_vertices_m,
            self.driver_indices,
            self.driver_weights,
            self.material_edges,
            self.mesh_indices,
            self.mesh_vertex_ranges,
        ):
            array = np.ascontiguousarray(value)
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes())
        return digest.hexdigest()


def _ascii_u8(value: str) -> np.ndarray:
    try:
        encoded = str(value).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("V8 runtime metadata must be ASCII") from exc
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def _decode_ascii_u8(
    fields: Mapping[str, np.ndarray],
    name: str,
    *,
    length: int | None = None,
) -> str:
    key = f"{_RUNTIME_PREFIX_V8}{name}"
    array = np.asarray(fields[key])
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional uint8 array")
    if length is not None and len(array) != int(length):
        raise ValueError(f"{key} has an invalid encoded length")
    try:
        return array.tobytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{key} is not valid ASCII") from exc


def _runtime_array_v8(
    fields: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: Any,
    ndim: int,
) -> np.ndarray:
    key = f"{_RUNTIME_PREFIX_V8}{name}"
    array = np.asarray(fields[key])
    expected = np.dtype(dtype)
    if array.dtype != expected or array.ndim != int(ndim):
        raise ValueError(
            f"{key} must have dtype {expected} and ndim {int(ndim)}"
        )
    return array


def tube_coupling_pack_to_runtime_fields_v8(
    pack: TubeCouplingPackV8,
) -> dict[str, np.ndarray]:
    """Flatten a pack into stable numeric L0/L1 runtime coefficients."""
    pack.validate()
    prefix = _RUNTIME_PREFIX_V8
    return {
        f"{prefix}schema_version": np.asarray(8, dtype=np.int32),
        f"{prefix}artifact_kind": _ascii_u8("TubeCouplingPackV8"),
        f"{prefix}vertex_count": np.asarray(pack.vertex_count, dtype=np.int64),
        f"{prefix}vertex_ids": np.asarray(pack.vertex_ids, dtype=np.int32).copy(),
        f"{prefix}rest_vertices_m": np.asarray(
            pack.rest_vertices_m, dtype=np.float32
        ).copy(),
        f"{prefix}driver_indices": np.asarray(
            pack.driver_indices, dtype=np.int16
        ).copy(),
        f"{prefix}driver_weights": np.asarray(
            pack.driver_weights, dtype=np.float32
        ).copy(),
        f"{prefix}material_edges": np.asarray(
            pack.material_edges, dtype=np.int32
        ).copy(),
        f"{prefix}mesh_indices": np.asarray(
            pack.mesh_indices, dtype=np.int32
        ).copy(),
        f"{prefix}mesh_vertex_ranges": np.asarray(
            pack.mesh_vertex_ranges, dtype=np.int32
        ).copy(),
        f"{prefix}topology_digest": _ascii_u8(pack.topology_digest),
        f"{prefix}domain_digest": _ascii_u8(pack.domain_digest),
        f"{prefix}rest_digest": _ascii_u8(pack.rest_digest),
        f"{prefix}weight_digest": _ascii_u8(pack.weight_digest),
        f"{prefix}source_bone_count": np.asarray(
            pack.source_bone_count, dtype=np.int32
        ),
        f"{prefix}influence_slots": np.asarray(
            pack.influence_slots, dtype=np.int32
        ),
        f"{prefix}backend": _ascii_u8(pack.backend),
        f"{prefix}content_digest": _ascii_u8(pack.content_digest()),
    }


def tube_coupling_pack_from_runtime_fields_v8(
    fields: Mapping[str, np.ndarray],
) -> TubeCouplingPackV8:
    """Restore and authenticate a pack from flat numeric coefficients."""
    if not isinstance(fields, Mapping):
        raise ValueError("V8 tube runtime fields must be a mapping")
    if _has_v7_marker(fields):
        raise ValueError("V8 rejects all tube_frame_v7 fields")
    prefix = _RUNTIME_PREFIX_V8
    present = {
        str(key)[len(prefix) :]
        for key in fields
        if str(key).startswith(prefix)
    }
    missing = sorted(_RUNTIME_FIELD_NAMES_V8 - present)
    unknown = sorted(present - _RUNTIME_FIELD_NAMES_V8)
    if missing:
        raise ValueError(f"V8 tube runtime fields missing required fields: {missing}")
    if unknown:
        raise ValueError(f"V8 tube runtime fields contain unknown fields: {unknown}")

    schema = _runtime_array_v8(
        fields, "schema_version", dtype=np.int32, ndim=0
    )
    if int(schema) != 8:
        raise ValueError("V8 tube runtime fields require schema_version 8")
    if _decode_ascii_u8(fields, "artifact_kind") != "TubeCouplingPackV8":
        raise ValueError("invalid V8 tube runtime artifact kind")
    vertex_count = _runtime_array_v8(
        fields, "vertex_count", dtype=np.int64, ndim=0
    )
    source_bone_count = _runtime_array_v8(
        fields, "source_bone_count", dtype=np.int32, ndim=0
    )
    influence_slots = _runtime_array_v8(
        fields, "influence_slots", dtype=np.int32, ndim=0
    )
    result = TubeCouplingPackV8(
        vertex_count=int(vertex_count),
        vertex_ids=_runtime_array_v8(
            fields, "vertex_ids", dtype=np.int32, ndim=1
        ),
        rest_vertices_m=_runtime_array_v8(
            fields, "rest_vertices_m", dtype=np.float32, ndim=2
        ),
        driver_indices=_runtime_array_v8(
            fields, "driver_indices", dtype=np.int16, ndim=2
        ),
        driver_weights=_runtime_array_v8(
            fields, "driver_weights", dtype=np.float32, ndim=2
        ),
        material_edges=_runtime_array_v8(
            fields, "material_edges", dtype=np.int32, ndim=2
        ),
        mesh_indices=_runtime_array_v8(
            fields, "mesh_indices", dtype=np.int32, ndim=1
        ),
        mesh_vertex_ranges=_runtime_array_v8(
            fields, "mesh_vertex_ranges", dtype=np.int32, ndim=2
        ),
        topology_digest=_decode_ascii_u8(fields, "topology_digest", length=64),
        domain_digest=_decode_ascii_u8(fields, "domain_digest", length=64),
        rest_digest=_decode_ascii_u8(fields, "rest_digest", length=64),
        weight_digest=_decode_ascii_u8(fields, "weight_digest", length=64),
        source_bone_count=int(source_bone_count),
        influence_slots=int(influence_slots),
        backend=_decode_ascii_u8(fields, "backend"),
    )
    expected = _decode_ascii_u8(fields, "content_digest", length=64)
    if result.content_digest() != expected:
        raise ValueError("V8 tube runtime content digest mismatch")
    return result


def _canonical_authored_weights(
    asset: AnatomyRiggedAsset, vertex_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if asset.driver_indices is None or asset.driver_weights is None:
        raise ValueError("V8 tube coupling requires original sparse Armature weights")
    all_indices = np.asarray(asset.driver_indices)
    all_weights = np.asarray(asset.driver_weights)
    expected = (len(asset.vertices_rest), INFLUENCE_SLOTS_V8)
    if all_indices.shape != expected or all_weights.shape != expected:
        raise ValueError("V8 requires exactly 14 sparse Armature slots")
    indices = np.asarray(all_indices[vertex_ids], dtype=np.int16)
    weights = np.asarray(all_weights[vertex_ids], dtype=np.float32)
    if (
        np.any(indices < 0)
        or np.any(indices >= SOURCE_BONE_COUNT_V8)
        or np.any(weights < 0.0)
        or not np.all(np.isfinite(weights))
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("invalid original sparse Armature weights")
    return indices, weights


def bake_tube_coupling_v8(
    asset: AnatomyRiggedAsset,
    *,
    runtime_fields: Mapping[str, Any] | None = None,
) -> tuple[TubeCouplingPackV8, dict[str, Any]]:
    """Freeze the subject tube domain and its exact authored 14-slot weights."""
    _require_full_local_fk(asset)
    reject_v7_tube_fields_v8(asset, runtime_fields)
    if len(asset.source_bone_names or []) != SOURCE_BONE_COUNT_V8:
        raise ValueError("V8 tube coupling requires exactly 235 source bones")
    domain = _tube_domain_v8(asset)
    indices, weights = _canonical_authored_weights(asset, domain.vertex_ids)
    rest = np.asarray(asset.vertices_rest, dtype=np.float32)[domain.vertex_ids]
    pack = TubeCouplingPackV8(
        vertex_count=len(asset.vertices_rest),
        vertex_ids=domain.vertex_ids,
        rest_vertices_m=rest,
        driver_indices=indices,
        driver_weights=weights,
        material_edges=domain.material_edges,
        mesh_indices=domain.mesh_indices,
        mesh_vertex_ranges=domain.mesh_vertex_ranges,
        topology_digest=tube_topology_digest_v8(
            len(asset.vertices_rest), asset.faces
        ),
        domain_digest=domain.digest,
        rest_digest=_rest_digest(domain.vertex_ids, rest),
        weight_digest=_weight_digest(domain.vertex_ids, indices, weights),
    )
    report = {
        "available": True,
        "passed": True,
        "backend": TUBE_COUPLING_BACKEND_V8,
        "source_bone_count": SOURCE_BONE_COUNT_V8,
        "influence_slots": INFLUENCE_SLOTS_V8,
        "tube_vertex_count": int(len(domain.vertex_ids)),
        "material_edge_count": int(len(domain.material_edges)),
        "topology_digest": pack.topology_digest,
        "domain_digest": pack.domain_digest,
        "weight_digest": pack.weight_digest,
        "runtime_kdtree": False,
        "runtime_graph_solve": False,
        "runtime_collision": False,
        "parallel_transport": False,
        "cross_section_reconstruction": False,
    }
    return pack, report


def _validate_live_asset(
    asset: AnatomyRiggedAsset,
    pack: TubeCouplingPackV8,
    runtime_fields: Mapping[str, Any] | None,
) -> None:
    pack.validate()
    _require_full_local_fk(asset)
    reject_v7_tube_fields_v8(asset, runtime_fields)
    if len(asset.source_bone_names or []) != SOURCE_BONE_COUNT_V8:
        raise ValueError("V8 tube coupling requires exactly 235 source bones")
    if len(asset.vertices_rest) != int(pack.vertex_count):
        raise ValueError("V8 tube coupling vertex count mismatch")
    if tube_topology_digest_v8(
        len(asset.vertices_rest), asset.faces
    ) != str(pack.topology_digest):
        raise ValueError("V8 tube coupling topology digest mismatch")
    domain = _tube_domain_v8(asset)
    if (
        domain.digest != str(pack.domain_digest)
        or not np.array_equal(domain.vertex_ids, pack.vertex_ids)
        or not np.array_equal(domain.material_edges, pack.material_edges)
    ):
        raise ValueError("V8 tube coupling frozen material domain mismatch")
    indices, weights = _canonical_authored_weights(asset, domain.vertex_ids)
    if _weight_digest(domain.vertex_ids, indices, weights) != str(pack.weight_digest):
        raise ValueError("V8 tube coupling authored weights were modified")
    rest = np.asarray(asset.vertices_rest, dtype=np.float32)[domain.vertex_ids]
    if _rest_digest(domain.vertex_ids, rest) != str(pack.rest_digest):
        raise ValueError("V8 tube coupling subject rest vertices were modified")


def predict_tube_vertices_v8(
    pack: TubeCouplingPackV8,
    transforms: np.ndarray,
    *,
    validate_pack: bool = True,
) -> np.ndarray:
    """Evaluate strict matrix LBS for only the frozen tube vertices."""
    if validate_pack:
        pack.validate()
    matrices = np.asarray(transforms, dtype=np.float64)
    if matrices.shape != (SOURCE_BONE_COUNT_V8, 4, 4):
        raise ValueError("V8 tube transforms must be [235,4,4]")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("V8 tube transforms contain non-finite values")
    affine_row = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    if not np.allclose(matrices[:, 3, :], affine_row, atol=1.0e-7, rtol=0.0):
        raise ValueError("V8 tube transforms must be affine")

    # Identity is a schema invariant, not a tolerance-based test.  Returning
    # the frozen array avoids a float row-sum changing a subject rest vertex by
    # one ULP while retaining exact matrix LBS for every non-identity action.
    if np.array_equal(
        matrices,
        np.tile(np.eye(4, dtype=np.float64), (SOURCE_BONE_COUNT_V8, 1, 1)),
    ):
        return np.asarray(pack.rest_vertices_m).copy()

    selected = matrices[np.asarray(pack.driver_indices, dtype=np.int64)]
    weights = np.asarray(pack.driver_weights, dtype=np.float64)
    blended = np.sum(selected * weights[..., None, None], axis=1)
    rest = np.asarray(pack.rest_vertices_m, dtype=np.float64)
    homogeneous = np.concatenate(
        (rest, np.ones((len(rest), 1), dtype=np.float64)), axis=1
    )
    return np.einsum("vij,vj->vi", blended[:, :3, :], homogeneous).astype(
        np.float32
    )


def apply_tube_coupling_v8(
    asset: AnatomyRiggedAsset,
    transforms: np.ndarray,
    posed_vertices: np.ndarray,
    pack: TubeCouplingPackV8,
    *,
    runtime_fields: Mapping[str, Any] | None = None,
    validate_live: bool = True,
) -> np.ndarray:
    """Overwrite the frozen material domain with its authoritative V71 LBS."""
    if validate_live:
        _validate_live_asset(asset, pack, runtime_fields)
    posed = np.asarray(posed_vertices)
    if posed.shape != (int(pack.vertex_count), 3) or not np.all(np.isfinite(posed)):
        raise ValueError("posed_vertices must be finite [vertex_count,3]")
    result = np.array(posed, dtype=np.float32, copy=True)
    result[np.asarray(pack.vertex_ids, dtype=np.int64)] = predict_tube_vertices_v8(
        pack, transforms, validate_pack=False
    )
    return result


def _unavailable(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "available": False,
        "pass": False,
        "passed": False,
        "reason": str(reason),
        "backend": TUBE_COUPLING_BACKEND_V8,
        **details,
    }


def tube_material_edge_metrics_v8(
    asset: AnatomyRiggedAsset,
    posed_vertices: np.ndarray,
    pack: TubeCouplingPackV8,
    *,
    maximum_edge_ratio_change: float = 0.05,
    runtime_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure frozen topology-edge length preservation on final vertices."""
    try:
        _validate_live_asset(asset, pack, runtime_fields)
    except ValueError as exc:
        return _unavailable(str(exc))
    posed = np.asarray(posed_vertices, dtype=np.float64)
    if posed.shape != (int(pack.vertex_count), 3) or not np.all(np.isfinite(posed)):
        return _unavailable("posed vertices are missing or invalid")
    edges = np.asarray(pack.material_edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return _unavailable("frozen material edges are unavailable")
    rest = np.asarray(pack.rest_vertices_m, dtype=np.float64)
    final = posed[np.asarray(pack.vertex_ids, dtype=np.int64)]
    before = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    after = np.linalg.norm(final[edges[:, 1]] - final[edges[:, 0]], axis=1)
    valid = before > 1.0e-10
    if not np.any(valid):
        return _unavailable("all frozen material edges are degenerate")
    ratio = after[valid] / before[valid]
    maximum = float(np.max(np.abs(ratio - 1.0)))
    passed = bool(maximum <= float(maximum_edge_ratio_change))
    return {
        "available": True,
        "pass": passed,
        "passed": passed,
        "backend": TUBE_COUPLING_BACKEND_V8,
        "edge_kind": "triangle_topology_material_edges",
        "parallel_transport": False,
        "fixed_edge_count": int(len(ratio)),
        "edge_ratio_q01": float(np.quantile(ratio, 0.01)),
        "edge_ratio_median": float(np.median(ratio)),
        "edge_ratio_q99": float(np.quantile(ratio, 0.99)),
        "edge_ratio_max_abs_change": maximum,
        "maximum_edge_ratio_change": float(maximum_edge_ratio_change),
    }


def v71_action_reference_gate_v8(
    pack: TubeCouplingPackV8,
    transforms: np.ndarray,
    reference_vertices: np.ndarray,
    *,
    provenance: Mapping[str, Any],
    maximum_rms_error_m: float = 0.0005,
    maximum_max_error_m: float | None = None,
) -> dict[str, Any]:
    """Compare compiled LBS with an authorized Blender/V71 action export."""
    content_digest = str(provenance.get("content_digest", ""))
    if (
        provenance.get("authorized") is not True
        or provenance.get("kind") != "blender_v71_action"
        or len(content_digest) != 64
        or any(c not in "0123456789abcdef" for c in content_digest)
    ):
        return _unavailable("authorized Blender/V71 action provenance is required")
    try:
        predicted = np.asarray(
            predict_tube_vertices_v8(pack, transforms), dtype=np.float64
        )
    except ValueError as exc:
        return _unavailable(str(exc))
    reference = np.asarray(reference_vertices, dtype=np.float64)
    if reference.shape == (int(pack.vertex_count), 3):
        reference = reference[np.asarray(pack.vertex_ids, dtype=np.int64)]
    if reference.shape != predicted.shape or not np.all(np.isfinite(reference)):
        return _unavailable("action reference vertices do not match the frozen domain")
    error = np.linalg.norm(predicted - reference, axis=1)
    rms = float(np.sqrt(np.mean(error * error)))
    maximum = float(np.max(error))
    rms_pass = rms <= float(maximum_rms_error_m)
    max_pass = (
        True
        if maximum_max_error_m is None
        else maximum <= float(maximum_max_error_m)
    )
    passed = bool(rms_pass and max_pass)
    return {
        "available": True,
        "pass": passed,
        "passed": passed,
        "backend": TUBE_COUPLING_BACKEND_V8,
        "reference_kind": "blender_v71_action",
        "reference_content_digest": content_digest,
        "vertex_count": int(len(error)),
        "rms_error_m": rms,
        "max_error_m": maximum,
        "maximum_rms_error_m": float(maximum_rms_error_m),
        "maximum_max_error_m": (
            None
            if maximum_max_error_m is None
            else float(maximum_max_error_m)
        ),
        "parallel_transport": False,
    }


__all__ = [
    "INFLUENCE_SLOTS_V8",
    "SOURCE_BONE_COUNT_V8",
    "TUBE_COUPLING_BACKEND_V8",
    "TubeCouplingPackV8",
    "apply_tube_coupling_v8",
    "bake_tube_coupling_v8",
    "predict_tube_vertices_v8",
    "reject_v7_tube_fields_v8",
    "tube_coupling_pack_from_runtime_fields_v8",
    "tube_coupling_pack_to_runtime_fields_v8",
    "tube_material_edge_metrics_v8",
    "tube_topology_digest_v8",
    "v71_action_reference_gate_v8",
]
