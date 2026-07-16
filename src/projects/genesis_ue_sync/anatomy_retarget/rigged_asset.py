"""Schema helpers for anatomy meshes driven by SMPL-X through a source rig."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POSE_FORMAT = "smplx_body55_axis_angle"
DEFAULT_COORDINATE_SYSTEM = "genesis_z_up_m"
ANATOMY_ASSET_SCHEMA_VERSION = 4
SOURCE_DRIVER_MODES: tuple[str, ...] = (
    "joint_local",
    "segment_root",
    "twist",
    "bind_follow",
    "rigid_group",
)

# Compact serialization for Blender's Bone.inherit_scale enum.  Keep this
# stable: source templates are intended to outlive the Blender process that
# produced them.
def _string_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "U", "O"}:
        return np.asarray([str(v.decode("utf-8") if isinstance(v, bytes) else v) for v in arr.reshape(-1)], dtype=object)
    return np.asarray([str(v) for v in arr.reshape(-1)], dtype=object)


def source_global_from_local(rest_local: Any, parents: Any) -> np.ndarray:
    """Reconstruct source-rig global bind frames from the only persisted frames."""
    local = np.asarray(rest_local, dtype=np.float64).reshape(-1, 4, 4)
    pa = np.asarray(parents, dtype=np.int64).reshape(-1)
    if len(local) != len(pa):
        raise ValueError("source rest_local/parents length mismatch")
    result = np.empty_like(local)
    for bone, parent in enumerate(pa.tolist()):
        if int(parent) < 0:
            result[bone] = local[bone]
        else:
            if int(parent) >= bone:
                raise ValueError("source parents must be parent-before-child")
            result[bone] = result[int(parent)] @ local[bone]
    return result.astype(np.float32)


def _points_to_bone_local(points: Any, rest_global: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inverse = np.linalg.inv(np.asarray(rest_global, dtype=np.float64).reshape(-1, 4, 4))
    return np.einsum("bij,bj->bi", inverse[:, :3, :3], pts) + inverse[:, :3, 3]


def _points_from_bone_local(points: Any, rest_global: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    global_bind = np.asarray(rest_global, dtype=np.float64).reshape(-1, 4, 4)
    return np.einsum("bij,bj->bi", global_bind[:, :3, :3], pts) + global_bind[:, :3, 3]


@dataclass(frozen=True)
class AnatomyRiggedAsset:
    vertices_rest: np.ndarray
    faces: np.ndarray
    lbs_weights: np.ndarray | None
    joint_names: list[str]
    parents: np.ndarray
    rest_joints: np.ndarray
    inverse_bind: np.ndarray
    source_mesh_names: list[str]
    source_vertex_ranges: np.ndarray | None = None
    source_tissues: list[str] | None = None
    driver_indices: np.ndarray | None = None
    driver_weights: np.ndarray | None = None
    source_bone_names: list[str] | None = None
    source_bone_parents: np.ndarray | None = None
    source_rest_global: np.ndarray | None = None
    source_rest_local: np.ndarray | None = None
    source_inverse_bind: np.ndarray | None = None
    source_bone_head: np.ndarray | None = None
    source_bone_tail: np.ndarray | None = None
    source_bone_smplx_a: np.ndarray | None = None
    source_bone_smplx_b: np.ndarray | None = None
    source_bone_blend: np.ndarray | None = None
    source_bone_driver_types: list[str] | None = None
    rigid_component_ids: np.ndarray | None = None
    leg_material_coordinates: np.ndarray | None = None
    registration_reference: np.ndarray | None = None
    source_skin_vertices: np.ndarray | None = None
    source_skin_faces: np.ndarray | None = None
    pose_cache_vertices: np.ndarray | None = None
    pose_cache_hash: str = ""
    pose_format: str = DEFAULT_POSE_FORMAT
    coordinate_system: str = DEFAULT_COORDINATE_SYSTEM
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        vertices = np.asarray(self.vertices_rest, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        weights = None if self.lbs_weights is None else np.asarray(self.lbs_weights, dtype=np.float32)
        parents = np.asarray(self.parents, dtype=np.int32).reshape(-1)
        rest_joints = np.asarray(self.rest_joints, dtype=np.float32)
        inverse_bind = np.asarray(self.inverse_bind, dtype=np.float32)
        joint_count = len(self.joint_names)

        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError(f"vertices_rest must be [N, 3], got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be [F, 3], got {faces.shape}")
        if weights is not None:
            if weights.shape != (vertices.shape[0], joint_count):
                raise ValueError(f"legacy lbs_weights must be {(vertices.shape[0], joint_count)}, got {weights.shape}")
            if np.any(weights < 0.0):
                raise ValueError("lbs_weights contains negative values")
        if self.driver_indices is not None or self.driver_weights is not None:
            if self.driver_indices is None or self.driver_weights is None:
                raise ValueError("driver_indices and driver_weights must be stored together")
            sparse_i = np.asarray(self.driver_indices, dtype=np.int32)
            sparse_w = np.asarray(self.driver_weights, dtype=np.float32)
            if sparse_i.shape != sparse_w.shape or sparse_i.ndim != 2 or sparse_i.shape[0] != vertices.shape[0]:
                raise ValueError("sparse drivers must both be [N, K]")
            source_count = len(self.source_bone_names or [])
            driver_count = source_count if source_count else joint_count
            if sparse_i.size and (int(sparse_i.min()) < 0 or int(sparse_i.max()) >= driver_count):
                raise ValueError("driver_indices contains an invalid source bone/joint")
            if not np.allclose(sparse_w.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0):
                raise ValueError("driver_weights rows must sum to one")
        if parents.shape != (joint_count,):
            raise ValueError(f"parents must be [{joint_count}], got {parents.shape}")
        if rest_joints.shape != (joint_count, 3):
            raise ValueError(f"rest_joints must be [{joint_count}, 3], got {rest_joints.shape}")
        if inverse_bind.shape != (joint_count, 4, 4):
            raise ValueError(f"inverse_bind must be [{joint_count}, 4, 4], got {inverse_bind.shape}")
        if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
            raise ValueError("faces contain vertex indices outside vertices_rest")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_rest contains non-finite values")
        if self.registration_reference is not None and np.asarray(self.registration_reference).shape != vertices.shape:
            raise ValueError("registration_reference must match vertices_rest")
        if self.source_skin_vertices is not None:
            skin_v = np.asarray(self.source_skin_vertices)
            skin_f = np.asarray(self.source_skin_faces)
            if skin_v.ndim != 2 or skin_v.shape[1] != 3 or skin_f.ndim != 2 or skin_f.shape[1] != 3:
                raise ValueError("source skin must be [N,3] vertices and [F,3] faces")
        if self.pose_cache_vertices is not None:
            cached = np.asarray(self.pose_cache_vertices)
            if cached.shape != vertices.shape or not np.all(np.isfinite(cached)):
                raise ValueError("pose_cache_vertices must be finite and match vertices_rest")
            if not str(self.pose_cache_hash):
                raise ValueError("pose_cache_hash is required with pose_cache_vertices")
        if weights is not None:
            if not np.all(np.isfinite(weights)):
                raise ValueError("lbs_weights contains non-finite values")
            row_sums = weights.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1.0e-5, rtol=0.0):
                raise ValueError(f"lbs_weights rows must sum to 1; max error={float(np.max(np.abs(row_sums - 1.0))):.6g}")
        if joint_count and int(parents[0]) not in (-1, 0):
            raise ValueError("parents[0] must be -1 or 0 for root")
        for idx, parent in enumerate(parents.tolist()):
            if idx == 0:
                continue
            if parent < 0 or parent >= idx:
                raise ValueError(f"parents[{idx}]={parent} must point to an earlier joint")

        if self.source_bone_names is not None:
            bone_count = len(self.source_bone_names)
            source_arrays = {
                "source_bone_parents": (self.source_bone_parents, (bone_count,)),
                "source_rest_global": (self.source_rest_global, (bone_count, 4, 4)),
                "source_inverse_bind": (self.source_inverse_bind, (bone_count, 4, 4)),
                "source_bone_smplx_a": (self.source_bone_smplx_a, (bone_count,)),
                "source_bone_smplx_b": (self.source_bone_smplx_b, (bone_count,)),
                "source_bone_blend": (self.source_bone_blend, (bone_count,)),
            }
            for name, (value, shape) in source_arrays.items():
                if value is None or np.asarray(value).shape != shape:
                    raise ValueError(f"{name} must be {shape} for source-rig v2")
            if self.source_bone_driver_types is None or len(self.source_bone_driver_types) != bone_count:
                raise ValueError("source_bone_driver_types must have one entry per source bone")
            unknown_modes = sorted(set(self.source_bone_driver_types) - set(SOURCE_DRIVER_MODES))
            if unknown_modes:
                raise ValueError(f"unknown source driver mode(s): {unknown_modes}")
            source_parents = np.asarray(self.source_bone_parents, dtype=np.int32)
            for idx, parent in enumerate(source_parents.tolist()):
                if parent >= idx or parent < -1:
                    raise ValueError(f"source_bone_parents[{idx}]={parent} is not topological")
            if self.driver_indices is None or self.driver_weights is None:
                raise ValueError("source-rig v2 requires sparse driver indices and weights")

            fk_values = (self.source_rest_local, self.source_bone_head, self.source_bone_tail)
            if any(value is not None for value in fk_values):
                if not all(value is not None for value in fk_values):
                    raise ValueError("source-rig FK metadata must be stored as one complete set")
                fk_arrays = {
                    "source_rest_local": (self.source_rest_local, (bone_count, 4, 4)),
                    "source_bone_head": (self.source_bone_head, (bone_count, 3)),
                    "source_bone_tail": (self.source_bone_tail, (bone_count, 3)),
                }
                for name, (value, shape) in fk_arrays.items():
                    arr = np.asarray(value)
                    if arr.shape != shape:
                        raise ValueError(f"{name} must be {shape} for source-rig v3")
                    if not np.all(np.isfinite(arr)):
                        raise ValueError(f"{name} contains non-finite values")
                source_global = np.asarray(self.source_rest_global, dtype=np.float64)
                source_local = np.asarray(self.source_rest_local, dtype=np.float64)
                for idx, parent in enumerate(source_parents.tolist()):
                    reconstructed = (
                        source_local[idx]
                        if int(parent) < 0
                        else source_global[int(parent)] @ source_local[idx]
                    )
                    if not np.allclose(reconstructed, source_global[idx], atol=1.0e-4, rtol=0.0):
                        raise ValueError(
                            f"source_rest_local[{idx}] does not reconstruct source_rest_global"
                        )
                heads = np.asarray(self.source_bone_head, dtype=np.float64)
                tails = np.asarray(self.source_bone_tail, dtype=np.float64)
                if np.any(np.linalg.norm(tails - heads, axis=1) <= 1.0e-8):
                    raise ValueError("source bone head/tail contains a zero-length bone")


def save_rigged_asset(path: Path | str, asset: AnatomyRiggedAsset) -> Path:
    asset.validate()
    if asset.source_bone_names is None:
        raise ValueError("schema v4 requires a complete source rig")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if asset.driver_indices is None or asset.driver_weights is None:
        if asset.lbs_weights is None:
            raise ValueError("asset requires either sparse drivers or legacy dense weights")
        driver_indices, driver_weights = sparse_driver_weights(asset.lbs_weights)
    else:
        driver_indices, driver_weights = asset.driver_indices, asset.driver_weights
    payload: dict[str, Any] = dict(
        schema_version=np.asarray(ANATOMY_ASSET_SCHEMA_VERSION, dtype=np.int32),
        vertices_rest=np.asarray(asset.vertices_rest, dtype=np.float32),
        faces=np.asarray(asset.faces, dtype=np.int32),
        joint_names=np.asarray(asset.joint_names, dtype=object),
        parents=np.asarray(asset.parents, dtype=np.int32),
        rest_joints=np.asarray(asset.rest_joints, dtype=np.float32),
        inverse_bind=np.asarray(asset.inverse_bind, dtype=np.float32),
        source_mesh_names=np.asarray(asset.source_mesh_names, dtype=object),
        source_vertex_ranges=np.asarray(
            asset.source_vertex_ranges if asset.source_vertex_ranges is not None else [], dtype=np.int32
        ).reshape(-1, 2),
        source_tissues=np.asarray(asset.source_tissues or [], dtype=object),
        driver_indices=np.asarray(driver_indices, dtype=np.int16),
        driver_weights=np.asarray(driver_weights, dtype=np.float32),
        pose_format=np.asarray(str(asset.pose_format), dtype=object),
        coordinate_system=np.asarray(str(asset.coordinate_system), dtype=object),
        metadata=np.asarray(asset.metadata or {}, dtype=object),
    )
    if asset.source_bone_names is not None:
        if asset.source_rest_local is None:
            raise ValueError("schema v4 requires source_rest_local")
        source_global = source_global_from_local(asset.source_rest_local, asset.source_bone_parents)
        head_local = _points_to_bone_local(asset.source_bone_head, source_global)
        tail_local = _points_to_bone_local(asset.source_bone_tail, source_global)
        payload.update(
            source_bone_names=np.asarray(asset.source_bone_names, dtype=object),
            source_bone_parents=np.asarray(asset.source_bone_parents, dtype=np.int16),
            source_rest_local=np.asarray(asset.source_rest_local, dtype=np.float32),
            source_bone_head_local=np.asarray(head_local, dtype=np.float32),
            source_bone_tail_local=np.asarray(tail_local, dtype=np.float32),
            source_bone_smplx_a=np.asarray(asset.source_bone_smplx_a, dtype=np.int16),
            source_bone_smplx_b=np.asarray(asset.source_bone_smplx_b, dtype=np.int16),
            source_bone_blend=np.asarray(asset.source_bone_blend, dtype=np.float32),
            source_bone_driver_types=np.asarray(asset.source_bone_driver_types, dtype=object),
            rigid_component_ids=np.asarray(
                asset.rigid_component_ids if asset.rigid_component_ids is not None else [], dtype=np.int32
            ),
            leg_material_coordinates=np.asarray(
                asset.leg_material_coordinates if asset.leg_material_coordinates is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            registration_reference=np.asarray(
                asset.registration_reference if asset.registration_reference is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            source_skin_vertices=np.asarray(
                asset.source_skin_vertices if asset.source_skin_vertices is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            source_skin_faces=np.asarray(
                asset.source_skin_faces if asset.source_skin_faces is not None else [], dtype=np.int32
            ).reshape(-1, 3),
            posed_vertices=np.asarray(
                asset.pose_cache_vertices if asset.pose_cache_vertices is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            pose_hash=np.asarray(str(asset.pose_cache_hash), dtype=object),
        )
    np.savez_compressed(
        out,
        **payload,
    )
    return out


def sparse_driver_weights(weights: Any, *, top_k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense Blender-derived drivers to normalized sparse top-k form."""
    dense = np.asarray(weights, dtype=np.float32)
    if dense.ndim != 2 or dense.shape[1] == 0:
        raise ValueError(f"weights must be [N, J], got {dense.shape}")
    k = max(1, min(int(top_k), int(dense.shape[1])))
    indices = np.argpartition(dense, -k, axis=1)[:, -k:]
    values = np.take_along_axis(dense, indices, axis=1)
    order = np.argsort(-values, axis=1)
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int16)
    values = np.take_along_axis(values, order, axis=1).astype(np.float32)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1.0e-8)
    return indices, values


def load_rigged_asset(path: Path | str, *, validate: bool = True) -> AnatomyRiggedAsset:
    data = np.load(Path(path), allow_pickle=True)
    schema = int(np.asarray(data["schema_version"]).reshape(-1)[0]) if "schema_version" in data.files else 0
    if schema != ANATOMY_ASSET_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses anatomy schema {schema}; schema {ANATOMY_ASSET_SCHEMA_VERSION} "
            "is required, rebuild from the source blend"
        )
    metadata: dict[str, Any] | None = None
    if "metadata" in data.files:
        raw_meta = data["metadata"]
        try:
            metadata = dict(raw_meta.item())
        except Exception:
            metadata = {}
    required = {
        "driver_indices", "driver_weights", "source_bone_names", "source_bone_parents",
        "source_rest_local", "source_bone_head_local", "source_bone_tail_local",
        "source_bone_smplx_a", "source_bone_smplx_b", "source_bone_blend",
        "source_bone_driver_types",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{path} is missing schema-v4 fields: {missing}")
    driver_indices = np.asarray(data["driver_indices"], dtype=np.int16)
    driver_weights = np.asarray(data["driver_weights"], dtype=np.float32)
    source_parents = (
        np.asarray(data["source_bone_parents"], dtype=np.int32)
        if "source_bone_parents" in data.files else None
    )
    source_local = (
        np.asarray(data["source_rest_local"], dtype=np.float32)
        if "source_rest_local" in data.files else None
    )
    source_global = (
        source_global_from_local(source_local, source_parents)
        if source_local is not None and source_parents is not None else None
    )
    source_head = (
        _points_from_bone_local(data["source_bone_head_local"], source_global).astype(np.float32)
        if "source_bone_head_local" in data.files and source_global is not None else None
    )
    source_tail = (
        _points_from_bone_local(data["source_bone_tail_local"], source_global).astype(np.float32)
        if "source_bone_tail_local" in data.files and source_global is not None else None
    )
    asset = AnatomyRiggedAsset(
        vertices_rest=np.asarray(data["vertices_rest"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int32),
        lbs_weights=None,
        joint_names=[str(v) for v in _string_array(data["joint_names"]).tolist()],
        parents=np.asarray(data["parents"], dtype=np.int32),
        rest_joints=np.asarray(data["rest_joints"], dtype=np.float32),
        inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float32),
        source_mesh_names=[str(v) for v in _string_array(data["source_mesh_names"]).tolist()],
        source_vertex_ranges=(
            np.asarray(data["source_vertex_ranges"], dtype=np.int32).reshape(-1, 2)
            if "source_vertex_ranges" in data.files
            else None
        ),
        source_tissues=(
            [str(v) for v in _string_array(data["source_tissues"]).tolist()]
            if "source_tissues" in data.files and data["source_tissues"].size
            else None
        ),
        driver_indices=driver_indices,
        driver_weights=driver_weights,
        source_bone_names=(
            [str(v) for v in _string_array(data["source_bone_names"]).tolist()]
            if "source_bone_names" in data.files
            else None
        ),
        source_bone_parents=source_parents,
        source_rest_global=source_global,
        source_rest_local=source_local,
        source_inverse_bind=(np.linalg.inv(source_global).astype(np.float32) if source_global is not None else None),
        source_bone_head=source_head,
        source_bone_tail=source_tail,
        source_bone_smplx_a=np.asarray(data["source_bone_smplx_a"], dtype=np.int32) if "source_bone_smplx_a" in data.files else None,
        source_bone_smplx_b=np.asarray(data["source_bone_smplx_b"], dtype=np.int32) if "source_bone_smplx_b" in data.files else None,
        source_bone_blend=np.asarray(data["source_bone_blend"], dtype=np.float32) if "source_bone_blend" in data.files else None,
        source_bone_driver_types=(
            [str(v) for v in _string_array(data["source_bone_driver_types"]).tolist()]
            if "source_bone_driver_types" in data.files
            else None
        ),
        rigid_component_ids=np.asarray(data["rigid_component_ids"], dtype=np.int32) if "rigid_component_ids" in data.files else None,
        leg_material_coordinates=np.asarray(data["leg_material_coordinates"], dtype=np.float32).reshape(-1, 3) if "leg_material_coordinates" in data.files and data["leg_material_coordinates"].size else None,
        registration_reference=np.asarray(data["registration_reference"], dtype=np.float32).reshape(-1, 3) if "registration_reference" in data.files and data["registration_reference"].size else None,
        source_skin_vertices=np.asarray(data["source_skin_vertices"], dtype=np.float32).reshape(-1, 3) if "source_skin_vertices" in data.files and data["source_skin_vertices"].size else None,
        source_skin_faces=np.asarray(data["source_skin_faces"], dtype=np.int32).reshape(-1, 3) if "source_skin_faces" in data.files and data["source_skin_faces"].size else None,
        pose_cache_vertices=np.asarray(data["posed_vertices"], dtype=np.float32).reshape(-1, 3) if "posed_vertices" in data.files and data["posed_vertices"].size else None,
        pose_cache_hash=str(data["pose_hash"].item()) if "pose_hash" in data.files else "",
        pose_format=str(data["pose_format"].item()) if "pose_format" in data.files else DEFAULT_POSE_FORMAT,
        coordinate_system=str(data["coordinate_system"].item()) if "coordinate_system" in data.files else DEFAULT_COORDINATE_SYSTEM,
        metadata=metadata,
    )
    if validate:
        asset.validate()
    return asset
