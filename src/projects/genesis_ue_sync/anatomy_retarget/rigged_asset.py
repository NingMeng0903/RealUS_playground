"""Schema helpers for anatomy meshes retargeted to SMPL-X runtime joints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POSE_FORMAT = "smplx_body55_axis_angle"
DEFAULT_COORDINATE_SYSTEM = "genesis_z_up_m"


def _string_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "U", "O"}:
        return np.asarray([str(v.decode("utf-8") if isinstance(v, bytes) else v) for v in arr.reshape(-1)], dtype=object)
    return np.asarray([str(v) for v in arr.reshape(-1)], dtype=object)


@dataclass(frozen=True)
class AnatomyRiggedAsset:
    vertices_rest: np.ndarray
    faces: np.ndarray
    lbs_weights: np.ndarray
    joint_names: list[str]
    parents: np.ndarray
    rest_joints: np.ndarray
    inverse_bind: np.ndarray
    source_mesh_names: list[str]
    pose_format: str = DEFAULT_POSE_FORMAT
    coordinate_system: str = DEFAULT_COORDINATE_SYSTEM
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        vertices = np.asarray(self.vertices_rest, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        weights = np.asarray(self.lbs_weights, dtype=np.float32)
        parents = np.asarray(self.parents, dtype=np.int32).reshape(-1)
        rest_joints = np.asarray(self.rest_joints, dtype=np.float32)
        inverse_bind = np.asarray(self.inverse_bind, dtype=np.float32)
        joint_count = len(self.joint_names)

        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError(f"vertices_rest must be [N, 3], got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be [F, 3], got {faces.shape}")
        if weights.shape != (vertices.shape[0], joint_count):
            raise ValueError(f"lbs_weights must be {(vertices.shape[0], joint_count)}, got {weights.shape}")
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
        if not np.all(np.isfinite(weights)):
            raise ValueError("lbs_weights contains non-finite values")
        row_sums = weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1.0e-4, rtol=0.0):
            raise ValueError(f"lbs_weights rows must sum to 1; max error={float(np.max(np.abs(row_sums - 1.0))):.6g}")
        if joint_count and int(parents[0]) not in (-1, 0):
            raise ValueError("parents[0] must be -1 or 0 for root")
        for idx, parent in enumerate(parents.tolist()):
            if idx == 0:
                continue
            if parent < 0 or parent >= idx:
                raise ValueError(f"parents[{idx}]={parent} must point to an earlier joint")


def save_rigged_asset(path: Path | str, asset: AnatomyRiggedAsset) -> Path:
    asset.validate()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices_rest=np.asarray(asset.vertices_rest, dtype=np.float32),
        faces=np.asarray(asset.faces, dtype=np.int32),
        lbs_weights=np.asarray(asset.lbs_weights, dtype=np.float32),
        joint_names=np.asarray(asset.joint_names, dtype=object),
        parents=np.asarray(asset.parents, dtype=np.int32),
        rest_joints=np.asarray(asset.rest_joints, dtype=np.float32),
        inverse_bind=np.asarray(asset.inverse_bind, dtype=np.float32),
        source_mesh_names=np.asarray(asset.source_mesh_names, dtype=object),
        pose_format=np.asarray(str(asset.pose_format), dtype=object),
        coordinate_system=np.asarray(str(asset.coordinate_system), dtype=object),
        metadata=np.asarray(asset.metadata or {}, dtype=object),
    )
    return out


def load_rigged_asset(path: Path | str, *, validate: bool = True) -> AnatomyRiggedAsset:
    data = np.load(Path(path), allow_pickle=True)
    metadata: dict[str, Any] | None = None
    if "metadata" in data.files:
        raw_meta = data["metadata"]
        try:
            metadata = dict(raw_meta.item())
        except Exception:
            metadata = {}
    asset = AnatomyRiggedAsset(
        vertices_rest=np.asarray(data["vertices_rest"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int32),
        lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32),
        joint_names=[str(v) for v in _string_array(data["joint_names"]).tolist()],
        parents=np.asarray(data["parents"], dtype=np.int32),
        rest_joints=np.asarray(data["rest_joints"], dtype=np.float32),
        inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float32),
        source_mesh_names=[str(v) for v in _string_array(data["source_mesh_names"]).tolist()],
        pose_format=str(data["pose_format"].item()) if "pose_format" in data.files else DEFAULT_POSE_FORMAT,
        coordinate_system=str(data["coordinate_system"].item()) if "coordinate_system" in data.files else DEFAULT_COORDINATE_SYSTEM,
        metadata=metadata,
    )
    if validate:
        asset.validate()
    return asset

