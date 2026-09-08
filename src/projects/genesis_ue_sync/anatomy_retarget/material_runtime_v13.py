"""Experimental once-compiled material transport on the frozen Blender rig.

The authored 235-controller motion and sparse weights remain the source
motion. Bone placement comes from an explicitly named shadow. A second,
fixed material attachment records how soft tissue follows that placement.
No skin queries, nearest-neighbour searches or optimization run in apply_pose.
This runtime does not assert anatomical containment or arbitrary-pose support.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import _weighted_rest_correction
from .pose_map_v1 import PoseMapV1
from .pose_map_v10 import apply_pose_map_global_v10
from .rigged_asset import load_rigged_asset, save_rigged_asset
from .terminal_reseat_v12 import FOREARM_SHAFT_ROOTS, _cluster_vertex_ids


def tissue_vertex_ids(asset: Any, tissues: set[str]) -> np.ndarray:
    chunks = [np.arange(int(a), int(b), dtype=np.int64)
              for tissue, (a, b) in zip(asset.source_tissues, asset.source_vertex_ranges)
              if str(tissue).lower() in tissues]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def rigid_correspondence(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover a persisted mesh-only correction without repeating its solver."""
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("rigid correspondence needs matching [N,3] points")
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
    r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    t = yc - r @ xc
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    error = float(np.linalg.norm(x @ r.T + t - y, axis=1).max())
    return transform, error


def recover_shared_weight_rest(base: Any, candidate: Any, asset: Any) -> tuple[np.ndarray, dict]:
    """Replay V12e's saved corrections on every weighted soft vertex once.

    The four forearm transforms are recovered from corresponding rigid bone
    points and checked to micrometre precision. Bones and target binds are
    retained byte-for-byte; this function does not refit them.
    """
    if not np.array_equal(base.faces, candidate.faces) or not np.array_equal(base.betas, candidate.betas):
        raise ValueError("base and candidate must share topology and beta")
    names = list(asset.source_bone_names)
    delta = np.asarray(candidate.B_final) @ np.linalg.inv(base.B_final)
    report = {}
    for name in FOREARM_SHAFT_ROOTS:
        controller = names.index(name)
        ids = _cluster_vertex_ids(asset, [controller])
        transform, error = rigid_correspondence(base.vertices_final[ids], candidate.vertices_final[ids])
        if error > 2.e-6:
            raise ValueError(f"{name} is not the saved rigid reseat: {error} m")
        delta[controller] = transform
        report[name] = {"reconstruction_max_m": error, "transform": transform.tolist()}
    rest = np.asarray(candidate.vertices_final).copy()
    soft = tissue_vertex_ids(asset, set(asset.source_tissues) - {"bone"})
    rest[soft] = _weighted_rest_correction(
        np.asarray(base.vertices_final)[soft], np.asarray(asset.driver_indices)[soft],
        np.asarray(asset.driver_weights)[soft], delta,
    )
    report["soft_vertex_count"] = len(soft)
    report["soft_transport_application_count"] = 1
    return rest, report


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class MaterialRuntimeV13:
    source_asset: Any
    pose_map: PoseMapV1
    target_rest: np.ndarray
    soft_ids: np.ndarray
    attachment: Any = None

    def apply_pose(self, pose_axis_angle: np.ndarray, *, mode: str = "weights",
                   transl: np.ndarray | None = None) -> np.ndarray:
        """Evaluate one supported SMPL-X pose without Blender or a skin model."""
        pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(55, 3)
        if not np.all(np.isfinite(pose)):
            raise ValueError("pose must be finite")
        if mode not in {"weights", "attachments", "source"}:
            raise ValueError(f"unknown material transport mode: {mode}")
        asset = self.source_asset
        source_global = source_bone_posed_global(asset, pose)
        source_transforms = source_global @ np.linalg.inv(self.pose_map.source_bind_global)
        source = _weighted_rest_correction(
            asset.vertices_rest, asset.driver_indices, asset.driver_weights, source_transforms)
        if mode == "source":
            result = source
        else:
            target_global = apply_pose_map_global_v10(
                self.pose_map, source_asset=asset, pose_axis_angle=pose)
            result = _weighted_rest_correction(
                self.target_rest, asset.driver_indices, asset.driver_weights,
                target_global @ self.pose_map.target_inverse_bind)
            if mode == "attachments":
                if self.attachment is None:
                    raise ValueError("runtime has no compiled material attachments")
                from .material_attachment_v13 import transport_material_attachment_v13
                result[self.soft_ids] = transport_material_attachment_v13(
                    self.attachment, source, result, source[self.soft_ids])
        if transl is not None:
            t = np.asarray(transl, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(t)):
                raise ValueError("translation must be finite")
            result = result + t
        if not np.all(np.isfinite(result)):
            raise ValueError("runtime produced non-finite vertices")
        return np.asarray(result, dtype=np.float32)

    def save(self, root: Path, *, provenance: dict) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        save_rigged_asset(root / "source_rig.npz", self.source_asset)
        arrays = {f.name: getattr(self.pose_map, f.name) for f in fields(self.pose_map)}
        arrays.update(target_rest=self.target_rest, soft_ids=self.soft_ids)
        np.savez_compressed(root / "runtime.npz", **arrays)
        names = ["source_rig.npz", "runtime.npz"]
        if self.attachment is not None:
            from .material_attachment_v13 import save_material_attachment_v13
            save_material_attachment_v13(root / "attachment.npz", self.attachment)
            names.append("attachment.npz")
        manifest = dict(schema_version=13, artifact_kind="MaterialRuntimeV13",
                        publishable=False, requires_blender_at_runtime=False,
                        requires_pose_rebake=False, requires_skin_queries_at_runtime=False,
                        provenance=provenance,
                        files={name: sha256_file(root / name) for name in names})
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def load_material_runtime_v13(root: Path) -> MaterialRuntimeV13:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("artifact_kind") != "MaterialRuntimeV13":
        raise ValueError("not a MaterialRuntimeV13 artifact")
    expected = {"source_rig.npz", "runtime.npz"}
    if not expected <= set(manifest["files"]):
        raise ValueError("incomplete runtime artifact")
    for name, digest in manifest["files"].items():
        if name not in expected | {"attachment.npz"} or sha256_file(root / name) != digest:
            raise ValueError(f"runtime file integrity check failed: {name}")
    asset = load_rigged_asset(root / "source_rig.npz")
    with np.load(root / "runtime.npz", allow_pickle=False) as data:
        kwargs = {f.name: data[f.name].copy() for f in fields(PoseMapV1)}
        for name in ("source_operator_digest", "subject_label", "oracle_sha256"):
            kwargs[name] = str(kwargs[name].item())
        pose_map = PoseMapV1(**kwargs)
        pose_map.validate()
        rest, ids = data["target_rest"].copy(), data["soft_ids"].copy()
    if rest.shape != asset.vertices_rest.shape or not np.all(np.isfinite(rest)):
        raise ValueError("invalid target rest")
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(rest)):
        raise ValueError("invalid soft vertex IDs")
    attachment = None
    if "attachment.npz" in manifest["files"]:
        from .material_attachment_v13 import load_material_attachment_v13
        attachment = load_material_attachment_v13(root / "attachment.npz")
    return MaterialRuntimeV13(asset, pose_map, rest, ids, attachment)
