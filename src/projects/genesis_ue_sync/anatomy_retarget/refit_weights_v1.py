"""V10 shadow: refit 14-slot LBS weights onto SMPL-X body shape.

Keeps the frozen Blender 235-controller hierarchy and pose_map right-multiply
linkage. Only selected knee-chain mesh rows may change ``driver_indices`` /
``driver_weights``. Faces, ranges, bone parents, and tubes stay frozen.

Contract: candidate-only. ``publishable=false``. Does not update trusted/latest.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .rigged_asset import AnatomyRiggedAsset


REFIT_WEIGHTS_KIND = "WeightRefitV1"
REFIT_WEIGHTS_SCHEMA = 1
N_DRIVER_SLOTS = 14

# Left/right knee chain meshes that may receive new skinning rows.
DEFAULT_TARGET_MESHES: tuple[str, ...] = (
    "Femur_L",
    "Patella_L",
    "Tibia_L",
    "Fibula_L",
    "Femur_R",
    "Patella_R",
    "Tibia_R",
    "Fibula_R",
)

# Per-side candidate controllers. Distal femur must be allowed to blend onto
# Knee_Rotate — authored Femur_* is ~100% Femur_Rot, which is why flexed
# containment fails after rest-only embed.
_SIDE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "L": (
        "Femur_Rot_L",
        "Knee_Rotate_L",
        "Patella_Rotate_L",
        "Tibia_Bone_L",
        "Tibia_Twist_L",
    ),
    "R": (
        "Femur_Rot_R",
        "Knee_Rotate_R",
        "Patella_Rotate_R",
        "Tibia_Bone_R",
        "Tibia_Twist_R",
    ),
}

_MESH_CORE_BONE: dict[str, str] = {
    "Femur_L": "Femur_Rot_L",
    "Femur_R": "Femur_Rot_R",
    "Patella_L": "Patella_Rotate_L",
    "Patella_R": "Patella_Rotate_R",
    "Tibia_L": "Tibia_Bone_L",
    "Tibia_R": "Tibia_Bone_R",
    "Fibula_L": "Tibia_Twist_L",
    "Fibula_R": "Tibia_Twist_R",
}


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mesh_vertex_ids(asset: AnatomyRiggedAsset, mesh_name: str) -> np.ndarray:
    names = [str(x) for x in asset.source_mesh_names]
    if mesh_name not in names:
        raise KeyError(f"mesh not in asset: {mesh_name}")
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[names.index(mesh_name)]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _mesh_local_faces(asset: AnatomyRiggedAsset, vertex_ids: np.ndarray) -> np.ndarray:
    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    global_to_local = np.full(int(np.asarray(asset.vertices_rest).shape[0]), -1, dtype=np.int64)
    global_to_local[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int64)
    mask = np.all(global_to_local[faces] >= 0, axis=1)
    local = global_to_local[faces[mask]]
    if local.size == 0:
        raise ValueError("selected mesh has no faces")
    return local


def _graph_laplacian(n_vertices: int, local_faces: np.ndarray) -> sparse.csr_matrix:
    edges = np.concatenate(
        (
            local_faces[:, (0, 1)],
            local_faces[:, (1, 2)],
            local_faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree) - adjacency


def _segment_distance(points: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(head, dtype=np.float64).reshape(3)
    b = np.asarray(tail, dtype=np.float64).reshape(3)
    ab = b - a
    length2 = float(np.dot(ab, ab))
    if length2 <= 1.0e-16:
        return np.linalg.norm(pts - a[None, :], axis=1)
    t = np.clip(np.einsum("ij,j->i", pts - a[None, :], ab) / length2, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - closest, axis=1)


def _axis_parameter(points: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(head, dtype=np.float64).reshape(3)
    b = np.asarray(tail, dtype=np.float64).reshape(3)
    ab = b - a
    length2 = float(np.dot(ab, ab))
    if length2 <= 1.0e-16:
        return np.zeros(len(pts), dtype=np.float64)
    return np.clip(np.einsum("ij,j->i", pts - a[None, :], ab) / length2, 0.0, 1.0)


def _softmax_neg_dist(distances: np.ndarray, temperature_m: float) -> np.ndarray:
    scaled = -np.asarray(distances, dtype=np.float64) / max(float(temperature_m), 1.0e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)


def _harmonic_blend_channels(
    laplacian: sparse.csr_matrix,
    *,
    seed: np.ndarray,
    pinned: np.ndarray,
) -> np.ndarray:
    """Diffuse seed weights; pinned vertices keep their seed values."""

    values = np.asarray(seed, dtype=np.float64).copy()
    n_vertices, n_channels = values.shape
    free = np.flatnonzero(~np.asarray(pinned, dtype=bool))
    if free.size == 0:
        return values
    system = laplacian[free][:, free].tocsc()
    fixed = np.flatnonzero(np.asarray(pinned, dtype=bool))
    for channel in range(n_channels):
        rhs = -laplacian[free][:, fixed] @ values[fixed, channel]
        solved = np.asarray(spsolve(system, rhs), dtype=np.float64)
        if not np.all(np.isfinite(solved)):
            raise ValueError("weight harmonic solve produced non-finite values")
        values[free, channel] = solved
    values = np.clip(values, 0.0, None)
    row_sum = values.sum(axis=1, keepdims=True)
    values /= np.maximum(row_sum, 1.0e-12)
    return values


def _pack_top_slots(
    bone_ids: np.ndarray,
    weights: np.ndarray,
    *,
    n_slots: int = N_DRIVER_SLOTS,
) -> tuple[np.ndarray, np.ndarray]:
    bone_ids = np.asarray(bone_ids, dtype=np.int64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64)
    n_vertices = int(w.shape[0])
    out_i = np.zeros((n_vertices, n_slots), dtype=np.int32)
    out_w = np.zeros((n_vertices, n_slots), dtype=np.float32)
    for vi in range(n_vertices):
        order = np.argsort(-w[vi])
        keep = order[:n_slots]
        chosen_bones = bone_ids[keep]
        chosen_w = w[vi, keep]
        chosen_w = np.clip(chosen_w, 0.0, None)
        total = float(chosen_w.sum())
        if total <= 1.0e-12:
            # fall back to strongest bone with unit weight
            out_i[vi, 0] = int(bone_ids[int(order[0])])
            out_w[vi, 0] = 1.0
            continue
        chosen_w /= total
        out_i[vi, : len(keep)] = chosen_bones.astype(np.int32)
        out_w[vi, : len(keep)] = chosen_w.astype(np.float32)
    return out_i, out_w


def _dense_from_slots(
    indices: np.ndarray,
    weights: np.ndarray,
    bone_ids: np.ndarray,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    bones = np.asarray(bone_ids, dtype=np.int64).reshape(-1)
    lookup = {int(b): i for i, b in enumerate(bones.tolist())}
    dense = np.zeros((idx.shape[0], len(bones)), dtype=np.float64)
    for vi in range(idx.shape[0]):
        for slot in range(idx.shape[1]):
            bi = int(idx[vi, slot])
            wi = float(w[vi, slot])
            if wi <= 0.0 or bi not in lookup:
                continue
            dense[vi, lookup[bi]] += wi
    row = dense.sum(axis=1, keepdims=True)
    dense /= np.maximum(row, 1.0e-12)
    return dense


@dataclass(frozen=True)
class WeightRefitV1:
    subject_label: str
    target_meshes: tuple[str, ...]
    affected_vertex_ids: np.ndarray
    driver_indices_refit: np.ndarray
    driver_weights_refit: np.ndarray
    source_driver_indices: np.ndarray
    source_driver_weights: np.ndarray
    candidate_bone_names: tuple[str, ...]
    prior_strength: float
    temperature_m: float
    core_axis_frac: float
    build_report: dict[str, Any]

    def apply_to_asset(self, asset: AnatomyRiggedAsset) -> AnatomyRiggedAsset:
        indices = np.asarray(asset.driver_indices, dtype=np.int32).copy()
        weights = np.asarray(asset.driver_weights, dtype=np.float32).copy()
        if indices.shape != self.driver_indices_refit.shape:
            raise ValueError("refit indices shape mismatch vs asset")
        if weights.shape != self.driver_weights_refit.shape:
            raise ValueError("refit weights shape mismatch vs asset")
        # identity check for untouched rows
        affected = np.asarray(self.affected_vertex_ids, dtype=np.int64)
        indices[affected] = self.driver_indices_refit[affected]
        weights[affected] = self.driver_weights_refit[affected]
        meta = dict(asset.metadata or {})
        meta["weight_refit_v1"] = {
            "kind": REFIT_WEIGHTS_KIND,
            "schema": REFIT_WEIGHTS_SCHEMA,
            "subject_label": self.subject_label,
            "n_affected": int(affected.size),
            "publishable": False,
        }
        return replace(
            asset,
            driver_indices=indices,
            driver_weights=weights,
            runtime_driver_indices_compressed=None,
            runtime_driver_weights_compressed=None,
            metadata=meta,
        )


def build_weight_refit_v1(
    asset: AnatomyRiggedAsset,
    *,
    subject_label: str,
    rest_vertices: np.ndarray | None = None,
    target_meshes: tuple[str, ...] = DEFAULT_TARGET_MESHES,
    prior_strength: float = 0.35,
    temperature_m: float = 0.025,
    core_axis_frac: float = 0.55,
) -> WeightRefitV1:
    """Refit selected knee-chain rows with Pinocchio-style distance+harmonic weights."""

    started = time.perf_counter()
    bone_names = [str(x) for x in (asset.source_bone_names or [])]
    if len(bone_names) != 235:
        raise ValueError(f"expected 235 controllers, got {len(bone_names)}")
    heads = np.asarray(asset.source_bone_head, dtype=np.float64)
    tails = np.asarray(asset.source_bone_tail, dtype=np.float64)
    rest = np.asarray(
        asset.vertices_rest if rest_vertices is None else rest_vertices,
        dtype=np.float64,
    )
    src_i = np.asarray(asset.driver_indices, dtype=np.int32).copy()
    src_w = np.asarray(asset.driver_weights, dtype=np.float32).copy()
    out_i = src_i.copy()
    out_w = src_w.copy()

    affected: list[int] = []
    mesh_reports: dict[str, Any] = {}
    used_bones: set[str] = set()

    for mesh_name in target_meshes:
        side = "L" if mesh_name.endswith("_L") else "R"
        candidates = _SIDE_CANDIDATES[side]
        for name in candidates:
            if name not in bone_names:
                raise KeyError(f"missing controller {name}")
        cand_ids = np.asarray([bone_names.index(n) for n in candidates], dtype=np.int64)
        used_bones.update(candidates)
        vids = _mesh_vertex_ids(asset, mesh_name)
        local_faces = _mesh_local_faces(asset, vids)
        lap = _graph_laplacian(len(vids), local_faces)
        pts = rest[vids]

        dist = np.stack(
            [_segment_distance(pts, heads[int(b)], tails[int(b)]) for b in cand_ids.tolist()],
            axis=1,
        )
        heat = _softmax_neg_dist(dist, temperature_m)

        blender = _dense_from_slots(src_i[vids], src_w[vids], cand_ids)
        # If blender had mass only on bones outside the candidate set, keep a
        # residual channel on the mesh core bone so rows stay normalized.
        core_name = _MESH_CORE_BONE[mesh_name]
        core_id = bone_names.index(core_name)
        core_col = int(np.where(cand_ids == core_id)[0][0])
        blender_sum = blender.sum(axis=1)
        thin = blender_sum < 0.5
        blender[thin, core_col] = 1.0
        blender /= np.maximum(blender.sum(axis=1, keepdims=True), 1.0e-12)

        mixed = float(prior_strength) * blender + (1.0 - float(prior_strength)) * heat
        mixed = np.clip(mixed, 0.0, None)
        mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1.0e-12)

        # Pin proximal/core vertices to authored-dominant behavior.
        axis_u = _axis_parameter(pts, heads[core_id], tails[core_id])
        knee_names = [n for n in candidates if n.startswith("Knee_Rotate")]
        knee_col = candidates.index(knee_names[0]) if knee_names else -1
        use_harmonic = True
        if mesh_name.startswith("Femur"):
            # Authored Femur is ~100% Femur_Rot; pure distance heat never
            # assigns Knee_Rotate because the femur shaft segment dominates.
            # Force a distal axial blend onto Knee_Rotate. Do NOT Dirichlet-
            # harmonicize channel-wise: pinned core=(1,0,...) forces the
            # Knee_Rotate field to identically zero on the free band.
            pinned = axis_u <= float(core_axis_frac)
            if knee_col < 0:
                raise RuntimeError(f"{mesh_name}: Knee_Rotate candidate missing")
            span = max(1.0e-6, 1.0 - float(core_axis_frac))
            blend = np.clip((axis_u - float(core_axis_frac)) / span, 0.0, 1.0)
            blend = blend * blend  # ease-in toward condyles
            distal_knee_mass = 0.70
            mixed[:] = 0.0
            mixed[:, core_col] = 1.0 - blend * distal_knee_mass
            mixed[:, knee_col] = blend * distal_knee_mass
            use_harmonic = False
        elif mesh_name.startswith("Patella"):
            pinned = np.ones(len(vids), dtype=bool)  # keep patella carrier
            mixed[:] = 0.0
            mixed[:, core_col] = 1.0
            use_harmonic = False
        elif mesh_name.startswith("Tibia") or mesh_name.startswith("Fibula"):
            pinned = axis_u <= 0.35
        else:
            pinned = np.zeros(len(vids), dtype=bool)

        # Enforce pin seeds: core bone = 1 on pinned verts.
        mixed[pinned] = 0.0
        mixed[pinned, core_col] = 1.0
        if use_harmonic:
            mixed = _harmonic_blend_channels(lap, seed=mixed, pinned=pinned)
        else:
            mixed = np.clip(mixed, 0.0, None)
            mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1.0e-12)

        packed_i, packed_w = _pack_top_slots(cand_ids, mixed)
        out_i[vids] = packed_i
        out_w[vids] = packed_w
        affected.extend(vids.tolist())

        # Diagnostics vs authored.
        src_dom = src_i[vids, 0]
        new_dom = packed_i[:, 0]
        knee_names = [n for n in candidates if n.startswith("Knee_Rotate")]
        knee_id = bone_names.index(knee_names[0]) if knee_names else -1
        knee_mass = 0.0
        if knee_id >= 0:
            knee_mass = float(
                np.sum(packed_w * (packed_i == knee_id)) / max(len(vids), 1)
            )
        mesh_reports[mesh_name] = {
            "n_vertices": int(len(vids)),
            "n_pinned": int(np.count_nonzero(pinned)),
            "mean_knee_rotate_weight": knee_mass,
            "dominant_bone_changed_fraction": float(np.mean(src_dom != new_dom)),
            "candidates": list(candidates),
        }

    affected_ids = np.unique(np.asarray(affected, dtype=np.int64))
    # Untouched rows must remain bit-identical.
    untouched = np.ones(len(out_i), dtype=bool)
    untouched[affected_ids] = False
    if not np.array_equal(out_i[untouched], src_i[untouched]):
        raise RuntimeError("refit mutated untouched driver_indices")
    if not np.allclose(out_w[untouched], src_w[untouched], atol=0.0, rtol=0.0):
        raise RuntimeError("refit mutated untouched driver_weights")

    report = {
        "kind": REFIT_WEIGHTS_KIND,
        "schema": REFIT_WEIGHTS_SCHEMA,
        "subject_label": subject_label,
        "publishable": False,
        "runtime_enabled": False,
        "trusted_latest_updated": False,
        "n_bones": 235,
        "n_affected_vertices": int(affected_ids.size),
        "prior_strength": float(prior_strength),
        "temperature_m": float(temperature_m),
        "core_axis_frac": float(core_axis_frac),
        "target_meshes": list(target_meshes),
        "meshes": mesh_reports,
        "faces_digest": _array_digest(asset.faces),
        "source_weights_digest": _array_digest(src_w),
        "refit_weights_digest": _array_digest(out_w),
        "source_indices_digest": _array_digest(src_i),
        "refit_indices_digest": _array_digest(out_i),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return WeightRefitV1(
        subject_label=str(subject_label),
        target_meshes=tuple(target_meshes),
        affected_vertex_ids=affected_ids,
        driver_indices_refit=out_i,
        driver_weights_refit=out_w,
        source_driver_indices=src_i,
        source_driver_weights=src_w,
        candidate_bone_names=tuple(sorted(used_bones)),
        prior_strength=float(prior_strength),
        temperature_m=float(temperature_m),
        core_axis_frac=float(core_axis_frac),
        build_report=report,
    )


def save_weight_refit_v1(path: Path, value: WeightRefitV1) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        path / "refit_weights.npz",
        affected_vertex_ids=value.affected_vertex_ids.astype(np.int64),
        driver_indices_refit=value.driver_indices_refit,
        driver_weights_refit=value.driver_weights_refit,
        source_driver_indices=value.source_driver_indices,
        source_driver_weights=value.source_driver_weights,
    )
    (path / "manifest.json").write_text(
        json.dumps(value.build_report, indent=2, sort_keys=True) + "\n"
    )


def load_weight_refit_v1(path: Path) -> WeightRefitV1:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "refit_weights.npz") as data:
        return WeightRefitV1(
            subject_label=str(manifest["subject_label"]),
            target_meshes=tuple(manifest["target_meshes"]),
            affected_vertex_ids=np.asarray(data["affected_vertex_ids"], dtype=np.int64),
            driver_indices_refit=np.asarray(data["driver_indices_refit"], dtype=np.int32),
            driver_weights_refit=np.asarray(data["driver_weights_refit"], dtype=np.float32),
            source_driver_indices=np.asarray(data["source_driver_indices"], dtype=np.int32),
            source_driver_weights=np.asarray(data["source_driver_weights"], dtype=np.float32),
            candidate_bone_names=tuple(manifest.get("candidate_bone_names", ())),
            prior_strength=float(manifest["prior_strength"]),
            temperature_m=float(manifest["temperature_m"]),
            core_axis_frac=float(manifest["core_axis_frac"]),
            build_report=manifest,
        )


__all__ = [
    "DEFAULT_TARGET_MESHES",
    "REFIT_WEIGHTS_KIND",
    "WeightRefitV1",
    "build_weight_refit_v1",
    "load_weight_refit_v1",
    "save_weight_refit_v1",
]
