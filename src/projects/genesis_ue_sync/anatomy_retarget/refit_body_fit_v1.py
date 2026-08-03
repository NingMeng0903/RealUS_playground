"""V10 offline body-fit: multipose LS rest bake + knee weight refit.

Offline only. Runtime remains Blender 235 FK + 14-slot LBS — no LS/SDF at pose
time. Training poses are bake evidence so the fixed rest+weights generalize.

Candidate-only. publishable=false.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _weighted_rest_correction
from .cli.run_amass_bedlam_retarget_matrix_v6 import (
    COUPLED_RBF_SUPPORT_RAD,
    _pick_frame,
    _pose_within_coupled_support,
    _resolve_bedlam_file,
)
from .dynamic_main_chain_retarget_v2 import _signed_distance_details
from .pose_adapter import (
    easymocap_fit_to_smplx55,
    pose_to_smplx55_axis_angle,
    smplh156_to_smplx55,
)
from .pose_map_v1 import PoseMapV1, apply_pose_map_global
from .refit_weights_v1 import WeightRefitV1, build_weight_refit_v1
from .rigged_asset import AnatomyRiggedAsset
from .smplx_body_surface_v7 import smplx_body_surface_v7


BODY_FIT_KIND = "BodyFitRefitV1"
BODY_FIT_SCHEMA = 2
KNEE_MESHES = (
    "Femur_L",
    "Patella_L",
    "Tibia_L",
    "Fibula_L",
    "Femur_R",
    "Patella_R",
    "Tibia_R",
    "Fibula_R",
)

# Mild locomotion / general — avoid crouch-to-walk extremes as bake cores.
DEFAULT_AMASS_RELS: tuple[str, ...] = (
    "ACCAD/Female1Walking_c3d/B1 - stand to walk_poses.npz",
    "ACCAD/Female1Walking_c3d/B4 - stand to walk back_poses.npz",
    "ACCAD/Female1Walking_c3d/B10 - walk turn left (45)_poses.npz",
    "ACCAD/Female1Walking_c3d/B12 - walk turn right (90)_poses.npz",
    "ACCAD/Female1Walking_c3d/B13 - walk turn right (45)_poses.npz",
    "ACCAD/Female1Walking_c3d/B15 - walk turn around (same direction)_poses.npz",
    "ACCAD/Female1Walking_c3d/B22 - side step left_poses.npz",
    "ACCAD/Female1Walking_c3d/B23 - side step right_poses.npz",
    "ACCAD/Female1General_c3d/A1 - Stand_poses.npz",
    "ACCAD/Female1General_c3d/A2 - Sway_poses.npz",
    "ACCAD/Male1Walking_c3d/Walk B10 - Walk turn left 45_poses.npz",
    "ACCAD/Male1Walking_c3d/Walk B22 - Side step left_poses.npz",
    "ACCAD/Male1General_c3d/General A1 - Stand_poses.npz",
    "ACCAD/Male2Walking_c3d/B5 -  Walk backwards_poses.npz",
    "ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses.npz",
    "ACCAD/Male2Walking_c3d/B13 -  Walk turn right 90_poses.npz",
    "ACCAD/Male2Walking_c3d/B22 -  side step left_poses.npz",
    "ACCAD/Male2Walking_c3d/B23 -  side step right_poses.npz",
    "ACCAD/Male2General_c3d/A1- Stand_poses.npz",
)
DEFAULT_BEDLAM_NAMES: tuple[str, ...] = (
    "it_4051_3XL_2304.npz",
    "it_4011_XL_2114.npz",
    "it_4019_2XL_2203.npz",
    "it_4046_2XL_2106.npz",
)


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mesh_ids(asset: AnatomyRiggedAsset, mesh_name: str) -> np.ndarray:
    names = [str(x) for x in asset.source_mesh_names]
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[names.index(mesh_name)]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _active_vertex_ids(asset: AnatomyRiggedAsset, meshes: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([_mesh_ids(asset, m) for m in meshes]))


def _mesh_adjacency(asset: AnatomyRiggedAsset, vertex_ids: np.ndarray) -> list[list[int]]:
    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    global_to_local = {int(v): i for i, v in enumerate(vertex_ids.tolist())}
    adj: list[set[int]] = [set() for _ in range(len(vertex_ids))]
    for tri in faces:
        locals_ = [global_to_local[int(v)] for v in tri.tolist() if int(v) in global_to_local]
        if len(locals_) < 2:
            continue
        for a in locals_:
            for b in locals_:
                if a != b:
                    adj[a].add(b)
    return [sorted(s) for s in adj]


def _smooth_displacements(
    disp: np.ndarray,
    adjacency: list[list[int]],
    *,
    iterations: int = 1,
) -> np.ndarray:
    values = np.asarray(disp, dtype=np.float64).copy()
    for _ in range(int(iterations)):
        nxt = values.copy()
        for i, nbrs in enumerate(adjacency):
            if not nbrs:
                continue
            nxt[i] = 0.5 * values[i] + 0.5 * np.mean(values[nbrs], axis=0)
        values = nxt
    return values


def _blended_affines(
    transforms: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    tf = np.asarray(transforms, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros((len(idx), 4, 4), dtype=np.float64)
    for slot in range(idx.shape[1]):
        wi = w[:, slot]
        if not np.any(wi > 0.0):
            continue
        out += wi[:, None, None] * tf[idx[:, slot]]
    return out


def _load_motion_frames(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        arr = data["poses"] if "poses" in data.files else data[data.files[0]]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return pose_to_smplx55_axis_angle(arr).reshape(1, 55, 3)
    if arr.shape[-1] == 165:
        return arr.reshape(arr.shape[0], 55, 3)
    if arr.shape[-1] == 156:
        return np.stack(
            [smplh156_to_smplx55(row) for row in arr.reshape(arr.shape[0], 156)],
            axis=0,
        )
    raise ValueError(f"unsupported motion pose width {arr.shape[-1]} in {path}")


def _clamp_pose_to_coupled_support(pose: np.ndarray, *, scale: float = 0.995) -> np.ndarray:
    """Keep body joints strictly inside coupled-RBF support (offline bake only)."""

    out = np.asarray(pose, dtype=np.float32).copy().reshape(55, 3)
    out[22] = 0.0
    body = out[1:22]
    norms = np.linalg.norm(body, axis=1, keepdims=True)
    limit = float(COUPLED_RBF_SUPPORT_RAD) * float(scale)
    body_scale = np.minimum(1.0, limit / np.maximum(norms, 1.0e-12))
    out[1:22] = body * body_scale
    return out


def _pick_percentile_frames(
    frames: np.ndarray,
    *,
    kind: str,
    percentiles: Sequence[float],
) -> list[tuple[int, np.ndarray]]:
    """Sample several in-support frames by joint-energy percentile (offline bake)."""

    frames = np.asarray(frames, dtype=np.float64)
    if kind == "upper":
        energy = np.linalg.norm(
            frames[:, [16, 17, 18, 19, 20, *range(25, 55)], :].reshape(len(frames), -1),
            axis=1,
        )
    elif kind == "lower":
        energy = np.linalg.norm(
            frames[:, [1, 2, 4, 5, 7, 8, 10, 11], :].reshape(len(frames), -1),
            axis=1,
        )
    else:
        energy = np.linalg.norm(frames.reshape(len(frames), -1), axis=1)
    support = np.array(
        [_pose_within_coupled_support(frame) for frame in frames], dtype=bool
    )
    if np.any(support):
        pool = np.flatnonzero(support)
    else:
        pool = np.arange(len(frames), dtype=np.int64)
    order = pool[np.argsort(energy[pool])]
    out: list[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    for pct in percentiles:
        rank = int(round((len(order) - 1) * float(pct) / 100.0))
        rank = int(np.clip(rank, 0, len(order) - 1))
        index = int(order[rank])
        if index in seen:
            continue
        seen.add(index)
        pose = _clamp_pose_to_coupled_support(frames[index])
        out.append((index, pose))
    return out


def _max_body_joint_rad(pose: np.ndarray) -> float:
    body = np.asarray(pose, dtype=np.float64).reshape(55, 3)[1:22]
    return float(np.linalg.norm(body, axis=1).max())


def _max_knee_rad(pose: np.ndarray) -> float:
    # SMPL-X knee joints: 4=left_knee, 5=right_knee
    joints = np.asarray(pose, dtype=np.float64).reshape(55, 3)
    return float(max(np.linalg.norm(joints[4]), np.linalg.norm(joints[5])))


def build_body_fit_pose_catalog_v1(
    *,
    repo_root: Path,
    model_path: Path,
    amass_root: Path | None = None,
    bedlam_root: Path | None = None,
    amass_rels: Sequence[str] = DEFAULT_AMASS_RELS,
    bedlam_names: Sequence[str] = DEFAULT_BEDLAM_NAMES,
    # Mild only: no high-energy peak / deep-flex extremes.
    percentiles: Sequence[float] = (15.0, 35.0, 55.0),
    max_knee_rad: float = float(np.radians(55.0)),
    max_body_joint_rad: float = float(np.radians(70.0)),
    include_peak_bedlam: bool = False,
) -> dict[str, np.ndarray]:
    """Offline bake pose catalog: captures + mild AMASS/BEDLAM (hard poses dropped)."""

    root = Path(repo_root)
    amass_root = Path(
        amass_root
        or "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/amass_hf/raw"
    )
    bedlam_root = Path(
        bedlam_root
        or "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions"
    )
    poses: dict[str, np.ndarray] = {"tpose": np.zeros((55, 3), np.float32)}
    for label, rel in (
        ("pose_213328", "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"),
        ("pose_213712", "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"),
    ):
        with np.load(root / rel) as data:
            poses[label] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            ).astype(np.float32)

    def _accept(pose: np.ndarray) -> np.ndarray | None:
        clamped = _clamp_pose_to_coupled_support(pose)
        if _max_knee_rad(clamped) > float(max_knee_rad) + 1.0e-9:
            return None
        if _max_body_joint_rad(clamped) > float(max_body_joint_rad) + 1.0e-9:
            return None
        return clamped

    rejected = 0
    for rel in amass_rels:
        path = amass_root / rel
        if not path.is_file():
            continue
        frames = _load_motion_frames(path)
        for index, pose in _pick_percentile_frames(
            frames, kind="full", percentiles=percentiles
        ):
            kept = _accept(pose)
            if kept is None:
                rejected += 1
                continue
            key = f"amass_{Path(rel).stem}_f{index}".replace(" ", "_")
            poses[key] = kept

    for name in bedlam_names:
        try:
            path = _resolve_bedlam_file(bedlam_root, name)
        except FileNotFoundError:
            continue
        frames = _load_motion_frames(path)
        kind = "lower" if "4051" in name or "4011" in name else "upper"
        for index, pose in _pick_percentile_frames(
            frames, kind=kind, percentiles=percentiles
        ):
            kept = _accept(pose)
            if kept is None:
                rejected += 1
                continue
            poses[f"bedlam_{Path(name).stem}_{kind}_f{index}"] = kept

    if include_peak_bedlam:
        for name, kind in (
            ("it_4051_3XL_2304.npz", "lower"),
            ("it_4011_XL_2114.npz", "lower"),
        ):
            try:
                path = _resolve_bedlam_file(bedlam_root, name)
            except FileNotFoundError:
                continue
            with np.load(path, allow_pickle=True) as data:
                arr = data["poses"] if "poses" in data.files else data[data.files[0]]
            fi, pose = _pick_frame(arr, kind=kind)
            kept = _accept(pose)
            if kept is None:
                rejected += 1
                continue
            poses[f"bedlam_{Path(name).stem}_{kind}_peak_f{fi}"] = kept

    for key in ("pose_213328", "pose_213712"):
        kept = _accept(poses[key])
        if kept is None:
            # Captures are required; soft-clamp only.
            poses[key] = _clamp_pose_to_coupled_support(poses[key])
        else:
            poses[key] = kept
    print(
        f"pose catalog: kept={len(poses)} rejected_hard={rejected} "
        f"(knee<{np.degrees(max_knee_rad):.0f}deg, no peak bedlam)",
        flush=True,
    )
    return poses


def _pose_knee_metrics(
    *,
    rest: np.ndarray,
    asset: AnatomyRiggedAsset,
    pose_map: PoseMapV1,
    pose: np.ndarray,
    model: Any,
    betas: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    pose = _clamp_pose_to_coupled_support(pose)
    skin, faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
    posed_global = apply_pose_map_global(
        pose_map, source_asset=asset, pose_axis_angle=pose
    )
    transforms = posed_global @ pose_map.target_inverse_bind
    posed = _weighted_rest_correction(
        rest, asset.driver_indices, asset.driver_weights, transforms
    )
    signed, closest, gradient = _signed_distance_details(posed[active], skin, faces)
    out = np.maximum(signed, 0.0)
    per_mesh: dict[str, float] = {}
    for mesh in KNEE_MESHES:
        ids = _mesh_ids(asset, mesh)
        mask = np.isin(active, ids)
        per_mesh[mesh] = float(out[mask].max()) if np.any(mask) else 0.0
    return {
        "max_outside_m": float(out.max()) if len(out) else 0.0,
        "n_outside": int(np.count_nonzero(out > 1.0e-4)),
        "meshes": per_mesh,
        "skin": skin,
        "faces": faces,
        "posed": posed,
        "transforms": transforms,
        "signed_active": signed,
        "closest": closest,
        "gradient": gradient,
    }


def filter_poses_by_baseline_outside(
    poses: Mapping[str, np.ndarray],
    *,
    rest: np.ndarray,
    asset: AnatomyRiggedAsset,
    pose_map: PoseMapV1,
    model: Any,
    betas: np.ndarray,
    active: np.ndarray,
    max_baseline_outside_m: float = 0.030,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Drop bake poses whose baseline knee outside is too hard to compromise."""

    kept: dict[str, np.ndarray] = {}
    dropped: list[dict[str, Any]] = []
    required = {"tpose", "pose_213328", "pose_213712"}
    for pose_id, pose in poses.items():
        metrics = _pose_knee_metrics(
            rest=rest,
            asset=asset,
            pose_map=pose_map,
            pose=np.asarray(pose, dtype=np.float32),
            model=model,
            betas=betas,
            active=active,
        )
        out = float(metrics["max_outside_m"])
        if out > float(max_baseline_outside_m) and str(pose_id) not in required:
            dropped.append({"pose_id": str(pose_id), "baseline_outside_m": out})
            continue
        kept[str(pose_id)] = np.asarray(pose, dtype=np.float32)
    print(
        f"baseline filter: kept={len(kept)} dropped_hard={len(dropped)} "
        f"(max_baseline_outside<{max_baseline_outside_m*1000:.0f}mm)",
        flush=True,
    )
    return kept, dropped


@dataclass(frozen=True)
class BodyFitRefitV1:
    subject_label: str
    weight_refit: WeightRefitV1
    vertices_final_refit: np.ndarray
    vertices_final_source: np.ndarray
    affected_vertex_ids: np.ndarray
    build_report: dict[str, Any]

    def apply_to_asset(self, asset: AnatomyRiggedAsset) -> AnatomyRiggedAsset:
        return self.weight_refit.apply_to_asset(asset)

    def apply_to_subject(self, value: ChainRestFitSubjectV1) -> ChainRestFitSubjectV1:
        src = np.asarray(value.vertices_final, dtype=np.float64)
        if src.shape != self.vertices_final_refit.shape:
            raise ValueError("subject vertex count mismatch")
        moved = np.unique(
            np.concatenate(
                (
                    np.asarray(value.moved_vertex_ids, dtype=np.int64),
                    np.asarray(self.affected_vertex_ids, dtype=np.int64),
                )
            )
        )
        report = dict(value.build_report or {})
        report["body_fit_refit_v1"] = {
            "kind": BODY_FIT_KIND,
            "schema": BODY_FIT_SCHEMA,
            "publishable": False,
            "n_affected": int(len(self.affected_vertex_ids)),
            "max_rest_displacement_m": float(
                np.linalg.norm(
                    self.vertices_final_refit - self.vertices_final_source, axis=1
                ).max()
            ),
        }
        return replace(
            value,
            vertices_final=np.asarray(self.vertices_final_refit, dtype=np.float32),
            moved_vertex_ids=moved.astype(np.int32),
            build_report=report,
        )


def build_body_fit_v1(
    asset: AnatomyRiggedAsset,
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    subject_label: str,
    model: Any,
    betas: np.ndarray,
    poses: Mapping[str, np.ndarray],
    target_meshes: tuple[str, ...] = KNEE_MESHES,
    prior_strength: float = 0.15,
    temperature_m: float = 0.025,
    core_axis_frac: float = 0.40,
    ls_iterations: int = 16,
    inset_margin_m: float = 0.0005,
    max_rest_step_m: float = 0.035,
    max_rest_total_m: float = 0.150,
    smooth_iterations: int = 1,
    outside_weight_gain: float = 20.0,
    inside_prior_weight: float = 0.15,
    # backward-compatible aliases
    inset_iterations: int | None = None,
    inset_step: float | None = None,
    max_rest_step_m_alias: float | None = None,
) -> BodyFitRefitV1:
    """Offline multipose LS rest bake. Does not run at pose time."""

    del inset_step, max_rest_step_m_alias
    if inset_iterations is not None:
        ls_iterations = int(inset_iterations)
    started = time.perf_counter()
    weight_refit = build_weight_refit_v1(
        asset,
        subject_label=subject_label,
        rest_vertices=np.asarray(value.vertices_final, dtype=np.float64),
        target_meshes=tuple(target_meshes),
        prior_strength=float(prior_strength),
        temperature_m=float(temperature_m),
        core_axis_frac=float(core_axis_frac),
    )
    asset_w = weight_refit.apply_to_asset(asset)

    rest0 = np.asarray(value.vertices_final, dtype=np.float64).copy()
    rest = rest0.copy()
    active = _active_vertex_ids(asset_w, target_meshes)
    adjacency = _mesh_adjacency(asset_w, active)
    pose_items = list(poses.items())
    history: list[dict[str, Any]] = []

    # Drop too-hard SMPL-X examples before LS (captures always kept).
    poses, dropped_hard = filter_poses_by_baseline_outside(
        dict(poses),
        rest=rest0,
        asset=asset_w,
        pose_map=pose_map,
        model=model,
        betas=betas,
        active=active,
        max_baseline_outside_m=0.030,
    )
    pose_items = list(poses.items())
    print(
        f"body_fit LS bake: n_poses={len(pose_items)} n_active={len(active)} "
        f"iters={ls_iterations} dropped_hard={len(dropped_hard)}",
        flush=True,
    )

    for it in range(int(ls_iterations)):
        pose_pack: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        pose_stats: dict[str, Any] = {}
        for pose_id, pose in pose_items:
            metrics = _pose_knee_metrics(
                rest=rest,
                asset=asset_w,
                pose_map=pose_map,
                pose=np.asarray(pose, dtype=np.float32),
                model=model,
                betas=betas,
                active=active,
            )
            pose_stats[pose_id] = {
                "max_outside_m": metrics["max_outside_m"],
                "n_outside": metrics["n_outside"],
                "meshes": metrics["meshes"],
            }
            affines = _blended_affines(
                metrics["transforms"],
                np.asarray(asset_w.driver_indices)[active],
                np.asarray(asset_w.driver_weights)[active],
            )
            pose_pack.append(
                (
                    pose_id,
                    metrics["signed_active"],
                    metrics["closest"],
                    metrics["gradient"],
                    affines,
                )
            )

        disp = np.zeros((len(active), 3), dtype=np.float64)
        for li in range(len(active)):
            rows: list[np.ndarray] = []
            rhs: list[np.ndarray] = []
            for _pid, signed, closest, gradient, affines in pose_pack:
                R = affines[li, :3, :3]
                t = affines[li, :3, 3]
                cur = R @ rest[active[li]] + t
                if float(signed[li]) > float(inset_margin_m):
                    target = closest[li] - float(inset_margin_m) * gradient[li]
                    weight = 1.0 + float(outside_weight_gain) * float(signed[li])
                else:
                    target = cur
                    weight = float(inside_prior_weight)
                rows.append(weight * R)
                rhs.append(weight * (target - t))
            A = np.concatenate(rows, axis=0)
            b = np.concatenate(rhs, axis=0)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            disp[li] = sol - rest[active[li]]

        disp = _smooth_displacements(disp, adjacency, iterations=int(smooth_iterations))
        already = np.linalg.norm(rest[active] - rest0[active], axis=1)
        step = np.linalg.norm(disp, axis=1)
        cap = float(max_rest_step_m) if it >= 4 else float(max_rest_step_m) * 1.25
        over = step > cap
        disp[over] *= (cap / np.maximum(step[over], 1.0e-12))[:, None]
        remain = np.maximum(float(max_rest_total_m) - already, 0.0)
        sn = np.linalg.norm(disp, axis=1)
        need = (already + sn) > float(max_rest_total_m)
        disp[need] *= (remain[need] / np.maximum(sn[need], 1.0e-12))[:, None]
        rest[active] = rest[active] + disp

        post_stats: dict[str, Any] = {}
        for pose_id, pose in pose_items:
            metrics = _pose_knee_metrics(
                rest=rest,
                asset=asset_w,
                pose_map=pose_map,
                pose=np.asarray(pose, dtype=np.float32),
                model=model,
                betas=betas,
                active=active,
            )
            post_stats[pose_id] = {
                "max_outside_m": metrics["max_outside_m"],
                "n_outside": metrics["n_outside"],
                "meshes": metrics["meshes"],
            }
        worst = max(float(v["max_outside_m"]) for v in post_stats.values())
        history.append(
            {
                "iteration": it,
                "worst_max_outside_m": worst,
                "mean_step_m": float(np.linalg.norm(disp, axis=1).mean()),
                "max_step_m": float(np.linalg.norm(disp, axis=1).max()),
                "poses_before": pose_stats,
                "poses_after": {
                    k: {"max_outside_m": v["max_outside_m"], "n_outside": v["n_outside"]}
                    for k, v in post_stats.items()
                },
            }
        )
        print(
            f"body_fit LS iter {it}: worst={worst*1000:.2f}mm "
            f"max_step={float(np.linalg.norm(disp, axis=1).max())*1000:.2f}mm "
            f"max_disp={float(np.linalg.norm(rest[active]-rest0[active], axis=1).max())*1000:.1f}mm",
            flush=True,
        )
        if worst <= float(inset_margin_m) + 5.0e-4:
            break

    final_poses = {
        pose_id: {
            "max_outside_m": cell["max_outside_m"],
            "n_outside": cell["n_outside"],
            "meshes": cell.get("meshes"),
        }
        for pose_id, cell in history[-1]["poses_after"].items()
    } if history else {}
    # refresh meshes on a cheap path from last post_stats if present
    if history:
        # recompute once with meshes for report completeness on key poses only
        final_poses = {}
        for pose_id, pose in pose_items:
            metrics = _pose_knee_metrics(
                rest=rest,
                asset=asset_w,
                pose_map=pose_map,
                pose=np.asarray(pose, dtype=np.float32),
                model=model,
                betas=betas,
                active=active,
            )
            final_poses[pose_id] = {
                "max_outside_m": metrics["max_outside_m"],
                "n_outside": metrics["n_outside"],
                "meshes": metrics["meshes"],
            }

    disp_norm = np.linalg.norm(rest - rest0, axis=1)
    affected = active[np.linalg.norm(rest[active] - rest0[active], axis=1) > 1.0e-7]
    report = {
        "kind": BODY_FIT_KIND,
        "schema": BODY_FIT_SCHEMA,
        "solver": "multipose_ls_rest_offline",
        "subject_label": subject_label,
        "publishable": False,
        "runtime_enabled": False,
        "trusted_latest_updated": False,
        "runtime_path": "blender_235_fk_plus_14slot_lbs",
        "n_train_poses": len(pose_items),
        "train_pose_ids": [str(k) for k, _ in pose_items],
        "target_meshes": list(target_meshes),
        "n_affected_vertices": int(len(affected)),
        "max_rest_displacement_m": float(disp_norm.max()),
        "mean_rest_displacement_m": float(disp_norm[active].mean()),
        "ls_iterations_ran": len(history),
        "inset_margin_m": float(inset_margin_m),
        "weight_refit": weight_refit.build_report,
        "history": history,
        "final_poses": final_poses,
        "worst_final_outside_m": float(
            max(v["max_outside_m"] for v in final_poses.values())
        )
        if final_poses
        else None,
        "source_rest_digest": _array_digest(rest0),
        "refit_rest_digest": _array_digest(rest),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return BodyFitRefitV1(
        subject_label=str(subject_label),
        weight_refit=weight_refit,
        vertices_final_refit=rest.astype(np.float64),
        vertices_final_source=rest0.astype(np.float64),
        affected_vertex_ids=np.asarray(affected, dtype=np.int64),
        build_report=report,
    )


def save_body_fit_v1(path: Path, value: BodyFitRefitV1) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        path / "body_fit.npz",
        vertices_final_refit=value.vertices_final_refit.astype(np.float32),
        vertices_final_source=value.vertices_final_source.astype(np.float32),
        affected_vertex_ids=value.affected_vertex_ids.astype(np.int64),
        driver_indices_refit=value.weight_refit.driver_indices_refit,
        driver_weights_refit=value.weight_refit.driver_weights_refit,
    )
    (path / "manifest.json").write_text(
        json.dumps(value.build_report, indent=2, sort_keys=True) + "\n"
    )


def sample_body_fit_betas_v1(
    *,
    repo_root: Path,
    amass_root: Path | None = None,
    n_amass_betas: int = 10,
) -> list[dict[str, Any]]:
    """Capture betas + many distinct AMASS identity betas (≥8 required in contract)."""

    if int(n_amass_betas) < 8:
        raise ValueError("n_amass_betas must be >= 8 (do not ship a 2-beta toy set)")

    root = Path(repo_root)
    amass_root = Path(
        amass_root
        or "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/amass_hf/raw"
    )
    rows: list[dict[str, Any]] = []
    for label, rel in (
        ("213328", "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"),
        ("213712", "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"),
    ):
        path = root / rel
        with np.load(path) as data:
            betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
        rows.append(
            {
                "subject_label": label,
                "source": "capture",
                "betas": betas,
                "path": str(path),
            }
        )
    exclude = [np.asarray(r["betas"], dtype=np.float64) for r in rows]
    # Prefer locomotion/general identities; martial-arts betas allowed for shape diversity.
    preferred = (
        "ACCAD/Female1Walking_c3d/B1 - stand to walk_poses.npz",
        "ACCAD/Female1General_c3d/A1 - Stand_poses.npz",
        "ACCAD/Female1Gestures_c3d/D1 - Urban 1_poses.npz",
        "ACCAD/Female1Running_c3d/C10 -  run backwards stop run forward_poses.npz",
        "ACCAD/Male1General_c3d/General A1 - Stand_poses.npz",
        "ACCAD/Male1Walking_c3d/Male1 Cal_poses.npz",
        "ACCAD/Male1Running_c3d/Run C24 - quick side step left_poses.npz",
        "ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses.npz",
        "ACCAD/Male2General_c3d/A1- Stand_poses.npz",
        "ACCAD/Male2Running_c3d/C1 - stand to run_poses.npz",
        "ACCAD/Male2MartialArtsStances_c3d/D1 - stand to ready_poses.npz",
        "ACCAD/s001/EricCamper04_poses.npz",
        "ACCAD/s007/QkWalk1_poses.npz",
        "ACCAD/s008/Run1_poses.npz",
        "ACCAD/s011/walkdog_poses.npz",
    )
    for rel in preferred:
        if len([r for r in rows if r["source"] == "amass"]) >= int(n_amass_betas):
            break
        path = amass_root / rel
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            if "betas" not in data.files:
                continue
            betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:10]
        if any(np.allclose(betas, other, atol=1.0e-5) for other in exclude):
            continue
        stem = Path(rel).parts[1] if len(Path(rel).parts) > 1 else Path(rel).stem
        label = f"amass_{stem}"
        # uniquify
        base = label
        k = 2
        while any(r["subject_label"] == label for r in rows):
            label = f"{base}_{k}"
            k += 1
        rows.append(
            {
                "subject_label": label,
                "source": "amass",
                "betas": betas,
                "path": str(path),
            }
        )
        exclude.append(betas)
    if len([r for r in rows if r["source"] == "amass"]) < int(n_amass_betas):
        for path in sorted((amass_root / "ACCAD").rglob("*_poses.npz")):
            if len([r for r in rows if r["source"] == "amass"]) >= int(n_amass_betas):
                break
            with np.load(path, allow_pickle=True) as data:
                if "betas" not in data.files:
                    continue
                betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:10]
            if any(np.allclose(betas, other, atol=1.0e-5) for other in exclude):
                continue
            label = f"amass_{path.parent.name}"
            base = label
            k = 2
            while any(r["subject_label"] == label for r in rows):
                label = f"{base}_{k}"
                k += 1
            rows.append(
                {
                    "subject_label": label,
                    "source": "amass",
                    "betas": betas,
                    "path": str(path),
                }
            )
            exclude.append(betas)
    n_amass = len([r for r in rows if r["source"] == "amass"])
    if n_amass < int(n_amass_betas):
        raise RuntimeError(
            f"only found {n_amass} distinct AMASS betas, need >= {n_amass_betas}"
        )
    print(
        f"beta catalog: captures=2 amass={n_amass} total={len(rows)}",
        flush=True,
    )
    return rows


__all__ = [
    "BODY_FIT_KIND",
    "BodyFitRefitV1",
    "DEFAULT_AMASS_RELS",
    "DEFAULT_BEDLAM_NAMES",
    "KNEE_MESHES",
    "build_body_fit_pose_catalog_v1",
    "build_body_fit_v1",
    "filter_poses_by_baseline_outside",
    "sample_body_fit_betas_v1",
    "save_body_fit_v1",
]
