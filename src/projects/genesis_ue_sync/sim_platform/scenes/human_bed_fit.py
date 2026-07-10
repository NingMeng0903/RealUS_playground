from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry_support import lower_shell_mask, support_plane_shift_masked
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence, evaluate_smpl_sequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import ProxyGeometry, sample_proxy_surface_points
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec


@dataclass(frozen=True)
class BedPlacementResult:
    world_offset: tuple[float, float, float]
    support_plane_z: float
    support_shift_z: float
    center_shift_xy: tuple[float, float]
    support_contact_ratio: float
    penetration_depth_m: float
    floating_height_m: float
    sample_indices: tuple[int, ...]
    lower_shell_snap_dz_m: float = 0.0


def _subset_sequence(sequence: HumanMotionSequence, indices: list[int]) -> HumanMotionSequence:
    idx = np.asarray(indices, dtype=np.int64)
    image_names = [sequence.image_names[i] for i in idx.tolist()] if sequence.image_names else []
    cam_int = sequence.cam_int[idx] if sequence.cam_int is not None else None
    cam_ext = sequence.cam_ext[idx] if sequence.cam_ext is not None else None
    return HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=f"{sequence.sequence_name}_bedfit",
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender,
        betas=np.asarray(sequence.betas, dtype=np.float32).copy(),
        poses=np.asarray(sequence.poses[idx], dtype=np.float32).copy(),
        trans=np.asarray(sequence.trans[idx], dtype=np.float32).copy(),
        image_names=image_names,
        cam_int=cam_int,
        cam_ext=cam_ext,
        metadata=dict(sequence.metadata),
    )


def _sample_indices(frame_count: int, sample_count: int) -> list[int]:
    n = max(int(frame_count), 1)
    k = max(1, min(int(sample_count), n))
    if k == 1:
        return [0]
    vals = np.linspace(0, n - 1, num=k)
    out: list[int] = []
    seen: set[int] = set()
    for v in vals:
        i = int(round(float(v)))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out or [0]


def _proxy_lower_mask(points: np.ndarray, quantile: float = 0.22) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    z = np.asarray(points[:, 2], dtype=np.float64)
    thr = float(np.quantile(z, float(np.clip(quantile, 0.05, 0.45))))
    return z <= thr


def _fit_xy_to_bed(
    vertices_seq: np.ndarray,
    *,
    target_center_xy: np.ndarray,
    bed_center_xy: np.ndarray,
    bed_size_xy: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    centers: list[np.ndarray] = []
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for vertices in vertices_seq:
        mask = lower_shell_mask(vertices)
        pts = vertices[mask] if np.any(mask) else vertices
        centers.append(np.mean(pts[:, :2], axis=0))
        mins.append(np.min(pts[:, :2], axis=0))
        maxs.append(np.max(pts[:, :2], axis=0))
    center_xy = np.median(np.stack(centers, axis=0), axis=0)
    delta = np.asarray(target_center_xy, dtype=np.float64) - center_xy
    bed_half = 0.5 * np.asarray(bed_size_xy, dtype=np.float64)
    allow_min = bed_center_xy - bed_half + float(margin_m)
    allow_max = bed_center_xy + bed_half - float(margin_m)
    shifted_min = np.min(np.stack(mins, axis=0) + delta[None, :], axis=0)
    shifted_max = np.max(np.stack(maxs, axis=0) + delta[None, :], axis=0)
    if shifted_min[0] < allow_min[0]:
        delta[0] += allow_min[0] - shifted_min[0]
    if shifted_max[0] > allow_max[0]:
        delta[0] -= shifted_max[0] - allow_max[0]
    if shifted_min[1] < allow_min[1]:
        delta[1] += allow_min[1] - shifted_min[1]
    if shifted_max[1] > allow_max[1]:
        delta[1] -= shifted_max[1] - allow_max[1]
    return delta.astype(np.float32)


def fit_human_sequence_to_bed(
    sequence: HumanMotionSequence,
    *,
    scene_spec: SyncSceneSpec,
    proxy_geometry: ProxyGeometry,
    device: str | None = "cpu",
    sample_count: int = 7,
    support_percentile: float = 4.0,
    support_band_m: float = 0.03,
    center_margin_m: float = 0.05,
    snap_lower_shell_min_z: bool = True,
    target_center_xy: tuple[float, float] | np.ndarray | None = None,
) -> BedPlacementResult:
    if scene_spec.support_surface is None:
        raise RuntimeError("Scene must define support_surface for bed fitting.")
    idx = _sample_indices(sequence.frame_count, sample_count)
    subset = _subset_sequence(sequence, idx)
    vertices_seq, joints_seq = evaluate_smpl_sequence(
        subset,
        device=device,
        include_vertices=True,
        include_joints=True,
    )
    assert vertices_seq is not None and joints_seq is not None
    bed_center_xy = np.asarray(scene_spec.support_surface.pos[:2], dtype=np.float32)
    bed_size_xy = np.asarray(scene_spec.support_surface.size[:2], dtype=np.float32)
    target_center_xy_arr = (
        np.asarray(target_center_xy, dtype=np.float32).reshape(2)
        if target_center_xy is not None
        else np.asarray(scene_spec.resolved_human_anchor()[:2], dtype=np.float32)
    )
    support_plane_z = float(scene_spec.support_surface_top_z + scene_spec.human.support_margin_m)

    delta_xy = _fit_xy_to_bed(
        vertices_seq,
        target_center_xy=target_center_xy_arr,
        bed_center_xy=bed_center_xy,
        bed_size_xy=bed_size_xy,
        margin_m=center_margin_m,
    )

    support_shifts: list[float] = []
    support_contact: list[float] = []
    penetrations: list[float] = []
    floating: list[float] = []
    for vertices, joints in zip(vertices_seq, joints_seq):
        verts = np.asarray(vertices, dtype=np.float32).copy()
        verts[:, :2] += delta_xy[None, :]
        vmask = lower_shell_mask(verts)
        dz = support_plane_shift_masked(
            verts,
            support_plane_z,
            vmask,
            percentile=float(support_percentile),
        )
        support_shifts.append(float(dz))

        cloud = sample_proxy_surface_points(joints, proxy_geometry)
        if cloud.size == 0:
            continue
        cloud = np.asarray(cloud, dtype=np.float32).copy()
        cloud[:, :2] += delta_xy[None, :]
        cloud[:, 2] += float(dz)
        cmask = _proxy_lower_mask(cloud)
        pts = cloud[cmask] if np.any(cmask) else cloud
        gap = np.asarray(pts[:, 2] - support_plane_z, dtype=np.float32)
        support_contact.append(float(np.mean(np.abs(gap) <= float(support_band_m))))
        penetrations.append(float(np.mean(np.clip(-gap, 0.0, None))))
        floating.append(float(np.mean(np.clip(gap, 0.0, None))))

    dz_final = float(np.median(np.asarray(support_shifts, dtype=np.float32)))
    snap_dz = 0.0
    if snap_lower_shell_min_z:
        touch_idx = int(idx[0])
        touch_sub = _subset_sequence(sequence, [touch_idx])
        v_touch, _ = evaluate_smpl_sequence(
            touch_sub,
            device=device,
            include_vertices=True,
            include_joints=False,
        )
        assert v_touch is not None
        verts = np.asarray(v_touch[0], dtype=np.float32).copy()
        verts[:, :2] += delta_xy[None, :]
        verts[:, 2] += dz_final
        vmask = lower_shell_mask(verts)
        pts = verts[vmask] if np.any(vmask) else verts
        min_z = float(np.min(pts[:, 2]))
        snap_dz = float(support_plane_z - min_z)
        dz_final += snap_dz

    return BedPlacementResult(
        world_offset=(float(delta_xy[0]), float(delta_xy[1]), dz_final),
        support_plane_z=support_plane_z,
        support_shift_z=float(dz_final),
        center_shift_xy=(float(delta_xy[0]), float(delta_xy[1])),
        support_contact_ratio=float(np.mean(support_contact)) if support_contact else 0.0,
        penetration_depth_m=float(np.mean(penetrations)) if penetrations else 0.0,
        floating_height_m=float(np.mean(floating)) if floating else 0.0,
        sample_indices=tuple(int(i) for i in idx),
        lower_shell_snap_dz_m=float(snap_dz),
    )
