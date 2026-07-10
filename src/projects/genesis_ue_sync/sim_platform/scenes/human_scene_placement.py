"""Canonical human placement artifact shared by Genesis and UE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence, evaluate_smpl_sequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import ProxyGeometry, build_proxy_geometry_for_sequence
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec
from projects.genesis_ue_sync.sim_platform.scenes.human_bed_fit import BedPlacementResult, fit_human_sequence_to_bed


SCHEMA_VERSION = 1


def _scene_fit_revision_token(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class HumanFramePlacement:
    frame_index: int
    root_translation_world_m: tuple[float, float, float]
    pelvis_world_m: tuple[float, float, float]
    mesh_aabb_min_m: tuple[float, float, float]
    mesh_aabb_max_m: tuple[float, float, float]
    mesh_lowest_z_m: float


@dataclass(frozen=True)
class HumanScenePlacement:
    """Single source of truth for SMPL root trajectory after scene fitting (Genesis canonical world, meters)."""

    output_world_convention: str
    scene_fit_revision: str
    world_offset_m: tuple[float, float, float]
    support_plane_z_m: float
    human_anchor_world_m: tuple[float, float, float]
    align_floor: bool
    display_vertical_offset_m: float
    display_vertical_sink_m: float
    display_pitch_forward_deg: float
    sequence_npz_path: str
    bed_placement: dict[str, Any]
    sampled_frames: tuple[HumanFramePlacement, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "output_world_convention": self.output_world_convention,
            "scene_fit_revision": self.scene_fit_revision,
            "world_offset_m": list(self.world_offset_m),
            "support_plane_z_m": float(self.support_plane_z_m),
            "human_anchor_world_m": list(self.human_anchor_world_m),
            "align_floor": bool(self.align_floor),
            "display_vertical_offset_m": float(self.display_vertical_offset_m),
            "display_vertical_sink_m": float(self.display_vertical_sink_m),
            "display_pitch_forward_deg": float(self.display_pitch_forward_deg),
            "sequence_npz_path": self.sequence_npz_path,
            "bed_placement": dict(self.bed_placement),
            "sampled_frames": [
                {
                    "frame_index": f.frame_index,
                    "root_translation_world_m": list(f.root_translation_world_m),
                    "pelvis_world_m": list(f.pelvis_world_m),
                    "mesh_aabb_min_m": list(f.mesh_aabb_min_m),
                    "mesh_aabb_max_m": list(f.mesh_aabb_max_m),
                    "mesh_lowest_z_m": float(f.mesh_lowest_z_m),
                }
                for f in self.sampled_frames
            ],
            "metadata": dict(self.metadata),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HumanScenePlacement:
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported human_scene_placement schema: {payload.get('schema_version')}")
        frames_raw = payload.get("sampled_frames") or []
        sampled: list[HumanFramePlacement] = []
        for fr in frames_raw:
            sampled.append(
                HumanFramePlacement(
                    frame_index=int(fr["frame_index"]),
                    root_translation_world_m=tuple(float(v) for v in fr["root_translation_world_m"]),
                    pelvis_world_m=tuple(float(v) for v in fr["pelvis_world_m"]),
                    mesh_aabb_min_m=tuple(float(v) for v in fr["mesh_aabb_min_m"]),
                    mesh_aabb_max_m=tuple(float(v) for v in fr["mesh_aabb_max_m"]),
                    mesh_lowest_z_m=float(fr["mesh_lowest_z_m"]),
                )
            )
        return cls(
            output_world_convention=str(payload["output_world_convention"]),
            scene_fit_revision=str(payload["scene_fit_revision"]),
            world_offset_m=tuple(float(v) for v in payload["world_offset_m"]),
            support_plane_z_m=float(payload["support_plane_z_m"]),
            human_anchor_world_m=tuple(float(v) for v in payload["human_anchor_world_m"]),
            align_floor=bool(payload.get("align_floor", True)),
            display_vertical_offset_m=float(payload.get("display_vertical_offset_m", 0.0)),
            display_vertical_sink_m=float(payload.get("display_vertical_sink_m", 0.0)),
            display_pitch_forward_deg=float(payload.get("display_pitch_forward_deg", 0.0)),
            sequence_npz_path=str(payload["sequence_npz_path"]),
            bed_placement=dict(payload.get("bed_placement") or {}),
            sampled_frames=tuple(sampled),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: Path) -> HumanScenePlacement:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object in {path}")
        return cls.from_dict(payload)


def _bed_placement_to_dict(result: BedPlacementResult) -> dict[str, Any]:
    return {
        "world_offset": list(result.world_offset),
        "support_plane_z": float(result.support_plane_z),
        "support_shift_z": float(result.support_shift_z),
        "center_shift_xy": list(result.center_shift_xy),
        "support_contact_ratio": float(result.support_contact_ratio),
        "penetration_depth_m": float(result.penetration_depth_m),
        "floating_height_m": float(result.floating_height_m),
        "sample_indices": list(result.sample_indices),
        "lower_shell_snap_dz_m": float(result.lower_shell_snap_dz_m),
    }


def _sample_frame_indices(frame_count: int, *, max_samples: int) -> list[int]:
    n = max(int(frame_count), 1)
    k = max(1, min(int(max_samples), n))
    if k >= n:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, num=k)})


def resolve_placement_target_center_xy(
    scene_spec: SyncSceneSpec,
    *,
    human_center_mode: str = "bed_center",
) -> tuple[float, float]:
    """XY target for bed-fit: bed center (default) or scene human anchor."""
    if str(human_center_mode).strip().lower() == "bed_center" and scene_spec.support_surface is not None:
        pos = scene_spec.support_surface.pos
        return float(pos[0]), float(pos[1])
    anchor = scene_spec.resolved_human_anchor()
    return float(anchor[0]), float(anchor[1])


def apply_first_frame_root_projection_xy(
    sequence: HumanMotionSequence,
    scene_spec: SyncSceneSpec,
    world_offset: tuple[float, float, float],
    *,
    device: str | None = "cpu",
    use_smpl_joint0_root: bool = True,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Shift XY so frame-0 root/pelvis lies on bed center (Genesis canonical meters)."""
    if scene_spec.support_surface is None:
        return world_offset, (0.0, 0.0)
    ox, oy, oz = (float(world_offset[0]), float(world_offset[1]), float(world_offset[2]))
    bed_center_xy = np.asarray(scene_spec.support_surface.pos[:2], dtype=np.float32).reshape(2)
    wo = np.asarray([ox, oy, oz], dtype=np.float32)
    root_xy = np.asarray(sequence.trans[0, :2], dtype=np.float32) + wo[:2]
    if use_smpl_joint0_root:
        try:
            sub = HumanMotionSequence(
                source_dataset=sequence.source_dataset,
                sequence_name=f"{sequence.sequence_name}_root_proj",
                source_path=sequence.source_path,
                model_type=sequence.model_type,
                fps=sequence.fps,
                gender=sequence.gender,
                betas=np.asarray(sequence.betas, dtype=np.float32).copy(),
                poses=np.asarray(sequence.poses[:1], dtype=np.float32),
                trans=np.asarray(sequence.trans[:1], dtype=np.float32),
                image_names=[sequence.image_names[0]] if sequence.image_names else [],
                cam_int=sequence.cam_int[:1] if sequence.cam_int is not None else None,
                cam_ext=sequence.cam_ext[:1] if sequence.cam_ext is not None else None,
                metadata=dict(sequence.metadata),
            )
            _, joints = evaluate_smpl_sequence(
                sub,
                device=device,
                include_vertices=False,
                include_joints=True,
            )
            if joints is not None and joints.ndim == 3 and joints.shape[1] > 0:
                root_xy = np.asarray(joints[0, 0, :2], dtype=np.float32) + wo[:2]
        except Exception:
            pass
    delta_xy = bed_center_xy - root_xy
    return (float(ox + delta_xy[0]), float(oy + delta_xy[1]), float(oz)), (
        float(delta_xy[0]),
        float(delta_xy[1]),
    )


def compute_human_scene_placement(
    sequence: HumanMotionSequence,
    *,
    scene_spec: SyncSceneSpec,
    sequence_npz_path: str,
    device: str | None = "cpu",
    placement_sample_frames: int = 11,
    proxy_geometry: ProxyGeometry | None = None,
    human_center_mode: str = "bed_center",
    root_projection_bed_center: bool = True,
    fit_samples: int | None = None,
    support_band_m: float = 0.03,
    center_margin_m: float = 0.05,
) -> HumanScenePlacement:
    if proxy_geometry is None:
        proxy_geometry = build_proxy_geometry_for_sequence(sequence, device=device)
    target_center_xy = resolve_placement_target_center_xy(scene_spec, human_center_mode=human_center_mode)
    sample_count = int(fit_samples) if fit_samples is not None else 7
    bed_fit = fit_human_sequence_to_bed(
        sequence,
        scene_spec=scene_spec,
        proxy_geometry=proxy_geometry,
        device=device,
        sample_count=sample_count,
        support_band_m=float(support_band_m),
        center_margin_m=float(center_margin_m),
        target_center_xy=target_center_xy,
    )
    ox, oy, oz = bed_fit.world_offset
    root_proj_delta = (0.0, 0.0)
    if bool(root_projection_bed_center):
        (ox, oy, oz), root_proj_delta = apply_first_frame_root_projection_xy(
            sequence,
            scene_spec,
            (ox, oy, oz),
            device=device,
        )
    anchor = scene_spec.resolved_human_anchor()

    revision = _scene_fit_revision_token(
        sequence_npz_path,
        str(scene_spec.name),
        str(human_center_mode),
        str(bool(root_projection_bed_center)),
        f"{ox:.6f},{oy:.6f},{oz:.6f}",
        f"{bed_fit.support_plane_z:.6f}",
        json.dumps(_bed_placement_to_dict(bed_fit), sort_keys=True),
    )

    indices = _sample_frame_indices(sequence.frame_count, max_samples=placement_sample_frames)
    sub = HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=sequence.sequence_name + "_placement_samples",
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender,
        betas=np.asarray(sequence.betas, dtype=np.float32).copy(),
        poses=np.asarray(sequence.poses[indices], dtype=np.float32),
        trans=np.asarray(sequence.trans[indices], dtype=np.float32),
        image_names=[sequence.image_names[i] for i in indices] if sequence.image_names else [],
        cam_int=sequence.cam_int[np.asarray(indices)] if sequence.cam_int is not None else None,
        cam_ext=sequence.cam_ext[np.asarray(indices)] if sequence.cam_ext is not None else None,
        metadata=dict(sequence.metadata),
    )
    verts, joints = evaluate_smpl_sequence(sub, device=device, include_vertices=True, include_joints=True)
    assert verts is not None and joints is not None

    dx, dy, dz = float(ox), float(oy), float(oz)
    sampled: list[HumanFramePlacement] = []
    for row, fi in enumerate(indices):
        v_raw = np.asarray(verts[row], dtype=np.float64)
        dz_floor = float(np.percentile(v_raw[:, 2], 2.0)) if scene_spec.human.align_floor else 0.0
        extra_z = float(scene_spec.human.display_vertical_sink_m + scene_spec.human.display_vertical_offset_m)

        pelvis = np.asarray(joints[row, 0], dtype=np.float64).copy()
        pelvis[2] -= dz_floor
        pelvis[0] += dx
        pelvis[1] += dy
        pelvis[2] += dz + extra_z

        v = v_raw.copy()
        v[:, 2] -= dz_floor
        v[:, 0] += dx
        v[:, 1] += dy
        v[:, 2] += dz + extra_z

        amin = np.min(v, axis=0)
        amax = np.max(v, axis=0)
        sampled.append(
            HumanFramePlacement(
                frame_index=int(fi),
                root_translation_world_m=tuple(float(x) for x in pelvis.reshape(-1).tolist()),
                pelvis_world_m=tuple(float(x) for x in pelvis.reshape(-1).tolist()),
                mesh_aabb_min_m=tuple(float(x) for x in amin.tolist()),
                mesh_aabb_max_m=tuple(float(x) for x in amax.tolist()),
                mesh_lowest_z_m=float(np.min(v[:, 2])),
            )
        )

    return HumanScenePlacement(
        output_world_convention="genesis_canonical_rh_z_up_m",
        scene_fit_revision=revision,
        world_offset_m=(float(ox), float(oy), float(oz)),
        support_plane_z_m=float(bed_fit.support_plane_z),
        human_anchor_world_m=tuple(float(v) for v in anchor),
        align_floor=bool(scene_spec.human.align_floor),
        display_vertical_offset_m=float(scene_spec.human.display_vertical_offset_m),
        display_vertical_sink_m=float(scene_spec.human.display_vertical_sink_m),
        display_pitch_forward_deg=float(scene_spec.human.display_pitch_forward_deg),
        sequence_npz_path=str(sequence_npz_path),
        bed_placement=_bed_placement_to_dict(bed_fit),
        sampled_frames=tuple(sampled),
        metadata={
            "support_surface_top_z": float(scene_spec.support_surface_top_z) if scene_spec.support_surface else None,
            "placement_sample_frames": int(placement_sample_frames),
            "proxy_shape_key": getattr(proxy_geometry, "shape_key", ""),
            "human_center_mode": str(human_center_mode),
            "root_projection_bed_center": bool(root_projection_bed_center),
            "root_projection_delta_xy": list(root_proj_delta),
            "target_center_xy": list(target_center_xy),
        },
    )


def load_human_scene_placement_from_scene_metadata(scene_spec: SyncSceneSpec, *, repo_root: Path) -> HumanScenePlacement | None:
    rel = scene_spec.metadata.get("human_scene_placement")
    if not rel:
        return None
    candidate = Path(str(rel))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.is_file():
        return None
    return HumanScenePlacement.load(candidate)
