"""Genesis debug-mesh drawer for retargeted anatomy assets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_pose_hash
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset, load_rigged_asset
from projects.genesis_ue_sync.multiview_realtime.viz.debug_mesh_draw import replace_colored_debug_mesh
from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime


def _rgba_float_to_uint8(color: tuple[float, float, float, float], opacity: float | None = None) -> np.ndarray:
    rgba = np.asarray(color, dtype=np.float32).reshape(4).copy()
    if opacity is not None:
        rgba[3] = float(opacity)
    if float(np.max(rgba)) <= 1.0:
        rgba = rgba * 255.0
    return np.clip(rgba, 0, 255).astype(np.uint8)


_TISSUE_RGBA = {
    "artery": (0.92, 0.05, 0.05, 0.88),
    "vein": (0.08, 0.25, 0.95, 0.88),
    "bone": (0.94, 0.92, 0.86, 0.95),
    "heart": (0.92, 0.05, 0.05, 0.90),
    "organ": (0.55, 0.55, 0.55, 0.75),
    "nerve": (0.95, 0.85, 0.20, 0.80),
    "connective_tissue": (0.70, 0.55, 0.40, 0.55),
    "default": (0.80, 0.05, 0.05, 0.85),
}


def _vertices_to_genesis_z_up(
    vertices: np.ndarray,
    *,
    coordinate_system: str,
) -> np.ndarray:
    """Map asset coordinates into the Genesis viewer frame.

    Live track overlays still draw SMPL-X meshes in ``smplx_y_up_m``, so anatomy
    must stay in that frame until both paths share the same viewer conversion.
    """
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    coordinate = str(coordinate_system)
    if coordinate in {"smplx_y_up_m", "genesis_z_up_m"}:
        return points.copy()
    raise ValueError(f"unsupported anatomy coordinate system: {coordinate!r}")


def _mesh_color_rgba(mesh_name: str, tissue: str) -> tuple[float, float, float, float]:
    lower = str(mesh_name).lower()
    tissue_key = str(tissue).lower()
    if tissue_key == "heart" or "heart" in lower:
        return _TISSUE_RGBA["heart"]
    if tissue_key == "bone":
        return _TISSUE_RGBA["bone"]
    if tissue_key == "organ":
        return _TISSUE_RGBA["organ"]
    if tissue_key == "nerve":
        return _TISSUE_RGBA["nerve"]
    if tissue_key == "connective_tissue":
        return _TISSUE_RGBA["connective_tissue"]
    if tissue_key == "vessel":
        if "vein" in lower:
            return _TISSUE_RGBA["vein"]
        if "arter" in lower:
            return _TISSUE_RGBA["artery"]
        return _TISSUE_RGBA["artery"]
    return _TISSUE_RGBA["default"]


def _vertex_colors_for_asset(
    asset: AnatomyRiggedAsset,
    *,
    fallback_rgba: tuple[float, float, float, float],
    opacity: float,
) -> np.ndarray:
    vertex_count = len(asset.vertices_rest)
    colors = np.tile(
        _rgba_float_to_uint8(fallback_rgba, opacity),
        (vertex_count, 1),
    )
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return colors
    mesh_names = asset.source_mesh_names or [""] * len(asset.source_tissues)
    for (start, stop), tissue, mesh_name in zip(
        asset.source_vertex_ranges, asset.source_tissues, mesh_names
    ):
        rgba = _rgba_float_to_uint8(_mesh_color_rgba(str(mesh_name), str(tissue)), opacity)
        colors[int(start) : int(stop)] = rgba
    return colors


def _reviewed_hidden_face_ids_v2(
    asset: AnatomyRiggedAsset,
    *,
    faces: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    raw = np.asarray(metadata.get("hidden_face_ids_v2", []))
    if not raw.size:
        return np.zeros(0, dtype=np.int64)
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("hidden_face_ids_v2 must contain integer face indices")
    face_ids = raw.astype(np.int64, copy=False).reshape(-1)
    if np.any(face_ids < 0) or np.any(face_ids >= len(faces)):
        raise ValueError("hidden_face_ids_v2 contains an invalid face index")
    if len(np.unique(face_ids)) != len(face_ids):
        raise ValueError("hidden_face_ids_v2 contains duplicate face indices")
    if not np.array_equal(face_ids, np.sort(face_ids)):
        raise ValueError("hidden_face_ids_v2 must be sorted")

    policy = metadata.get("oral_visibility_policy_v2")
    if not isinstance(policy, dict):
        raise ValueError(
            "hidden_face_ids_v2 requires oral_visibility_policy_v2 metadata"
        )
    if int(policy.get("hidden_face_count", -1)) != len(face_ids):
        raise ValueError("oral visibility hidden face count disagrees with draw list")
    expected_digest = str(policy.get("hidden_face_ids_sha256", ""))
    digest = hashlib.sha256(
        np.ascontiguousarray(face_ids, dtype="<i4").tobytes()
    ).hexdigest()
    if not expected_digest or digest != expected_digest:
        raise ValueError("oral visibility hidden face digest disagrees with draw list")

    reviewed_names = [
        str(value)
        for value in policy.get("hidden_face_source_mesh_names", [])
    ]
    expected_counts = {
        str(name): int(value)
        for name, value in dict(
            policy.get("hidden_face_counts_by_mesh", {})
        ).items()
    }
    if not reviewed_names or set(reviewed_names) != set(expected_counts):
        raise ValueError("oral visibility reviewed face domains are incomplete")
    if (
        asset.source_vertex_ranges is None
        or asset.source_tissues is None
        or not asset.source_mesh_names
    ):
        raise ValueError(
            "reviewed hidden faces require complete source mesh metadata"
        )
    mesh_names = list(asset.source_mesh_names)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = list(asset.source_tissues)
    if len(ranges) != len(mesh_names) or len(tissues) != len(mesh_names):
        raise ValueError("reviewed hidden face source metadata is incomplete")

    reviewed_face_mask = np.zeros(len(faces), dtype=bool)
    actual_counts: dict[str, int] = {}
    for mesh_name in reviewed_names:
        try:
            mesh_index = mesh_names.index(mesh_name)
        except ValueError as exc:
            raise ValueError(
                f"reviewed hidden face mesh {mesh_name!r} is missing"
            ) from exc
        if str(tissues[mesh_index]).strip().lower() != "organ":
            raise ValueError(
                f"reviewed hidden face mesh {mesh_name!r} is not organ tissue"
            )
        start, stop = (int(value) for value in ranges[mesh_index])
        mesh_face_mask = np.all(
            (faces >= start) & (faces < stop),
            axis=1,
        )
        reviewed_face_mask |= mesh_face_mask
        actual_counts[mesh_name] = int(
            np.count_nonzero(mesh_face_mask[face_ids])
        )
    if not np.all(reviewed_face_mask[face_ids]):
        raise ValueError(
            "hidden_face_ids_v2 includes a face outside reviewed organ domains"
        )
    if actual_counts != expected_counts:
        raise ValueError("oral visibility per-mesh face counts changed")

    hidden_mesh_names_v2 = {
        str(value) for value in metadata.get("hidden_mesh_names_v2", [])
    }
    reviewed_whole_mesh_names = {
        str(name)
        for name in dict(
            policy.get("hidden_whole_mesh_face_counts", {})
        )
    }
    if hidden_mesh_names_v2 != reviewed_whole_mesh_names:
        raise ValueError("oral visibility whole-mesh exclusions changed")
    return face_ids


def render_faces_for_asset_v2(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return the Genesis draw list after reviewed tissue exclusions."""

    faces = np.asarray(asset.faces, dtype=np.int32)
    metadata = dict(asset.metadata or {})
    excluded_faces = np.zeros(len(faces), dtype=bool)
    face_ids = _reviewed_hidden_face_ids_v2(
        asset,
        faces=faces,
        metadata=metadata,
    )
    if len(face_ids):
        excluded_faces[face_ids] = True
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return faces[~excluded_faces]
    hidden = np.zeros(len(asset.vertices_rest), dtype=bool)
    show_connective = bool(metadata.get("show_connective_tissue", False))
    show_vessels = bool(metadata.get("show_vessels", True))
    hidden_mesh_names = {
        str(value)
        for key in ("hidden_mesh_names_v1", "hidden_mesh_names_v2")
        for value in metadata.get(key, [])
    }
    mesh_names = list(
        asset.source_mesh_names or [""] * len(asset.source_tissues)
    )
    for (start, stop), tissue, mesh_name in zip(
        asset.source_vertex_ranges,
        asset.source_tissues,
        mesh_names,
    ):
        tissue_key = str(tissue)
        if str(mesh_name) in hidden_mesh_names:
            hidden[int(start) : int(stop)] = True
        elif tissue_key == "connective_tissue" and not show_connective:
            hidden[int(start) : int(stop)] = True
        elif tissue_key == "vessel" and not show_vessels:
            hidden[int(start) : int(stop)] = True
    excluded_faces |= np.any(hidden[faces], axis=1)
    return faces[~excluded_faces]


class AnatomyLbsDrawer:
    def __init__(
        self,
        runtime: GenesisPlatformRuntime,
        *,
        asset: AnatomyRiggedAsset,
        model_id: str,
        color_rgba: tuple[float, float, float, float] = (0.8, 0.05, 0.05, 0.85),
    ) -> None:
        self.runtime = runtime
        self.asset = asset
        self.model_id = str(model_id)
        self.default_color_rgba = tuple(float(v) for v in color_rgba)
        self.opacity = float(color_rgba[3])
        self.visible = True
        self._mesh_node: Any = None
        self._last_pose: np.ndarray | None = None
        self._last_transl: np.ndarray | None = None

    def _render_faces(self) -> np.ndarray:
        """Hide optional tissue layers without deleting them from the asset."""
        return render_faces_for_asset_v2(self.asset)

    @classmethod
    def from_npz(
        cls,
        runtime: GenesisPlatformRuntime,
        *,
        path: Path | str,
        model_id: str,
        color_rgba: tuple[float, float, float, float] = (0.8, 0.05, 0.05, 0.85),
    ) -> "AnatomyLbsDrawer":
        return cls(runtime, asset=load_rigged_asset(path), model_id=model_id, color_rgba=color_rgba)

    def clear_node(self) -> None:
        if self._mesh_node is None:
            return
        node = self._mesh_node
        for attempt in (0.15, 0.35, 0.75):
            with try_viewer_render_lock(self.runtime, timeout_s=attempt) as acquired:
                if not acquired:
                    continue
                try:
                    self.runtime.scene.clear_debug_object(node)
                except Exception:
                    pass
                else:
                    self._mesh_node = None
                    return
        self._mesh_node = None

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if not self.visible:
            self.clear_node()
        else:
            self.redraw_last()

    def set_opacity(self, opacity: float) -> None:
        self.opacity = float(max(0.0, min(1.0, float(opacity))))
        if self.opacity <= 0.0:
            self.clear_node()
        else:
            self.visible = True
            self.redraw_last()

    def restore_opacity(self) -> None:
        self.opacity = float(self.default_color_rgba[3])
        self.visible = True
        self.redraw_last()

    def set_render_mode(self, mode: str, *, transparent_alpha: float = 0.35) -> None:
        text = str(mode).strip().lower()
        if text == "hidden":
            self.set_visible(False)
        elif text == "transparent":
            self.set_opacity(float(transparent_alpha))
        elif text == "opaque":
            self.set_opacity(1.0)
        else:
            raise ValueError(f"Unsupported anatomy render mode: {mode}")

    def redraw_last(self) -> bool:
        if self._last_pose is None:
            return False
        return self.draw(self._last_pose, transl=self._last_transl, force=True)

    def draw(self, pose_axis_angle: Any, *, transl: Any | None = None, force: bool = False) -> bool:
        pose = np.asarray(pose_axis_angle, dtype=np.float32).reshape(-1)
        new_transl = None if transl is None else np.asarray(transl, dtype=np.float32).reshape(3)
        if (
            not force
            and self._mesh_node is not None
            and self._last_pose is not None
            and pose.shape == self._last_pose.shape
            and np.allclose(pose, self._last_pose, atol=1.0e-5)
            and (
                (new_transl is None and self._last_transl is None)
                or (
                    new_transl is not None
                    and self._last_transl is not None
                    and np.allclose(new_transl, self._last_transl, atol=1.0e-5)
                )
            )
        ):
            return True
        self._last_pose = pose.copy()
        self._last_transl = None if new_transl is None else new_transl.copy()
        if not self.visible or self.opacity <= 0.0:
            self.clear_node()
            return True
        cache_hit = (
            not bool(os.environ.get("AMONGUS_ANATOMY_FORCE_LIVE_LBS", "").strip())
            and self.asset.pose_cache_vertices is not None
            and self.asset.pose_cache_hash == smplx_pose_hash(pose, new_transl)
        )
        vertices = (
            np.asarray(self.asset.pose_cache_vertices, dtype=np.float32)
            if cache_hit
            else skin_vertices(self.asset, pose, transl=transl)
        )
        if not np.all(np.isfinite(vertices)):
            return False
        vertices = _vertices_to_genesis_z_up(
            vertices,
            coordinate_system=self.asset.coordinate_system,
        )
        span_m = float(np.max(np.ptp(vertices, axis=0)))
        if span_m < 0.05 or span_m > 10.0:
            import logging

            logging.getLogger(__name__).warning(
                "anatomy draw skipped model_id=%s span_m=%.3f (expected 0.05..10)",
                self.model_id,
                span_m,
            )
            return False

        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=self._render_faces(), process=False)
        rgba = _vertex_colors_for_asset(
            self.asset,
            fallback_rgba=self.default_color_rgba,
            opacity=self.opacity,
        )
        mesh.visual.vertex_colors = rgba

        self._mesh_node = replace_colored_debug_mesh(
            self.runtime,
            mesh,
            self._mesh_node,
            double_sided=True,
            # Blender renders the vascular meshes with smooth shading (and a
            # post-Armature subdivision modifier).  Flat debug-mesh normals
            # made the unchanged low-poly tube topology look kinked in
            # Genesis even when its skinned centerline was continuous.
            smooth=True,
        )
        return self._mesh_node is not None
