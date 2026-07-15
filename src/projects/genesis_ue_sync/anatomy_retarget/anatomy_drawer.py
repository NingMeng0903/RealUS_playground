"""Genesis debug-mesh drawer for retargeted anatomy assets."""

from __future__ import annotations

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
        faces = np.asarray(self.asset.faces, dtype=np.int32)
        if bool((self.asset.metadata or {}).get("show_connective_tissue", False)):
            return faces
        if self.asset.source_vertex_ranges is None or self.asset.source_tissues is None:
            return faces
        hidden = np.zeros(len(self.asset.vertices_rest), dtype=bool)
        for (start, stop), tissue in zip(self.asset.source_vertex_ranges, self.asset.source_tissues):
            if str(tissue) == "connective_tissue":
                hidden[int(start) : int(stop)] = True
        return faces[~np.any(hidden[faces], axis=1)]

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
        with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
            if not acquired:
                return
            try:
                self.runtime.scene.clear_debug_object(self._mesh_node)
            except Exception:
                pass
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
        )
        return self._mesh_node is not None
