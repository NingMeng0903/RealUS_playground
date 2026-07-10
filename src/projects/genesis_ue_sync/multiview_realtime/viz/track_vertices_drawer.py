"""Draw fitted mesh vertices sent over the track ZMQ topic."""

from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime


class TrackVerticesDrawer:
    """Orange mesh overlay from explicit vertices/faces (SMPL-X, SMPL, or any fitted mesh)."""

    def __init__(
        self,
        runtime: GenesisPlatformRuntime,
        *,
        mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 235),
        display_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.runtime = runtime
        self._rgba = tuple(int(c) for c in mesh_rgba)
        self._offset = np.asarray(display_offset_m, dtype=np.float32).reshape(3)
        self._mesh_node: Any = None
        self._last_sig: tuple[float, ...] | None = None

    def draw(self, vertices: np.ndarray, faces: np.ndarray) -> bool:
        verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3) + self._offset.reshape(1, 3)
        tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if verts.size == 0 or tri.size == 0:
            return False
        if not np.all(np.isfinite(verts)):
            return False
        span = np.ptp(verts, axis=0)
        if float(np.max(span)) < 0.25 or float(np.max(span)) > 5.0:
            return False
        sig_arr = np.concatenate([verts.reshape(-1)[:60], tri.reshape(-1)[:30].astype(np.float32)])
        sig = tuple(float(v) for v in sig_arr.tolist())
        if self._last_sig is not None and sig == self._last_sig and self._mesh_node is not None:
            return True

        import trimesh

        mesh = trimesh.Trimesh(vertices=verts, faces=tri, process=False)
        mesh.visual.vertex_colors = np.tile(np.asarray(self._rgba, dtype=np.uint8), (len(mesh.vertices), 1))

        with try_viewer_render_lock(self.runtime, timeout_s=0.15) as acquired:
            if not acquired:
                return False
            ctx = self.runtime.scene._visualizer.context
            if self._mesh_node is not None:
                try:
                    ctx.clear_debug_object(self._mesh_node)
                except Exception:
                    pass
                self._mesh_node = None
            self._mesh_node = ctx.draw_debug_mesh(mesh)
        self._last_sig = sig
        return True

    def clear(self) -> None:
        if self._mesh_node is None:
            return
        with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
            if not acquired:
                return
            ctx = self.runtime.scene._visualizer.context
            try:
                ctx.clear_debug_object(self._mesh_node)
            except Exception:
                pass
        self._mesh_node = None
        self._last_sig = None
