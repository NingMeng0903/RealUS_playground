"""Orange capsule human from triangulated Body25 3D joints (no SMPL skin mesh).

Visualizes the same filtered 3D keypoints used by the realtime SMPL fit, as a readable
humanoid proxy when the SMPL mesh payload is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock
from projects.genesis_ue_sync.multiview_realtime.viz.track_skeleton_drawer import (
    CENTER_EDGES,
    LEFT_EDGES,
    RIGHT_EDGES,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime

_ALL_EDGES = CENTER_EDGES + RIGHT_EDGES + LEFT_EDGES


def _align_z_to_vector(direction: np.ndarray) -> np.ndarray:
    import trimesh.transformations as tf

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return np.eye(4, dtype=np.float64)
    d /= n
    axis = np.cross(z, d)
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-8:
        if float(np.dot(z, d)) > 0.0:
            return np.eye(4, dtype=np.float64)
        return tf.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    axis /= axis_n
    angle = float(np.arccos(np.clip(np.dot(z, d), -1.0, 1.0)))
    return tf.rotation_matrix(angle, axis)


class TrackCapsuleDrawer:
    """Orange capsule humanoid overlay from Body25 (J,4) world joints."""

    def __init__(
        self,
        runtime: GenesisPlatformRuntime,
        *,
        mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 235),
        bone_radius_m: float = 0.032,
        joint_radius_m: float = 0.038,
        min_draw_conf: float = 0.35,
        display_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.runtime = runtime
        self._rgba = tuple(int(c) for c in mesh_rgba)
        self._bone_radius = float(bone_radius_m)
        self._joint_radius = float(joint_radius_m)
        self._min_draw_conf = float(min_draw_conf)
        self._offset = np.asarray(display_offset_m, dtype=np.float64).reshape(3)
        self._mesh_node: Any = None
        self._last_sig: tuple[float, ...] | None = None

    def _build_mesh(self, keypoints3d: np.ndarray):
        import trimesh

        kp = np.asarray(keypoints3d, dtype=np.float64).reshape(-1, 4)
        pos = kp[:, :3] + self._offset
        drawable = kp[:, 3] >= self._min_draw_conf
        parts: list[Any] = []
        rgba = np.asarray(self._rgba, dtype=np.uint8)

        for a, b in _ALL_EDGES:
            if a >= kp.shape[0] or b >= kp.shape[0] or not (drawable[a] and drawable[b]):
                continue
            p0 = pos[a]
            p1 = pos[b]
            seg = p1 - p0
            height = float(np.linalg.norm(seg))
            if height < 1e-4:
                continue
            cyl = trimesh.creation.cylinder(radius=self._bone_radius, height=height, sections=10)
            mid = 0.5 * (p0 + p1)
            T = _align_z_to_vector(seg)
            T[:3, 3] = mid
            cyl.apply_transform(T)
            cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
            parts.append(cyl)

        for i in range(kp.shape[0]):
            if not drawable[i]:
                continue
            sph = trimesh.creation.icosphere(radius=self._joint_radius, subdivisions=1)
            sph.apply_translation(pos[i])
            sph.visual.vertex_colors = np.tile(rgba, (len(sph.vertices), 1))
            parts.append(sph)

        if not parts:
            return None
        return trimesh.util.concatenate(parts)

    def draw(self, keypoints3d: np.ndarray, schema: str = "body25") -> None:
        if str(schema) != "body25":
            return
        kp = np.asarray(keypoints3d, dtype=np.float32).reshape(-1, 4)
        sig = tuple(float(v) for v in kp[np.isfinite(kp)].reshape(-1)[: min(60, kp.size)])
        if self._last_sig is not None and sig == self._last_sig and self._mesh_node is not None:
            return

        mesh = self._build_mesh(kp)
        with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
            if not acquired:
                return
            ctx = self.runtime.scene._visualizer.context
            if self._mesh_node is not None:
                try:
                    ctx.clear_debug_object(self._mesh_node)
                except Exception:
                    pass
                self._mesh_node = None
            if mesh is not None:
                self._mesh_node = ctx.draw_debug_mesh(mesh)
        self._last_sig = sig

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
