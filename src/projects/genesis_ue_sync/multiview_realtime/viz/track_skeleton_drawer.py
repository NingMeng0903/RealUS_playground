"""Draw a Body25 3D skeleton (joints + bones) on a live Genesis runtime.

Replaces the offline SMPL mesh overlay: consumes filtered triangulated 3D
keypoints from the realtime triangulation backend and renders them as colored
spheres connected by bone segments. Left/right limbs use distinct shades and a
confidence gate suppresses noisy low-score joints (e.g. occluded head points).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime

# OpenPose BODY_25 bone connectivity, grouped by body side for readable coloring.
CENTER_EDGES: tuple[tuple[int, int], ...] = ((1, 8), (1, 0))
RIGHT_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (8, 9), (9, 10), (10, 11),
    (0, 15), (15, 17), (11, 22), (22, 23), (11, 24),
)
LEFT_EDGES: tuple[tuple[int, int], ...] = (
    (1, 5), (5, 6), (6, 7), (8, 12), (12, 13), (13, 14),
    (0, 16), (16, 18), (14, 19), (19, 20), (14, 21),
)

RIGHT_JOINTS = frozenset({2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24})
LEFT_JOINTS = frozenset({5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21})


def _uint8_rgba_to_float(rgba: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    return (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0)


class TrackSkeletonDrawer:
    """Colored 3D joint/bone skeleton overlay on an existing Genesis runtime."""

    def __init__(
        self,
        runtime: GenesisPlatformRuntime,
        *,
        joint_rgba: tuple[int, int, int, int] = (250, 122, 31, 235),
        joint_radius_m: float = 0.028,
        bone_radius_m: float = 0.013,
        min_draw_conf: float = 0.35,
        display_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.runtime = runtime
        # Center keeps the requested orange; left/right are warm-cool variants for readability.
        self._center_color = _uint8_rgba_to_float(joint_rgba)
        self._right_color = _uint8_rgba_to_float((250, 90, 40, joint_rgba[3]))
        self._left_color = _uint8_rgba_to_float((255, 200, 70, joint_rgba[3]))
        self._joint_radius = float(joint_radius_m)
        self._bone_radius = float(bone_radius_m)
        self._min_draw_conf = float(min_draw_conf)
        self._offset = np.asarray(display_offset_m, dtype=np.float64).reshape(3)
        self._nodes: list[Any] = []

    def _clear(self) -> None:
        for node in self._nodes:
            try:
                self.runtime.scene._visualizer.context.clear_debug_object(node)
            except Exception:
                pass
        self._nodes.clear()

    def _draw_joint_group(self, pos: np.ndarray, mask: np.ndarray, color, ctx) -> None:
        if not np.any(mask):
            return
        self._nodes.append(ctx.draw_debug_spheres(pos[mask], radius=self._joint_radius, color=color))

    def _draw_edges(self, pos: np.ndarray, drawable: np.ndarray, edges, color, n: int, ctx) -> None:
        for a, b in edges:
            if a >= n or b >= n or not (drawable[a] and drawable[b]):
                continue
            self._nodes.append(
                ctx.draw_debug_line(pos[a].tolist(), pos[b].tolist(), radius=self._bone_radius, color=color)
            )

    def draw(self, keypoints3d: np.ndarray, schema: str = "body25") -> None:
        kp = np.asarray(keypoints3d, dtype=np.float64).reshape(-1, 4)
        n = kp.shape[0]
        drawable = kp[:, 3] >= self._min_draw_conf
        pos = kp[:, :3] + self._offset
        if not np.any(drawable):
            with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
                if acquired:
                    self._clear()
            return

        idx = np.arange(n)
        right_mask = drawable & np.isin(idx, list(RIGHT_JOINTS))
        left_mask = drawable & np.isin(idx, list(LEFT_JOINTS))
        center_mask = drawable & ~right_mask & ~left_mask
        with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
            if not acquired:
                return
            ctx = self.runtime.scene._visualizer.context
            self._clear()
            self._draw_joint_group(pos, center_mask, self._center_color, ctx)
            self._draw_joint_group(pos, right_mask, self._right_color, ctx)
            self._draw_joint_group(pos, left_mask, self._left_color, ctx)
            if str(schema) == "body25":
                self._draw_edges(pos, drawable, CENTER_EDGES, self._center_color, n, ctx)
                self._draw_edges(pos, drawable, RIGHT_EDGES, self._right_color, n, ctx)
                self._draw_edges(pos, drawable, LEFT_EDGES, self._left_color, n, ctx)
