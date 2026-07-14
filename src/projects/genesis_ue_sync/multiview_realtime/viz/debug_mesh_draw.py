"""Shared Genesis debug-mesh draw helpers (pyrender markers)."""

from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock


def _visualizer_context(runtime: Any) -> Any:
    return runtime.scene._visualizer.context


def draw_colored_debug_mesh(
    runtime: Any,
    mesh: Any,
    *,
    double_sided: bool = False,
    smooth: bool = False,
) -> Any | None:
    """Draw a trimesh marker; returns the pyrender node (or None if lock busy)."""
    import genesis as gs
    import genesis.utils.geom as gu
    from genesis.ext import pyrender
    from genesis.utils.misc import tensor_to_array

    ctx = _visualizer_context(runtime)
    n_envs = len(ctx.rendered_envs_idx)
    T = gu.trans_to_T(np.zeros(3, dtype=np.float32))
    poses = tensor_to_array(T)
    if poses.ndim != 3:
        poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))

    node = pyrender.Mesh.from_trimesh(
        mesh,
        name=f"debug_mesh_{gs.UID()}",
        poses=poses,
        is_marker=True,
        double_sided=bool(double_sided),
        smooth=bool(smooth),
    )
    ctx.add_external_node(node)
    return node


def replace_colored_debug_mesh(
    runtime: Any,
    mesh: Any,
    old_node: Any | None,
    *,
    double_sided: bool = False,
    smooth: bool = False,
    lock_timeout_s: float = 0.15,
) -> Any | None:
    with try_viewer_render_lock(runtime, timeout_s=lock_timeout_s) as acquired:
        if not acquired:
            return old_node
        ctx = _visualizer_context(runtime)
        if old_node is not None:
            try:
                ctx.clear_debug_object(old_node)
            except Exception:
                try:
                    runtime.scene.clear_debug_object(old_node)
                except Exception:
                    pass
        return draw_colored_debug_mesh(runtime, mesh, double_sided=double_sided, smooth=smooth)
