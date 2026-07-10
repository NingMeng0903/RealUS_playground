"""Non-blocking access to the Genesis pyrender viewer lock.

``scene.clear_debug_object`` / ``draw_debug_mesh`` acquire ``render_lock`` with no
timeout. When the viewer thread is rendering, the main thread can block indefinitely
and Ctrl+C may not be delivered until the lock is released. Use a timed acquire and
batch debug updates under a single lock hold.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator


def _pyrender_render_lock(runtime: Any):
    vis = getattr(runtime.scene, "_visualizer", None)
    if vis is None:
        return None
    viewer_lock = getattr(vis, "viewer_lock", None)
    pyrender_viewer = getattr(viewer_lock, "_pyrender_viewer", None) if viewer_lock is not None else None
    if pyrender_viewer is None:
        return None
    return getattr(pyrender_viewer, "render_lock", None)


@contextmanager
def try_viewer_render_lock(runtime: Any, *, timeout_s: float = 0.05) -> Generator[bool, None, None]:
    lock = _pyrender_render_lock(runtime)
    if lock is None:
        yield True
        return
    acquired = bool(lock.acquire(timeout=max(0.0, float(timeout_s))))
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()
