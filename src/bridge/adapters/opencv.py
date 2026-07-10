from __future__ import annotations

import numpy as np

from bridge.core.camera import CanonicalCamera, opencv_camera_matrices_from_lookat


def canonical_camera_from_scene_camera(camera_spec) -> CanonicalCamera:
    return CanonicalCamera(
        pos=tuple(float(v) for v in camera_spec.pos),
        lookat=tuple(float(v) for v in camera_spec.lookat),
        up=tuple(float(v) for v in camera_spec.up),
        roll_deg=float(getattr(camera_spec, 'roll_deg', 0.0) or 0.0),
    )


def opencv_camera_matrices_from_scene_camera(camera_spec) -> tuple[np.ndarray, np.ndarray]:
    camera = canonical_camera_from_scene_camera(camera_spec)
    return opencv_camera_matrices_from_lookat(camera.pos, camera.lookat, camera.up, roll_deg=camera.roll_deg)
