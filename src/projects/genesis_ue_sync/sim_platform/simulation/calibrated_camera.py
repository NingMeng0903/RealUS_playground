"""Per-camera full-intrinsics support for the local Genesis rasterizer.

Genesis 0.4.4 exposes a scalar vertical FOV on its high-level camera object.
The bundled rasterizer, however, uses pyrender camera objects internally and
its ``IntrinsicsCamera`` already implements an off-centre, non-square-focal
length projection.  This module installs that object on one camera node after
Genesis has built the scene.  The adaptation is deliberately per-instance:
no Genesis or pyrender class is monkey-patched.

The adapter is intentionally strict about the private structures it has to
touch.  They are not a public Genesis API, so an unsupported Genesis version
or renderer fails with a useful error instead of silently falling back to an
incorrect scalar-FOV projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import inspect
from types import MethodType
from typing import Any, Mapping

import numpy as np


# The local RealUS setup uses this checked-in Genesis fork.  Keep this exact
# until a new adapter has been audited against a newer private rasterizer API.
SUPPORTED_GENESIS_VERSION = "0.4.4"
_PYRENDER_CAMERA_MODULE = "genesis.ext.pyrender.camera"
_RASTERIZER_MODULE = "genesis.vis.rasterizer"


def _as_intrinsics_matrix(raw: Any, *, source: str) -> np.ndarray:
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a numeric 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{source} must be a finite 3x3 matrix, got shape {matrix.shape}")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{source} must have homogeneous bottom row [0, 0, 1]")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError(f"{source} must have positive fx and fy")
    # pyrender.IntrinsicsCamera has no skew parameter.  Refusing skew is
    # preferable to rendering a subtly different camera than the calibrated K.
    if not np.isclose(matrix[0, 1], 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(
            "Genesis Rasterizer full-K adapter cannot represent non-zero K skew "
            f"(got {matrix[0, 1]!r})"
        )
    matrix.setflags(write=False)
    return matrix


def full_k_from_camera_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    resolution: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Read a calibrated K from a scene camera metadata mapping.

    The scene compiler currently writes ``metadata['intrinsics']`` as a
    matrix.  ``intrinsics.K`` and ``image_geometry.effective_K`` are accepted
    as well because those are the two frame metadata forms used by the UE
    ingress path.  Missing K means that the caller should retain Genesis'
    scalar-FOV camera.
    """

    metadata = metadata or {}
    raw: Any = None
    source = "camera metadata"

    intrinsics = metadata.get("intrinsics")
    if isinstance(intrinsics, Mapping):
        for key in ("K", "matrix", "intrinsics"):
            if key in intrinsics:
                raw = intrinsics[key]
                source = f"camera metadata intrinsics.{key}"
                break
        if raw is None and all(key in intrinsics for key in ("fx", "fy", "cx", "cy")):
            raw = [
                [intrinsics["fx"], intrinsics.get("skew", 0.0), intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ]
            source = "camera metadata intrinsics.fx/fy/cx/cy"
    elif intrinsics is not None:
        raw = intrinsics
        source = "camera metadata intrinsics"

    if raw is None:
        geometry = metadata.get("image_geometry")
        if isinstance(geometry, Mapping) and geometry.get("effective_K") is not None:
            raw = geometry["effective_K"]
            source = "camera metadata image_geometry.effective_K"

    if raw is None:
        return None

    matrix = _as_intrinsics_matrix(raw, source=source)
    if resolution is not None:
        width, height = (int(resolution[0]), int(resolution[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera resolution must be positive, got {resolution!r}")
    distortion = metadata.get("distortion")
    if distortion is not None:
        try:
            distortion_arr = np.asarray(distortion, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Camera distortion metadata must be numeric") from exc
        if not np.all(np.isfinite(distortion_arr)):
            raise ValueError("Camera distortion metadata must be finite")
        if np.any(np.abs(distortion_arr) > 1e-12):
            raise ValueError(
                "Genesis Rasterizer full-K adapter supports a pinhole K only; "
                "non-zero lens distortion is not representable"
            )
    return matrix


def camera_intrinsics_to_matrix(value: Any) -> np.ndarray | None:
    """Convert a ``CameraIntrinsics``-like object to a validated K matrix."""

    if value is None:
        return None
    if all(hasattr(value, key) for key in ("fx", "fy", "cx", "cy")):
        value = [
            [value.fx, getattr(value, "skew", 0.0), value.cx],
            [0.0, value.fy, value.cy],
            [0.0, 0.0, 1.0],
        ]
    return _as_intrinsics_matrix(value, source="camera intrinsics")


def opengl_projection_matrix_from_k(
    intrinsics: Any,
    *,
    width: int,
    height: int,
    znear: float,
    zfar: float | None,
) -> np.ndarray:
    """Return the OpenGL projection used by pyrender.IntrinsicsCamera."""

    K = _as_intrinsics_matrix(intrinsics, source="intrinsics")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Projection resolution must be positive, got {(width, height)!r}")
    znear = float(znear)
    if znear < 0.0:
        raise ValueError(f"z-near must be non-negative, got {znear!r}")
    if zfar is not None:
        zfar = float(zfar)
        if zfar <= znear:
            raise ValueError(f"z-far must be greater than z-near, got {(znear, zfar)!r}")

    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    width_f, height_f = float(width), float(height)
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * fx / width_f
    P[1, 1] = 2.0 * fy / height_f
    P[0, 2] = 1.0 - 2.0 * cx / width_f
    P[1, 2] = 2.0 * cy / height_f - 1.0
    P[3, 2] = -1.0
    if zfar is None:
        P[2, 2] = -1.0
        P[2, 3] = -2.0 * znear
    else:
        P[2, 2] = (zfar + znear) / (znear - zfar)
        P[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    P.setflags(write=False)
    return P


def _require_supported_private_api(camera: Any) -> tuple[Any, Any, Any]:
    """Validate the exact local Genesis 0.4.4 rasterizer instance layout."""

    visualizer = getattr(camera, "_visualizer", None)
    rasterizer = getattr(camera, "_rasterizer", None)
    if visualizer is None or rasterizer is None:
        raise RuntimeError(
            "Full-K Genesis camera installation requires a built camera with a rasterizer instance; "
            "call GenesisPlatformRuntime.build() first."
        )

    try:
        import genesis as gs
    except ImportError as exc:  # pragma: no cover - runtime-only failure
        raise RuntimeError("Full-K Genesis camera installation requires the local Genesis package") from exc
    version = str(getattr(gs, "__version__", ""))
    if version != SUPPORTED_GENESIS_VERSION:
        raise RuntimeError(
            "Full-K Genesis camera adapter only supports Genesis "
            f"{SUPPORTED_GENESIS_VERSION}; loaded {version or 'unknown'}"
        )

    if type(rasterizer).__module__ != _RASTERIZER_MODULE or type(rasterizer).__name__ != "Rasterizer":
        raise RuntimeError(
            "Full-K camera projection is only supported by the Genesis Rasterizer; "
            f"got {type(rasterizer).__module__}.{type(rasterizer).__name__}"
        )
    if getattr(camera, "_raytracer", None) is not None or getattr(camera, "_batch_renderer", None) is not None:
        raise RuntimeError(
            "Full-K camera projection is unsupported when Genesis uses RayTracer or BatchRenderer; "
            "select the Rasterizer renderer."
        )
    if getattr(visualizer, "_renderer", rasterizer) is not rasterizer:
        renderer = getattr(visualizer, "_renderer", None)
        raise RuntimeError(
            "Full-K camera projection requires the active Genesis renderer to be Rasterizer; "
            f"got {type(renderer).__module__}.{type(renderer).__name__}"
        )

    camera_nodes = getattr(rasterizer, "_camera_nodes", None)
    if not isinstance(camera_nodes, dict):
        raise RuntimeError(
            "Unsupported Genesis Rasterizer camera API: expected per-instance _camera_nodes dict"
        )
    uid = getattr(camera, "uid", None)
    node = camera_nodes.get(uid)
    if node is None:
        raise RuntimeError("Genesis Rasterizer has no node for the camera being calibrated")
    node_camera = getattr(node, "camera", None)
    if (
        type(node_camera).__module__ != _PYRENDER_CAMERA_MODULE
        or type(node_camera).__name__ != "PerspectiveCamera"
    ):
        raise RuntimeError(
            "Unsupported Genesis Rasterizer camera node API: expected pyrender.PerspectiveCamera, "
            f"got {type(node_camera).__module__}.{type(node_camera).__name__}"
        )
    update_camera = getattr(rasterizer, "update_camera", None)
    if not callable(update_camera) or len(inspect.signature(update_camera).parameters) != 1:
        raise RuntimeError("Unsupported Genesis Rasterizer update_camera API")
    return rasterizer, node, node_camera


def _locked_intrinsics_camera_type(intrinsics_camera_type: type) -> type:
    """Create a node-local camera type whose legacy yfov writes are harmless."""

    class LockedIntrinsicsCamera(intrinsics_camera_type):
        @property
        def yfov(self):
            # Rasterizer.update_camera writes yfov on every pose update.  The
            # actual projection comes from fx/fy/cx/cy and must stay locked.
            return getattr(self, "_realus_ignored_yfov", 0.0)

        @yfov.setter
        def yfov(self, value):
            self._realus_ignored_yfov = float(value)

    LockedIntrinsicsCamera.__name__ = "RealUSLockedIntrinsicsCamera"
    LockedIntrinsicsCamera.__qualname__ = "RealUSLockedIntrinsicsCamera"
    return LockedIntrinsicsCamera


def _replace_registered_camera(scene: Any, node: Any, replacement: Any) -> None:
    """Re-register a camera node so pyrender's object maps remain consistent."""
    if not scene.has_node(node) or node.children:
        raise RuntimeError("Full-K camera replacement requires a registered leaf camera node")
    parents = list(scene._digraph.neighbors(node))
    if len(parents) != 1:
        raise RuntimeError("Unsupported pyrender camera parent graph")
    parent = None if parents[0] == "world" else parents[0]
    main_camera = scene.main_camera_node
    old_camera = node.camera
    scene.remove_node(node)
    try:
        node.camera = replacement
        scene.add_node(node, parent_node=parent)
    except Exception:
        node.camera = old_camera
        if not scene.has_node(node):
            scene.add_node(node, parent_node=parent)
        raise
    finally:
        if main_camera is not None and scene.has_node(main_camera):
            scene.main_camera_node = main_camera


@dataclass
class FullKCameraAdapter:
    """Installed state for one Genesis camera and its rasterizer node."""

    camera: Any
    intrinsics: np.ndarray
    width: int
    height: int
    projection_matrix: np.ndarray
    node: Any
    node_camera: Any
    source: str = "scene_camera_metadata"

    @property
    def metadata(self) -> dict[str, Any]:
        K = self.intrinsics
        return {
            "projection_model": "opencv_pinhole",
            "intrinsics_source": self.source,
            "intrinsics": K.tolist(),
            "projection_matrix": self.projection_matrix.tolist(),
            "pointcloud_intrinsics": K.tolist(),
        }

    def render_pointcloud(self, camera: Any, world_frame: bool = True):
        """Genesis Camera.render_pointcloud with the calibrated fx/fy/cx/cy."""

        if not bool(getattr(camera, "_is_built", False)):
            raise RuntimeError("Genesis camera must be built before render_pointcloud()")
        if getattr(camera, "_batch_renderer", None) is not None:
            raise RuntimeError("Full-K point-cloud rendering is unsupported with BatchRenderer")

        rasterizer = getattr(camera, "_rasterizer", None)
        if rasterizer is None:
            raise RuntimeError("Genesis camera has no rasterizer for point-cloud rendering")
        rasterizer.update_scene(force_render=False)
        _, depth_arr, _, _ = rasterizer.render_camera(
            camera, rgb=False, depth=True, segmentation=False, normal=False
        )
        depth_arr = np.asarray(depth_arr)

        width, height = self.width, self.height
        fx, fy, cx, cy = (
            float(self.intrinsics[0, 0]),
            float(self.intrinsics[1, 1]),
            float(self.intrinsics[0, 2]),
            float(self.intrinsics[1, 2]),
        )
        mask = (float(camera.near) < depth_arr) & (depth_arr < float(camera.far) * (1.0 - 1e-3))
        v, u = np.meshgrid(np.arange(height, dtype=np.int32), np.arange(width, dtype=np.int32), indexing="ij")
        u = u.reshape((-1,))
        v = v.reshape((-1,))
        depth_grid = depth_arr[..., v, u]
        world_x = depth_grid * (u + 0.5 - cx) / fx
        world_y = depth_grid * (v + 0.5 - cy) / fy
        world_z = depth_grid
        point_cloud = np.stack((world_x, world_y, world_z, np.ones_like(world_z)), axis=-1)
        if world_frame:
            T_OPENGL_TO_OPENCV = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            cam_pose = np.asarray(camera.transform) @ T_OPENGL_TO_OPENCV
            point_cloud = point_cloud @ np.swapaxes(cam_pose, -1, -2)
        point_cloud = point_cloud[..., :3].reshape((*depth_arr.shape, 3))
        return point_cloud, mask


def install_full_k_raster_camera(
    camera: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    resolution: tuple[int, int] | None = None,
    intrinsics: Any | None = None,
    source: str = "scene_camera_metadata",
) -> FullKCameraAdapter | None:
    """Install full K on one already-built Genesis Rasterizer camera.

    ``None`` is returned when no calibrated K is present.  Supplying an
    explicit ``intrinsics`` (for example from a sensor profile) takes
    precedence over metadata.  If both are present they must agree exactly
    within numerical tolerance; otherwise the render and sensor contracts
    would describe different cameras and installation fails.
    """

    metadata = metadata or {}
    if resolution is None:
        camera_resolution = getattr(camera, "res", None)
        if camera_resolution is None:
            raise ValueError("Camera resolution is required for full-K installation")
        resolution = (int(camera_resolution[0]), int(camera_resolution[1]))
    width, height = (int(resolution[0]), int(resolution[1]))

    metadata_K = full_k_from_camera_metadata(metadata, resolution=(width, height))
    explicit_K = None if intrinsics is None else _as_intrinsics_matrix(intrinsics, source="explicit intrinsics")
    if explicit_K is None:
        K = metadata_K
    elif metadata_K is None:
        K = explicit_K
    else:
        if not np.allclose(explicit_K, metadata_K, atol=1e-9, rtol=1e-9):
            raise ValueError("Explicit sensor intrinsics do not match scene camera metadata K")
        K = explicit_K
    if K is None:
        return None

    rasterizer, node, _old_node_camera = _require_supported_private_api(camera)
    try:
        from genesis.ext import pyrender
    except ImportError as exc:  # pragma: no cover - runtime-only failure
        raise RuntimeError("The local Genesis pyrender extension is required for full-K cameras") from exc
    intrinsics_type = getattr(pyrender, "IntrinsicsCamera", None)
    if not isinstance(intrinsics_type, type):
        raise RuntimeError("The local Genesis pyrender extension has no IntrinsicsCamera")
    init_params = inspect.signature(intrinsics_type.__init__).parameters
    required_params = ("fx", "fy", "cx", "cy", "znear", "zfar")
    if any(name not in init_params for name in required_params):
        raise RuntimeError("Unsupported pyrender.IntrinsicsCamera constructor; expected fx/fy/cx/cy/znear/zfar")
    projection_method = getattr(intrinsics_type, "get_projection_matrix", None)
    if not callable(projection_method):
        raise RuntimeError("Unsupported pyrender.IntrinsicsCamera: get_projection_matrix is missing")

    locked_type = _locked_intrinsics_camera_type(intrinsics_type)
    node_camera = locked_type(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=float(camera.near),
        zfar=float(camera.far),
    )
    context = getattr(rasterizer, "_context", None)
    render_scene = getattr(context, "_scene", None)
    if render_scene is None:
        raise RuntimeError("Unsupported Genesis rasterizer context: registered pyrender scene is missing")
    # Mutating node.camera in place leaves Scene._obj_to_nodes pointing at the
    # discarded camera and breaks destruction. Use its registration API.
    lock = getattr(camera._visualizer, "viewer_lock", None)
    with lock if lock is not None else nullcontext():
        _replace_registered_camera(render_scene, node, node_camera)
    if node.camera is not node_camera:
        raise RuntimeError("Genesis Rasterizer rejected the per-camera IntrinsicsCamera replacement")

    projection = opengl_projection_matrix_from_k(
        K,
        width=width,
        height=height,
        znear=float(camera.near),
        zfar=float(camera.far),
    )
    # Camera.intrinsics and Camera.projection_matrix are cached_property
    # descriptors in Genesis 0.4.4.  Populate their instance caches so all
    # downstream callers see the same K as the rasterizer node.
    camera.__dict__["intrinsics"] = K
    camera.__dict__["projection_matrix"] = projection

    adapter = FullKCameraAdapter(
        camera=camera,
        intrinsics=K,
        width=width,
        height=height,
        projection_matrix=projection,
        node=node,
        node_camera=node_camera,
        source=source,
    )
    # Point-cloud conversion in Genesis' stock Camera assumes fx == fy and a
    # centred principal point.  Bind the calibrated implementation only to
    # this camera instance; other cameras and Genesis classes are untouched.
    camera.__dict__["render_pointcloud"] = MethodType(adapter.render_pointcloud, camera)
    camera.__dict__["_realus_full_k_adapter"] = adapter
    return adapter


__all__ = [
    "SUPPORTED_GENESIS_VERSION",
    "FullKCameraAdapter",
    "full_k_from_camera_metadata",
    "camera_intrinsics_to_matrix",
    "install_full_k_raster_camera",
    "opengl_projection_matrix_from_k",
]
