from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bridge.adapters.opencv import opencv_camera_matrices_from_scene_camera
from bridge.core.camera import build_intrinsics_from_fov as bridge_build_intrinsics_from_fov
from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.core.messages import CameraIntrinsics
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import SensorProfile
from projects.genesis_ue_sync.sim_platform.scenes import SceneCameraSpec, SyncSceneSpec, load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.simulation.runtime import StaticCameraConfig

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _expand_path(raw: str | Path | None, *, anchor: Path) -> Path | None:
    if raw is None:
        return None
    text = os.path.expandvars(str(raw)).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (anchor / path).resolve()


def _normalize(v: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        if fallback is None:
            raise ValueError("Cannot normalize a near-zero vector without a fallback.")
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return arr / norm


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis.tolist()
    c = float(math.cos(float(angle_rad)))
    s = float(math.sin(float(angle_rad)))
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def build_intrinsics_from_fov(*, width: int, height: int, fov_deg: float) -> np.ndarray:
    return bridge_build_intrinsics_from_fov(width=width, height=height, fov_deg=fov_deg)


def scale_intrinsics(K: np.ndarray, *, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    k = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    fw, fh = int(from_wh[0]), int(from_wh[1])
    tw, th = int(to_wh[0]), int(to_wh[1])
    if fw <= 0 or fh <= 0 or (fw, fh) == (tw, th):
        return k
    sx = float(tw) / float(fw)
    sy = float(th) / float(fh)
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def build_camera_from_scene_camera(camera: SceneCameraSpec) -> tuple[np.ndarray, np.ndarray]:
    return opencv_camera_matrices_from_scene_camera(camera)


@dataclass(frozen=True)
class AlignmentConvention:
    world_frame: str = "world"
    ue_frame: str = "ue_world"
    genesis_frame: str = "genesis_world"
    world_up_axis: str = "z"
    world_handedness: str = "right"
    ue_handedness: str = "left"
    genesis_handedness: str = "right"
    units: str = "meters"
    image_origin: str = "top_left"
    camera_forward_axis: str = "+z"
    notes: list[str] = field(default_factory=list)
    world_from_ue: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    world_from_genesis: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_frame": self.world_frame,
            "ue_frame": self.ue_frame,
            "genesis_frame": self.genesis_frame,
            "world_up_axis": self.world_up_axis,
            "world_handedness": self.world_handedness,
            "ue_handedness": self.ue_handedness,
            "genesis_handedness": self.genesis_handedness,
            "units": self.units,
            "image_origin": self.image_origin,
            "camera_forward_axis": self.camera_forward_axis,
            "notes": list(self.notes),
            "world_from_ue": [list(row) for row in self.world_from_ue],
            "world_from_genesis": [list(row) for row in self.world_from_genesis],
        }


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    image_size: tuple[int, int]
    intrinsics: np.ndarray
    camera_from_world: np.ndarray
    world_from_camera: np.ndarray
    distortion: np.ndarray = field(default_factory=lambda: np.zeros((5,), dtype=np.float64))
    source: str = "scene_spec"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", (int(self.image_size[0]), int(self.image_size[1])))
        object.__setattr__(self, "intrinsics", np.asarray(self.intrinsics, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "camera_from_world", np.asarray(self.camera_from_world, dtype=np.float64).reshape(4, 4))
        object.__setattr__(self, "world_from_camera", np.asarray(self.world_from_camera, dtype=np.float64).reshape(4, 4))
        object.__setattr__(self, "distortion", np.asarray(self.distortion, dtype=np.float64).reshape(-1))

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    @property
    def projection(self) -> np.ndarray:
        return self.intrinsics @ self.camera_from_world[:3, :]

    @property
    def camera_center_world(self) -> np.ndarray:
        return self.world_from_camera[:3, 3].copy()

    def to_camera_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=float(self.intrinsics[0, 0]),
            fy=float(self.intrinsics[1, 1]),
            cx=float(self.intrinsics[0, 2]),
            cy=float(self.intrinsics[1, 2]),
            skew=float(self.intrinsics[0, 1]),
        )

    def to_sensor_profile(self) -> SensorProfile:
        return SensorProfile(
            name=self.camera_id,
            modality="rgb",
            frame_id=f"camera_frame/{self.camera_id}",
            resolution=(self.width, self.height),
            intrinsics=self.to_camera_intrinsics(),
            extrinsics=self.camera_from_world.tolist(),
            metadata={
                "camera_id": self.camera_id,
                "distortion": self.distortion.tolist(),
                **dict(self.metadata),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "image_size": [self.width, self.height],
            "intrinsics": self.intrinsics.tolist(),
            "camera_from_world": self.camera_from_world.tolist(),
            "world_from_camera": self.world_from_camera.tolist(),
            "projection": self.projection.tolist(),
            "distortion": self.distortion.tolist(),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationBundle:
    scene_spec_path: Path | None
    scene_spec: SyncSceneSpec | None
    cameras: dict[str, CameraCalibration]
    convention: AlignmentConvention
    calibration_path: Path | None = None
    alignment_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def camera(self, camera_id: str) -> CameraCalibration:
        if camera_id not in self.cameras:
            raise KeyError(f"Unknown calibrated camera: {camera_id}")
        return self.cameras[camera_id]

    def ordered_camera_ids(self) -> list[str]:
        return list(self.cameras.keys())

    def static_camera_configs(self) -> list[StaticCameraConfig]:
        if self.scene_spec is not None:
            out: list[StaticCameraConfig] = []
            for scene_camera in self.scene_spec.cameras:
                if scene_camera.name in self.cameras:
                    out.append(
                        StaticCameraConfig(
                            name=scene_camera.name,
                            res=scene_camera.res,
                            pos=scene_camera.pos,
                            lookat=scene_camera.lookat,
                            up=scene_camera.up,
                            fov=scene_camera.fov,
                            near=scene_camera.near,
                            far=scene_camera.far,
                            gui=scene_camera.gui,
                            mount_entity=scene_camera.mount_entity,
                            mount_link=scene_camera.mount_link,
                            pose_rel=scene_camera.pose_rel,
                            follow_entity=scene_camera.follow_entity,
                            metadata=dict(scene_camera.metadata),
                        )
                    )
            return out
        out = []
        for camera_id, camera in self.cameras.items():
            center = camera.camera_center_world
            forward = camera.world_from_camera[:3, 2]
            up = -camera.world_from_camera[:3, 1]
            lookat = center + forward
            out.append(
                StaticCameraConfig(
                    name=camera_id,
                    res=camera.image_size,
                    pos=tuple(float(v) for v in center.tolist()),
                    lookat=tuple(float(v) for v in lookat.tolist()),
                    up=tuple(float(v) for v in up.tolist()),
                    fov=45.0,
                )
            )
        return out

    def sensor_profiles(self) -> dict[str, SensorProfile]:
        return {camera_id: camera.to_sensor_profile() for camera_id, camera in self.cameras.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_spec_path": str(self.scene_spec_path) if self.scene_spec_path is not None else None,
            "calibration_path": str(self.calibration_path) if self.calibration_path is not None else None,
            "alignment_path": str(self.alignment_path) if self.alignment_path is not None else None,
            "convention": self.convention.as_dict(),
            "cameras": {camera_id: camera.as_dict() for camera_id, camera in self.cameras.items()},
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


def _load_payload(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if yaml is not None and path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload in {path}, got {type(payload).__name__}")
    return payload


def _alignment_from_payload(payload: dict[str, Any]) -> AlignmentConvention:
    return AlignmentConvention(
        world_frame=str(payload.get("world_frame", "world")),
        ue_frame=str(payload.get("ue_frame", "ue_world")),
        genesis_frame=str(payload.get("genesis_frame", "genesis_world")),
        world_up_axis=str(payload.get("world_up_axis", "z")),
        world_handedness=str(payload.get("world_handedness", "right")),
        ue_handedness=str(payload.get("ue_handedness", "left")),
        genesis_handedness=str(payload.get("genesis_handedness", "right")),
        units=str(payload.get("units", "meters")),
        image_origin=str(payload.get("image_origin", "top_left")),
        camera_forward_axis=str(payload.get("camera_forward_axis", "+z")),
        notes=[str(item) for item in payload.get("notes", [])],
        world_from_ue=tuple(tuple(float(v) for v in row) for row in payload.get("world_from_ue", np.eye(4).tolist())),
        world_from_genesis=tuple(
            tuple(float(v) for v in row) for row in payload.get("world_from_genesis", np.eye(4).tolist())
        ),
    )


def _camera_from_payload(camera_id: str, payload: dict[str, Any]) -> CameraCalibration:
    image_size = tuple(int(v) for v in payload["image_size"])
    return CameraCalibration(
        camera_id=camera_id,
        image_size=(int(image_size[0]), int(image_size[1])),
        intrinsics=np.asarray(payload["intrinsics"], dtype=np.float64),
        camera_from_world=np.asarray(payload["camera_from_world"], dtype=np.float64),
        world_from_camera=np.asarray(payload["world_from_camera"], dtype=np.float64),
        distortion=np.asarray(payload.get("distortion", [0.0] * 5), dtype=np.float64),
        source=str(payload.get("source", "calibration_file")),
        metadata=dict(payload.get("metadata", {})),
    )


def calibration_from_scene_camera(
    camera: SceneCameraSpec,
    *,
    intrinsics: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = "scene_spec",
) -> CameraCalibration:
    camera_from_world, world_from_camera = build_camera_from_scene_camera(camera)
    K = intrinsics
    if K is None:
        K = build_intrinsics_from_fov(width=int(camera.res[0]), height=int(camera.res[1]), fov_deg=float(camera.fov))
    return CameraCalibration(
        camera_id=camera.name,
        image_size=(int(camera.res[0]), int(camera.res[1])),
        intrinsics=K,
        camera_from_world=camera_from_world,
        world_from_camera=world_from_camera,
        distortion=np.zeros((5,), dtype=np.float64) if distortion is None else distortion,
        source=source,
        metadata=dict(metadata or {}),
    )


def _bundle_from_scene(scene_spec: SyncSceneSpec, *, scene_spec_path: Path | None, metadata: dict[str, Any]) -> CalibrationBundle:
    cameras = {
        scene_camera.name: calibration_from_scene_camera(
            scene_camera,
            metadata={"fov_deg": float(scene_camera.fov), "derived_from_scene_spec": True, **dict(scene_camera.metadata)},
            source="scene_spec",
        )
        for scene_camera in scene_spec.cameras
    }
    return CalibrationBundle(
        scene_spec_path=scene_spec_path,
        scene_spec=scene_spec,
        cameras=cameras,
        convention=AlignmentConvention(),
        metadata=dict(metadata),
    )


def load_calibration_bundle(
    calibration_path: str | Path,
    *,
    scene_spec_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
) -> CalibrationBundle:
    calibration_path = project_paths(__file__).resolve_from_root(calibration_path)
    payload = _load_payload(calibration_path)
    anchor = calibration_path.parent
    scene_path = _expand_path(scene_spec_path or payload.get("scene_spec"), anchor=anchor)
    alignment_file = _expand_path(alignment_path or payload.get("alignment_path"), anchor=anchor)
    scene_spec = load_sync_scene_spec(scene_path) if scene_path is not None and scene_path.is_file() else None
    convention = AlignmentConvention()
    if alignment_file is not None and alignment_file.is_file():
        convention = _alignment_from_payload(_load_payload(alignment_file))
    elif isinstance(payload.get("convention"), dict):
        convention = _alignment_from_payload(dict(payload["convention"]))
    cameras_payload = payload.get("cameras")
    cameras: dict[str, CameraCalibration]
    if isinstance(cameras_payload, dict) and cameras_payload:
        cameras = {str(camera_id): _camera_from_payload(str(camera_id), dict(camera_payload)) for camera_id, camera_payload in cameras_payload.items()}
    elif scene_spec is not None:
        cameras = _bundle_from_scene(scene_spec, scene_spec_path=scene_path, metadata=dict(payload.get("metadata", {}))).cameras
    else:
        raise ValueError(f"Calibration file {calibration_path} defines no cameras and no usable scene spec.")
    return CalibrationBundle(
        scene_spec_path=scene_path,
        scene_spec=scene_spec,
        cameras=cameras,
        convention=convention,
        calibration_path=calibration_path,
        alignment_path=alignment_file,
        metadata=dict(payload.get("metadata", {})),
    )


def bundle_from_scene_spec(scene_spec_path: str | Path) -> CalibrationBundle:
    scene_path = project_paths(__file__).resolve_from_root(scene_spec_path)
    scene_spec = load_sync_scene_spec(scene_path)
    bundle = _bundle_from_scene(scene_spec, scene_spec_path=scene_path, metadata={"generated": True})
    return bundle


__all__ = [
    "AlignmentConvention",
    "CalibrationBundle",
    "CameraCalibration",
    "build_camera_from_scene_camera",
    "build_intrinsics_from_fov",
    "bundle_from_scene_spec",
    "calibration_from_scene_camera",
    "load_calibration_bundle",
]
