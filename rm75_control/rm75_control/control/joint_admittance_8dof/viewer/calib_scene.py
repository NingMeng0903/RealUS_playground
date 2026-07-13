"""Load multicam calibration bundle for Genesis viewer scene decoration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.paths import PACKAGE_DIR

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(frozen=True)
class CameraCalib:
    camera_id: str
    image_size: tuple[int, int]
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    camera_from_world: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    @property
    def camera_center_world(self) -> np.ndarray:
        return self.world_from_camera[:3, 3].copy()

    def vertical_fov_deg(self) -> float:
        fy = float(self.intrinsics[1, 1])
        h = float(self.height)
        return math.degrees(2.0 * math.atan(h / (2.0 * max(fy, 1e-6))))

    def genesis_mount(self) -> dict[str, Any]:
        """pos / lookat / up for ``scene.add_camera`` (OpenCV +Z forward, Y down)."""
        center = self.camera_center_world
        forward = self.world_from_camera[:3, 2]
        up = -self.world_from_camera[:3, 1]
        lookat = center + forward
        return {
            "pos": tuple(float(v) for v in center.tolist()),
            "lookat": tuple(float(v) for v in lookat.tolist()),
            "up": tuple(float(v) for v in up.tolist()),
            "fov": self.vertical_fov_deg(),
            "res": (self.width, self.height),
        }


@dataclass(frozen=True)
class BedSurfaceCalib:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation_deg: float
    color: tuple[float, float, float, float] = (0.55, 0.78, 0.95, 1.0)

    @classmethod
    def from_bundle(cls, payload: dict[str, Any]) -> BedSurfaceCalib | None:
        """Build a floor-anchored bed box from ``genesis_bundle.yaml`` (no fixed thickness).

        Vertical extent is always ``[z=0, z=bed_top_z_m]`` where ``bed_top_z_m`` comes from
        calibration (``bed.height_m`` or ``bed.support_surface.top_z_m``). Re-run Stage 2 /
        re-export bundle after bed height changes — viewer picks it up automatically.
        """
        bed = payload.get("bed")
        if not isinstance(bed, dict):
            return None

        size_m = bed.get("size_m")
        if not size_m or len(size_m) < 2:
            return None
        lx, ly = float(size_m[0]), float(size_m[1])
        rot = float(bed.get("rotation_deg", 0.0))
        name = "bed_surface"

        bed_top_z_m: float | None = None
        if bed.get("height_m") is not None:
            bed_top_z_m = float(bed["height_m"])
        support = bed.get("support_surface")
        if isinstance(support, dict) and support.get("top_z_m") is not None:
            bed_top_z_m = float(support["top_z_m"])
        if bed_top_z_m is None:
            center_world = bed.get("center_world")
            if isinstance(center_world, (list, tuple)) and len(center_world) >= 3:
                bed_top_z_m = float(center_world[2])
        if bed_top_z_m is None or bed_top_z_m <= 0.0:
            return None

        center_floor = bed.get("center_on_floor")
        if isinstance(center_floor, (list, tuple)) and len(center_floor) >= 2:
            cx, cy = float(center_floor[0]), float(center_floor[1])
        else:
            cx, cy = 0.0, 0.0

        # Box bottom on world floor (z=0); top at calibrated bed surface height.
        center = (cx, cy, bed_top_z_m / 2.0)
        size = (lx, ly, bed_top_z_m)
        return cls(
            name=name,
            center=center,
            size=size,
            rotation_deg=rot,
        )


@dataclass
class CalibrationSceneSpec:
    bundle_path: Path
    cameras: dict[str, CameraCalib] = field(default_factory=dict)
    bed: BedSurfaceCalib | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_ids(self) -> list[str]:
        return list(self.cameras.keys())


def repo_root() -> Path:
    # joint_admittance_8dof -> control -> rm75_control(pkg) -> repo root
    return PACKAGE_DIR.parents[2]


def playground_root() -> Path | None:
    root = repo_root()
    if (root.parent / "camera_calibration").is_dir():
        return root.parent
    env = os.environ.get("REALUS_PLAYGROUND_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    return None


def default_calib_bundle_path() -> Path | None:
    env = os.environ.get("CAMERA_CALIB_BUNDLE", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()

    pg = playground_root()
    if pg is not None:
        candidate = pg / "camera_calibration/calibration_results/genesis_bundle.yaml"
        if candidate.is_file():
            return candidate.resolve()

    local = repo_root() / "data/calibration/genesis_bundle.yaml"
    if local.is_file():
        return local.resolve()
    return None


def load_calibration_scene(path: str | Path | None = None) -> CalibrationSceneSpec | None:
    bundle_path = Path(path).expanduser().resolve() if path is not None else default_calib_bundle_path()
    if bundle_path is None or not bundle_path.is_file():
        return None
    if yaml is None:
        raise ImportError("PyYAML is required to load calibration bundles")

    payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {}
    cameras_raw = payload.get("cameras")
    if not isinstance(cameras_raw, dict) or not cameras_raw:
        return None

    cameras: dict[str, CameraCalib] = {}
    for cam_id, cam_payload in cameras_raw.items():
        if not isinstance(cam_payload, dict):
            continue
        size = cam_payload.get("image_size") or [1280, 720]
        cameras[str(cam_id)] = CameraCalib(
            camera_id=str(cam_id),
            image_size=(int(size[0]), int(size[1])),
            intrinsics=np.asarray(cam_payload["intrinsics"], dtype=np.float64).reshape(3, 3),
            world_from_camera=np.asarray(cam_payload["world_from_camera"], dtype=np.float64).reshape(4, 4),
            camera_from_world=np.asarray(cam_payload["camera_from_world"], dtype=np.float64).reshape(4, 4),
        )

    bed = BedSurfaceCalib.from_bundle(payload)
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return CalibrationSceneSpec(bundle_path=bundle_path, cameras=cameras, bed=bed, metadata=dict(meta))


def quat_wxyz_from_euler_z(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(float(deg)) * 0.5
    return (float(math.cos(half)), 0.0, 0.0, float(math.sin(half)))


def add_calibration_scene(scene: Any, gs: Any, spec: CalibrationSceneSpec) -> dict[str, Any]:
    """Add ground (Z=0), bed box, and static cameras from a calibration bundle."""
    added: dict[str, Any] = {"cameras": {}, "bed": None, "bundle": str(spec.bundle_path)}

    if spec.bed is not None:
        bed = spec.bed
        entity = scene.add_entity(
            gs.morphs.Box(
                pos=bed.center,
                size=bed.size,
                quat=quat_wxyz_from_euler_z(bed.rotation_deg),
                fixed=True,
                collision=False,
                visualization=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=bed.color),
            name=bed.name,
        )
        added["bed"] = entity

    for cam_id in sorted(spec.cameras.keys()):
        cam = spec.cameras[cam_id]
        mount = cam.genesis_mount()
        camera = scene.add_camera(
            res=mount["res"],
            pos=mount["pos"],
            lookat=mount["lookat"],
            up=mount["up"],
            fov=float(mount["fov"]),
            GUI=False,
        )
        added["cameras"][cam_id] = camera

    return added
