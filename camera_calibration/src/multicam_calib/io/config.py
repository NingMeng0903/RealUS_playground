"""YAML config loading with the project's canonical paths.

All paths are resolved relative to the project root (`camera_calibration/`).
No global state; every helper takes/returns an explicit path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "calibration_results"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


# -------------------- cameras.yaml --------------------

@dataclass
class CameraEntry:
    alias: str
    serial: str
    driver: str = "realsense"
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "serial": self.serial,
            "driver": self.driver,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraEntry":
        return cls(
            alias=str(d["alias"]),
            serial=str(d["serial"]),
            driver=str(d.get("driver", "realsense")),
            model=str(d.get("model", "")),
        )


def cameras_path() -> Path:
    return CONFIGS_DIR / "cameras.yaml"


def load_camera_roster(path: Path | None = None) -> list[CameraEntry]:
    raw = _read_yaml(path or cameras_path())
    cams = raw.get("cameras") or []
    return [CameraEntry.from_dict(c) for c in cams]


def save_camera_roster(entries: list[CameraEntry], path: Path | None = None) -> None:
    _write_yaml(path or cameras_path(), {"cameras": [e.to_dict() for e in entries]})


# -------------------- board.yaml --------------------

@dataclass
class BoardConfig:
    family: str
    rows: int
    cols: int
    id_row_step: int
    id_col_step: int
    id_origin: int
    tag_size_m: float
    tag_spacing_m: float
    # pupil_apriltags corner index → board-frame [BL, BR, TR, TL].
    # Large bed board: (3,0,1,2). EE board tags are printed 90° from that.
    pupil_to_board_corner_perm: tuple[int, int, int, int] = (3, 0, 1, 2)

    @property
    def pitch_m(self) -> float:
        """Center-to-center distance between adjacent tags along one axis."""
        return self.tag_size_m + self.tag_spacing_m

    @property
    def n_tags(self) -> int:
        return self.rows * self.cols

    def tag_id(self, row: int, col: int) -> int:
        return self.id_origin + self.id_row_step * row + self.id_col_step * col

    def all_expected_ids(self) -> set[int]:
        return {self.tag_id(r, c) for r in range(self.rows) for c in range(self.cols)}


def board_path() -> Path:
    return CONFIGS_DIR / "board.yaml"


def _board_from_raw(raw: dict[str, Any]) -> BoardConfig:
    perm = raw.get("pupil_to_board_corner_perm", (3, 0, 1, 2))
    perm_t = tuple(int(v) for v in perm)
    if len(perm_t) != 4:
        raise ValueError(f"pupil_to_board_corner_perm must have 4 indices, got {perm!r}")
    return BoardConfig(
        family=str(raw["family"]),
        rows=int(raw["rows"]),
        cols=int(raw["cols"]),
        id_row_step=int(raw["id_row_step"]),
        id_col_step=int(raw["id_col_step"]),
        id_origin=int(raw["id_origin"]),
        tag_size_m=float(raw["tag_size_m"]),
        tag_spacing_m=float(raw["tag_spacing_m"]),
        pupil_to_board_corner_perm=perm_t,  # type: ignore[arg-type]
    )


def load_board(path: Path | None = None) -> BoardConfig:
    return _board_from_raw(_read_yaml(path or board_path()))


def board_ee_path() -> Path:
    return CONFIGS_DIR / "board_ee.yaml"


def load_board_ee(path: Path | None = None) -> BoardConfig:
    """4×4 end-effector AprilTag board (robot-world / hand-eye phase)."""
    return _board_from_raw(_read_yaml(path or board_ee_path()))


# -------------------- app.yaml --------------------

@dataclass
class StreamConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    color_format: str = "bgr8"


@dataclass
class SyncConfig:
    max_spread_ms: float = 80.0
    capture_poll_attempts: int = 30
    capture_poll_interval_ms: float = 2.0
    use_device_timestamp: bool = True


@dataclass
class PreviewConfig:
    source: str = "local"  # local | zmq
    zmq_connect: str = "tcp://127.0.0.1:17356"
    zmq_preview_topic: str = "amongus_camera_preview_v1"
    zmq_capture_topic: str = "amongus_camera_frame_v1"


@dataclass
class DetectorConfig:
    nthreads: int = 4
    quad_decimate: float = 1.0       # full-res for saved captures / calibration
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25
    preview_quad_decimate: float = 1.0  # decimate=2.0 drops oblique 4cm tags even at 1080p


@dataclass
class UIConfig:
    preview_refresh_hz: int = 15


@dataclass
class RecordingConfig:
    root_dir: str = "data"
    save_images: bool = True
    jpeg_quality: int = 95


@dataclass
class BAConfig:
    loss: str = "cauchy"
    f_scale: float = 1.0
    max_nfev: int = 200
    verbose: int = 1


@dataclass
class CalibrationConfig:
    min_tags_per_view: int = 8
    min_frames: int = 5
    min_qualifying_cameras_hint: int = 3
    ba: BAConfig = field(default_factory=BAConfig)


@dataclass
class AppConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def app_path() -> Path:
    return CONFIGS_DIR / "app.yaml"


def load_app(path: Path | None = None) -> AppConfig:
    raw = _read_yaml(path or app_path())

    def _sec(name: str) -> dict[str, Any]:
        v = raw.get(name) or {}
        return v if isinstance(v, dict) else {}

    stream = StreamConfig(**{**StreamConfig().__dict__, **_sec("stream")})
    sync = SyncConfig(**{**SyncConfig().__dict__, **_sec("sync")})
    preview = PreviewConfig(**{**PreviewConfig().__dict__, **_sec("preview")})
    detector = DetectorConfig(**{**DetectorConfig().__dict__, **_sec("detector")})
    recording = RecordingConfig(**{**RecordingConfig().__dict__, **_sec("recording")})
    ui = UIConfig(**{**UIConfig().__dict__, **_sec("ui")})

    calib_raw = _sec("calibration")
    ba_raw = calib_raw.get("ba") or {}
    ba = BAConfig(**{**BAConfig().__dict__, **(ba_raw if isinstance(ba_raw, dict) else {})})
    calib_kwargs = {k: v for k, v in calib_raw.items() if k != "ba"}
    calibration = CalibrationConfig(
        **{**{k: v for k, v in CalibrationConfig().__dict__.items() if k not in ("ba",)}, **calib_kwargs},
        ba=ba,
    )
    return AppConfig(stream=stream, sync=sync, preview=preview, detector=detector, recording=recording, calibration=calibration, ui=ui)


# -------------------- orbbec.yaml (Stage 3 RGB-D check) --------------------

@dataclass
class OrbbecConfig:
    serial: str = ""
    color_width: int = 640
    color_height: int = 480
    depth_width: int = 0
    depth_height: int = 0
    fps: int = 30
    depth_fps: int = 15
    depth_flip_h: bool = True
    align: str = "d2c_sw"
    min_depth_m: float = 0.20
    max_depth_m: float = 3.00
    overlay_alpha: float = 0.45
    min_valid_points: int = 500
    min_valid_frac: float = 0.02
    bundled_sdk_root: str = "../tmp/OrbbecSDK_Python_v1.1.4_linux_x64_release"
    v1_python: str = "/media/camp/EXT_DRIVE/envs/orbbec_sdk39/bin/python"


def orbbec_path() -> Path:
    return CONFIGS_DIR / "orbbec.yaml"


def load_orbbec(path: Path | None = None) -> OrbbecConfig:
    raw = _read_yaml(path or orbbec_path())
    body = raw.get("orbbec", raw) if isinstance(raw, dict) else {}
    if not isinstance(body, dict):
        body = {}
    defaults = OrbbecConfig()
    return OrbbecConfig(
        serial=str(body.get("serial", defaults.serial) or ""),
        color_width=int(body.get("color_width", defaults.color_width)),
        color_height=int(body.get("color_height", defaults.color_height)),
        depth_width=int(body.get("depth_width", defaults.depth_width)),
        depth_height=int(body.get("depth_height", defaults.depth_height)),
        fps=int(body.get("fps", defaults.fps)),
        depth_fps=int(body.get("depth_fps", body.get("fps", defaults.depth_fps))),
        depth_flip_h=bool(body.get("depth_flip_h", defaults.depth_flip_h)),
        align=str(body.get("align", defaults.align)),
        min_depth_m=float(body.get("min_depth_m", defaults.min_depth_m)),
        max_depth_m=float(body.get("max_depth_m", defaults.max_depth_m)),
        overlay_alpha=float(body.get("overlay_alpha", defaults.overlay_alpha)),
        min_valid_points=int(body.get("min_valid_points", defaults.min_valid_points)),
        min_valid_frac=float(body.get("min_valid_frac", defaults.min_valid_frac)),
        bundled_sdk_root=str(body.get("bundled_sdk_root", defaults.bundled_sdk_root)),
        v1_python=str(body.get("v1_python", defaults.v1_python) or defaults.v1_python),
    )


# -------------------- world.yaml (Stage 2) --------------------

@dataclass
class CornerTagsConfig:
    tl: int = 151
    tr: int = 1
    bl: int = 162
    br: int = 12

    def all_ids(self) -> list[int]:
        return [int(self.tl), int(self.tr), int(self.bl), int(self.br)]


@dataclass
class WorldConfig:
    corner_tags: CornerTagsConfig = field(default_factory=CornerTagsConfig)
    min_robot_samples: int = 3
    min_bed_samples: int = 3
    min_corner_samples: int = 4
    min_cameras_corner_fusion: int = 2
    require_all_corner_tags: bool = True
    # Max 3D deviation (mm) between per-camera board pose and fused pose at corner tags.
    corner_fusion_max_std_mm: float = 40.0
    # Corners capture: lower tag count than full-board BA (partial views at bed edges).
    min_tags_corner_view: int = 20
    min_cameras_robot: int = 2
    min_tags_robot_view: int = 7
    origin_mode: str = "bed_center_projected_to_floor"
    # Rotate world +X/+Y about +Z so bed edges are parallel to world axes (after origin shift).
    # Default false: world +X is the rail axis (see xy_reference).
    align_xy_to_bed: bool = False
    xy_reference: str = "rail"
    bed_skew_warn_deg: float = 5.0
    # Bed capture: board must be at least this high (mm) above floor plane.
    bed_min_height_above_floor_mm: float = 200.0
    # Reject bed-height match diagnostics (unused for robot phase).
    bed_height_match_tolerance_mm: float = 100.0
    # Large-board thickness; subtracted from the fitted tag-plane height.
    board_thickness_m: float = 0.0035
    # Kept so older callers/tests that still name the first-phase minimum compile.
    min_floor_samples: int = 3

    def __post_init__(self) -> None:
        self.min_floor_samples = int(self.min_robot_samples)


def world_path() -> Path:
    return CONFIGS_DIR / "world.yaml"


def load_world(path: Path | None = None) -> WorldConfig:
    raw = _read_yaml(path or world_path())
    ct_raw = raw.get("corner_tags") or {}
    corner_tags = CornerTagsConfig(
        **{**CornerTagsConfig().__dict__, **(ct_raw if isinstance(ct_raw, dict) else {})}
    )
    defaults = WorldConfig()
    min_robot = int(
        raw.get("min_robot_samples", raw.get("min_floor_samples", defaults.min_robot_samples))
    )
    return WorldConfig(
        corner_tags=corner_tags,
        min_robot_samples=min_robot,
        min_bed_samples=int(raw.get("min_bed_samples", defaults.min_bed_samples)),
        min_corner_samples=int(raw.get("min_corner_samples", defaults.min_corner_samples)),
        min_cameras_corner_fusion=int(raw.get("min_cameras_corner_fusion", defaults.min_cameras_corner_fusion)),
        require_all_corner_tags=bool(raw.get("require_all_corner_tags", defaults.require_all_corner_tags)),
        corner_fusion_max_std_mm=float(raw.get("corner_fusion_max_std_mm", defaults.corner_fusion_max_std_mm)),
        min_tags_corner_view=int(raw.get("min_tags_corner_view", defaults.min_tags_corner_view)),
        min_cameras_robot=int(raw.get("min_cameras_robot", defaults.min_cameras_robot)),
        min_tags_robot_view=int(raw.get("min_tags_robot_view", defaults.min_tags_robot_view)),
        origin_mode=str(raw.get("origin_mode", defaults.origin_mode)),
        align_xy_to_bed=bool(raw.get("align_xy_to_bed", defaults.align_xy_to_bed)),
        xy_reference=str(raw.get("xy_reference", defaults.xy_reference)),
        bed_skew_warn_deg=float(raw.get("bed_skew_warn_deg", defaults.bed_skew_warn_deg)),
        bed_min_height_above_floor_mm=float(
            raw.get("bed_min_height_above_floor_mm", defaults.bed_min_height_above_floor_mm)
        ),
        bed_height_match_tolerance_mm=float(
            raw.get("bed_height_match_tolerance_mm", defaults.bed_height_match_tolerance_mm)
        ),
        board_thickness_m=float(raw.get("board_thickness_m", defaults.board_thickness_m)),
        min_floor_samples=min_robot,
    )


# -------------------- robot.yaml (Stage 2 robot / hand-eye) --------------------

@dataclass
class RobotShmConfig:
    name: str = "rm75_state"
    max_age_s: float = 0.25


@dataclass
class RobotStillnessConfig:
    window_s: float = 0.15
    trans_m: float = 0.001
    rot_deg: float = 0.3
    rail_m: float = 0.0005


@dataclass
class KinematicFitConfig:
    joint_offsets: bool = True
    rail_span_min_m: float = 0.15
    board_scale_warn: float = 0.002


@dataclass
class RobotConfig:
    base_link_height_above_floor_m: float = 0.274
    rail_y_origin_in_railbase_m: tuple[float, float, float] = (0.0, -0.4, 0.0)
    shm: RobotShmConfig = field(default_factory=RobotShmConfig)
    stillness: RobotStillnessConfig = field(default_factory=RobotStillnessConfig)
    wbc_urdf: str = "../rm75_control/rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf"
    kinematic_fit: KinematicFitConfig = field(default_factory=KinematicFitConfig)

    def wbc_urdf_path(self) -> Path:
        p = Path(self.wbc_urdf)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p


def robot_path() -> Path:
    return CONFIGS_DIR / "robot.yaml"


def load_robot(path: Path | None = None) -> RobotConfig:
    raw = _read_yaml(path or robot_path())
    defaults = RobotConfig()
    shm_raw = raw.get("shm") or {}
    still_raw = raw.get("stillness") or {}
    origin = raw.get("rail_y_origin_in_railbase_m") or list(defaults.rail_y_origin_in_railbase_m)
    fit_raw = raw.get("kinematic_fit") or {}
    return RobotConfig(
        base_link_height_above_floor_m=float(
            raw.get("base_link_height_above_floor_m", defaults.base_link_height_above_floor_m)
        ),
        rail_y_origin_in_railbase_m=(float(origin[0]), float(origin[1]), float(origin[2])),
        shm=RobotShmConfig(
            name=str(shm_raw.get("name", defaults.shm.name)),
            max_age_s=float(shm_raw.get("max_age_s", defaults.shm.max_age_s)),
        ),
        stillness=RobotStillnessConfig(
            window_s=float(still_raw.get("window_s", defaults.stillness.window_s)),
            trans_m=float(still_raw.get("trans_m", defaults.stillness.trans_m)),
            rot_deg=float(still_raw.get("rot_deg", defaults.stillness.rot_deg)),
            rail_m=float(still_raw.get("rail_m", defaults.stillness.rail_m)),
        ),
        wbc_urdf=str(raw.get("wbc_urdf", defaults.wbc_urdf)),
        kinematic_fit=KinematicFitConfig(
            joint_offsets=bool(fit_raw.get("joint_offsets", defaults.kinematic_fit.joint_offsets)),
            rail_span_min_m=float(
                fit_raw.get("rail_span_min_m", defaults.kinematic_fit.rail_span_min_m)
            ),
            board_scale_warn=float(
                fit_raw.get("board_scale_warn", defaults.kinematic_fit.board_scale_warn)
            ),
        ),
    )
