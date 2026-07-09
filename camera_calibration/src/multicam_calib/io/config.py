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


def load_board(path: Path | None = None) -> BoardConfig:
    raw = _read_yaml(path or board_path())
    return BoardConfig(
        family=str(raw["family"]),
        rows=int(raw["rows"]),
        cols=int(raw["cols"]),
        id_row_step=int(raw["id_row_step"]),
        id_col_step=int(raw["id_col_step"]),
        id_origin=int(raw["id_origin"]),
        tag_size_m=float(raw["tag_size_m"]),
        tag_spacing_m=float(raw["tag_spacing_m"]),
    )


# -------------------- app.yaml --------------------

@dataclass
class StreamConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    color_format: str = "bgr8"


@dataclass
class SyncConfig:
    # Reject capture when host-timestamp spread across cameras exceeds this (ms).
    # 4x D435 on one USB3 hub typically sees 40–90 ms host spread at 30 fps; 80 ms
    # is safe for static-board calibration (board does not move during capture).
    max_spread_ms: float = 80.0
    # On capture, poll latest frames several times and keep the tightest-sync set.
    capture_poll_attempts: int = 30
    capture_poll_interval_ms: float = 2.0


@dataclass
class DetectorConfig:
    nthreads: int = 4
    quad_decimate: float = 1.0       # full-res for saved captures / calibration
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25
    preview_quad_decimate: float = 1.0  # must stay 1.0 for reliable preview on all cameras


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
    return AppConfig(stream=stream, sync=sync, detector=detector, recording=recording, calibration=calibration, ui=ui)


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
    min_floor_samples: int = 3
    min_bed_samples: int = 3
    min_corner_samples: int = 4
    min_cameras_corner_fusion: int = 3
    require_all_corner_tags: bool = True
    # Max 3D deviation (mm) between per-camera board pose and fused pose at corner tags.
    corner_fusion_max_std_mm: float = 40.0
    # Corners capture: lower tag count than full-board BA (partial views at bed edges).
    min_tags_corner_view: int = 20
    origin_mode: str = "bed_center_projected_to_floor"
    # Ground capture: board origin must be within this height (mm) above floor plane.
    floor_max_height_above_plane_mm: float = 120.0
    # Bed capture: board must be at least this high (mm) above floor plane.
    bed_min_height_above_floor_mm: float = 200.0
    # Reject ground capture if within this band (mm) of known bed height.
    bed_height_match_tolerance_mm: float = 100.0


def world_path() -> Path:
    return CONFIGS_DIR / "world.yaml"


def load_world(path: Path | None = None) -> WorldConfig:
    raw = _read_yaml(path or world_path())
    ct_raw = raw.get("corner_tags") or {}
    corner_tags = CornerTagsConfig(
        **{**CornerTagsConfig().__dict__, **(ct_raw if isinstance(ct_raw, dict) else {})}
    )
    defaults = WorldConfig()
    return WorldConfig(
        corner_tags=corner_tags,
        min_floor_samples=int(raw.get("min_floor_samples", defaults.min_floor_samples)),
        min_bed_samples=int(raw.get("min_bed_samples", defaults.min_bed_samples)),
        min_corner_samples=int(raw.get("min_corner_samples", defaults.min_corner_samples)),
        min_cameras_corner_fusion=int(raw.get("min_cameras_corner_fusion", defaults.min_cameras_corner_fusion)),
        require_all_corner_tags=bool(raw.get("require_all_corner_tags", defaults.require_all_corner_tags)),
        corner_fusion_max_std_mm=float(raw.get("corner_fusion_max_std_mm", defaults.corner_fusion_max_std_mm)),
        min_tags_corner_view=int(raw.get("min_tags_corner_view", defaults.min_tags_corner_view)),
        origin_mode=str(raw.get("origin_mode", defaults.origin_mode)),
        floor_max_height_above_plane_mm=float(
            raw.get("floor_max_height_above_plane_mm", defaults.floor_max_height_above_plane_mm)
        ),
        bed_min_height_above_floor_mm=float(
            raw.get("bed_min_height_above_floor_mm", defaults.bed_min_height_above_floor_mm)
        ),
        bed_height_match_tolerance_mm=float(
            raw.get("bed_height_match_tolerance_mm", defaults.bed_height_match_tolerance_mm)
        ),
    )
