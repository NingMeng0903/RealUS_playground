"""Stage 2 multi-phase recording session (robot / bed / bed corners).

On disk only two fixed folders under ``data/stage2_world/``:

- ``working/`` — current UI session (cleared on every app start)
- ``last/``     — previous completed calibration (optional load; overwritten on export)
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from multicam_calib.board.detector import AprilTagDetector, TagDetection
from multicam_calib.io.config import DATA_DIR, RecordingConfig, RESULTS_DIR, RobotConfig, load_robot
from multicam_calib.io.results import (
    WorldMeta,
    extrinsics_world_path,
    load_robot_world,
    load_world_meta,
    robot_world_path,
    world_meta_path,
)
from multicam_calib.recording.session import RecordingSession, Sample, ViewDetections

Stage2Phase = Literal["robot", "bed", "corners"]
PHASES: tuple[Stage2Phase, ...] = ("robot", "bed", "corners")
ALIGNED_STATE_NAME = "aligned_state.json"
WORKING_DIR_NAME = "working"
LAST_DIR_NAME = "last"


@dataclass
class Stage2AlignedState:
    """Persisted after each per-phase Run (robot → bed → corners).

    ``floor_aligned`` still means "world +Z / temporary origin are known" —
    they now come from robot geometry rather than a floor-board plane fit.
    """

    floor_aligned: bool = False
    bed_aligned: bool = False
    corners_aligned: bool = False
    floor_plane_residual_mm: float = 0.0
    floor_normal: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    floor_d: float = 0.0
    x_axis: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    y_axis: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    z_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    origin_tmp_ref: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bed_height_m: float | None = None
    bed_plane_residual_mm: float | None = None
    T_ref_railbase: list[list[float]] | None = None
    T_tcp_board: list[list[float]] | None = None
    rail_direction_ref: list[float] | None = None
    baselink_z_tilt_from_world_z_deg: float | None = None
    robot_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "Stage2AlignedState":
        return cls(
            floor_aligned=bool(d.get("floor_aligned", False)),
            bed_aligned=bool(d.get("bed_aligned", False)),
            corners_aligned=bool(d.get("corners_aligned", False)),
            floor_plane_residual_mm=float(d.get("floor_plane_residual_mm", 0.0)),
            floor_normal=list(d.get("floor_normal") or [0.0, 0.0, 1.0]),
            floor_d=float(d.get("floor_d", 0.0)),
            x_axis=list(d.get("x_axis") or [1.0, 0.0, 0.0]),
            y_axis=list(d.get("y_axis") or [0.0, 1.0, 0.0]),
            z_axis=list(d.get("z_axis") or [0.0, 0.0, 1.0]),
            origin_tmp_ref=list(d.get("origin_tmp_ref") or [0.0, 0.0, 0.0]),
            bed_height_m=(float(d["bed_height_m"]) if d.get("bed_height_m") is not None else None),
            bed_plane_residual_mm=(
                float(d["bed_plane_residual_mm"]) if d.get("bed_plane_residual_mm") is not None else None
            ),
            T_ref_railbase=d.get("T_ref_railbase"),
            T_tcp_board=d.get("T_tcp_board"),
            rail_direction_ref=list(d["rail_direction_ref"]) if d.get("rail_direction_ref") is not None else None,
            baselink_z_tilt_from_world_z_deg=(
                float(d["baselink_z_tilt_from_world_z_deg"])
                if d.get("baselink_z_tilt_from_world_z_deg") is not None
                else None
            ),
            robot_diagnostics=dict(d.get("robot_diagnostics") or {}),
        )


def _count_phase_samples(phase_dir: Path) -> int:
    sd = phase_dir / "samples"
    if not sd.is_dir():
        return 0
    return sum(1 for e in sd.iterdir() if (e / "snapshot.json").is_file())


def session_sample_counts(session_root: Path) -> tuple[int, int, int]:
    return (
        _count_phase_samples(session_root / "robot"),
        _count_phase_samples(session_root / "bed"),
        _count_phase_samples(session_root / "corners"),
    )


def _session_has_data(session_root: Path) -> bool:
    return sum(session_sample_counts(session_root)) > 0


def reconstruct_aligned_state_from_exports(
    robot_world: dict[str, Any] | None = None,
    world_meta: WorldMeta | None = None,
    robot_cfg: RobotConfig | None = None,
) -> Stage2AlignedState | None:
    """Rebuild robot/bed alignment from ``robot_world.yaml`` + ``world_meta.yaml``.

    ``last/aligned_state.json`` is not always archived with per-phase samples.
    The exported yamls are the durable result of those Runs.
    """
    from multicam_calib.calib.robot_world import T_railbase_baselink, world_axes_from_railbase

    rw = robot_world if robot_world is not None else load_robot_world()
    if not rw or not rw.get("T_ref_railbase"):
        return None
    cfg = robot_cfg or load_robot()
    T_ref = np.asarray(rw["T_ref_railbase"], dtype=np.float64)
    x, y, z = world_axes_from_railbase(T_ref)
    T_bl0 = T_ref @ T_railbase_baselink(0.0, cfg.rail_y_origin_in_railbase_m)
    h = float(cfg.base_link_height_above_floor_m)
    origin_tmp = T_bl0[:3, 3] - h * z
    floor_d = float(origin_tmp @ z)
    meta = world_meta if world_meta is not None else load_world_meta()
    phases = list((meta.phases_completed if meta is not None else None) or [])
    bed_h = float(meta.bed_height_m) if meta is not None and meta.bed_height_m else None
    bed_res = (
        float(meta.bed_plane_residual_mm)
        if meta is not None and meta.bed_plane_residual_mm is not None
        else None
    )
    bed_ok = bed_h is not None and ("bed" in phases or (bed_h or 0.0) > 0.05)
    return Stage2AlignedState(
        floor_aligned=True,
        bed_aligned=bool(bed_ok),
        corners_aligned=False,
        floor_plane_residual_mm=float((meta.floor_plane_residual_mm if meta is not None else 0.0) or 0.0),
        floor_normal=z.tolist(),
        floor_d=floor_d,
        x_axis=x.tolist(),
        y_axis=y.tolist(),
        z_axis=z.tolist(),
        origin_tmp_ref=origin_tmp.tolist(),
        bed_height_m=bed_h if bed_ok else None,
        bed_plane_residual_mm=bed_res if bed_ok else None,
        T_ref_railbase=T_ref.tolist(),
        T_tcp_board=rw.get("T_tcp_board"),
        rail_direction_ref=T_ref[:3, 1].tolist(),
        baselink_z_tilt_from_world_z_deg=float(
            ((rw.get("diagnostics") or {}).get("baselink_z_tilt_from_world_z_deg")) or 0.0
        ),
        robot_diagnostics=dict(rw.get("diagnostics") or {}),
    )


def _ensure_phase_dirs(session_root: Path) -> None:
    for phase in PHASES:
        (session_root / phase / "samples").mkdir(parents=True, exist_ok=True)


def _wipe_session_tree(session_root: Path) -> None:
    if session_root.exists():
        shutil.rmtree(session_root)
    _ensure_phase_dirs(session_root)


def _copy_session_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _next_sample_index(samples: list[Sample]) -> int:
    if not samples:
        return 0
    return max(s.index for s in samples) + 1


@dataclass
class PhaseRecordingSession:
    """One phase (robot/bed/corners) inside a Stage 2 session root."""

    session_root: Path
    phase: Stage2Phase
    aliases: list[str]
    detector: AprilTagDetector
    recording_cfg: RecordingConfig
    samples: list[Sample] = field(default_factory=list)

    @property
    def phase_dir(self) -> Path:
        return self.session_root / self.phase

    @property
    def samples_dir(self) -> Path:
        return self.phase_dir / "samples"

    def add_sample(
        self,
        images_bgr: dict[str, np.ndarray],
        host_timestamp_ns: int,
        metadata: dict | None = None,
    ) -> Sample:
        idx = _next_sample_index(self.samples)
        sample_dir = self.samples_dir / f"{idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        views: dict[str, ViewDetections] = {}
        image_paths: dict[str, Path] = {}
        for alias, img in images_bgr.items():
            dets: list[TagDetection] = self.detector.detect(img)
            views[alias] = ViewDetections(alias=alias, tags=self.detector.detections_to_dict(dets))
            if self.recording_cfg.save_images:
                img_path = sample_dir / f"{alias}.png"
                cv2.imwrite(str(img_path), img)
                image_paths[alias] = img_path
            (sample_dir / f"{alias}_detections.json").write_text(json.dumps(views[alias].to_json()))
        sample = Sample(
            index=idx,
            host_timestamp_ns=int(host_timestamp_ns),
            views=views,
            image_paths=image_paths,
            metadata={**(metadata or {}), "phase": self.phase},
        )
        (sample_dir / "snapshot.json").write_text(json.dumps(sample.to_json(), indent=2))
        self.samples.append(sample)
        return sample

    def remove_last(self) -> Sample | None:
        if not self.samples:
            return None
        s = self.samples.pop()
        sample_dir = self.samples_dir / f"{s.index:06d}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        return s

    def remove_sample_by_index(self, sample_index: int) -> Sample | None:
        """Delete one sample by its index; remaining samples keep their original numbers."""
        pos = next((i for i, s in enumerate(self.samples) if s.index == sample_index), None)
        if pos is None:
            return None
        removed = self.samples.pop(pos)
        sample_dir = self.samples_dir / f"{removed.index:06d}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        return removed

    def clear(self) -> None:
        for s in list(self.samples):
            sample_dir = self.samples_dir / f"{s.index:06d}"
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
        self.samples.clear()

    def load_existing(self) -> None:
        self.samples.clear()
        if not self.samples_dir.exists():
            return
        for entry in sorted(self.samples_dir.iterdir()):
            snap = entry / "snapshot.json"
            if not snap.exists():
                continue
            d = json.loads(snap.read_text())
            views = {alias: ViewDetections.from_json(v) for alias, v in d["views"].items()}
            self.samples.append(
                Sample(
                    index=int(d["index"]),
                    host_timestamp_ns=int(d["host_timestamp_ns"]),
                    views=views,
                    image_paths={alias: entry / p for alias, p in (d.get("image_paths") or {}).items()},
                    metadata=d.get("metadata") or {},
                )
            )
        self.samples.sort(key=lambda s: s.index)


@dataclass
class Stage2SessionBundle:
    """Robot + bed + corners phases sharing one session root."""

    session_root: Path
    aliases: list[str]
    detector: AprilTagDetector
    recording_cfg: RecordingConfig
    detector_ee: AprilTagDetector | None = None
    robot: PhaseRecordingSession = field(init=False)
    bed: PhaseRecordingSession = field(init=False)
    corners: PhaseRecordingSession = field(init=False)

    def __post_init__(self) -> None:
        _ensure_phase_dirs(self.session_root)
        self.robot = self._phase("robot")
        self.bed = self._phase("bed")
        self.corners = self._phase("corners")

    def _phase(self, phase: Stage2Phase) -> PhaseRecordingSession:
        det = self.detector_ee if phase == "robot" and self.detector_ee is not None else self.detector
        return PhaseRecordingSession(
            session_root=self.session_root,
            phase=phase,
            aliases=list(self.aliases),
            detector=det,
            recording_cfg=self.recording_cfg,
        )

    @classmethod
    def stage2_root(cls, root: Path | None = None) -> Path:
        return (root or DATA_DIR) / "stage2_world"

    @classmethod
    def working_path(cls, root: Path | None = None) -> Path:
        return cls.stage2_root(root) / WORKING_DIR_NAME

    @classmethod
    def last_path(cls, root: Path | None = None) -> Path:
        return cls.stage2_root(root) / LAST_DIR_NAME

    @classmethod
    def last_exists(cls, root: Path | None = None) -> bool:
        cls._migrate_legacy_to_last_if_needed(root)
        return _session_has_data(cls.last_path(root))

    @classmethod
    def last_counts_label(cls, root: Path | None = None) -> str:
        f, b, c = session_sample_counts(cls.last_path(root))
        return f"last  (robot {f}, bed {b}, corners {c})"

    @classmethod
    def last_phase_sample_count(cls, phase: Stage2Phase, root: Path | None = None) -> int:
        cls._migrate_legacy_to_last_if_needed(root)
        return _count_phase_samples(cls.last_path(root) / phase)

    @classmethod
    def last_phase_exists(cls, phase: Stage2Phase, root: Path | None = None) -> bool:
        return cls.last_phase_sample_count(phase, root) > 0

    @classmethod
    def copy_last_phase_to_working(cls, phase: Stage2Phase, root: Path | None = None) -> bool:
        """Copy one phase tree from ``last/`` into ``working/`` (other phases untouched)."""
        cls._migrate_legacy_to_last_if_needed(root)
        src = cls.last_path(root) / phase
        dst = cls.working_path(root) / phase
        if _count_phase_samples(src) == 0:
            return False
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True

    @classmethod
    def last_aligned_state(cls, root: Path | None = None) -> "Stage2AlignedState | None":
        p = cls.last_path(root) / ALIGNED_STATE_NAME
        if not p.exists():
            return None
        return Stage2AlignedState.from_json_dict(json.loads(p.read_text()))

    @classmethod
    def _migrate_legacy_to_last_if_needed(cls, root: Path | None = None) -> None:
        """One-time: promote the most recent non-empty ``session_*`` folder to ``last/``.

        Recency is derived from the ``session_YYYYMMDD_HHMMSS`` name rather than
        filesystem mtimes (see ``RecordingSession._migrate_legacy_to_last_if_needed``).
        """
        last_p = cls.last_path(root)
        if _session_has_data(last_p):
            return
        base = cls.stage2_root(root)
        if not base.is_dir():
            return
        candidates: list[Path] = []
        for p in base.iterdir():
            if not p.is_dir() or not p.name.startswith("session_"):
                continue
            if sum(session_sample_counts(p)) > 0:
                candidates.append(p)
        if not candidates:
            return
        candidates.sort(key=lambda p: p.name, reverse=True)
        _copy_session_tree(candidates[0], last_p)

    @classmethod
    def create_fresh_for_ui(
        cls,
        *,
        aliases: list[str],
        detector: AprilTagDetector,
        recording_cfg: RecordingConfig,
        root: Path | None = None,
        detector_ee: AprilTagDetector | None = None,
    ) -> "Stage2SessionBundle":
        """Empty working session — called on every app start."""
        cls.stage2_root(root).mkdir(parents=True, exist_ok=True)
        wp = cls.working_path(root)
        _wipe_session_tree(wp)
        last_state = cls.last_path(root) / ALIGNED_STATE_NAME
        if last_state.is_file():
            shutil.copy2(last_state, wp / ALIGNED_STATE_NAME)
        bundle = cls(
            session_root=wp,
            aliases=list(aliases),
            detector=detector,
            recording_cfg=recording_cfg,
            detector_ee=detector_ee,
        )
        bundle.write_manifest()
        return bundle

    @classmethod
    def load_last_into_working(
        cls,
        *,
        aliases: list[str],
        detector: AprilTagDetector,
        recording_cfg: RecordingConfig,
        root: Path | None = None,
        detector_ee: AprilTagDetector | None = None,
    ) -> "Stage2SessionBundle | None":
        """Copy ``last/`` → ``working/`` and open it. Returns None if no last data."""
        if not cls.last_exists(root):
            return None
        wp = cls.working_path(root)
        _copy_session_tree(cls.last_path(root), wp)
        return cls.open_existing(
            wp,
            aliases=aliases,
            detector=detector,
            recording_cfg=recording_cfg,
            detector_ee=detector_ee,
        )

    @classmethod
    def archive_working_as_last(cls, root: Path | None = None) -> None:
        """Overwrite ``last/`` with the current ``working/`` tree."""
        wp = cls.working_path(root)
        if not _session_has_data(wp):
            return
        _copy_session_tree(wp, cls.last_path(root))

    @classmethod
    def archive_working_phase_as_last(cls, phase: Stage2Phase, root: Path | None = None) -> None:
        """Overwrite ``last/<phase>/`` with ``working/<phase>/`` (other last phases untouched)."""
        wp = cls.working_path(root) / phase
        if _count_phase_samples(wp) == 0:
            return
        last_root = cls.last_path(root)
        _ensure_phase_dirs(last_root)
        dst = last_root / phase
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(wp, dst)
        src_state = cls.working_path(root) / ALIGNED_STATE_NAME
        if src_state.is_file():
            shutil.copy2(src_state, last_root / ALIGNED_STATE_NAME)

    @classmethod
    def open_existing(
        cls,
        session_root: Path,
        *,
        aliases: list[str],
        detector: AprilTagDetector,
        recording_cfg: RecordingConfig,
        detector_ee: AprilTagDetector | None = None,
    ) -> "Stage2SessionBundle":
        bundle = cls(
            session_root=session_root,
            aliases=list(aliases),
            detector=detector,
            recording_cfg=recording_cfg,
            detector_ee=detector_ee,
        )
        bundle.robot.load_existing()
        bundle.bed.load_existing()
        bundle.corners.load_existing()
        return bundle

    def phase_session(self, phase: Stage2Phase | str) -> PhaseRecordingSession:
        key = "robot" if phase == "floor" else phase
        return {"robot": self.robot, "bed": self.bed, "corners": self.corners}[key]

    def write_manifest(self) -> None:
        manifest = {
            "stage": "stage2_world",
            "aliases": self.aliases,
            "phases_completed": self.phases_completed(),
            "aligned": self.load_aligned_state().to_json_dict(),
            "recording": {
                "save_images": bool(self.recording_cfg.save_images),
                "jpeg_quality": int(self.recording_cfg.jpeg_quality),
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.session_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def aligned_state_path(self) -> Path:
        return self.session_root / ALIGNED_STATE_NAME

    def load_aligned_state(self) -> Stage2AlignedState:
        p = self.aligned_state_path()
        if not p.exists():
            return Stage2AlignedState()
        return Stage2AlignedState.from_json_dict(json.loads(p.read_text()))

    def save_aligned_state(self, state: Stage2AlignedState) -> None:
        self.aligned_state_path().write_text(json.dumps(state.to_json_dict(), indent=2))
        self.write_manifest()

    def inherit_prereq_alignment_from_last(
        self, phase: Stage2Phase, root: Path | None = None
    ) -> list[str]:
        """Fill in missing prerequisite alignment (robot/bed) from ``last/`` state.

        Loading only one phase's samples from ``last/`` (e.g. "Load last
        corners" into an otherwise-empty working session) leaves the earlier
        phases' *alignment results* missing even though the loaded samples
        were captured and validated against exactly that earlier alignment.
        Running a later phase only needs the fitted robot/bed geometry — not
        the raw robot/bed sample images — so pull those numbers in from
        ``last/aligned_state.json`` whenever this working session doesn't
        already have its own (never overwrites a fresher in-working result).
        """
        last_state = Stage2SessionBundle.last_aligned_state(root)
        if last_state is None:
            last_state = reconstruct_aligned_state_from_exports()
        if last_state is None:
            return []
        state = self.load_aligned_state()
        notes: list[str] = []
        if phase in ("bed", "corners") and not state.floor_aligned and last_state.floor_aligned:
            state.floor_aligned = True
            state.floor_plane_residual_mm = last_state.floor_plane_residual_mm
            state.floor_normal = list(last_state.floor_normal)
            state.floor_d = last_state.floor_d
            state.x_axis = list(last_state.x_axis)
            state.y_axis = list(last_state.y_axis)
            state.z_axis = list(last_state.z_axis)
            state.origin_tmp_ref = list(last_state.origin_tmp_ref)
            state.T_ref_railbase = last_state.T_ref_railbase
            state.T_tcp_board = last_state.T_tcp_board
            state.rail_direction_ref = last_state.rail_direction_ref
            state.baselink_z_tilt_from_world_z_deg = last_state.baselink_z_tilt_from_world_z_deg
            state.robot_diagnostics = dict(last_state.robot_diagnostics)
            notes.append(f"robot/floor alignment ({last_state.floor_plane_residual_mm:.2f} mm RMSE)")
        if phase == "corners" and not state.bed_aligned and last_state.bed_aligned:
            state.bed_aligned = True
            state.bed_height_m = last_state.bed_height_m
            state.bed_plane_residual_mm = last_state.bed_plane_residual_mm
            bed_mm = (last_state.bed_height_m or 0.0) * 1000.0
            notes.append(f"bed alignment (z={bed_mm:.1f} mm)")
        if notes:
            self.save_aligned_state(state)
        return notes

    def clear_aligned_state(self) -> None:
        p = self.aligned_state_path()
        if p.exists():
            p.unlink()

    def phases_completed(self) -> list[str]:
        done: list[str] = []
        if self.robot.samples:
            done.append("robot")
        if self.bed.samples:
            done.append("bed")
        if self.corners.samples:
            done.append("corners")
        return done

    def invalidate_from_robot(self) -> None:
        """Clear bed + corners samples and stale world results."""
        self.bed.clear()
        self.corners.clear()
        self.clear_aligned_state()
        self._remove_world_results()
        self.write_manifest()

    # Back-compat alias.
    invalidate_from_floor = invalidate_from_robot

    def invalidate_from_bed(self) -> None:
        """Clear corners and bed-dependent world metadata."""
        self.corners.clear()
        state = self.load_aligned_state()
        state.bed_aligned = False
        state.corners_aligned = False
        state.bed_height_m = None
        state.bed_plane_residual_mm = None
        self.save_aligned_state(state)
        self._remove_world_results()
        self.write_manifest()

    @staticmethod
    def _remove_world_results() -> None:
        for p in (extrinsics_world_path(), world_meta_path(), robot_world_path()):
            if p.exists():
                p.unlink()

    def as_legacy_session(self, phase: Stage2Phase) -> RecordingSession:
        """Adapt a phase to RecordingSession for pipeline reuse."""
        ps = self.phase_session(phase)
        legacy = RecordingSession(
            session_dir=ps.phase_dir,
            stage=f"stage2_{phase}",
            aliases=ps.aliases,
            detector=ps.detector,
            recording_cfg=ps.recording_cfg,
            samples=list(ps.samples),
        )
        return legacy
