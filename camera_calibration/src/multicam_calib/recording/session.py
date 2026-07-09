"""Persistent calibration session.

Every "capture" produces one *sample*: RGB frames from every camera plus the
AprilTag detections computed on each frame. Samples are indexed by an integer
starting at 0 and are stored under ``data/<stage>/<session_id>/``::

    session_id/
        manifest.json              # session metadata
        samples/
            000000/
                cam1.png
                cam1_detections.json
                cam2.png
                cam2_detections.json
                ...
                snapshot.json

On disk only two fixed session ids are used per stage, mirroring Stage 2's
scheme: ``working/`` (current UI session, reset on every app start) and
``last/`` (previous completed calibration, optionally reloaded into
``working/``). This keeps only one calibration's worth of data around instead
of accumulating a timestamped folder per app launch.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from multicam_calib.board.detector import AprilTagDetector, TagDetection
from multicam_calib.io.config import DATA_DIR, RecordingConfig

WORKING_DIR_NAME = "working"
LAST_DIR_NAME = "last"


def _next_sample_index(samples: list["Sample"]) -> int:
    if not samples:
        return 0
    return max(s.index for s in samples) + 1


def _count_samples(session_dir: Path) -> int:
    sd = session_dir / "samples"
    if not sd.is_dir():
        return 0
    return sum(1 for e in sd.iterdir() if (e / "snapshot.json").is_file())


def _session_has_data(session_dir: Path) -> bool:
    return _count_samples(session_dir) > 0


def _wipe_session_tree(session_dir: Path) -> None:
    if session_dir.exists():
        shutil.rmtree(session_dir)
    (session_dir / "samples").mkdir(parents=True, exist_ok=True)


def _copy_session_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@dataclass
class ViewDetections:
    """Detections for one (frame, camera) pair, ready for downstream algorithms."""

    alias: str
    tags: dict[int, np.ndarray]  # tag_id -> (4, 2) pixel corners in [BL,BR,TR,TL]

    def num_tags(self) -> int:
        return len(self.tags)

    def to_json(self) -> dict:
        return {"alias": self.alias, "tags": {str(k): v.tolist() for k, v in self.tags.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "ViewDetections":
        return cls(
            alias=str(d["alias"]),
            tags={int(k): np.asarray(v, dtype=np.float64).reshape(4, 2) for k, v in (d.get("tags") or {}).items()},
        )


@dataclass
class Sample:
    """One capture-button press: all cameras' detections + optional images."""

    index: int
    host_timestamp_ns: int
    views: dict[str, ViewDetections]
    image_paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "host_timestamp_ns": self.host_timestamp_ns,
            "views": {alias: v.to_json() for alias, v in self.views.items()},
            "image_paths": {alias: str(p.name) for alias, p in self.image_paths.items()},
            "metadata": self.metadata,
        }


@dataclass
class RecordingSession:
    """A directory-backed collection of samples for one calibration run."""

    session_dir: Path
    stage: str
    aliases: list[str]
    detector: AprilTagDetector
    recording_cfg: RecordingConfig
    samples: list[Sample] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        aliases: list[str],
        detector: AprilTagDetector,
        recording_cfg: RecordingConfig,
        session_dir: Path,
    ) -> "RecordingSession":
        (session_dir / "samples").mkdir(parents=True, exist_ok=True)
        session = cls(
            session_dir=session_dir,
            stage=stage,
            aliases=list(aliases),
            detector=detector,
            recording_cfg=recording_cfg,
        )
        session._write_manifest()
        return session

    # --- working/last lifecycle (mirrors Stage2SessionBundle) ---

    @classmethod
    def stage_root(cls, stage: str, root: Path | None = None) -> Path:
        return (root or DATA_DIR) / stage

    @classmethod
    def working_path(cls, stage: str, root: Path | None = None) -> Path:
        return cls.stage_root(stage, root) / WORKING_DIR_NAME

    @classmethod
    def last_path(cls, stage: str, root: Path | None = None) -> Path:
        return cls.stage_root(stage, root) / LAST_DIR_NAME

    @classmethod
    def last_exists(cls, stage: str, root: Path | None = None) -> bool:
        cls._migrate_legacy_to_last_if_needed(stage, root)
        return _session_has_data(cls.last_path(stage, root))

    @classmethod
    def last_count_label(cls, stage: str, root: Path | None = None) -> str:
        return f"last  ({_count_samples(cls.last_path(stage, root))} samples)"

    @classmethod
    def _migrate_legacy_to_last_if_needed(cls, stage: str, root: Path | None = None) -> None:
        """One-time: promote the most recent non-empty ``session_*`` folder to ``last/``.

        Recency is derived from the ``session_YYYYMMDD_HHMMSS`` name (sortable
        lexicographically) rather than filesystem mtimes, since a session's
        directory mtime does not necessarily bump when files are written into
        nested subdirectories on every platform.
        """
        last_p = cls.last_path(stage, root)
        if _session_has_data(last_p):
            return
        base = cls.stage_root(stage, root)
        if not base.is_dir():
            return
        candidates: list[Path] = []
        for p in base.iterdir():
            if not p.is_dir() or not p.name.startswith("session_"):
                continue
            if _count_samples(p) > 0:
                candidates.append(p)
        if not candidates:
            return
        candidates.sort(key=lambda p: p.name, reverse=True)
        _copy_session_tree(candidates[0], last_p)

    @classmethod
    def create_fresh_for_ui(
        cls,
        *,
        stage: str,
        aliases: list[str],
        detector: AprilTagDetector,
        recording_cfg: RecordingConfig,
        root: Path | None = None,
    ) -> "RecordingSession":
        """Empty working session — called on every app start."""
        cls.stage_root(stage, root).mkdir(parents=True, exist_ok=True)
        wp = cls.working_path(stage, root)
        _wipe_session_tree(wp)
        return cls.create(
            stage=stage,
            aliases=aliases,
            detector=detector,
            recording_cfg=recording_cfg,
            session_dir=wp,
        )

    @classmethod
    def load_last_into_working(
        cls,
        *,
        stage: str,
        aliases: list[str],
        detector: AprilTagDetector,
        recording_cfg: RecordingConfig,
        root: Path | None = None,
    ) -> "RecordingSession | None":
        """Copy ``last/`` → ``working/`` and open it. Returns None if no last data."""
        if not cls.last_exists(stage, root):
            return None
        wp = cls.working_path(stage, root)
        _copy_session_tree(cls.last_path(stage, root), wp)
        session = cls(
            session_dir=wp,
            stage=stage,
            aliases=list(aliases),
            detector=detector,
            recording_cfg=recording_cfg,
        )
        session.load_existing()
        return session

    @classmethod
    def archive_working_as_last(cls, stage: str, root: Path | None = None) -> None:
        """Overwrite ``last/`` with the current ``working/`` tree."""
        wp = cls.working_path(stage, root)
        if not _session_has_data(wp):
            return
        _copy_session_tree(wp, cls.last_path(stage, root))

    def _write_manifest(self) -> None:
        manifest = {
            "stage": self.stage,
            "aliases": self.aliases,
            "recording": {
                "save_images": bool(self.recording_cfg.save_images),
                "jpeg_quality": int(self.recording_cfg.jpeg_quality),
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def add_sample(
        self,
        images_bgr: dict[str, np.ndarray],
        host_timestamp_ns: int,
        metadata: dict | None = None,
    ) -> Sample:
        """Detect tags in every image and persist the sample."""
        idx = _next_sample_index(self.samples)
        sample_dir = self.session_dir / "samples" / f"{idx:06d}"
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
            # Always persist per-view detections in json for later re-runs.
            (sample_dir / f"{alias}_detections.json").write_text(json.dumps(views[alias].to_json()))
        sample = Sample(
            index=idx,
            host_timestamp_ns=int(host_timestamp_ns),
            views=views,
            image_paths=image_paths,
            metadata=dict(metadata or {}),
        )
        (sample_dir / "snapshot.json").write_text(json.dumps(sample.to_json(), indent=2))
        self.samples.append(sample)
        return sample

    def remove_last(self) -> Sample | None:
        if not self.samples:
            return None
        s = self.samples.pop()
        sample_dir = self.session_dir / "samples" / f"{s.index:06d}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        return s

    def clear(self) -> None:
        for s in list(self.samples):
            sample_dir = self.session_dir / "samples" / f"{s.index:06d}"
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
        self.samples.clear()

    def remove_sample_by_index(self, sample_index: int) -> Sample | None:
        """Delete one sample by its index; remaining samples keep their original numbers."""
        pos = next((i for i, s in enumerate(self.samples) if s.index == sample_index), None)
        if pos is None:
            return None
        removed = self.samples.pop(pos)
        sample_dir = self.session_dir / "samples" / f"{removed.index:06d}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        return removed

    def load_existing(self) -> None:
        """Repopulate `samples` from disk (used when reopening a session)."""
        samples_root = self.session_dir / "samples"
        if not samples_root.exists():
            return
        self.samples.clear()
        for entry in sorted(samples_root.iterdir()):
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
