#!/usr/bin/env python3
"""Render synchronized UltraPoC comparison dashboards for the PPT.

The script is deliberately self contained at the output boundary.  It reads
the existing raw H5 recordings and the source camera files, then writes only
``data_*`` assets below the PPT staging directory.  Each card uses the run's
own elapsed source time.  The four cards therefore share a playback clock for
presentation, but are never presented as simultaneous physical trials.

The lower right profile is a replay visualization.  It uses the repository's
random-walk Welleweerd-v3 confidence solver, but computes a frame-wise robust
near-field ROI from confidence quantiles.  It is explicitly separate from the
controller/paper confidence values logged in JSONL; no logged metric is
recomputed or silently substituted.

Typical invocation (with the project's HDF5-capable environment)::

    python data_video_dashboards.py --family all

The default output root is the parent of this script's ``ppt`` directory, so
the copied script remains reproducible outside the source repository.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_DEFAULT = Path("/media/camp/EXT_DRIVE/RealUS_playground")
DATA_ROOT = Path("/media/camp/PEI_T7/icra 2027_contact")
COMPARISON_ROOT = DATA_ROOT / "Comparison Study"
FIGURE8_ROOT = DATA_ROOT / "figure8" / "005"
CAMERA_ROOT = DATA_ROOT / "video"
CONFIG_DEFAULT = REPO_DEFAULT / "peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    law: str
    family: str
    directory: Path
    h5_name: str = "raw.h5"
    scene_video: Path | None = None
    alias_of: str | None = None
    scene_mode: str = "pending"
    # Human scene mappings are supplied by the footage cut workflow.  The
    # fields are optional because the H5/B-mode/force dashboard is still
    # useful while that mapping is being prepared.
    scene_start_s: float | None = None
    scene_end_s: float | None = None
    scene_note: str = ""


PHANTOM_RUNS = (
    RunSpec(
        "ultrapoc_021", "UltraPoC | 021", "UltraPoC", "phantom",
        COMPARISON_ROOT / "ultrapoc" / "021",
        scene_video=CAMERA_ROOT / "C0176.MP4", scene_mode="epoch_exact",
        alias_of="ultrapoc_021",
        scene_note="C0176 mapped by H5 epoch timestamp and MP4 creation_time; 021 - 完美 is a byte-identical alias.",
    ),
    RunSpec(
        "ac2d_006", "FT-AC | AC2D 006", "FT-AC", "phantom",
        COMPARISON_ROOT / "ac2d" / "006",
        scene_video=CAMERA_ROOT / "C0180.MP4", scene_mode="epoch_exact",
        scene_note="C0180 mapped by H5 epoch timestamp and MP4 creation_time.",
    ),
    RunSpec(
        "fac_003", "F-AC | admittance 003", "F-AC", "phantom",
        COMPARISON_ROOT / "admittance_1d" / "003",
        scene_video=CAMERA_ROOT / "C0181.MP4", scene_mode="epoch_exact",
        scene_note="C0181 mapped by H5 epoch timestamp and MP4 creation_time.",
    ),
    RunSpec(
        "tafac_002", "TAF-AC | TAFAC 002", "TAF-AC", "phantom",
        COMPARISON_ROOT / "tafac" / "002",
        scene_video=CAMERA_ROOT / "C0182.MP4", scene_mode="epoch_exact",
        scene_note="C0182 mapped by H5 epoch timestamp and MP4 creation_time.",
    ),
)

HUMAN_RUNS = (
    RunSpec(
        "human_ultrapoc_005", "UltraPoC | human 005", "UltraPoC", "human",
        FIGURE8_ROOT / "ultrapoc",
        scene_video=CAMERA_ROOT / "C0221.MP4", scene_mode="mapping_required",
        scene_note="C0221 contains all four controller trials; exact cuts are read from metadata/scene_mapping.json when supplied.",
    ),
    RunSpec(
        "human_ac2d_005", "FT-AC | human 005", "FT-AC", "human",
        FIGURE8_ROOT / "ac2d",
        scene_video=CAMERA_ROOT / "C0221.MP4", scene_mode="mapping_required",
        scene_note="C0221 contains all four controller trials; exact cuts are read from metadata/scene_mapping.json when supplied.",
    ),
    RunSpec(
        "human_ac1d_005", "F-AC | human 005", "F-AC", "human",
        FIGURE8_ROOT / "ac1d",
        scene_video=CAMERA_ROOT / "C0221.MP4", scene_mode="mapping_required",
        scene_note="C0221 contains all four controller trials; exact cuts are read from metadata/scene_mapping.json when supplied.",
    ),
    RunSpec(
        "human_tafac_005", "TAF-AC | human 005", "TAF-AC", "human",
        FIGURE8_ROOT / "tafac",
        scene_video=CAMERA_ROOT / "C0221.MP4", scene_mode="mapping_required",
        scene_note="C0221 contains all four controller trials; exact cuts are read from metadata/scene_mapping.json when supplied.",
    ),
)


WIDTH = 1920
HEIGHT = 960
FPS = 15
PLAYBACK_SPEED = 2.0

# BGR colors: source PPT is a dark navy dashboard with teal scientific marks.
BG = (11, 16, 21)
CARD = (22, 33, 43)
PANEL = (15, 24, 31)
GRID = (58, 77, 87)
WHITE = (244, 248, 249)
MUTED = (164, 182, 186)
TEAL = (196, 213, 101)
TEAL_BRIGHT = (221, 231, 112)
CYAN = (225, 209, 93)
ORANGE = (85, 157, 240)
RED = (86, 86, 235)
PURPLE = (196, 117, 202)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _path_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _creation_epoch(path: Path) -> float | None:
    """Read MP4 creation_time without requiring an external ffprobe binary."""
    try:
        import av

        with av.open(str(path)) as container:
            raw = container.metadata.get("creation_time") or container.streams.video[0].metadata.get("creation_time")
    except Exception:
        raw = None
    if not raw:
        return None
    value = str(raw).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        stamp = _dt.datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_dt.timezone.utc)
        return stamp.timestamp()
    except ValueError:
        return None


def _import_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "h5py is required. Use the genesis environment or set "
            "PYTHONPATH to the local h5py dependency staging directory."
        ) from exc


def _load_feature_api(config_path: Path):
    # Import through the repository so the exact shipped implementation is
    # used; this file does not duplicate the confidence solver.
    repo = config_path.parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from peirastic.contact_qp.features import load_feature_config, random_walk_confidence

    return load_feature_config(str(config_path)), random_walk_confidence


def _motion_tracking_interval(directory: Path) -> tuple[float, float]:
    path = directory / "motion_trace.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    import csv

    mono: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "tracking":
                continue
            try:
                mono.append(float(row["t_mono_s"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not mono:
        raise ValueError(f"no tracking stage in {path}")
    return min(mono), max(mono)


def _scene_mapping_files(out_root: Path) -> list[Path]:
    metadata = out_root / "metadata"
    return [
        metadata / "scene_mapping.json",
        metadata / "data_scene_mapping.json",
        metadata / "footage_scene_mapping.json",
    ]


def _lookup_scene_mapping(out_root: Path, spec: RunSpec) -> dict[str, Any] | None:
    """Accept the footage worker's mapping with a few friendly nestings."""
    for path in _scene_mapping_files(out_root):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        candidates: list[Any] = [
            data.get(spec.key),
            (data.get(spec.family) or {}).get(spec.key) if isinstance(data.get(spec.family), dict) else None,
            (data.get(spec.family) or {}).get(spec.law) if isinstance(data.get(spec.family), dict) else None,
            (data.get("runs") or {}).get(spec.key) if isinstance(data.get("runs"), dict) else None,
        ]
        for item in candidates:
            if isinstance(item, dict):
                return dict(item, _mapping_source=str(path))
    return None


class SceneReader:
    """Sequential MP4 reader for monotonic output timestamps."""

    def __init__(self, path: Path, *, start_s: float = 0.0):
        self.path = path
        self.start_s = float(start_s)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open scene video {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.last_index = -1
        self.last_frame: np.ndarray | None = None

    def frame_at(self, source_elapsed_s: float, *, duration_s: float) -> np.ndarray | None:
        # If no exact mapping is available, the caller never constructs this
        # reader.  For representative clips this normalized mapping keeps the
        # camera motion visible while the manifest records it as such.
        target = self.start_s + max(0.0, float(source_elapsed_s))
        index = max(0, int(round(target * self.fps)))
        if self.count > 0:
            index = min(index, self.count - 1)
        if index < self.last_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.last_index = -1
            self.last_frame = None
        while self.last_index < index:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.last_index += 1
            self.last_frame = frame
        return self.last_frame

    def close(self) -> None:
        self.cap.release()


class RunData:
    def __init__(self, spec: RunSpec, out_root: Path, config_path: Path, random_walk: Any, feature_cfg: Any):
        self.spec = spec
        self.out_root = out_root
        self.h5py = _import_h5py()
        self.h5_path = spec.directory / spec.h5_name
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        self.h5 = self.h5py.File(self.h5_path, "r")
        self.random_walk = random_walk
        self.feature_cfg = feature_cfg
        self.config_path = config_path
        self.us_mono = np.asarray(self.h5["ultrasound/timestamp_mono_ns"], dtype=np.float64).ravel() * 1e-9
        self.us_epoch = np.asarray(self.h5["ultrasound/timestamp_ns"], dtype=np.float64).ravel() * 1e-9
        self.force_mono = np.asarray(self.h5["force/timestamp_mono_ns"], dtype=np.float64).ravel() * 1e-9
        self.force = np.asarray(self.h5["force/contact_force_n"], dtype=np.float64).ravel()
        if self.force.size != self.force_mono.size:
            wrench = np.asarray(self.h5["force/wrench_tcp"], dtype=np.float64)
            self.force = wrench[:, 2]
        self.epoch_offset_s = float(np.median(self.us_epoch - self.us_mono))
        self.t_start_mono, self.t_end_mono = self._interval()
        us_keep = (self.us_mono >= self.t_start_mono) & (self.us_mono <= self.t_end_mono)
        self.us_indices = np.flatnonzero(us_keep)
        if self.us_indices.size == 0:
            # Keep nearest valid image if an H5 capture ends a few ms before
            # the motion trace endpoint.
            self.us_indices = np.arange(len(self.us_mono), dtype=int)
        self.duration_s = float(self.t_end_mono - self.t_start_mono)
        self.trace_keep = (self.force_mono >= self.t_start_mono) & (self.force_mono <= self.t_end_mono)
        self.trace_t = self.force_mono[self.trace_keep] - self.t_start_mono
        self.trace_force = self.force[self.trace_keep]
        finite = np.isfinite(self.trace_force)
        self.trace_t = self.trace_t[finite]
        self.trace_force = self.trace_force[finite]
        self.profile_cache: dict[int, tuple[np.ndarray, int, int]] = {}
        self.scene_mapping = _lookup_scene_mapping(out_root, spec)
        self.scene_reader: SceneReader | None = None
        self.scene_start_s: float | None = None
        self.scene_exact = False
        self.scene_partial = False
        self.scene_status = "pending"
        self._init_scene()

    def _interval(self) -> tuple[float, float]:
        if self.spec.family == "phantom":
            return _motion_tracking_interval(self.spec.directory)
        session = _read_json(FIGURE8_ROOT / "session.json") or {}
        trial_name = self.spec.key.removeprefix("human_").removesuffix("_005")
        trial = next((x for x in session.get("trials", []) if x.get("name") == trial_name), None)
        if not trial:
            raise KeyError(f"figure8 trial missing for {self.spec.key}")
        episode = trial.get("episode") or {}
        if "start_time_ns" not in episode or "end_time_ns" not in episode:
            raise KeyError(f"figure8 episode has no timestamps for {self.spec.key}")
        start_epoch = float(episode["start_time_ns"]) * 1e-9
        end_epoch = float(episode["end_time_ns"]) * 1e-9
        return start_epoch - self.epoch_offset_s, end_epoch - self.epoch_offset_s

    def _init_scene(self) -> None:
        path = self.spec.scene_video
        mapping = self.scene_mapping
        if mapping:
            mapped_path = mapping.get("path") or mapping.get("video") or mapping.get("scene_video")
            if mapped_path:
                candidate = Path(str(mapped_path))
                if not candidate.is_absolute():
                    candidate = (self.out_root / candidate).resolve()
                path = candidate
            self.scene_exact = str(mapping.get("sync_mode", mapping.get("mode", ""))).lower() in {
                "exact", "epoch_exact", "h5_epoch", "source_exact"
            }
            start = mapping.get("video_start_s", mapping.get("source_start_s", mapping.get("start_s")))
            self.scene_start_s = float(start) if start is not None else 0.0
            self.scene_status = "mapped_exact" if self.scene_exact else "mapped_representative"
        elif self.spec.scene_mode == "epoch_exact" and path and path.exists():
            created = _creation_epoch(path)
            if created is not None:
                self.scene_start_s = float(self.t_start_mono + self.epoch_offset_s - created)
                self.scene_exact = True
                self.scene_status = "epoch_exact"
            else:
                self.scene_status = "missing_creation_time"
        elif self.spec.family == "phantom" and path and path.exists():
            self.scene_start_s = 0.0
            self.scene_status = "mapped_representative"
        else:
            self.scene_status = "mapping_pending"
        if path and path.exists() and self.scene_status in {"epoch_exact", "mapped_exact", "mapped_representative"}:
            try:
                self.scene_reader = SceneReader(path, start_s=self.scene_start_s or 0.0)
                # The source cameras stop before the H5 scan in these
                # comparison recordings.  Preserve the exact epoch mapping
                # for the overlap, but disclose the incomplete coverage in
                # both the card and manifest instead of implying a full scene
                # synchronization.
                if self.scene_exact and self.scene_reader.count > 0:
                    clip_duration = self.scene_reader.count / max(self.scene_reader.fps, 1e-9)
                    self.scene_partial = (self.scene_start_s or 0.0) + self.duration_s > clip_duration + 0.25
                    if self.scene_partial:
                        self.scene_status = "epoch_exact_partial"
            except RuntimeError:
                self.scene_reader = None
                self.scene_status = "scene_open_failed"

    def source_elapsed(self, presentation_elapsed_s: float) -> float:
        return float(np.clip(presentation_elapsed_s * PLAYBACK_SPEED, 0.0, self.duration_s))

    def _nearest_image_index(self, source_elapsed_s: float) -> int:
        target = self.t_start_mono + source_elapsed_s
        i = int(np.searchsorted(self.us_mono, target, side="left"))
        i = min(max(i, 0), len(self.us_mono) - 1)
        if i > 0 and abs(self.us_mono[i - 1] - target) < abs(self.us_mono[i] - target):
            i -= 1
        return i

    def _decode(self, index: int) -> np.ndarray:
        payload = np.asarray(self.h5["ultrasound/jpeg"][int(index)], dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"invalid JPEG frame {index} in {self.h5_path}")
        return image

    @staticmethod
    def _adaptive_roi(conf: np.ndarray) -> tuple[int, int]:
        """Return a frame-wise near-field band from profile quantiles.

        Only broad anti-degeneracy bounds are used.  The band itself comes
        from the current frame's confidence distribution, avoiding the fixed
        raw intensity/crop threshold that the slide is meant to explain.
        """
        row = np.nanmedian(conf, axis=1)
        row = np.asarray(row, dtype=np.float64)
        h = len(row)
        max_depth = max(12, int(round(h * 0.45)))
        q10, q85 = np.nanpercentile(row[:max_depth], [10.0, 85.0])
        target = q10 + 0.28 * (q85 - q10)
        # Ignore the seeded top row and choose the first sustained descent.
        smooth = np.convolve(row[:max_depth], np.ones(5) / 5.0, mode="same")
        candidates = np.flatnonzero(smooth[3:] <= target) + 3
        y1 = int(candidates[0]) if candidates.size else int(round(h * 0.26))
        y1 = int(np.clip(y1, max(8, int(round(h * 0.10))), int(round(h * 0.42))))
        y0 = 1 if h > 1 else 0
        if y1 <= y0 + 2:
            y1 = min(h, y0 + max(8, int(round(h * 0.12))))
        return y0, y1

    def profile(self, index: int) -> tuple[np.ndarray, int, int]:
        cached = self.profile_cache.get(int(index))
        if cached is not None:
            return cached
        image = self._decode(index)
        confidence = self.random_walk(image, self.feature_cfg)
        y0, y1 = self._adaptive_roi(confidence)
        columns = np.asarray(confidence[y0:y1], dtype=np.float64)
        n = 10
        values = np.empty(n, dtype=np.float32)
        for j, (left, right) in enumerate(zip(np.linspace(0, columns.shape[1], n, endpoint=False),
                                              np.linspace(0, columns.shape[1], n + 1)[1:])):
            a, b = int(round(left)), max(int(round(right)), int(round(left)) + 1)
            values[j] = float(np.nanquantile(columns[:, a:b], 0.25))
        result = (values, y0, y1)
        self.profile_cache[int(index)] = result
        return result

    def raw_frame(self, index: int) -> np.ndarray:
        return self._decode(index)

    def close(self) -> None:
        if self.scene_reader is not None:
            self.scene_reader.close()
        self.h5.close()

    def manifest(self) -> dict[str, Any]:
        meta = _read_json(self.spec.directory / "meta.json") or {}
        result = _read_json(self.spec.directory / "result.json") or {}
        created = _creation_epoch(self.spec.scene_video) if self.spec.scene_video else None
        item = {
            "key": self.spec.key,
            "label": self.spec.label,
            "law": self.spec.law,
            "family": self.spec.family,
            "directory": str(self.spec.directory),
            "raw_h5": str(self.h5_path),
            "source_meta": meta,
            "result": result,
            "h5_ultrasound_frames": int(len(self.us_mono)),
            "h5_force_samples": int(len(self.force_mono)),
            "h5_epoch_offset_s": self.epoch_offset_s,
            "segment_start_monotonic_s": self.t_start_mono,
            "segment_end_monotonic_s": self.t_end_mono,
            "segment_duration_s": self.duration_s,
            "scene_video": str(self.spec.scene_video) if self.spec.scene_video else None,
            "scene_status": self.scene_status,
            "scene_exact": bool(self.scene_exact),
            "scene_partial_coverage": bool(self.scene_partial),
            "scene_video_creation_epoch_s": created,
            "scene_video_start_s": self.scene_start_s,
            "scene_note": self.spec.scene_note,
            "alias_of": self.spec.alias_of,
            "logged_confidence_source": (
                str(self.spec.directory / "contact_qp.jsonl")
                if (self.spec.directory / "contact_qp.jsonl").exists()
                else None
            ),
            "visualization_confidence_is_logged_metric": False,
        }
        if self.scene_mapping:
            item["scene_mapping"] = self.scene_mapping
        return item


def _fit_image(image: np.ndarray, width: int, height: int, *, background: tuple[int, int, int] = PANEL) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ih, iw = image.shape[:2]
    scale = min(width / max(iw, 1), height / max(ih, 1))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _text(canvas: np.ndarray, value: str, x: int, y: int, size: float = 0.65,
          color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(canvas, str(value), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                float(size), color, int(thickness), cv2.LINE_AA)


def _wrapped_text(canvas: np.ndarray, value: str, x: int, y: int, width: int,
                  size: float = 0.52, color: tuple[int, int, int] = MUTED,
                  line_gap: int = 22) -> None:
    words = str(value).split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = (line + " " + word).strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    for i, text in enumerate(lines):
        _text(canvas, text, x, y + i * line_gap, size, color)


def _panel(canvas: np.ndarray, rect: tuple[int, int, int, int], *, fill=PANEL,
           border=GRID, radius: int = 8) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)


def _axis(canvas: np.ndarray, rect: tuple[int, int, int, int], *, x_max: float,
          y_min: float, y_max: float, y_label: str) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    left, right, top, bottom = x + 42, x + w - 12, y + 16, y + h - 28
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    for frac in np.linspace(0, 1, 5):
        xx = int(round(left + frac * (right - left)))
        cv2.line(canvas, (xx, top), (xx, bottom), GRID, 1, cv2.LINE_AA)
        label = f"{frac * x_max:.0f}"
        _text(canvas, label, xx - 12, bottom + 20, 0.42, MUTED)
    for frac in np.linspace(0, 1, 4):
        yy = int(round(bottom - frac * (bottom - top)))
        cv2.line(canvas, (left, yy), (right, yy), GRID, 1, cv2.LINE_AA)
        value = y_min + frac * (y_max - y_min)
        _text(canvas, f"{value:.1f}", x + 3, yy + 4, 0.40, MUTED)
    _text(canvas, y_label, x + 5, y + 15, 0.46, MUTED)
    return left, right, top, bottom


def _polyline(canvas: np.ndarray, x: np.ndarray, y: np.ndarray, bounds: tuple[int, int, int, int],
              x_max: float, y_min: float, y_max: float, color: tuple[int, int, int], thickness: int = 2) -> None:
    left, right, top, bottom = bounds
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return
    px = left + (np.clip(x[finite], 0, x_max) / max(x_max, 1e-9) * (right - left))
    py = bottom - ((np.clip(y[finite], y_min, y_max) - y_min) / max(y_max - y_min, 1e-9) * (bottom - top))
    points = np.column_stack([px, py]).astype(np.int32).reshape(-1, 1, 2)
    if len(points) > 1:
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)


def _draw_scene(canvas: np.ndarray, rect: tuple[int, int, int, int], run: RunData,
                source_elapsed_s: float) -> None:
    x, y, w, h = rect
    scene = None
    if run.scene_reader is not None:
        # Mappings marked representative still expose the source camera motion
        # but are labeled below so a viewer cannot mistake them for H5 sync.
        scene = run.scene_reader.frame_at(source_elapsed_s, duration_s=run.duration_s)
    if scene is None:
        _panel(canvas, rect)
        _text(canvas, "SCENE CLIP", x + 20, y + 40, 0.64, MUTED, 2)
        _text(canvas, "mapping pending", x + 20, y + 77, 0.58, WHITE)
        _wrapped_text(canvas, "B-mode + force traces remain directly synchronized to H5 timestamps.",
                      x + 20, y + 115, w - 40, 0.48, MUTED, 21)
        return
    crop = _fit_image(scene, w, h)
    canvas[y:y + h, x:x + w] = crop
    if run.scene_exact and run.scene_partial:
        status = "SCENE | H5/MP4 epoch sync (partial)"
    elif run.scene_exact:
        status = "SCENE | H5/MP4 epoch sync"
    else:
        status = "SCENE | representative timing"
    cv2.rectangle(canvas, (x + 8, y + 8), (x + 360, y + 36), (0, 0, 0), -1)
    _text(canvas, status, x + 16, y + 29, 0.46, TEAL_BRIGHT, 1)


def _draw_bmode(canvas: np.ndarray, rect: tuple[int, int, int, int], image: np.ndarray,
                y0: int, y1: int, source_elapsed_s: float) -> None:
    x, y, w, h = rect
    pane = _fit_image(image, w, h)
    # Draw ROI in image coordinates before fit would be more exact, but the
    # aspect-preserving fit makes this transform simple and avoids distortion.
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    ox, oy = (w - nw) // 2, (h - nh) // 2
    yy0 = y + oy + int(round(y0 * scale))
    yy1 = y + oy + int(round(y1 * scale))
    cv2.line(pane, (ox, int(round(y0 * scale))), (ox + nw, int(round(y0 * scale))), CYAN, 2)
    cv2.line(pane, (ox, int(round(y1 * scale))), (ox + nw, int(round(y1 * scale))), CYAN, 2)
    canvas[y:y + h, x:x + w] = pane
    cv2.rectangle(canvas, (x + 8, y + 8), (x + 210, y + 36), (0, 0, 0), -1)
    _text(canvas, "RAW B-MODE", x + 16, y + 29, 0.46, WHITE, 1)
    _text(canvas, f"t={source_elapsed_s:.1f}s", x + w - 112, y + 29, 0.45, WHITE, 1)
    # Use a second box for the dynamically identified ROI, keeping it clear
    # that this is a visualization cue and not a logged contact decision.
    _text(canvas, f"adaptive ROI {100*y0/ih:.0f}-{100*y1/ih:.0f}%", x + 10, y + h - 12, 0.43, CYAN, 1)


def _draw_force(canvas: np.ndarray, rect: tuple[int, int, int, int], run: RunData,
                source_elapsed_s: float) -> None:
    x, y, w, h = rect
    xmax = max(run.duration_s, 1e-6)
    bounds = _axis(canvas, rect, x_max=xmax, y_min=0.0, y_max=5.2, y_label="Fz [N]")
    _polyline(canvas, run.trace_t, run.trace_force, bounds, xmax, 0.0, 5.2, TEAL_BRIGHT, 2)
    left, right, top, bottom = bounds
    target_y = int(round(bottom - 4.0 / 5.2 * (bottom - top)))
    cv2.line(canvas, (left, target_y), (right, target_y), MUTED, 1, cv2.LINE_AA)
    cursor = int(round(left + np.clip(source_elapsed_s, 0, xmax) / xmax * (right - left)))
    cv2.line(canvas, (cursor, top), (cursor, bottom), CYAN, 2, cv2.LINE_AA)
    _text(canvas, "measured", x + w - 116, y + 18, 0.40, TEAL_BRIGHT)
    _text(canvas, "target 4 N", x + w - 116, y + 38, 0.40, MUTED)


def _draw_profile(canvas: np.ndarray, rect: tuple[int, int, int, int], profile: np.ndarray,
                  y0: int, y1: int, image_height: int) -> None:
    x, y, w, h = rect
    _panel(canvas, rect)
    _text(canvas, "REPLAY CONFIDENCE | VISUALIZATION ONLY", x + 16, y + 25, 0.45, TEAL_BRIGHT, 1)
    _text(canvas, "10 equal regions | q25 in adaptive near-field ROI", x + 16, y + 48, 0.40, MUTED)
    left, right, top, bottom = x + 34, x + w - 16, y + 68, y + h - 25
    for frac in (0.0, 0.5, 1.0):
        yy = int(round(bottom - frac * (bottom - top)))
        cv2.line(canvas, (left, yy), (right, yy), GRID, 1, cv2.LINE_AA)
        _text(canvas, f"{frac:.1f}", x + 5, yy + 4, 0.38, MUTED)
    n = len(profile)
    gap = max(2, int(round((right - left) * 0.012)))
    bw = max(3, int((right - left - gap * (n + 1)) / n))
    for i, value in enumerate(profile):
        value = float(np.clip(value, 0, 1))
        bx = left + gap + i * (bw + gap)
        by = int(round(bottom - value * (bottom - top)))
        cv2.rectangle(canvas, (bx, by), (bx + bw, bottom), TEAL, -1)
        _text(canvas, str(i + 1), bx + 2, bottom + 17, 0.36, MUTED)
    _text(canvas, f"ROI rows {y0}:{y1} / {image_height}", x + 16, y + h - 7, 0.38, CYAN)


def _run_scene_source_elapsed(run: RunData, source_elapsed_s: float) -> float:
    """Translate H5 elapsed time to mapped video elapsed time.

    Exact phantom mappings use H5 epoch versus MP4 creation_time.  Footage
    mappings can override the same field in scene_mapping.json.  A
    representative mapping is normalized to the available video duration in
    SceneReader only when explicitly supplied by the footage workflow.
    """
    mapping = run.scene_mapping or {}
    if run.scene_exact and mapping.get("video_start_s") is not None:
        return float(mapping["video_start_s"] + source_elapsed_s)
    return float((run.scene_start_s or 0.0) + source_elapsed_s)


def _draw_card(canvas: np.ndarray, rect: tuple[int, int, int, int], run: RunData,
               source_elapsed_s: float, frame_index: int, frame_cache: dict[tuple[str, int], tuple[np.ndarray, int, int]]) -> None:
    x, y, w, h = rect
    _panel(canvas, rect, fill=CARD, border=(62, 88, 93))
    header_h = 48
    _text(canvas, run.spec.label, x + 18, y + 31, 0.72, WHITE, 2)
    _text(canvas, f"elapsed source {source_elapsed_s:.1f}/{run.duration_s:.1f}s | 2x playback",
          x + w - 370, y + 29, 0.42, MUTED)
    inner_x, inner_y, inner_w, inner_h = x + 14, y + header_h, w - 28, h - header_h - 14
    upper_h = int(round(inner_h * 0.53))
    lower_h = inner_h - upper_h - 10
    scene_w = int(round(inner_w * 0.61))
    bmode_w = inner_w - scene_w - 10
    scene_rect = (inner_x, inner_y, scene_w, upper_h)
    bmode_rect = (inner_x + scene_w + 10, inner_y, bmode_w, upper_h)
    force_w = int(round(inner_w * 0.61))
    profile_w = inner_w - force_w - 10
    force_rect = (inner_x, inner_y + upper_h + 10, force_w, lower_h)
    profile_rect = (inner_x + force_w + 10, inner_y + upper_h + 10, profile_w, lower_h)
    _draw_scene(canvas, scene_rect, run, source_elapsed_s)
    key = (run.spec.key, int(frame_index))
    if key not in frame_cache:
        frame_cache[key] = run.profile(int(frame_index))
    profile, y0, y1 = frame_cache[key]
    image = run.raw_frame(int(frame_index))
    _draw_bmode(canvas, bmode_rect, image, y0, y1, source_elapsed_s)
    _draw_force(canvas, force_rect, run, source_elapsed_s)
    _draw_profile(canvas, profile_rect, profile, y0, y1, image.shape[0])


def _family_specs(family: str) -> tuple[RunSpec, ...]:
    if family == "phantom":
        return PHANTOM_RUNS
    if family == "human":
        return HUMAN_RUNS
    raise ValueError(family)


def _cache_path(out_root: Path, family: str) -> Path:
    return out_root / "cache" / f"data_{family}_comparison.npz"


def _video_path(out_root: Path, family: str) -> Path:
    return out_root / "videos" / f"data_{family}_comparison.mp4"


def _manifest_path(out_root: Path, family: str) -> Path:
    return out_root / "metadata" / f"data_{family}_comparison.json"


def _md_path(out_root: Path, family: str) -> Path:
    return out_root / "metadata" / f"data_{family}_comparison.md"


def _cover_path(out_root: Path, family: str) -> Path:
    return out_root / "images" / f"data_{family}_comparison_cover.png"


def _load_or_create_cache(out_root: Path, family: str, runs: list[RunData]) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    path = _cache_path(out_root, family)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as loaded:
                elapsed = np.asarray(loaded["presentation_elapsed_s"], dtype=np.float64)
                cached: dict[str, dict[str, np.ndarray]] = {}
                for run in runs:
                    prefix = run.spec.key
                    cached[prefix] = {
                        "source_elapsed_s": np.asarray(loaded[f"{prefix}/source_elapsed_s"], dtype=np.float64),
                        "frame_index": np.asarray(loaded[f"{prefix}/frame_index"], dtype=np.int64),
                        "profile": np.asarray(loaded[f"{prefix}/profile"], dtype=np.float32),
                        "roi": np.asarray(loaded[f"{prefix}/roi"], dtype=np.int16),
                    }
                return elapsed, cached
        except (KeyError, ValueError, OSError):
            pass
    duration = max(run.duration_s for run in runs) / PLAYBACK_SPEED
    count = int(math.ceil(duration * FPS)) + 1
    presentation_elapsed = np.arange(count, dtype=np.float64) / FPS
    cached: dict[str, dict[str, np.ndarray]] = {}
    for run in runs:
        source_elapsed = np.minimum(presentation_elapsed * PLAYBACK_SPEED, run.duration_s)
        frame_index = np.asarray([run._nearest_image_index(float(t)) for t in source_elapsed], dtype=np.int64)
        profiles = np.empty((count, 10), dtype=np.float32)
        roi = np.empty((count, 2), dtype=np.int16)
        # Deduplicate frozen/frequently repeated frame indices at 15 fps.
        for index in np.unique(frame_index):
            values, y0, y1 = run.profile(int(index))
            mask = frame_index == index
            profiles[mask] = values
            roi[mask] = (y0, y1)
        cached[run.spec.key] = {
            "source_elapsed_s": source_elapsed,
            "frame_index": frame_index,
            "profile": profiles,
            "roi": roi,
        }
    payload: dict[str, np.ndarray] = {"presentation_elapsed_s": presentation_elapsed}
    for run in runs:
        prefix = run.spec.key
        for name, values in cached[prefix].items():
            payload[f"{prefix}/{name}"] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return presentation_elapsed, cached


def _write_video(path: Path, frames: Any, *, fps: int = FPS) -> dict[str, Any]:
    """Encode RGB frames with libx264 through PyAV."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(fps, 1))
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "medium", "crf": "18", "profile": "high"}
    n = 0
    try:
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format="rgb24")
            video_frame.pts = n
            for packet in stream.encode(video_frame):
                container.mux(packet)
            n += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return {"frames": n, "fps": fps, "duration_s": n / fps, "codec": "H.264/libx264", "pix_fmt": "yuv420p"}


def _render_family(out_root: Path, family: str, repo: Path, config_path: Path) -> dict[str, Any]:
    feature_cfg, random_walk = _load_feature_api(config_path)
    runs = [RunData(spec, out_root, config_path, random_walk, feature_cfg) for spec in _family_specs(family)]
    presentation_elapsed, cache = _load_or_create_cache(out_root, family, runs)
    frame_cache: dict[tuple[str, int], tuple[np.ndarray, int, int]] = {}
    manifest_runs = [run.manifest() for run in runs]
    output_path = _video_path(out_root, family)
    cover_path = _cover_path(out_root, family)

    def frames():
        for i, ptime in enumerate(presentation_elapsed):
            frame = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
            _text(frame, f"{family.upper()} COMPARISON | SYNCHRONIZED DATA REPLAY", 28, 39, 0.82, WHITE, 2)
            _text(frame, "Each card follows its own elapsed scan time | raw H5 B-mode + force clocks preserved", 28, 65, 0.45, MUTED)
            _text(frame, f"presentation {ptime:.1f}s | playback 2x", WIDTH - 275, 39, 0.45, TEAL_BRIGHT)
            margin_x, margin_y, gap_x, gap_y = 24, 84, 18, 18
            card_w = (WIDTH - 2 * margin_x - gap_x) // 2
            card_h = (HEIGHT - margin_y - 22 - gap_y) // 2
            for j, run in enumerate(runs):
                row, col = divmod(j, 2)
                rect = (margin_x + col * (card_w + gap_x), margin_y + row * (card_h + gap_y), card_w, card_h)
                state = cache[run.spec.key]
                src = float(state["source_elapsed_s"][i])
                index = int(state["frame_index"][i])
                # Cached profiles avoid another solver call during encoding;
                # use the cached value directly while preserving the same card
                # drawing path for the raw image and force traces.
                frame_key = (run.spec.key, index)
                if frame_key not in frame_cache:
                    frame_cache[frame_key] = (state["profile"][i], int(state["roi"][i, 0]), int(state["roi"][i, 1]))
                _draw_card(frame, rect, run, src, index, frame_cache)
            yield frame

    video_info = _write_video(output_path, frames())
    # Cover is a plain PNG to make slide placement and review quick.
    # Re-rendering one frame is cheap because cache values are ready.
    first = next(iter(frames()))
    cv2.imwrite(str(cover_path), first, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    manifest = {
        "schema": "ultrapoc_ppt_data_video_v1",
        "family": family,
        "output_video": str(output_path),
        "cover_image": str(cover_path),
        "cache": str(_cache_path(out_root, family)),
        "output": {"width": WIDTH, "height": HEIGHT, **video_info, "playback_speed_x": PLAYBACK_SPEED},
        "alignment": {
            "basis": "elapsed scan time per run",
            "simultaneous_physical_time_claim": False,
            "ultrasound_clock": "raw.h5 ultrasound/timestamp_mono_ns",
            "force_clock": "raw.h5 force/timestamp_mono_ns",
            "force_signal": "raw.h5 force/contact_force_n (fallback force/wrench_tcp[:,2])",
            "cursor_policy": "each card advances at 2x its own source elapsed time and freezes after its segment ends",
            "phantom_tracking_interval": "motion_trace.csv stage == tracking",
            "human_tracking_interval": "figure8/005/session.json accepted episode start_time_ns/end_time_ns converted through H5 epoch offset",
        },
        "confidence": {
            "logged_metric_label": "controller/paper confidence (JSONL; kept separate, not plotted as replay output)",
            "visualization_metric_label": "Replay confidence · visualization only",
            "algorithm": getattr(feature_cfg, "algorithm_version", "unknown"),
            "config_path": str(config_path),
            "nearfield_roi": "frame-wise robust confidence quantiles with broad anti-degeneracy bounds; not fixed raw intensity/crop thresholds",
            "profile": "10 equal lateral regions, q25 over the adaptive ROI",
            "metrics_fabricated": False,
        },
        "runs": manifest_runs,
        "notes": [
            "The two UltraPoC directories 021 and 021 - 完美 have identical raw.h5 and contact_qp.jsonl SHA-256 values; 021 - 完美 is recorded as an alias, not a second trial.",
            "Scene clips are camera context. Phantom C0176/C0180/C0181/C0182 use H5 epoch timestamps and MP4 creation_time. Human C0221 requires an explicit scene_mapping.json cut map before exact scene attribution is claimed.",
        ],
    }
    _manifest_path(out_root, family).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    md = _markdown_manifest(manifest)
    _md_path(out_root, family).write_text(md, encoding="utf-8")
    for run in runs:
        run.close()
    return manifest


def _markdown_manifest(manifest: dict[str, Any]) -> str:
    out = manifest["output"]
    lines = [
        f"# Data video: {manifest['family']}",
        "",
        f"- Output: `{manifest['output_video']}`",
        f"- Video: {out['width']}×{out['height']} H.264, {out['fps']} fps, {out['duration_s']:.2f} s, {out['playback_speed_x']:.1f}× replay",
        f"- Alignment: {manifest['alignment']['basis']}; cards are not simultaneous physical trials.",
        f"- Confidence panel: **{manifest['confidence']['visualization_metric_label']}** using `{manifest['confidence']['algorithm']}` and a frame-wise adaptive ROI.",
        f"- Logged controller/paper confidence remains separate: `{manifest['confidence']['logged_metric_label']}`.",
        "",
        "## Runs",
        "",
        "| Card | Segment | Scene status | H5 B-mode frames | Force samples |",
        "|---|---:|---|---:|---:|",
    ]
    for item in manifest["runs"]:
        lines.append(
            f"| {item['label']} | {item['segment_duration_s']:.2f} s | {item['scene_status']} | "
            f"{item['h5_ultrasound_frames']} | {item['h5_force_samples']} |"
        )
    lines += [
        "",
        "The phantom scene mapping is derived from the camera MP4 `creation_time` and the H5 epoch/monotonic timestamp offset. The human C0221 camera file is retained as a source in the run metadata; exact per-method cuts are only used when a footage mapping file is present.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("phantom", "human", "all"), default="all")
    parser.add_argument("--out", type=Path, default=None, help="PPT staging root (default: this script's parent)")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    args = parser.parse_args()
    out_root = (args.out or Path(__file__).resolve().parents[1]).resolve()
    for name in ("scripts", "videos", "images", "metadata", "cache"):
        (out_root / name).mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    families = ("phantom", "human") if args.family == "all" else (args.family,)
    all_manifest = []
    for family in families:
        print(f"[data-video] rendering {family}", flush=True)
        manifest = _render_family(out_root, family, args.repo.resolve(), config_path)
        all_manifest.append(manifest)
        print(f"[data-video] wrote {manifest['output_video']} ({manifest['output']['duration_s']:.2f}s)", flush=True)
    index_path = out_root / "metadata" / "data_video_manifest.json"
    existing = _read_json(index_path) or {}
    by_family = {
        str(item.get("family")): item
        for item in existing.get("families", [])
        if isinstance(item, dict) and item.get("family")
    }
    by_family.update({str(item.get("family")): item for item in all_manifest})
    index_path.write_text(
        json.dumps({"schema": "ultrapoc_ppt_data_video_index_v1", "families": list(by_family.values())}, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
