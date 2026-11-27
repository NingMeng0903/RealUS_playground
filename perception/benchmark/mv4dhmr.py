"""Adapter for the official Att100/MV4DHMR implementation.

The reference implementation lives in ``ref_code_library/MV4DHMR`` and is
loaded lazily so importing this module never imports torch, downloads DINOv2,
or changes the RealUS runtime.  The adapter consumes the recorder's
``manifest.jsonl`` and per-frame ``metadata.json`` files, keeps the recorded
full K matrices, and supplies calibrated world-to-camera transforms to the
official multi-view refinement pipeline.

The measured latency is around one call to the reference ``Pipeline.step``:
all views enter the HMR frontend and the returned predictions go through the
reference cross-view matching, triangulation, refinement, and tracking path.
No single-view fallback is used.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence
import urllib.request

import numpy as np


REPO_URL = "https://github.com/Att100/MV4DHMR"
CHECKPOINT_BASE_URL = "https://download.europe.naverlabs.com/ComputerVision/MultiHMR"
GOOGLE_DRIVE_LARGE_672_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1NR8l55wFwRyppwZBewM0sDd4J_Tgq_rp&export=download&confirm=t"
)
# The original benchmark was recorded with four cameras.  Keep the exported
# ``CAMERA_COUNT`` name as the backwards-compatible default, while allowing a
# strict five-view recording from the optimized UE rig.  The official
# Pipeline is configured with the selected count below; no view is silently
# dropped unless the caller explicitly requests ``camera_names``.
DEFAULT_CAMERA_COUNT = 4
SUPPORTED_CAMERA_COUNTS: tuple[int, ...] = (4, 5)
CAMERA_COUNT = DEFAULT_CAMERA_COUNT
SMPLX_JOINT_COUNT = 55

# ``env.sh`` deliberately sets PYTHONNOUSERSITE=1 so the RealUS runtime does
# not accidentally inherit unrelated user packages.  The official MV4DHMR
# checkout, however, uses a few packages which are already installed in this
# machine's user site (notably top-level ``pyrender`` and ``smplx``) while the
# shared genesis environment does not contain them.  Keep the normal genesis
# site-packages first (especially its CUDA/Torch stack) and append only the
# explicitly selected supplemental site-packages directory when an official
# dependency is missing.  No package is installed or upgraded at runtime.
_OFFICIAL_OPTIONAL_SITE_ENV = "MV4DHMR_EXTRA_SITE_PACKAGES"
_OFFICIAL_TORCH_HOME_ENV = "MV4DHMR_TORCH_HOME"
_OFFICIAL_IMPORTS = ("pyrender", "smplx", "roma", "einops")


def _missing_official_imports() -> tuple[str, ...]:
    """Return official imports unavailable on the current ``sys.path``."""
    return tuple(name for name in _OFFICIAL_IMPORTS if importlib.util.find_spec(name) is None)


def _bootstrap_official_dependencies() -> dict[str, Any]:
    """Make the pinned reference checkout importable without mutating the env.

    The regular command is expected to work immediately after ``source
    env.sh``.  Since that environment opts out of Python's user site, use the
    user site as a *fallback* only for modules absent from the genesis env.  A
    caller may provide a reproducible wheel/site-packages directory via
    ``MV4DHMR_EXTRA_SITE_PACKAGES`` (colon-separated, like ``PYTHONPATH``).
    Paths are appended, never prepended, so environment-owned Torch and other
    packages retain precedence.
    """
    missing_before = _missing_official_imports()
    if not missing_before:
        return {
            "missing_before": [],
            "supplemental_paths": [],
            "missing_after": [],
        }

    candidates: list[Path] = []
    configured = os.environ.get(_OFFICIAL_OPTIONAL_SITE_ENV, "")
    for raw_path in configured.split(os.pathsep):
        if raw_path.strip():
            candidates.append(Path(raw_path).expanduser())
    try:
        user_site = Path(site.getusersitepackages()).expanduser()
    except (AttributeError, TypeError):
        user_site = None
    if user_site is not None:
        candidates.append(user_site)

    appended: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        # Keep genesis site-packages (and therefore its Torch/CUDA ABI) ahead
        # of the fallback path.  Avoid duplicate entries on repeated calls.
        if key not in {str(Path(item).resolve()) for item in sys.path if item}:
            sys.path.append(key)
            appended.append(key)
        if not _missing_official_imports():
            break

    missing_after = _missing_official_imports()
    if missing_after:
        configured_hint = (
            f" or set {_OFFICIAL_OPTIONAL_SITE_ENV}=/path/to/site-packages"
            if not configured
            else ""
        )
        raise RuntimeError(
            "MV4DHMR official dependencies are unavailable: "
            f"{', '.join(missing_after)}. The selected genesis environment has "
            "no compatible package and the user-site fallback did not provide it. "
            "Install the pinned reference requirements into a separate target "
            f"(without changing Torch){configured_hint}. "
            "The adapter never installs or upgrades dependencies automatically."
        )
    return {
        "missing_before": list(missing_before),
        "supplemental_paths": appended,
        "missing_after": [],
    }


def _resolve_torch_home() -> Path:
    """Return the persistent Torch Hub root used by the MV4 adapter.

    An explicit ``MV4DHMR_TORCH_HOME`` remains an escape hatch for a
    machine-local cache.  The generic ``TORCH_HOME`` is intentionally ignored:
    a stale shell setting from a previous experiment must not silently make a
    benchmark depend on ``/tmp``.  The default is inside the repository
    outputs tree so a shell that only sourced ``env.sh`` is reproducible.
    """
    configured = os.environ.get(_OFFICIAL_TORCH_HOME_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "outputs/cache/mv4dhmr_torch").resolve()


@contextlib.contextmanager
def _cached_dinov2_loader(torch_module: Any, torch_home: Path) -> Iterator[None]:
    """Use installed official Hub source without GitHub branch probing.

    Torch Hub otherwise makes a network request even with a complete cache.
    Scope the redirect to construction and restore the loader on failures.
    Other repositories retain Torch Hub's normal behavior.
    """
    cache = torch_home / "hub/facebookresearch_dinov2_main"
    if not (cache / "hubconf.py").is_file():
        yield
        return
    original_load = torch_module.hub.load

    def load(repo_or_dir: Any, *args: Any, **kwargs: Any) -> Any:
        if str(repo_or_dir) in ("facebookresearch/dinov2", "facebookresearch/dinov2:main"):
            kwargs["source"] = "local"
            return original_load(str(cache), *args, **kwargs)
        return original_load(repo_or_dir, *args, **kwargs)

    torch_module.hub.load = load
    try:
        yield
    finally:
        torch_module.hub.load = original_load

# This ordering matches projects.genesis_ue_sync.anatomy_retarget's canonical
# SMPL-X export and the 55-joint protocol consumed by the UE bridge.
SMPLX_JOINT_NAMES_55: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)

CHECKPOINTS: dict[str, dict[str, Any]] = {
    "small": {
        "filename": "multiHMR_672_S.pt",
        "checkpoint_subdir": "checkpoints/multiHMR",
        "checkpoint_url": f"{CHECKPOINT_BASE_URL}/multiHMR_672_S.pt",
        "checkpoint_source": "Naver Multi-HMR zero-shot",
        "eval_mode": "zeroshot",
        "image_size": 672,
        "backbone": "dinov2_vits14",
        "official_runtime_ms_v100": 61.12081336975098,
        "official_runtime_variant": "MV4DHMR README, 5 views",
    },
    "large": {
        "filename": "model_large_672_chi3d_sv.pt",
        "checkpoint_subdir": "checkpoints/saved",
        "checkpoint_url": GOOGLE_DRIVE_LARGE_672_URL,
        "checkpoint_source": "MV4DHMR Google Drive CHI3D fine-tuned Large-672",
        "eval_mode": "finetune",
        "image_size": 672,
        "backbone": "dinov2_vitl14",
        "official_runtime_ms_v100": 244.543958902359,
        "official_runtime_variant": "MV4DHMR README, Large-672 5-view reference; CHI3D fine-tuned checkpoint",
    },
    "large896": {
        "filename": "multiHMR_896_L.pt",
        "checkpoint_subdir": "checkpoints/multiHMR",
        "checkpoint_url": f"{CHECKPOINT_BASE_URL}/multiHMR_896_L.pt",
        "checkpoint_source": "Naver Multi-HMR zero-shot Large-896",
        "eval_mode": "zeroshot",
        "image_size": 896,
        "backbone": "dinov2_vitl14",
        "official_runtime_ms_v100": 244.543958902359,
        "official_runtime_variant": "MV4DHMR README, Large-672 5-view reference; selected checkpoint is official 896-L",
    },
}


@dataclasses.dataclass(frozen=True)
class CameraCalibration:
    """One static camera calibration in the reference pipeline convention."""

    name: str
    world_from_camera: np.ndarray

    def world_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        transform = np.asarray(self.world_from_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"camera {self.name!r} world_from_camera must be finite 4x4")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError(f"camera {self.name!r} has invalid homogeneous last row")
        r_wc = transform[:3, :3]
        p_wc = transform[:3, 3]
        # MV4DHMR's world2cam uses R @ p_world + t.  The scene stores the
        # opposite transform, camera pose in world coordinates.
        r_cw = r_wc.T
        t_cw = -r_cw @ p_wc
        return r_cw.astype(np.float32), t_cw.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class RecordingFrame:
    """A validated all-camera frame from the UE strict recorder."""

    record: Mapping[str, Any]
    image_paths: tuple[Path, ...]
    camera_names: tuple[str, ...]
    intrinsics: np.ndarray
    rotations_world_to_camera: np.ndarray
    translations_world_to_camera: np.ndarray

    @property
    def frame_index(self) -> int:
        return int(self.record.get("frame_index", 0))

    @property
    def motion_frame_index(self) -> int | None:
        value = self.record.get("motion_frame_index")
        return None if value is None else int(value)

    @property
    def motion_time_ns(self) -> int | None:
        value = self.record.get("motion_time_ns")
        return None if value is None else int(value)


@dataclasses.dataclass(frozen=True)
class Mv4dHmrResult:
    """One output row written by :func:`run_recording`."""

    frame_index: int
    motion_frame_index: int | None
    motion_time_ns: int | None
    latency_ms: float
    stage_timings_ms: Mapping[str, float]
    humans: tuple[dict[str, Any], ...]
    phase: str = "motion"
    # Keep the legacy four-view default for callers constructing result rows
    # directly; recording-backed rows always set this from the selected views.
    camera_count: int = DEFAULT_CAMERA_COUNT

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": 1,
            "method": "mv4dhmr",
            "status": "ok" if len(self.humans) == 1 else "rejected",
            "reason": None if len(self.humans) == 1 else (
                "no_human_detected" if not self.humans else "multiple_humans_not_supported"
            ),
            "frame_index": int(self.frame_index),
            "motion_frame_index": self.motion_frame_index,
            "motion_time_ns": self.motion_time_ns,
            "phase": self.phase,
            "camera_count": int(self.camera_count),
            "latency_ms": float(self.latency_ms),
            "latency_elapsed_s": float(self.latency_ms) / 1000.0,
            "stage_timings_ms": dict(self.stage_timings_ms),
            "latency_scope": (
                "official Pipeline.step: all-camera HMR frontend + cross-view "
                "matching/triangulation/refinement/tracking"
            ),
            "humans": list(self.humans),
        }
        # The benchmark evaluator consumes one common SMPL-X row.  Keep the
        # complete ``humans`` list for multi-person recordings, while exposing
        # the single-person fields at the row level used by the recorder/GT
        # protocol.  These are aliases of official MV4DHMR output, never GT.
        if len(self.humans) == 1:
            human = self.humans[0]
            pose = human.get("smplx_pose_axis_angle", [])
            row.update(
                {
                    "track_id": human.get("track_id"),
                    "joints_world_m": human.get("joints_world_m"),
                    "pelvis_world_m": human.get("pelvis_world_m"),
                    "root_translation_world_m": human.get("root_translation_world_m"),
                    "smplx_transl_world_m": human.get("smplx_transl_world_m"),
                    "easymocap_Th_world_m": human.get("smplx_transl_world_m"),
                    "betas": human.get("betas", []),
                    "beta": human.get("betas", []),
                    "shapes": human.get("betas", []),
                    "gender": human.get("gender", "neutral"),
                    "expression": human.get("expression", []),
                    "smplx_pose_axis_angle": pose,
                    "smplx_pose_aa_165": human.get("smplx_pose_aa_165", pose),
                    "poses165": pose,
                    "Rh": pose[:3],
                    # ``Th`` is the SMPL-X/EasyMocap model translation.  The
                    # world pelvis is exposed separately because it includes
                    # the model's local pelvis offset under the root pose.
                    "Th": human.get("smplx_transl_world_m"),
                    "smplx_joint_names": human.get("smplx_joint_names"),
                    "smplx_joint_count": human.get("smplx_joint_count"),
                    "native_joint_count": human.get("native_joint_count"),
                }
            )
            if "vertices_world_m" in human:
                row["vertices_world_m"] = human["vertices_world_m"]
            if "faces" in human:
                row["faces"] = human["faces"]
        return row


def _as_float_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite with shape {shape}, got {array.shape}")
    return array


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _validate_camera_count(value: int, *, label: str = "camera count") -> int:
    """Validate a supported MV4DHMR view count and return it as ``int``."""
    count = int(value)
    if count not in SUPPORTED_CAMERA_COUNTS:
        supported = ", ".join(str(item) for item in SUPPORTED_CAMERA_COUNTS)
        raise ValueError(f"{label} must be one of {supported}, found {count}")
    return count


def _normalize_camera_names(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    """Normalize and validate a manifest/CLI camera-name sequence."""
    names = tuple(str(item).strip() for item in value)
    if not names or any(not name for name in names):
        raise ValueError(f"{label} must contain non-empty camera names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate camera names: {names}")
    _validate_camera_count(len(names), label=f"{label} camera count")
    return names


def _camera_keys(value: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    """Return JSON camera-map keys while rejecting non-string collisions."""
    keys = tuple(str(item) for item in value.keys())
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicate camera names")
    return keys


def _require_camera_key_match(
    actual: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
    allow_extra: bool,
) -> None:
    """Fail early when a frame's camera metadata disagrees with its images."""
    actual_names = set(_camera_keys(actual, label=label))
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or (extra and not allow_extra):
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra and not allow_extra:
            detail.append(f"extra={extra}")
        expected_text = ", ".join(expected)
        raise ValueError(
            f"{label} camera metadata mismatch: expected [{expected_text}]; "
            + "; ".join(detail)
        )


def _recording_camera_names(
    recording_dir: Path,
    records: Sequence[Mapping[str, Any]],
    camera_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Infer four/five recording views, or select an explicit subset."""
    if not records:
        raise ValueError("cannot infer camera names from an empty recording manifest")
    manifest = recording_dir / "manifest.json"
    manifest_names: tuple[str, ...] | None = None
    if manifest.exists():
        value = _read_json(manifest).get("camera_names")
        if isinstance(value, list) and value:
            manifest_names = _normalize_camera_names(value, label=f"{manifest}")

    image_paths = records[0].get("image_paths", {})
    if not isinstance(image_paths, dict):
        raise ValueError("manifest records must contain an image_paths object")
    image_names = _normalize_camera_names(tuple(str(name) for name in image_paths), label="first frame image_paths")
    if manifest_names is not None:
        if set(manifest_names) != set(image_names):
            raise ValueError(
                f"manifest camera metadata mismatch: manifest.json has {manifest_names}, "
                f"but first frame image_paths has {image_names}"
            )
        source_names = manifest_names
    else:
        source_names = image_names

    if camera_names is None:
        return source_names

    selected = _normalize_camera_names(camera_names, label="camera_names override")
    missing = sorted(set(selected) - set(source_names))
    if missing:
        raise ValueError(
            f"camera_names override selects cameras absent from the recording: {missing}; "
            f"available={list(source_names)}"
        )
    return selected


def load_camera_calibration(path: Path, camera_names: Sequence[str]) -> dict[str, CameraCalibration]:
    """Load scene ``world_from_camera`` matrices without changing their values."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required to read the calibrated scene YAML") from exc
    try:
        scene = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read camera calibration {path}: {exc}") from exc
    entries = scene.get("cameras") if isinstance(scene, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"scene calibration {path} has no cameras list")
    by_name: dict[str, CameraCalibration] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        name = str(entry["name"])
        matrix = entry.get("world_from_camera")
        if matrix is None:
            continue
        by_name[name] = CameraCalibration(name=name, world_from_camera=np.asarray(matrix, dtype=np.float64))
    missing = [name for name in camera_names if name not in by_name]
    if missing:
        raise ValueError(f"scene calibration {path} is missing world_from_camera for {missing}")
    return {name: by_name[name] for name in camera_names}


def load_recording_frames(
    recording_dir: Path,
    *,
    scene_spec: Path,
    manifest_path: Path | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
    camera_names: Sequence[str] | None = None,
) -> list[RecordingFrame]:
    """Read and validate a strict four- or five-camera recording.

    With no override, the camera order and count come from ``manifest.json``
    (falling back to the first JSONL frame's image map).  ``camera_names`` may
    select a four-view subset from a five-view recording; all selected images,
    K matrices, and world transforms must still be present in every frame.
    """
    recording_dir = recording_dir.resolve()
    manifest_path = (manifest_path or recording_dir / "manifest.jsonl").resolve()
    try:
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read recording manifest {manifest_path}: {exc}") from exc
    records = [record for record in records if isinstance(record, dict) and record.get("status", "ok") == "ok"]
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    records = records[::frame_stride]
    if max_frames is not None:
        records = records[: int(max_frames)]
    if not records:
        raise ValueError(f"recording manifest has no usable frames: {manifest_path}")
    # Resolve the source contract separately from an optional selected subset
    # so a five-view manifest cannot hide a later frame whose camera map has
    # silently changed.  The subset only controls which valid source views are
    # sent to MV4DHMR.
    source_camera_names = _recording_camera_names(recording_dir, records)
    selected_camera_names = _recording_camera_names(recording_dir, records, camera_names)
    calibration = load_camera_calibration(scene_spec.resolve(), selected_camera_names)
    rotations = []
    translations = []
    for name in selected_camera_names:
        rotation, translation = calibration[name].world_to_camera()
        rotations.append(rotation)
        translations.append(translation)
    rotations_array = np.stack(rotations, axis=0)
    translations_array = np.stack(translations, axis=0)[:, None, :]

    frames: list[RecordingFrame] = []
    for record in records:
        # The strict recorder keeps source motion timing in canonical.json,
        # while manifest.jsonl primarily indexes images.  Copy only timing and
        # source-frame metadata into this local record; no pose, shape, or GT
        # values are read or passed to the official estimator.
        record = dict(record)
        canonical_ref = record.get("canonical_path")
        if canonical_ref:
            canonical_path = recording_dir / str(canonical_ref)
            if canonical_path.is_file():
                canonical = _read_json(canonical_path)
                canonical_human = canonical.get("human")
                if isinstance(canonical_human, dict):
                    for key in ("motion_frame_index", "motion_time_ns"):
                        if record.get(key) is None and canonical_human.get(key) is not None:
                            record[key] = canonical_human[key]
                if record.get("benchmark_phase") is None and canonical.get("benchmark_phase") is not None:
                    record["benchmark_phase"] = canonical["benchmark_phase"]
        image_map = record.get("image_paths")
        if not isinstance(image_map, dict):
            raise ValueError(f"frame {record.get('frame_index')} has no image_paths object")
        image_names = _camera_keys(image_map, label=f"frame {record.get('frame_index')} image_paths")
        _require_camera_key_match(
            image_map,
            source_camera_names,
            label=f"frame {record.get('frame_index')} image_paths",
            allow_extra=False,
        )
        metadata_ref = record.get("metadata_path")
        if not metadata_ref:
            raise ValueError(f"frame {record.get('frame_index')} has no metadata_path")
        metadata_path = recording_dir / str(metadata_ref)
        metadata = _read_json(metadata_path)
        by_camera = metadata.get("by_camera")
        if not isinstance(by_camera, dict):
            raise ValueError(f"metadata {metadata_path} has no by_camera object")
        if set(_camera_keys(by_camera, label=f"metadata {metadata_path}.by_camera")) != set(image_names):
            raise ValueError(
                f"metadata {metadata_path} camera metadata mismatch with image_paths: "
                f"metadata={list(_camera_keys(by_camera, label=f'metadata {metadata_path}.by_camera'))}, "
                f"images={list(image_names)}"
            )
        _require_camera_key_match(
            by_camera,
            selected_camera_names,
            label=f"metadata {metadata_path}.by_camera",
            allow_extra=camera_names is not None,
        )
        for metadata_name, metadata_value in by_camera.items():
            if not isinstance(metadata_value, dict):
                raise ValueError(f"metadata {metadata_path} camera {metadata_name!r} is not an object")
            declared_name = metadata_value.get("camera_name")
            if declared_name is not None and str(declared_name) != str(metadata_name):
                raise ValueError(
                    f"metadata {metadata_path} camera_name mismatch: key={metadata_name!r}, "
                    f"declared={declared_name!r}"
                )
        paths: list[Path] = []
        intrinsics: list[np.ndarray] = []
        for name in selected_camera_names:
            image_path = recording_dir / str(image_map.get(name, ""))
            if not image_path.is_file():
                raise FileNotFoundError(f"missing {name} image for frame {record.get('frame_index')}: {image_path}")
            camera_meta = by_camera.get(name)
            if not isinstance(camera_meta, dict):
                raise ValueError(f"metadata {metadata_path} missing {name}")
            intrinsics_meta = camera_meta.get("intrinsics", {})
            matrix = intrinsics_meta.get("K") if isinstance(intrinsics_meta, dict) else None
            if matrix is None:
                matrix = camera_meta.get("K")
            intrinsics.append(_as_float_array(matrix, shape=(3, 3), label=f"{name}.K"))
            paths.append(image_path.resolve())
        frames.append(
            RecordingFrame(
                record=record,
                image_paths=tuple(paths),
                camera_names=selected_camera_names,
                intrinsics=np.stack(intrinsics, axis=0),
                rotations_world_to_camera=rotations_array.copy(),
                translations_world_to_camera=translations_array.copy(),
            )
        )
    return frames


def _find_body_models(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/media/camp/EXT_DRIVE/Among_US/dataset/intermediate/humans/body_models"),
            Path("/media/camp/EXT_DRIVE/Among_US/ref_code_library/GVHMR/inputs/checkpoints/body_models"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "smpl" / "SMPL_NEUTRAL.pkl").is_file() and any(
            (candidate / "smplx" / filename).is_file() for filename in ("SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL.pkl")
        ):
            return candidate
    detail = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "MV4DHMR needs SMPL_NEUTRAL.pkl and SMPLX_NEUTRAL.npz/pkl under one body-model root; checked " + detail
    )


def _find_mean_params(explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/media/camp/EXT_DRIVE/Among_US/dataset/extra/smpl_assets/smpl_mean_params.npz"),
            Path("/media/camp/EXT_DRIVE/Among_US/ref_code_library/GVHMR/hmr4d/network/hmr2/configs/smpl_mean_params.npz"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("could not find official Multi-HMR smpl_mean_params.npz")


def _checkpoint_path(repo: Path, size: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    info = CHECKPOINTS[size]
    filename = str(info["filename"])
    subdir = Path(str(info["checkpoint_subdir"]))
    candidates = [
        repo / subdir / filename,
        Path.home() / "models" / "multiHMR" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    url = str(info["checkpoint_url"])
    raise FileNotFoundError(
        f"official {size} checkpoint is not present; download {url} into {candidates[0]} "
        "or pass --checkpoint"
    )


def _default_checkpoint_destination(repo: Path, size: str) -> Path:
    info = CHECKPOINTS[size]
    return (repo / str(info["checkpoint_subdir"]) / str(info["filename"])).resolve()


def download_official_checkpoint(repo: Path, size: str, destination: Path | None = None) -> Path:
    """Download an official MV4DHMR/Naver weight without fabricating a model."""
    info = CHECKPOINTS[size]
    destination = destination or repo / str(info["checkpoint_subdir"]) / str(info["filename"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = str(info["checkpoint_url"])
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@contextlib.contextmanager
def _official_runtime_context(
    repo: Path,
    body_models: Path,
    mean_params: Path,
    *,
    torch_home: Path | None = None,
) -> Iterator[Path]:
    """Expose official relative paths and a persistent Torch Hub cache."""
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_torch_home = os.environ.get("TORCH_HOME")
    selected_torch_home = (torch_home or _resolve_torch_home()).expanduser().resolve()
    selected_torch_home.mkdir(parents=True, exist_ok=True)
    # Keep this scoped to the official runner.  Other RealUS processes in the
    # same shell retain their own Torch cache setting after the context exits.
    os.environ["TORCH_HOME"] = str(selected_torch_home)
    with tempfile.TemporaryDirectory(prefix="mv4dhmr-runtime-") as temporary_name:
        temporary = Path(temporary_name)
        models = temporary / "models"
        models.mkdir()
        for directory in ("smpl", "smplx"):
            os.symlink(body_models / directory, models / directory, target_is_directory=True)
        os.symlink(mean_params, models / "smpl_mean_params.npz")
        repo_string = str(repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        os.chdir(temporary)
        try:
            yield models
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
            if old_torch_home is None:
                os.environ.pop("TORCH_HOME", None)
            else:
                os.environ["TORCH_HOME"] = old_torch_home


def _pipeline_args(
    *,
    size: str,
    checkpoint: Path,
    models_dir: Path,
    save_dir: Path,
    n_humans: int,
    image_size_org: tuple[int, int],
    camera_count: int = DEFAULT_CAMERA_COUNT,
) -> SimpleNamespace:
    info = CHECKPOINTS[size]
    camera_count = _validate_camera_count(camera_count, label="pipeline camera count")
    return SimpleNamespace(
        pretrained_path=str(checkpoint),
        checkpoint_path=str(checkpoint),
        smplx_dir=str(models_dir),
        smplx2smpl_path="",
        j_regressor_h36m_path="",
        save_dir=str(save_dir),
        backbone=str(info["backbone"]),
        smplx_type="smplx",
        image_size=int(info["image_size"]),
        image_size_org=f"{image_size_org[0]},{image_size_org[1]}",
        eval_mode=str(info["eval_mode"]),
        n_humans=int(n_humans),
        n_views=camera_count,
        data_smplx_type="smplx",
        recenter=False,
        smooth=False,
        auto_interpolate=False,
        vis=False,
        vis_mode="scatter",
        vid=False,
        save=False,
    )


def _first_vector(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return array.reshape(-1)


def _human_vector(value: Any, human_index: int, *, label: str, width: int) -> np.ndarray:
    """Select one human from a vector-shaped official output."""
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.ndim == 1:
        if array.size % width != 0:
            raise ValueError(f"official output {label} has incompatible shape {array.shape}")
        rows = array.reshape(-1, width)
    else:
        if array.shape[-1] != width:
            raise ValueError(f"official output {label} must end in {width}, got {array.shape}")
        rows = array.reshape(-1, width)
    if human_index >= rows.shape[0]:
        raise ValueError(f"official output {label} has no human index {human_index}")
    return rows[human_index].reshape(-1)


def _pack_common_smplx(
    params: Mapping[str, Any],
    human_index: int,
    *,
    joints_world_m: Any | None = None,
    vertices_world_m: Any | None = None,
    faces: Any | None = None,
) -> dict[str, Any]:
    """Map official refined keys to the repository's 55-joint SMPL-X order."""
    pose = np.zeros((SMPLX_JOINT_COUNT, 3), dtype=np.float32)
    pose[0] = _human_vector(params["global_orient"], human_index, label="global_orient", width=3).reshape(3)
    body = np.asarray(params["body_pose"], dtype=np.float32).reshape(-1, 3)
    if body.shape[0] >= (human_index + 1) * 21:
        body = body[human_index * 21 : (human_index + 1) * 21]
    pose[1 : 1 + min(21, len(body))] = body[:21]
    pose[22] = _human_vector(params["jaw_pose"], human_index, label="jaw_pose", width=3).reshape(3)
    if params.get("leye_pose") is not None:
        pose[23] = _human_vector(params["leye_pose"], human_index, label="leye_pose", width=3).reshape(3)
    if params.get("reye_pose") is not None:
        pose[24] = _human_vector(params["reye_pose"], human_index, label="reye_pose", width=3).reshape(3)
    left_hand = np.asarray(params["left_hand_pose"], dtype=np.float32).reshape(-1, 3)
    right_hand = np.asarray(params["right_hand_pose"], dtype=np.float32).reshape(-1, 3)
    if left_hand.shape[0] >= (human_index + 1) * 15:
        left_hand = left_hand[human_index * 15 : (human_index + 1) * 15]
    if right_hand.shape[0] >= (human_index + 1) * 15:
        right_hand = right_hand[human_index * 15 : (human_index + 1) * 15]
    pose[25 : 25 + min(15, len(left_hand))] = left_hand[:15]
    pose[40 : 40 + min(15, len(right_hand))] = right_hand[:15]
    transl = _first_vector(params["transl"], label="transl").reshape(-1, 3)[human_index]
    betas = _first_vector(params["betas"], label="betas").reshape(-1, 10)[human_index]
    expression_value = params.get("expression")
    expression = np.zeros((0,), dtype=np.float32) if expression_value is None else np.asarray(expression_value, dtype=np.float32)
    if expression.size:
        expression = expression.reshape(-1, expression.shape[-1] if expression.ndim > 1 else expression.size)[human_index]
    # Official ``postprocess_refined`` rewrites params["transl"] as the
    # SMPL-model translation needed to place its local pelvis joint at the
    # triangulated world pelvis.  It is therefore not itself the world pelvis
    # consumed by the common RealUS/UE protocol.  Prefer pred_j3d[0], which is
    # already reconstructed in world coordinates by the official refiner, and
    # retain the model translation separately for provenance.
    joints: np.ndarray | None = None
    if joints_world_m is not None:
        joints = np.asarray(joints_world_m, dtype=np.float32).reshape(-1, 3)
        if not np.all(np.isfinite(joints)):
            raise ValueError("official pred_j3d contains non-finite values")
        if joints.shape[0] < SMPLX_JOINT_COUNT:
            raise ValueError(f"official pred_j3d has only {joints.shape[0]} joints, need native SMPL-X 55")
    pelvis_world = transl if joints is None else joints[0]
    human: dict[str, Any] = {
        "root_translation_world_m": pelvis_world.astype(float).tolist(),
        "pelvis_world_m": pelvis_world.astype(float).tolist(),
        "smplx_transl_world_m": transl.astype(float).tolist(),
        "easymocap_Th_world_m": transl.astype(float).tolist(),
        "smplx_pose_axis_angle": pose.astype(float).reshape(-1).tolist(),
        "smplx_pose_aa_165": pose.astype(float).reshape(-1).tolist(),
        "smplx_joint_names": list(SMPLX_JOINT_NAMES_55),
        "smplx_joint_count": SMPLX_JOINT_COUNT,
        "betas": betas.astype(float).tolist(),
        "gender": "neutral",
        "expression": expression.astype(float).tolist(),
        "source_id": "mv4dhmr",
        "source_pivot": "smpl_pelvis_world",
    }
    if joints is not None:
        human["native_joint_count"] = int(joints.shape[0])
        human["joints_world_m"] = joints[:SMPLX_JOINT_COUNT].astype(float).tolist()
        human["joint_selection"] = "first_55_smplx_joints_from_official_pred_j3d"
    if vertices_world_m is not None:
        vertices = np.asarray(vertices_world_m, dtype=np.float32).reshape(-1, 3)
        if not np.all(np.isfinite(vertices)):
            raise ValueError("official pred_v3d contains non-finite values")
        human["vertices_world_m"] = vertices.astype(float).tolist()
    if faces is not None:
        triangles = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        if np.any(triangles < 0):
            raise ValueError("official SMPL-X faces contain negative indices")
        human["faces"] = triangles.tolist()
    return human


def _tracks_to_humans(
    tracks: Any,
    *,
    include_vertices: bool = False,
    faces: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    if tracks is None:
        return ()
    humans: list[dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, (tuple, list)) or len(track) != 2:
            continue
        joints_with_id, payload = track
        if not isinstance(payload, Mapping) or not isinstance(payload.get("smplx_params"), Mapping):
            continue
        joints = np.asarray(joints_with_id)
        if joints.ndim != 2 or joints.shape[1] < 1:
            continue
        track_id = int(round(float(joints[0, -1])))
        params = payload["smplx_params"]
        # Official refinement returns a batch with one entry per tracked human;
        # this payload is indexed in the same order as the tracker output.
        human_index = 0
        human = _pack_common_smplx(
            params,
            human_index,
            joints_world_m=payload.get("pred_j3d"),
            vertices_world_m=payload.get("pred_v3d") if include_vertices else None,
            faces=faces if include_vertices else None,
        )
        human["track_id"] = track_id
        humans.append(human)
    return tuple(humans)


class MV4DHMRRunner:
    """Own one official MV4DHMR pipeline and one all-view recording pass."""

    def __init__(
        self,
        *,
        official_repo: Path,
        size: str,
        checkpoint: Path,
        body_models: Path,
        mean_params: Path,
        image_size_org: tuple[int, int],
        n_humans: int = 1,
        device: str = "cuda:0",
        save_dir: Path | None = None,
        include_vertices: bool = False,
        camera_count: int = DEFAULT_CAMERA_COUNT,
    ) -> None:
        if size not in CHECKPOINTS:
            raise ValueError(f"size must be one of {sorted(CHECKPOINTS)}, got {size!r}")
        initialization_started = time.perf_counter()
        self.official_repo = official_repo.resolve()
        self.size = size
        self.device_name = str(device)
        self.checkpoint = checkpoint.resolve()
        self.include_vertices = bool(include_vertices)
        self.camera_count = _validate_camera_count(camera_count, label="runner camera count")
        self.cold_init_elapsed_s: float | None = None
        self.gpu_name: str | None = None
        self.official_dependency_info: dict[str, Any] = {}
        self.torch_home = _resolve_torch_home()
        self._runtime_context = _official_runtime_context(
            self.official_repo,
            body_models.resolve(),
            mean_params.resolve(),
            torch_home=self.torch_home,
        )
        models_dir = self._runtime_context.__enter__()
        self._closed = False
        try:
            # Do this before importing the official pipeline.  Its
            # ``utils`` package imports top-level ``pyrender`` eagerly.
            self.official_dependency_info = _bootstrap_official_dependencies()
            import torch
            from pipeline import Pipeline

            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    f"requested {self.device_name}, but torch.cuda.is_available() is false; "
                    "run in the configured GPU environment or pass --device cpu"
                )
            self.device = torch.device(self.device_name)
            if self.device.type == "cuda":
                self.gpu_name = str(torch.cuda.get_device_name(self.device))
            self.args = _pipeline_args(
                size=size,
                checkpoint=self.checkpoint,
                models_dir=models_dir,
                save_dir=(save_dir or Path(tempfile.mkdtemp(prefix="mv4dhmr-output-"))).resolve(),
                n_humans=n_humans,
                image_size_org=image_size_org,
                camera_count=self.camera_count,
            )
            Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
            with _cached_dinov2_loader(torch, self.torch_home):
                self.pipeline = Pipeline(self.args, self.device)
            self.mesh_faces = np.asarray(self.pipeline.smplx_model.faces, dtype=np.int32).reshape(-1, 3)
            self.mesh_vertex_count = int(self.pipeline.smplx_model.v_template.shape[0])
            # Pipeline.step calls visualisation unconditionally.  Suppress it
            # per instance so benchmark latency measures inference/fusion and
            # does not write matplotlib frames for every camera group.
            self.pipeline.visuaize_frame = lambda *args, **kwargs: None
            self.cold_init_elapsed_s = time.perf_counter() - initialization_started
        except Exception:
            self._runtime_context.__exit__(*sys.exc_info())
            self._closed = True
            raise

    def _synchronize(self) -> None:
        """Make wall-clock stage boundaries include queued CUDA work."""
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def infer(
        self, frame: RecordingFrame
    ) -> tuple[tuple[dict[str, Any], ...], float, dict[str, float]]:
        """Run exactly the official all-view pipeline with stage timings.

        This mirrors ``Pipeline.step`` from the pinned official checkout.  It
        is deliberately expanded here so the output records the wall time of
        preprocessing, the HMR frontend, cross-view refinement, tracking and
        the adapter packaging separately.  The visualizer remains disabled per
        instance in ``__init__``.
        """
        if self._closed:
            raise RuntimeError("MV4DHMRRunner is closed")
        if len(frame.camera_names) != self.camera_count:
            raise ValueError(
                f"runner is configured for {self.camera_count} cameras, "
                f"but frame {frame.frame_index} contains {len(frame.camera_names)}"
            )
        for label, value in (
            ("intrinsics", frame.intrinsics),
            ("rotations_world_to_camera", frame.rotations_world_to_camera),
            ("translations_world_to_camera", frame.translations_world_to_camera),
        ):
            if np.asarray(value).shape[0] != self.camera_count:
                raise ValueError(
                    f"frame {frame.frame_index} {label} camera count does not match "
                    f"{self.camera_count}: shape={np.asarray(value).shape}"
                )
        pipeline = self.pipeline
        started = time.perf_counter()
        stages: dict[str, float] = {}

        # Keep the counter update in the same position as Pipeline.step;
        # forward_tracking uses it for the tracker history.
        pipeline.counter += 1

        self._synchronize()
        stage_started = time.perf_counter()
        imgs, K, R, t = pipeline.preprocess_input(
            [str(path) for path in frame.image_paths],
            frame.intrinsics,
            frame.rotations_world_to_camera,
            frame.translations_world_to_camera,
        )
        self._synchronize()
        stages["preprocess_ms"] = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        output_hmr = pipeline.forward_hmr(imgs, K)
        self._synchronize()
        stages["frontend_hmr_ms"] = (time.perf_counter() - stage_started) * 1000.0
        if output_hmr is None:
            stages["cross_view_refine_ms"] = 0.0
            stages["tracking_ms"] = 0.0
            stages["adapter_pack_ms"] = 0.0
            return (), (time.perf_counter() - started) * 1000.0, stages

        stage_started = time.perf_counter()
        smplx_params, pred_v3d, pred_j3d, _labels_matched = pipeline.forward_refine(
            output_hmr, K, R, t
        )
        self._synchronize()
        stages["cross_view_refine_ms"] = (time.perf_counter() - stage_started) * 1000.0
        if smplx_params is None:
            stages["tracking_ms"] = 0.0
            stages["adapter_pack_ms"] = 0.0
            return (), (time.perf_counter() - started) * 1000.0, stages

        stage_started = time.perf_counter()
        tracks = pipeline.forward_tracking(smplx_params, pred_v3d, pred_j3d)
        self._synchronize()
        stages["tracking_ms"] = (time.perf_counter() - stage_started) * 1000.0

        # Pipeline.step invokes visualization after tracking.  The instance
        # method was replaced with a no-op in __init__, preserving the official
        # call order without creating images or adding plotting work.
        pipeline.visuaize_frame(pipeline.counter, imgs, tracks, K, R, t)
        stage_started = time.perf_counter()
        humans = _tracks_to_humans(
            tracks,
            include_vertices=self.include_vertices,
            faces=self.mesh_faces,
        )
        self._synchronize()
        stages["adapter_pack_ms"] = (time.perf_counter() - stage_started) * 1000.0
        return humans, (time.perf_counter() - started) * 1000.0, stages

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime_context.__exit__(None, None, None)

    def __enter__(self) -> "MV4DHMRRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def run_recording(
    frames: Sequence[RecordingFrame],
    runner: MV4DHMRRunner,
    *,
    warmup_frames: int = 0,
    post_warmup_stride: int = 1,
) -> tuple[list[Mv4dHmrResult], dict[str, Any]]:
    """Run every selected frame and return post-warmup outputs plus summary."""
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    if post_warmup_stride <= 0:
        raise ValueError("post_warmup_stride must be positive")
    observed_counts = {len(frame.camera_names) for frame in frames}
    if len(observed_counts) > 1:
        raise ValueError(f"recording frames disagree on camera count: {sorted(observed_counts)}")
    inferred_count = next(iter(observed_counts), DEFAULT_CAMERA_COUNT)
    camera_count = _validate_camera_count(
        int(getattr(runner, "camera_count", inferred_count)),
        label="recording camera count",
    )
    mismatched_frames = [
        frame.frame_index for frame in frames if len(frame.camera_names) != camera_count
    ]
    if mismatched_frames:
        raise ValueError(
            f"recording frames contain camera counts different from {camera_count}: "
            f"frame_indices={mismatched_frames[:8]}"
        )
    results: list[Mv4dHmrResult] = []
    warmup_latencies: list[float] = []
    all_latencies: list[float] = []
    measured_latencies: list[float] = []
    for index, frame in enumerate(frames):
        is_warmup = index < warmup_frames
        is_measurement = not is_warmup and (index - warmup_frames) % post_warmup_stride == 0
        if not is_warmup and not is_measurement:
            continue
        humans, latency_ms, stage_timings_ms = runner.infer(frame)
        all_latencies.append(float(latency_ms))
        if is_warmup:
            warmup_latencies.append(float(latency_ms))
            continue
        measured_latencies.append(float(latency_ms))
        results.append(
            Mv4dHmrResult(
                frame_index=frame.frame_index,
                motion_frame_index=frame.motion_frame_index,
                motion_time_ns=frame.motion_time_ns,
                latency_ms=latency_ms,
                stage_timings_ms=stage_timings_ms,
                humans=humans,
                camera_count=camera_count,
            )
        )
    measured = measured_latencies
    summary = {
        "schema_version": 1,
        "method": "mv4dhmr",
        "official_repository": REPO_URL,
        "size": runner.size,
        "checkpoint": str(runner.checkpoint),
        "checkpoint_source": CHECKPOINTS[runner.size]["checkpoint_source"],
        "checkpoint_eval_mode": CHECKPOINTS[runner.size]["eval_mode"],
        "mesh_vertex_count": int(getattr(runner, "mesh_vertex_count", 0)),
        "mesh_face_count": int(getattr(runner, "mesh_faces", np.empty((0, 3))).shape[0]),
        "device": runner.device_name,
        "gpu_name": runner.gpu_name,
        "torch_home": str(getattr(runner, "torch_home", _resolve_torch_home())),
        "official_dependency_info": dict(getattr(runner, "official_dependency_info", {})),
        "shape_policy": (
            "official MV4DHMR predicts betas per frame; warmup frames only warm the runtime/tracker "
            "and do not freeze shape"
        ),
        "camera_count": camera_count,
        "selected_frames": len(frames),
        "inference_frames": len(all_latencies),
        "warmup_frames": min(warmup_frames, len(frames)),
        "measured_frames": len(results),
        "post_warmup_frame_stride": int(post_warmup_stride),
        "cold_init_elapsed_s": runner.cold_init_elapsed_s,
        "latency_sample_note": (
            "Each retained sample still runs all recorded views through the official frontend and fusion; "
            "post_warmup_frame_stride only changes which source frames are measured."
        ),
        "latency_scope": results[0].as_dict()["latency_scope"] if results else "",
        "latency_ms": {
            "warmup_mean": float(np.mean(warmup_latencies)) if warmup_latencies else None,
            "measured_mean": float(np.mean(measured)) if measured else None,
            "measured_p50": float(np.percentile(measured, 50)) if measured else None,
            "measured_p95": float(np.percentile(measured, 95)) if measured else None,
            "measured_min": float(np.min(measured)) if measured else None,
            "measured_max": float(np.max(measured)) if measured else None,
        },
        "stage_timings_ms": {
            key: float(np.mean([result.stage_timings_ms[key] for result in results if key in result.stage_timings_ms]))
            for key in sorted({key for result in results for key in result.stage_timings_ms})
        },
        "official_reference_runtime_ms_v100": CHECKPOINTS[runner.size]["official_runtime_ms_v100"],
        "official_reference_runtime_note": (
            CHECKPOINTS[runner.size]["official_runtime_variant"]
            + f"; this pass uses {camera_count} recorded views."
        ),
        "humans_detected_total": int(sum(len(result.humans) for result in results)),
    }
    return results, summary


def _write_results(path: Path, results: Iterable[Mv4dHmrResult], summary: Mapping[str, Any], summary_path: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for result in results:
            output.write(json.dumps(result.as_dict(), separators=(",", ":")) + "\n")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(dict(summary), indent=2) + "\n")


def _write_mesh_npz(path: Path, results: Sequence[Mv4dHmrResult], faces: np.ndarray) -> None:
    """Write compact native mesh/joint arrays beside the human-readable JSONL."""
    valid = [
        result
        for result in results
        if len(result.humans) == 1
        and "vertices_world_m" in result.humans[0]
        and "joints_world_m" in result.humans[0]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        vertices = np.asarray([result.humans[0]["vertices_world_m"] for result in valid], dtype=np.float32)
        joints = np.asarray([result.humans[0]["joints_world_m"] for result in valid], dtype=np.float32)
        poses = np.asarray([result.humans[0]["smplx_pose_axis_angle"] for result in valid], dtype=np.float32)
        betas = np.asarray([result.humans[0]["betas"] for result in valid], dtype=np.float32)
        frame_indices = np.asarray([result.frame_index for result in valid], dtype=np.int64)
    else:
        vertices = np.empty((0, 0, 3), dtype=np.float32)
        joints = np.empty((0, SMPLX_JOINT_COUNT, 3), dtype=np.float32)
        poses = np.empty((0, SMPLX_JOINT_COUNT * 3), dtype=np.float32)
        betas = np.empty((0, 0), dtype=np.float32)
        frame_indices = np.empty((0,), dtype=np.int64)
    np.savez_compressed(
        path,
        frame_index=frame_indices,
        vertices_world_m=vertices,
        joints_world_m=joints,
        poses165=poses,
        betas=betas,
        faces=np.asarray(faces, dtype=np.int32).reshape(-1, 3),
    )


SMPLX_EDGES_55: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # SMPL-X collars branch from spine3 (9), independently of neck (12).
    # Keeping them under neck rotates both arm chains when the head turns.
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
    (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30),
    (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
    (20, 37), (37, 38), (38, 39),
    (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45),
    (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51),
    (21, 52), (52, 53), (53, 54),
)


def _project_world_points(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    camera = (np.asarray(rotation_world_to_camera, dtype=np.float64) @ points.T).T
    camera += np.asarray(translation_world_to_camera, dtype=np.float64).reshape(1, 3)
    depth = camera[:, 2]
    projected_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected_h[:, :2] / projected_h[:, 2:3]
    visible = np.isfinite(pixels).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
    return pixels, visible


def _load_gt_world_joints(recording_dir: Path, frame: RecordingFrame) -> np.ndarray | None:
    """Load only recorder bone locations for visualization, never estimator input."""
    gt_ref = frame.record.get("gt_path")
    if not gt_ref:
        return None
    gt_path = recording_dir / str(gt_ref)
    if not gt_path.is_file():
        return None
    gt = _read_json(gt_path)
    if gt.get("missing_joints"):
        return None
    bones = gt.get("bones")
    names = gt.get("joint_names")
    if not isinstance(bones, list) or not isinstance(names, list) or len(names) != SMPLX_JOINT_COUNT:
        return None
    by_name = {str(item.get("joint_name")): item for item in bones if isinstance(item, dict)}
    try:
        locations_cm = np.asarray([by_name[str(name)]["world_location_cm"] for name in names], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    # UE left-handed centimetres -> calibrated Genesis right-handed metres.
    return (locations_cm * np.asarray([0.01, -0.01, 0.01], dtype=np.float64)).astype(np.float32)


def _write_overlays(
    recording_dir: Path,
    frames: Sequence[RecordingFrame],
    results: Sequence[Mv4dHmrResult],
    output_dir: Path,
) -> int:
    """Save per-camera RGB/predicted-mesh/GT-skeleton contact sheets.

    This runs after inference and is excluded from latency.  A row contains
    original RGB, the translucent predicted native SMPL-X mesh, and the mesh
    with actual recorder bones overlaid.  Rejected rows remain visible with a
    rejection label and GT-only panel.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("--overlay-dir requires OpenCV in the selected Python environment") from exc

    frame_by_index = {frame.frame_index: frame for frame in frames}
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for result in results:
        frame = frame_by_index.get(result.frame_index)
        if frame is None:
            continue
        human = result.humans[0] if len(result.humans) == 1 else None
        vertices = None if human is None else human.get("vertices_world_m")
        faces = None if human is None else human.get("faces")
        gt_world = _load_gt_world_joints(recording_dir, frame)
        camera_rows = []
        for camera_index, image_path in enumerate(frame.image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            original = image.copy()
            mesh_panel = image.copy()
            if vertices is not None and faces is not None:
                projected, visible = _project_world_points(
                    np.asarray(vertices, dtype=np.float32),
                    frame.intrinsics[camera_index],
                    frame.rotations_world_to_camera[camera_index],
                    frame.translations_world_to_camera[camera_index],
                )
                overlay = mesh_panel.copy()
                for triangle in np.asarray(faces, dtype=np.int32).reshape(-1, 3):
                    if not visible[triangle].all():
                        continue
                    polygon = np.rint(projected[triangle]).astype(np.int32)
                    cv2.fillConvexPoly(overlay, polygon, (40, 180, 255), lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.20, mesh_panel, 0.80, 0.0, mesh_panel)
            else:
                cv2.putText(mesh_panel, "REJECTED: no native mesh", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            gt_panel = mesh_panel.copy()
            if gt_world is not None:
                gt_pixels, gt_visible = _project_world_points(
                    gt_world,
                    frame.intrinsics[camera_index],
                    frame.rotations_world_to_camera[camera_index],
                    frame.translations_world_to_camera[camera_index],
                )
                for start, end in SMPLX_EDGES_55:
                    if gt_visible[start] and gt_visible[end]:
                        cv2.line(
                            gt_panel,
                            tuple(np.rint(gt_pixels[start]).astype(int)),
                            tuple(np.rint(gt_pixels[end]).astype(int)),
                            (40, 40, 255), 3, cv2.LINE_AA,
                        )
                for point, visible_point in zip(gt_pixels, gt_visible):
                    if visible_point:
                        cv2.circle(gt_panel, tuple(np.rint(point).astype(int)), 4, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.putText(original, f"{frame.camera_names[camera_index]} RGB", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(mesh_panel, f"{frame.camera_names[camera_index]} MV4 mesh", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(gt_panel, f"{frame.camera_names[camera_index]} mesh + UE GT", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 255), 2, cv2.LINE_AA)
            target_size = (width // 2, height // 2)
            camera_rows.append(cv2.hconcat([cv2.resize(panel, target_size) for panel in (original, mesh_panel, gt_panel)]))
        if camera_rows:
            mosaic = cv2.vconcat(camera_rows)
            output_path = output_dir / f"frame_{result.frame_index:08d}.jpg"
            cv2.imwrite(str(output_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            written += 1
    return written


def _default_repo(root: Path) -> Path:
    return root / "ref_code_library" / "MV4DHMR"


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run official Att100/MV4DHMR on a four- or five-camera UE recording")
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--scene-spec", type=Path, default=root / "configs/scenes/realus_bed_rail_scene.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help=(
            "optional ordered subset of recording cameras (for example cam1 cam2 cam3 cam4 "
            "when selecting four views from a five-view manifest); default: infer all manifest views"
        ),
    )
    parser.add_argument("--official-repo", type=Path, default=_default_repo(root))
    parser.add_argument("--size", choices=tuple(CHECKPOINTS), default="small")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--body-models", type=Path, default=None)
    parser.add_argument("--mean-params", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--n-humans", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--motion-stride",
        type=int,
        default=1,
        help="Run every Nth frame after warmup (15 gives the requested 2 Hz motion samples at 30 fps).",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="optional compact native mesh/joint NPZ sidecar; requires --include-vertices",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="optional post-inference per-camera RGB/mesh/UE-GT contact sheets",
    )
    parser.add_argument(
        "--include-vertices",
        action="store_true",
        help="include official native vertices in each prediction row (large JSON output)",
    )
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate all inferred/selected views, K, and WFC without loading torch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    official_repo = args.official_repo.resolve()
    size = str(args.size)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if args.download_checkpoint:
        checkpoint = download_official_checkpoint(official_repo, size, checkpoint)
    elif checkpoint is None:
        if args.dry_run:
            checkpoint = _default_checkpoint_destination(official_repo, size)
        else:
            checkpoint = _checkpoint_path(official_repo, size, None)
    dry_run_dependency_info: dict[str, Any] | None = None
    if args.dry_run:
        # Validate the same lazy official imports used by a real run without
        # constructing Torch models or touching CUDA.
        dry_run_dependency_info = _bootstrap_official_dependencies()
    frames = load_recording_frames(
        args.recording_dir,
        scene_spec=args.scene_spec,
        manifest_path=args.manifest,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        camera_names=args.camera_names,
    )
    metadata_ref = frames[0].record.get("metadata_path")
    metadata = _read_json((args.recording_dir / str(metadata_ref)).resolve())
    first_camera = metadata["by_camera"][frames[0].camera_names[0]]
    width = int(first_camera["intrinsics"]["width"])
    height = int(first_camera["intrinsics"]["height"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "recording_dir": str(args.recording_dir.resolve()),
                    "frames": len(frames),
                    "camera_names": list(frames[0].camera_names),
                    "image_size": [width, height],
                    "checkpoint": str(checkpoint),
                    "checkpoint_present": checkpoint.is_file(),
                    "official_repo": str(official_repo),
                    "scene_spec": str(args.scene_spec.resolve()),
                    "torch_home": str(_resolve_torch_home()),
                    "official_dependency_info": dry_run_dependency_info,
                },
                indent=2,
            )
        )
        return 0
    body_models = _find_body_models(args.body_models)
    mean_params = _find_mean_params(args.mean_params)
    if args.output_npz is not None and not args.include_vertices:
        parser.error("--output-npz requires --include-vertices")
    if args.overlay_dir is not None and not args.include_vertices:
        parser.error("--overlay-dir requires --include-vertices")
    output_jsonl = args.output_jsonl or Path("outputs/realus_sync/tests/mv4dhmr") / f"{size}_predictions.jsonl"
    output_summary = args.output_summary or output_jsonl.with_suffix(".summary.json")
    output_npz = args.output_npz
    overlay_dir = args.overlay_dir
    mesh_faces = np.empty((0, 3), dtype=np.int32)
    with MV4DHMRRunner(
        official_repo=official_repo,
        size=size,
        checkpoint=checkpoint,
        body_models=body_models,
        mean_params=mean_params,
        image_size_org=(width, height),
        n_humans=int(args.n_humans),
        device=str(args.device),
        include_vertices=bool(args.include_vertices),
        camera_count=len(frames[0].camera_names),
    ) as runner:
        results, summary = run_recording(
            frames,
            runner,
            warmup_frames=int(args.warmup_frames),
            post_warmup_stride=int(args.motion_stride),
        )
        mesh_faces = runner.mesh_faces.copy()
    if output_npz is not None:
        summary["mesh_npz"] = str(output_npz)
    if overlay_dir is not None:
        summary["overlay_dir"] = str(overlay_dir)
    _write_results(output_jsonl, results, summary, output_summary)
    if output_npz is not None:
        _write_mesh_npz(output_npz, results, mesh_faces)
    if overlay_dir is not None:
        summary["overlay_frames_written"] = _write_overlays(
            args.recording_dir.resolve(), frames, results, overlay_dir.resolve()
        )
        if output_summary is not None:
            output_summary.write_text(json.dumps(dict(summary), indent=2) + "\n")
    print(json.dumps({"status": "ok", "output_jsonl": str(output_jsonl), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
