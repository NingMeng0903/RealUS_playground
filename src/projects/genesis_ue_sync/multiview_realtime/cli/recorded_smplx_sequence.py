"""Offline SMPL-X fitting for strict UE benchmark recorder outputs.

The benchmark recorder writes one ``manifest.jsonl`` row per applied UE
capture group.  This module consumes those rows in file order and decodes the
four JPEGs from each group directly; it never connects to the live camera
PUB/SUB stream and therefore cannot skip frames because of realtime backlog.
"""

from __future__ import annotations

import json
import inspect
import logging
import math
import shutil
import time
from dataclasses import dataclass
from collections.abc import Mapping, Iterator
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.benchmark.ue_benchmark_recorder import validate_capture_metadata
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_metadata import (
    is_ue_source_profile,
    normalize_frame_for_source,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.multiview_realtime.ingress.undistort_burst import ensure_undistorted_burst
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle

LOG = logging.getLogger(__name__)


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_under(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _manifest_rows(recording_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Read recorder rows without reordering the manifest source sequence."""

    root = Path(recording_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"recording directory not found: {root}")
    manifest_jsonl = root / "manifest.jsonl"
    if not manifest_jsonl.is_file():
        summary_path = root / "manifest.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"recording has no manifest.jsonl/manifest.json: {root}")
        summary = _json_object(summary_path)
        raw_jsonl = summary.get("manifest_jsonl")
        if raw_jsonl:
            manifest_jsonl = _resolve_under(root, str(raw_jsonl))
        else:
            raw_rows = summary.get("records", summary.get("groups"))
            if not isinstance(raw_rows, list):
                raise ValueError(f"recording summary does not point to manifest rows: {summary_path}")
            return summary_path, [dict(row) for row in raw_rows if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    with manifest_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid recording manifest line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"recording manifest line {line_no} is not an object")
            if str(row.get("status", "ok")).lower() not in {"ok", "success", ""}:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"recording manifest has no successful groups: {manifest_jsonl}")
    return manifest_jsonl, rows


def _load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"recorded camera image not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _run_frontend_warmup(
    *,
    detector: Any,
    synced: SyncedMultiviewFrame,
    camera_ids: list[str],
) -> dict[str, Any]:
    """Run one real DWPose multiview forward without fitting or mutating tracking state.

    Reused HybrIK initialization skips the static EasyMocap burst, which would
    otherwise leave the detector's first motion frame paying the ONNX/TensorRT
    execution cost.  Keep this helper deliberately limited to detector output;
    no triangulation, GT, shape, or backend state is touched.
    """

    started = time.perf_counter()
    _annots, _det_meta, batch_meta = detector.infer_easymocap_annot_multiview(
        synced.views_rgb,
        list(camera_ids),
    )
    inference_mode = str(batch_meta.get("inference_mode", "")) if isinstance(batch_meta, dict) else ""
    result: dict[str, Any] = {
        "frontend_warmup_elapsed_s": float(time.perf_counter() - started),
        "frontend_warmup_camera_ids": list(camera_ids),
        "frontend_warmup_batched": bool(
            isinstance(batch_meta, dict)
            and (batch_meta.get("batched", False) or inference_mode.startswith("batched"))
        ),
    }
    if inference_mode:
        result["frontend_warmup_inference_mode"] = inference_mode
    providers = getattr(detector, "_execution_providers", ())
    if providers:
        result["frontend_warmup_execution_providers"] = [str(value) for value in providers]
    return result


def _motion_timing(canonical_state: dict[str, Any]) -> tuple[int | None, float | None]:
    """Return the motion-clock stamp and rate carried by the canonical human."""

    human = canonical_state.get("human")
    if not isinstance(human, dict):
        return None, None
    motion_time_ns: int | None = None
    raw_time = human.get("motion_time_ns")
    if raw_time is not None:
        try:
            motion_time_ns = int(raw_time)
        except (TypeError, ValueError):
            motion_time_ns = None
    motion_fps: float | None = None
    raw_fps = human.get("motion_fps")
    if raw_fps is not None:
        try:
            value = float(raw_fps)
            if np.isfinite(value) and value > 0.0:
                motion_fps = value
        except (TypeError, ValueError):
            motion_fps = None
    return motion_time_ns, motion_fps


def _benchmark_timing(canonical_state: dict[str, Any]) -> tuple[int | None, str | None]:
    """Return the benchmark replay clock and phase when recorder metadata has it."""

    human = canonical_state.get("human")
    if not isinstance(human, dict):
        return None, None
    benchmark_time_ns: int | None = None
    raw_time = human.get("benchmark_time_ns")
    if raw_time is not None:
        try:
            benchmark_time_ns = int(raw_time)
        except (TypeError, ValueError):
            benchmark_time_ns = None
    raw_phase = human.get("benchmark_phase")
    benchmark_phase = str(raw_phase) if raw_phase not in (None, "") else None
    return benchmark_time_ns, benchmark_phase


def _source_timestamp(
    record: dict[str, Any],
    metadata_by_camera: dict[str, dict[str, Any]],
    frame_index: int,
    canonical_state: dict[str, Any] | None = None,
) -> int:
    # The benchmark recorder's 30 Hz clock is the replay clock when present.
    # It is deliberately distinct from the raw AMASS motion clock and the
    # Genesis sim clock, both of which remain provenance fields below.
    benchmark_time_ns, _benchmark_phase = _benchmark_timing(canonical_state or {})
    if benchmark_time_ns is not None:
        return int(benchmark_time_ns)
    # AMASS motion time is the replay clock for older recordings. Genesis
    # sim_time_ns is kept as provenance, but can run at a different dt.
    motion_time_ns, _motion_fps = _motion_timing(canonical_state or {})
    if motion_time_ns is not None:
        return int(motion_time_ns)
    # Recorder state is authoritative, including a valid zero timestamp for
    # the first frame.  Older manifests may omit sim_time_ns, so fall back to
    # the validated per-camera source clock before using frame index.
    if "sim_time_ns" in record and record.get("sim_time_ns") is not None:
        try:
            return int(record.get("sim_time_ns") or 0)
        except (TypeError, ValueError):
            pass
    values: list[int] = []
    for metadata in metadata_by_camera.values():
        for key in ("source_time_ns", "sim_time_ns", "wall_time_ns"):
            try:
                value = int(metadata.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                values.append(value)
                break
    return int(min(values)) if values else int(frame_index)


def _canonical_context(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    raw_path = record.get("canonical_path")
    if not raw_path:
        return {}
    path = _resolve_under(root, str(raw_path))
    if not path.is_file():
        return {}
    try:
        return _json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        LOG.warning("ignoring unreadable canonical context %s", path)
        return {}


def _nested_human(state: dict[str, Any]) -> dict[str, Any]:
    human = state.get("human")
    return dict(human) if isinstance(human, dict) else {}


@dataclass(frozen=True)
class RecordedFrameGroup:
    """One manifest row decoded into the normal multiview ingress shape."""

    sequence_index: int
    frame_index: int
    timestamp_ns: int
    source_sim_time_ns: int
    motion_time_ns: int | None
    source_motion_fps: float | None
    benchmark_time_ns: int | None
    benchmark_phase: str | None
    source_time_domain: str
    loop_id: str
    loop_index: int
    capture_group_id: str
    motion_frame_index: int | None
    views_rgb: Mapping[str, np.ndarray]
    metadata_by_camera: dict[str, dict[str, Any]]
    canonical_state: dict[str, Any]
    record: dict[str, Any]

    def synced_frame(self) -> SyncedMultiviewFrame:
        return SyncedMultiviewFrame(
            frame_index=int(self.frame_index),
            views_rgb=dict(self.views_rgb),
            metadata_by_camera={key: dict(value) for key, value in self.metadata_by_camera.items()},
            timestamp_ns=int(self.timestamp_ns),
        )


class _RecordedRGBPaths(Mapping[str, np.ndarray]):
    """Decode only the current camera group, without retaining earlier RGB."""

    def __init__(self, paths: dict[str, Path]):
        self.paths = dict(paths)

    def __getitem__(self, camera_id: str) -> np.ndarray:
        return _load_image(self.paths[camera_id])

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


def load_recorded_frame_groups(
    recording_dir: Path | str,
    camera_ids: Iterable[str] = ("cam1", "cam2", "cam3", "cam4"),
    *, lazy_images: bool = False,
) -> tuple[Path, list[RecordedFrameGroup]]:
    """Decode manifest groups in recorder order, preserving source identity."""

    root = Path(recording_dir).expanduser().resolve()
    manifest_path, rows = _manifest_rows(root)
    ids = [str(camera_id) for camera_id in camera_ids]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("camera_ids must be unique and nonempty")
    groups: list[RecordedFrameGroup] = []
    for sequence_index, record in enumerate(rows):
        try:
            frame_index = int(record["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"recording row {sequence_index} has no valid frame_index") from exc
        group_id = str(record.get("capture_group_id") or "")
        if not group_id:
            raise ValueError(f"recording row {sequence_index} has no capture_group_id")

        metadata_path = record.get("metadata_path")
        if metadata_path:
            metadata_doc = _json_object(_resolve_under(root, str(metadata_path)))
        else:
            metadata_doc = _json_object(root / "frames" / group_id / "metadata.json")
        raw_by_camera = metadata_doc.get("by_camera", metadata_doc)
        if not isinstance(raw_by_camera, dict):
            raise ValueError(f"recording row {sequence_index} metadata has no by_camera object")
        metadata_by_camera: dict[str, dict[str, Any]] = {}
        for camera_id in ids:
            metadata = raw_by_camera.get(camera_id)
            if not isinstance(metadata, dict):
                raise ValueError(f"recording row {sequence_index} missing metadata for {camera_id}")
            errors = validate_capture_metadata(
                metadata,
                capture_group=group_id,
                frame_index=frame_index,
                session_id=str(record.get("session_id") or ""),
                loop_id=str(record.get("loop_id") or ""),
            )
            if errors:
                raise ValueError(
                    f"recording row {sequence_index} invalid metadata for {camera_id}: {','.join(errors)}"
                )
            metadata_by_camera[camera_id] = dict(metadata)

        raw_images = record.get("image_paths")
        image_paths = dict(raw_images) if isinstance(raw_images, dict) else {}
        image_root = _resolve_under(root, Path(str(metadata_path)).parent if metadata_path else Path("frames") / group_id)
        resolved_images: dict[str, Path] = {}
        for camera_id in ids:
            image_raw = image_paths.get(camera_id)
            image_path = _resolve_under(root, str(image_raw)) if image_raw else image_root / f"{camera_id}.jpg"
            if lazy_images and not image_path.is_file():
                raise FileNotFoundError(image_path)
            resolved_images[camera_id] = image_path
        views_rgb = (_RecordedRGBPaths(resolved_images) if lazy_images else
                     {camera_id: _load_image(path) for camera_id, path in resolved_images.items()})

        canonical_state = _canonical_context(root, record)
        human = _nested_human(canonical_state)
        motion_time_ns, source_motion_fps = _motion_timing(canonical_state)
        benchmark_time_ns, benchmark_phase = _benchmark_timing(canonical_state)
        raw_loop_id = record.get("loop_id") or human.get("loop_id") or metadata_by_camera[ids[0]].get("loop_id") or ""
        try:
            loop_index = int(human.get("loop_index", record.get("loop_index", 0)) or 0)
        except (TypeError, ValueError):
            loop_index = 0
        motion_raw = human.get(
            "motion_frame_index",
            record.get("motion_frame_index", metadata_by_camera[ids[0]].get("applied_motion_frame_index")),
        )
        try:
            motion_frame_index = int(motion_raw) if motion_raw is not None else None
        except (TypeError, ValueError):
            motion_frame_index = None
        groups.append(
            RecordedFrameGroup(
                sequence_index=int(sequence_index),
                frame_index=frame_index,
                timestamp_ns=_source_timestamp(
                    record,
                    metadata_by_camera,
                    frame_index,
                    canonical_state=canonical_state,
                ),
                source_sim_time_ns=int(record.get("sim_time_ns") or 0),
                motion_time_ns=motion_time_ns,
                source_motion_fps=source_motion_fps,
                benchmark_time_ns=benchmark_time_ns,
                benchmark_phase=benchmark_phase,
                source_time_domain=str(metadata_by_camera[ids[0]].get("source_time_domain") or ""),
                loop_id=str(raw_loop_id),
                loop_index=loop_index,
                capture_group_id=group_id,
                motion_frame_index=motion_frame_index,
                views_rgb=views_rgb,
                metadata_by_camera=metadata_by_camera,
                canonical_state=canonical_state,
                record=dict(record),
            )
        )
    return manifest_path, groups


def _first_fit_betas(moment_dir: Path) -> np.ndarray | None:
    path = Path(moment_dir) / "smplx_result.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "shapes" not in data:
            return None
        betas = np.asarray(data["shapes"], dtype=np.float32).reshape(-1)[:10]
    if betas.size != 10 or not np.all(np.isfinite(betas)):
        return None
    return betas


def _torch_runtime_info() -> dict[str, Any]:
    """Return serializable CUDA evidence without making torch mandatory."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised in minimal CPU envs
        return {
            "torch_available": False,
            "torch_import_error": str(exc),
            "cuda_available": False,
            "device_name": None,
        }
    available = bool(torch.cuda.is_available())
    name = None
    index = None
    if available:
        try:
            index = int(torch.cuda.current_device())
            name = str(torch.cuda.get_device_name(index))
        except Exception as exc:  # pragma: no cover - driver race/partial install
            name = f"cuda_device_query_failed: {exc}"
    return {
        "torch_available": True,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": available,
        "cuda_device_index": index,
        "device_name": name,
    }


def _model_device(model: Any) -> str | None:
    """Best-effort device string for the actual EasyMocap body model."""

    try:
        device = getattr(model, "device", None)
        if device is not None:
            return str(device)
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            return str(next(parameters()).device)
    except Exception:
        pass
    return None


def _source_fields(group: RecordedFrameGroup) -> dict[str, Any]:
    """Copy source identity/timing into an index row without losing raw clocks."""

    return {
        "recording_sequence_index": int(group.sequence_index),
        "capture_group_id": group.capture_group_id,
        "source_frame_index": int(group.frame_index),
        "source_timestamp_ns": int(group.timestamp_ns),
        "source_sim_time_ns": int(group.source_sim_time_ns),
        "motion_time_ns": group.motion_time_ns,
        "source_motion_fps": group.source_motion_fps,
        "benchmark_time_ns": group.benchmark_time_ns,
        "benchmark_phase": group.benchmark_phase,
        "source_time_domain": group.source_time_domain,
        "loop_id": group.loop_id,
        "loop_index": int(group.loop_index),
    }


def _preserve_triangulation_artifact(
    *,
    moment_dir: Path,
    run_dir: Path,
    moment_id: int,
) -> Path | None:
    """Keep only the compact 3D artifact after a rejected moment is removed."""

    candidates = [
        moment_dir / "triangulation.npz",
        moment_dir.parent / f"{moment_dir.name}.triangulation.npz",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return None
    target = run_dir / "triangulation" / f"moment_{int(moment_id):04d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(source), str(target))
    except OSError:
        return None
    return target


def apply_dwpose_temporal_bbox_fallback(
    cfg: MultiviewRealtimeConfig,
    override: bool | None,
) -> MultiviewRealtimeConfig:
    """Apply an explicit CLI override while preserving the YAML value by default."""

    if override is None:
        return cfg
    pose_backend = dict(cfg.pose_backend)
    dwpose = dict(pose_backend.get("dwpose") or {})
    dwpose["temporal_bbox_fallback"] = bool(override)
    pose_backend["dwpose"] = dwpose
    return MultiviewRealtimeConfig(
        calibration_path=cfg.calibration_path,
        scene_spec_path=cfg.scene_spec_path,
        camera_ids=cfg.camera_ids,
        primary_camera_id=cfg.primary_camera_id,
        ingress=cfg.ingress,
        pose_backend=pose_backend,
        world_reconstruction=cfg.world_reconstruction,
        robot_kinematic_mask=cfg.robot_kinematic_mask,
        genesis=cfg.genesis,
    )


def run_recorded_smplx_sequence(
    *,
    args: Any,
    cfg: MultiviewRealtimeConfig,
    mesh_rgb: tuple[int, int, int],
) -> int:
    """Fit saved recorder groups with optional static burst shape initialization.

    ``warmup_moments`` (15 by default for a 0.5 s / 30 Hz benchmark burst)
    is fitted once with shared shape optimization.  Its ten betas are frozen
    for sampled motion groups. ``max_moments`` limits motion groups after the
    warmup, while ``sample_stride`` selects them directly from the manifest;
    no realtime replay or interval sampling is involved.
    """

    cfg = apply_dwpose_temporal_bbox_fallback(
        cfg,
        getattr(args, "dwpose_temporal_bbox_fallback", None),
    )
    rgb_pose_weight = float(getattr(args, "hybrik_rgb_pose_weight", 0.0) or 0.0)
    if not math.isfinite(rgb_pose_weight) or rgb_pose_weight < 0.0:
        raise ValueError("--hybrik-rgb-pose-weight must be finite and nonnegative")
    observation_geometry_filter = bool(
        getattr(args, "hybrik_observation_geometry_filter", False)
    )
    occlusion_rotation_weight = float(
        getattr(args, "hybrik_occlusion_rotation_weight", 0.0) or 0.0
    )
    if not math.isfinite(occlusion_rotation_weight) or occlusion_rotation_weight < 0.0:
        raise ValueError("--hybrik-occlusion-rotation-weight must be finite and nonnegative")
    rgb_root_fallback = bool(getattr(args, "hybrik_rgb_root_fallback", False))
    recording_dir = Path(args.recording_dir).expanduser().resolve()
    manifest_path, all_groups = load_recorded_frame_groups(recording_dir, cfg.camera_ids, lazy_images=True)
    if not all_groups:
        raise ValueError(f"recording has no groups selected for fitting: {recording_dir}")

    max_moments = int(getattr(args, "max_moments", 0) or 0)
    if bool(getattr(args, "max_moments4", False)):
        max_moments = 4
    sample_stride = max(1, int(getattr(args, "sample_stride", 1) or 1))
    if bool(getattr(args, "sample_stride15", False)):
        sample_stride = 15
    requested_warmup = max(0, int(getattr(args, "warmup_moments", 15) or 0))

    out_root = project_paths(__file__).resolve_from_root(str(args.output_root))
    default_name = f"{recording_dir.name}_smplx"
    run_name = str(args.run_name or default_name)
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration_bundle(
        project_paths(__file__).resolve_from_root(cfg.calibration_path)
    )

    from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
        _jsonable,
        load_fixed_betas,
        process_burst,
        process_one_hybrik_moment,
        process_one_moment,
    )
    from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector

    betas_path = getattr(args, "betas_path", None)
    fixed_betas = load_fixed_betas(betas_path) if betas_path else None
    if betas_path and fixed_betas is None:
        raise ValueError(f"unable to load ten SMPL betas from {betas_path}")
    if fixed_betas is not None:
        np.save(run_dir / "sequence_betas.npy", np.asarray(fixed_betas, dtype=np.float32))
    tracking_backend = str(getattr(args, "tracking_backend", "easymocap") or "easymocap").strip().lower()
    if tracking_backend not in {"easymocap", "hybrik"}:
        raise ValueError(f"unsupported tracking backend: {tracking_backend}")
    initialization_npz_raw = getattr(args, "initialization_npz", None)
    initialization_npz: Path | None = None
    if initialization_npz_raw is not None:
        initialization_npz = project_paths(__file__).resolve_from_root(str(initialization_npz_raw))
        if not initialization_npz.is_file():
            raise FileNotFoundError(f"HybrIK initialization NPZ not found: {initialization_npz}")
    if tracking_backend == "hybrik" and initialization_npz is not None:
        with np.load(initialization_npz) as initialization_data:
            if "pelvis_world_m" not in initialization_data or "poses" not in initialization_data:
                raise ValueError(
                    "--initialization-npz must contain EasyMocap poses and measured pelvis_world_m"
                )
    write_debug_images = bool(getattr(args, "write_debug_images", True))
    triangulation_pair_hypotheses = int(
        dict(cfg.pose_backend.get("triangulation") or {}).get("simcc_pair_hypotheses", 5)
    )
    # A HybrIK run without an explicit seed owns the static EasyMocap burst;
    # an explicit seed is already the validated GPU warmup and can be reused
    # without paying the fit twice.
    hybrik_reuses_initialization = tracking_backend == "hybrik" and initialization_npz is not None
    hybrik_needs_warmup = tracking_backend == "hybrik" and not hybrik_reuses_initialization
    # Keep small legacy/unit recordings on the old per-frame path. Real
    # benchmark recordings have >=15 rows, so their first 0.5 s is always the
    # deterministic static initialization burst.
    warmup_count = 0
    if not bool(getattr(args, "skip_fit", False)):
        if requested_warmup > 0 and len(all_groups) >= requested_warmup:
            if hybrik_needs_warmup or hybrik_reuses_initialization:
                # With a reused validated seed, reserve/skip the same static
                # manifest burst so motion indexing starts at source 15 by
                # default.  ``--warmup-moments 0`` explicitly includes frame 0.
                warmup_count = requested_warmup
            elif fixed_betas is None:
                warmup_count = requested_warmup
    warmup_groups = all_groups[:warmup_count]
    motion_groups = all_groups[warmup_count:]
    if sample_stride > 1:
        motion_groups = motion_groups[::sample_stride]
    if max_moments > 0:
        motion_groups = motion_groups[:max_moments]
    groups = motion_groups
    if not groups and not warmup_groups:
        raise ValueError(f"recording has no groups selected for fitting: {recording_dir}")
    if tracking_backend == "hybrik":
        betas_mode = (
            "initialization_npz_reused" if hybrik_reuses_initialization
            else "warmup_burst_then_hybrik_frozen"
        )
    else:
        betas_mode = (
            "explicit" if fixed_betas is not None
            else "warmup_burst_then_frozen" if warmup_groups
            else "first_successful_fit"
        )

    scene_spec_path = (
        project_paths(__file__).resolve_from_root(str(args.scene_spec_path))
        if getattr(args, "scene_spec_path", None)
        else cfg.scene_spec_path
    )
    detector = DwposeOnnxDetector(DwposeOnnxConfig.from_dict(dict(cfg.pose_backend.get("dwpose") or {})))
    body_model_cache: dict[str, Any] = {}
    publisher = None
    if bool(getattr(args, "publish_genesis", False)):
        from projects.genesis_ue_sync.multiview_realtime.publish.static_smplx_track import SmplxTrackPublisher

        publisher = SmplxTrackPublisher(str(getattr(args, "publish_bind", "tcp://127.0.0.1:5598")))
        publisher.open()

    index: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_profile = str(cfg.ingress.source_profile or "ue").strip().lower()
    warmup_summary: dict[str, Any] | None = None
    warmup_valid = hybrik_reuses_initialization or not bool(warmup_groups)
    frontend_warmup_info: dict[str, Any] | None = None
    hybrik_backend: Any | None = None
    rgb_pose_prior: Any | None = None
    hybrik_info: dict[str, Any] = {}
    if hybrik_reuses_initialization:
        warmup_summary = {
            "fit_ok": True,
            "warmup_init_valid": True,
            "betas_frozen": True,
            "betas_source": "initialization_npz_reused",
            "initialization_npz": str(initialization_npz.resolve()),
            "source_groups": 0,
            "source_groups_skipped": len(warmup_groups),
            "warmup_mode": "initialization_npz_reused",
        }

    def _normalize_group(group: RecordedFrameGroup) -> tuple[SyncedMultiviewFrame, dict[str, Any]]:
        nonlocal calibration
        synced = group.synced_frame()
        if is_ue_source_profile(source_profile):
            synced, calibration, metadata_report = normalize_frame_for_source(
                synced,
                calibration,
                list(cfg.camera_ids),
                source_profile=source_profile,
            )
        else:
            synced = ensure_undistorted_burst([synced], list(cfg.camera_ids))[0]
            metadata_report = {"source_profile": source_profile, "cameras": {}}
        return synced, metadata_report

    triangulator = None
    triangulation_parallel = None
    try:
        parallel_workers = int(getattr(args, "triangulation_workers", 0) or 0)
        if tracking_backend == "hybrik" and parallel_workers:
            from projects.genesis_ue_sync.multiview_realtime.triangulation.parallel import BodyHandTriangulator
            triangulator = BodyHandTriangulator(parallel_workers)
            parallel_started = time.perf_counter()
            triangulator.preload()
            triangulation_parallel = {"workers": parallel_workers, "start_method": "spawn",
                                      "preload_elapsed_s": float(time.perf_counter() - parallel_started)}
        detector.preload()
        if warmup_groups and hybrik_reuses_initialization:
            # The static groups are reserved for source-clock alignment even
            # when their validated NPZ is reused. Run one RGB group through the
            # real DWPose frontend so the first motion frame does not pay the
            # ONNX/TensorRT execution warmup. This intentionally skips DLT,
            # GT, EasyMocap fitting, and all HybrIK state mutation.
            try:
                warmup_synced, _warmup_report = _normalize_group(warmup_groups[0])
                frontend_warmup_info = _run_frontend_warmup(
                    detector=detector,
                    synced=warmup_synced,
                    camera_ids=list(cfg.camera_ids),
                )
                frontend_warmup_info.update(
                    {
                        "frontend_warmup_valid": True,
                        "frontend_warmup_source_sequence_index": int(warmup_groups[0].sequence_index),
                        "frontend_warmup_source_frame_index": int(warmup_groups[0].frame_index),
                    }
                )
                warmup_summary["frontend_warmup"] = frontend_warmup_info
                warmup_summary["frontend_warmup_valid"] = True
            except Exception as exc:
                LOG.exception("reused-init DWPose frontend warmup failed: %s", exc)
                frontend_warmup_info = {
                    "frontend_warmup_valid": False,
                    "failure": "frontend_warmup_exception",
                    "error": str(exc),
                    "frontend_warmup_source_sequence_index": int(warmup_groups[0].sequence_index),
                    "frontend_warmup_source_frame_index": int(warmup_groups[0].frame_index),
                }
                warmup_summary.update(frontend_warmup_info)
                warmup_valid = False
                groups = []
        if warmup_groups and not hybrik_reuses_initialization:
            warmup_dir = run_dir / "warmup_init"
            try:
                warmup_synced: list[SyncedMultiviewFrame] = []
                warmup_reports: list[dict[str, Any]] = []
                for group in warmup_groups:
                    synced, report = _normalize_group(group)
                    warmup_synced.append(synced)
                    warmup_reports.append(report)
                warmup_summary = process_burst(
                    moment_dir=warmup_dir,
                    synced_frames=warmup_synced,
                    cfg=cfg,
                    calibration=calibration,
                    detector=detector,
                    camera_ids=list(cfg.camera_ids),
                    gender=str(args.gender),
                    fit_model=str(args.fit_model),
                    thres2d=float(args.thres2d),
                    max_repro_error=float(args.max_repro_error),
                    body_model_cache=body_model_cache,
                    fixed_betas=(fixed_betas if tracking_backend == "hybrik" else None),
                    bed_sdf=bool(args.bed_sdf),
                    scene_spec_path=scene_spec_path,
                    motion_frame_indices=[group.motion_frame_index for group in warmup_groups],
                    write_debug_images=bool(getattr(args, "write_debug_images", True)),
                    disable_face_iteration=bool(getattr(args, "disable_face_iteration", False)),
                )
                warmup_summary["fit_device"] = _model_device(body_model_cache.get("model"))
                warmup_summary["torch_runtime"] = _torch_runtime_info()
                warmup_summary.update(
                    {
                        "source_groups": len(warmup_groups),
                        "source_sequence_indices": [int(group.sequence_index) for group in warmup_groups],
                        "metadata_reports": warmup_reports,
                        "source_start": _source_fields(warmup_groups[0]),
                        "source_end": _source_fields(warmup_groups[-1]),
                    }
                )
                inferred = _first_fit_betas(warmup_dir)
                if bool(warmup_summary.get("fit_ok")) and inferred is not None:
                    fixed_betas = inferred
                    np.save(run_dir / "sequence_betas.npy", inferred)
                    warmup_summary["betas_frozen"] = True
                    warmup_summary["betas_source"] = "warmup_init"
                    warmup_summary["warmup_init_valid"] = True
                    warmup_valid = True
                else:
                    warmup_summary["betas_frozen"] = False
                    warmup_summary["betas_source"] = None
                    warmup_summary["warmup_init_valid"] = False
                    warmup_summary["failure"] = (
                        "warmup_fit_gate_failed" if not bool(warmup_summary.get("fit_ok"))
                        else "warmup_missing_finite_betas"
                    )
            except Exception as exc:
                LOG.exception("warmup burst failed: %s", exc)
                warmup_summary = {
                    "fit_ok": False,
                    "error": str(exc),
                    "source_groups": len(warmup_groups),
                    "warmup_dir": str(warmup_dir.resolve()),
                    "warmup_init_valid": False,
                    "failure": "warmup_exception",
                }
            if not warmup_valid:
                # A benchmark run must never silently switch from the declared
                # static shared-shape initialization to per-frame shape fits.
                # Leave the diagnostic warmup directory and index.json, then
                # stop before processing motion groups.
                groups = []

        if tracking_backend == "hybrik" and warmup_valid:
            # Release EasyMocap's CUDA model before constructing the official
            # SMPL-X layer.  The static NPZ remains the sole seed for shape,
            # root and fixed face/twist state.
            init_path = initialization_npz or (run_dir / "warmup_init" / "smplx_result.npz")
            if not init_path.is_file():
                warmup_valid = False
                groups = []
                warmup_summary = dict(warmup_summary or {})
                warmup_summary.update(
                    {
                        "fit_ok": False,
                        "warmup_init_valid": False,
                        "failure": "hybrik_initialization_npz_missing",
                        "initialization_npz": str(init_path.resolve()),
                    }
                )
            else:
                try:
                    from perception.benchmark.hybrik_backend import InitializedHybrIKX

                    body_model_cache.clear()
                    runtime_before = _torch_runtime_info()
                    try:
                        import torch

                        if bool(runtime_before.get("cuda_available")):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    backend_started = time.perf_counter()
                    refine_steps = max(0, int(getattr(args, "hybrik_refine_steps", 0) or 0))
                    backend_kwargs: dict[str, Any] = {
                        "device": str(getattr(args, "hybrik_device", "cuda")),
                        "gender": str(args.gender),
                    }
                    if "refine_steps" in inspect.signature(InitializedHybrIKX).parameters:
                        backend_kwargs["refine_steps"] = refine_steps
                    elif refine_steps:
                        raise TypeError(
                            "Installed InitializedHybrIKX does not support --hybrik-refine-steps"
                        )
                    pose_initialization = str(getattr(args, "hybrik_pose_initialization", "hybrik"))
                    temporal_weight = float(getattr(args, "hybrik_temporal_weight", 0.0))
                    if pose_initialization != "hybrik" or temporal_weight != 0:
                        backend_kwargs.update(pose_initialization=pose_initialization,
                                              temporal_weight=temporal_weight)
                    stationary_mm = getattr(args, "hybrik_stationary_tolerance_mm", None)
                    if stationary_mm is not None:
                        backend_kwargs["stationary_tolerance_m"] = float(stationary_mm) / 1000.0
                    cutoff = getattr(args, "hybrik_observation_min_cutoff_hz", None)
                    beta = getattr(args, "hybrik_observation_speed_beta", None)
                    if cutoff is not None:
                        backend_kwargs["observation_min_cutoff_hz"] = float(cutoff)
                    if beta is not None:
                        backend_kwargs["observation_speed_beta"] = float(beta)
                    if rgb_pose_weight > 0.0:
                        backend_kwargs["rgb_pose_weight"] = rgb_pose_weight
                    if rgb_root_fallback:
                        backend_kwargs["rgb_root_fallback"] = True
                    # Preserve the legacy backend constructor call when both
                    # optional observation policies are disabled.
                    if observation_geometry_filter:
                        backend_kwargs["observation_geometry_filter"] = True
                    if occlusion_rotation_weight > 0.0:
                        backend_kwargs["occlusion_rotation_weight"] = occlusion_rotation_weight
                    landmark_convention = getattr(args, "hybrik_landmark_convention", None)
                    if landmark_convention is not None:
                        backend_kwargs["landmark_convention"] = landmark_convention
                    hybrik_backend = InitializedHybrIKX(init_path, **backend_kwargs)
                    hybrik_info = {
                        "device_requested": str(getattr(args, "hybrik_device", "cuda")),
                        "device_actual": str(getattr(hybrik_backend, "device", "unknown")),
                        "initialization_npz": str(init_path.resolve()),
                        "construction_elapsed_s": float(time.perf_counter() - backend_started),
                        # The backend performs one compile/cache-warming step
                        # during construction. Keep it separate from the
                        # steady-state per-frame timing in benchmark output.
                        "warmup_elapsed_s": (
                            float(getattr(hybrik_backend, "warmup_elapsed_s"))
                            if getattr(hybrik_backend, "warmup_elapsed_s", None) is not None
                            else None
                        ),
                        "refine_steps": refine_steps,
                        "pose_initialization": pose_initialization,
                        "landmark_convention": getattr(hybrik_backend, "landmark_convention", "easymocap"),
                        "temporal_weight": temporal_weight,
                        "observation_filter_enabled": getattr(hybrik_backend, "observation_filter", None) is not None,
                        "observation_min_cutoff_hz": float(getattr(hybrik_backend, "observation_min_cutoff_hz", 0)),
                        "observation_speed_beta": float(getattr(hybrik_backend, "observation_speed_beta", 0)),
                        "observation_geometry_filter": observation_geometry_filter,
                        "occlusion_rotation_weight": occlusion_rotation_weight,
                        "rgb_root_fallback": rgb_root_fallback,
                        "refine_loss_policy": (
                            ("confidence-linear body2 hand2 native-FK shoulders derived neck excluded Huber delta .03m"
                             if getattr(hybrik_backend, "landmark_convention", "easymocap") == "dwpose"
                             else "confidence-linear body2 hand2 Huber delta .03m")
                            if refine_steps > 0 else None
                        ),
                        "torch_runtime": runtime_before,
                    }
                    if frontend_warmup_info is not None:
                        hybrik_info.update(frontend_warmup_info)
                    if rgb_pose_weight > 0.0:
                        from perception.benchmark.rgb_tracking import RGBPosePrior

                        rgb_pose_prior = RGBPosePrior(
                            device=str(getattr(args, "hybrik_device", "cuda")),
                            warmup_views=len(cfg.camera_ids),
                        )
                        hybrik_info["rgb"] = {
                            "enabled": True,
                            "weight": rgb_pose_weight,
                            "rgb_metadata": getattr(rgb_pose_prior, "rgb_metadata", None),
                            "warmup_elapsed_s": (
                                float(getattr(rgb_pose_prior, "warmup_elapsed_s"))
                                if getattr(rgb_pose_prior, "warmup_elapsed_s", None) is not None
                                else None
                            ),
                        }
                    if warmup_summary is not None:
                        warmup_summary["hybrik_initialization"] = hybrik_info
                except Exception as exc:
                    LOG.exception("InitializedHybrIKX construction failed: %s", exc)
                    warmup_valid = False
                    groups = []
                    warmup_summary = dict(warmup_summary or {})
                    warmup_summary.update(
                        {
                            "fit_ok": False,
                            "warmup_init_valid": False,
                            "failure": "hybrik_initialization_exception",
                            "initialization_npz": str(init_path.resolve()),
                            "error": str(exc),
                        }
                    )

        for moment_id, group in enumerate(groups):
            moment_dir = run_dir / f"moment_{moment_id:04d}"
            try:
                synced, metadata_report = _normalize_group(group)
                if tracking_backend == "hybrik":
                    if hybrik_backend is None:
                        raise RuntimeError("HybrIK backend was not initialized")
                    process_kwargs = dict(
                        moment_dir=moment_dir,
                        synced=synced,
                        cfg=cfg,
                        calibration=calibration,
                        detector=detector,
                        camera_ids=list(cfg.camera_ids),
                        hybrik_backend=hybrik_backend,
                        triangulator=triangulator,
                        thres2d=float(args.thres2d),
                        max_repro_error=float(args.max_repro_error),
                        motion_frame_index=group.motion_frame_index,
                        keep_rejected=bool(getattr(args, "keep_rejected", False)),
                        write_debug_images=write_debug_images,
                    )
                    if rgb_pose_prior is not None:
                        process_kwargs["rgb_pose_prior"] = rgb_pose_prior
                        process_kwargs["overlap_rgb"] = bool(getattr(args, "hybrik_overlap_rgb", True))
                    summary = process_one_hybrik_moment(**process_kwargs)
                else:
                    summary = process_one_moment(
                        moment_dir=moment_dir,
                        synced=synced,
                        cfg=cfg,
                        calibration=calibration,
                        detector=detector,
                        camera_ids=list(cfg.camera_ids),
                        gender=str(args.gender),
                        fit_model=str(args.fit_model),
                        thres2d=float(args.thres2d),
                        max_repro_error=float(args.max_repro_error),
                        mesh_alpha=float(args.mesh_alpha),
                        mesh_rgb=mesh_rgb,
                        face_stride=int(args.face_stride),
                        max_triangle_px=float(args.max_triangle_px),
                        body_model_cache=body_model_cache,
                        skip_fit=bool(args.skip_fit),
                        fixed_betas=fixed_betas,
                        bed_sdf=bool(args.bed_sdf),
                        scene_spec_path=scene_spec_path,
                        motion_frame_index=group.motion_frame_index,
                        keep_rejected=bool(getattr(args, "keep_rejected", False)),
                        disable_face_iteration=bool(getattr(args, "disable_face_iteration", False)),
                        write_debug_images=write_debug_images,
                    )
                summary.update(
                    {
                        "moment_id": int(moment_id),
                        "tracking_backend": tracking_backend,
                        "fit_device": (
                            _model_device(body_model_cache.get("model"))
                            if tracking_backend == "easymocap" else None
                        ),
                        "torch_runtime": (
                            _torch_runtime_info() if tracking_backend == "easymocap" else None
                        ),
                        "triangulation_pair_hypotheses": triangulation_pair_hypotheses,
                        "hybrik_refine_steps": (
                            max(0, int(getattr(args, "hybrik_refine_steps", 0) or 0))
                            if tracking_backend == "hybrik" else None
                        ),
                        "hybrik_refine_loss_policy": (
                            "confidence-linear body2 hand2 Huber delta .03m"
                            if tracking_backend == "hybrik"
                            and int(getattr(args, "hybrik_refine_steps", 0) or 0) > 0 else None
                        ),
                        "observation_geometry_filter": observation_geometry_filter,
                        "occlusion_rotation_weight": occlusion_rotation_weight,
                        "rgb_root_fallback": rgb_root_fallback,
                        **_source_fields(group),
                        "source_manifest": str(manifest_path),
                        "source_profile": source_profile,
                        "metadata_report": metadata_report,
                    }
                )
                if summary.get("skip_reason") or not summary.get("fit_ok"):
                    rejected.append(summary)
                    if not bool(getattr(args, "keep_rejected", False)):
                        preserved = _preserve_triangulation_artifact(
                            moment_dir=moment_dir, run_dir=run_dir, moment_id=moment_id
                        )
                        if preserved is not None:
                            summary["triangulation_npz_preserved"] = str(preserved.relative_to(run_dir))
                        shutil.rmtree(moment_dir, ignore_errors=True)
                    continue
                if fixed_betas is None and tracking_backend != "hybrik":
                    inferred = _first_fit_betas(moment_dir)
                    if inferred is not None:
                        fixed_betas = inferred
                        np.save(run_dir / "sequence_betas.npy", inferred)
                        summary["betas_reused_from_moment"] = int(moment_id)
                if publisher is not None:
                    publisher.send_moment(
                        moment_dir=moment_dir,
                        frame_index=int(group.frame_index),
                        timestamp_ns=int(group.timestamp_ns),
                        publish_kind=str(getattr(args, "publish_kind", "smplx_mesh")),
                        gender=str(args.gender),
                    )
                index.append(summary)
            except Exception as exc:
                LOG.exception("recorded group %d failed: %s", moment_id, exc)
                failed = {"moment_id": int(moment_id), **_source_fields(group), "fit_ok": False, "error": str(exc)}
                preserved = _preserve_triangulation_artifact(
                    moment_dir=moment_dir, run_dir=run_dir, moment_id=moment_id
                )
                if preserved is not None:
                    failed["triangulation_npz_preserved"] = str(preserved.relative_to(run_dir))
                rejected.append(failed)
                if not bool(getattr(args, "keep_rejected", False)):
                    shutil.rmtree(moment_dir, ignore_errors=True)
    finally:
        if triangulator is not None:
            triangulator.close()
        detector.close()
        if publisher is not None:
            publisher.close()

    run_summary = {
        "run_dir": str(run_dir.resolve()),
        "recording_dir": str(recording_dir),
        "source_manifest": str(manifest_path),
        "source_order": "manifest_jsonl",
        "source_profile": source_profile,
        "camera_ids": list(cfg.camera_ids),
        "groups_available": len(all_groups),
        "warmup_groups_selected": len(warmup_groups),
        "motion_groups_selected": len(groups),
        "groups_selected": len(warmup_groups) + len(groups),
        "warmup_moments": int(warmup_count),
        "sample_stride": int(sample_stride),
        "max_moments": int(max_moments),
        "moments_produced": len(index),
        "moments_ok": sum(1 for row in index if row.get("fit_ok")),
        "rejected_count": len(rejected),
        "tracking_backend": tracking_backend,
        "triangulation_parallel": triangulation_parallel,
        "initialization_npz": str(initialization_npz.resolve()) if initialization_npz is not None else None,
        "hybrik_device_requested": (
            str(getattr(args, "hybrik_device", "cuda")) if tracking_backend == "hybrik" else None
        ),
        "hybrik_refine_steps_requested": (
            max(0, int(getattr(args, "hybrik_refine_steps", 0) or 0))
            if tracking_backend == "hybrik" else None
        ),
        "hybrik_refine_loss_policy": (
            "confidence-linear body2 hand2 Huber delta .03m"
            if tracking_backend == "hybrik"
            and int(getattr(args, "hybrik_refine_steps", 0) or 0) > 0 else None
        ),
        "observation_geometry_filter": observation_geometry_filter,
        "occlusion_rotation_weight": occlusion_rotation_weight,
        "rgb_root_fallback": rgb_root_fallback,
        "hybrik": hybrik_info if tracking_backend == "hybrik" else None,
        "write_debug_images": write_debug_images,
        "triangulation_pair_hypotheses": triangulation_pair_hypotheses,
        "betas_mode": betas_mode,
        "betas_path": str(betas_path) if betas_path else None,
        "sequence_betas_path": str((run_dir / "sequence_betas.npy").resolve())
        if (run_dir / "sequence_betas.npy").is_file()
        else None,
        "publish_genesis": bool(getattr(args, "publish_genesis", False)),
        "publish_bind": str(getattr(args, "publish_bind", "tcp://127.0.0.1:5598")),
        "disable_face_iteration": bool(getattr(args, "disable_face_iteration", False)),
        "warmup": warmup_summary,
        "warmup_init_valid": bool(warmup_valid),
        "index": index,
        "rejected_index": rejected,
    }
    (run_dir / "index.json").write_text(
        json.dumps(_jsonable(run_summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    LOG.info(
        "recorded sequence done: %d/%d fitted -> %s",
        len(index),
        len(groups),
        run_dir,
    )
    return 0 if index or warmup_valid else 2


__all__ = [
    "RecordedFrameGroup",
    "apply_dwpose_temporal_bbox_fallback",
    "load_recorded_frame_groups",
    "run_recorded_smplx_sequence",
]
