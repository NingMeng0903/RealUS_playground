"""Frozen, continuous BABEL/AMASS validation clips for the V15 protocol.

The six clips in :data:`CLIPS_V15` are validation/regression identities.  They
are deliberately kept separate from fitting data and are replayed at the
native AMASS frame rate.  Source SMPL-H rotations are mapped to the runtime
SMPL-X 55-joint layout without clipping, retiming, shape transfer, or root
normalisation.

The source files contain SMPL-H ``poses`` (52 * 3) and ``trans``.  The source
``betas`` field is never read and is never written to a frozen clip.  A frozen
directory is write-once: an existing output path is rejected before any work
is performed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .pose_adapter import smplh156_to_smplx55
from .validation_poses_v13 import DEFAULT_AMASS_ROOT


DEFAULT_BABEL_ROOT = Path(
    "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/babel/babel_v1.0_release"
)


@dataclass(frozen=True)
class BabelSegmentV15:
    """One expected BABEL frame annotation for a validation clip."""

    raw_label: str
    proc_label: str
    start_t: float
    end_t: float


@dataclass(frozen=True)
class ValidationClipSpecV15:
    """Immutable source and frame selection for one V15 clip."""

    name: str
    sid: int
    source_relative_path: str
    babel_feat_p: str
    source_fps: float
    start_frame: int
    stop_frame: int
    babel_segments: tuple[BabelSegmentV15, ...]
    selection_note: str = ""

    @property
    def frame_ids(self) -> np.ndarray:
        """Return the exact native frame IDs selected by this spec."""
        return np.arange(self.start_frame, self.stop_frame, dtype=np.int64)

    @property
    def babel_label(self) -> str:
        return " + ".join(segment.raw_label for segment in self.babel_segments)


# These are the previously audited windows.  Every stop is exclusive and the
# reach window intentionally ends before the following BABEL knocking label.
CLIPS_V15: tuple[ValidationClipSpecV15, ...] = (
    ValidationClipSpecV15(
        name="walk_sid8836",
        sid=8836,
        source_relative_path="KIT/11/WalkingStraightForwards01_poses.npz",
        babel_feat_p="KIT/KIT/11/WalkingStraightForwards01_poses.npz",
        source_fps=100.0,
        start_frame=169,
        stop_frame=429,
        babel_segments=(BabelSegmentV15("walk", "walk", 1.686, 4.289),),
    ),
    ValidationClipSpecV15(
        name="turn_sid11003",
        sid=11003,
        source_relative_path=(
            "BioMotionLab_NTroje/rub056/0014_knocking2_poses.npz"
        ),
        babel_feat_p=(
            "BMLrub/BioMotionLab_NTroje/rub056/0014_knocking2_poses.npz"
        ),
        source_fps=120.0,
        start_frame=490,
        stop_frame=614,
        babel_segments=(BabelSegmentV15("turn around", "turn around", 4.081, 5.115),),
    ),
    ValidationClipSpecV15(
        name="sitstand_sid4336",
        sid=4336,
        source_relative_path=(
            "BioMotionLab_NTroje/rub101/0014_sitting1_poses.npz"
        ),
        babel_feat_p=(
            "BMLrub/BioMotionLab_NTroje/rub101/0014_sitting1_poses.npz"
        ),
        source_fps=120.0,
        start_frame=353,
        stop_frame=793,
        babel_segments=(
            BabelSegmentV15("sit", "sit", 2.940, 6.148),
            BabelSegmentV15("stand up", "stand up", 6.148, 6.606),
        ),
    ),
    ValidationClipSpecV15(
        name="reach_sid12951",
        sid=12951,
        source_relative_path=(
            "BioMotionLab_NTroje/rub030/0014_knocking2_poses.npz"
        ),
        babel_feat_p=(
            "BMLrub/BioMotionLab_NTroje/rub030/0014_knocking2_poses.npz"
        ),
        source_fps=120.0,
        start_frame=206,
        stop_frame=291,
        babel_segments=(
            BabelSegmentV15(
                "reaching out with right hand",
                "reach out with right hand",
                1.711,
                2.461,
            ),
        ),
        selection_note=(
            "Strict reach-only window ends at frame 291 before the following "
            "BABEL knocking segment beginning at 2.419 s."
        ),
    ),
    ValidationClipSpecV15(
        name="drink_sid3307",
        sid=3307,
        source_relative_path="KIT/3/Drinking03_poses.npz",
        babel_feat_p="KIT/KIT/3/Drinking03_poses.npz",
        source_fps=100.0,
        start_frame=275,
        stop_frame=479,
        babel_segments=(
            BabelSegmentV15("drink from bottle", "drink from bottle", 2.748, 4.788),
        ),
    ),
    ValidationClipSpecV15(
        name="armup_sid4010",
        sid=4010,
        source_relative_path=(
            "BioMotionLab_NTroje/rub026/0013_knocking1_poses.npz"
        ),
        babel_feat_p=(
            "BMLrub/BioMotionLab_NTroje/rub026/0013_knocking1_poses.npz"
        ),
        source_fps=120.0,
        start_frame=277,
        stop_frame=375,
        babel_segments=(BabelSegmentV15("hold arm up", "hold arm up", 2.304, 3.120),),
    ),
)

CLIP_INDEX_V15: Mapping[str, ValidationClipSpecV15] = {
    spec.name: spec for spec in CLIPS_V15
}


@dataclass(frozen=True)
class ValidationClipV15:
    """In-memory native-frame clip ready for metrics or Genesis replay."""

    spec: ValidationClipSpecV15
    source_path: Path
    source_fps: float
    frame_ids: np.ndarray
    poses: np.ndarray
    transl: np.ndarray
    source_gender: str = ""
    source_sha256: str = ""

    def __post_init__(self) -> None:
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        poses = np.asarray(self.poses, dtype=np.float32)
        transl = np.asarray(self.transl)
        if frame_ids.ndim != 1:
            raise ValueError("frame_ids must be one-dimensional")
        if poses.shape != (len(frame_ids), 55, 3):
            raise ValueError(
                f"poses must have shape ({len(frame_ids)}, 55, 3), got {poses.shape}"
            )
        if transl.shape != (len(frame_ids), 3):
            raise ValueError(
                f"transl must have shape ({len(frame_ids)}, 3), got {transl.shape}"
            )
        if not np.isfinite(poses).all() or not np.isfinite(transl).all():
            raise ValueError("validation clip contains non-finite pose or translation")
        if not np.isfinite(float(self.source_fps)) or float(self.source_fps) <= 0:
            raise ValueError("source_fps must be finite and positive")
        if not np.array_equal(frame_ids, self.spec.frame_ids):
            raise ValueError("frame_ids do not match the immutable clip specification")
        frame_ids.setflags(write=False)
        poses.setflags(write=False)
        transl.setflags(write=False)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "transl", transl)
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def sid(self) -> int:
        return self.spec.sid

    @property
    def frame_count(self) -> int:
        return int(self.frame_ids.size)

    @property
    def fps(self) -> float:
        return float(self.source_fps)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_relative(value: str) -> str:
    return Path(str(value).strip()).as_posix()


def _source_path_variants(spec: ValidationClipSpecV15) -> tuple[str, ...]:
    """Return the known BABEL/AMASS path aliases without broad filesystem search."""
    values: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = _normalise_relative(value)
        if value and value not in seen:
            values.append(value)
            seen.add(value)

    add(spec.source_relative_path)
    add(spec.babel_feat_p)
    for value in tuple(values):
        parts = value.split("/")
        if len(parts) >= 2 and parts[0] == parts[1]:
            add("/".join(parts[1:]))
        if parts and parts[0] in {
            "BMLrub",
            "DFaust67",
            "EyesJapanDataset",
            "MPIHDM05",
            "MPILimits",
            "MPImosh",
            "SSMsynced",
            "TCDhandMocap",
            "Transitionsmocap",
        }:
            add("/".join(parts[1:]))
    return tuple(values)


def _resolve_source_path(
    spec: ValidationClipSpecV15,
    amass_root: str | Path,
) -> Path:
    root = Path(amass_root).expanduser().resolve()
    bases = (root, root / "raw")
    for base in bases:
        for relative in _source_path_variants(spec):
            candidate = (base / relative).resolve()
            if candidate.is_file():
                return candidate
    expected = ", ".join(_source_path_variants(spec))
    raise FileNotFoundError(
        f"{spec.name}: no AMASS NPZ under {root}; expected one of {expected}"
    )


def _resolve_babel_release_dir(babel_root: str | Path) -> Path:
    root = Path(babel_root).expanduser().resolve()
    if root.is_file() and root.suffix.lower() == ".json":
        root = root.parent
    if (root / "train.json").is_file():
        return root
    nested = root / "babel_v1.0_release"
    if (nested / "train.json").is_file():
        return nested
    raise FileNotFoundError(
        f"No BABEL release JSON under {root}; expected train.json or "
        "babel_v1.0_release/train.json"
    )


def _load_babel_provenance(
    spec: ValidationClipSpecV15,
    babel_root: str | Path,
) -> tuple[Path, str, dict[str, Any], list[dict[str, Any]]]:
    release = _resolve_babel_release_dir(babel_root)
    for split in ("train", "val", "extra_train", "extra_val", "test"):
        annotation_path = release / f"{split}.json"
        if not annotation_path.is_file():
            continue
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        record = payload.get(str(spec.sid))
        if record is None:
            continue
        if int(record.get("babel_sid", -1)) != spec.sid:
            raise ValueError(f"BABEL SID mismatch in {annotation_path}: {spec.sid}")
        if _normalise_relative(str(record.get("feat_p", ""))) != _normalise_relative(
            spec.babel_feat_p
        ):
            raise ValueError(
                f"BABEL feat_p mismatch for SID {spec.sid}: "
                f"{record.get('feat_p')!r} != {spec.babel_feat_p!r}"
            )
        labels = record.get("frame_ann", {}).get("labels", [])
        selected: list[dict[str, Any]] = []
        for expected in spec.babel_segments:
            matches = [
                label
                for label in labels
                if str(label.get("raw_label", "")) == expected.raw_label
                and str(label.get("proc_label", "")) == expected.proc_label
                and np.isclose(float(label.get("start_t")), expected.start_t, atol=1e-6)
                and np.isclose(float(label.get("end_t")), expected.end_t, atol=1e-6)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"BABEL label lookup for SID {spec.sid} expected one "
                    f"{expected.raw_label!r} segment, found {len(matches)}"
                )
            selected.append(dict(matches[0]))
        return annotation_path, split, dict(record), selected
    raise FileNotFoundError(
        f"BABEL SID {spec.sid} with feat_p {spec.babel_feat_p!r} was not found "
        f"under {release}"
    )


def recover_smplh156_from_smplx55(pose55: Any) -> np.ndarray:
    """Recover the SMPL-H 52-joint block from an adapted SMPL-X pose.

    This is intentionally strict: non-zero SMPL-X face slots would make the
    inverse lossy and therefore raise instead of silently dropping data.
    """
    array = np.asarray(pose55, dtype=np.float32)
    if array.ndim < 2 or array.shape[-2:] != (55, 3):
        raise ValueError(f"pose55 must end with shape (55, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("pose55 contains non-finite rotations")
    if not np.array_equal(array[..., 22:25, :], np.zeros_like(array[..., 22:25, :])):
        raise ValueError("SMPL-X face slots 22:25 must be zero for SMPL-H recovery")
    return np.concatenate(
        (array[..., :22, :], array[..., 25:40, :], array[..., 40:55, :]), axis=-2
    ).astype(np.float32, copy=False)


def _resolve_spec(clip: str | ValidationClipSpecV15) -> ValidationClipSpecV15:
    if isinstance(clip, ValidationClipSpecV15):
        return clip
    try:
        return CLIP_INDEX_V15[str(clip)]
    except KeyError as exc:
        names = ", ".join(CLIP_INDEX_V15)
        raise KeyError(f"Unknown V15 validation clip {clip!r}; choose one of {names}") from exc


def load_validation_clip_v15(
    clip: str | ValidationClipSpecV15,
    amass_root: str | Path = DEFAULT_AMASS_ROOT,
) -> ValidationClipV15:
    """Load one full-resolution frozen-window clip into memory.

    No source ``betas`` key is accessed.  The returned arrays are read-only,
    while ``poses`` retains every selected root/body/hand rotation and
    ``transl`` retains the source translation values without a transform.
    """
    spec = _resolve_spec(clip)
    source_path = _resolve_source_path(spec, amass_root)
    frame_ids = spec.frame_ids
    with np.load(source_path, allow_pickle=False) as data:
        if "poses" not in data.files:
            raise ValueError(f"{source_path}: missing poses array")
        raw = np.asarray(data["poses"])
        if raw.ndim != 2 or raw.shape[1] != 156:
            raise ValueError(f"{source_path}: expected poses shape [frames, 156], got {raw.shape}")
        if spec.start_frame < 0 or spec.stop_frame <= spec.start_frame:
            raise ValueError(f"{spec.name}: invalid frame window {spec.start_frame}:{spec.stop_frame}")
        if spec.stop_frame > raw.shape[0]:
            raise ValueError(
                f"{spec.name}: frame window {spec.start_frame}:{spec.stop_frame} "
                f"exceeds source length {raw.shape[0]}"
            )
        if "trans" not in data.files:
            raise ValueError(f"{source_path}: missing trans array")
        trans = np.asarray(data["trans"])
        if trans.ndim != 2 or trans.shape[1] != 3 or trans.shape[0] < spec.stop_frame:
            raise ValueError(
                f"{source_path}: expected trans shape [frames, 3] covering stop "
                f"{spec.stop_frame}, got {trans.shape}"
            )
        try:
            source_fps = float(np.asarray(data["mocap_framerate"]).item())
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source_path}: invalid mocap_framerate") from exc
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"{source_path}: invalid mocap_framerate {source_fps!r}")
        if not np.isclose(source_fps, spec.source_fps, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{spec.name}: source fps {source_fps} disagrees with frozen "
                f"fps {spec.source_fps}"
            )
        source_pose = np.asarray(raw[frame_ids])
        source_trans = np.asarray(trans[frame_ids])
        if not np.isfinite(source_pose).all():
            raise ValueError(f"{spec.name}: selected source poses contain non-finite values")
        if not np.isfinite(source_trans).all():
            raise ValueError(f"{spec.name}: selected source trans contains non-finite values")
        poses = np.stack(
            [smplh156_to_smplx55(row) for row in source_pose], axis=0
        ).astype(np.float32, copy=False)
        if not np.isfinite(poses).all():
            raise ValueError(f"{spec.name}: adapted poses contain non-finite values")
        recovered = recover_smplh156_from_smplx55(poses)
        source_float32 = source_pose.astype(np.float32).reshape(-1, 52, 3)
        if not np.array_equal(recovered, source_float32):
            raise ValueError(f"{spec.name}: SMPL-H rotations changed during adaptation")
        transl = source_trans.copy()
        gender_value = data["gender"] if "gender" in data.files else ""
        try:
            source_gender = str(np.asarray(gender_value).item())
        except (TypeError, ValueError):
            source_gender = str(gender_value)
    return ValidationClipV15(
        spec=spec,
        source_path=source_path,
        source_fps=source_fps,
        frame_ids=frame_ids,
        poses=poses,
        transl=transl,
        source_gender=source_gender,
        source_sha256=_sha256_file(source_path),
    )


def load_validation_clips_v15(
    names: Iterable[str] | None = None,
    amass_root: str | Path = DEFAULT_AMASS_ROOT,
) -> dict[str, ValidationClipV15]:
    """Load all V15 clips, or the requested names, for metrics/replay."""
    specs = CLIPS_V15 if names is None else tuple(_resolve_spec(name) for name in names)
    return {
        spec.name: load_validation_clip_v15(spec, amass_root=amass_root)
        for spec in specs
    }


def _manifest_babel_fields(
    spec: ValidationClipSpecV15,
    babel_info: tuple[Path, str, dict[str, Any], list[dict[str, Any]]],
) -> dict[str, Any]:
    annotation_path, split, record, selected = babel_info
    label_hash_payload = {
        "babel_sid": spec.sid,
        "feat_p": spec.babel_feat_p,
        "segments": selected,
    }
    return {
        "babel_split": split,
        "babel_annotation_path": str(annotation_path.resolve()),
        "babel_feat_p": spec.babel_feat_p,
        "babel_label": spec.babel_label,
        "babel_labels": selected,
        "babel_label_sha256": _canonical_hash(label_hash_payload),
        "babel_record_sha256": _canonical_hash(record),
        "babel_json_sha256": _sha256_file(annotation_path),
    }


def freeze_validation_motion_v15(
    output: str | Path,
    amass_root: str | Path = DEFAULT_AMASS_ROOT,
    babel_root: str | Path = DEFAULT_BABEL_ROOT,
    *,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write a write-once directory containing the selected V15 NPZ clips.

    The preflight loads and validates all source clips and BABEL annotations
    before creating ``output``.  Each archive contains only ``poses``,
    ``transl``, ``frame_ids`` and ``source_fps``; source SMPL-H betas are never
    transferred.
    """
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"V15 validation output already exists: {output_path}")
    specs = CLIPS_V15 if names is None else tuple(_resolve_spec(name) for name in names)
    clips = [load_validation_clip_v15(spec, amass_root=amass_root) for spec in specs]
    babel_infos = [
        _load_babel_provenance(spec, babel_root=babel_root) for spec in specs
    ]

    output_path.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "protocol": "v15_babel_amass_continuous_validation_v1",
        "immutable": True,
        "used_for_fit": False,
        "source_dataset": "AMASS",
        "source_pose_layout": "smplh156",
        "target_pose_layout": "smplx55",
        "native_frame_stride": 1,
        "pose_clipped": False,
        "amass_betas_transferred": False,
        "trans_key": "transl",
        "clips": {},
    }
    for clip, babel_info in zip(clips, babel_infos):
        spec = clip.spec
        filename = f"{spec.name}.npz"
        target = output_path / filename
        np.savez_compressed(
            target,
            poses=np.asarray(clip.poses, dtype=np.float32),
            transl=np.asarray(clip.transl),
            frame_ids=np.asarray(clip.frame_ids, dtype=np.int64),
            source_fps=np.asarray(clip.source_fps, dtype=np.float64),
        )
        record: dict[str, Any] = {
            "name": spec.name,
            "sid": spec.sid,
            "source_path": str(clip.source_path),
            "source_relative_path": spec.source_relative_path,
            "source_sha256": clip.source_sha256,
            "source_gender": clip.source_gender,
            "source_fps": clip.fps,
            "expected_source_fps": spec.source_fps,
            "start_frame": spec.start_frame,
            "stop_frame_exclusive": spec.stop_frame,
            "frame_ids": spec.frame_ids.tolist(),
            "frame_count": clip.frame_count,
            "window_start_seconds": spec.start_frame / clip.fps,
            "window_stop_seconds_exclusive": spec.stop_frame / clip.fps,
            "babel_annotation_start_seconds": spec.babel_segments[0].start_t,
            "babel_annotation_end_seconds": spec.babel_segments[-1].end_t,
            "selection_note": spec.selection_note,
            "native_frame_stride": 1,
            "used_for_fit": False,
            "pose_clipped": False,
            "amass_betas_transferred": False,
            "source_pose_layout": "smplh156",
            "target_pose_layout": "smplx55",
            "output_file": filename,
            "output_keys": ["poses", "transl", "frame_ids", "source_fps"],
        }
        record.update(_manifest_babel_fields(spec, babel_info))
        record["archive_sha256"] = _sha256_file(target)
        manifest["clips"][spec.name] = record
    manifest["clip_names"] = [spec.name for spec in specs]
    manifest["clip_count"] = len(specs)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frozen_validation_clip_v15(
    directory: str | Path,
    name: str,
) -> ValidationClipV15:
    """Load and integrity-check one NPZ from a V15 frozen directory."""
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing V15 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "v15_babel_amass_continuous_validation_v1":
        raise ValueError("unsupported V15 validation manifest protocol")
    if manifest.get("used_for_fit") is not False:
        raise ValueError("V15 validation manifest must remain used_for_fit=false")
    record = manifest.get("clips", {}).get(str(name))
    if record is None:
        raise KeyError(f"V15 frozen clip not found: {name}")
    if record.get("used_for_fit") is not False:
        raise ValueError(f"frozen clip {name} is marked used_for_fit")
    target = root / str(record.get("output_file", f"{name}.npz"))
    if not target.is_file():
        raise FileNotFoundError(f"Missing V15 frozen archive: {target}")
    expected_archive_hash = record.get("archive_sha256")
    if expected_archive_hash and _sha256_file(target) != expected_archive_hash:
        raise ValueError(f"V15 frozen archive hash mismatch: {target}")
    spec = _resolve_spec(str(name))
    with np.load(target, allow_pickle=False) as data:
        required = {"poses", "transl", "frame_ids", "source_fps"}
        if set(data.files) != required:
            raise ValueError(
                f"{target}: expected exactly {sorted(required)}, got {sorted(data.files)}"
            )
        poses = np.asarray(data["poses"], dtype=np.float32)
        transl = np.asarray(data["transl"])
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        source_fps = float(np.asarray(data["source_fps"]).item())
    if not np.array_equal(frame_ids, np.asarray(record.get("frame_ids"), dtype=np.int64)):
        raise ValueError(f"{target}: frame IDs disagree with manifest")
    if record.get("sid") != spec.sid:
        raise ValueError(f"{target}: SID disagrees with immutable specification")
    if record.get("source_fps") != source_fps:
        raise ValueError(f"{target}: source fps disagrees with manifest")
    recover_smplh156_from_smplx55(poses)
    source_path = Path(str(record.get("source_path", "")))
    return ValidationClipV15(
        spec=spec,
        source_path=source_path,
        source_fps=source_fps,
        frame_ids=frame_ids,
        poses=poses,
        transl=transl,
        source_gender=str(record.get("source_gender", "")),
        source_sha256=str(record.get("source_sha256", "")),
    )


# A short alias mirrors the V14 function while keeping the version explicit in
# the primary API name.
freeze_validation_motion = freeze_validation_motion_v15


__all__ = [
    "BabelSegmentV15",
    "CLIPS_V15",
    "CLIP_INDEX_V15",
    "DEFAULT_AMASS_ROOT",
    "DEFAULT_BABEL_ROOT",
    "ValidationClipSpecV15",
    "ValidationClipV15",
    "freeze_validation_motion",
    "freeze_validation_motion_v15",
    "load_frozen_validation_clip_v15",
    "load_validation_clip_v15",
    "load_validation_clips_v15",
    "recover_smplh156_from_smplx55",
]
