from __future__ import annotations

import contextlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.layout import HumanDatasetLayout


def _normalize_gender(value: Any, default: str = "neutral") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
        return _normalize_gender(value, default=default)
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return default


def _repo_root() -> Path:
    return project_paths(__file__).root


_numpy_chumpy_shim_done = False


def ensure_numpy_aliases_for_chumpy() -> None:
    """chumpy (SMPL .pkl) does ``from numpy import int, ...``; removed in NumPy 1.24+."""
    global _numpy_chumpy_shim_done
    if _numpy_chumpy_shim_done:
        return
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64
    np.complex = np.complex128
    np.object = np.object_
    np.unicode = np.str_
    np.str = np.str_
    _numpy_chumpy_shim_done = True


@dataclass
class HumanMotionSequence:
    source_dataset: str
    sequence_name: str
    source_path: str
    model_type: str
    fps: float
    gender: str
    betas: np.ndarray
    poses: np.ndarray
    trans: np.ndarray
    image_names: list[str] = field(default_factory=list)
    cam_int: np.ndarray | None = None
    cam_ext: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.betas = np.asarray(self.betas, dtype=np.float32).reshape(-1)
        self.poses = np.asarray(self.poses, dtype=np.float32)
        self.trans = np.asarray(self.trans, dtype=np.float32)
        if self.poses.ndim != 2:
            raise ValueError(f"'poses' must be 2D, got shape {self.poses.shape}.")
        if self.trans.ndim != 2 or self.trans.shape[0] != self.poses.shape[0] or self.trans.shape[1] < 3:
            raise ValueError(f"'trans' must have shape (F, 3+), got {self.trans.shape}.")
        if self.cam_int is not None:
            self.cam_int = np.asarray(self.cam_int, dtype=np.float32)
        if self.cam_ext is not None:
            self.cam_ext = np.asarray(self.cam_ext, dtype=np.float32)
        self.gender = _normalize_gender(self.gender)
        self.model_type = self.model_type.lower()

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])

    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "source_dataset": self.source_dataset,
            "sequence_name": self.sequence_name,
            "source_path": self.source_path,
            "model_type": self.model_type,
            "fps": np.asarray([self.fps], dtype=np.float32),
            "gender": np.asarray([self.gender]),
            "betas": self.betas.astype(np.float32),
            "poses": self.poses.astype(np.float32),
            "trans": self.trans.astype(np.float32),
            "image_names": np.asarray(self.image_names),
            "metadata_json": np.asarray([json.dumps(self.metadata, ensure_ascii=True)]),
        }
        if self.cam_int is not None:
            payload["cam_int"] = self.cam_int.astype(np.float32)
        if self.cam_ext is not None:
            payload["cam_ext"] = self.cam_ext.astype(np.float32)
        np.savez_compressed(output_path, **payload)
        return output_path

    @classmethod
    def load(cls, input_path: Path) -> "HumanMotionSequence":
        with np.load(input_path, allow_pickle=True) as payload:
            metadata_json = payload["metadata_json"][0] if "metadata_json" in payload else "{}"
            metadata = json.loads(str(metadata_json))
            return cls(
                source_dataset=str(payload["source_dataset"]),
                sequence_name=str(payload["sequence_name"]),
                source_path=str(payload["source_path"]),
                model_type=str(payload["model_type"]),
                fps=float(np.asarray(payload["fps"]).reshape(-1)[0]),
                gender=_normalize_gender(payload["gender"]),
                betas=np.asarray(payload["betas"], dtype=np.float32),
                poses=np.asarray(payload["poses"], dtype=np.float32),
                trans=np.asarray(payload["trans"], dtype=np.float32),
                image_names=[str(item) for item in payload.get("image_names", np.asarray([])).tolist()],
                cam_int=np.asarray(payload["cam_int"], dtype=np.float32) if "cam_int" in payload else None,
                cam_ext=np.asarray(payload["cam_ext"], dtype=np.float32) if "cam_ext" in payload else None,
                metadata=metadata,
            )


def npz_shape_reference_for_retarget_cache(npz_path: Path) -> dict[str, Any]:
    """Shape identity for retarget cache signatures without loading full ``poses`` / ``trans``.

    Decompressing large ``poses`` from ``.npz`` on slow storage can take minutes; signing the cache
    only requires betas and metadata fields present beside ``poses`` in the archive.
    """
    npz_path = Path(npz_path)
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            required = ("betas", "source_dataset", "model_type", "gender")
            if not all(k in z.files for k in required):
                raise KeyError("missing npz keys for lightweight shape ref")
            betas = np.asarray(z["betas"], dtype=np.float32).reshape(-1)
            sd_raw = z["source_dataset"]
            mt_raw = z["model_type"]
            if isinstance(sd_raw, np.ndarray):
                sd = str(sd_raw.reshape(-1)[0])
            else:
                sd = str(sd_raw)
            if isinstance(mt_raw, np.ndarray):
                mt = str(mt_raw.reshape(-1)[0])
            else:
                mt = str(mt_raw)
            gender = _normalize_gender(z["gender"])
            return {
                "source_dataset": sd,
                "model_type": mt.lower(),
                "gender": gender,
                "betas_dim": int(betas.size),
                "betas": [float(v) for v in betas.tolist()],
            }
    except (KeyError, OSError, ValueError):
        seq = HumanMotionSequence.load(npz_path)
        return {
            "source_dataset": str(seq.source_dataset),
            "model_type": str(seq.model_type),
            "gender": str(seq.gender),
            "betas_dim": int(seq.betas.size),
            "betas": [float(v) for v in np.asarray(seq.betas, dtype=np.float32).reshape(-1).tolist()],
        }


def load_amass_sequence(npz_path: Path, *, fps: float = 60.0) -> HumanMotionSequence:
    with np.load(npz_path, allow_pickle=True) as payload:
        poses = np.asarray(payload["poses"], dtype=np.float32)
        trans = np.asarray(payload["trans"], dtype=np.float32)
        betas = np.asarray(payload["betas"], dtype=np.float32).reshape(-1)
        gender = _normalize_gender(payload["gender"] if "gender" in payload else "neutral")
        mocap_framerate = float(np.asarray(payload["mocap_framerate"]).reshape(-1)[0]) if "mocap_framerate" in payload else fps
    return HumanMotionSequence(
        source_dataset="amass",
        sequence_name=npz_path.stem,
        source_path=str(npz_path),
        model_type="smpl",
        fps=mocap_framerate,
        gender=gender,
        betas=betas[:10],
        poses=poses,
        trans=trans[:, :3],
        metadata={"raw_fields": ["poses", "trans", "betas"]},
    )


def load_bedlam_sequence(
    npz_path: Path,
    *,
    subject_id: str | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    pose_key: str = "pose_world",
    trans_key: str = "trans_world",
) -> HumanMotionSequence:
    with np.load(npz_path, allow_pickle=True) as payload:
        if pose_key not in payload:
            raise KeyError(f"Missing '{pose_key}' in BEDLAM npz: {npz_path}")
        if trans_key not in payload:
            raise KeyError(f"Missing '{trans_key}' in BEDLAM npz: {npz_path}")
        pose_world = np.asarray(payload[pose_key], dtype=np.float32)
        trans_world = np.asarray(payload[trans_key], dtype=np.float32)
        shape = np.asarray(payload["shape"], dtype=np.float32)
        genders = payload["gender"] if "gender" in payload else np.asarray(["neutral"] * len(pose_world))
        subjects = payload["sub"] if "sub" in payload else np.asarray(["subject_0"] * len(pose_world))
        image_names = payload["imgname"] if "imgname" in payload else np.asarray([])
        motion_info = payload["motion_info"] if "motion_info" in payload else np.asarray([])
        cam_int = np.asarray(payload["cam_int"], dtype=np.float32) if "cam_int" in payload else None
        cam_ext = np.asarray(payload["cam_ext"], dtype=np.float32) if "cam_ext" in payload else None

    subject_labels = np.asarray([str(item) for item in subjects.reshape(-1)], dtype=object)
    if subject_id is None:
        subject_id = str(subject_labels[0])
    mask = subject_labels == str(subject_id)
    if not np.any(mask):
        raise ValueError(f"Subject '{subject_id}' not found in BEDLAM npz: {npz_path}")

    indices = np.nonzero(mask)[0]
    indices = indices[:: max(stride, 1)]
    if max_frames is not None:
        indices = indices[:max_frames]
    if len(indices) == 0:
        raise ValueError("No BEDLAM frames selected after applying subject/stride/max_frames filters.")

    shape_seq = shape[indices]
    betas = np.median(shape_seq, axis=0).astype(np.float32)
    image_names_seq = [str(item) for item in np.asarray(image_names)[indices].tolist()] if image_names.size else []
    cam_int_seq = cam_int[indices] if cam_int is not None else None
    cam_ext_seq = cam_ext[indices] if cam_ext is not None else None
    motion_info_seq = np.asarray(motion_info)[indices].tolist() if motion_info.size else []
    fps = 30.0
    return HumanMotionSequence(
        source_dataset="bedlam",
        sequence_name=f"{npz_path.stem}_{subject_id}",
        source_path=str(npz_path),
        model_type="smplx",
        fps=fps,
        gender=_normalize_gender(np.asarray(genders)[indices]),
        betas=betas,
        poses=pose_world[indices],
        trans=trans_world[indices, :3],
        image_names=image_names_seq,
        cam_int=cam_int_seq,
        cam_ext=cam_ext_seq,
        metadata={
            "subject_id": subject_id,
            "source_pose_key": pose_key,
            "source_trans_key": trans_key,
            "motion_info": motion_info_seq,
        },
    )


def resolve_body_model_dir(model_type: str) -> Path:
    model_type = model_type.lower()
    if model_type == "smpl":
        env_raw = os.environ.get("AMONGUS_SMPL_MODEL_DIR", "").strip()
        env_value = Path(env_raw).expanduser() if env_raw else None
        candidates = [
            env_value,
            HumanDatasetLayout.default().body_models_root / "smpl",
            _repo_root() / "ref_code_library" / "GVHMR" / "inputs" / "checkpoints" / "body_models",
            Path.home() / ".cache" / "smpl" / "models",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if (candidate / "SMPL_NEUTRAL.pkl").exists():
                return candidate
        raise FileNotFoundError(
            "SMPL models not found. Set AMONGUS_SMPL_MODEL_DIR or populate dataset/intermediate/humans/body_models/smpl."
        )

    layout = HumanDatasetLayout.default()
    candidates = [
        layout.body_models_root / "smplx" / "models",
        _repo_root() / "ref_code_library" / "BEDLAM" / "data" / "body_models" / "smplx" / "models",
        Path.home() / ".cache" / "smplx" / "models",
    ]
    for candidate in candidates:
        if (candidate / "smplx" / "SMPLX_NEUTRAL.npz").exists():
            return candidate
    raise FileNotFoundError(
        "SMPL-X models not found. Populate dataset/intermediate/humans/body_models/smplx/models or the BEDLAM data path."
    )


def resolve_torch_device(preferred: str | None = None):
    import torch

    requested = str(preferred or os.environ.get("AMONGUS_TORCH_DEVICE", "auto")).strip().lower() or "auto"
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA for body-model evaluation, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported body-model torch device: {requested}")


def _resolve_available_gender(model_dir: Path, requested_gender: str, *, extension: str) -> str:
    requested = requested_gender.upper()
    candidates = [requested, "NEUTRAL", "MALE", "FEMALE"]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (Path(model_dir) / f"SMPL_{candidate}.{extension}").exists():
            return candidate
    return "NEUTRAL"


_SMPL_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _smpl_model_cache_key(sequence: HumanMotionSequence, torch_device) -> tuple[Any, ...]:
    model_type = sequence.model_type.lower()
    if model_type == "smpl":
        model_dir = resolve_body_model_dir("smpl")
        gender = _resolve_available_gender(model_dir, sequence.gender, extension="pkl")
        return ("smpl", str(model_dir.resolve()), gender, str(torch_device))
    if model_type == "smplx":
        model_dir = resolve_body_model_dir("smplx")
        return ("smplx", str(model_dir.resolve()), sequence.gender.upper(), str(torch_device))
    raise ValueError(f"Unsupported model_type: {sequence.model_type}")


def _resolve_frame_slice_bounds(frame_slice: slice | None, frame_count: int) -> tuple[int, int]:
    if frame_slice is None:
        return 0, int(frame_count)
    start, stop, step = frame_slice.indices(int(frame_count))
    if step != 1:
        raise ValueError("Only contiguous frame slices are supported for SMPL evaluation.")
    return int(start), int(max(start, stop))


def _create_smpl_model(sequence: HumanMotionSequence, torch_device):
    import smplx

    ensure_numpy_aliases_for_chumpy()
    key = _smpl_model_cache_key(sequence, torch_device)
    cached = _SMPL_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model_type = sequence.model_type.lower()
    with contextlib.redirect_stdout(io.StringIO()):
        if model_type == "smpl":
            model_dir = resolve_body_model_dir("smpl")
            model = smplx.SMPL(
                model_path=str(model_dir),
                gender=_resolve_available_gender(model_dir, sequence.gender, extension="pkl"),
                ext="pkl",
            ).to(torch_device)
        elif model_type == "smplx":
            model = smplx.create(
                model_path=str(resolve_body_model_dir("smplx")),
                model_type="smplx",
                gender=sequence.gender.upper(),
                ext="npz",
                use_pca=False,
                flat_hand_mean=True,
            ).to(torch_device)
        else:
            raise ValueError(f"Unsupported model_type: {sequence.model_type}")
    _SMPL_MODEL_CACHE[key] = model
    return model


def _build_smpl_kwargs(sequence: HumanMotionSequence, *, torch_device, frame_slice: slice | None = None):
    import torch

    start, stop = _resolve_frame_slice_bounds(frame_slice, sequence.frame_count)
    pose = np.asarray(sequence.poses[start:stop], dtype=np.float32)
    trans = np.asarray(sequence.trans[start:stop, :3], dtype=np.float32)
    frame_count = int(stop - start)
    model_type = sequence.model_type.lower()
    if model_type == "smpl":
        return {
            "betas": torch.from_numpy(np.repeat(sequence.betas[None, :10], frame_count, axis=0)).float().to(torch_device),
            "global_orient": torch.from_numpy(pose[:, :3]).float().to(torch_device),
            "body_pose": torch.from_numpy(pose[:, 3:72]).float().to(torch_device),
            "transl": torch.from_numpy(trans).float().to(torch_device),
        }

    if model_type == "smplx":
        kwargs = {
            "betas": torch.from_numpy(np.repeat(sequence.betas[None, : min(len(sequence.betas), 16)], frame_count, axis=0)).float().to(torch_device),
            "global_orient": torch.from_numpy(pose[:, :3]).float().to(torch_device),
            "body_pose": torch.from_numpy(pose[:, 3:66]).float().to(torch_device),
            "transl": torch.from_numpy(trans).float().to(torch_device),
        }
        if pose.shape[1] >= 111:
            kwargs["left_hand_pose"] = torch.from_numpy(pose[:, 66:111]).float().to(torch_device)
        if pose.shape[1] >= 156:
            kwargs["right_hand_pose"] = torch.from_numpy(pose[:, 111:156]).float().to(torch_device)
        if pose.shape[1] >= 159:
            kwargs["jaw_pose"] = torch.from_numpy(pose[:, 156:159]).float().to(torch_device)
        if pose.shape[1] >= 162:
            kwargs["leye_pose"] = torch.from_numpy(pose[:, 159:162]).float().to(torch_device)
        if pose.shape[1] >= 165:
            kwargs["reye_pose"] = torch.from_numpy(pose[:, 162:165]).float().to(torch_device)
        return kwargs

    raise ValueError(f"Unsupported model_type: {sequence.model_type}")


def _build_smpl_model(sequence: HumanMotionSequence, *, device: str | None = None):
    torch_device = resolve_torch_device(device)
    model = _create_smpl_model(sequence, torch_device)
    kwargs = _build_smpl_kwargs(sequence, torch_device=torch_device)
    return model, kwargs


def _smpl_frame_chunk_size(sequence: HumanMotionSequence, *, frame_chunk_size: int | None, torch_device) -> int:
    if str(torch_device) != "cuda":
        return int(sequence.frame_count)
    if frame_chunk_size is None:
        raw = os.environ.get("AMONGUS_SMPL_FRAME_CHUNK_SIZE", "").strip()
        frame_chunk_size = int(raw) if raw else 128
    size = int(frame_chunk_size)
    if size <= 0:
        return int(sequence.frame_count)
    return min(size, int(sequence.frame_count))


def evaluate_smpl_sequence(
    sequence: HumanMotionSequence,
    *,
    device: str | None = None,
    frame_chunk_size: int | None = None,
    include_vertices: bool = True,
    include_joints: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    import torch

    torch_device = resolve_torch_device(device)
    chunk_size = _smpl_frame_chunk_size(sequence, frame_chunk_size=frame_chunk_size, torch_device=torch_device)
    model = _create_smpl_model(sequence, torch_device)
    vertex_chunks: list[np.ndarray] | None = [] if include_vertices else None
    joint_chunks: list[np.ndarray] | None = [] if include_joints else None
    try:
        with torch.inference_mode():
            for start in range(0, int(sequence.frame_count), chunk_size):
                stop = min(start + chunk_size, int(sequence.frame_count))
                kwargs = _build_smpl_kwargs(sequence, torch_device=torch_device, frame_slice=slice(start, stop))
                model_out = model(**kwargs)
                if vertex_chunks is not None:
                    vertex_chunks.append(model_out.vertices.detach().cpu().numpy().astype(np.float32))
                if joint_chunks is not None:
                    joint_chunks.append(model_out.joints.detach().cpu().numpy().astype(np.float32))
                del model_out
                del kwargs
                if str(torch_device) == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if str(torch_device) == "cuda":
            torch.cuda.empty_cache()
    vertices = None if vertex_chunks is None else np.concatenate(vertex_chunks, axis=0)
    joints = None if joint_chunks is None else np.concatenate(joint_chunks, axis=0)
    return vertices, joints


def build_trimesh_sequence(
    sequence: HumanMotionSequence,
    *,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_floor: bool = True,
    color: tuple[int, int, int, int] = (184, 209, 245, 255),
) -> list[trimesh.Trimesh]:
    import torch
    import trimesh

    model, kwargs = _build_smpl_model(sequence)
    with torch.no_grad():
        model_out = model(**kwargs)
    vertices_seq = model_out.vertices.detach().cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int32)
    offset = np.asarray(world_offset, dtype=np.float32).reshape(3)
    meshes: list[trimesh.Trimesh] = []
    for vertices in vertices_seq:
        verts = vertices.copy()
        if align_floor:
            verts[:, 2] -= float(np.percentile(verts[:, 2], 2.0))
        verts += offset[None, :]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
        meshes.append(mesh)
    return meshes


def build_joint_sequence(sequence: HumanMotionSequence) -> np.ndarray:
    import torch

    model, kwargs = _build_smpl_model(sequence)
    with torch.no_grad():
        model_out = model(**kwargs)
    return model_out.joints.detach().cpu().numpy().astype(np.float32)


def build_shape_neutral_geometry(
    sequence: HumanMotionSequence,
    *,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    neutral_sequence = HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=f"{sequence.sequence_name}_shape_neutral",
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender,
        betas=sequence.betas.copy(),
        poses=np.zeros((1, sequence.poses.shape[1]), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        image_names=[],
        cam_int=None,
        cam_ext=None,
        metadata={},
    )
    model, kwargs = _build_smpl_model(neutral_sequence, device=device)
    with torch.no_grad():
        model_out = model(**kwargs)
    vertices = model_out.vertices.detach().cpu().numpy().astype(np.float32)[0]
    joints = model_out.joints.detach().cpu().numpy().astype(np.float32)[0]
    return vertices, joints


def build_pose_neutral_template_geometry(
    sequence: HumanMotionSequence,
    *,
    device: str | None = None,
    betas_mode: str = "sequence",
    gender: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """SMPL mesh with pose=0 and trans=0; betas from sequence or zeros."""

    import torch

    mode = str(betas_mode).strip().lower()
    if mode not in ("sequence", "neutral"):
        raise ValueError(f"betas_mode must be 'sequence' or 'neutral', got {betas_mode!r}")
    if mode == "neutral":
        beta_dim = int(np.asarray(sequence.betas).reshape(-1).size)
        betas = np.zeros(beta_dim, dtype=np.float32)
    else:
        betas = sequence.betas.copy()
    neutral_sequence = HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=f"{sequence.sequence_name}_pose_neutral_template",
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender if gender is None else str(gender),
        betas=betas,
        poses=np.zeros((1, sequence.poses.shape[1]), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        image_names=[],
        cam_int=None,
        cam_ext=None,
        metadata={},
    )
    model, kwargs = _build_smpl_model(neutral_sequence, device=device)
    with torch.no_grad():
        model_out = model(**kwargs)
    vertices = model_out.vertices.detach().cpu().numpy().astype(np.float32)[0]
    joints = model_out.joints.detach().cpu().numpy().astype(np.float32)[0]
    skin_weights = getattr(model, "lbs_weights", None)
    if skin_weights is None:
        raise AttributeError("The SMPL model does not expose lbs_weights needed for body proxy sizing.")
    weights = skin_weights.detach().cpu().numpy().astype(np.float32) if hasattr(skin_weights, "detach") else np.asarray(skin_weights, dtype=np.float32)
    template_info = {
        "pose": "zero",
        "trans": "zero",
        "betas_mode": mode,
        "betas": [float(v) for v in np.asarray(betas, dtype=np.float32).reshape(-1).tolist()],
        "model_type": str(sequence.model_type),
        "source_gender": str(sequence.gender),
        "template_gender": str(neutral_sequence.gender),
    }
    return vertices, joints, weights, template_info


def build_shape_neutral_body_geometry(
    sequence: HumanMotionSequence,
    *,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, joints, weights, _ = build_pose_neutral_template_geometry(sequence, device=device, betas_mode="sequence")
    return vertices, joints, weights


def compute_genesis_matched_root_translation(
    sequence: HumanMotionSequence,
    *,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_floor: bool = True,
    root_joint_index: int = 0,
) -> np.ndarray:
    """World position of the SMPL/BiRP root joint each frame, matching ``build_trimesh_sequence``.

    Uses the **same** ``model_out`` forward as the debug mesh: for each frame we take
    ``joints[t, root_joint_index]`` (default pelvis), apply the same floor percentile on Z as
    vertices, then add ``world_offset``. NPZ ``trans`` alone is not used here so Genesis mesh
    centroids / pelvis and UE ``root_translation_world_m`` describe one geometric root.

    Parameters
    ----------
    root_joint_index
        Body-model root / pelvis joint index (0 for SMPL/SMPL-X body layouts in this repo).
    """
    import torch

    model, kwargs = _build_smpl_model(sequence)
    with torch.no_grad():
        model_out = model(**kwargs)
    vertices_seq = model_out.vertices.detach().cpu().numpy().astype(np.float32)
    joints_seq = model_out.joints.detach().cpu().numpy().astype(np.float32)
    off = np.asarray(world_offset, dtype=np.float32).reshape(3)
    n = int(joints_seq.shape[0])
    ji = int(max(min(int(root_joint_index), int(joints_seq.shape[1]) - 1), 0))
    out = np.zeros((n, 3), dtype=np.float32)
    for t in range(n):
        p = np.asarray(joints_seq[t, ji, :3], dtype=np.float32).copy()
        if align_floor:
            p[2] -= float(np.percentile(vertices_seq[t, :, 2], 2.0))
        p += off
        out[t] = p
    return out


def export_mesh_sequence(
    sequence: HumanMotionSequence,
    output_root: Path,
    *,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_floor: bool = True,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    meshes = build_trimesh_sequence(sequence, world_offset=world_offset, align_floor=align_floor)
    if not meshes:
        raise RuntimeError("No meshes generated for sequence export.")
    faces_path = output_root / "faces.npy"
    np.save(faces_path, meshes[0].faces.astype(np.int32))
    frames = []
    for frame_idx, mesh in enumerate(meshes):
        frame_stem = f"frame_{frame_idx:05d}"
        frame_path = output_root / f"{frame_stem}.npz"
        np.savez_compressed(frame_path, vertices_world=mesh.vertices.astype(np.float32))
        frames.append({"frame_idx": frame_idx, "frame_stem": frame_stem, "world_smpl_path": frame_path.name})
    manifest = {
        "sequence_name": sequence.sequence_name,
        "source_dataset": sequence.source_dataset,
        "model_type": sequence.model_type,
        "fps": sequence.fps,
        "frame_count": len(frames),
        "faces_path": faces_path.name,
        "frames": frames,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_mesh_sequence_from_manifest(manifest_path: Path) -> list[trimesh.Trimesh]:
    import trimesh

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    faces_path = manifest_path.parent / manifest["faces_path"]
    faces = np.load(faces_path).astype(np.int32)
    meshes: list[trimesh.Trimesh] = []
    for frame in manifest["frames"]:
        frame_path = manifest_path.parent / frame["world_smpl_path"]
        payload = np.load(frame_path)
        vertices = np.asarray(payload["vertices_world"], dtype=np.float32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = np.tile(np.asarray([184, 209, 245, 255], dtype=np.uint8), (len(mesh.vertices), 1))
        meshes.append(mesh)
    return meshes
