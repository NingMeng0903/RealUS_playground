"""Export subject-beta SMPL-X T-pose assets for anatomy retargeting."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    _build_smpl_kwargs,
    _create_smpl_model,
    resolve_torch_device,
)


SMPLX_JOINT_NAMES_55 = [
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
]


@dataclass(frozen=True)
class CanonicalExportResult:
    output_dir: Path
    subject_obj: Path
    neutral_obj: Path
    skeleton_json: Path
    weights_npz: Path
    manifest_json: Path


def load_betas(path_or_run_dir: Path | str) -> np.ndarray:
    path = Path(path_or_run_dir)
    if path.is_dir():
        candidates = [
            path / "beta_calibration" / "betas.npy",
            path / "betas.npy",
            path / "moment_0000" / "smplx_result.npz",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find betas file from {path_or_run_dir}")
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)[:10]
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        for key in ("betas", "shapes"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32).reshape(-1)[:10]
    raise ValueError(f"Unsupported betas input: {path}")


def _make_sequence(*, betas: np.ndarray, gender: str, pose_dim: int = 165) -> HumanMotionSequence:
    return HumanMotionSequence(
        source_dataset="anatomy_retarget",
        sequence_name="canonical_tpose",
        source_path="",
        model_type="smplx",
        fps=30.0,
        gender=str(gender).lower(),
        betas=np.asarray(betas, dtype=np.float32).reshape(-1),
        poses=np.zeros((1, int(pose_dim)), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        image_names=[],
        cam_int=None,
        cam_ext=None,
        metadata={},
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_smplx_pkl_model(*, gender: str) -> tuple[Path, str]:
    requested = str(gender).upper()
    repo = _repo_root()
    roots = [
        repo / "ref_code_library" / "EasyMocap" / "data" / "smplx",
        repo / "ref_code_library" / "InteractVLM" / "data" / "body_models",
        repo / "ref_code_library" / "HybrIK" / "model_files",
    ]
    for root in roots:
        for candidate_gender in (requested, "NEUTRAL", "MALE", "FEMALE"):
            candidate = root / "smplx" / f"SMPLX_{candidate_gender}.pkl"
            if candidate.is_file():
                return root, candidate_gender
    raise FileNotFoundError(
        "SMPL-X PKL models not found. Place SMPLX_MALE.pkl under ref_code_library/EasyMocap/data/smplx/smplx "
        "or provide SMPL-X NPZ models under dataset/intermediate/humans/body_models/smplx/models."
    )


def _forward_zero_pose_pkl_fallback(
    *,
    betas: np.ndarray,
    gender: str,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    import smplx
    import torch

    torch_device = resolve_torch_device(device)
    model_root, resolved_gender = _resolve_smplx_pkl_model(gender=gender)
    model = smplx.create(
        model_path=str(model_root),
        model_type="smplx",
        gender=resolved_gender,
        ext="pkl",
        use_pca=False,
        flat_hand_mean=True,
    ).to(torch_device)
    beta = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
    if beta.size < 10:
        beta = np.pad(beta, (0, 10 - beta.size))
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=torch_device)
    with torch.inference_mode():
        out = model(
            betas=torch.as_tensor(beta.reshape(1, 10), dtype=torch.float32, device=torch_device),
            global_orient=zeros3,
            body_pose=torch.zeros((1, 63), dtype=torch.float32, device=torch_device),
            left_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=torch_device),
            right_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=torch_device),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            transl=zeros3,
            return_verts=True,
        )
    vertices = out.vertices.detach().cpu().numpy().astype(np.float32)[0]
    joints = out.joints.detach().cpu().numpy().astype(np.float32)[0]
    faces = np.asarray(model.faces, dtype=np.int32)
    return vertices, joints, faces, model


def _forward_zero_pose(*, betas: np.ndarray, gender: str, device: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    import torch

    torch_device = resolve_torch_device(device)
    seq = _make_sequence(betas=betas, gender=gender)
    try:
        model = _create_smpl_model(seq, torch_device)
        kwargs = _build_smpl_kwargs(seq, torch_device=torch_device)
        with torch.inference_mode():
            out = model(**kwargs)
    except FileNotFoundError:
        return _forward_zero_pose_pkl_fallback(betas=betas, gender=gender, device=device)
    vertices = out.vertices.detach().cpu().numpy().astype(np.float32)[0]
    joints = out.joints.detach().cpu().numpy().astype(np.float32)[0]
    faces = np.asarray(model.faces, dtype=np.int32)
    return vertices, joints, faces, model


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# SMPL-X canonical T-pose exported by anatomy_retarget\n")
        for vertex in np.asarray(vertices, dtype=np.float32).reshape(-1, 3):
            handle.write(f"v {float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f}\n")
        for face in np.asarray(faces, dtype=np.int32).reshape(-1, 3):
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _global_bind_mats(rest_joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    joints = np.asarray(rest_joints, dtype=np.float32).reshape(-1, 3)
    pa = np.asarray(parents, dtype=np.int32).reshape(-1)
    out = np.tile(np.eye(4, dtype=np.float32), (joints.shape[0], 1, 1))
    for idx in range(joints.shape[0]):
        local = np.eye(4, dtype=np.float32)
        if idx == 0 or int(pa[idx]) < 0:
            local[:3, 3] = joints[idx]
            out[idx] = local
        else:
            parent = int(pa[idx])
            local[:3, 3] = joints[idx] - joints[parent]
            out[idx] = out[parent] @ local
    return out


def _model_arrays(model: Any, *, joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    weights = getattr(model, "lbs_weights", None)
    parents = getattr(model, "parents", None)
    if weights is None or parents is None:
        raise AttributeError("SMPL-X model must expose lbs_weights and parents")
    if hasattr(weights, "detach"):
        weights_np = weights.detach().cpu().numpy().astype(np.float32)
    else:
        weights_np = np.asarray(weights, dtype=np.float32)
    if hasattr(parents, "detach"):
        parents_np = parents.detach().cpu().numpy().astype(np.int32)
    else:
        parents_np = np.asarray(parents, dtype=np.int32)
    return weights_np[:, :joint_count], parents_np[:joint_count]


def _model_blendshape_arrays(model: Any) -> dict[str, np.ndarray]:
    """Extract neutral template-space shape/pose bases when available."""
    out: dict[str, np.ndarray] = {}
    for name in ("shapedirs", "posedirs", "v_template"):
        arr = getattr(model, name, None)
        if arr is None:
            continue
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        out[name] = np.asarray(arr, dtype=np.float32)
    return out


def export_canonical_tpose(
    *,
    betas: np.ndarray,
    output_dir: Path,
    gender: str = "male",
    device: str | None = "cpu",
    staging_dir: Path | None = None,
    source: str | None = None,
) -> CanonicalExportResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    beta_arr = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
    neutral_betas = np.zeros_like(beta_arr, dtype=np.float32)

    vertices_subject, joints_subject_all, faces, model = _forward_zero_pose(betas=beta_arr, gender=gender, device=device)
    vertices_neutral, joints_neutral_all, _faces2, _model2 = _forward_zero_pose(betas=neutral_betas, gender=gender, device=device)
    weights, parents = _model_arrays(model, joint_count=min(len(SMPLX_JOINT_NAMES_55), joints_subject_all.shape[0]))
    blendshape_arrays = _model_blendshape_arrays(model)
    joint_count = int(weights.shape[1])
    joint_names = SMPLX_JOINT_NAMES_55[:joint_count]
    joints_subject = np.asarray(joints_subject_all[:joint_count], dtype=np.float32)
    joints_neutral = np.asarray(joints_neutral_all[:joint_count], dtype=np.float32)
    bind = _global_bind_mats(joints_subject, parents)
    inverse_bind = np.linalg.inv(bind).astype(np.float32)

    subject_obj = out / "smpl_canonical_tpose.obj"
    neutral_obj = out / "smpl_canonical_tpose_neutral.obj"
    skeleton_json = out / "smpl_canonical_skeleton.json"
    weights_npz = out / "smpl_canonical_weights.npz"
    manifest_json = out / "source_manifest.json"

    _write_obj(subject_obj, vertices_subject, faces)
    _write_obj(neutral_obj, vertices_neutral, faces)
    skeleton_payload = {
        "model_type": "smplx",
        "gender": str(gender).lower(),
        "joint_names": joint_names,
        "parents": [int(v) for v in parents.tolist()],
        "rest_joints_subject": joints_subject.astype(float).tolist(),
        "rest_joints_neutral": joints_neutral.astype(float).tolist(),
        "inverse_bind": inverse_bind.astype(float).tolist(),
    }
    skeleton_json.write_text(json.dumps(skeleton_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    np.savez_compressed(
        weights_npz,
        lbs_weights=weights.astype(np.float32),
        faces=faces.astype(np.int32),
        joint_names=np.asarray(joint_names, dtype=object),
        parents=parents.astype(np.int32),
        rest_joints=joints_subject.astype(np.float32),
        inverse_bind=inverse_bind.astype(np.float32),
        **blendshape_arrays,
    )
    manifest = {
        "source": str(source or ""),
        "gender": str(gender).lower(),
        "betas": [float(v) for v in beta_arr.tolist()],
        "subject_obj": str(subject_obj),
        "neutral_obj": str(neutral_obj),
        "skeleton_json": str(skeleton_json),
        "weights_npz": str(weights_npz),
        "has_shapedirs": bool("shapedirs" in blendshape_arrays),
        "has_posedirs": bool("posedirs" in blendshape_arrays),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    result = CanonicalExportResult(
        output_dir=out,
        subject_obj=subject_obj,
        neutral_obj=neutral_obj,
        skeleton_json=skeleton_json,
        weights_npz=weights_npz,
        manifest_json=manifest_json,
    )
    if staging_dir is not None:
        stage = Path(staging_dir)
        stage.mkdir(parents=True, exist_ok=True)
        for src in (subject_obj, neutral_obj, skeleton_json, weights_npz, manifest_json):
            shutil.copy2(src, stage / src.name)
    return result

