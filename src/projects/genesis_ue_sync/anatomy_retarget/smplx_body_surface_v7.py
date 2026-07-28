"""Pure-numpy SMPL-X body surface for V7 vessel/nerve containment gates.

Torch is not available in this environment, so the ``smplx`` package cannot be
used.  This module loads an offline ``SMPLX_*.pkl`` and runs the standard LBS
forward pass in float64 so containment can be measured against the exact
subject beta and pose without a second solver.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SMPLX_BODY_SURFACE_SCHEMA_VERSION = 7
_NUM_JOINTS = 55
_NUM_SHAPE_BETAS = 10


def _as_float64_array(value: Any, *, label: str) -> np.ndarray:
    # Chumpy / object wrappers appear in some SMPL-X pickles; force a numeric array.
    if hasattr(value, "r"):
        value = value.r
    if hasattr(value, "todense"):
        value = value.todense()
    array = np.asarray(value, dtype=np.float64)
    if array.dtype == object:
        array = np.asarray(array.tolist(), dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains a non-finite value")
    return array


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle < 1.0e-8:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    x, y, z = axis.tolist()
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def load_smplx_model_v7(model_path: Path | str) -> dict[str, np.ndarray]:
    """Load a chumpy-free SMPL-X pickle into the arrays needed for numpy LBS."""
    path = Path(model_path).expanduser().resolve()
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, Mapping):
        raise ValueError(f"SMPL-X model at {path.name} is not a mapping")

    missing = [
        name
        for name in (
            "v_template",
            "shapedirs",
            "posedirs",
            "J_regressor",
            "weights",
            "f",
            "kintree_table",
        )
        if name not in payload
    ]
    if missing:
        raise ValueError(
            f"SMPL-X model at {path.name} is missing required fields: {missing}"
        )

    v_template = _as_float64_array(payload["v_template"], label="v_template")
    if v_template.ndim != 2 or v_template.shape[1] != 3 or not len(v_template):
        raise ValueError(
            f"v_template must be [V,3], got {v_template.shape} in {path.name}"
        )
    vertex_count = int(v_template.shape[0])

    shapedirs = _as_float64_array(payload["shapedirs"], label="shapedirs")
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != (vertex_count, 3):
        raise ValueError(
            f"shapedirs must be [V,3,S], got {shapedirs.shape} in {path.name}"
        )
    if shapedirs.shape[2] < _NUM_SHAPE_BETAS:
        raise ValueError(
            f"shapedirs must provide at least {_NUM_SHAPE_BETAS} betas, "
            f"got {shapedirs.shape[2]} in {path.name}"
        )

    posedirs = _as_float64_array(payload["posedirs"], label="posedirs")
    pose_basis = 9 * (_NUM_JOINTS - 1)
    if posedirs.ndim == 3 and posedirs.shape == (vertex_count, 3, pose_basis):
        pass
    elif posedirs.ndim == 2 and posedirs.shape == (pose_basis, vertex_count * 3):
        posedirs = posedirs.T.reshape(vertex_count, 3, pose_basis)
    elif posedirs.ndim == 2 and posedirs.shape == (vertex_count * 3, pose_basis):
        posedirs = posedirs.reshape(vertex_count, 3, pose_basis)
    else:
        raise ValueError(
            f"posedirs must be [V,3,{pose_basis}] or [{pose_basis},V*3], "
            f"got {posedirs.shape} in {path.name}"
        )

    j_regressor = _as_float64_array(payload["J_regressor"], label="J_regressor")
    if j_regressor.shape != (_NUM_JOINTS, vertex_count):
        raise ValueError(
            f"J_regressor must be [{_NUM_JOINTS},V], got {j_regressor.shape} "
            f"in {path.name}"
        )

    weights = _as_float64_array(payload["weights"], label="weights")
    if weights.shape != (vertex_count, _NUM_JOINTS):
        raise ValueError(
            f"weights must be [V,{_NUM_JOINTS}], got {weights.shape} in {path.name}"
        )

    faces = np.asarray(payload["f"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"faces must be [F,3], got {faces.shape} in {path.name}")
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError(f"faces reference an invalid vertex in {path.name}")

    kintree = np.asarray(payload["kintree_table"], dtype=np.int64)
    if kintree.ndim != 2 or kintree.shape[1] != _NUM_JOINTS:
        raise ValueError(
            f"kintree_table must be [2,{_NUM_JOINTS}], got {kintree.shape} "
            f"in {path.name}"
        )
    parents = np.asarray(kintree[0], dtype=np.int64).copy()
    parents[0] = -1

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": j_regressor,
        "weights": weights,
        "kintree_parents": parents,
        "faces": faces,
        "model_path": str(path),
    }


def smplx_body_surface_v7(
    model: Mapping[str, np.ndarray],
    *,
    betas: Any,
    pose_axis_angle: Any,
    transl: Any | None = None,
    expression: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard SMPL-X LBS in float64.  Returns ``(vertices [V,3], faces [F,3])``."""
    required = (
        "v_template",
        "shapedirs",
        "posedirs",
        "J_regressor",
        "weights",
        "kintree_parents",
        "faces",
    )
    missing = [name for name in required if name not in model]
    if missing:
        raise ValueError(f"SMPL-X model dict is missing fields: {missing}")

    v_template = np.asarray(model["v_template"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    posedirs = np.asarray(model["posedirs"], dtype=np.float64)
    j_regressor = np.asarray(model["J_regressor"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    parents = np.asarray(model["kintree_parents"], dtype=np.int64).reshape(-1)
    faces = np.asarray(model["faces"], dtype=np.int32)

    if v_template.shape[1] != 3 or shapedirs.shape[:2] != v_template.shape:
        raise ValueError("model arrays have inconsistent vertex shapes")
    if parents.shape != (_NUM_JOINTS,) or j_regressor.shape[0] != _NUM_JOINTS:
        raise ValueError(f"model must describe exactly {_NUM_JOINTS} joints")
    if weights.shape != (len(v_template), _NUM_JOINTS):
        raise ValueError("weights shape does not match template vertices")

    beta = np.asarray(betas, dtype=np.float64).reshape(-1)
    if beta.size < _NUM_SHAPE_BETAS:
        raise ValueError(f"betas must provide at least {_NUM_SHAPE_BETAS} values")
    if not np.all(np.isfinite(beta[:_NUM_SHAPE_BETAS])):
        raise ValueError("betas contain a non-finite value")

    pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(_NUM_JOINTS, 3)
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose_axis_angle contains a non-finite value")

    shape = beta[:_NUM_SHAPE_BETAS]
    v_shaped = v_template + np.einsum(
        "vks,s->vk", shapedirs[:, :, :_NUM_SHAPE_BETAS], shape
    )
    if expression is not None:
        expr = np.asarray(expression, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(expr)):
            raise ValueError("expression contains a non-finite value")
        available = int(shapedirs.shape[2] - _NUM_SHAPE_BETAS)
        count = min(int(expr.size), available)
        if count:
            v_shaped = v_shaped + np.einsum(
                "vks,s->vk",
                shapedirs[:, :, _NUM_SHAPE_BETAS : _NUM_SHAPE_BETAS + count],
                expr[:count],
            )

    joints = j_regressor @ v_shaped
    rotations = np.stack(
        [_axis_angle_to_rotation(pose[index]) for index in range(_NUM_JOINTS)],
        axis=0,
    )
    pose_feature = (rotations[1:] - np.eye(3, dtype=np.float64)[None, :, :]).reshape(
        -1
    )
    if posedirs.shape[-1] != pose_feature.size:
        raise ValueError(
            f"posedirs last dimension {posedirs.shape[-1]} does not match "
            f"pose feature length {pose_feature.size}"
        )
    v_posed = v_shaped + np.einsum("vkp,p->vk", posedirs, pose_feature)

    transforms = np.zeros((_NUM_JOINTS, 4, 4), dtype=np.float64)
    for index in range(_NUM_JOINTS):
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = rotations[index]
        parent = int(parents[index])
        if parent < 0:
            local[:3, 3] = joints[index]
            transforms[index] = local
        else:
            local[:3, 3] = joints[index] - joints[parent]
            transforms[index] = transforms[parent] @ local

    relative = np.zeros((_NUM_JOINTS, 4, 4), dtype=np.float64)
    for index in range(_NUM_JOINTS):
        rest = np.eye(4, dtype=np.float64)
        rest[:3, 3] = joints[index]
        relative[index] = transforms[index] @ np.linalg.inv(rest)

    blended = np.einsum("vj,jab->vab", weights, relative)
    homogeneous = np.concatenate(
        (v_posed, np.ones((len(v_posed), 1), dtype=np.float64)),
        axis=1,
    )
    vertices = np.einsum("vab,vb->va", blended, homogeneous)[:, :3]
    if transl is not None:
        translation = np.asarray(transl, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(translation)):
            raise ValueError("transl contains a non-finite value")
        vertices = vertices + translation
    if not np.all(np.isfinite(vertices)):
        raise ValueError("SMPL-X forward pass produced non-finite vertices")
    return vertices.astype(np.float64, copy=False), faces.copy()


def _capture_pose55(capture_path: Path) -> np.ndarray:
    from .pose_adapter import load_easymocap_smplx_fit_drive

    pose55, _transl = load_easymocap_smplx_fit_drive(capture_path)
    return np.asarray(pose55, dtype=np.float64).reshape(_NUM_JOINTS, 3)


def _poses_match(first: np.ndarray, second: np.ndarray, *, atol: float = 1.0e-5) -> bool:
    a = np.asarray(first, dtype=np.float64).reshape(_NUM_JOINTS, 3)
    b = np.asarray(second, dtype=np.float64).reshape(_NUM_JOINTS, 3)
    return bool(np.allclose(a, b, atol=atol, rtol=0.0))


def body_surface_for_cell_v7(
    *,
    capture_result_path: Path | str | None = None,
    model_path: Path | str | None = None,
    betas: Any | None = None,
    pose_axis_angle: Any | None = None,
    transl: Any | None = None,
    expression: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the SMPL-X skin for one capture cell, preferring stored fit vertices."""
    if capture_result_path is not None:
        capture_path = Path(capture_result_path).expanduser().resolve()
        with np.load(capture_path) as data:
            has_surface = "vertices" in data.files and "faces" in data.files
            if has_surface:
                capture_vertices = np.asarray(data["vertices"], dtype=np.float64)
                capture_faces = np.asarray(data["faces"], dtype=np.int32)
                use_capture = pose_axis_angle is None
                if pose_axis_angle is not None:
                    try:
                        use_capture = _poses_match(
                            pose_axis_angle, _capture_pose55(capture_path)
                        )
                    except Exception:
                        use_capture = False
                if use_capture:
                    provenance = {
                        "source": "capture_fit_vertices",
                        "schema_version": SMPLX_BODY_SURFACE_SCHEMA_VERSION,
                        "capture_result_path": capture_path.name,
                        "vertex_count": int(len(capture_vertices)),
                        "face_count": int(len(capture_faces)),
                    }
                    return capture_vertices, capture_faces, provenance

    if model_path is None:
        raise ValueError(
            "numpy SMPL-X forward requires model_path when capture vertices "
            "are unavailable or the requested pose does not match the capture"
        )
    if betas is None or pose_axis_angle is None:
        raise ValueError(
            "numpy SMPL-X forward requires betas and pose_axis_angle when "
            "capture vertices are not used"
        )
    resolved_model = Path(model_path).expanduser().resolve()
    model = load_smplx_model_v7(resolved_model)
    vertices, faces = smplx_body_surface_v7(
        model,
        betas=betas,
        pose_axis_angle=pose_axis_angle,
        transl=transl,
        expression=expression,
    )
    provenance = {
        "source": "numpy_smplx_forward",
        "schema_version": SMPLX_BODY_SURFACE_SCHEMA_VERSION,
        "model_path": resolved_model.name,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }
    return vertices, faces, provenance


__all__ = [
    "SMPLX_BODY_SURFACE_SCHEMA_VERSION",
    "body_surface_for_cell_v7",
    "load_smplx_model_v7",
    "smplx_body_surface_v7",
]
