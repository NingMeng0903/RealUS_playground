"""Independent checker for the Blender bone/tube linkage oracle.

This module consumes only Blender-evaluated matrices/vertices and frozen V8
source arrays.  It never reads candidate acceptance flags and contains no
retargeting logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .tube_frames_v8 import tube_coupling_pack_from_runtime_fields_v8
from .v8_artifacts import load_source_operator


ORACLE_SCHEMA_VERSION = 71
EXPECTED_BLEND_SHA256 = (
    "34945b610c9efbbd40b07bacd2933e0586264f06d8413e1f6ffd8e2b98a7b67c"
)
EXPECTED_BLENDER_VERSION = "4.5.8 LTS"
EXPECTED_SOURCE_COMMIT = "142ece5f0bc646978ae3e8c9add76deea71c26a2"
EXPECTED_ORACLE_SHA256 = (
    "60bf4c3f7803b62b2113fe2715e9b53b35d16caf35410ce4bd1f9b9c47e8dd3d"
)
EXPECTED_ORACLE_REPORT_SHA256 = (
    "d1bb299e4aa5069c88e95d8f61556dd75cdf5de402c3ef481fdee7ded1885850"
)
EXPECTED_OPERATOR_MANIFEST_SHA256 = (
    "a53455ec2c60c60c5b522b8a4ee78d5596abba86494cff42c51f1ce84c3d332e"
)
EXPECTED_OPERATOR_RUNTIME_DIGEST = (
    "17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b"
)
EXPECTED_OPERATOR_RIG_ARRAY_SHA256 = {
    "parents": "86fb3ec34b848902f589707aed146fc91c3e143f368d8418bd8cb01f7607a00e",
    "rest_global": "65bef55136b7f160500df319fc9b991795628084b4e8a6181dd4736065c58286",
    "rest_local": "9d75a97b5393786e9529b6c7f8379f14cb2fc629ecee5c04da827dbeac5b00ec",
    "use_connect": "1c6f4df4243803c0c3b60216f37217212d468f6d8dcc99209ff85b6bc26c2010",
    "inherit_scale": "22b94c6893bfc091be2a9f454a045184df6c0398cffa2b4e90c0065dd6eeb1b0",
}
EXPECTED_ACTION_FRAMES = np.arange(0, 271, dtype=np.int32)
EXPECTED_MESH_FRAMES = np.unique(
    np.concatenate(
        (np.arange(0, 271, 15, dtype=np.int32), np.asarray((250, 260), dtype=np.int32))
    )
)
EXPECTED_TUBE_DIGESTS = {
    "topology": "765293284200c8d3a88204ce71c547aa767544092d1246ef02fd9a56ddf33ff5",
    "domain": "1e99d47507868fd6e5aa8394d6454147639607a507338d12ac4181a9bec317a0",
    "weight": "9e7e2f6ad8f9f451405fddcf01970b4b2dde588ecf18c72e083273215acd64ff",
}
KEY_BONE_MESHES = (
    "Ilium_L", "Ilium_R", "Femur_L", "Femur_R", "Tibia_L", "Tibia_R",
    "Patella_L", "Patella_R", "Humerus_L", "Humerus_R", "Radius_L",
    "Radius_R", "Ulna_L", "Ulna_R",
)
TUBE_MESHES = (
    "Artery", "Vein", "Autonomic", "Cervical_Nerves_L",
    "Cervical_Nerves_R", "Coccygeal_Nerve_L", "Coccygeal_Nerve_R",
    "Facial_Nerves_L", "Facial_Nerves_R", "Lumbar_Nerves_L",
    "Lumbar_Nerves_R", "Optic_Chiasm", "Sacral_Nerves_L",
    "Sacral_Nerves_R", "Spinal_Cord", "Thoracic_Nerves_L",
    "Thoracic_Nerves_R",
)
EXPECTED_MESHES = KEY_BONE_MESHES + TUBE_MESHES


def _key(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _frozen_operator_contract(operator: Any, operator_path: Path) -> dict[str, Any]:
    """Authenticate the 142 operator independently of the raw Blender rest frame.

    The oracle rest matrices are the authored armature-object-space V71 frames;
    the 142 operator stores the later canonical metric retarget frame.  They are
    intentionally different coordinate products and must not be compared
    elementwise.  Instead, freeze both products: the oracle sidecar authenticates
    the raw Blender source, while these digests authenticate the complete 142
    runtime rig, including Blender's connect/inherit-scale semantics.
    """

    manifest = Path(operator_path) / "manifest.json"
    if not manifest.is_file():
        raise ValueError("frozen 142 operator manifest is missing")
    asset = operator.template_asset
    arrays = {
        "parents": np.asarray(asset.source_bone_parents, dtype=np.int32),
        "rest_global": np.asarray(asset.source_rest_global, dtype=np.float32),
        "rest_local": np.asarray(asset.source_rest_local, dtype=np.float32),
        "use_connect": np.asarray(asset.source_bone_use_connect, dtype=np.uint8),
        "inherit_scale": np.asarray(
            asset.source_bone_inherit_scale, dtype=np.uint8
        ),
    }
    hashes = {name: _array_sha256(value) for name, value in arrays.items()}
    manifest_sha256 = _sha256(manifest)
    runtime_digest = str(operator.runtime_digest(validate=False))
    passed = bool(
        manifest_sha256 == EXPECTED_OPERATOR_MANIFEST_SHA256
        and runtime_digest == EXPECTED_OPERATOR_RUNTIME_DIGEST
        and hashes == EXPECTED_OPERATOR_RIG_ARRAY_SHA256
    )
    if not passed:
        raise ValueError("frozen 142 operator rig contract changed")
    return {
        "pass": True,
        "manifest_sha256": manifest_sha256,
        "runtime_digest": runtime_digest,
        "bone_count": int(len(arrays["parents"])),
        "array_sha256": hashes,
        "raw_blender_rest_is_distinct": True,
    }


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _polar_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    result = u @ vt
    if float(np.linalg.det(result)) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64).reshape(-1, 3, 3)
    b = np.asarray(second, dtype=np.float64).reshape(-1, 3, 3)
    values = np.empty(len(a), dtype=np.float64)
    for index, (ra, rb) in enumerate(zip(a, b)):
        delta = _polar_rotation(ra).T @ _polar_rotation(rb)
        cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        values[index] = math.degrees(math.acos(cosine))
    return values


def _mesh_rows(asset: Any, name: str) -> tuple[int, int, np.ndarray]:
    names = list(asset.source_mesh_names or ())
    if name not in names:
        raise ValueError(f"frozen operator is missing oracle mesh {name!r}")
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[
        names.index(name)
    ]
    faces = np.asarray(asset.faces, dtype=np.int64)
    rows = np.flatnonzero(
        np.all((faces >= int(start)) & (faces < int(stop)), axis=1)
    )
    return int(start), int(stop), faces[rows] - int(start)


def _matrix_metrics(
    *,
    global_matrices: np.ndarray,
    local_matrices: np.ndarray,
    parents: np.ndarray,
    unit_scale_m: float,
) -> dict[str, Any]:
    expected = np.empty_like(global_matrices, dtype=np.float64)
    for frame in range(len(global_matrices)):
        for bone, parent in enumerate(parents.tolist()):
            expected[frame, bone] = (
                local_matrices[frame, bone]
                if parent < 0
                else expected[frame, parent] @ local_matrices[frame, bone]
            )
    delta_translation = np.linalg.norm(
        expected[..., :3, 3] - global_matrices[..., :3, 3], axis=-1
    ) * float(unit_scale_m)
    rotation = _rotation_error_deg(
        expected[..., :3, :3], global_matrices[..., :3, :3]
    ).reshape(delta_translation.shape)
    matrix_max = float(np.max(np.abs(expected - global_matrices)))
    row_error = float(
        max(
            np.max(
                np.abs(
                    global_matrices[..., 3, :]
                    - np.asarray((0.0, 0.0, 0.0, 1.0))
                )
            ),
            np.max(
                np.abs(
                    local_matrices[..., 3, :]
                    - np.asarray((0.0, 0.0, 0.0, 1.0))
                )
            ),
        )
    )
    passed = bool(
        float(np.sqrt(np.mean(delta_translation * delta_translation))) <= 1.0e-6
        and float(np.max(delta_translation)) <= 1.0e-5
        and float(np.sqrt(np.mean(rotation * rotation))) <= 1.0e-4
        and float(np.max(rotation)) <= 1.0e-3
        and row_error <= 1.0e-6
    )
    return {
        "available": True,
        "pass": passed,
        "translation_rms_m": float(
            np.sqrt(np.mean(delta_translation * delta_translation))
        ),
        "translation_max_m": float(np.max(delta_translation)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation * rotation))),
        "rotation_max_deg": float(np.max(rotation)),
        "matrix_element_max": matrix_max,
        "homogeneous_row_max_error": row_error,
        "thresholds": {
            "translation_rms_m": 1.0e-6,
            "translation_max_m": 1.0e-5,
            "rotation_rms_deg": 1.0e-4,
            "rotation_max_deg": 1.0e-3,
        },
    }


def _basis_fk_metrics(
    *,
    global_matrices: np.ndarray,
    basis_matrices: np.ndarray,
    rest_global: np.ndarray,
    rest_local: np.ndarray,
    parents: np.ndarray,
    unit_scale_m: float,
) -> dict[str, Any]:
    """Rebuild Blender's ordinary FULL-inherit pose FK from frozen inputs.

    Unlike ``_matrix_metrics``, this does not consume child matrices derived
    from the sampled globals.  It starts from bind-local transforms and the
    authored Action ``matrix_basis`` channels, so it is the independent gate
    that proves Blender can be removed from pose-time execution for this rig.
    """

    expected = np.empty_like(global_matrices, dtype=np.float64)
    for frame in range(len(global_matrices)):
        for bone, parent in enumerate(parents.tolist()):
            if parent < 0:
                expected[frame, bone] = rest_global[bone] @ basis_matrices[frame, bone]
            else:
                expected[frame, bone] = (
                    expected[frame, parent]
                    @ rest_local[bone]
                    @ basis_matrices[frame, bone]
                )
    delta_translation = np.linalg.norm(
        expected[..., :3, 3] - global_matrices[..., :3, 3], axis=-1
    ) * float(unit_scale_m)
    rotation = _rotation_error_deg(
        expected[..., :3, :3], global_matrices[..., :3, :3]
    ).reshape(delta_translation.shape)
    scale_shear_max = float(
        np.max(np.abs(expected[..., :3, :3] - global_matrices[..., :3, :3]))
    )
    translation_rms = float(np.sqrt(np.mean(delta_translation * delta_translation)))
    rotation_rms = float(np.sqrt(np.mean(rotation * rotation)))
    passed = bool(
        translation_rms <= 1.0e-6
        and float(np.max(delta_translation)) <= 1.0e-5
        and rotation_rms <= 1.0e-4
        and float(np.max(rotation)) <= 1.0e-3
        and scale_shear_max <= 1.0e-5
    )
    return {
        "available": True,
        "pass": passed,
        "translation_rms_m": translation_rms,
        "translation_max_m": float(np.max(delta_translation)),
        "rotation_rms_deg": rotation_rms,
        "rotation_max_deg": float(np.max(rotation)),
        "scale_shear_element_max": scale_shear_max,
        "thresholds": {
            "translation_rms_m": 1.0e-6,
            "translation_max_m": 1.0e-5,
            "rotation_rms_deg": 1.0e-4,
            "rotation_max_deg": 1.0e-3,
            "scale_shear_element_max": 1.0e-5,
        },
    }


def _lbs_mesh_metrics(
    data: Any,
    *,
    name: str,
    rest_global: np.ndarray,
    action_global: np.ndarray,
    action_frames: np.ndarray,
    mesh_frames: np.ndarray,
    parents: np.ndarray,
    unit_scale_m: float,
) -> dict[str, Any]:
    prefix = f"mesh__{_key(name)}"
    rest = np.asarray(data[f"{prefix}__bind_vertices"], dtype=np.float64)
    expected_frames = np.asarray(data[f"{prefix}__vertices"], dtype=np.float32)
    indices = np.asarray(data[f"{prefix}__driver_indices"], dtype=np.int64)
    weights = np.asarray(data[f"{prefix}__driver_weights"], dtype=np.float64)
    frame_lookup = {int(frame): index for index, frame in enumerate(action_frames.tolist())}
    try:
        sample_indices = np.asarray(
            [frame_lookup[int(frame)] for frame in mesh_frames.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("mesh parity frame is absent from Action matrices") from error
    sampled_action_global = np.asarray(action_global[sample_indices], dtype=np.float64)
    if (
        expected_frames.shape[1:] != rest.shape
        or len(expected_frames) != len(mesh_frames)
        or indices.shape != weights.shape
        or indices.shape != (len(rest), 14)
        or np.any(indices < 0)
        or np.any(indices >= len(rest_global))
    ):
        raise ValueError(f"oracle mesh {name!r} has inconsistent arrays")
    inverse = np.linalg.inv(np.asarray(rest_global, dtype=np.float64))
    per_frame: list[dict[str, Any]] = []
    sum_squared = 0.0
    sample_count = 0
    maximum = 0.0
    displacement_max = 0.0
    root_relative_displacement_max = 0.0
    reference = np.asarray(expected_frames[0], dtype=np.float64)
    roots = np.flatnonzero(np.asarray(parents, dtype=np.int64) < 0)
    if len(roots) != 1:
        raise ValueError("oracle linkage check requires exactly one rig root")
    root_index = int(roots[0])
    root_inverse_bind = np.linalg.inv(np.asarray(rest_global[root_index], dtype=np.float64))
    first_root_skin = sampled_action_global[0, root_index] @ root_inverse_bind
    first_unroot = (
        np.einsum("ij,nj->ni", np.linalg.inv(first_root_skin)[:3, :3], reference)
        + np.linalg.inv(first_root_skin)[:3, 3]
    )
    for frame_index, global_frame in enumerate(sampled_action_global):
        transforms = np.asarray(global_frame, dtype=np.float64) @ inverse
        posed = np.zeros_like(rest, dtype=np.float64)
        for slot in range(indices.shape[1]):
            selected = transforms[indices[:, slot]]
            transformed = (
                np.einsum("nij,nj->ni", selected[:, :3, :3], rest)
                + selected[:, :3, 3]
            )
            posed += weights[:, slot : slot + 1] * transformed
        error_m = np.linalg.norm(
            posed - np.asarray(expected_frames[frame_index], dtype=np.float64),
            axis=1,
        ) * float(unit_scale_m)
        rms_m = float(np.sqrt(np.mean(error_m * error_m)))
        q99_m = float(np.quantile(error_m, 0.99))
        max_m = float(np.max(error_m))
        frame_pass = bool(
            rms_m <= 1.0e-5 and q99_m <= 5.0e-5 and max_m <= 2.0e-4
        )
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "frame": int(mesh_frames[frame_index]),
                "rms_m": rms_m,
                "q99_m": q99_m,
                "max_m": max_m,
                "pass": frame_pass,
            }
        )
        sum_squared += float(np.sum(error_m * error_m))
        sample_count += int(len(error_m))
        maximum = max(maximum, max_m)
        displacement_max = max(
            displacement_max,
            float(
                np.max(
                    np.linalg.norm(
                        np.asarray(expected_frames[frame_index], dtype=np.float64)
                        - reference,
                        axis=1,
                    )
                )
                * float(unit_scale_m)
            ),
        )
        root_skin = np.asarray(global_frame[root_index], dtype=np.float64) @ root_inverse_bind
        inverse_root_skin = np.linalg.inv(root_skin)
        unroot = (
            np.einsum(
                "ij,nj->ni",
                inverse_root_skin[:3, :3],
                np.asarray(expected_frames[frame_index], dtype=np.float64),
            )
            + inverse_root_skin[:3, 3]
        )
        root_relative_displacement_max = max(
            root_relative_displacement_max,
            float(np.max(np.linalg.norm(unroot - first_unroot, axis=1)))
            * float(unit_scale_m),
        )
    failures = [item["frame"] for item in per_frame if not item["pass"]]
    influenced_bones = np.unique(indices[weights > 0.0]).astype(np.int64)
    root_inverse_pose = np.linalg.inv(
        np.asarray(action_global[:, root_index], dtype=np.float64)
    )
    relative_pose = root_inverse_pose[:, None, :, :] @ np.asarray(
        action_global[:, influenced_bones], dtype=np.float64
    )
    relative_pose_change = float(np.max(np.abs(relative_pose - relative_pose[0:1])))
    relative_translation_m = float(
        np.max(
            np.linalg.norm(
                relative_pose[..., :3, 3] - relative_pose[0:1, ..., :3, 3],
                axis=-1,
            )
        )
        * float(unit_scale_m)
    )
    relative_rotation_deg = float(
        np.max(
            _rotation_error_deg(
                relative_pose[..., :3, :3],
                np.broadcast_to(
                    relative_pose[0:1, ..., :3, :3],
                    relative_pose[..., :3, :3].shape,
                ),
            )
        )
    )
    relative_linear_element_max = float(
        np.max(
            np.abs(
                relative_pose[..., :3, :3]
                - relative_pose[0:1, ..., :3, :3]
            )
        )
    )
    expected_non_root_dynamic = bool(
        relative_translation_m > 1.0e-5
        or relative_rotation_deg > 1.0e-3
        or relative_linear_element_max > 1.0e-5
    )
    return {
        "available": True,
        "pass": not failures,
        "vertex_count": int(len(rest)),
        "frame_count": int(len(mesh_frames)),
        "rms_m": float(np.sqrt(sum_squared / max(sample_count, 1))),
        "maximum_frame_q99_m": float(max(item["q99_m"] for item in per_frame)),
        "max_m": maximum,
        "maximum_action_displacement_m": displacement_max,
        "non_static": bool(displacement_max > 1.0e-6),
        "maximum_root_relative_displacement_m": root_relative_displacement_max,
        "non_root_dynamic": bool(root_relative_displacement_max > 1.0e-6),
        "influenced_bone_count": int(len(influenced_bones)),
        "influenced_bone_relative_pose_element_max": relative_pose_change,
        "influenced_bone_relative_translation_max_m": relative_translation_m,
        "influenced_bone_relative_rotation_max_deg": relative_rotation_deg,
        "influenced_bone_relative_linear_element_max": relative_linear_element_max,
        "expected_non_root_dynamic": expected_non_root_dynamic,
        "expected_dynamic_but_static": bool(
            expected_non_root_dynamic and root_relative_displacement_max <= 1.0e-6
        ),
        "failed_frame_indices": failures,
        "frames": per_frame,
        "thresholds": {"rms_m": 1.0e-5, "q99_m": 5.0e-5, "max_m": 2.0e-4},
    }


def _aggregate_mesh_metrics(
    metrics: Mapping[str, Mapping[str, Any]], names: tuple[str, ...]
) -> dict[str, Any]:
    samples = sum(
        int(metrics[name]["vertex_count"]) * int(metrics[name]["frame_count"])
        for name in names
    )
    squared = sum(
        float(metrics[name]["rms_m"]) ** 2
        * int(metrics[name]["vertex_count"])
        * int(metrics[name]["frame_count"])
        for name in names
    )
    return {
        "pass": bool(all(bool(metrics[name]["pass"]) for name in names)),
        "mesh_count": int(len(names)),
        "vertex_frame_sample_count": int(samples),
        "rms_m": float(np.sqrt(squared / max(samples, 1))),
        "maximum_frame_q99_m": float(
            max(float(metrics[name]["maximum_frame_q99_m"]) for name in names)
        ),
        "max_m": float(max(float(metrics[name]["max_m"]) for name in names)),
    }


def check_blender_link_oracle_v7(
    *,
    oracle_npz: Path | str,
    oracle_report: Path | str,
    operator_path: Path | str,
    require_full_action: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    npz_path = Path(oracle_npz).resolve()
    report_path = Path(oracle_report).resolve()
    sidecar = json.loads(report_path.read_text(encoding="utf-8"))
    if _sha256(report_path) != EXPECTED_ORACLE_REPORT_SHA256:
        raise ValueError("Blender linkage oracle sidecar differs from the frozen report")
    if int(sidecar.get("schema_version", -1)) != ORACLE_SCHEMA_VERSION:
        raise ValueError("unsupported Blender linkage oracle schema")
    if sidecar.get("source_blend_sha256") != EXPECTED_BLEND_SHA256:
        raise ValueError("Blender linkage oracle source .blend digest changed")
    if sidecar.get("blender_version") != EXPECTED_BLENDER_VERSION:
        raise ValueError("Blender linkage oracle version changed")
    if sidecar.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("Blender linkage oracle source commit changed")
    if sidecar.get("artifact_sha256") != _sha256(npz_path):
        raise ValueError("Blender linkage oracle artifact digest mismatch")
    if _sha256(npz_path) != EXPECTED_ORACLE_SHA256:
        raise ValueError("Blender linkage oracle differs from the frozen full oracle")
    resolved_operator = Path(operator_path).resolve()
    operator = load_source_operator(resolved_operator, mmap=True)
    operator_contract = _frozen_operator_contract(operator, resolved_operator)
    asset = operator.template_asset
    with np.load(npz_path, allow_pickle=False) as data:
        if int(np.asarray(data["schema_version"]).reshape(-1)[0]) != ORACLE_SCHEMA_VERSION:
            raise ValueError("oracle NPZ schema disagrees with sidecar")
        names = tuple(str(value) for value in np.asarray(data["mesh_names"]).tolist())
        if names != EXPECTED_MESHES:
            raise ValueError("oracle does not contain the fixed bone+tube mesh list")
        bone_names = tuple(str(value) for value in np.asarray(data["bone_names"]).tolist())
        parents = np.asarray(data["bone_parents"], dtype=np.int32)
        if bone_names != tuple(asset.source_bone_names or ()):
            raise ValueError("oracle bone names differ from frozen 142")
        if not np.array_equal(parents, np.asarray(asset.source_bone_parents, dtype=np.int32)):
            raise ValueError("oracle bone parents differ from frozen 142")
        frames = np.asarray(data["frames"], dtype=np.int32)
        mesh_frames = np.asarray(data["mesh_frames"], dtype=np.int32)
        rest_global = np.asarray(data["bone_rest_global"], dtype=np.float64)
        rest_local = np.asarray(data["bone_rest_local"], dtype=np.float64)
        action_global = np.asarray(data["bone_action_global"], dtype=np.float64)
        action_local = np.asarray(data["bone_action_local"], dtype=np.float64)
        action_basis = np.asarray(data["bone_action_basis"], dtype=np.float64)
        unit_scale_m = float(np.asarray(data["unit_scale_m"]).reshape(-1)[0])
        if (
            rest_global.shape != (235, 4, 4)
            or rest_local.shape != rest_global.shape
            or action_global.shape != (len(frames), 235, 4, 4)
            or action_local.shape != action_global.shape
            or action_basis.shape != action_global.shape
            or not np.isclose(unit_scale_m, 0.01, atol=0.0, rtol=0.0)
        ):
            raise ValueError("oracle FK arrays or unit scale are invalid")
        frame_coverage = {
            "pass": bool(
                np.array_equal(frames, EXPECTED_ACTION_FRAMES)
                and np.array_equal(mesh_frames, EXPECTED_MESH_FRAMES)
            ),
            "required": bool(require_full_action),
            "sampled_frame_count": int(len(frames)),
            "expected_frame_count": int(len(EXPECTED_ACTION_FRAMES)),
            "mesh_sampled_frame_count": int(len(mesh_frames)),
            "expected_mesh_sampled_frame_count": int(len(EXPECTED_MESH_FRAMES)),
            "first": None if not len(frames) else int(frames[0]),
            "last": None if not len(frames) else int(frames[-1]),
        }
        topology: dict[str, Any] = {}
        total_tube_vertices = 0
        for name in names:
            prefix = f"mesh__{_key(name)}"
            rest = np.asarray(data[f"{prefix}__rest_vertices"])
            faces = np.asarray(data[f"{prefix}__faces"], dtype=np.int64)
            indices = np.asarray(data[f"{prefix}__driver_indices"], dtype=np.int16)
            weights = np.asarray(data[f"{prefix}__driver_weights"], dtype=np.float32)
            start, stop, frozen_faces = _mesh_rows(asset, name)
            topology[name] = {
                "vertex_count": int(len(rest)),
                "frozen_vertex_count": int(stop - start),
                "faces_exact": bool(np.array_equal(faces, frozen_faces)),
                "indices_exact": bool(
                    np.array_equal(indices, np.asarray(asset.driver_indices)[start:stop])
                ),
                "weights_max_abs": float(
                    np.max(
                        np.abs(
                            weights
                            - np.asarray(asset.driver_weights, dtype=np.float32)[start:stop]
                        )
                    )
                ),
            }
            topology[name]["pass"] = bool(
                len(rest) == stop - start
                and topology[name]["faces_exact"]
                and topology[name]["indices_exact"]
                and topology[name]["weights_max_abs"] <= 1.0e-6
            )
            if name in TUBE_MESHES:
                total_tube_vertices += int(len(rest))
        tube_pack = tube_coupling_pack_from_runtime_fields_v8(
            operator.runtime_coefficients
        )
        coverage = {
            "pass": bool(
                total_tube_vertices == 55337
                and len(tube_pack.material_edges) == 165659
                and tube_pack.topology_digest == EXPECTED_TUBE_DIGESTS["topology"]
                and tube_pack.domain_digest == EXPECTED_TUBE_DIGESTS["domain"]
                and tube_pack.weight_digest == EXPECTED_TUBE_DIGESTS["weight"]
                and all(topology[name]["pass"] for name in names)
            ),
            "tube_mesh_count": int(len(TUBE_MESHES)),
            "tube_vertex_count": int(total_tube_vertices),
            "tube_material_edge_count": int(len(tube_pack.material_edges)),
            "tube_topology_digest": tube_pack.topology_digest,
            "tube_domain_digest": tube_pack.domain_digest,
            "tube_weight_digest": tube_pack.weight_digest,
            "meshes": topology,
        }
        fk = _matrix_metrics(
            global_matrices=action_global,
            local_matrices=action_local,
            parents=parents,
            unit_scale_m=unit_scale_m,
        )
        basis_fk = _basis_fk_metrics(
            global_matrices=action_global,
            basis_matrices=action_basis,
            rest_global=rest_global,
            rest_local=rest_local,
            parents=parents,
            unit_scale_m=unit_scale_m,
        )
        neutral_global = np.asarray(data["bone_neutral_global"], dtype=np.float64)
        neutral_delta = neutral_global @ np.linalg.inv(rest_global)
        neutral_matrix_error = float(
            np.max(
                np.abs(
                    neutral_delta
                    - np.eye(4, dtype=np.float64)[None, :, :]
                )
            )
        )
        neutral_mesh_error = 0.0
        raw_bind_offsets: dict[str, dict[str, float]] = {}
        for name in names:
            prefix = f"mesh__{_key(name)}"
            raw = np.asarray(data[f"{prefix}__rest_vertices"], dtype=np.float64)
            bind = np.asarray(data[f"{prefix}__bind_vertices"], dtype=np.float64)
            if raw.shape != bind.shape:
                raise ValueError(f"oracle mesh {name!r} raw/bind shape changed")
            offset_m = np.linalg.norm(bind - raw, axis=1) * unit_scale_m
            raw_bind_offsets[name] = {
                "rms_m": float(np.sqrt(np.mean(offset_m * offset_m))),
                "max_m": float(np.max(offset_m)),
            }
            indices = np.asarray(data[f"{prefix}__driver_indices"], dtype=np.int64)
            weights = np.asarray(data[f"{prefix}__driver_weights"], dtype=np.float64)
            neutral_posed = np.zeros_like(bind)
            for slot in range(indices.shape[1]):
                selected = neutral_delta[indices[:, slot]]
                transformed = (
                    np.einsum("nij,nj->ni", selected[:, :3, :3], bind)
                    + selected[:, :3, 3]
                )
                neutral_posed += weights[:, slot : slot + 1] * transformed
            neutral_mesh_error = max(
                neutral_mesh_error,
                float(np.max(np.linalg.norm(neutral_posed - bind, axis=1)))
                * unit_scale_m,
            )
        neutral = {
            "pass": bool(
                neutral_matrix_error <= 1.0e-6
                and neutral_mesh_error <= 1.0e-6
            ),
            "bone_transform_max_error": neutral_matrix_error,
            "vertex_max_error_m": neutral_mesh_error,
            "raw_to_bind_offsets": raw_bind_offsets,
        }
        mesh_metrics = {
            name: _lbs_mesh_metrics(
                data,
                name=name,
                rest_global=rest_global,
                action_global=action_global,
                action_frames=frames,
                mesh_frames=mesh_frames,
                parents=parents,
                unit_scale_m=unit_scale_m,
            )
            for name in names
        }
    non_static_tubes = [
        name for name in TUBE_MESHES if mesh_metrics[name]["non_static"]
    ]
    non_root_dynamic_tubes = [
        name for name in TUBE_MESHES if mesh_metrics[name]["non_root_dynamic"]
    ]
    linkage = {
        "pass": bool(
            all(value["pass"] for value in mesh_metrics.values())
            and mesh_metrics["Artery"]["non_static"]
            and mesh_metrics["Vein"]["non_static"]
            and not any(
                mesh_metrics[name]["expected_dynamic_but_static"]
                for name in TUBE_MESHES
            )
        ),
        "non_static_tube_count": int(len(non_static_tubes)),
        "non_root_dynamic_tube_count": int(len(non_root_dynamic_tubes)),
        "categories": {
            "representative_hard_bone": _aggregate_mesh_metrics(
                mesh_metrics, KEY_BONE_MESHES
            ),
            "vessel": _aggregate_mesh_metrics(mesh_metrics, ("Artery", "Vein")),
            "nerve": _aggregate_mesh_metrics(
                mesh_metrics,
                tuple(name for name in TUBE_MESHES if name not in {"Artery", "Vein"}),
            ),
        },
        "meshes": mesh_metrics,
    }
    elapsed = float(time.perf_counter() - started)
    maximum_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    semantic_pass = bool(
        operator_contract["pass"]
        and coverage["pass"]
        and fk["pass"]
        and basis_fk["pass"]
        and neutral["pass"]
        and linkage["pass"]
        and (frame_coverage["pass"] or not require_full_action)
    )
    blender_sampler_seconds = float(sidecar.get("elapsed_seconds", float("inf")))
    end_to_end_seconds = blender_sampler_seconds + elapsed
    performance_pass = bool(end_to_end_seconds <= 120.0)
    passed = bool(semantic_pass and performance_pass)
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "artifact_kind": "BlenderLinkOracleParityV7",
        "passed": passed,
        "source_blend_sha256": EXPECTED_BLEND_SHA256,
        "oracle_npz": str(npz_path),
        "oracle_sha256": _sha256(npz_path),
        "operator": str(Path(operator_path).resolve()),
        "operator_contract": operator_contract,
        "frame_count": int(len(frames)),
        "frame_coverage": frame_coverage,
        "fk": fk,
        "basis_fk": basis_fk,
        "neutral": neutral,
        "coverage": coverage,
        "linkage": linkage,
        "smplx_mapping_available": False,
        "claim": "raw_blender_action_basis_fk_and_armature_lbs_only",
        "elapsed_seconds": elapsed,
        "blender_sampler_seconds": blender_sampler_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "maximum_rss_mb": maximum_rss_mb,
        "performance_limit_seconds": 120.0,
        "performance_pass": performance_pass,
        "publishable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-npz", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--allow-partial-action",
        action="store_true",
        help="diagnostic smoke mode only; a publishable oracle still requires all 271 frames",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_blender_link_oracle_v7(
        oracle_npz=args.oracle_npz,
        oracle_report=args.oracle_report,
        operator_path=args.operator,
        require_full_action=not args.allow_partial_action,
    )
    _json_write(args.output_json, report)
    print(
        f"BlenderLinkOracleParityV7 passed={str(report['passed']).lower()} "
        f"frames={report['frame_count']} seconds={report['elapsed_seconds']:.3f} "
        f"-> {args.output_json}"
    )
    return 0 if report["passed"] and report["performance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
