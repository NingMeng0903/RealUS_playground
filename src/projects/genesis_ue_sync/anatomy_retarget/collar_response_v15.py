"""Bilateral collar response with anatomically located SMPL-X pivots.

V14 established one narrow response for ``Clavicle_Rot_L``: the source
controller consumed the local SMPL-X collar rotation and its target bind
origin was moved to the corresponding SMPL-X collar joint.  This module
applies that same offline operation to both sides, discovering controller IDs
from the authenticated source-rig names instead of assuming that the right
side is adjacent to the left side.

The operation changes only the two collar driver declarations, their two
rebaked coupling rows, and the coherent target bind triple.  Source geometry,
topology, hierarchy, authored source bind, sparse driver indices and weights
remain unchanged.  The result is returned together with the changed IDs and a
JSON-ready provenance record so a caller can pass the effective asset to
``BakedMotionResponseV14.from_assets`` without rebuilding anything at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Any

import numpy as np

from .anatomy_lbs import build_source_driver_coupling


COLLAR_RESPONSE_SCHEMA_V15 = "collar_bilateral_joint_local_pivot_v15"
COLLAR_RESPONSE_METADATA_KEY_V15 = "source_collar_response_v14"
COLLAR_RESPONSE_MODE_V15 = "joint_local"
COLLAR_ORIGINAL_MODE_V15 = "segment_root"
COLLAR_PARENT_NAME_V15 = "Spine_T1"
COLLAR_RESPONSE_VARIANT_V15 = "bilateral_joint13_joint14_pivot"
COLLAR_SIDES_V15 = ("L", "R")
COLLAR_SMPLX_JOINTS_V15 = {"L": 13, "R": 14}
COLLAR_SHOULDER_SMPLX_JOINTS_V15 = {"L": 16, "R": 17}
COLLAR_ORIGINAL_DRIVER_A_V15 = 9


@dataclass(frozen=True)
class _CollarSpecV15:
    side: str
    bone_id: int
    bone_name: str
    parent_id: int
    pivot_joint: int
    original_a: int
    original_b: int
    original_frame: tuple[int, int, int]
    response_frame: tuple[int, int, int]


def _metadata(asset: Any) -> dict[str, Any]:
    value = getattr(asset, "metadata", None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("source rig metadata must be a mapping")
    return copy.deepcopy(value)


def _source_count(asset: Any) -> int:
    names = getattr(asset, "source_bone_names", None)
    if names is None or len(names) != 235:
        raise ValueError("V15 collar response requires the frozen 235-controller source rig")
    return 235


def _require_rigid(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} has invalid shape or non-finite values")
    if shape[-2:] == (4, 4):
        rotation = array[..., :3, :3]
        if not np.allclose(array[..., 3, :], (0.0, 0.0, 0.0, 1.0), atol=2e-6, rtol=0.0):
            raise ValueError(f"{label} has invalid homogeneous rows")
        identity = np.eye(3, dtype=np.float64)
        if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, identity, atol=3e-6, rtol=0.0):
            raise ValueError(f"{label} contains scale or shear")
        if not np.allclose(np.linalg.det(rotation), 1.0, atol=3e-6, rtol=0.0):
            raise ValueError(f"{label} contains a reflection or degenerate frame")
    return array


def _target_bind_local_from_global(global_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    matrices = _require_rigid(global_bind, (235, 4, 4), "target bind global")
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    if parent_ids.shape != (235,):
        raise ValueError("source parents must contain 235 controllers")
    local = matrices.copy()
    for bone, parent in enumerate(parent_ids.tolist()):
        if parent < -1 or parent >= bone:
            raise ValueError(f"source parent {parent} for controller {bone} is not topological")
        if parent >= 0:
            local[bone] = np.linalg.inv(matrices[parent]) @ matrices[bone]
    return local


def _discover_original_specs(asset: Any) -> tuple[_CollarSpecV15, ...]:
    """Discover and validate both original collar declarations."""

    _source_count(asset)
    names = [str(value) for value in asset.source_bone_names]
    if len(set(names)) != len(names):
        raise ValueError("source controller names must be unique")
    by_name = {name: index for index, name in enumerate(names)}
    if COLLAR_PARENT_NAME_V15 not in by_name:
        raise ValueError(f"source rig is missing {COLLAR_PARENT_NAME_V15}")
    parent_id = by_name[COLLAR_PARENT_NAME_V15]
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    modes = list(asset.source_bone_driver_types or ())
    source_a = np.asarray(asset.source_bone_smplx_a, dtype=np.int64).reshape(-1)
    source_b = np.asarray(asset.source_bone_smplx_b, dtype=np.int64).reshape(-1)
    frame = np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
    if (
        parents.shape != (235,)
        or len(modes) != 235
        or source_a.shape != (235,)
        or source_b.shape != (235,)
        or frame.shape != (235, 3)
    ):
        raise ValueError("source collar driver metadata has invalid shapes")
    if (
        np.any(source_a < 0)
        or np.any(source_a >= 55)
        or np.any(source_b < 0)
        or np.any(source_b >= 55)
        or np.any(frame < -1)
        or np.any(frame >= 55)
        or not np.array_equal(frame[:, 0], source_a)
    ):
        raise ValueError("source collar driver metadata has invalid SMPL-X identifiers")

    specs = []
    for side in COLLAR_SIDES_V15:
        collar_name = f"Clavicle_Rot_{side}"
        shoulder_name = f"Shoulder_Rotate_{side}"
        if collar_name not in by_name or shoulder_name not in by_name:
            raise ValueError(f"source rig is missing {collar_name} or {shoulder_name}")
        bone_id = by_name[collar_name]
        shoulder_id = by_name[shoulder_name]
        pivot_joint = int(COLLAR_SMPLX_JOINTS_V15[side])
        expected_shoulder_joint = int(COLLAR_SHOULDER_SMPLX_JOINTS_V15[side])
        original_a = int(source_a[bone_id])
        original_b = int(source_b[bone_id])
        original_frame = tuple(int(value) for value in frame[bone_id])
        expected_frame = (original_a, original_b, int(source_a[shoulder_id]))
        if (
            int(parents[bone_id]) != parent_id
            or modes[bone_id] != COLLAR_ORIGINAL_MODE_V15
            or original_a != COLLAR_ORIGINAL_DRIVER_A_V15
            or int(source_a[shoulder_id]) != expected_shoulder_joint
            or original_b != pivot_joint
            or original_frame != expected_frame
            or len(set(original_frame)) != 3
        ):
            raise ValueError(
                f"{collar_name} is not the expected explicit segment-root collar frame"
            )
        specs.append(
            _CollarSpecV15(
                side=side,
                bone_id=bone_id,
                bone_name=collar_name,
                parent_id=parent_id,
                pivot_joint=pivot_joint,
                original_a=original_a,
                original_b=original_b,
                original_frame=original_frame,
                response_frame=(pivot_joint, pivot_joint, -1),
            )
        )
    if len({spec.bone_id for spec in specs}) != 2:
        raise ValueError("left and right collar controller IDs must be distinct")
    return tuple(specs)


def _record_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return a validated V15 marker, rejecting an unrelated V14 marker."""

    record = metadata.get(COLLAR_RESPONSE_METADATA_KEY_V15)
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("source collar response metadata must be a mapping")
    if record.get("schema") != COLLAR_RESPONSE_SCHEMA_V15:
        raise ValueError(
            "source asset already carries a different collar response; refusing double apply"
        )
    if record.get("variant") != COLLAR_RESPONSE_VARIANT_V15:
        raise ValueError("source collar response metadata has an unexpected V15 variant")
    return record


def _validate_target_triple(asset: Any, specs: tuple[_CollarSpecV15, ...]) -> None:
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    global_bind = _require_rigid(asset.target_bind_global, (235, 4, 4), "target bind global")
    local_bind = _require_rigid(asset.target_bind_local, (235, 4, 4), "target bind local")
    inverse_bind = _require_rigid(asset.target_inverse_bind, (235, 4, 4), "target inverse bind")
    expected_local = _target_bind_local_from_global(global_bind, parents)
    if not np.allclose(local_bind, expected_local, atol=3e-6, rtol=0.0):
        raise ValueError("target bind local is not derived from target bind global")
    if not np.allclose(inverse_bind @ global_bind, np.eye(4)[None], atol=3e-6, rtol=0.0):
        raise ValueError("target inverse bind is not the inverse of target bind global")
    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    if joints.shape != (55, 3) or not np.all(np.isfinite(joints)):
        raise ValueError("source rest joints are invalid")
    for spec in specs:
        if not np.allclose(
            global_bind[spec.bone_id, :3, 3], joints[spec.pivot_joint], atol=3e-6, rtol=0.0
        ):
            raise ValueError(f"{spec.bone_name} target pivot is not SMPL-X J{spec.pivot_joint}")


def _validate_applied(asset: Any, record: dict[str, Any]) -> tuple[_CollarSpecV15, ...]:
    specs = _discover_applied_specs(asset)
    ids = [int(value) for value in record.get("changed_controller_ids", [])]
    if ids != [spec.bone_id for spec in specs]:
        raise ValueError("V15 collar metadata changed-controller IDs are invalid")
    names = list(asset.source_bone_names)
    modes = list(asset.source_bone_driver_types or ())
    source_a = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    source_b = np.asarray(asset.source_bone_smplx_b, dtype=np.int64)
    frame = np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
    for spec in specs:
        if (
            names[spec.bone_id] != spec.bone_name
            or modes[spec.bone_id] != COLLAR_RESPONSE_MODE_V15
            or int(source_a[spec.bone_id]) != spec.pivot_joint
            or int(source_b[spec.bone_id]) != spec.pivot_joint
            or tuple(int(value) for value in frame[spec.bone_id]) != spec.response_frame
        ):
            raise ValueError("V15 collar response metadata does not match driver arrays")
    _validate_target_triple(asset, specs)
    return specs


def _discover_applied_specs(asset: Any) -> tuple[_CollarSpecV15, ...]:
    """Discover sides without requiring the original segment-root declaration."""

    _source_count(asset)
    names = [str(value) for value in asset.source_bone_names]
    by_name = {name: index for index, name in enumerate(names)}
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    source_a = np.asarray(asset.source_bone_smplx_a, dtype=np.int64).reshape(-1)
    source_b = np.asarray(asset.source_bone_smplx_b, dtype=np.int64).reshape(-1)
    frame = np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
    specs = []
    parent_id = by_name.get(COLLAR_PARENT_NAME_V15, -1)
    for side in COLLAR_SIDES_V15:
        name = f"Clavicle_Rot_{side}"
        if name not in by_name:
            raise ValueError(f"source rig is missing {name}")
        bone_id = by_name[name]
        pivot = int(COLLAR_SMPLX_JOINTS_V15[side])
        specs.append(
            _CollarSpecV15(
                side=side,
                bone_id=bone_id,
                bone_name=name,
                parent_id=parent_id,
                pivot_joint=pivot,
                original_a=int(source_a[bone_id]),
                original_b=int(source_b[bone_id]),
                original_frame=tuple(int(value) for value in frame[bone_id]),
                response_frame=(pivot, pivot, -1),
            )
        )
    if parent_id < 0 or any(int(parents[spec.bone_id]) != parent_id for spec in specs):
        raise ValueError("V15 applied collar parent contract is invalid")
    return tuple(specs)


def _provenance(specs: tuple[_CollarSpecV15, ...], *, idempotent: bool) -> dict[str, Any]:
    return {
        "schema": COLLAR_RESPONSE_SCHEMA_V15,
        "variant": COLLAR_RESPONSE_VARIANT_V15,
        "changed_controller_ids": [int(spec.bone_id) for spec in specs],
        "changed_controller_names": [spec.bone_name for spec in specs],
        "pivot_smplx_joint_ids": [int(spec.pivot_joint) for spec in specs],
        "source_parent_name": COLLAR_PARENT_NAME_V15,
        "response_mode": COLLAR_RESPONSE_MODE_V15,
        "response_driver": "joint_local_a=b=pivot_joint",
        "target_bind_pivot": "rest_joints[pivot_smplx_joint_id]",
        "coupling_rows_rebaked": [int(spec.bone_id) for spec in specs],
        "coupling_rows_preserved": 235 - len(specs),
        "source_bind_mesh_topology_weights_immutable": True,
        "world_space_wrist_restore": False,
        "runtime_rebuild": False,
        "idempotent_reuse": bool(idempotent),
        "sides": [
            {
                "side": spec.side,
                "bone_id": int(spec.bone_id),
                "bone_name": spec.bone_name,
                "pivot_smplx_joint": int(spec.pivot_joint),
                "original_mode": COLLAR_ORIGINAL_MODE_V15,
                "original_smplx_a": int(spec.original_a),
                "original_smplx_b": int(spec.original_b),
                "original_frame_joints": list(spec.original_frame),
                "response_mode": COLLAR_RESPONSE_MODE_V15,
                "response_smplx_a": int(spec.pivot_joint),
                "response_smplx_b": int(spec.pivot_joint),
                "response_frame_joints": list(spec.response_frame),
            }
            for spec in specs
        ],
    }


def make_collar_response_asset_v15(asset: Any) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
    """Build the bilateral collar effective asset and return details.

    Returns ``(effective_asset, changed_controller_ids, provenance)``.  The
    effective asset is suitable for ``BakedMotionResponseV14.from_assets``;
    no response is recomputed at runtime.  Existing V14 or malformed V15
    markers fail closed.  Calling this function on its own valid V15 output is
    idempotent and returns that same asset object.
    """

    metadata = _metadata(asset)
    record = _record_from_metadata(metadata)
    if record is not None:
        specs = _validate_applied(asset, record)
        provenance = copy.deepcopy(record.get("provenance"))
        if not isinstance(provenance, dict):
            provenance = _provenance(specs, idempotent=True)
        provenance["idempotent_reuse"] = True
        return asset, tuple(spec.bone_id for spec in specs), provenance
    specs = _discover_original_specs(asset)
    # A V14 effective source has already changed the left declaration.  It is
    # intentionally rejected instead of being silently stacked with V15.
    if "source_collar_response_v14" in metadata:
        raise ValueError("source asset already carries a V14 collar response")
    _validate_target_triple_without_pivot(asset)

    source_a = np.asarray(asset.source_bone_smplx_a, dtype=np.int32).copy()
    source_b = np.asarray(asset.source_bone_smplx_b, dtype=np.int32).copy()
    frame = np.asarray(asset.source_bone_frame_joints, dtype=np.int32).copy()
    modes = list(asset.source_bone_driver_types or ())
    for spec in specs:
        source_a[spec.bone_id] = spec.pivot_joint
        source_b[spec.bone_id] = spec.pivot_joint
        frame[spec.bone_id] = np.asarray(spec.response_frame, dtype=np.int32)
        modes[spec.bone_id] = COLLAR_RESPONSE_MODE_V15

    original_coupling_raw = np.asarray(asset.source_driver_coupling)
    original_coupling = _require_rigid(
        original_coupling_raw, (235, 4, 4), "source driver coupling"
    )
    target_global_raw = np.asarray(asset.target_bind_global)
    target_global = _require_rigid(
        target_global_raw, (235, 4, 4), "target bind global"
    ).copy()
    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    if joints.shape != (55, 3) or not np.all(np.isfinite(joints)):
        raise ValueError("source rest joints are invalid")
    for spec in specs:
        target_global[spec.bone_id, :3, 3] = joints[spec.pivot_joint]
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    target_local_raw = np.asarray(asset.target_bind_local)
    target_inverse_raw = np.asarray(asset.target_inverse_bind)
    target_local = _target_bind_local_from_global(target_global, parents)
    target_inverse = np.linalg.inv(target_global)

    staged = replace(
        asset,
        source_bone_smplx_a=source_a,
        source_bone_smplx_b=source_b,
        source_bone_frame_joints=frame,
        source_bone_driver_types=modes,
        source_driver_coupling=None,
        # Preserve the source array dtypes so an untouched target row remains
        # bit-identical for float64 test assets as well as the authenticated
        # float32 packs.  Only the declared collar rows are numerically moved.
        target_rest_global=target_global.astype(target_global_raw.dtype, copy=False),
        target_rest_local=target_local.astype(target_local_raw.dtype, copy=False),
        target_inverse_bind=target_inverse.astype(target_inverse_raw.dtype, copy=False),
    )
    # Build once offline from the fully modified declarations and pivoted
    # target bind.  Copy all unrelated rows from the source exactly, as V14
    # does, so independent response identities survive the rebake.
    rebaked = np.asarray(build_source_driver_coupling(staged), dtype=np.float32)
    if rebaked.shape != original_coupling.shape:
        raise ValueError("rebaked source driver coupling has an invalid shape")
    coupling = np.array(original_coupling_raw, copy=True)
    for spec in specs:
        coupling[spec.bone_id] = rebaked[spec.bone_id].astype(
            coupling.dtype, copy=False
        )

    provenance = _provenance(specs, idempotent=False)
    metadata = _metadata(asset)
    record = copy.deepcopy(provenance)
    record.update(
        {
            "schema": COLLAR_RESPONSE_SCHEMA_V15,
            "variant": COLLAR_RESPONSE_VARIANT_V15,
            "metadata_compatibility_key": COLLAR_RESPONSE_METADATA_KEY_V15,
            "provenance": copy.deepcopy(provenance),
            "target_bind_global_recomputed": True,
            "target_bind_local_recomputed": True,
            "target_inverse_bind_recomputed": True,
            "target_bind_global_noncollar_preserved": True,
            "source_bind_mesh_topology_weights_immutable": True,
        }
    )
    # BakedMotionResponseV14 currently allowlists this metadata key.  The
    # nested schema remains explicitly V15, so V14 builders reject it rather
    # than interpreting it as a V14 response.  The caller also receives the
    # same record as a standalone provenance object.
    metadata[COLLAR_RESPONSE_METADATA_KEY_V15] = record
    effective = replace(
        staged,
        source_driver_coupling=coupling,
        metadata=metadata,
    )
    _validate_applied(effective, record)
    return effective, tuple(spec.bone_id for spec in specs), provenance


def _validate_target_triple_without_pivot(asset: Any) -> None:
    global_bind = _require_rigid(asset.target_bind_global, (235, 4, 4), "target bind global")
    local_bind = _require_rigid(asset.target_bind_local, (235, 4, 4), "target bind local")
    inverse_bind = _require_rigid(asset.target_inverse_bind, (235, 4, 4), "target inverse bind")
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    expected = _target_bind_local_from_global(global_bind, parents)
    if not np.allclose(local_bind, expected, atol=3e-6, rtol=0.0):
        raise ValueError("source target bind local is incoherent")
    if not np.allclose(inverse_bind @ global_bind, np.eye(4)[None], atol=3e-6, rtol=0.0):
        raise ValueError("source target inverse bind is incoherent")


def apply_collar_response_v15(asset: Any) -> Any:
    """Return only the effective asset for callers that do not need details."""

    return make_collar_response_asset_v15(asset)[0]


def make_collar_pivot_response_asset_v15(asset: Any) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
    """Explicit alias naming the J13/J14 pivot variant."""

    return make_collar_response_asset_v15(asset)


def make_bilateral_collar_pivot_response_asset_v15(asset: Any) -> Any:
    """Return only the bilateral J13/J14 effective asset.

    The compiler-facing convenience entry point deliberately has the same
    single-asset return shape as the older V14 collar helper.  Callers that
    need the changed controller IDs and the standalone provenance record can
    use :func:`make_collar_response_asset_v15`; the exact same provenance is
    also persisted in ``effective.metadata`` for the response baker.
    """

    return make_collar_response_asset_v15(asset)[0]


__all__ = [
    "COLLAR_ORIGINAL_DRIVER_A_V15",
    "COLLAR_ORIGINAL_MODE_V15",
    "COLLAR_RESPONSE_METADATA_KEY_V15",
    "COLLAR_RESPONSE_MODE_V15",
    "COLLAR_RESPONSE_SCHEMA_V15",
    "COLLAR_SHOULDER_SMPLX_JOINTS_V15",
    "COLLAR_RESPONSE_VARIANT_V15",
    "apply_collar_response_v15",
    "make_bilateral_collar_pivot_response_asset_v15",
    "make_collar_pivot_response_asset_v15",
    "make_collar_response_asset_v15",
]
