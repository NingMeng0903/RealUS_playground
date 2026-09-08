"""Analytic V14 collar-driver response override.

The frozen 142 source rig maps ``Clavicle_Rot_L`` (controller 129) with an
explicit 9--13--16 position frame.  That frame transports positions, but it
does not consume the local SMPL-X collar rotation at joint 13.  This module
provides a deliberately narrow counterfactual: controller 129 becomes a
joint-local driver for SMPL-X joint 13, with its coupling rebaked at rest.

The returned asset is an effective immutable copy.  Its topology, vertices,
weights, source hierarchy, fitted bind and every controller except 129 are
preserved.  The normal full-local-FK evaluator then recomputes parent-local
matrices for all 235 controllers, so downstream directly driven world
orientations remain tied to their original SMPL-X frames while 130
``bind_follow`` inherits the collar response.  No mesh edit and no
world-space wrist restore are performed here.

``make_collar_pivot_response_asset_v14`` is the separate J13-pivot ablation:
it applies the same response and coherently re-derives the three target bind
matrices after moving only target controller 129's neutral origin to
``rest_joints[13]``.  The source bind remains untouched.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .anatomy_lbs import (
    build_source_driver_coupling,
    source_bone_posed_global,
)


COLLAR_RESPONSE_SCHEMA_V14 = "collar_joint_local_response_v14"
COLLAR_BONE_ID_V14 = 129
COLLAR_SMPLX_JOINT_ID_V14 = 13
COLLAR_ORIGINAL_MODE_V14 = "segment_root"
COLLAR_RESPONSE_MODE_V14 = "joint_local"
COLLAR_ORIGINAL_DRIVER_A_V14 = 9
COLLAR_ORIGINAL_DRIVER_B_V14 = 13
COLLAR_RESPONSE_DRIVER_A_V14 = 13
COLLAR_RESPONSE_DRIVER_B_V14 = 13
COLLAR_ORIGINAL_FRAME_JOINTS_V14 = (9, 13, 16)
COLLAR_RESPONSE_FRAME_JOINTS_V14 = (13, 13, -1)
COLLAR_PIVOT_VARIANT_V14 = "joint13_pivot"


def _source_count(asset: Any) -> int:
    names = getattr(asset, "source_bone_names", None)
    if names is None or len(names) != 235:
        raise ValueError("V14 collar response requires the frozen 235-controller source rig")
    return int(len(names))


def _metadata(asset: Any) -> dict[str, Any]:
    value = getattr(asset, "metadata", None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("source rig metadata must be a mapping")
    return dict(value)


def _validate_original_collar_contract(asset: Any) -> None:
    _source_count(asset)
    names = list(asset.source_bone_names)
    if str(names[COLLAR_BONE_ID_V14]) != "Clavicle_Rot_L":
        raise ValueError(
            "collar response expects source bone 129 to be Clavicle_Rot_L"
        )
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if parents.shape != (235,) or int(parents[COLLAR_BONE_ID_V14]) != 107:
        raise ValueError("collar response expects Clavicle_Rot_L parent 107")
    modes = list(asset.source_bone_driver_types or ())
    a_ids = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    b_ids = np.asarray(asset.source_bone_smplx_b, dtype=np.int64)
    frame = np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
    if (
        len(modes) != 235
        or a_ids.shape != (235,)
        or b_ids.shape != (235,)
        or frame.shape != (235, 3)
    ):
        raise ValueError("source collar driver metadata has invalid shape")
    if (
        modes[COLLAR_BONE_ID_V14] != COLLAR_ORIGINAL_MODE_V14
        or int(a_ids[COLLAR_BONE_ID_V14]) != COLLAR_ORIGINAL_DRIVER_A_V14
        or int(b_ids[COLLAR_BONE_ID_V14]) != COLLAR_ORIGINAL_DRIVER_B_V14
        or tuple(int(value) for value in frame[COLLAR_BONE_ID_V14])
        != COLLAR_ORIGINAL_FRAME_JOINTS_V14
    ):
        raise ValueError(
            "source collar driver is not the expected 9->13 explicit frame; "
            "refusing to stack an unverified override"
        )


def _already_applied(asset: Any) -> bool:
    metadata = _metadata(asset)
    record = metadata.get("source_collar_response_v14")
    if not isinstance(record, dict) or record.get("schema") != COLLAR_RESPONSE_SCHEMA_V14:
        return False
    _source_count(asset)
    names = list(asset.source_bone_names)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if (
        names[COLLAR_BONE_ID_V14] != "Clavicle_Rot_L"
        or parents.shape != (235,)
        or int(parents[COLLAR_BONE_ID_V14]) != 107
    ):
        raise ValueError("collar response asset has an invalid source contract")
    if (
        int(record.get("bone_id", -1)) != COLLAR_BONE_ID_V14
        or record.get("bone_name") != "Clavicle_Rot_L"
        or record.get("response_mode") != COLLAR_RESPONSE_MODE_V14
    ):
        raise ValueError("collar response metadata does not match its source contract")
    modes = list(asset.source_bone_driver_types or ())
    a_ids = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    b_ids = np.asarray(asset.source_bone_smplx_b, dtype=np.int64)
    frame = np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
    if (
        len(modes) != 235
        or a_ids.shape != (235,)
        or b_ids.shape != (235,)
        or frame.shape != (235, 3)
    ):
        return False
    return bool(
        modes[COLLAR_BONE_ID_V14] == COLLAR_RESPONSE_MODE_V14
        and int(a_ids[COLLAR_BONE_ID_V14]) == COLLAR_RESPONSE_DRIVER_A_V14
        and int(b_ids[COLLAR_BONE_ID_V14]) == COLLAR_RESPONSE_DRIVER_B_V14
        and tuple(int(value) for value in frame[COLLAR_BONE_ID_V14])
        == COLLAR_RESPONSE_FRAME_JOINTS_V14
    )


def make_collar_response_asset_v14(asset: Any) -> Any:
    """Return an effective asset with the fixed collar response override.

    The original asset is never mutated.  The only source driver fields
    changed are controller 129's ``a``, ``b``, mode and padded frame-joint
    declaration.  Its source-driver coupling is rebuilt from the modified
    rest frame, which keeps ``target_bind_global[129]`` as the rest anchor.
    """

    if _already_applied(asset):
        return asset
    _validate_original_collar_contract(asset)
    source_a = np.array(asset.source_bone_smplx_a, dtype=np.int32, copy=True)
    source_b = np.array(asset.source_bone_smplx_b, dtype=np.int32, copy=True)
    frame_joints = np.array(asset.source_bone_frame_joints, dtype=np.int32, copy=True)
    modes = list(asset.source_bone_driver_types or ())
    source_a[COLLAR_BONE_ID_V14] = COLLAR_RESPONSE_DRIVER_A_V14
    source_b[COLLAR_BONE_ID_V14] = COLLAR_RESPONSE_DRIVER_B_V14
    frame_joints[COLLAR_BONE_ID_V14] = COLLAR_RESPONSE_FRAME_JOINTS_V14
    modes[COLLAR_BONE_ID_V14] = COLLAR_RESPONSE_MODE_V14
    # ``build_source_driver_coupling`` consumes the modified driver declaration
    # at neutral pose.  Passing None makes this an explicit offline rebake,
    # never a blend-time mutation of the original coupling.
    staged = replace(
        asset,
        source_bone_smplx_a=source_a,
        source_bone_smplx_b=source_b,
        source_bone_frame_joints=frame_joints,
        source_bone_driver_types=modes,
        source_driver_coupling=None,
    )
    rebaked_coupling = np.asarray(
        build_source_driver_coupling(staged), dtype=np.float32
    )
    # Rebuilding the neutral frames is needed for row 129, but it must not
    # rewrite unrelated persisted response rows (even by float-rounding).
    # This matters when a source pack has separately audited couplings for the
    # other 234 controllers.
    if asset.source_driver_coupling is None:
        coupling = rebaked_coupling
    else:
        original_coupling = np.asarray(asset.source_driver_coupling, dtype=np.float32)
        if original_coupling.shape != rebaked_coupling.shape:
            raise ValueError("source driver coupling has an invalid shape")
        coupling = original_coupling.copy()
        coupling[COLLAR_BONE_ID_V14] = rebaked_coupling[COLLAR_BONE_ID_V14]
    metadata = _metadata(asset)
    metadata["source_collar_response_v14"] = {
        "schema": COLLAR_RESPONSE_SCHEMA_V14,
        "bone_id": COLLAR_BONE_ID_V14,
        "bone_name": "Clavicle_Rot_L",
        "original_mode": COLLAR_ORIGINAL_MODE_V14,
        "original_smplx_a": COLLAR_ORIGINAL_DRIVER_A_V14,
        "original_smplx_b": COLLAR_ORIGINAL_DRIVER_B_V14,
        "original_frame_joints": list(COLLAR_ORIGINAL_FRAME_JOINTS_V14),
        "response_mode": COLLAR_RESPONSE_MODE_V14,
        "response_smplx_a": COLLAR_RESPONSE_DRIVER_A_V14,
        "response_smplx_b": COLLAR_RESPONSE_DRIVER_B_V14,
        "response_frame_joints": list(COLLAR_RESPONSE_FRAME_JOINTS_V14),
        "coupling": "offline_build_source_driver_coupling_at_neutral_pose",
        "coupling_rows_rebaked": [COLLAR_BONE_ID_V14],
        "coupling_rows_preserved": 234,
        "topology_weights_mesh_immutable": True,
        "world_space_wrist_restore": False,
    }
    return replace(
        staged,
        source_driver_coupling=np.asarray(coupling, dtype=np.float32),
        metadata=metadata,
    )


def _target_bind_local_from_global(global_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Derive all parent-local target binds from one global authority."""

    matrices = np.asarray(global_bind, dtype=np.float64)
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    if matrices.shape != (len(parent_ids), 4, 4):
        raise ValueError("target bind and source parent shapes do not agree")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("target bind contains non-finite values")
    local = matrices.copy()
    for bone, parent in enumerate(parent_ids.tolist()):
        if parent >= 0:
            if parent >= bone:
                raise ValueError("source parents must be topological")
            local[bone] = np.linalg.inv(matrices[parent]) @ matrices[bone]
    return local


def _pivot_already_applied(asset: Any) -> bool:
    metadata = _metadata(asset)
    record = metadata.get("source_collar_response_v14")
    if not isinstance(record, dict) or record.get("variant") != COLLAR_PIVOT_VARIANT_V14:
        return False
    if not _already_applied(asset):
        raise ValueError("collar pivot metadata lacks the joint-local response contract")
    if (
        int(record.get("pivot_bone_id", -1)) != COLLAR_BONE_ID_V14
        or int(record.get("pivot_smplx_joint_id", -1)) != COLLAR_SMPLX_JOINT_ID_V14
        or record.get("pivot_target_bind_origin") != "rest_joints[13]"
    ):
        raise ValueError("collar pivot metadata does not match the J13 pivot contract")
    # Validate the complete target-bind triple before treating an asset as an
    # already compiled pivot.  A stale metadata marker must never suppress a
    # required rebake.
    _source_count(asset)
    global_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    local_bind = np.asarray(asset.target_bind_local, dtype=np.float64)
    # Do not accept the source-inverse fallback here: a pivot asset must carry
    # its own explicit target inverse bind, even when the source and target
    # happen to be numerically similar.
    target_inverse = getattr(asset, "target_inverse_bind", None)
    if target_inverse is None:
        raise ValueError("collar pivot target inverse bind is missing")
    inverse_bind = np.asarray(target_inverse, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    rest_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    if (
        global_bind.shape != (235, 4, 4)
        or local_bind.shape != global_bind.shape
        or inverse_bind.shape != global_bind.shape
        or rest_joints.shape != (55, 3)
        or not np.all(np.isfinite(global_bind))
        or not np.all(np.isfinite(local_bind))
        or not np.all(np.isfinite(inverse_bind))
        or not np.all(np.isfinite(rest_joints))
    ):
        raise ValueError("collar pivot target bind has invalid shapes")
    if not np.allclose(
        global_bind[COLLAR_BONE_ID_V14, :3, 3],
        rest_joints[COLLAR_SMPLX_JOINT_ID_V14],
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("collar pivot target origin is not SMPL-X J13")
    expected_local = _target_bind_local_from_global(global_bind, parents)
    if not np.allclose(local_bind, expected_local, atol=2.0e-6, rtol=0.0):
        raise ValueError("collar pivot target local bind is incoherent")
    if not np.allclose(
        inverse_bind @ global_bind,
        np.eye(4, dtype=np.float64)[None],
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("collar pivot target inverse bind is incoherent")
    return True


def make_collar_pivot_response_asset_v14(asset: Any) -> Any:
    """Return the collar response with controller 129 pivoted to SMPL-X J13.

    This is the measured J13-pivot ablation used by the V14 compile tool.  It
    first applies the narrow joint-local response, then changes only
    ``target_bind_global[129]``'s translation to the materialized neutral
    ``rest_joints[13]`` point.  All parent-local target binds and the target
    inverse bind are derived from that global matrix; target B130's global
    frame remains unchanged.  The source bind, mesh, sparse weights and
    topology are untouched.  Coupling row 129 is rebaked against the new
    target pivot and all other persisted rows are copied exactly.
    """

    if _pivot_already_applied(asset):
        return asset
    response = make_collar_response_asset_v14(asset)
    global_bind = np.asarray(response.target_bind_global, dtype=np.float64).copy()
    rest_joints = np.asarray(response.rest_joints, dtype=np.float64)
    if rest_joints.shape != (55, 3) or not np.all(np.isfinite(rest_joints)):
        raise ValueError("collar pivot requires finite SMPL-X rest joints")
    global_bind[COLLAR_BONE_ID_V14, :3, 3] = rest_joints[COLLAR_SMPLX_JOINT_ID_V14]
    parents = np.asarray(response.source_bone_parents, dtype=np.int64)
    local_bind = _target_bind_local_from_global(global_bind, parents)
    inverse_bind = np.linalg.inv(global_bind)
    staged = replace(
        response,
        target_rest_global=global_bind.astype(np.float32),
        target_rest_local=local_bind.astype(np.float32),
        target_inverse_bind=inverse_bind.astype(np.float32),
        source_driver_coupling=None,
    )
    rebaked_coupling = np.asarray(
        build_source_driver_coupling(staged), dtype=np.float32
    )
    if response.source_driver_coupling is None:
        coupling = rebaked_coupling
    else:
        existing_coupling = np.asarray(response.source_driver_coupling, dtype=np.float32)
        if existing_coupling.shape != rebaked_coupling.shape:
            raise ValueError("source driver coupling has an invalid shape")
        coupling = existing_coupling.copy()
        coupling[COLLAR_BONE_ID_V14] = rebaked_coupling[COLLAR_BONE_ID_V14]
    metadata = _metadata(response)
    record = dict(metadata.get("source_collar_response_v14") or {})
    record.update(
        {
            "schema": COLLAR_RESPONSE_SCHEMA_V14,
            "variant": COLLAR_PIVOT_VARIANT_V14,
            "pivot_bone_id": COLLAR_BONE_ID_V14,
            "pivot_smplx_joint_id": COLLAR_SMPLX_JOINT_ID_V14,
            "pivot_target_bind_origin": "rest_joints[13]",
            "target_bind_global_recomputed": True,
            "target_bind_local_recomputed": True,
            "target_inverse_bind_recomputed": True,
            "target_bind_global_130_preserved": True,
            "coupling_rows_rebaked": [COLLAR_BONE_ID_V14],
            "coupling_rows_preserved": 234,
            "source_bind_mesh_topology_weights_immutable": True,
            "rest_refit_performed": False,
        }
    )
    metadata["source_collar_response_v14"] = record
    return replace(
        staged,
        source_driver_coupling=np.asarray(coupling, dtype=np.float32),
        metadata=metadata,
    )


def apply_collar_response_v14(asset: Any) -> Any:
    """Alias for :func:`make_collar_response_asset_v14`."""

    return make_collar_response_asset_v14(asset)


def collar_response_runtime_arrays_v14(asset: Any) -> dict[str, np.ndarray]:
    """Return the baked response fields as detached, array-only values.

    The response is carried by the effective asset for in-process evaluation,
    but a runtime pack may want to persist the override without serializing a
    Python object.  Every returned array is a copy, so a caller cannot mutate
    either the source pack or the effective asset through this view.  Driver
    modes are represented as a Unicode array because they are part of the
    response contract rather than free-form runtime state.
    """

    effective = make_collar_response_asset_v14(asset)
    return {
        "source_bone_smplx_a": np.asarray(
            effective.source_bone_smplx_a, dtype=np.int32
        ).copy(),
        "source_bone_smplx_b": np.asarray(
            effective.source_bone_smplx_b, dtype=np.int32
        ).copy(),
        "source_bone_frame_joints": np.asarray(
            effective.source_bone_frame_joints, dtype=np.int32
        ).copy(),
        "source_driver_coupling": np.asarray(
            effective.source_driver_coupling, dtype=np.float32
        ).copy(),
        "source_bone_driver_types": np.asarray(
            list(effective.source_bone_driver_types or ()), dtype="<U32"
        ).copy(),
    }


def restore_collar_response_asset_v14(asset: Any) -> Any:
    """Restore the original 129 response contract on an effective asset.

    This is an in-memory inverse for tools that need to compare a baked
    response with the immutable source.  It only accepts this module's own
    response record and rebuilds the original neutral coupling; an unrelated
    source asset is returned unchanged.  The caller still owns persistence of
    whichever variant it selects.
    """

    if not _already_applied(asset):
        return asset
    names = list(asset.source_bone_names or ())
    if len(names) != 235 or names[COLLAR_BONE_ID_V14] != "Clavicle_Rot_L":
        raise ValueError("collar response asset has an invalid source contract")
    source_a = np.array(asset.source_bone_smplx_a, dtype=np.int32, copy=True)
    source_b = np.array(asset.source_bone_smplx_b, dtype=np.int32, copy=True)
    frame_joints = np.array(asset.source_bone_frame_joints, dtype=np.int32, copy=True)
    modes = list(asset.source_bone_driver_types or ())
    source_a[COLLAR_BONE_ID_V14] = COLLAR_ORIGINAL_DRIVER_A_V14
    source_b[COLLAR_BONE_ID_V14] = COLLAR_ORIGINAL_DRIVER_B_V14
    frame_joints[COLLAR_BONE_ID_V14] = COLLAR_ORIGINAL_FRAME_JOINTS_V14
    modes[COLLAR_BONE_ID_V14] = COLLAR_ORIGINAL_MODE_V14
    metadata = _metadata(asset)
    metadata.pop("source_collar_response_v14", None)
    staged = replace(
        asset,
        source_bone_smplx_a=source_a,
        source_bone_smplx_b=source_b,
        source_bone_frame_joints=frame_joints,
        source_bone_driver_types=modes,
        source_driver_coupling=None,
        metadata=metadata,
    )
    rebaked_coupling = np.asarray(
        build_source_driver_coupling(staged), dtype=np.float32
    )
    if asset.source_driver_coupling is None:
        coupling = rebaked_coupling
    else:
        existing_coupling = np.asarray(asset.source_driver_coupling, dtype=np.float32)
        if existing_coupling.shape != rebaked_coupling.shape:
            raise ValueError("source driver coupling has an invalid shape")
        coupling = existing_coupling.copy()
        coupling[COLLAR_BONE_ID_V14] = rebaked_coupling[COLLAR_BONE_ID_V14]
    return replace(
        staged,
        source_driver_coupling=np.asarray(coupling, dtype=np.float32),
    )


def source_bone_posed_global_collar_v14(asset: Any, pose_axis_angle: Any) -> np.ndarray:
    """Evaluate corrected source globals through the unchanged 235-bone FK."""

    effective = make_collar_response_asset_v14(asset)
    return source_bone_posed_global(effective, pose_axis_angle)


def source_bone_local_matrices_collar_v14(asset: Any, pose_axis_angle: Any) -> np.ndarray:
    """Return corrected parent-local FK matrices for all 235 controllers."""

    effective = make_collar_response_asset_v14(asset)
    globals_ = source_bone_posed_global(effective, pose_axis_angle)
    parents = np.asarray(effective.source_bone_parents, dtype=np.int64)
    local = np.asarray(globals_, dtype=np.float64).copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            local[bone] = np.linalg.inv(globals_[int(parent)]) @ globals_[bone]
    return local


def source_bone_local_correction_v14(asset: Any, pose_axis_angle: Any) -> np.ndarray:
    """Return right-multiplied local deltas from original to collar FK.

    If ``L`` is the original source local FK and ``C`` is this result, then
    ``L[i] @ delta[i]`` equals the corrected local matrix for every controller.
    The complete 235 array is intentional: descendants recompute their
    parent-local driver delta while retaining their original desired world
    orientation.
    """

    original = source_bone_posed_global(asset, pose_axis_angle)
    corrected = source_bone_posed_global_collar_v14(asset, pose_axis_angle)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    original_local = np.asarray(original, dtype=np.float64).copy()
    corrected_local = np.asarray(corrected, dtype=np.float64).copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            original_local[bone] = np.linalg.inv(original[parent]) @ original[bone]
            corrected_local[bone] = np.linalg.inv(corrected[parent]) @ corrected[bone]
    deltas = np.empty_like(original_local)
    for bone in range(len(deltas)):
        deltas[bone] = np.linalg.inv(original_local[bone]) @ corrected_local[bone]
    return deltas


__all__ = [
    "COLLAR_BONE_ID_V14",
    "COLLAR_ORIGINAL_DRIVER_A_V14",
    "COLLAR_ORIGINAL_DRIVER_B_V14",
    "COLLAR_ORIGINAL_FRAME_JOINTS_V14",
    "COLLAR_ORIGINAL_MODE_V14",
    "COLLAR_RESPONSE_DRIVER_A_V14",
    "COLLAR_RESPONSE_DRIVER_B_V14",
    "COLLAR_RESPONSE_FRAME_JOINTS_V14",
    "COLLAR_RESPONSE_MODE_V14",
    "COLLAR_RESPONSE_SCHEMA_V14",
    "COLLAR_PIVOT_VARIANT_V14",
    "COLLAR_SMPLX_JOINT_ID_V14",
    "apply_collar_response_v14",
    "collar_response_runtime_arrays_v14",
    "make_collar_pivot_response_asset_v14",
    "make_collar_response_asset_v14",
    "restore_collar_response_asset_v14",
    "source_bone_local_correction_v14",
    "source_bone_local_matrices_collar_v14",
    "source_bone_posed_global_collar_v14",
]
