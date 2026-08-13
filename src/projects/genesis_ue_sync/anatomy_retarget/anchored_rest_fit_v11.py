"""V11 anchored rest-fit: undo harmful hinge translations on frozen V7 rest.

V7 rest-fit violated OSSO-style socket seating (Ej):

- ``result[elbow] = humerus`` copied the shoulder correction onto Elbow_Rot,
  amplifying left-elbow bind error 7.83 → 19.79 mm via a ~265 mm lever.
- Knee bind was placed on a station/centerline ray, pushing left knee
  6.66 → 21.76 mm from anatomical A_subj.

V11 keeps the frozen V7 *mesh* (contact / embed geometry) and restores the
corrupted hinge controller *origins* to ``B_prefit``.  That:

1. Splits elbow from humerus (elbow translation no longer equals shoulder).
2. Satisfies ``|B_final − A_subj| ≤ |B_prefit − A_subj|`` with equality.
3. Removes the 17–20 mm left/right bind-Δ asymmetry on knee/elbow.
4. Preserves V7 flex contact (full A_subj snap opens the medial gap).

Pose authority remains hybrid ``joint_anchored_fk_v10`` (identity-142
hand/foot).  Station L/R femur span asymmetry (~19 mm) is real SMPL-X male
geometry; A_subj (migrated Node1) is the symmetric anatomical target.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _global_to_local,
    _weighted_rest_correction,
)
from .segment_similarity_rest_v10 import subject_anatomical_pivots_v10


ANCHORED_REST_V11_KIND = "AnchoredRestFitV11"
ANCHORED_REST_V11_METHOD = "prefit_hinge_origin_restore_v11"

# Controllers whose V7 translation was corrupted by elbow=humerus / knee-ray.
# Femur_Rot is intentionally NOT restored: undoing the V7 hip seat breaks
# posed femur containment and the hybrid FK flex gap.  Hips stay at V7
# (symmetric ~10.5 mm from A) and are gated by non-regression vs V7.
HINGE_RESTORE_CONTROLLERS = (
    "Knee_Rotate_L",
    "Knee_Rotate_R",
    "Patella_Rotate_L",
    "Patella_Rotate_R",
    "Elbow_Rot_L",
    "Elbow_Rot_R",
    "Forearm_Bone_L",
    "Forearm_Bone_R",
    "Forearm_Twist_L",
    "Forearm_Twist_R",
    "Shoulder_Rotate_L",
    "Shoulder_Rotate_R",
)

# Leaving the arm chain on the V7 seat is not a legal option: it fails V11's
# own ``rest_anatomical_anchor_v11`` and ``lr_symmetry_v11``, which is the
# whole reason the elbow was restored.  Kept only so that failure is
# reproducible, not as a shipping preset.
LEG_ONLY_HINGE_RESTORE_CONTROLLERS = (
    "Knee_Rotate_L",
    "Knee_Rotate_R",
    "Patella_Rotate_L",
    "Patella_Rotate_R",
)

HINGE_RESTORE_PRESETS = {
    "full": HINGE_RESTORE_CONTROLLERS,
    "leg_only": LEG_ONLY_HINGE_RESTORE_CONTROLLERS,
}

# Carrying the whole mesh with the restored bind fails knee containment: V7's
# leg mesh placement is load-bearing for condyle-plateau contact.  Carrying
# nothing leaves the elbow bind describing a different joint than the elbow
# geometry.  The arm chain is the only part that needs to move, so the carry
# is restricted to it and blended by LBS weight so the shoulder seam does not
# tear.
ARM_CARRY_CONTROLLERS = (
    "Elbow_Rot_L",
    "Elbow_Rot_R",
    "Forearm_Bone_L",
    "Forearm_Bone_R",
    "Forearm_Twist_L",
    "Forearm_Twist_R",
)

CARRY_MESH_PRESETS = {
    "none": (),
    "arm": ARM_CARRY_CONTROLLERS,
    "all": None,
}

# Joints that must meet |B_final−A| ≤ |B_prefit−A| (OSSO Ej hard).
ANATOMICAL_ANCHOR_HARD_JOINTS = (
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

# Hips keep V7 seating; hard rule is non-regression vs V7.
ANATOMICAL_ANCHOR_V7_NONREGRESS_JOINTS = ("left_hip", "right_hip")

SYMMETRY_PAIRS = (
    ("Knee_Rotate_L", "Knee_Rotate_R"),
    ("Elbow_Rot_L", "Elbow_Rot_R"),
    ("Patella_Rotate_L", "Patella_Rotate_R"),
    ("Forearm_Bone_L", "Forearm_Bone_R"),
    ("Femur_Rot_L", "Femur_Rot_R"),
    ("Shoulder_Rotate_L", "Shoulder_Rotate_R"),
)


def _controller_index(names: list[str], controller: str) -> int:
    if controller not in names:
        raise ValueError(f"missing controller: {controller}")
    return names.index(controller)


def build_anchored_rest_fit_v11(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    restore_controllers: tuple[str, ...] = HINGE_RESTORE_CONTROLLERS,
    carry_mesh_controllers: tuple[str, ...] | None = (),
) -> ChainRestFitSubjectV1:
    """Return V7 rest with hinge origins restored to prefit (elbow split).

    ``carry_mesh_controllers`` (V12a) re-derives ``vertices_final`` from the
    restored ``C_bone`` for the listed controllers instead of keeping the V7
    mesh everywhere; ``None`` carries the whole body, ``()`` is V11.

    V11 kept the entire V7 mesh to preserve its contact geometry, but that
    leaves the bind and the geometry describing different elbows: the forearm
    rotates about a pivot ~15 mm from the mesh condyle, which is where the
    left-forearm poke-through and the 17.7 mm station-to-hinge-axis error come
    from.  Carrying everything instead fails knee containment, because V7's
    leg mesh placement is load-bearing.  Hence the per-controller selection,
    blended by LBS weight so no seam tears.
    """

    started = time.perf_counter()
    names = [str(n) for n in asset.source_bone_names]
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    b_prefit = np.asarray(value.B_prefit, dtype=np.float64)
    b_v7 = np.asarray(value.B_final, dtype=np.float64)
    b_final = b_v7.copy()
    a_subj = subject_anatomical_pivots_v10(asset, calibration)

    restored: dict[str, Any] = {}
    for controller in restore_controllers:
        index = _controller_index(names, controller)
        before = b_final[index, :3, 3].copy()
        b_final[index, :3, 3] = b_prefit[index, :3, 3]
        restored[controller] = {
            "translation_restored_m": (b_prefit[index, :3, 3] - before).tolist(),
            "delta_norm_m": float(np.linalg.norm(b_prefit[index, :3, 3] - before)),
        }

    # Only meaningful when the elbow was restored; otherwise the arm chain is
    # deliberately left on the V7 seat, elbow=humerus lever included.
    for suffix in ("L", "R"):
        if f"Elbow_Rot_{suffix}" not in restore_controllers:
            continue
        elbow = _controller_index(names, f"Elbow_Rot_{suffix}")
        shoulder = _controller_index(names, f"Shoulder_Rotate_{suffix}")
        if np.allclose(b_final[elbow], b_final[shoulder], atol=1.0e-9):
            raise ValueError(
                f"Elbow_Rot_{suffix} still matches Shoulder_Rotate_{suffix} after split"
            )

    c_bone = b_final @ np.linalg.inv(b_prefit)
    target_local = _global_to_local(b_final, parents)
    inverse = np.linalg.inv(b_final)

    vertices_final = np.asarray(value.vertices_final, dtype=np.float32)
    mesh_shift_m = 0.0
    if carry_mesh_controllers is None or len(carry_mesh_controllers):
        driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)
        driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
        carried = _weighted_rest_correction(
            np.asarray(value.vertices_prefit, dtype=np.float64),
            driver_indices,
            driver_weights,
            c_bone,
        )
        base = np.asarray(vertices_final, dtype=np.float64)
        if carry_mesh_controllers is None:
            blended = carried
        else:
            selected = {
                _controller_index(names, controller)
                for controller in carry_mesh_controllers
            }
            member = np.isin(driver_indices, sorted(selected))
            # Fraction of a vertex's LBS mass that sits on carried
            # controllers, so the transition is as smooth as the skinning.
            alpha = np.sum(driver_weights * member, axis=1)
            alpha = np.clip(alpha, 0.0, 1.0)[:, None]
            blended = (1.0 - alpha) * base + alpha * carried
        mesh_shift_m = float(np.max(np.linalg.norm(blended - base, axis=1)))
        vertices_final = np.asarray(blended, dtype=np.float32)

    joint_rows: dict[str, Any] = {}
    for joint_index, spec in enumerate(JOINT_SPECS):
        ctrl = _controller_index(names, spec.controller)
        origin_a = a_subj[joint_index, :3, 3]
        d_pre = float(np.linalg.norm(b_prefit[ctrl, :3, 3] - origin_a))
        d_v7 = float(np.linalg.norm(b_v7[ctrl, :3, 3] - origin_a))
        d_final = float(np.linalg.norm(b_final[ctrl, :3, 3] - origin_a))
        joint_rows[spec.name] = {
            "controller": spec.controller,
            "prefit_to_anatomical_m": d_pre,
            "v7_to_anatomical_m": d_v7,
            "final_to_anatomical_m": d_final,
            "anchor_ok": bool(d_final <= d_pre + 1.0e-9),
        }

    symmetry_rows: dict[str, Any] = {}
    for left, right in SYMMETRY_PAIRS:
        li = _controller_index(names, left)
        ri = _controller_index(names, right)
        d_l = float(np.linalg.norm(b_final[li, :3, 3] - b_prefit[li, :3, 3]))
        d_r = float(np.linalg.norm(b_final[ri, :3, 3] - b_prefit[ri, :3, 3]))
        symmetry_rows[f"{left}/{right}"] = {
            "left_bind_delta_m": d_l,
            "right_bind_delta_m": d_r,
            "abs_delta_diff_m": abs(d_l - d_r),
        }

    report = dict(value.build_report)
    report.update(
        {
            "schema_version": 11,
            "artifact_kind": ANCHORED_REST_V11_KIND,
            "method": ANCHORED_REST_V11_METHOD,
            "accepted_scope": "full_main_chain_shadow_v11",
            "elbow_policy": (
                "independent_of_humerus_prefit_origin"
                if "Elbow_Rot_L" in restore_controllers
                else "left_on_v7_seat"
            ),
            "knee_policy": "prefit_origin_restore_keep_v7_mesh",
            "hip_policy": "keep_v7_femur_origin_for_containment",
            "mesh_policy_note": (
                "vertices_final unchanged from V7 (contact-preserving)"
                if carry_mesh_controllers == ()
                else "vertices_final carried with the restored C_bone by LBS weight"
            ),
            "carry_mesh_controllers": (
                None
                if carry_mesh_controllers is None
                else list(carry_mesh_controllers)
            ),
            "mesh_shift_vs_v7_max_m": mesh_shift_m,
            "hinge_restore_controllers": list(restore_controllers),
            "hinge_restore": restored,
            "anatomical_anchor": joint_rows,
            "anatomical_anchor_hard_joints": list(ANATOMICAL_ANCHOR_HARD_JOINTS),
            "anatomical_anchor_v7_nonregress_joints": list(
                ANATOMICAL_ANCHOR_V7_NONREGRESS_JOINTS
            ),
            "lr_symmetry": symmetry_rows,
            "station_femur_asymmetry_note": (
                "SMPL-X male station hip→knee L/R differs ~19 mm; "
                "A_subj (Node1 migrate) is nearly symmetric — not a derivation bug"
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    return replace(
        value,
        vertices_final=vertices_final,
        B_final=b_final.astype(np.float64),
        C_bone=c_bone.astype(np.float64),
        target_local_bind=target_local.astype(np.float64),
        inverse_bind=inverse.astype(np.float64),
        build_report=report,
    )


def anatomical_bind_distances_v11(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    bind: str = "final",
) -> dict[str, float]:
    """Return ``|B[controller].origin − A_subj|`` for the 12 joints (metres)."""

    names = [str(n) for n in asset.source_bone_names]
    a_subj = subject_anatomical_pivots_v10(asset, calibration)
    if bind == "final":
        matrices = np.asarray(value.B_final, dtype=np.float64)
    elif bind == "prefit":
        matrices = np.asarray(value.B_prefit, dtype=np.float64)
    else:
        raise ValueError(f"unknown bind={bind!r}")
    out: dict[str, float] = {}
    for joint_index, spec in enumerate(JOINT_SPECS):
        ctrl = _controller_index(names, spec.controller)
        out[spec.name] = float(
            np.linalg.norm(matrices[ctrl, :3, 3] - a_subj[joint_index, :3, 3])
        )
    return out


__all__ = [
    "ANATOMICAL_ANCHOR_HARD_JOINTS",
    "ANATOMICAL_ANCHOR_V7_NONREGRESS_JOINTS",
    "ANCHORED_REST_V11_KIND",
    "ANCHORED_REST_V11_METHOD",
    "ARM_CARRY_CONTROLLERS",
    "CARRY_MESH_PRESETS",
    "HINGE_RESTORE_CONTROLLERS",
    "HINGE_RESTORE_PRESETS",
    "LEG_ONLY_HINGE_RESTORE_CONTROLLERS",
    "SYMMETRY_PAIRS",
    "anatomical_bind_distances_v11",
    "build_anchored_rest_fit_v11",
]
