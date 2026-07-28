"""Measure V7 controller and local-FK observations from final posed state.

Older joint reports copied metadata ``pass`` flags and refit landmarks after
subject fitting, so a bad fit could redefine its own probes.  This module
recomputes every gated quantity from the posed bone matrices and the final
posed vertex array; asset metadata may supply pose coefficients only, never a
boolean acceptance flag.

Hip controller gating uses ``direction_error_deg`` (posed femur head→knee
pivot vs SMPL-X hip→knee), limit unchanged at 1.0 deg.  The hinge-constrained
leg solve deliberately twists the femur about its long axis to keep the
authored knee hinge on the drive plane; that residual is reported as
``axial_twist_deg`` and is not gated.  ``rotation_error_deg`` remains in the
payload for continuity with older reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .anatomy_lbs import joint_global_transforms, source_bone_posed_global
from .joint_contact_v7 import (
    REQUIRED_LOCAL_FK_LINKS,
    SIDES,
    evaluate_controller_gate_v7,
    evaluate_local_fk_gate_v7,
    fit_sphere_fixed_radius_v7,
    fit_sphere_v7,
)

FK_OBSERVATION_SCHEMA_VERSION = 7

# Leg link keys must match REQUIRED_LOCAL_FK_LINKS exactly.
_LEG_LINK_SPECS = (
    ("Femur_Rot", "Knee_Rotate", "femur_knee"),
    ("Knee_Rotate", "Tibia_Bone", "knee_tibia"),
    ("Tibia_Bone", "Patella_Rotate", "tibia_patella"),
)

# Asset lookup found only Elbow_Rot_{S}; Humerus/Ulna/Radius bases are absent
# and those links report available=False rather than inventing Shoulder/Forearm.
# Authored arm chain, read off the source rig: Shoulder_Rotate > Elbow_Rot >
# Forearm_Bone > Forearm_Twist, carrying the Humerus, Ulna and Radius meshes in
# that order. An earlier revision looked for Humerus_Rot/Ulna_Bone/Radius_Bone,
# which exist in no source rig, so every arm link reported "bone is absent".
_ARM_LINK_SPECS = (
    ("Shoulder_Rotate", "Elbow_Rot", "humerus_elbow"),
    ("Elbow_Rot", "Forearm_Bone", "elbow_ulna"),
    ("Forearm_Bone", "Forearm_Twist", "ulna_radius"),
)


@dataclass(frozen=True)
class FkReferenceV7:
    """Independent reference for authorized degrees of freedom."""

    patella_response_rad: Callable[[str, float], float] | None
    patella_axis_local: Mapping[str, np.ndarray] | None
    knee_axis_local: Mapping[str, np.ndarray] | None
    # The trochlear corridor correction is an authorized translation, in the same
    # sense as the tibia glide bound: it must come from the frozen oracle plus the
    # subject rest geometry, never from a coefficient stored in the candidate.
    patella_translation_parent_local: Callable[[str, float], np.ndarray] | None = None
    patella_translation_bound_m: float = 0.0
    tibia_glide_bound_m: float = 0.001
    source: str = ""


def default_fk_reference_v7() -> FkReferenceV7:
    """Fail-closed reference: every reference-dependent rotation is unavailable."""
    return FkReferenceV7(
        patella_response_rad=None,
        patella_axis_local=None,
        knee_axis_local=None,
        source="",
    )


def fk_reference_from_patella_oracle_v7(
    law: Any,
    *,
    contact_translations: Mapping[str, np.ndarray] | None = None,
) -> FkReferenceV7:
    """Adapt a ``PatellaOracleLawV7`` into an observation reference.

    ``contact_translations`` maps a side to the ``[K,3]`` parent-local corridor
    table recomputed by the caller from this law and the subject rest geometry.
    """
    try:
        from . import patella_oracle_v7 as _patella_oracle_v7  # noqa: F401
    except Exception as exc:  # pragma: no cover - sibling may arrive later
        raise ValueError(
            "patella_oracle_v7 is required to build an FkReferenceV7 from an oracle law"
        ) from exc

    def _response(side: str, flexion_rad: float) -> float:
        return float(law.response_rad(side, float(flexion_rad)))

    axes_patella = {
        str(side): np.asarray(law.axis_patella_local[side], dtype=np.float64).reshape(3)
        for side in SIDES
    }
    axes_knee = {
        str(side): np.asarray(law.axis_knee_local[side], dtype=np.float64).reshape(3)
        for side in SIDES
    }
    tables: dict[str, np.ndarray] = {}
    if contact_translations is not None:
        knots = np.asarray(law.knots_deg, dtype=np.float64).reshape(-1)
        for side, table in dict(contact_translations).items():
            values = np.asarray(table, dtype=np.float64)
            if values.shape != (len(knots), 3) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"contact_translations[{side!r}] must be finite [K,3] matching the law knots"
                )
            tables[str(side)] = values

    def _translation(side: str, flexion_rad: float) -> np.ndarray:
        table = tables.get(str(side))
        if table is None:
            raise ValueError(f"no oracle contact translation table for side {side!r}")
        knots = np.asarray(law.knots_deg, dtype=np.float64).reshape(-1)
        angle = float(np.degrees(max(float(flexion_rad), 0.0)))
        return np.asarray(
            [np.interp(angle, knots, table[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )

    return FkReferenceV7(
        patella_response_rad=_response,
        patella_axis_local=axes_patella,
        knee_axis_local=axes_knee,
        patella_translation_parent_local=_translation if tables else None,
        patella_translation_bound_m=float(law.max_contact_translation_m),
        source=str(law.content_digest()),
    )


def _side_suffix(side: str) -> str:
    if side == "left":
        return "L"
    if side == "right":
        return "R"
    raise ValueError(f"side must be left or right, got {side!r}")


def _bone_index(names: Sequence[str], bone_name: str) -> int:
    try:
        return list(names).index(bone_name)
    except ValueError as exc:
        raise ValueError(f"required source bone {bone_name!r} is missing") from exc


def _bone_index_optional(names: Sequence[str], bone_name: str) -> int | None:
    try:
        return list(names).index(bone_name)
    except ValueError:
        return None


def _as_vertices(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{label} must be a non-empty [N,3] array")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return points


def _as_matrices(value: Any, label: str, count: int) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.shape != (count, 4, 4):
        raise ValueError(f"{label} must have shape [{count},4,4], got {matrices.shape}")
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{label} contains a non-finite entry")
    return matrices


def _rotation_angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    delta = Rotation.from_matrix(np.asarray(ra, dtype=np.float64)) * Rotation.from_matrix(
        np.asarray(rb, dtype=np.float64)
    ).inv()
    return float(np.degrees(np.linalg.norm(delta.as_rotvec())))


def _acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    ua = np.asarray(a, dtype=np.float64).reshape(3)
    ub = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(ua))
    nb = float(np.linalg.norm(ub))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    cosine = float(np.clip(np.dot(ua / na, ub / nb), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(abs(cosine))))
    return angle


def _apply_rigid(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


def _orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Reflection-corrected orthogonal Procrustes mapping source onto target."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("Procrustes inputs must be matching [N,3] arrays with N>=3")
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    covariance = (src - src_center).T @ (dst - dst_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    aligned = (src - src_center) @ rotation.T + dst_center
    residual = dst - aligned
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rotation.astype(np.float64), rms


def _unit_axis(value: Any, label: str) -> np.ndarray | None:
    if value is None:
        return None
    axis = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{label} contains a non-finite entry")
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return None
    return axis / norm


def _observation(
    *,
    translation_error_m: float | None,
    rotation_error_deg: float | None,
    available: bool,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "translation_error_m": (
            None if translation_error_m is None else float(translation_error_m)
        ),
        "rotation_error_deg": (
            None if rotation_error_deg is None else float(rotation_error_deg)
        ),
        "available": bool(available),
        "reason": "" if available else str(reason),
    }
    payload.update(extra)
    return payload


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return _observation(
        translation_error_m=None,
        rotation_error_deg=None,
        available=False,
        reason=reason,
        **extra,
    )


def _local_pair(
    parent: np.ndarray,
    child: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(parent) @ child


def _on_axis_decomposition(
    posed_local: np.ndarray,
    bind_local: np.ndarray,
    axis_child_local: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return (on_axis_rad, off_axis_rad, hinge_axis_parent_local)."""
    axis_child = _unit_axis(axis_child_local, "hinge axis")
    if axis_child is None:
        raise ValueError("hinge axis is degenerate")
    r_bind = bind_local[:3, :3]
    r_posed = posed_local[:3, :3]
    # Parent-local expression of the child-local hinge axis.
    axis_parent = r_bind @ axis_child
    axis_parent = axis_parent / max(float(np.linalg.norm(axis_parent)), 1.0e-12)
    q = r_posed @ r_bind.T
    rotvec = Rotation.from_matrix(q).as_rotvec()
    on_axis = float(np.dot(rotvec, axis_parent))
    perp = rotvec - on_axis * axis_parent
    off_axis = float(np.linalg.norm(perp))
    return on_axis, off_axis, axis_parent


def _authorized_child_rotation(
    bind_local: np.ndarray,
    axis_child_local: np.ndarray | None,
    angle_rad: float,
) -> np.ndarray:
    if axis_child_local is None or abs(float(angle_rad)) <= 0.0:
        return np.asarray(bind_local[:3, :3], dtype=np.float64)
    axis = _unit_axis(axis_child_local, "authorized axis")
    if axis is None:
        raise ValueError("authorized axis is degenerate")
    auth = Rotation.from_rotvec(axis * float(angle_rad)).as_matrix()
    return np.asarray(bind_local[:3, :3], dtype=np.float64) @ auth


def _fit_acetabulum_center(
    vertices: np.ndarray,
    domains: Any,
    side: str,
) -> tuple[np.ndarray | None, str]:
    head = domains.require(f"{side}/femoral_head")
    socket = domains.require(f"{side}/acetabulum")
    head_fit = fit_sphere_v7(vertices[head])
    if not head_fit["available"]:
        return None, str(head_fit.get("reason") or "femoral head sphere unavailable")
    socket_fit = fit_sphere_fixed_radius_v7(
        vertices[socket],
        radius_m=float(head_fit["radius_m"]),
    )
    if not socket_fit["available"]:
        return None, str(socket_fit.get("reason") or "acetabulum sphere unavailable")
    center = np.asarray(socket_fit["center"], dtype=np.float64).reshape(3)
    return center, ""


def _observe_hip(
    *,
    side: str,
    femur_index: int,
    knee_index: int,
    hip_joint: int,
    knee_joint: int,
    pg: np.ndarray,
    bg: np.ndarray,
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    domains: Any,
    pose_points: np.ndarray,
    target_joint_delta: np.ndarray,
) -> dict[str, Any]:
    c_rest, rest_reason = _fit_acetabulum_center(rest_vertices, domains, side)
    c_posed, posed_reason = _fit_acetabulum_center(posed_vertices, domains, side)
    delta = pg[femur_index] @ np.linalg.inv(bg[femur_index])
    femur_idx = domains.require(f"{side}/femur")
    if len(femur_idx) < 3:
        return _unavailable(
            f"{side}/femur domain needs at least three vertices",
            socket_center_rest_m=None if c_rest is None else c_rest.tolist(),
            socket_center_posed_m=None if c_posed is None else c_posed.tolist(),
            femur_procrustes_rms_m=None,
            direction_error_deg=None,
            axial_twist_deg=None,
        )
    procrustes_r, procrustes_rms = _orthogonal_procrustes(
        rest_vertices[femur_idx],
        posed_vertices[femur_idx],
    )
    rotation_error = _rotation_angle_deg(delta[:3, :3], procrustes_r)

    femur_dir = np.asarray(
        pg[knee_index][:3, 3] - pg[femur_index][:3, 3], dtype=np.float64
    )
    bind_dir = np.asarray(
        bg[knee_index][:3, 3] - bg[femur_index][:3, 3], dtype=np.float64
    )
    smplx_dir = np.asarray(
        pose_points[knee_joint] - pose_points[hip_joint], dtype=np.float64
    )
    femur_norm = float(np.linalg.norm(femur_dir))
    bind_norm = float(np.linalg.norm(bind_dir))
    smplx_norm = float(np.linalg.norm(smplx_dir))
    if femur_norm <= 1.0e-12 or smplx_norm <= 1.0e-12 or bind_norm <= 1.0e-12:
        direction_error = None
        axial_twist = None
    else:
        femur_u = femur_dir / femur_norm
        bind_u = bind_dir / bind_norm
        smplx_u = smplx_dir / smplx_norm
        # Neutral pose recovers the fitted bind exactly (leg hinge solve is
        # inactive), so gate against the bind femur direction.  Under a
        # nonzero pose the hinge solve tracks the SMPL-X hip→knee direction.
        at_bind = bool(
            np.allclose(pg[femur_index], bg[femur_index], atol=2.0e-6, rtol=0.0)
            and np.allclose(
                pg[knee_index, :3, 3], bg[knee_index, :3, 3], atol=2.0e-6, rtol=0.0
            )
        )
        target_u = bind_u if at_bind else smplx_u
        direction_error = float(
            np.degrees(
                np.arccos(float(np.clip(np.dot(femur_u, target_u), -1.0, 1.0)))
            )
        )
        # Residual rotation of the femur rigid delta vs the SMPL-X hip delta,
        # projected onto the posed femur long axis (authorized ball-joint DoF).
        smplx_delta_r = np.asarray(
            target_joint_delta[hip_joint, :3, :3], dtype=np.float64
        )
        relative = smplx_delta_r.T @ delta[:3, :3]
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        axial_twist = float(np.degrees(float(np.dot(rotvec, femur_u))))

    extras = {
        "socket_center_rest_m": None if c_rest is None else c_rest.tolist(),
        "socket_center_posed_m": None if c_posed is None else c_posed.tolist(),
        "femur_procrustes_rms_m": procrustes_rms,
        "direction_error_deg": direction_error,
        "axial_twist_deg": axial_twist,
    }
    if c_rest is None or c_posed is None:
        reason = rest_reason or posed_reason or "acetabulum centre unavailable"
        return _observation(
            translation_error_m=None,
            rotation_error_deg=rotation_error,
            available=False,
            reason=reason,
            **extras,
        )
    predicted = _apply_rigid(delta, c_rest)
    translation_error = float(np.linalg.norm(predicted - c_posed))
    if direction_error is None:
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=rotation_error,
            available=False,
            reason="femur or SMPL-X hip→knee direction is degenerate",
            **extras,
        )
    return _observation(
        translation_error_m=translation_error,
        rotation_error_deg=rotation_error,
        available=True,
        reason="",
        **extras,
    )


def _observe_knee(
    *,
    side: str,
    femur_index: int,
    knee_index: int,
    pg: np.ndarray,
    bg: np.ndarray,
    posed_vertices: np.ndarray,
    domains: Any,
    reference: FkReferenceV7,
    flexion_clamped_deg: float | None = None,
) -> dict[str, Any]:
    posed_local = _local_pair(pg[femur_index], pg[knee_index])
    bind_local = _local_pair(bg[femur_index], bg[knee_index])
    translation_error = float(
        np.linalg.norm(posed_local[:3, 3] - bind_local[:3, 3])
    )
    knee_axes = reference.knee_axis_local
    if knee_axes is None or side not in knee_axes:
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=None,
            available=False,
            reason="knee hinge axis reference is missing",
            flexion_deg=None,
            flexion_clamped_deg=flexion_clamped_deg,
            hinge_axis_parent_local=None,
            epicondylar_axis_error_deg=None,
        )
    axis_local = _unit_axis(knee_axes[side], f"knee_axis_local[{side}]")
    if axis_local is None:
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=None,
            available=False,
            reason=f"knee_axis_local[{side}] is degenerate",
            flexion_deg=None,
            flexion_clamped_deg=flexion_clamped_deg,
            hinge_axis_parent_local=None,
            epicondylar_axis_error_deg=None,
        )
    on_axis, off_axis, axis_parent = _on_axis_decomposition(
        posed_local, bind_local, axis_local
    )
    lateral = domains.require(f"{side}/femoral_condyle_lateral")
    medial = domains.require(f"{side}/femoral_condyle_medial")
    epicondylar = np.mean(posed_vertices[lateral], axis=0) - np.mean(
        posed_vertices[medial], axis=0
    )
    hinge_world = pg[knee_index][:3, :3] @ axis_local
    epicondylar_error = _acute_angle_deg(hinge_world, epicondylar)
    return _observation(
        translation_error_m=translation_error,
        rotation_error_deg=float(np.degrees(off_axis)),
        available=True,
        reason="",
        flexion_deg=float(np.degrees(on_axis)),
        flexion_clamped_deg=flexion_clamped_deg,
        hinge_axis_parent_local=axis_parent.tolist(),
        epicondylar_axis_error_deg=epicondylar_error,
    )


def _link_key(side: str, parent_base: str, child_base: str) -> str:
    return f"{side}/{parent_base}>{child_base}"


def _arm_reference_authorized_angle_rad(
    link_kind: str,
    reference: FkReferenceV7,
) -> tuple[float | None, str]:
    """Extension point for independent arm DoF references.

    ``Ulna>Radius`` is treated as rigid (authorized angle 0).  Proximal arm
    links stay unavailable until an independent arm reference is supplied.
    """
    del reference  # reserved for a future arm oracle
    if link_kind == "ulna_radius":
        return 0.0, ""
    return None, "no independent arm reference"


def _observe_local_link(
    *,
    side: str,
    parent_index: int,
    child_index: int,
    pg: np.ndarray,
    bg: np.ndarray,
    link_kind: str,
    reference: FkReferenceV7,
    knee_flexion_rad: float | None,
) -> dict[str, Any]:
    posed_local = _local_pair(pg[parent_index], pg[child_index])
    bind_local = _local_pair(bg[parent_index], bg[child_index])
    translation_error = float(
        np.linalg.norm(posed_local[:3, 3] - bind_local[:3, 3])
    )
    bind_t = bind_local[:3, 3].tolist()
    posed_t = posed_local[:3, 3].tolist()
    extras = {
        "flexion_deg": None,
        "authorized_angle_deg": None,
        "off_axis_residual_deg": None,
        "response_error_deg": None,
        "bind_local_translation_m": bind_t,
        "posed_local_translation_m": posed_t,
    }

    if link_kind == "femur_knee":
        knee_axes = reference.knee_axis_local
        if knee_axes is None or side not in knee_axes:
            return _observation(
                translation_error_m=translation_error,
                rotation_error_deg=None,
                available=False,
                reason="knee hinge axis reference is missing",
                **extras,
            )
        axis = _unit_axis(knee_axes[side], f"knee_axis_local[{side}]")
        if axis is None:
            return _observation(
                translation_error_m=translation_error,
                rotation_error_deg=None,
                available=False,
                reason=f"knee_axis_local[{side}] is degenerate",
                **extras,
            )
        on_axis, off_axis, _axis_parent = _on_axis_decomposition(
            posed_local, bind_local, axis
        )
        extras.update(
            {
                "flexion_deg": float(np.degrees(on_axis)),
                "off_axis_residual_deg": float(np.degrees(off_axis)),
            }
        )
        # Taking the authorized angle from the candidate's own on-axis angle made
        # the reference a function of the thing under test: the rotation error was
        # then only the off-axis residual, which the runtime drives to ~1e-6 deg by
        # construction, and response_error_deg was a hardcoded 0.0. The knee flexion
        # is solved by the leg IK rather than read from the drive, so no independent
        # authorized angle exists yet; per spec 2.3 that is recorded unavailable.
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=None,
            available=False,
            reason=(
                "no independent knee flexion reference: the authorized angle "
                "would have to come from the candidate's own posed bones"
            ),
            **extras,
        )

    if link_kind == "knee_tibia":
        r_ref = bind_local[:3, :3]
        rotation_error = _rotation_angle_deg(posed_local[:3, :3], r_ref)
        extras.update(
            {
                "flexion_deg": 0.0 if knee_flexion_rad is None else float(
                    np.degrees(knee_flexion_rad)
                ),
                "authorized_angle_deg": 0.0,
                "off_axis_residual_deg": rotation_error,
                "response_error_deg": None,
            }
        )
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=rotation_error,
            available=True,
            reason="",
            **extras,
        )

    if link_kind == "tibia_patella":
        if reference.patella_response_rad is None or reference.patella_axis_local is None:
            return _observation(
                translation_error_m=translation_error,
                rotation_error_deg=None,
                available=False,
                reason="patella oracle reference is missing",
                **extras,
            )
        if side not in reference.patella_axis_local:
            return _observation(
                translation_error_m=translation_error,
                rotation_error_deg=None,
                available=False,
                reason=f"patella_axis_local[{side}] is missing",
                **extras,
            )
        if knee_flexion_rad is None:
            return _observation(
                translation_error_m=translation_error,
                rotation_error_deg=None,
                available=False,
                reason="knee flexion is unavailable for patella coupling",
                **extras,
            )
        axis = _unit_axis(
            reference.patella_axis_local[side],
            f"patella_axis_local[{side}]",
        )
        if axis is None:
            return _observation(
                translation_error_m=translation_error,
                rotation_error_deg=None,
                available=False,
                reason=f"patella_axis_local[{side}] is degenerate",
                **extras,
            )
        auth_angle = float(reference.patella_response_rad(side, float(knee_flexion_rad)))
        authorized_translation = np.zeros(3, dtype=np.float64)
        if reference.patella_translation_parent_local is not None:
            authorized_translation = np.asarray(
                reference.patella_translation_parent_local(
                    side, float(knee_flexion_rad)
                ),
                dtype=np.float64,
            ).reshape(3)
            bound = float(reference.patella_translation_bound_m)
            if bound > 0.0 and float(np.linalg.norm(authorized_translation)) > bound + 1.0e-9:
                return _observation(
                    translation_error_m=translation_error,
                    rotation_error_deg=None,
                    available=False,
                    reason="oracle corridor translation exceeds its frozen bound",
                    **extras,
                )
        translation_error = float(
            np.linalg.norm(
                posed_local[:3, 3] - bind_local[:3, 3] - authorized_translation
            )
        )
        on_axis, off_axis, _axis_parent = _on_axis_decomposition(
            posed_local, bind_local, axis
        )
        r_ref = _authorized_child_rotation(bind_local, axis, auth_angle)
        rotation_error = _rotation_angle_deg(posed_local[:3, :3], r_ref)
        extras.update(
            {
                "flexion_deg": float(np.degrees(knee_flexion_rad)),
                "authorized_angle_deg": float(np.degrees(auth_angle)),
                "off_axis_residual_deg": float(np.degrees(off_axis)),
                "response_error_deg": float(np.degrees(on_axis - auth_angle)),
                "authorized_translation_m": authorized_translation.tolist(),
                "authorized_translation_norm_m": float(
                    np.linalg.norm(authorized_translation)
                ),
            }
        )
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=rotation_error,
            available=True,
            reason="",
            **extras,
        )

    # Arm links.
    auth_angle, auth_reason = _arm_reference_authorized_angle_rad(link_kind, reference)
    if auth_angle is None:
        return _observation(
            translation_error_m=translation_error,
            rotation_error_deg=None,
            available=False,
            reason=auth_reason,
            **extras,
        )
    r_ref = _authorized_child_rotation(bind_local, None, float(auth_angle))
    rotation_error = _rotation_angle_deg(posed_local[:3, :3], r_ref)
    extras.update(
        {
            "flexion_deg": None,
            "authorized_angle_deg": float(np.degrees(auth_angle)),
            "off_axis_residual_deg": rotation_error,
            "response_error_deg": None,
        }
    )
    return _observation(
        translation_error_m=translation_error,
        rotation_error_deg=rotation_error,
        available=True,
        reason="",
        **extras,
    )


def observe_fk_v7(
    asset: Any,
    *,
    pose_axis_angle: Any,
    posed_vertices: Any,
    domains: Any,
    reference: FkReferenceV7,
    faces: Any = None,
) -> dict[str, Any]:
    """Measure controller and local-FK observations from posed bones and mesh."""
    del faces  # reserved for optional topology checks by callers
    if asset.source_bone_names is None:
        raise ValueError("observe_fk_v7 requires schema-v6 source_bone_names")
    names = list(asset.source_bone_names)
    bone_count = len(names)
    if asset.source_bone_parents is None:
        raise ValueError("observe_fk_v7 requires source_bone_parents")
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if parents.shape != (bone_count,):
        raise ValueError("source_bone_parents shape does not match source_bone_names")

    rest_vertices = _as_vertices(asset.vertices_rest, "vertices_rest")
    posed = _as_vertices(posed_vertices, "posed_vertices")
    if posed.shape != rest_vertices.shape:
        raise ValueError("posed_vertices must match vertices_rest shape")

    # Prefer the subject's baked (posterior-positive) knee hinge axes over the
    # oracle's raw axes when they differ by a bake-time sign flip.
    metadata = getattr(asset, "metadata", None) or {}
    hinge_splines = metadata.get("source_knee_hinge_splines_v7") or {}
    subject_knee_axes: dict[str, np.ndarray] = {}
    if isinstance(hinge_splines, dict):
        for _key, spline in hinge_splines.items():
            if not isinstance(spline, dict):
                continue
            side = str(spline.get("side", ""))
            axis = np.asarray(spline.get("axis_local", []), dtype=np.float64).reshape(-1)
            if side in SIDES and axis.shape == (3,) and np.all(np.isfinite(axis)):
                subject_knee_axes[side] = axis
    if subject_knee_axes:
        merged = dict(reference.knee_axis_local or {})
        merged.update(subject_knee_axes)
        reference = FkReferenceV7(
            patella_response_rad=reference.patella_response_rad,
            patella_axis_local=reference.patella_axis_local,
            knee_axis_local=merged,
            patella_translation_parent_local=reference.patella_translation_parent_local,
            patella_translation_bound_m=reference.patella_translation_bound_m,
            tibia_glide_bound_m=reference.tibia_glide_bound_m,
            source=reference.source,
        )

    bg = _as_matrices(asset.target_bind_global, "target_bind_global", bone_count)
    pg = _as_matrices(
        source_bone_posed_global(asset, pose_axis_angle),
        "posed_global",
        bone_count,
    )
    target_pose_global = joint_global_transforms(
        pose_axis_angle=pose_axis_angle,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    target_rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    target_joint_delta = target_pose_global @ np.linalg.inv(target_rest_global)
    pose_points = target_pose_global[:, :3, 3]

    pose_array = np.asarray(pose_axis_angle)
    notes: list[str] = []
    controller: dict[str, Any] = {}
    local_fk: dict[str, Any] = {}
    knee_flexion: dict[str, float | None] = {side: None for side in SIDES}

    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        solve_leg_hinge_v1,
        source_bone_driver_frames,
        _validate_leg_hinge_solve_entry_v1,
    )
    from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
        pose_to_smplx55_axis_angle,
    )

    input_pose = pose_to_smplx55_axis_angle(pose_axis_angle).astype(np.float64)
    leg_solve_raw = metadata.get("source_leg_hinge_solve_v1") or {}
    clamp_by_side: dict[str, float] = {}
    if (
        isinstance(leg_solve_raw, dict)
        and leg_solve_raw
        and bool(np.any(input_pose))
        and getattr(asset, "source_driver_coupling", None) is not None
    ):
        contact_rest = np.asarray(
            asset.source_driver_rest_joints
            if asset.source_driver_rest_joints is not None
            else asset.rest_joints,
            dtype=np.float64,
        )
        contact_delta = target_joint_delta.copy()
        for joint, parent_j in enumerate(
            np.asarray(asset.parents, dtype=np.int64).tolist()
        ):
            if int(parent_j) >= 0:
                contact_delta[joint] = target_joint_delta[int(parent_j)]
        contact_pose = (
            np.einsum("bij,bj->bi", contact_delta[:, :3, :3], contact_rest)
            + contact_delta[:, :3, 3]
        )
        drivers = source_bone_driver_frames(asset, pose_axis_angle)
        coupling = np.asarray(asset.source_driver_coupling, dtype=np.float64)
        for side, raw_entry in leg_solve_raw.items():
            entry = _validate_leg_hinge_solve_entry_v1(
                raw_entry,
                side=str(side),
                bone_count=bone_count,
                joint_count=int(pose_points.shape[0]),
            )
            fb = int(entry["femur_bone"])
            kb = int(entry["knee_bone"])
            ab = int(entry["ankle_bone"])
            hj = int(entry["smplx_hip"])
            kj = int(entry["smplx_knee"])
            aj = int(entry["smplx_ankle"])
            hip = contact_pose[hj]
            knee_pt = hip + (pose_points[kj] - pose_points[hj])
            ankle_pt = knee_pt + (pose_points[aj] - pose_points[kj])
            driver_desired = drivers[fb] @ coupling[fb]
            _R, theta_applied, theta_raw, _axis = solve_leg_hinge_v1(
                hip=hip,
                knee=knee_pt,
                ankle=ankle_pt,
                bind_hip=bg[fb, :3, 3],
                bind_knee=bg[kb, :3, 3],
                bind_ankle=bg[ab, :3, 3],
                bind_femur_rotation=bg[fb, :3, :3],
                hinge_axis_femur_local=entry["hinge_axis_femur_local"],
                driver_femur_rotation=driver_desired[:3, :3],
                blend_lo_deg=float(entry["blend_lo_deg"]),
                blend_hi_deg=float(entry["blend_hi_deg"]),
            )
            clamp_by_side[str(side)] = float(
                np.degrees(float(theta_raw) - float(theta_applied))
            )

    for side in SIDES:
        suffix = _side_suffix(side)
        femur = _bone_index(names, f"Femur_Rot_{suffix}")
        knee = _bone_index(names, f"Knee_Rotate_{suffix}")
        tibia = _bone_index(names, f"Tibia_Bone_{suffix}")
        patella = _bone_index(names, f"Patella_Rotate_{suffix}")
        if asset.source_bone_smplx_a is None or asset.source_bone_smplx_b is None:
            raise ValueError("observe_fk_v7 requires source_bone_smplx_a/b")
        hip_joint = int(asset.source_bone_smplx_a[femur])
        knee_joint = int(asset.source_bone_smplx_b[femur])
        if (
            hip_joint < 0
            or knee_joint < 0
            or hip_joint >= len(pose_points)
            or knee_joint >= len(pose_points)
            or hip_joint == knee_joint
        ):
            raise ValueError(f"{side} femur SMPL-X hip/knee mapping is invalid")

        hip = _observe_hip(
            side=side,
            femur_index=femur,
            knee_index=knee,
            hip_joint=hip_joint,
            knee_joint=knee_joint,
            pg=pg,
            bg=bg,
            rest_vertices=rest_vertices,
            posed_vertices=posed,
            domains=domains,
            pose_points=pose_points,
            target_joint_delta=target_joint_delta,
        )
        controller[f"hip_{side}"] = hip
        if not hip["available"]:
            notes.append(f"hip_{side}: {hip['reason']}")

        clamp_deg = float(clamp_by_side.get(side, 0.0))
        knee_obs = _observe_knee(
            side=side,
            femur_index=femur,
            knee_index=knee,
            pg=pg,
            bg=bg,
            posed_vertices=posed,
            domains=domains,
            reference=reference,
            flexion_clamped_deg=clamp_deg if abs(clamp_deg) > 1.0e-9 else 0.0,
        )
        controller[f"knee_{side}"] = knee_obs
        if knee_obs.get("flexion_deg") is not None:
            knee_flexion[side] = float(np.radians(float(knee_obs["flexion_deg"])))
        elif reference.knee_axis_local is not None and side in reference.knee_axis_local:
            # Still attempt flexion from matrices when rotation was otherwise ok.
            posed_local = _local_pair(pg[femur], pg[knee])
            bind_local = _local_pair(bg[femur], bg[knee])
            axis = _unit_axis(reference.knee_axis_local[side], f"knee_axis_local[{side}]")
            if axis is not None:
                on_axis, _off, _parent = _on_axis_decomposition(
                    posed_local, bind_local, axis
                )
                knee_flexion[side] = float(on_axis)
        if not knee_obs["available"]:
            notes.append(f"knee_{side}: {knee_obs['reason']}")
        if abs(clamp_deg) > 1.0e-6:
            notes.append(
                f"knee_{side}: hyperextension clamped by {clamp_deg:.3f} deg"
            )

        indices = {
            "Femur_Rot": femur,
            "Knee_Rotate": knee,
            "Tibia_Bone": tibia,
            "Patella_Rotate": patella,
        }
        for parent_base, child_base, kind in _LEG_LINK_SPECS:
            key = _link_key(side, parent_base, child_base)
            obs = _observe_local_link(
                side=side,
                parent_index=indices[parent_base],
                child_index=indices[child_base],
                pg=pg,
                bg=bg,
                link_kind=kind,
                reference=reference,
                knee_flexion_rad=knee_flexion[side],
            )
            local_fk[key] = obs
            if not obs["available"]:
                notes.append(f"{key}: {obs['reason']}")

        for parent_base, child_base, kind in _ARM_LINK_SPECS:
            parent_name = f"{parent_base}_{suffix}"
            child_name = f"{child_base}_{suffix}"
            key = f"{side}/{parent_name}>{child_name}"
            parent_i = _bone_index_optional(names, parent_name)
            child_i = _bone_index_optional(names, child_name)
            if parent_i is None or child_i is None:
                missing = parent_name if parent_i is None else child_name
                obs = _unavailable(f"arm bone {missing!r} is absent")
            else:
                obs = _observe_local_link(
                    side=side,
                    parent_index=parent_i,
                    child_index=child_i,
                    pg=pg,
                    bg=bg,
                    link_kind=kind,
                    reference=reference,
                    knee_flexion_rad=None,
                )
            local_fk[key] = obs
            if not obs["available"]:
                notes.append(f"{key}: {obs['reason']}")

    # Confirm required leg keys exist exactly.
    for key in REQUIRED_LOCAL_FK_LINKS:
        if key not in local_fk:
            raise ValueError(f"required local-FK link {key!r} was not measured")

    return {
        "schema_version": int(FK_OBSERVATION_SCHEMA_VERSION),
        "controller_observations": controller,
        "local_fk_observations": local_fk,
        "diagnostics": {
            "reference_source": str(reference.source),
            "pose_axis_angle_shape": list(pose_array.shape),
            "knee_flexion_rad": {
                side: knee_flexion[side] for side in SIDES
            },
            "notes": notes,
        },
    }


def observations_report_v7(
    asset: Any,
    *,
    pose_axis_angle: Any,
    posed_vertices: Any,
    domains: Any,
    reference: FkReferenceV7,
) -> dict[str, Any]:
    """Measure observations and evaluate controller / local-FK gates."""
    observations = observe_fk_v7(
        asset,
        pose_axis_angle=pose_axis_angle,
        posed_vertices=posed_vertices,
        domains=domains,
        reference=reference,
    )
    controller = evaluate_controller_gate_v7(
        observations["controller_observations"]
    )
    local_fk = evaluate_local_fk_gate_v7(
        observations["local_fk_observations"],
        required=REQUIRED_LOCAL_FK_LINKS,
    )
    arm_keys = [
        key
        for key in observations["local_fk_observations"]
        if key not in REQUIRED_LOCAL_FK_LINKS
    ]
    local_fk_arms = evaluate_local_fk_gate_v7(
        observations["local_fk_observations"],
        required=arm_keys,
    )
    return {
        "observations": observations,
        "controller": controller,
        "local_fk": local_fk,
        "local_fk_arms": local_fk_arms,
    }


__all__ = [
    "FK_OBSERVATION_SCHEMA_VERSION",
    "FkReferenceV7",
    "default_fk_reference_v7",
    "fk_reference_from_patella_oracle_v7",
    "observe_fk_v7",
    "observations_report_v7",
]
