"""Rotation-minimizing tool frames for a measured phantom surface.

The scanner supplies an outward surface normal at every Cartesian waypoint;
the probe's tool ``+Z`` points into the contact surface, so it must align to
the *negative* of that normal.
There are infinitely many tool frames with a given tool ``+Z`` direction.  A
world-fixed yaw hint at every point can therefore introduce an artificial
spin (and, at an S-turn, a nearly 180 degree flip).  This module transports
the measured initial frame along the normal sequence instead: each update is
the shortest rotation which maps the previous tool ``+Z`` to the inward target
``-normal``.
The update axis is tangent to the old and new ``+Z`` directions, so its body
frame rotation vector has no ``Z`` component.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


_UNIT_TOL = 1.0e-5
_ROTATION_TOL = 1.0e-6
_AXIS_TOL = 1.0e-12


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_normals(normals: Any) -> np.ndarray:
    array = _as_float_array(normals, name="normals")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("normals must have shape (N, 3)")
    if array.shape[0] < 1:
        raise ValueError("normals must contain at least one vector")
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths <= _AXIS_TOL):
        raise ValueError("normals must be non-zero")
    if np.max(np.abs(lengths - 1.0)) > _UNIT_TOL:
        raise ValueError("normals must be unit vectors")
    # Surface fitting normally already returns unit normals.  Renormalizing
    # the small floating point residual keeps every subsequent cross/dot
    # operation well-conditioned without accepting a scaled normal.
    return array / lengths[:, None]


def _validate_rotation(rotation: Any, *, name: str) -> np.ndarray:
    array = _as_float_array(rotation, name=name)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    gram_error = float(np.max(np.abs(array.T @ array - np.eye(3))))
    determinant = float(np.linalg.det(array))
    if gram_error > _ROTATION_TOL or abs(determinant - 1.0) > _ROTATION_TOL:
        raise ValueError(f"{name} must be a proper rotation matrix")
    return array


def _skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=np.float64)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a proper Rodrigues rotation for a unit axis and angle."""

    unit_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(unit_axis))
    if not np.isfinite(length) or length <= _AXIS_TOL:
        raise ValueError("rotation axis must be non-zero")
    unit_axis = unit_axis / length
    cross = _skew(unit_axis)
    sine = float(np.sin(angle))
    cosine = float(np.cos(angle))
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def _shortest_z_alignment(previous: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest rotation mapping the old tool Z onto ``target``.

    ``previous`` is the old world-frame tool rotation matrix and ``target`` is the
    next inward tool ``+Z`` target (the negative of the outward normal).  The
    antiparallel case has no unique shortest axis;
    the old tool ``+X`` is supplied by the caller so that the result is
    deterministic while still avoiding a rotation around tool ``+Z``.
    """

    old_z = np.array(previous[:, 2], dtype=np.float64, copy=True)
    old_x = np.asarray(previous[:, 0], dtype=np.float64)
    old_z /= np.linalg.norm(old_z)
    target_z = np.array(target, dtype=np.float64, copy=True)
    target_z /= np.linalg.norm(target_z)

    cross = np.cross(old_z, target_z)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(old_z, target_z), -1.0, 1.0))
    if sine <= _AXIS_TOL:
        if cosine >= 0.0:
            return np.eye(3, dtype=np.float64)
        # The old X axis is orthogonal to old Z for a valid frame.  Project
        # once to suppress only roundoff; using old Z here would leave the
        # antiparallel mapping undefined and create an arbitrary tool spin.
        axis = old_x - float(np.dot(old_x, old_z)) * old_z
        return _axis_angle(axis, np.pi)

    axis = cross / sine
    angle = float(np.arctan2(sine, cosine))
    return _axis_angle(axis, angle)


def rotation_minimizing_frames(
    normals: Any,
    initial_rotation: Any,
) -> np.ndarray:
    """Transport ``initial_rotation`` over an outward-normal sequence.

    Parameters
    ----------
    normals:
        Array of shape ``(N, 3)`` containing finite, approximately unit
        outward normals.  The tool ``+Z`` target is ``-normals[i]``; the
        supplied outward sign is therefore retained explicitly.
    initial_rotation:
        A finite proper ``(3, 3)`` measured tool rotation matrix.

    Returns
    -------
    numpy.ndarray
        ``(N, 3, 3)`` proper rotation matrices.  Frame ``0`` is obtained by
        the shortest tilt from the measured initial tool ``+Z`` to
        ``-normals[0]``.  Every following frame uses only the shortest
        ``+Z`` alignment from the preceding frame to ``-normals[i]``.
    """

    outward_normals = _validate_normals(normals)
    # The fitted surface normal points out of the phantom.  The probe's tool
    # +Z is defined into the phantom, hence every alignment target is the
    # inward direction.  Keeping this sign conversion here makes it
    # impossible for callers to accidentally mix outward surface normals
    # with a tool frame that points away from the contact surface.
    tool_z_targets = -outward_normals
    previous = _validate_rotation(initial_rotation, name="initial_rotation")
    frames = np.empty((tool_z_targets.shape[0], 3, 3), dtype=np.float64)

    for index, target in enumerate(tool_z_targets):
        alignment = _shortest_z_alignment(previous, target)
        current = alignment @ previous
        # Rodrigues construction is already orthogonal to machine precision.
        # A final SVD projection prevents long paths from accumulating a
        # measurable matrix error while preserving the proper (det +1) side.
        u, _, vh = np.linalg.svd(current)
        current = u @ vh
        if float(np.linalg.det(current)) < 0.0:
            u[:, -1] *= -1.0
            current = u @ vh
        frames[index] = current
        previous = current
    return frames


def _relative_rotvec(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative_body = np.asarray(previous, dtype=np.float64).T @ np.asarray(current, dtype=np.float64)
    return Rotation.from_matrix(relative_body).as_rotvec()


def frame_diagnostics(rotations: Any, initial_rotation: Any) -> dict[str, Any]:
    """Return JSON-compatible checks for initial and adjacent frame motion.

    The first entry in ``step_rotation_deg`` and ``tool_z_spin_deg`` is the
    motion from ``initial_rotation`` to frame 0.  Remaining entries are the
    adjacent path steps.  ``tool_z_spin_deg`` is the absolute body-frame Z
    component of each relative rotation vector; it directly audits the
    rotation-minimizing constraint.
    """

    frames = _as_float_array(rotations, name="rotations")
    if frames.ndim != 3 or frames.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if frames.shape[0] < 1:
        raise ValueError("rotations must contain at least one frame")
    for index, frame in enumerate(frames):
        _validate_rotation(frame, name=f"rotations[{index}]")
    initial = _validate_rotation(initial_rotation, name="initial_rotation")

    previous = np.concatenate((initial[None, :, :], frames[:-1]), axis=0)
    relative = np.einsum("nij,njk->nik", previous.transpose(0, 2, 1), frames)
    rotvecs = Rotation.from_matrix(relative).as_rotvec()
    step_rad = np.linalg.norm(rotvecs, axis=1)
    spin_rad = np.abs(rotvecs[:, 2])
    path_step_rad = step_rad[1:]
    path_spin_rad = spin_rad[1:]
    frame_gram_error = np.max(
        np.abs(np.einsum("nji,njk->nik", frames, frames) - np.eye(3)),
        axis=(1, 2),
    )
    frame_det_error = np.abs(np.linalg.det(frames) - 1.0)

    def _degrees(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.degrees(values)]

    return {
        "n_frames": int(frames.shape[0]),
        "initial_step_rotation_deg": float(np.degrees(step_rad[0])),
        "initial_tool_z_spin_deg": float(np.degrees(spin_rad[0])),
        "max_step_rotation_deg": float(np.degrees(np.max(step_rad))),
        "max_tool_z_spin_deg": float(np.degrees(np.max(spin_rad))),
        "max_path_step_rotation_deg": float(
            np.degrees(np.max(path_step_rad)) if path_step_rad.size else 0.0
        ),
        "max_path_tool_z_spin_deg": float(
            np.degrees(np.max(path_spin_rad)) if path_spin_rad.size else 0.0
        ),
        "step_rotation_deg": _degrees(step_rad),
        "tool_z_spin_deg": _degrees(spin_rad),
        "max_body_rotvec_z_rad": float(np.max(np.abs(rotvecs[:, 2]))),
        "max_frame_orthogonality_error": float(np.max(frame_gram_error)),
        "max_frame_det_error": float(np.max(frame_det_error)),
    }


__all__ = ["frame_diagnostics", "rotation_minimizing_frames"]
