"""Power-paired geometry at the original TCP and the acoustic face."""
from __future__ import annotations

import numpy as np

from .types import ProbeGeometry, TwistConstraints, vector


def skew(p):
    x, y, z = np.asarray(p, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def twist_tcp_to_face(geometry: ProbeGeometry):
    t = geometry.T_tcp_face
    r = t[:3, :3].T
    return np.block([[r, -r @ skew(t[:3, 3])], [np.zeros((3, 3)), r]])


def wrench_tcp_to_face(wrench_tcp, geometry: ProbeGeometry):
    # W_face^T V_face = W_tcp^T V_tcp.
    return np.linalg.solve(twist_tcp_to_face(geometry).T, vector(wrench_tcp, (6,)))


def motion_basis(path_twist):
    b = np.array(vector(path_twist, (6,)), copy=True)
    if abs(b[2]) > 1e-12 or abs(b[4]) > 1e-12:
        raise ValueError("path must relinquish TCP z translation and y rotation")
    h = np.zeros((6, 3))
    h[2, 0], h[4, 1] = 1., 1.
    h[:, 2] = b
    return h


def point_normal_row(geometry: ProbeGeometry, face_x_m: float):
    t = geometry.T_tcp_face
    point = t[:3, 3] + float(face_x_m) * t[:3, 0]
    normal = t[:3, 2]
    return normal @ np.concatenate((np.eye(3), -skew(point)), axis=1)


def window_rows(geometry: ProbeGeometry, lateral_windows=((.04, .34), (.34, .66), (.66, .96))):
    windows = vector(lateral_windows, (3, 2), name="lateral_windows")
    if np.any(windows[:, 0] >= windows[:, 1]) or np.any(windows < 0) or np.any(windows > 1):
        raise ValueError("invalid lateral window fractions")
    locations = geometry.image_x_sign * geometry.half_length_m * (windows.sum(axis=1)-1.)
    return np.stack([point_normal_row(geometry, x) for x in locations])


def aperture_rows(geometry: ProbeGeometry, center_interval_m=None):
    interval = (-geometry.half_length_m, geometry.half_length_m) if center_interval_m is None else center_interval_m
    if len(interval) != 2 or not (-geometry.half_length_m <= interval[0] <= interval[1] <= geometry.half_length_m):
        raise ValueError("center interval must lie within physical face aperture")
    return np.stack([point_normal_row(geometry, c) for c in interval])


def constraints_to_base(constraints: TwistConstraints, rotation_base_tcp):
    if constraints.frame != "tcp_tool":
        raise ValueError("constraint is already transformed or has the wrong frame")
    r = vector(rotation_base_tcp, (3, 3), name="rotation_base_tcp")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-9) or np.linalg.det(r) < 0:
        raise ValueError("invalid TCP rotation")
    tool_from_base = np.block([[r.T, np.zeros((3, 3))], [np.zeros((3, 3)), r.T]])
    return TwistConstraints(constraints.A @ tool_from_base, constraints.lower, constraints.upper,
                            "tcp_base", constraints.valid_until_s, constraints.sequence,
                            constraints.stop_epoch, constraints.labels)


def accepted_alpha(h, proposed_alpha, sent_tool, predicted_tool, tolerance):
    """Reference progress only, not measured acquisition progress."""
    h = vector(h, (6, 3), name="H")
    tol = vector(tolerance, (6,), name="command tolerance")
    if (tol <= 0).any():
        raise ValueError("positive component command tolerances required")
    if not 0 <= proposed_alpha <= 1:
        raise ValueError("invalid proposed alpha")
    zero_path = np.linalg.norm(h[:, 2]) <= 1e-14
    values = []
    scaled = h / tol[:, None]
    for v in (sent_tool, predicted_tool):
        velocity = vector(v, (6,))
        y, *_ = np.linalg.lstsq(scaled, velocity/tol, rcond=None)
        if np.any(np.abs(h @ y-velocity) > tol):
            raise ValueError("final velocity left the admitted motion subspace")
        alpha_tol = 1.0 / max(np.linalg.norm(h[:, 2]/tol), 1.0)
        if y[2] < -alpha_tol or y[2] > proposed_alpha + alpha_tol:
            raise ValueError("final velocity reverses or exceeds admitted progress")
        values.append(max(0., float(y[2])))
    return 0.0 if zero_path else min(float(proposed_alpha), *values)
