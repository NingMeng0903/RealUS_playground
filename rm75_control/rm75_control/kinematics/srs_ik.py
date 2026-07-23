"""Shimizu 2008 closed-form IK for the RM75-6F 7-DOF SRS arm on the 8-DOF rail.

The RM75 URDF simplifies to the canonical SRS Euler chain

    R_tcp = R_z(q1)·R_y(q2)·R_z(q3) · R_y(q4) · R_z(q5)·R_y(q6)·R_z(q7)

with joint origins that make joints 1,2,3 intersect at the shoulder centre S
(offset D_BS = 0.2405 m above the base_link along +Z), joint 4 (elbow) sitting
D_SE = 0.256 m from S along the upper arm, joints 5,6,7 intersecting at the
wrist centre W (D_EW = 0.210 m from E), and the TCP on the joint_7 axis
D_WT = 0.1612 + 0.220 = 0.3812 m from W.  ψ is the Shimizu swivel angle: the
signed rotation of E about the SW axis measured from the base −Z reference
vector projected off SW — identical to
``tasks.arm_angle.ArmAngleTask.arm_angle`` (matched to 1e-6 rad in tests).

The rail (q[0]) only translates S in +Y (it is a rigid base shift) and ψ is
invariant to it (see ``tests/test_arm_angle_rail_invariance.py``).

Branch encoding (3 bits → 8 branches):

    branch_id = (shoulder_bit << 2) | (elbow_bit << 1) | wrist_bit

    shoulder_bit = 0  →  q2 >= 0  (elbow "above" shoulder)
                = 1  →  q2 <  0
    elbow_bit    = 0  →  q4 >= 0  (elbow bent one way)
                = 1  →  q4 <  0
    wrist_bit    = 0  →  q6 >= 0  (wrist not flipped)
                = 1  →  q6 <  0

References
----------
* M. Shimizu et al., "Analytical inverse kinematic computation for 7-DOF
  redundant manipulators with joint limits and its application to redundancy
  resolution control", IEEE T-RO 24(5), 2008.
* J. K. Kreutz-Delgado et al., "Kinematic analysis of 7-DOF manipulators",
  Int. J. Robotics Research 11(5), 1992 (SRS decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# URDF-derived geometric constants (see rm75_control/assets/robots/
# rm75_6f_8dof/RM75-6F-8dof.urdf).  DO NOT change without also updating the
# URDF: the whole point of this module is a *closed-form* IK that agrees with
# the Pinocchio FK to numerical precision.
# ---------------------------------------------------------------------------
D_BS: float = 0.2405   # base_link (below rail slide) → S (joint_2 origin) along +Z
D_SE: float = 0.256    # |S-E| upper arm length
D_EW: float = 0.210    # |E-W| forearm length
D_WT_FLANGE: float = 0.1612  # |W→link_7| along flange / tool Z (URDF joint_7)
D_WT: float = 0.3812   # default |W-TCP| = D_WT_FLANGE + 0.220 (stock gripper)


def d_wt_from_tcp_offset(tcp_offset_pose: np.ndarray) -> float:
    """|W→TCP| from live link_7→tcp offset (m): flange + |Δz| (tool along Z)."""
    off = np.asarray(tcp_offset_pose, dtype=float).reshape(-1)
    if off.size < 3:
        return float(D_WT)
    z = float(off[2])
    if abs(z) < 1e-9:
        z = float(np.linalg.norm(off[:3]))
    return float(D_WT_FLANGE + abs(z))


def d_wt_from_kin(kin) -> float:
    """|W→TCP| from ``RobotKinematics`` after TCP sync (link_7→tcp frame)."""
    try:
        off = np.asarray(kin.tcp_offset_pose, dtype=float).reshape(6)
        return d_wt_from_tcp_offset(off)
    except Exception:
        r = np.asarray(getattr(kin, "_r_link7_tcp", [0.0, 0.0, 0.22]), dtype=float).reshape(3)
        z = float(r[2]) if abs(float(r[2])) > 1e-9 else float(np.linalg.norm(r))
        return float(D_WT_FLANGE + abs(z))

# Reference direction for ψ = 0.  Matches _V_REF in tasks/arm_angle.py.
_V_REF: np.ndarray = np.array([0.0, 0.0, -1.0])

# URDF joint limits (rad) for the seven arm joints.  Rail limits are handled
# separately by the SafetyLimits / QP boxes; this module only filters arm q.
Q_LOWER: np.ndarray = np.array(
    [-3.106, -2.2689, -3.106, -2.356, -3.106, -2.234, -6.28], dtype=float
)
Q_UPPER: np.ndarray = np.array(
    [ 3.106,  2.2689,  3.106,  2.356,  3.106,  2.234,  6.28], dtype=float
)

_EPS_SIN: float = 1e-6   # sin(q_2) / sin(q_6) singularity threshold


# ---------------------------------------------------------------------------
# Basic rotations & pose conversions
# ---------------------------------------------------------------------------
def _rot_z(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def _rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])


def _euler_xyz_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """scipy ``from_euler('xyz', [rx, ry, rz])`` → rotation matrix.

    scipy's lowercase ``'xyz'`` is EXTRINSIC — the resulting matrix maps a
    body-frame vector to the base frame as ``R = R_z(rz) · R_y(ry) · R_x(rx)``.
    (Also equivalent to the intrinsic ZYX convention.)  Matches
    ``model.py::RobotKinematics.fk_pose`` exactly.
    """
    return _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)


def _pose_to_Rp(pose: np.ndarray, euler_order: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
    """Convert a 6-vec pose ``[x, y, z, rx, ry, rz]`` → (R, p)."""
    pose = np.asarray(pose, dtype=float).reshape(-1)
    if pose.size != 6:
        raise ValueError(f"pose must be length 6, got {pose.size}")
    if euler_order != "xyz":
        # We only intrinsically support xyz here; anything else must be
        # converted by the caller (matches the rest of the 8-DOF stack, which
        # is xyz throughout — see model.py fk_pose).
        raise ValueError(f"srs_ik only supports euler_order='xyz', got {euler_order!r}")
    p = pose[:3].copy()
    R = _euler_xyz_to_R(pose[3], pose[4], pose[5])
    return R, p


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# ψ (swivel angle) — geometric helpers.  Kept identical to the ones in
# tasks/arm_angle.py so servo-layer ψ and planner-layer ψ never disagree.
# ---------------------------------------------------------------------------
def _psi_from_SEW(S: np.ndarray, E: np.ndarray, W: np.ndarray) -> float:
    """Signed rotation of E about the SW axis, measured from V_REF perp SW.

    Returns 0 when the arm is fully stretched (|e_perp| ≈ 0) or when SW is
    parallel to V_REF (|r_perp| ≈ 0).  Matches ``ArmAngleTask.arm_angle``.
    """
    sw = W - S
    n_sw = float(np.linalg.norm(sw))
    if n_sw < 1e-9:
        return 0.0
    w_hat = sw / n_sw
    e_perp = (E - S) - np.dot(E - S, w_hat) * w_hat
    r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
    ne = float(np.linalg.norm(e_perp))
    nr = float(np.linalg.norm(r_perp))
    if ne < 1e-9 or nr < 1e-9:
        return 0.0
    e_u = e_perp / ne
    r_u = r_perp / nr
    return float(np.arctan2(np.dot(np.cross(r_u, e_u), w_hat), np.dot(r_u, e_u)))


def _E_from_psi(S: np.ndarray, W: np.ndarray, psi: float) -> np.ndarray | None:
    """Elbow position on the ψ-parametrised circle of centre E_center, radius r_e.

    Returns ``None`` if SW is outside the reachable range ``[|D_SE-D_EW|, D_SE+D_EW]``
    or if SW is parallel to V_REF (ψ reference undefined).
    """
    sw = W - S
    dsw = float(np.linalg.norm(sw))
    if dsw < abs(D_SE - D_EW) + 1e-9 or dsw > D_SE + D_EW - 1e-9:
        return None
    w_hat = sw / dsw
    # Law of cosines at S: |SE|² + |SW|² − |EW|² = 2·|SE|·|SW|·cos(θ)
    cos_theta = float(
        np.clip((D_SE * D_SE + dsw * dsw - D_EW * D_EW) / (2.0 * D_SE * dsw), -1.0, 1.0)
    )
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    c = D_SE * cos_theta               # centre offset from S along SW
    r_e = D_SE * sin_theta             # circle radius
    E_center = S + c * w_hat
    r_perp = _V_REF - float(np.dot(_V_REF, w_hat)) * w_hat
    nr = float(np.linalg.norm(r_perp))
    if nr < 1e-9:
        return None                    # SW parallel to V_REF → ψ undefined
    r_u = r_perp / nr
    r_bin = np.cross(w_hat, r_u)       # already unit, r_u ⊥ w_hat
    E = E_center + r_e * (np.cos(psi) * r_u + np.sin(psi) * r_bin)
    return E


# ---------------------------------------------------------------------------
# Forward kinematics of the shoulder/elbow chain, from arm joints only.  Used
# by ``psi_from_q`` and by the srs_ik round-trip test.  Rail is not needed
# because ψ is invariant to it; call sites that need world S must add
# ``(0, y_rail, 0)`` themselves.
# ---------------------------------------------------------------------------
def _fk_SEW_arm(q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S, E, W) in the base_link frame (rail excluded)."""
    q1, q2, q3, q4 = q_arm[0], q_arm[1], q_arm[2], q_arm[3]
    S = np.array([0.0, 0.0, D_BS], dtype=float)
    # SE direction in base = R_shoulder · Z_local = column 3 of R_z(q1)R_y(q2)R_z(q3)
    SE_dir = np.array(
        [
            float(np.cos(q1) * np.sin(q2)),
            float(np.sin(q1) * np.sin(q2)),
            float(np.cos(q2)),
        ]
    )
    E = S + D_SE * SE_dir
    R_sh = _rot_z(q1) @ _rot_y(q2) @ _rot_z(q3)
    EW_local = np.array([float(np.sin(q4)), 0.0, float(np.cos(q4))])
    W = E + D_EW * (R_sh @ EW_local)
    return S, E, W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def psi_from_q(q_arm: np.ndarray) -> float:
    """Swivel angle ψ ∈ (−π, π] from the seven arm joints (rad).

    Rail-invariant: caller may pass either a 7-vec (arm only) or an 8-vec
    (rail + arm); q[0] is ignored either way.  Numerically matches
    ``tasks/arm_angle.py:ArmAngleTask.arm_angle`` (verified to 1e-6 rad).
    """
    q = np.asarray(q_arm, dtype=float).reshape(-1)
    if q.size == 8:
        q = q[1:]
    if q.size != 7:
        raise ValueError(f"q_arm must be length 7 or 8, got {q.size}")
    S, E, W = _fk_SEW_arm(q)
    return _psi_from_SEW(S, E, W)


def branch_from_q(q_arm: np.ndarray) -> int:
    """Discrete branch id ∈ {0..7} that ``srs_ik`` needs to reproduce ``q_arm``.

    Sign convention matches the SRS decomposition below: q_2 sign is the
    shoulder branch (upper arm above vs below), q_4 sign is the elbow branch
    (bent one way vs the other), q_6 sign is the wrist branch (wrist flip).
    """
    q = np.asarray(q_arm, dtype=float).reshape(-1)
    if q.size == 8:
        q = q[1:]
    if q.size != 7:
        raise ValueError(f"q_arm must be length 7 or 8, got {q.size}")
    b_sh = 0 if q[1] >= 0.0 else 1
    b_el = 0 if q[3] >= 0.0 else 1
    b_wr = 0 if q[5] >= 0.0 else 1
    return (b_sh << 2) | (b_el << 1) | b_wr


def is_reachable(
    pose_tcp: np.ndarray,
    y_rail: float = 0.0,
    euler_order: str = "xyz",
    *,
    d_wt: float | None = None,
) -> bool:
    """True iff the TCP pose lies inside the SRS reachable annulus.

    Ignores joint limits (those are branch-dependent and are enforced inside
    ``srs_ik``); this is the fast geometric feasibility check for a planner.
    """
    L = float(D_WT if d_wt is None else d_wt)
    R, p = _pose_to_Rp(pose_tcp, euler_order)
    S = np.array([0.0, float(y_rail), D_BS])
    W = p - L * R[:, 2]
    dsw = float(np.linalg.norm(W - S))
    return abs(D_SE - D_EW) + 1e-6 < dsw < D_SE + D_EW - 1e-6


@dataclass(frozen=True)
class SrsSolution:
    """Full IK output when the caller wants intermediate quantities too."""
    q_arm: np.ndarray
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray
    psi_realised: float
    branch: int


def srs_ik(
    pose_tcp: np.ndarray,
    psi: float,
    branch_id: int,
    y_rail: float = 0.0,
    *,
    euler_order: str = "xyz",
    check_limits: bool = True,
    d_wt: float | None = None,
) -> np.ndarray | None:
    """Closed-form IK for (pose_tcp, ψ, branch, y_rail) → q_arm (7-vec, rad).

    Returns ``None`` if any of the following fail:

    * the pose is outside the reachable annulus (|SW| ∉ [|D_SE-D_EW|, D_SE+D_EW])
    * the shoulder configuration hits an algorithmic singularity
      (sin(q_2) ≈ 0 — arm vertical, ψ tangent to the SW axis)
    * the wrist configuration hits a gimbal-lock singularity (sin(q_6) ≈ 0)
    * ``check_limits`` is True and any q_i falls outside the URDF limits

    ``branch_id`` encodes 3 discrete choices; see the module docstring.

    ``d_wt`` is |W→TCP| along tool Z.  Pass :func:`d_wt_from_kin` after TCP
    sync so a changed gripper offset is honoured (default ``D_WT`` = stock 220 mm).

    Preconditions
    -------------
    ``euler_order`` must be ``'xyz'`` (matches ``RobotKinematics.fk_pose``).
    """
    L = float(D_WT if d_wt is None else d_wt)
    R_tcp, p_tcp = _pose_to_Rp(pose_tcp, euler_order)
    branch_id = int(branch_id) & 0b111
    b_sh = (branch_id >> 2) & 1
    b_el = (branch_id >> 1) & 1
    b_wr = branch_id & 1

    # --- 1. Wrist centre and shoulder centre --------------------------------
    S = np.array([0.0, float(y_rail), D_BS], dtype=float)
    # TCP frame Z axis points from W to TCP (verified against the URDF joint
    # chain: link_7 Z at q_7 = 0 equals joint_7 axis; TCP is on that axis).
    W = p_tcp - L * R_tcp[:, 2]

    # --- 2. Elbow angle from law of cosines --------------------------------
    dsw = float(np.linalg.norm(W - S))
    if dsw <= abs(D_SE - D_EW) + 1e-6 or dsw >= D_SE + D_EW - 1e-6:
        return None
    cos_q4 = (dsw * dsw - D_SE * D_SE - D_EW * D_EW) / (2.0 * D_SE * D_EW)
    cos_q4 = float(np.clip(cos_q4, -1.0, 1.0))
    q4_mag = float(np.arccos(cos_q4))            # ∈ [0, π]
    q4 = q4_mag if b_el == 0 else -q4_mag

    # --- 3. Elbow position from ψ ------------------------------------------
    E = _E_from_psi(S, W, float(psi))
    if E is None:
        return None

    # --- 4. Shoulder joints from SE direction and (S, E, W) plane ----------
    SE_dir = (E - S) / D_SE
    # ZYZ Euler on the column-3 vector: q_2 = acos(z),  q_1 = atan2(y, x)
    z = float(np.clip(SE_dir[2], -1.0, 1.0))
    q2_mag = float(np.arccos(z))
    if q2_mag < _EPS_SIN or q2_mag > np.pi - _EPS_SIN:
        # sin(q_2) ≈ 0: arm vertical, q_1 is degenerate (ZYZ gimbal) and ψ
        # loses observability.  Reject to keep the ψ-controlled semantics.
        return None
    if b_sh == 0:
        q2 = q2_mag
        q1 = float(np.arctan2(SE_dir[1], SE_dir[0]))
    else:
        q2 = -q2_mag
        q1 = _wrap_pi(float(np.arctan2(SE_dir[1], SE_dir[0])) + np.pi)

    # q_3 aligns the ZYZ shoulder frame so the wrist centre sits in its
    # (X, Z) plane.  In the shoulder frame the wrist-centre vector has
    # coordinates V_local = (|EW|·sin(q4), 0, |SE|+|EW|·cos(q4)); conjugating
    # only q_1 and q_2 away gives U = R_z(q3)·V_local, so
    #     U_x = |EW|·sin(q4)·cos(q3)
    #     U_y = |EW|·sin(q4)·sin(q3)
    # The sign of ``sin(q4)`` flips both — atan2 must use the sign, otherwise
    # q_3 comes back with a ±π offset whenever the elbow bends "the other
    # way" (b_el = 1).
    U = _rot_y(-q2) @ _rot_z(-q1) @ (W - S)
    if float(U[0] * U[0] + U[1] * U[1]) < 1e-16:
        # Wrist centre on the shoulder-frame Z axis: another degenerate case
        # (arm-stretched-through-shoulder), which the reachability check
        # above already forbids in practice; guard anyway.
        return None
    sign_sin_q4 = 1.0 if np.sin(q4) >= 0.0 else -1.0
    q3 = float(np.arctan2(sign_sin_q4 * U[1], sign_sin_q4 * U[0]))

    # --- 5. Wrist joints from residual orientation (ZYZ) -------------------
    R_pre_wrist = _rot_z(q1) @ _rot_y(q2) @ _rot_z(q3) @ _rot_y(q4)
    R_wrist = R_pre_wrist.T @ R_tcp

    # ZYZ Euler extraction on R_wrist.  The two branches differ by q_6 sign.
    cos_q6 = float(np.clip(R_wrist[2, 2], -1.0, 1.0))
    q6_mag = float(np.arccos(cos_q6))
    if q6_mag < _EPS_SIN or q6_mag > np.pi - _EPS_SIN:
        # Wrist gimbal lock (q_6 ≈ 0 or ±π): q_5 and q_7 are coupled and the
        # branch choice becomes underdetermined.  Reject; the planner must
        # pick a different ψ.
        return None
    if b_wr == 0:
        q6 = q6_mag
        sin_q6 = float(np.sin(q6))
        q5 = float(np.arctan2(R_wrist[1, 2], R_wrist[0, 2]))
        q7 = float(np.arctan2(R_wrist[2, 1], -R_wrist[2, 0]))
    else:
        q6 = -q6_mag
        sin_q6 = float(np.sin(q6))
        q5 = float(np.arctan2(-R_wrist[1, 2], -R_wrist[0, 2]))
        q7 = float(np.arctan2(-R_wrist[2, 1], R_wrist[2, 0]))
    del sin_q6                                    # (kept for future debug)

    q_arm = np.array([q1, q2, q3, q4, q5, q6, q7], dtype=float)

    # --- 6. Joint-limit filter --------------------------------------------
    if check_limits:
        if bool(np.any(q_arm < Q_LOWER - 1e-9)) or bool(np.any(q_arm > Q_UPPER + 1e-9)):
            return None

    return q_arm


def srs_ik_with_diagnostics(
    pose_tcp: np.ndarray,
    psi: float,
    branch_id: int,
    y_rail: float = 0.0,
    *,
    euler_order: str = "xyz",
    check_limits: bool = True,
    d_wt: float | None = None,
) -> SrsSolution | None:
    """Same as ``srs_ik`` but also returns S/E/W/ψ_realised for diagnostics."""
    q_arm = srs_ik(
        pose_tcp,
        psi,
        branch_id,
        y_rail=y_rail,
        euler_order=euler_order,
        check_limits=check_limits,
        d_wt=d_wt,
    )
    if q_arm is None:
        return None
    S, E, W = _fk_SEW_arm(q_arm)
    S = S + np.array([0.0, float(y_rail), 0.0])
    E = E + np.array([0.0, float(y_rail), 0.0])
    W = W + np.array([0.0, float(y_rail), 0.0])
    psi_realised = _psi_from_SEW(S, E, W)
    return SrsSolution(
        q_arm=q_arm,
        S=S,
        E=E,
        W=W,
        psi_realised=psi_realised,
        branch=branch_from_q(q_arm),
    )


__all__ = [
    "D_BS",
    "D_SE",
    "D_EW",
    "D_WT",
    "D_WT_FLANGE",
    "d_wt_from_kin",
    "d_wt_from_tcp_offset",
    "Q_LOWER",
    "Q_UPPER",
    "SrsSolution",
    "branch_from_q",
    "is_reachable",
    "psi_from_q",
    "srs_ik",
    "srs_ik_with_diagnostics",
]
