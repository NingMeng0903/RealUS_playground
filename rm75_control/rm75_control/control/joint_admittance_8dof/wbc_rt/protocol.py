"""Shared-memory protocol for the wbc_rt C++ inner process.

Layouts must match ``native/wbc_rt/include/wbc_rt/protocol.hpp``.
"""

from __future__ import annotations

import numpy as np

WBC_MAGIC = 0x57424331
WBC_VERSION = 7
DEFAULT_IN_NAME = "rm75_wbc_in"
DEFAULT_OUT_NAME = "rm75_wbc_out"

CMD_NONE = 0
CMD_STEP = 1
CMD_ENABLE = 2
CMD_STOP = 3
CMD_RESET = 4
CMD_BEGIN_HYBRID = 5
CMD_SET_RAIL_MODE = 6
CMD_SET_FLAGS = 7
CMD_PLAN_STROKE = 8
CMD_SET_STROKE = 9
CMD_SET_RAIL_POSE_TARGET = 10
CMD_CAPTURE_RAIL_EXT_REF = 11
CMD_SET_RAIL_EXT_MODE = 12
CMD_SHUTDOWN = 13

IN_CONTACT = 1 << 0
IN_STALE = 1 << 1
IN_HAS_QDOT_FF = 1 << 2
IN_HAS_POSE_D = 1 << 3
IN_HAS_VEL_FF = 1 << 4
IN_HAS_RAIL_V = 1 << 5
IN_HAS_V_FORCE = 1 << 6
IN_HAS_PATH_TWIST = 1 << 7
IN_HAS_FEEDBACK_TWIST = 1 << 8
IN_SEED_QCMD = 1 << 9
IN_HAS_POSTURE = 1 << 10
IN_HAS_QSTAR = 1 << 11
IN_COMMIT_PREV = 1 << 12
IN_ABORT_PREV = 1 << 13
IN_AUTO_COMMIT = 1 << 14

FLAG_PLAN_DRIVES_RAIL = 1 << 0
FLAG_DIRECT_PTP = 1 << 1
FLAG_ARM_SUPPRESS = 1 << 2
FLAG_CENTER_SUPPRESS = 1 << 3
FLAG_MANIP_ACTIVE = 1 << 4
FLAG_RAIL_EXT_ACTIVE = 1 << 5

OUT_JOINT_LIMITED = 1 << 0
OUT_RAIL_LIMITED = 1 << 1
OUT_WALL_ACTIVE = 1 << 2
OUT_SEC_SUPPRESSED = 1 << 3
OUT_STALE = 1 << 4
OUT_READY = 1 << 5
OUT_FAILED = 1 << 6

STATUS_BOOT = 0
STATUS_READY = 1
STATUS_OK = 2
STATUS_FAIL = 3
STATUS_SHUTDOWN = 4

QP_NOT_RUN = 0
QP_SOLVED = 1
QP_MAX_ITER = 2
QP_FAILED = 3
QP_PRIMAL_INFEASIBLE = 4
QP_DUAL_INFEASIBLE = 5
QP_CLOSEST_PRIMAL = 6
QP_P0_CONFLICT = 7

RAIL_COUPLED = 0
RAIL_LOCKED = 1
STYLE_HOLD = 0
STYLE_RAIL_ONLY = 1
STYLE_TCP_FIXED = 2

WBC_IN_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u4"),
        ("seq", "<u8"),
        ("cmd_seq", "<u8"),
        ("cmd", "<u4"),
        ("flags", "<u4"),
        ("t_mono", "<f8"),
        ("dt_wall", "<f8"),
        ("dt_nom", "<f8"),
        ("v_cmd", "<f8", (6,)),
        ("q_meas", "<f8", (8,)),
        ("rail_q", "<f8"),
        ("rail_v", "<f8"),
        ("v_force_z", "<f8"),
        ("pose_d", "<f8", (6,)),
        ("vel_ff", "<f8", (6,)),
        ("qdot_ff", "<f8", (8,)),
        ("path_twist", "<f8", (6,)),
        ("feedback_twist", "<f8", (6,)),
        ("cmd_f", "<f8", (16,)),
        ("cmd_u", "<u4", (8,)),
        ("rail_refresh_dt", "<f8"),
    ],
    align=False,
)

WBC_OUT_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u4"),
        ("seq", "<u8"),
        ("cmd_ack", "<u8"),
        ("status", "<u4"),
        ("flags", "<u4"),
        ("q_cmd", "<f8", (8,)),
        ("qdot", "<f8", (8,)),
        ("v_cmd_received", "<f8", (6,)),
        ("v_cmd_feasible", "<f8", (6,)),
        ("v_tcp_estimated", "<f8", (6,)),
        ("task_residual", "<f8", (6,)),
        ("slack", "<f8"),
        ("e_qp", "<f8"),
        ("u_alloc", "<f8"),
        ("u_mid", "<f8"),
        ("v_r_ref", "<f8"),
        ("psi", "<f8"),
        ("d_star", "<f8"),
        ("d_pref", "<f8"),
        ("solve_ms", "<f8"),
        ("sigma_min", "<f8"),
        ("sigma_arm", "<f8"),
        ("cmd_f", "<f8", (8,)),
        ("joint_limited", "<u4"),
        ("rail_limited", "<u4"),
        ("wall_active", "<u4"),
        ("secondary_suppressed", "<u4"),
        ("ns_norm", "<f8"),
        ("ns_centering", "<f8"),
        ("ns_manip", "<f8"),
        ("ns_arm_angle", "<f8"),
        ("ns_damping", "<f8"),
        ("ns_rail_lock", "<f8"),
        ("sat_scale", "<f8"),
        ("sec_target_norm", "<f8"),
        ("homotopy_s", "<f8"),
        ("psi_star", "<f8"),
        ("rail_motion_share", "<f8"),
        ("u_task_raw", "<f8"),
        ("u_task_feasible", "<f8"),
        ("u_pi_raw", "<f8"),
        ("u_mid_cmd", "<f8"),
        ("u_post_raw", "<f8"),
        ("u_post_feasible", "<f8"),
        ("u_mid_applied", "<f8"),
        ("d_star_dot_cmd", "<f8"),
        ("u_escape_raw", "<f8"),
        ("u_escape_feasible", "<f8"),
        ("escape_active", "<f8"),
        ("escape_dir", "<f8"),
        ("u_base", "<f8"),
        ("u_feasible", "<f8"),
        ("v_r_lpf", "<f8"),
        ("e_d", "<f8"),
        ("V_d_proxy", "<f8"),
        ("j4_design_slack", "<f8"),
        ("sigma_slack", "<f8"),
        ("rail_box_lo", "<f8"),
        ("rail_box_hi", "<f8"),
        ("rail_bind_lo", "<u4"),
        ("rail_bind_hi", "<u4"),
        ("rail_task_vel_used", "<f8"),
        ("rail_h1", "<f8"),
        ("rail_h2", "<f8"),
        ("rail_qdot_prev", "<f8"),
        ("rail_qdot_prev2", "<f8"),
        ("qp1_status", "<u4"),
        ("qp2_status", "<u4"),
        ("qp1_iter", "<u4"),
        ("qp2_iter", "<u4"),
        ("n_cbf_active", "<u4"),
        ("box_degenerate", "<u4"),
        ("box_infeasible", "<u4"),
        ("manip_active", "<u4"),
        ("qp1_solve_ms", "<f8"),
        ("qp2_solve_ms", "<f8"),
        ("assembly_ms", "<f8"),
        ("fallback_ms", "<f8"),
        ("hard_residual_max", "<f8"),
        ("equality_residual_max", "<f8"),
        ("rail_exec", "<f8"),
        ("box_excess_max", "<f8"),
        ("follow_err_rad", "<f8"),
        ("qdot_qp_vs_sent_max", "<f8"),
        ("dual_cancel", "<f8"),
        ("secondary_alpha", "<f8"),
        ("box_lo", "<f8", (8,)),
        ("box_hi", "<f8", (8,)),
        ("qdot_prev", "<f8", (8,)),
        ("qdot_prev2", "<f8", (8,)),
        ("task_progress_alpha", "<f8"),
        ("task_progress_scale_used", "<f8"),
        ("rail_task_projection", "<f8"),
        ("rail_task_committed", "<f8"),
        ("rail_escape_committed", "<f8"),
        ("rail_post_committed", "<f8"),
        ("rail_total_committed", "<f8"),
        ("rail_preview_arm_norm", "<f8"),
        ("rail_preview_residual", "<f8"),
        ("task_paused", "<u4"),
        ("task_pause_reason", "<u4"),
        ("v_task_actual", "<f8", (6,)),
        ("rail_preview_arm", "<f8", (8,)),
        ("rail_base_shaped", "<f8"),
        ("rail_base_raw", "<f8"),
        ("rail_pi_xi", "<f8"),
        ("rail_d_ref", "<f8"),
        ("rail_ref_acceleration", "<f8"),

    ],
    align=False,
)

WBC_IN_SIZE = int(WBC_IN_DTYPE.itemsize)
WBC_OUT_SIZE = int(WBC_OUT_DTYPE.itemsize)
# Packed C++ layouts in native/wbc_rt/include/wbc_rt/protocol.hpp.
assert WBC_IN_SIZE == 616, WBC_IN_SIZE
assert WBC_OUT_SIZE == 1440, WBC_OUT_SIZE


def view_in(buf) -> np.ndarray:
    return np.ndarray((1,), dtype=WBC_IN_DTYPE, buffer=buf)


def view_out(buf) -> np.ndarray:
    return np.ndarray((1,), dtype=WBC_OUT_DTYPE, buffer=buf)
