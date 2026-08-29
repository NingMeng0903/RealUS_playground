"""CSV schema checks for the single stable force controller."""

from __future__ import annotations

import csv

import numpy as np

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkStep,
    _TickLogger,
)


def test_force_log_has_energy_aware_reference_and_actual_tcp_velocity(tmp_path):
    path = tmp_path / "force.csv"
    logger = _TickLogger(str(path))
    step = JointIkStep(
        q_send=np.zeros(8),
        qdot=np.zeros(8),
        twist_base=np.zeros(6),
        sigma_min=0.2,
        manip=0.1,
        slack_norm=0.0,
        n_cbf_active=0,
        follow_err_rad=0.0,
    )
    controller = AdmittanceController(0.005, AdmittanceConfig())

    class Outer:
        pass

    outer = Outer()
    outer.controller = controller
    controller.force_reference_fast_clear = True
    controller.force_fast_z = 1.234
    controller.retract_guard_armed = True
    controller.retract_fast_hold = True
    controller.retract_fast_stop_count = 2
    controller.retract_fast_rearm_count = 3
    controller.force_task_latched = True
    controller.physical_contact_state = "suspect_loss"
    controller.physical_contact_acquire_event = True
    controller.physical_contact_loss_event = False
    controller.physical_contact_reacquire_event = True
    controller.physical_contact_low_timer_s = 0.012
    controller.physical_contact_high_timer_s = 0.034
    logger.write(
        0.0,
        "scan",
        0.0,
        step,
        np.zeros(8),
        np.zeros(6),
        np.zeros(6),
        outer=outer,
        dt_actual_s=0.005,
        sensor_age_s=0.001,
        f_ext_raw=np.zeros(6),
        twist_achieved_base=np.zeros(6),
        v_tcp_z_actual=0.002,
    )
    logger.close()

    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert len(rows[0]) == len(rows[1])
    header = rows[0]
    assert "force_reference_scale_n" in header
    assert "force_reference_drive" in header
    assert "force_reference_gate_scale" in header
    assert "force_reference_accel_m_s2" in header
    assert "force_reference_reversal_reset" in header
    assert "force_reference_fast_clear" in header
    assert "force_fast_z" in header
    assert "retract_guard_armed" in header
    assert "retract_fast_hold" in header
    assert "retract_fast_stop_count" in header
    assert "retract_fast_rearm_count" in header
    assert "force_task_latched" in header
    assert "physical_contact_state" in header
    assert "physical_contact_acquire_event" in header
    assert "physical_contact_loss_event" in header
    assert "physical_contact_reacquire_event" in header
    assert "physical_contact_low_timer_s" in header
    assert "physical_contact_high_timer_s" in header
    assert "mass_z_eff" in header
    assert "damping_ke_z" in header
    assert "damping_dimeas_z" in header
    assert "vz_achieved_tool" in header
    assert "pose_d_x" in header
    assert "pose_meas_x" in header
    assert "motion_err_lin_y_mm" in header
    assert "motion_err_rms_mm" in header
    assert "motion_axis_peak_mm" in header
    assert "vel_ff_vy" in header
    assert "rail_contrib_m_s" in header
    assert "arm_contrib_m_s" in header
    assert "arm_y_qdot" in header
    assert "rail_motion_share" in header
    assert "rail_escape_active" in header
    assert "tool_y_des_m" in header
    assert "psi_deg" in header
    assert "psi_ref_deg" in header
    assert "d_pref_m" in header
    assert "waste_ratio" in header
    assert "rail_ff_m" in header
    assert "d_star_m" in header
    assert "psi_star_deg" in header
    assert "homotopy_s" in header
    assert "contact_phase" in header
    assert "ke_hat" in header
    assert "dob_v" in header
    assert "barrier_cap_floor" in header
    assert "elbow_margin_rad" in header
    assert "wrist_open_rad" in header
    assert "family_ok" in header
    assert "tool_y_err_mm" in header
    assert "rail_sat" in header
    assert "sigma_arm" in header
    assert "qdot_meas_0" in header
    assert "v_cmd_vy" in header
    assert "path_twist_vy" in header
    assert "feedback_twist_vy" in header
    assert "sns_scale" in header
    assert "v_escape" in header
    assert "cbf_min_dist" in header
    assert "comfort_slack_j4" in header
    assert "pad_lx" in header
    assert "pad_vcmd_base_vy" in header
    assert "u_dob_z" in header
    assert "v_force_cmd_z" in header
    assert "tdpa_e_obs_j" in header
    assert "tdpa_alpha" in header
    assert "tdpa_clamped" in header
    assert "tdpa_passivity_holds" in header
    assert "corridor_applied" in header
    assert "ke_cap_n_m" in header
    assert "cdyob_corr_m_s" in header
    assert "cdyob_qtinv_vm" in header
    assert "cdyob_q_vi" in header
    assert "cdyob_n1_force" in header
    assert "cdyob_pert_unclipped" in header
    assert "cdyob_blend" in header
    assert "cdyob_vi" in header
    assert "cdyob_candidate" in header
    assert "cdyob_antiwindup_error" in header
    assert "cdyob_residual" in header
    assert "cdyob_saturated" in header
    assert "cdyob_constrained" in header
    assert "cdyob_linear_equivalent" in header
    assert "cdyob_apply_ready" in header
    assert "cdyob_ready_s" in header
    assert "overforce_escape" in header
    assert "u_nom_raw" in header
    assert "u_nom_capped" in header
    assert "u_shield_hyp" in header
    assert "u_sent" in header
    assert "lambda_obs" in header
    assert "shield_applied" in header
    assert "shield_feasible" in header
    assert "f_ub_n" in header
    assert "e_lb_j" in header
    assert "w_lb_j" in header
    assert "rho_v2_w" in header
    assert "n_stop" in header
    assert "tube_violation" in header
    assert "solver_us" in header
    assert "shield_infeasible_reason" in header
    assert "f_constraint_margin_n" in header
    assert "energy_margin_j" in header
    assert "terminal_ok" in header
    assert "aj_ok" in header
    assert "domain_ok" in header
    assert "uncertified_brake" in header
    assert "recovery_latched" in header
    assert "recontact_slow_latched" in header
    assert "v_recontact_cap" in header

    values = dict(zip(header, rows[1], strict=True))
    assert values["rail_escape_active"] == "0"
    assert values["force_reference_fast_clear"] == "1"
    assert values["force_fast_z"] == "1.234"
    assert values["retract_guard_armed"] == "1"
    assert values["retract_fast_hold"] == "1"
    assert values["retract_fast_stop_count"] == "2"
    assert values["retract_fast_rearm_count"] == "3"
    assert values["force_task_latched"] == "1"
    assert values["physical_contact_state"] == "suspect_loss"
    assert values["physical_contact_acquire_event"] == "1"
    assert values["physical_contact_loss_event"] == "0"
    assert values["physical_contact_reacquire_event"] == "1"
    assert values["physical_contact_low_timer_s"] == "0.012000"
    assert values["physical_contact_high_timer_s"] == "0.034000"


def test_motion_axis_accuracy_columns_populated(tmp_path):
    path = tmp_path / "motion.csv"
    logger = _TickLogger(str(path))
    step = JointIkStep(
        q_send=np.zeros(8),
        qdot=np.zeros(8),
        twist_base=np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]),
        sigma_min=0.2,
        manip=0.1,
        slack_norm=0.0,
        n_cbf_active=0,
        follow_err_rad=0.0,
        rail_contrib_m_s=0.01,
        arm_contrib_m_s=0.01,
        rail_motion_share=0.5,
        rail_escape_active=False,
    )
    cfg = AdmittanceConfig(
        track_axes=np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
        force_axes=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    controller = AdmittanceController(0.005, cfg)

    class Outer:
        pass

    outer = Outer()
    outer.controller = controller
    outer.last_pose_d = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    outer.last_vel_ff = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    pose_meas = np.array([0.0, 0.04, 0.01, 0.0, 0.0, 0.0])
    logger.write(
        0.0,
        "scan",
        0.0,
        step,
        np.zeros(8),
        pose_meas,
        np.zeros(6),
        outer=outer,
    )
    logger.close()
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    values = dict(zip(rows[0], rows[1], strict=True))
    assert values["pose_d_y"] == "0.050000"
    assert values["pose_meas_y"] == "0.040000"
    assert values["motion_err_lin_y_mm"] != ""
    assert values["motion_err_lin_z_mm"] == ""  # force axis excluded
    assert values["motion_err_rms_mm"] != ""
    assert values["vel_ff_vy"] == "0.030000"
    assert values["rail_contrib_m_s"] == "0.010000"
    assert values["rail_escape_active"] == "0"
    assert values["tool_y_err_mm"] != ""


def test_tick_logger_appends_on_restart(tmp_path):
    path = tmp_path / "run.csv"
    step = JointIkStep(
        q_send=np.zeros(8),
        qdot=np.zeros(8),
        twist_base=np.zeros(6),
        sigma_min=0.2,
        manip=0.1,
        slack_norm=0.0,
        n_cbf_active=0,
        follow_err_rad=0.0,
    )
    controller = AdmittanceController(0.005, AdmittanceConfig())

    class Outer:
        pass

    outer = Outer()
    outer.controller = controller
    first = _TickLogger(str(path))
    first.write(0.0, "scan", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6), outer=outer)
    first.close()
    second = _TickLogger(str(path))
    second.write(0.01, "scan", 0.01, step, np.zeros(8), np.zeros(6), np.zeros(6), outer=outer)
    second.close()
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][0] == "t_wall_s"
    assert len(rows) == 3
    assert rows[1][0] != "t_wall_s"
    assert rows[2][0] != "t_wall_s"
    assert len(rows[0]) == len(rows[1]) == len(rows[2])
