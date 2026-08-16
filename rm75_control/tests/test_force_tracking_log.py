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
    assert "contact_phase" in header
    assert "ke_hat" in header
    assert "dob_v" in header
    assert "barrier_cap_floor" in header
    assert "elbow_margin_rad" in header
    assert "wrist_open_rad" in header
    assert "family_ok" in header
    assert "tool_y_err_mm" in header
    assert "rail_sat" in header
    assert "rail_feedback_fresh" in header
    assert "rail_command_mode" in header
    assert header.index("rail_feedback_fresh") < header.index("rail_command_mode")
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
    assert values["rail_feedback_fresh"] == "0"
    assert values["rail_command_mode"] == ""


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
