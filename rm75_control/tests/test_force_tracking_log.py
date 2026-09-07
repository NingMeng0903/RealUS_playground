"""CSV schema checks for the single stable force controller."""

from __future__ import annotations

import csv
import os
import queue
import time

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


def test_tick_logger_drops_when_queue_is_full(tmp_path):
    path = tmp_path / "full.csv"
    logger = _TickLogger(str(path))
    logger._stop.set()
    logger._worker.join(timeout=2.0)
    while True:
        try:
            logger._q.put_nowait(None)
        except queue.Full:
            break
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
    logger.write(0.0, "scan", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6), outer=outer)
    assert logger.dropped >= 1
    logger.close()


def test_tick_logger_queues_raw_request_without_callable_or_deepcopy(tmp_path):
    path = tmp_path / "raw.csv"
    logger = _TickLogger(str(path))
    # Stop the child before enqueueing so the request can be inspected without
    # racing its formatter.  Keep admission enabled to exercise the producer.
    logger._worker.terminate()
    logger._worker.join(timeout=2.0)
    logger._worker_alive = lambda: True
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
    logger.write(0.0, "scan", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6))
    step.fallback_reason = "mutated_after_enqueue"
    step.q_send[:] = 7.0
    request = logger._q.get(timeout=2.0)
    assert isinstance(request, tuple)
    assert not callable(request)
    assert request[0] == logger._REQUEST_TAG
    assert request[1][3].fallback_reason == ""
    np.testing.assert_array_equal(request[1][3].q_send, np.zeros(8))
    logger.close()


def test_tick_logger_live_blocked_sink_cannot_block_producer(tmp_path):
    # A FIFO without a reader blocks the live child in open(), exactly where
    # a stalled filesystem must be isolated from the controller.
    path = tmp_path / "blocked.csv"
    os.mkfifo(path)
    logger = _TickLogger(str(path))
    step = JointIkStep(
        q_send=np.zeros(8), qdot=np.zeros(8), twist_base=np.zeros(6),
        sigma_min=.2, manip=.1, slack_norm=0., n_cbf_active=0,
        follow_err_rad=0., fallback_reason="native_timeout_coast",
    )
    try:
        started = time.monotonic()
        for _ in range(logger._QUEUE_MAX + 20):
            logger.write(0., "scan", 0., step, np.zeros(8), np.zeros(6), np.zeros(6))
        elapsed = time.monotonic() - started
        assert logger._worker.is_alive()
        assert logger._q.full()
        assert logger.dropped == 20
        assert elapsed < .5  # 100 requests must not wait on the blocked sink.
    finally:
        logger.close()
    assert not logger._worker.is_alive()


def test_tick_logger_writer_failure_only_drops_telemetry(tmp_path):
    path = tmp_path / "directory.csv"
    path.mkdir()
    logger = _TickLogger(str(path))
    logger._worker.join(timeout=2.0)
    assert logger._failed.is_set()
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
    logger.write(0.0, "scan", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6))
    assert logger.dropped >= 1
    logger.close()


def test_tick_logger_writes_qpik_stage_timing_columns(tmp_path):
    path = tmp_path / "timing.csv"
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
    for name, value in (
        ("qp_kinematics_ms", 1.25),
        ("qp_collision_ms", 2.5),
        ("qp_solve_phase_ms", 3.75),
        ("native_dispatch_ms", 4.0),
        ("native_roundtrip_ms", 5.0),
        ("native_transport_ms", 6.0),
    ):
        setattr(step, name, value)
    logger.write(0.0, "scan", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6))
    logger.close()
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["qpik_kinematics_ms"] == "1.250000"
    assert row["qpik_collision_ms"] == "2.500000"
    assert row["qpik_solve_phase_ms"] == "3.750000"
    assert row["qpik_native_dispatch_ms"] == "4.000000"
    assert row["qpik_native_roundtrip_ms"] == "5.000000"
    assert row["qpik_native_transport_ms"] == "6.000000"


def test_tick_logger_snapshots_tilt_telemetry_and_records_stop_flags(tmp_path):
    path = tmp_path / "tilt.csv"
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

    class Outer:
        pass

    outer = Outer()
    outer.last_tau_y = 0.125
    outer.last_tau_error_y = -0.375
    outer.last_omega_y = 0.5
    outer.last_theta_tilt = -0.0625
    outer.last_tilt_engaged = False
    outer.last_tilt_frozen = False
    outer.last_tilt_capped = False
    outer.last_tilt_stalled = True
    outer.last_tilt_stop_reason = "cop_stall"
    outer.last_cop_x = 0.0065
    outer.last_cop_r = 0.0085
    outer.last_on_tube = False
    outer.last_tilt_deadband_nm = 0.025

    logger.write(
        0.0,
        "scan",
        0.0,
        step,
        np.zeros(8),
        np.zeros(6),
        np.zeros(6),
        outer=outer,
    )
    # The request must contain a value snapshot, rather than the mutable
    # HybridTffOuter instance that produced it.
    outer.last_tau_y = 9.0
    outer.last_tilt_stalled = False
    outer.last_tilt_stop_reason = "mutated_after_enqueue"
    outer.last_cop_x = 9.0
    logger.close()

    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["tilt_tau_y_nm"] == "0.125000"
    assert row["tilt_tau_error_y_nm"] == "-0.375000"
    assert row["tilt_omega_y_rad_s"] == "0.500000"
    assert row["tilt_theta_rad"] == "-0.062500"
    assert row["tilt_engaged"] == "0"
    assert row["tilt_frozen"] == "0"
    assert row["tilt_capped"] == "0"
    assert row["tilt_stalled"] == "1"
    assert row["tilt_stop_reason"] == "cop_stall"
    assert row["tilt_cop_x_m"] == "0.006500000"
    assert row["tilt_cop_r_m"] == "0.008500000"
    assert row["tilt_on_tube"] == "0"
    assert row["tilt_deadband_nm"] == "0.025000"
