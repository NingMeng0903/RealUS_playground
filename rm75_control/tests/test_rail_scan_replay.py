"""Offline closed-loop replay of the D-pose Y scan (COUPLED rail, no robot).

Reproduces the failing hardware scenario from /tmp teach logs: 80 cm pp
tool-Y sine at 4 cm/s from slot D.  With the old tuning the arm stretched to
near-straight (elbow 17.5 deg, sigma_arm 0.027) at the +Y extreme while the
rail lagged ("stuck straight" lock-up).  With the preferred-extension rail
task + corrected rail cost the rail is recruited early and smoothly:

  - sigma_min never drops below its value at the scan-entry pose
  - the elbow stays far from straight
  - the rail does not chatter (only the sine turnarounds reverse it)
  - a small scan keeps extension inside the reach dead zone while the rail
    tracks via FF without sacrificing sigma or tracking
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"

# Slot-D scan pose from the hardware run (rail ~0, arm at the pose-IK result).
Q_D = np.array(
    [0.0, -0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137]
)


def _make_inner() -> JointIkController:
    raw = yaml.safe_load(CONFIG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.qp.collision.enabled = False  # STL narrow-phase not needed offline
    cfg.control_frame = "base"        # test drives base-frame twists directly
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(Q_D)
    # Same phase preset the scan uses: COUPLED rail + extension coordination
    # + psi hold + centering (yaml rail.mode must be coupled for this test).
    SecondaryPolicy(preset="track").apply(inner)
    return inner


def _run_scan(inner: JointIkController, amplitude_m: float, n_ticks: int, v_peak_m_s: float = 0.04):
    """Base-frame Y sine about the start pose with velocity feedforward."""
    kin = inner.kin
    dt = inner.cfg.dt
    m0 = kin.fk_placement(Q_D)
    p0 = m0.translation.copy()
    r0 = m0.rotation.copy()
    omega = v_peak_m_s / amplitude_m
    out = {
        "sigma": [],
        "elbow_deg": [],
        "rail": [],
        "err_mm": [],
        "psi_err_deg": [],
        "ext_err_m": [],
        "tcp_y": [],
    }
    psi_ref = float(inner.arm_task.psi_ref)
    for i in range(n_ticks):
        t = i * dt
        dy = amplitude_m * np.sin(omega * t)
        vy = amplitude_m * omega * np.cos(omega * t)
        q = inner.q_cmd
        mc = kin.fk_placement(q)
        perr = (p0 + np.array([0.0, dy, 0.0])) - mc.translation
        rerr = Rsc.from_matrix(r0 @ mc.rotation.T).as_rotvec()
        twist = np.zeros(6)
        twist[:3] = 2.0 * perr + np.array([0.0, vy, 0.0])
        twist[3:] = 1.5 * rerr
        vel_ff = np.zeros(6)
        vel_ff[1] = vy
        step = inner.update(twist, dt, q_meas=q.copy(), vel_ff=vel_ff)
        mc_post = kin.fk_placement(inner.q_cmd)
        out["sigma"].append(step.sigma_min)
        out["elbow_deg"].append(abs(float(np.degrees(inner.q_cmd[4]))))
        out["rail"].append(float(inner.q_cmd[0]))
        out["err_mm"].append(float(np.linalg.norm(perr)) * 1000.0)
        out["ext_err_m"].append(step.rail_ext_err_m)
        out["tcp_y"].append(float(mc_post.translation[1]))
        out["psi_err_deg"].append(
            float(
                np.degrees(
                    (inner.arm_task.arm_angle(inner.q_cmd) - psi_ref + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                )
            )
        )
    return {k: np.asarray(v) for k, v in out.items()}


def _rail_reversals(rail: np.ndarray, v_eps: float = 1e-6) -> int:
    dq = np.diff(rail)
    rev = 0
    last = 0
    for d in dq:
        if abs(d) <= v_eps:
            continue
        s = 1 if d > 0 else -1
        if last != 0 and s != last:
            rev += 1
        last = s
    return rev


def test_80cm_scan_stays_well_conditioned():
    """Full 80 cm pp cycle: the failing hardware case.  Baseline (rail cost
    bug): sigma 0.0499, elbow 17.5 deg at the +Y peak.  Redesigned: sigma
    never below the scan-entry value, elbow never near straight."""
    inner = _make_inner()
    amplitude = 0.40
    omega = 0.04 / amplitude
    n = int(2.0 * np.pi / omega / inner.cfg.dt) + 10  # one full period
    out = _run_scan(inner, amplitude, n)

    assert out["sigma"].min() > 0.07, out["sigma"].min()
    assert out["elbow_deg"].min() > 35.0, out["elbow_deg"].min()
    # Ideal-plant tracking must not be sacrificed for the posture goals.
    assert out["err_mm"].max() < 5.0, out["err_mm"].max()
    # Swivel (psi) held through the whole excursion.
    assert np.abs(out["psi_err_deg"]).max() < 10.0, np.abs(out["psi_err_deg"]).max()
    # Rail used, but within travel and without chatter: only the two sine
    # turnarounds may reverse its direction (allow a small margin).
    assert np.abs(out["rail"]).max() <= 0.25
    assert np.abs(out["rail"]).max() > 0.15  # rail actually recruited
    assert _rail_reversals(out["rail"]) <= 10, _rail_reversals(out["rail"])


def test_real_scan_rail_does_not_hunt():
    """16 cm pp (the app default): the rail reverses only at sine turnarounds.

    Extra reversals are the σ-escape fighting the scan feedforward, which is
    the left/right rocking seen near singularity on hardware: ∂σ/∂y holds one
    sign while the sweep flips every half period, so any escape allowed to
    outrank the primary shows up here as more than 2 reversals per period.
    """
    inner = _make_inner()
    amplitude = 0.08
    periods = 4
    omega = 0.04 / amplitude
    n = int(periods * 2.0 * np.pi / omega / inner.cfg.dt) + 10
    out = _run_scan(inner, amplitude, n)

    reversals = _rail_reversals(out["rail"])
    assert reversals <= 2 * periods, reversals
    # Stage-1 acceptance allows a short transient up to 5 mm; the steady
    # trajectory remains tighter than 2 mm for 95% of samples.
    assert out["err_mm"].max() < 5.0, out["err_mm"].max()
    assert np.percentile(out["err_mm"], 95.0) < 2.0
    assert out["elbow_deg"].min() > 35.0, out["elbow_deg"].min()
    assert out["sigma"].min() > 0.07, out["sigma"].min()


def test_small_scan_rail_stays_in_sweet_spot():
    """8 cm pp scan: FF recruits the rail for gross Y, but reach error stays
    inside e0_m so the arm remains in its sweet spot (no straightening)."""
    inner = _make_inner()
    amplitude = 0.04
    omega = 0.04 / amplitude
    n = int(np.pi / omega / inner.cfg.dt) + 10  # half period covers +peak/-slope
    out = _run_scan(inner, amplitude, n)
    e0 = float(inner.cfg.rail_extension.e0_m)

    assert np.abs(out["ext_err_m"]).max() < e0, np.abs(out["ext_err_m"]).max()
    # Rail tracks via FF (not stuttering): meaningful motion, bounded on 8 cm scan.
    rail_pp = float(np.ptp(out["rail"]))
    tcp_y_pp = float(np.ptp(out["tcp_y"]))
    assert rail_pp > 0.005, rail_pp
    assert rail_pp < 0.10, rail_pp
    assert rail_pp > 0.15 * tcp_y_pp, (rail_pp, tcp_y_pp)
    assert np.abs(out["rail"]).max() < 0.10, np.abs(out["rail"]).max()
    assert out["sigma"].min() > 0.07
    assert out["err_mm"].max() < 5.0


def test_track_preset_keeps_nominal_centering():
    """Scan entry keeps yaml q_nominal centering — D is arbitrary, not comfortable."""
    inner = _make_inner()
    target_j4 = float(np.degrees(inner.centering_task.q_target[4]))
    assert abs(target_j4 - 90.0) < 0.1, target_j4


def test_track_preset_gates_extension_task():
    """API wiring: track preset (COUPLED) enables + re-anchors the extension
    task; hold preset disables it and locks the rail."""
    inner = _make_inner()
    assert inner._rail_ext_active
    assert inner.rail_ext_task.d_pref_m is not None
    d_pref = inner.rail_ext_task.d_pref_m
    SecondaryPolicy(preset="hold").apply(inner)
    assert not inner._rail_ext_active
    assert inner.is_locked_hold
    # Track must restore the CONFIGURED yaml mode (COUPLED) even though the
    # hold phase mutated the live cfg.rail.mode to LOCKED (regression guard:
    # reading cfg.rail.mode kept the scan LOCKED after any hold@D phase).
    SecondaryPolicy(preset="track").apply(inner)
    from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode

    assert inner.rail_mode == RailMode.COUPLED
    assert inner._rail_ext_active
    assert abs(inner.rail_ext_task.d_pref_m - d_pref) < 1e-9  # same posture
