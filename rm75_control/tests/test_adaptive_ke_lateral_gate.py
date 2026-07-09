"""
Regression guard for adaptive_ke's lateral-velocity learning gate.

On /tmp/scan_v4.csv the median tangential (tool-XY) scan speed was
0.017 m/s, but adaptive_ke.lateral_vel_gate_m_s was 0.005 m/s -- well below
that median. K_e learning (`_should_update_ke`) was therefore gated shut for
93.2% of scan ticks, so the "online stiffness adaptation" feature never
actually engaged during real scanning; it only opened in the near-zero-speed
turnaround at each sine-scan extremum, which is not where a genuine
muscle<->bone stiffness transition would be sampled.

These tests pin the widened gate (~0.02 m/s) that keeps learning open across
a realistic fraction of scan speeds while still freezing K_e during the
fastest, most geometry-coupled mid-stroke motion.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rm75_control.control.hybrid_motion.adaptive_ke import AdaptiveKeConfig, EnvironmentStiffnessEstimator


def test_yaml_lateral_vel_gate_widened():
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    gate = raw["hybrid_motion"]["adaptive_ke"]["lateral_vel_gate_m_s"]
    assert 0.015 <= gate <= 0.03, (
        f"lateral_vel_gate_m_s={gate} -- must stay near the data-driven ~0.02 m/s "
        "band (measured median scan speed 0.017 m/s on /tmp/scan_v4.csv); the old "
        "0.005 m/s froze K_e learning for 93.2% of scan ticks (see MD/debug.md "
        "Round 4), and values above ~0.03 stop gating out the fastest, most "
        "geometry-coupled mid-stroke motion."
    )


def test_median_scan_speed_keeps_learning_open():
    """The measured median tangential scan speed (0.017 m/s) must pass the gate."""
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    cfg = AdaptiveKeConfig.from_dict(raw["hybrid_motion"], raw["hybrid_motion"])
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005)

    median_scan_speed_m_s = 0.017
    assert est._should_update_ke(
        f_ext_z=3.0,
        f_err_z=0.0,
        v_lateral_m_s=median_scan_speed_m_s,
        df=0.0,
        f_err_gate_n=est._f_err_gate_eff_n(3.0),
    ), "widened gate must keep K_e learning open at the measured median scan speed"


def test_fast_midstroke_speed_still_gated():
    """Fast mid-stroke tangential motion (strong geometric F_z coupling) must
    still freeze K_e learning -- the widened gate must not disable protection
    entirely."""
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    cfg = AdaptiveKeConfig.from_dict(raw["hybrid_motion"], raw["hybrid_motion"])
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005)

    fast_midstroke_speed_m_s = 0.06
    assert not est._should_update_ke(
        f_ext_z=3.0,
        f_err_z=0.0,
        v_lateral_m_s=fast_midstroke_speed_m_s,
        df=0.0,
        f_err_gate_n=est._f_err_gate_eff_n(3.0),
    ), "fast mid-stroke scan speed must still gate K_e learning shut"
