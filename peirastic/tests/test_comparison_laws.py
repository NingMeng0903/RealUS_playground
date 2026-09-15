"""Hardware-free 1D AC vs TAFAC-z comparison laws."""
import numpy as np
import pytest

from peirastic.api.payloads import HfpcPayload
from peirastic.realman8dof.force.comparison_laws import (
    ComparisonAdmittance1D,
    ComparisonAdmittance2D,
    ComparisonContactLatch,
    TafacNormalLaw,
    comparison_law_from_payload,
)


def _step(law, *, fz=3.0, fd=4.0, v_in=0.0, dt=0.005):
    path = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    raw = np.array([0.01, 0.0, v_in, 0.0, 0.0, 0.02])
    return law.update(
        dt_s=dt,
        pose=np.zeros(6),
        f_ext=np.array([0.0, 0.0, fz, 0.0, 0.05, 0.0]),
        f_des=np.array([0.0, 0.0, fd, 0.0, 0.0, 0.0]),
        path_twist=path,
        path_twist_raw=raw,
        contact=True,
    )


def test_admittance_1d_writes_only_tool_z():
    law = ComparisonAdmittance1D(mass=1.0, damping=40.0, vmax_m_s=0.02, dt=0.005)
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    out = _step(law, v_in=0.01)
    assert out.v_force[2] != 0.0
    np.testing.assert_array_equal(out.v_force[[0, 1, 3, 4, 5]], 0.0)
    assert law.last["v_in"] == pytest.approx(0.01)
    assert law.last["v_z"] == pytest.approx(out.v_force_z)
    assert abs(out.v_force_z) <= 0.02 + 1e-12


def test_ac2d_writes_tool_z_and_omega_y():
    law = ComparisonAdmittance2D(mass=1.0, damping=40.0, vmax_m_s=0.02, dt=0.005)
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    out = _step(law, v_in=0.01)
    assert out.v_force[2] != 0.0
    assert out.v_force[4] != 0.0
    np.testing.assert_array_equal(out.v_force[[0, 1, 3, 5]], 0.0)
    assert law.last["law"] == "ac2d"
    assert abs(out.v_force[4]) <= 0.21 + 1e-12


def test_ac2d_matches_1d_on_z_and_holds_omega_before_contact():
    one = ComparisonAdmittance1D(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    two = ComparisonAdmittance2D(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    one.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    two.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    air = two.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.array([0.0, 0.0, -2.0, 0.0, 0.05, 0.0]),
        f_des=np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),
        path_twist=np.zeros(6),
        path_twist_raw=np.zeros(6),
        contact=False,
    )
    assert air.v_force[4] == pytest.approx(0.0)
    one.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    two.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    last_one = last_two = None
    for _ in range(20):
        last_one = _step(one, v_in=0.0, fz=3.5)
        last_two = _step(two, v_in=0.0, fz=3.5)
    assert last_two.v_force_z == pytest.approx(last_one.v_force_z)
    assert last_two.v_force[4] != 0.0


def test_ac2d_moment_track_is_75_percent_and_has_coulomb_deadband():
    from peirastic.realman8dof.force.comparison_laws import (
        DEFAULT_COULOMB_YY,
        DEFAULT_DAMPING_YY,
        DEFAULT_VMAX_OMEGA_Y,
        MOMENT_TRACK_SCALE,
    )

    assert MOMENT_TRACK_SCALE == pytest.approx(0.75)
    assert DEFAULT_DAMPING_YY == pytest.approx(0.22 / 0.75)
    assert DEFAULT_VMAX_OMEGA_Y == pytest.approx(0.21)
    assert DEFAULT_COULOMB_YY == pytest.approx(0.02)
    law = ComparisonAdmittance2D(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    quiet = law.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.array([0.0, 0.0, 3.5, 0.0, 0.015, 0.0]),
        f_des=np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),
        path_twist=np.zeros(6),
        path_twist_raw=np.zeros(6),
        contact=True,
    )
    assert quiet.v_force[4] == pytest.approx(0.0)
    assert law.last["e_m"] == pytest.approx(0.0)
    last = None
    for _ in range(40):
        last = _step(law, fz=3.5)
    assert last.v_force[4] != 0.0
    assert abs(last.v_force[4]) < 0.05 / 0.22 * 0.8


def test_tafac_adds_reference_normal_velocity():
    ac = ComparisonAdmittance1D(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    tafac = TafacNormalLaw(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    ac.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    tafac.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    last_ac = last_tafac = None
    for i in range(40):
        v_in = 0.008 + 0.002 * np.sin(i * 0.2)
        last_ac = _step(ac, v_in=v_in)
        last_tafac = _step(tafac, v_in=v_in)
    assert tafac.last["v_z"] == pytest.approx(tafac.last["v_in"] + tafac.last["delta_v"])
    assert tafac.last["tau_s"] == pytest.approx(0.005)
    assert tafac.last["v_z"] != pytest.approx(ac.last["v_z"], abs=1e-6)


def test_tafac_matches_ac_when_reference_normal_is_still():
    ac = ComparisonAdmittance1D(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    tafac = TafacNormalLaw(mass=1.0, damping=40.0, vmax_m_s=0.05, dt=0.005)
    ac.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    tafac.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    # Warm the delayed TAFAC force term so both see a constant error.
    for _ in range(80):
        ac_out = _step(ac, v_in=0.0, fz=3.5)
        tafac_out = _step(tafac, v_in=0.0, fz=3.5)
    assert ac_out.v_force_z == pytest.approx((4.0 - 3.5) / 40.0, rel=0.05)
    assert tafac.last["v_in"] == pytest.approx(0.0)
    assert tafac.last["a_in"] == pytest.approx(0.0)
    assert tafac_out.v_force_z == pytest.approx(tafac.last["v_z"])


def test_comparison_contact_latch_matches_scan_enter_window():
    latch = ComparisonContactLatch(enter_n=0.8, confirm_s=0.05)
    for _ in range(9):
        assert latch.observe(2.0, 0.005) is False
    assert latch.observe(2.0, 0.005) is True
    latch.observe(-1.0, 0.005)
    assert latch.contact_present is False
    latch.reset()
    assert latch.observe(0.4, 0.005) is False


def test_comparison_law_from_payload_builds_ac2d():
    law = comparison_law_from_payload(
        0.005,
        {
            "law": "ac2d",
            "admittance_inertia_yy": 0.051,
            "admittance_damping_yy": 0.22 / 0.75,
            "max_omega_y_rad_s": 0.21,
            "admittance_coulomb_yy": 0.02,
        },
    )
    assert isinstance(law, ComparisonAdmittance2D)
    assert law.inertia_yy == pytest.approx(0.051)
    assert law.damping_yy == pytest.approx(0.22 / 0.75)
    assert law.vmax_omega_y == pytest.approx(0.21)
    assert law.coulomb_yy == pytest.approx(0.02)
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    out = _step(law, fz=3.0)
    assert out.v_force[4] != 0.0


def test_comparison_law_exposes_controller_contact_present():
    law = comparison_law_from_payload(
        0.005,
        {"law": "admittance_1d", "contact_enter_n": 0.8, "enter_confirm_s": 0.005},
    )
    assert hasattr(law.controller, "contact_present")
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    _step(law, fz=-2.0)
    assert law.controller.contact_present is False
    out = _step(law, fz=2.0)
    assert law.controller.contact_present is True
    assert law.controller.v_force_cmd_z == pytest.approx(out.v_force_z)
    assert np.isfinite(law.controller.v_force_cmd_z)


def test_wrap_comparison_phase_calls_original_sample_once(tmp_path):
    from types import SimpleNamespace

    from peirastic.realman8dof.force.comparison_laws import wrap_comparison_phase

    calls = []

    class Outer:
        force_law = SimpleNamespace(last={"v_z": 0.01})

        def sample(self, t_s, pose, f_ext, **kwargs):
            calls.append(t_s)
            return np.array([0.01, 0.0, 0.002, 0.0, 0.0, 0.0])

    phase = SimpleNamespace(outer=Outer(), on_tick=None, on_exit=None)
    wrap_comparison_phase(phase, {
        "comparison_log_path": str(tmp_path / "comparison_law.jsonl"),
        "law": "admittance_1d",
    })
    twist = phase.outer.sample(0.1, np.zeros(6), np.zeros(6))
    assert calls == [0.1]
    np.testing.assert_allclose(twist[2], 0.002)
    phase.on_exit()


def test_hfpc_payload_accepts_comparison_laws_and_polyline_contact_qp():
    ac = HfpcPayload(reference="polyline", law="admittance_1d", poses=[[0] * 6] * 2).to_json()
    assert ac["law"] == "admittance_1d"
    assert ac["use_tff_split"] is True
    tafac = HfpcPayload(reference="polyline", law="tafac", poses=[[0] * 6] * 2).to_json()
    assert tafac["law"] == "tafac"
    ac2d = HfpcPayload(reference="polyline", law="ac2d", poses=[[0] * 6] * 2).to_json()
    assert ac2d["law"] == "ac2d"
    study = HfpcPayload(
        reference="polyline", contact_qp={"mode": "baseline"}, poses=[[0] * 6] * 2
    ).to_json()
    assert study["contact_qp"] == {"mode": "baseline"}
    with pytest.raises(ValueError, match="icra_path or polyline"):
        HfpcPayload(reference="hold", contact_qp={"mode": "baseline"}).to_json()
