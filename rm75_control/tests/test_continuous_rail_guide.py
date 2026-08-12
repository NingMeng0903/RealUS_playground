from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.continuous_guide import (
    ContinuousGuideLimits,
    ContinuousRailGuide,
    GuideStatus,
    GuideTarget,
    RedundancyState,
    RuckigUnavailableError,
    SrsMappingResult,
)


def _mapped_q(rail: float, psi: float) -> np.ndarray:
    q = np.zeros(8)
    q[0] = rail
    q[1] = psi
    q[2] = 10.0 * rail
    q[3] = -0.5 * psi
    return q


def test_production_rail_limits_and_explicit_unavailable_state() -> None:
    limits = ContinuousGuideLimits()
    assert limits.rail_v_max_m_s == pytest.approx(0.08)
    assert limits.rail_a_max_m_s2 == pytest.approx(0.30)
    assert limits.rail_j_max_m_s3 == pytest.approx(1.5)

    guide = ContinuousRailGuide(
        lambda ctx: _mapped_q(ctx.rail_m, ctx.psi_unwrapped_rad),
        ruckig_module=False,
    )
    assert not guide.available
    assert guide.status is GuideStatus.UNAVAILABLE
    assert guide.reset(rail_m=0.0, psi_rad=0.0) is GuideStatus.UNAVAILABLE
    sample = guide.update()
    assert sample.status is GuideStatus.UNAVAILABLE
    assert "Ruckig" in sample.reason
    with pytest.raises(RuckigUnavailableError):
        guide.require_available()


def test_ruckig_is_continuous_and_srs_mapping_runs_every_tick() -> None:
    calls = []

    def mapping(ctx):
        calls.append(ctx)
        return _mapped_q(ctx.rail_m, ctx.psi_unwrapped_rad)

    limits = ContinuousGuideLimits(
        arm_v_max_rad_s=0.20,
        arm_a_max_rad_s2=0.50,
    )
    guide = ContinuousRailGuide(mapping, dt_s=0.01, limits=limits)
    guide.require_available()
    guide.reset(
        RedundancyState(rail_m=0.0, psi_rad=0.0, branch=2),
        q_goal=_mapped_q(0.0, 0.0),
    )
    guide.set_target(
        GuideTarget(
            rail_m=0.03,
            psi_rad=0.20,
            branch=2,
            winding=0,
            mapping_input={"pose_id": 7},
        )
    )

    samples = []
    for _ in range(1500):
        sample = guide.update()
        samples.append(sample)
        assert sample.valid, sample.reason
        assert sample.q_goal is not None and sample.q_goal.shape == (8,)
        assert sample.qdot_guide is not None and sample.qdot_guide.shape == (8,)
        assert sample.qddot_guide is not None and sample.qddot_guide.shape == (8,)
        assert sample.q_goal[0] == pytest.approx(sample.rail_m)
        assert sample.q_goal[1] == pytest.approx(sample.psi_unwrapped_rad)
        assert sample.q_goal[2] == pytest.approx(10.0 * sample.rail_m)
        assert sample.qdot_guide[0] == pytest.approx(sample.rail_velocity_m_s)
        assert np.max(np.abs(sample.qdot_guide[1:])) <= 0.20 + 1e-9
        assert np.max(np.abs(sample.qddot_guide[1:])) <= 0.50 + 1e-8
        if sample.finished:
            break

    assert samples[-1].finished
    assert len(calls) == len(samples)
    assert samples[-1].rail_m == pytest.approx(0.03, abs=1e-8)
    assert samples[-1].psi_unwrapped_rad == pytest.approx(0.20, abs=1e-8)
    assert all(call.mapping_input == {"pose_id": 7} for call in calls)
    with pytest.raises(ValueError):
        samples[-1].q_goal[0] = 1.0

    rail_accels = np.array([s.rail_acceleration_m_s2 for s in samples])
    rail_jerk = np.diff(np.r_[0.0, rail_accels]) / guide.dt_s
    assert np.max(np.abs(rail_jerk)) <= 1.5 + 1e-4
    assert max(abs(s.rail_velocity_m_s) for s in samples) <= 0.08 + 1e-8
    assert max(abs(s.rail_acceleration_m_s2) for s in samples) <= 0.30 + 1e-8


def test_nearest_winding_takes_short_psi_route_across_pi() -> None:
    start = math.pi - 0.02
    target = -math.pi + 0.02
    guide = ContinuousRailGuide(
        lambda ctx: _mapped_q(ctx.rail_m, ctx.psi_unwrapped_rad),
        dt_s=0.01,
    )
    guide.reset(
        rail_m=0.0,
        psi_rad=start,
        branch=1,
        winding=0,
        q_goal=_mapped_q(0.0, start),
    )
    resolved = guide.set_target(rail_m=0.0, psi_rad=target, branch=1, winding=None)
    assert resolved.winding == 1
    values = []
    for _ in range(1000):
        sample = guide.update()
        assert sample.valid, sample.reason
        values.append(sample.psi_unwrapped_rad)
        if sample.finished:
            break
    assert values[-1] == pytest.approx(math.pi + 0.02, abs=1e-8)
    assert max(values) - min(values) < 0.05
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_mapping_failure_and_arm_shaping_are_explicit() -> None:
    bad = ContinuousRailGuide(lambda _ctx: None)
    bad.reset(rail_m=0.0, psi_rad=0.0, q_goal=np.zeros(8))
    bad.set_target(rail_m=0.01, psi_rad=0.01, branch=0)
    failed = bad.update()
    assert failed.status is GuideStatus.MAPPING_FAILED
    assert not failed.valid
    assert "mapping" in failed.reason.lower()

    def huge_velocity(ctx):
        return SrsMappingResult(
            _mapped_q(ctx.rail_m, ctx.psi_unwrapped_rad),
            qdot_guide=np.array([0.0, 100.0, -50.0, 25.0, 0.0, 0.0, 0.0, 0.0]),
        )

    limits = ContinuousGuideLimits(arm_v_max_rad_s=0.4, arm_a_max_rad_s2=0.8)
    shaped = ContinuousRailGuide(huge_velocity, dt_s=0.01, limits=limits)
    shaped.reset(rail_m=0.0, psi_rad=0.0, q_goal=np.zeros(8))
    shaped.set_target(rail_m=0.001, psi_rad=0.001, branch=0)
    sample = shaped.update()
    assert sample.valid
    assert np.max(np.abs(sample.qdot_guide[1:])) <= 0.4 + 1e-9
    assert np.max(np.abs(sample.qddot_guide[1:])) <= 0.8 + 1e-9

