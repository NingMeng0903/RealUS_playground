"""Apex-held +tool-Z sine: kinematics and TRACK_CARTESIAN compile."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.api import OK, PeirasticArm
from peirastic.core.modes import Mode
from peirastic.realman8dof.session import compile_request
from peirastic.tests.test_api_facade import _SEED, _arm, _ctx
from rm75_control.control.joint_admittance_8dof.reference import (
    SinToolZReference,
    tool_z_sine_motion,
)


def test_tool_z_sine_starts_at_apex_and_only_presses_plus_z():
    dz0, vz0 = tool_z_sine_motion(0.0, 0.002, 2 * math.pi, soft_start=False)
    assert dz0 == pytest.approx(0.0)
    assert vz0 == pytest.approx(0.0)
    times = np.linspace(0, 1, 401)
    dz = np.array([tool_z_sine_motion(t, 0.002, 2 * math.pi, soft_start=False)[0] for t in times])
    assert dz.min() >= -1e-15
    assert dz.max() == pytest.approx(0.002, abs=1e-12)
    assert dz[0] == pytest.approx(0.0, abs=1e-15)
    assert dz[200] == pytest.approx(0.002, abs=1e-12)


def test_sin_tool_z_reference_is_plus_tool_z_and_soft_start_is_still():
    origin = np.array([0.2, 0.1, 0.3, 0.0, 0.0, 0.0])
    ref = SinToolZReference(0.002, period_s=1.0, duration_s=4.0, ramp_s=0.4, soft_start=True)
    ref.set_origin(origin, t_s=0.0)
    first = ref.sample(0.0)
    np.testing.assert_allclose(first.pose_d, origin)
    np.testing.assert_allclose(first.vel_ff, 0.0, atol=1e-12)
    last = ref.sample(ref.duration_s)
    np.testing.assert_allclose(last.vel_ff, 0.0, atol=1e-9)
    assert float((last.pose_d[:3] - origin[:3]) @ Rotation.from_euler("xyz", origin[3:]).as_matrix()[:, 2]) >= -1e-12
    mid = ref.sample(0.5 + 0.2)
    axis = Rotation.from_euler("xyz", origin[3:]).as_matrix()[:, 2]
    delta = mid.pose_d[:3] - origin[:3]
    lateral = delta - float(delta @ axis) * axis
    np.testing.assert_allclose(lateral, 0.0, atol=1e-12)
    assert float(delta @ axis) > 0.0
    assert float(delta @ axis) <= 0.002 + 1e-9
    assert ref.duration_s == pytest.approx(4.5)


def test_sin_tool_z_explicit_origin_survives_live_reseed_and_ends_at_apex():
    origin = np.array([0.20, 0.10, 0.30, 0.10, -0.20, 0.30])
    live = np.array([0.90, -0.30, 0.70, -0.40, 0.50, -0.60])
    ref = SinToolZReference(
        0.002, period_s=1.0, duration_s=15.0, ramp_s=0.4,
        stop_ramp_s=0.4, origin_pose=origin,
    )
    ref.set_origin(live, t_s=2.0)
    np.testing.assert_allclose(ref.sample(2.0).pose_d, origin)
    np.testing.assert_allclose(ref.sample(2.0 + ref.duration_s).pose_d, origin,
                               atol=1e-12)
    np.testing.assert_allclose(ref.sample(2.0 + ref.duration_s).vel_ff, 0.0,
                               atol=1e-12)


@pytest.mark.parametrize("frequency_hz", [1.0, 1.5, 2.0])
def test_sin_tool_z_three_frequencies_complete_integer_cycles(frequency_hz):
    origin = np.array([0.2, 0.1, 0.3, 0.0, 0.0, 0.0])
    duration = 15.0 / frequency_hz
    ramp = 0.4
    ref = SinToolZReference(
        0.002, period_s=1.0 / frequency_hz, duration_s=duration,
        ramp_s=ramp, stop_ramp_s=ramp, origin_pose=origin,
    )
    ref.set_origin(np.full(6, 9.0), t_s=0.0)
    end = ref.sample(ref.duration_s)
    np.testing.assert_allclose(end.pose_d, origin, atol=1e-12)
    np.testing.assert_allclose(end.vel_ff, 0.0, atol=1e-12)


def test_sin_tool_z_rejects_oversized_amplitude():
    with pytest.raises(ValueError, match="5 mm"):
        SinToolZReference(0.006, period_s=1.0)


def test_cartesian_track_tool_z_sine_compiles():
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    assert arm.cartesian_track(
        reference="tool_z_sine",
        amplitude_z_m=0.002,
        period_s=1.0,
        duration_s=3.0,
        ramp_s=0.4,
        stop_ramp_s=0.4,
        origin_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        block=0,
        label="delay_cal_z",
    ) == OK
    req = arm.last_request
    assert req.mode == Mode.TRACK_CARTESIAN
    assert req.payload["reference"] == "tool_z_sine"
    assert req.payload["amplitude_z_m"] == pytest.approx(0.002)
    assert req.payload["stop_ramp_s"] == pytest.approx(0.4)
    assert req.payload["origin_pose"] == pytest.approx([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    phase = compile_request(ctx, req, raw=raw)
    assert isinstance(phase.outer.reference, SinToolZReference)
    assert phase.duration_s == pytest.approx(3.4)
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose, t_s=0.0)
    twist0 = np.asarray(phase.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert twist0.shape == (6,)
    assert np.all(np.isfinite(twist0))
