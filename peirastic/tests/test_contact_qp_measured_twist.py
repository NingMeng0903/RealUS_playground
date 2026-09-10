"""Full measurement forwarding cannot reuse the rail command predictor."""
from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import _FullMeasuredTwistTracker


class Kinematics:
    def __init__(self):
        self.last_q = None

    def jacobian(self, q):
        self.last_q = q.copy()
        result = np.zeros((6, 8))
        result[0, 0] = 1.
        result[1, 1] = 1.
        result[2, 2] = 1.
        result[3, 3] = 1.
        result[4, 4] = 1.
        result[5, 5] = 1.
        return result


def arm(seq, t, *, q=0., speed=10.):
    return SimpleNamespace(ok=True, seq=seq, t_s=t, q_deg=np.full(7, q),
                           qdot_deg_s=None if speed is None else np.full(7, speed))


def rail(seq, t, x, **kwargs):
    return SimpleNamespace(valid=True, motion_seq=seq, sample_mono_s=t, position_m=x,
                           v_meas_m_s=88., v_cmd_m_s=99., a_cmd_m_s2=999., **kwargs)


def update(tracker, snap, feedback, kin=None, **kwargs):
    return tracker.update(snap, feedback, Kinematics() if kin is None else kin,
                          now_s=kwargs.get('now_s', snap.t_s), freshness_s=.08,
                          max_skew_s=kwargs.get('max_skew_s', .02))


def test_raw_encoder_difference_and_sdk_speed_not_command_prediction():
    tracker, kin = _FullMeasuredTwistTracker(), Kinematics()
    assert not update(tracker, arm(1, 1.), rail(1, 1., .2))['measured_twist_valid']
    result = update(tracker, arm(2, 1.02), rail(2, 1.02, .2004), kin)
    assert result['measured_twist_valid'] and result['measured_twist_fresh']
    np.testing.assert_allclose(result['measured_twist_base'], [.02, *([np.deg2rad(10.)] * 5)])
    assert kin.last_q[0] == .2004  # raw position, no observer or command extrapolation
    metadata = result['measured_twist_metadata']
    assert metadata['rail_velocity_source'] == 'raw_encoder_difference'
    assert metadata['rail_velocity_interval_s'] == (1., 1.02)
    assert metadata['port_verified'] is False
    assert result['measurement_time_s'] == pytest.approx(1.01)


def test_duplicate_pair_is_not_fresh_and_input_identity_cannot_change():
    tracker = _FullMeasuredTwistTracker()
    update(tracker, arm(1, 1.), rail(1, 1., .2))
    update(tracker, arm(2, 1.02), rail(2, 1.02, .2004))
    duplicate = update(tracker, arm(2, 1.02), rail(2, 1.02, .2004))
    assert duplicate['measured_twist_valid'] and not duplicate['measured_twist_fresh']
    changed = update(tracker, arm(2, 1.02), rail(2, 1.02, .201))
    assert changed['measured_twist_metadata']['reason'] == 'rail_identity_changed'


def test_relay_republication_sequence_does_not_replace_source_measurement_identity():
    tracker=_FullMeasuredTwistTracker()
    update(tracker,arm(1,1.),rail(1,1.,.2))
    original=update(tracker,arm(2,1.02),rail(2,1.02,.2004))
    held=update(tracker,arm(99,1.02),rail(2,1.02,.2004),now_s=1.025)
    assert held['measured_twist_valid'] and not held['measured_twist_fresh']
    assert held['measured_twist_metadata']['arm_sequence']==99
    assert held['measurement_time_s']==original['measurement_time_s']
    np.testing.assert_array_equal(held['measured_twist_base'],original['measured_twist_base'])
    invalid=update(tracker,arm(100,1.02,q=.1),rail(2,1.02,.2004),now_s=1.025)
    assert invalid['measured_twist_metadata']['reason']=='arm_identity_changed'


@pytest.mark.parametrize('bad', ['missing', 'invalid', 'stale', 'future', 'nonfinite'])
def test_missing_or_untrusted_rail_never_defaults_to_zero(bad):
    tracker = _FullMeasuredTwistTracker()
    update(tracker, arm(1, 1.), rail(1, 1., .2))
    r = rail(2, 1.02, .2004)
    if bad == 'missing':
        r = None
    elif bad == 'invalid':
        r.valid = False
    elif bad == 'stale':
        r.sample_mono_s = .9
    elif bad == 'future':
        r.sample_mono_s = 1.1
    else:
        r.position_m = np.nan
    result = update(tracker, arm(2, 1.02), r)
    assert not result['measured_twist_valid']
    assert result['measured_twist_base'] is None
    assert result['measurement_time_s'] is None


def test_missing_sdk_speed_uses_only_raw_arm_position_difference():
    tracker = _FullMeasuredTwistTracker()
    update(tracker, arm(1, 1., q=1., speed=None), rail(1, 1., .2))
    result = update(tracker, arm(2, 1.02, q=1.2, speed=None), rail(2, 1.02, .2004))
    np.testing.assert_allclose(result['measured_twist_base'], [.02, *([np.deg2rad(10.)] * 5)])
    assert result['measured_twist_metadata']['arm_velocity_interval_s'] == (1., 1.02)


def test_joint_freshness_and_time_skew_are_checked():
    tracker = _FullMeasuredTwistTracker()
    update(tracker, arm(1, 1.), rail(1, 1., .2))
    result = update(tracker, arm(2, 1.04), rail(2, 1.02, .2004), max_skew_s=.01)
    assert result['measured_twist_metadata']['reason'] == 'measurement_time_skew'
    stale = update(tracker, arm(3, 1.041), rail(3, 1.04, .2005), now_s=1.2)
    assert stale['measured_twist_metadata']['reason'] == 'arm_stale_or_future'


def test_invalid_arm_and_regressing_sample_are_not_measurements():
    tracker = _FullMeasuredTwistTracker()
    update(tracker, arm(2, 1.), rail(1, 1., .2))
    result = update(tracker, arm(1, .99), rail(2, 1.02, .2004), now_s=1.02)
    assert result['measured_twist_metadata']['reason'] == 'arm_out_of_order'
    snap = arm(3, 1.02)
    snap.ok = False
    assert not update(tracker, snap, rail(2, 1.02, .2004))['measured_twist_valid']
