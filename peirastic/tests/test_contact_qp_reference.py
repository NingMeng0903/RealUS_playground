"""Hardware-free reference geometry and accepted-clock contract checks."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.scan_path import ForearmReference, make_spec
from rm75_control.control.admittance_common.reference import MotionReference


def forearm(shape="L", direction="DtP", rotating=False):
    start = np.array([0.1, 0.2, 0.3, 0.2, -0.3, 0.4])
    end = start.copy()
    end[1] += 0.06
    if rotating:
        end[3:] = [-0.3, 0.4, 0.8]
    return ForearmReference(make_spec(start, end, shape, direction, 31, speed=0.02))


@pytest.mark.parametrize("shape", ["L", "C", "S"])
@pytest.mark.parametrize("direction", ["DtP", "PtD"])
def test_production_ramp_starts_without_advancing_pose_or_clock(shape, direction):
    source = forearm(shape, direction)
    adapter = FiniteIntervalReference(source, dt_s=0.005)
    assert np.array_equal(source.sample(0).vel_ff, np.zeros(6))
    before = source.sample(0).pose_d
    candidate, h_ref = adapter.candidate(0)
    assert np.array_equal(candidate.pose_d, before)
    assert h_ref == 0.005
    assert np.linalg.norm(candidate.vel_ff[:3]) > 0.0
    np.testing.assert_allclose(candidate.vel_ff[:3] * 0.005,
                               source.sample(0.005).pose_d[:3] - before[:3], atol=1e-20)
    # Rejected proposals, retries, and arbitrary sampling do not move the clock.
    adapter.sample(2.0)
    assert np.array_equal(adapter.sample(0).pose_d, before)
    assert adapter.commit_time(0, 0) == 0


@pytest.mark.parametrize("direction", ["DtP", "PtD"])
def test_general_orientation_uses_world_rotation_displacement(direction):
    source = forearm("C", direction, rotating=True)
    adapter = FiniteIntervalReference(source)
    start, dt = 0.35, 0.03
    candidate, h_ref = adapter.candidate(start, dt)
    r0 = Rotation.from_euler("xyz", source.sample(start).pose_d[3:])
    r1 = Rotation.from_euler("xyz", source.sample(start + h_ref).pose_d[3:])
    integrated = Rotation.from_rotvec(candidate.vel_ff[3:] * dt) * r0
    np.testing.assert_allclose(integrated.as_matrix(), r1.as_matrix(), atol=1e-14)
    # The existing instantaneous world twist points along the same Slerp axis.
    np.testing.assert_allclose(np.cross(candidate.vel_ff[3:], source.sample(start).vel_ff[3:]),
                               np.zeros(3), atol=1e-14)


class LinearReference:
    duration_s = 1.0

    def sample(self, t_s):
        pose = np.array([t_s, 2 * t_s, 0.0, 0.0, 0.0, np.pi - 0.1 + 0.2 * t_s])
        pose[3:] = Rotation.from_euler("xyz", pose[3:]).as_euler("xyz")
        return MotionReference(pose, np.zeros(6), t_s)


def test_short_final_interval_uses_full_execution_hold_and_short_commit():
    adapter = FiniteIntervalReference(LinearReference())
    sample, h_ref = adapter.candidate(0.99, 0.1)
    assert h_ref == pytest.approx(0.01)
    assert sample.pose_d[0] == 0.99
    np.testing.assert_allclose(sample.vel_ff, [0.1, 0.2, 0, 0, 0, 0.02], atol=1e-14)
    assert adapter.commit_time(0.99, 0.25, 0.1) == pytest.approx(0.9925)
    assert adapter.commit_time(0.99, 1, 0.1) == 1.0
    assert np.array_equal(adapter.sample(1, 0.1).vel_ff, np.zeros(6))


def test_rotation_wrap_is_shortest_world_rotation():
    adapter = FiniteIntervalReference(LinearReference(), dt_s=0.2)
    sample = adapter.sample(0.4)
    assert sample.vel_ff[5] == pytest.approx(0.2)
    assert np.linalg.norm(sample.vel_ff[3:]) < 1.0


@pytest.mark.parametrize("with_origin_hook", [False, True])
def test_hot_install_and_bounded_global_clock(with_origin_hook):
    source = forearm() if with_origin_hook else LinearReference()
    adapter = FiniteIntervalReference(source, dt_s=0.005)
    original = source.sample(0).pose_d.copy()
    adapter.set_origin(np.zeros(6), t_s=73.25)
    assert adapter.origin_s == 73.25
    assert np.array_equal(adapter.sample(73.25).pose_d, original)
    assert np.array_equal(adapter.sample(-100).pose_d, original)
    assert adapter.local_time(-100) == 0.0
    assert adapter.local_time(1e8) == adapter.duration_s
    assert adapter.local_time(adapter.origin_s + adapter.duration_s) == adapter.duration_s
    assert adapter.exhaustion_reason(adapter.origin_s + adapter.duration_s) == "reference_clock_end"
    assert adapter.commit_time(73.25, 0.4) == pytest.approx(73.252)
    assert adapter.sample(73.25).t_ref == 73.25
    adapter.set_origin(np.zeros(6), t_s=None)
    assert adapter.local_time(0.25) == 0.25


def test_legacy_origin_hook_uses_local_clock():
    class Legacy(LinearReference):
        def set_origin(self, pose0):
            self.pose0 = pose0.copy()

    source = Legacy()
    adapter = FiniteIntervalReference(source, 0.1)
    adapter.set_origin(np.ones(6), t_s=9.0)
    assert np.array_equal(source.pose0, np.ones(6))
    assert adapter.sample(9.25).pose_d[0] == 0.25


def test_alpha_zero_and_roundoff_snap_are_distinct():
    adapter = FiniteIntervalReference(LinearReference(), 0.005)
    near_end = np.nextafter(1.0, 0.0)
    assert adapter.commit_time(near_end, 0.0) == near_end
    assert adapter.commit_time(near_end, np.nextafter(0.0, 1.0)) == near_end
    assert adapter.commit_time(near_end, 0.41674) == 1.0
    assert adapter.commit_time(0.2, -1) == 0.2
    assert adapter.commit_time(0.2, 2) == pytest.approx(0.205)
    accepted = 0.0
    for _ in range(1000):
        accepted = adapter.commit_time(accepted, 0.41674)
    assert accepted == 1.0


def test_final_geometry_exhaustion_does_not_invent_accepted_time():
    source = ForearmReference(make_spec(np.zeros(6), [0, .06, 0, 0, 0, 0],
                                        "L", "DtP", 0, speed=.02))
    adapter = FiniteIntervalReference(source, 0.005)
    accepted = source.duration_s - 1e-6
    assert np.array_equal(source.sample(accepted).pose_d, source.sample(source.duration_s).pose_d)
    assert adapter.exhaustion_reason(accepted) == "reference_geometry_roundoff"
    assert adapter.commit_time(accepted, 0.0) == accepted
    assert np.array_equal(adapter.sample(accepted).vel_ff, np.zeros(6))
    assert adapter.exhaustion_reason(source.duration_s) == "reference_clock_end"


def test_equal_position_with_different_orientation_is_not_geometry_exhaustion():
    class Rotating(LinearReference):
        ramp = 0.4

        def sample(self, t_s):
            value = super().sample(t_s)
            value.pose_d[:3] = 0.0
            return value

    adapter = FiniteIntervalReference(Rotating(), 0.1)
    assert adapter.exhaustion_reason(0.9) == ""


def test_endpoint_equality_outside_final_ramp_does_not_complete():
    class Hold(LinearReference):
        ramp = 0.4

        def sample(self, t_s):
            return MotionReference.from_pose_hold(np.zeros(6))

    adapter = FiniteIntervalReference(Hold(), 0.1)
    assert adapter.exhaustion_reason(0.0) == ""
    assert adapter.exhaustion_reason(0.5) == ""
    assert adapter.exhaustion_reason(0.7) == "reference_geometry_roundoff"
    assert adapter.commit_time(0.7, 0) == 0.7


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_clock_hold_and_alpha_fail(bad):
    adapter = FiniteIntervalReference(LinearReference(), 0.1)
    with pytest.raises(ValueError):
        adapter.sample(bad)
    with pytest.raises(ValueError):
        adapter.candidate(0, bad)
    with pytest.raises(ValueError):
        adapter.commit_time(0, bad)
    with pytest.raises(ValueError):
        adapter.set_origin(np.zeros(6), t_s=bad)


@pytest.mark.parametrize("hold", [0.0, -0.1])
def test_nonpositive_hold_fails(hold):
    with pytest.raises(ValueError):
        FiniteIntervalReference(LinearReference(), hold)


def test_invalid_source_sample_stays_invalid_and_pose_is_copied():
    class Invalid(LinearReference):
        def sample(self, t_s):
            return MotionReference(np.zeros(6), np.zeros(6), valid=t_s == 0)

    adapter = FiniteIntervalReference(Invalid(), 0.1)
    assert not adapter.sample(0).valid
