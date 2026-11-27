from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from perception.benchmark.hybrik_backend import (  # noqa: E402
    InitializedHybrIKX,
    NOMINAL_FRAME_DT_S,
    so3_chordal_loss,
    temporal_dt,
)


def _rotation_z(angle: float) -> torch.Tensor:
    cosine, sine = np.cos(angle), np.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _decode_rotation_6d(value: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(value[..., :3], dim=-1)
    second = value[..., 3:] - (first * value[..., 3:]).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def test_chordal_prior_uses_decoded_rotation_not_6d_coordinates() -> None:
    reference = _rotation_z(0.2).reshape(1, 1, 3, 3)
    current = _rotation_z(0.35).reshape(1, 1, 3, 3)

    # The two 6D vectors below are different redundant representations of the
    # same first two rotation columns.  Their decoded SO(3) matrices are the
    # same, so a matrix-space prior must give one identical value.
    six_d = torch.cat((current[..., :, 0], current[..., :, 1]), dim=-1)
    equivalent_six_d = six_d * torch.tensor([2, 2, 2, 3, 3, 3], dtype=six_d.dtype)
    decoded_current = _decode_rotation_6d(six_d)
    decoded_equivalent = _decode_rotation_6d(equivalent_six_d)
    torch.testing.assert_close(decoded_current, decoded_equivalent, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        so3_chordal_loss(decoded_current, reference),
        so3_chordal_loss(decoded_equivalent, reference),
        atol=1e-7,
        rtol=1e-6,
    )


def test_temporal_dt_normalizes_30_10_and_2_hz_and_rejects_rewind() -> None:
    dt_30, scale_30 = temporal_dt(1_000_000_000 + 33_333_333, 1_000_000_000)
    dt_10, scale_10 = temporal_dt(1_000_000_000 + 100_000_000, 1_000_000_000)
    dt_2, scale_2 = temporal_dt(1_000_000_000 + 500_000_000, 1_000_000_000)

    np.testing.assert_allclose(dt_30, 33_333_333e-9, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale_30, 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(dt_10, 0.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale_10, 1.0 / 3.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(dt_2, 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale_2, 1.0 / 15.0, rtol=0, atol=1e-6)
    assert temporal_dt(None, None) == (NOMINAL_FRAME_DT_S, 1.0)
    with pytest.raises(ValueError, match="increase strictly"):
        temporal_dt(10, 10)


def test_step_rolls_back_pose_prior_root_and_timestamp_after_failure() -> None:
    backend = InitializedHybrIKX.__new__(InitializedHybrIKX)
    backend.pose = torch.eye(3, dtype=torch.float32).repeat(1, 55, 1, 1)
    backend.prior = np.arange(55 * 3, dtype=np.float32).reshape(55, 3)
    backend.pelvis = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    backend.last_timestamp_ns = 123
    saved_pose = backend.pose.clone()
    saved_prior = backend.prior.copy()
    saved_pelvis = backend.pelvis.copy()

    def fail_after_mutating(self, points, *, timestamp_ns=None):
        self.pose[0, 0, 0, 0] = 99.0
        self.prior[0, 0] = 99.0
        self.pelvis[:] = 99.0
        self.last_timestamp_ns = timestamp_ns
        raise ValueError("synthetic rejected sample")

    backend._step_impl = MethodType(fail_after_mutating, backend)
    with pytest.raises(ValueError, match="synthetic rejected"):
        backend.step(np.zeros((67, 4), dtype=np.float32), timestamp_ns=456)

    torch.testing.assert_close(backend.pose, saved_pose)
    np.testing.assert_array_equal(backend.prior, saved_prior)
    np.testing.assert_array_equal(backend.pelvis, saved_pelvis)
    assert backend.last_timestamp_ns == 123


def _stationary_backend():
    from perception.benchmark.stationary_observation import StationaryObservationGate

    backend = InitializedHybrIKX.__new__(InitializedHybrIKX)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.min_confidence = .05
    backend.pose_initialization = "previous"
    backend.phis = torch.zeros((1, 54, 2))
    backend.pose = torch.eye(3).repeat(1, 55, 1, 1)
    backend.prior = np.zeros((71, 3), np.float32)
    backend.pelvis = np.zeros(3, np.float32)
    backend.last_timestamp_ns = 0
    backend.stationary_gate = StationaryObservationGate()
    points = np.column_stack((np.zeros((67, 3)), np.ones(67))).astype(np.float32)
    backend.stationary_gate.commit(points)
    backend._stationary_result = dict(
        poses=np.zeros((1, 165), np.float32), vertices=np.ones((4, 3), np.float32),
        joints_world_m=np.ones((55, 3), np.float32), stationary_tolerance_m=.002,
    )
    return backend, points


@pytest.mark.parametrize("interval_ns", [33_333_333, 625_000_000])
def test_static_hold_preserves_full_geometry_and_advances_source_clock(interval_ns):
    backend, points = _stationary_backend()
    pose_before, prior_before = backend.pose.clone(), backend.prior.copy()
    result = backend.step(points, timestamp_ns=interval_ns)
    assert result["stationary_held"]
    assert backend.last_timestamp_ns == interval_ns
    torch.testing.assert_close(backend.pose, pose_before)
    np.testing.assert_array_equal(backend.prior, prior_before)
    np.testing.assert_array_equal(result["vertices"], backend._stationary_result["vertices"])
    # A downstream consumer cannot mutate the cached pose/mesh or the anchor.
    result["vertices"][0, 0] = 99
    repeated = backend.step(points, timestamp_ns=2 * interval_ns)
    assert repeated["vertices"][0, 0] == 1
    np.testing.assert_array_equal(backend.stationary_gate.snapshot().points, points)
    with pytest.raises(ValueError, match="increase strictly"):
        backend.step(points, timestamp_ns=2 * interval_ns)
    assert backend.last_timestamp_ns == 2 * interval_ns


def test_failed_step_restores_stationary_anchor_and_result_cache():
    backend, points = _stationary_backend()
    cached = backend._stationary_result

    def fail_after_stationary_update(self, points, *, timestamp_ns=None):
        changed = points.copy()
        changed[:, 0] = 1
        self.stationary_gate.commit(changed)
        self._stationary_result = {"vertices": np.zeros((4, 3))}
        self.last_timestamp_ns = timestamp_ns
        raise ValueError("synthetic rejected sample")

    backend._step_impl = MethodType(fail_after_stationary_update, backend)
    with pytest.raises(ValueError, match="synthetic rejected"):
        backend.step(points, timestamp_ns=100)
    assert backend.last_timestamp_ns == 0
    assert backend._stationary_result is cached
    assert backend.stationary_gate.matches(points)


def test_failed_step_restores_adaptive_observation_history():
    from perception.benchmark.adaptive_observation import AdaptiveObservationFilter

    backend, points = _stationary_backend()
    backend.observation_filter = AdaptiveObservationFilter()
    backend.observation_filter.apply(points, 0)
    before = backend.observation_filter.snapshot()

    def fail_after_filtering(self, points, *, timestamp_ns=None):
        changed = points.copy()
        changed[:, 0] = 1
        self.observation_filter.apply(changed, timestamp_ns)
        self.last_timestamp_ns = timestamp_ns
        raise ValueError("rejected after observation filtering")

    backend._step_impl = MethodType(fail_after_filtering, backend)
    with pytest.raises(ValueError, match="rejected after"):
        backend.step(points, timestamp_ns=33_333_333)
    after = backend.observation_filter.snapshot()
    assert after.timestamp_ns == before.timestamp_ns == 0
    for name in ("previous", "filtered", "velocity", "valid"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    assert backend.last_timestamp_ns == 0
