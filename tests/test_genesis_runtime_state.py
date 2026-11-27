from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perception.apps.genesis_runtime_state import (
    decode_fitted_track_payload,
    RM75_8DOF_JOINT_ORDER,
    RM75_8DOF_Q_SEMANTICS,
    RM75_8DOF_UNITS,
    RM75_8DOF_ZERO_OFFSET_POLICY,
    ShmRobotStateAdapter,
    SimulationRobotStateAdapter,
)


def test_rm75_contract_is_explicit_and_simulation_accepts_legacy_arm_home() -> None:
    assert RM75_8DOF_JOINT_ORDER == (
        "rail_y",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    )
    assert RM75_8DOF_UNITS == ("m", "rad", "rad", "rad", "rad", "rad", "rad", "rad")
    assert RM75_8DOF_Q_SEMANTICS == "raw_measured"
    assert RM75_8DOF_ZERO_OFFSET_POLICY == "not_applied_in_adapter"
    sample = SimulationRobotStateAdapter(np.arange(7, dtype=np.float64)).read()
    np.testing.assert_allclose(sample.q8dof, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert sample.source == "simulation"


def test_shm_adapter_converts_feedback_degrees_and_prepends_rail() -> None:
    class FakeBus:
        last_rail_m = 0.125

        def read(self):
            return SimpleNamespace(
                ok=True,
                seq=17,
                wall_time_ns=123456789,
                t_s=4.5,
                q_deg=np.array([0.0, 90.0, -90.0, 180.0, 45.0, 0.0, -45.0]),
            )

        def stop(self):
            self.stopped = True

    bus = FakeBus()
    adapter = ShmRobotStateAdapter(bus=bus, timeout_s=0.5)
    sample = adapter.read()
    assert sample is not None
    np.testing.assert_allclose(
        sample.q8dof,
        [0.125, 0.0, np.pi / 2.0, -np.pi / 2.0, np.pi, np.pi / 4.0, 0.0, -np.pi / 4.0],
        atol=1e-12,
    )
    assert sample.sequence == 17
    assert sample.wall_time_ns == 123456789
    assert sample.source_time_s == pytest.approx(4.5)
    adapter.close()
    assert bus.stopped is True


def test_shm_adapter_holds_no_invalid_or_stale_frame() -> None:
    class FakeBus:
        last_rail_m = 0.0

        def __init__(self):
            self.snapshot = SimpleNamespace(ok=False, seq=0, q_deg=None)

        def read(self):
            return self.snapshot

        def stop(self):
            pass

    adapter = ShmRobotStateAdapter(bus=FakeBus(), timeout_s=0.01)
    assert adapter.read() is None

    adapter._bus.snapshot = SimpleNamespace(
        ok=True,
        seq=1,
        wall_time_ns=0,
        t_s=0.0,
        q_deg=np.zeros(7, dtype=np.float64),
    )
    assert adapter.read() is not None
    adapter._last_new_frame_mono -= 1.0
    assert adapter.read() is None


def test_simulation_rejects_ambiguous_home_size() -> None:
    with pytest.raises(ValueError, match="7 or 8"):
        SimulationRobotStateAdapter(np.zeros(6))


def test_fitted_track_decoder_preserves_full_pose_and_metadata() -> None:
    payload = {
        "payload_kind": "mesh_vertices",
        "frame_index": 7,
        "timestamp_ns": 1234,
        "smplx_pose_aa_165": np.zeros((55, 3), dtype=np.float32).reshape(-1).tolist(),
        "translation_m": [0.1, 0.2, 0.3],
        "betas": [0.2] * 10,
        "gender": "female",
        "expression": [0.4] * 10,
        "source_id": "synthetic_fit",
    }
    sample = decode_fitted_track_payload(payload)
    assert sample is not None
    assert sample.pose55.size == 165
    assert sample.frame_index == 7
    assert sample.timestamp_ns == 1234
    assert sample.gender == "female"
    assert sample.source_id == "synthetic_fit"
    assert sample.expression.size == 10
    np.testing.assert_allclose(sample.betas, 0.2)


def test_fitted_track_decoder_maps_legacy_smpl_pose() -> None:
    payload = {
        "payload_kind": "smpl_pose",
        "frame_index": 3,
        "pose_aa": np.zeros(72, dtype=np.float32).tolist(),
        "translation_m": [0.0, 0.0, 0.0],
        "betas": [0.0] * 10,
    }
    sample = decode_fitted_track_payload(payload)
    assert sample is not None
    assert sample.pose55.shape == (165,)


def test_fitted_track_decoder_preserves_zero_frame_and_timestamp() -> None:
    payload = {
        "payload_kind": "mesh_vertices",
        "frame_index": 0,
        "timestamp_ns": 0,
        "smplx_pose_aa_165": np.zeros(165, dtype=np.float32).tolist(),
        "translation_m": [0.0, 0.0, 0.0],
    }
    sample = decode_fitted_track_payload(payload)
    assert sample is not None
    assert sample.frame_index == 0
    assert sample.timestamp_ns == 0


def test_fitted_track_decoder_preserves_pelvis_pivot_separately_from_raw_translation() -> None:
    payload = {
        "payload_kind": "mesh_vertices",
        "frame_index": 0,
        "timestamp_ns": 0,
        "smplx_pose_aa_165": np.zeros(165, dtype=np.float32).tolist(),
        "translation_m": [0.2, 0.3, 0.4],
        "pelvis_world_m": [0.01, 0.02, 0.53],
    }
    sample = decode_fitted_track_payload(payload)
    assert sample is not None
    np.testing.assert_allclose(sample.transl, [0.2, 0.3, 0.4])
    np.testing.assert_allclose(sample.pelvis_world_m, [0.01, 0.02, 0.53])
