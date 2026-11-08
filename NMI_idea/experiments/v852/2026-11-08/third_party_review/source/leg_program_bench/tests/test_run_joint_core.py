"""Small contracts for the independent joint-core experiment runner."""
import numpy as np
import pytest

from leg_program_bench.run_joint_core import image_pose, default_specs, content_hash, PlannedProgramExecutor


def test_public_pose_uses_reference_only():
    class World:
        def reference_at(self, progress, mode):
            assert progress == .6 and mode == 1
            return np.array([1., 2., 3.]), np.eye(3)
        @property
        def spec(self):
            raise AssertionError("hidden spec must not be read")
    T = image_pose(World(), .6, 1)
    np.testing.assert_allclose(T[:3, 3], [1., 2., 3.])
    np.testing.assert_allclose(T[:3, :3], np.eye(3))


def test_train_test_configuration_families_are_disjoint():
    train = default_specs(4, "train", 100, 0.)
    test = default_specs(4, "test", 200, 60.)
    assert not ({s.geometry_seed for s in train} & {s.geometry_seed for s in test})
    assert not ({s.object_id for s in train} & {s.object_id for s in test})
    assert [s.knee_deg for s in train] == [0.] * 4
    assert [s.knee_deg for s in test] == [60.] * 4


def test_evidence_content_hash_ignores_dict_insertion_order():
    assert content_hash({"near": .1, "kind": "old"}) == content_hash({"kind": "old", "near": .1})


def test_unsupported_public_target_does_not_tick_or_reset():
    class World:
        def reference_at(self, progress, mode):
            return np.zeros(3), np.eye(3)
    class Backend:
        ticks = 0
        def state(self):return {"q_meas": [0.] * 8}
        def execute_path(self, *a, **k):
            self.ticks += 1
            raise AssertionError("no candidate must not execute")
    class Planner:
        def candidates(self, *a):return []
    engine = PlannedProgramExecutor.__new__(PlannedProgramExecutor)
    engine.world = World(); engine.backend = Backend(); engine.planner = Planner(); engine.transfer_events = []; engine.safe = True
    assert not engine.move(.6, 0)
    assert engine.backend.ticks == 0
    assert engine.transfer_events[-1]["candidate_count"] == 0


def test_safety_stop_prevents_another_global_move():
    engine = PlannedProgramExecutor.__new__(PlannedProgramExecutor)
    engine.safe = False
    assert not engine.move(.6, 0)
