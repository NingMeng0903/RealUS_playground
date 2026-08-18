"""Regression tests for the conservative collision broadphase."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


URDF = (
    Path(__file__).resolve().parents[1]
    / "rm75_control"
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.collision.urdf"
)


def _collision() -> tuple[RobotKinematics, CollisionModel]:
    kin = RobotKinematics(URDF)
    return kin, CollisionModel(kin.model)


def test_broadphase_contains_every_pair_inside_activation_band() -> None:
    kin, collision = _collision()
    q = np.zeros(kin.nq, dtype=float)
    threshold = 0.05

    collision.update(q)
    full = np.array(
        [collision.geom_data.distanceResults[i].min_distance for i in collision.exact_pair_indices],
        dtype=float,
    )
    collision.update(q, distance_threshold=threshold)

    active_full = {
        i
        for i, distance in enumerate(full)
        if np.isfinite(distance) and distance <= threshold
    }
    assert active_full.issubset(set(collision.exact_pair_indices))
    for i in active_full:
        assert collision.geom_data.distanceResults[i].min_distance == full[i]
    assert collision.distance_query_count <= len(full)


def test_skipped_distance_result_cannot_be_reused_as_active_or_closest() -> None:
    kin, collision = _collision()
    q = np.zeros(kin.nq, dtype=float)

    # Populate every result first, then run a broadphase pass which evaluates
    # only the pair needed for nearest-pair telemetry.  A stale narrow-phase
    # result from the first pass must not leak into active_pairs/closest_pair.
    collision.update(q)
    previous = {
        i: float(collision.geom_data.distanceResults[i].min_distance)
        for i in collision.exact_pair_indices
    }
    collision.update(q, distance_threshold=0.0)
    assert collision.skipped_pair_indices
    assert set(collision.exact_pair_indices).isdisjoint(collision.skipped_pair_indices)

    active = collision.active_pairs(0.10)
    assert all(p.pair_index in collision.exact_pair_indices for p in active)
    closest = collision.closest_pair()
    assert closest is None or closest.pair_index in collision.exact_pair_indices
    # Ensure the test actually exercised a previously populated skipped slot.
    assert any(i in previous for i in collision.skipped_pair_indices)


def test_cbf_build_uses_activation_broadphase() -> None:
    from rm75_control.control.joint_admittance_8dof.collision_model import (
        CollisionConfig,
    )
    from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
        build_cbf_rows,
    )

    kin, collision = _collision()
    q = np.zeros(kin.nq, dtype=float)
    collision.update(q)
    n_full = collision.distance_query_count
    build_cbf_rows(collision, kin, q, CollisionConfig(enabled=True, d_activate=0.04))
    assert collision.distance_query_count < n_full


def test_default_update_remains_full_narrow_phase() -> None:
    kin, collision = _collision()
    collision.update(np.zeros(kin.nq, dtype=float))
    assert collision.distance_query_count == len(collision.geom_model.collisionPairs)
    assert not collision.skipped_pair_indices
    assert len(collision.all_pairs()) == len(collision.geom_model.collisionPairs)


def test_nonunit_mesh_scale_disables_broadphase_for_safety() -> None:
    geometry = SimpleNamespace(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
    )
    sphere = CollisionModel._make_bounding_sphere(
        geometry, mesh_scale=np.array([0.001, 0.001, 0.001])
    )
    assert np.isinf(sphere.radius)
