"""Regression tests for the conservative collision broadphase."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_default_update_remains_full_narrow_phase() -> None:
    kin, collision = _collision()
    collision.update(np.zeros(kin.nq, dtype=float))
    assert collision.distance_query_count == len(collision.geom_model.collisionPairs)
    assert not collision.skipped_pair_indices
    assert len(collision.all_pairs()) == len(collision.geom_model.collisionPairs)


def test_vectorized_lower_bound_matches_scalar() -> None:
    kin, collision = _collision()
    q = np.zeros(kin.nq, dtype=float)
    collision.update(q)
    vec = collision._pair_lower_bounds()
    assert vec.shape == (len(collision.geom_model.collisionPairs),)
    for i, pair in enumerate(collision.geom_model.collisionPairs):
        sphere_a = collision.bounding_spheres[int(pair.first)]
        sphere_b = collision.bounding_spheres[int(pair.second)]
        if not np.isfinite(sphere_a.radius) or not np.isfinite(sphere_b.radius):
            expected = -float("inf")
        else:
            t_a = collision.geom_data.oMg[int(pair.first)]
            t_b = collision.geom_data.oMg[int(pair.second)]
            center_a = np.asarray(t_a.translation, dtype=float) + np.asarray(
                t_a.rotation, dtype=float
            ) @ sphere_a.center
            center_b = np.asarray(t_b.translation, dtype=float) + np.asarray(
                t_b.rotation, dtype=float
            ) @ sphere_b.center
            expected = float(
                np.linalg.norm(center_a - center_b) - sphere_a.radius - sphere_b.radius
            )
        assert vec[i] == pytest.approx(expected, rel=0.0, abs=1.0e-12)


def test_cbf_broadphase_matches_full_narrow_phase_slots() -> None:
    from rm75_control.control.joint_admittance_8dof.collision_model import (
        CollisionConfig,
    )
    from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
        CbfSlotTracker,
        build_cbf_rows,
    )

    kin, collision_bp = _collision()
    _, collision_full = _collision()
    cfg = CollisionConfig(enabled=True, d_safe=0.01, d_activate=0.04, gamma=5.0)
    tracker_bp = CbfSlotTracker(max_pairs=cfg.max_pairs)
    tracker_full = CbfSlotTracker(max_pairs=cfg.max_pairs)
    poses = [
        np.zeros(kin.nq, dtype=float),
        np.array([0.40, 0.1, -0.5, 0.2, 1.0, -0.1, 0.8, 0.0]),
        np.array([0.20, 0.4, -1.0, 0.5, 1.4, 0.3, 0.2, -0.2]),
    ]
    for q in poses:
        kin.jacobian(q)
        rows_bp = build_cbf_rows(
            collision_bp,
            kin,
            q,
            cfg,
            tracker=tracker_bp,
            kinematics_ready=True,
        )
        collision_full.update(q)
        d_keep = cfg.d_activate + tracker_full.hyst_m
        raw = collision_full.active_pairs(d_keep)
        slotted = tracker_full.update(raw, cfg.d_activate)
        names_full = tuple(
            f"self_collision:{p.name_a}:{p.name_b}" for p in slotted if p is not None
        )
        assert rows_bp.names == names_full
        in_band_full = sorted(
            p.distance
            for p in collision_full.all_pairs()
            if np.isfinite(p.distance) and p.distance < d_keep
        )
        in_band_bp = sorted(
            p.distance
            for p in collision_bp.all_pairs()
            if np.isfinite(p.distance) and p.distance < d_keep
        )
        assert in_band_bp == pytest.approx(in_band_full, rel=0.0, abs=1.0e-9)
        assert collision_bp.distance_query_count <= collision_full.distance_query_count
    geometry = SimpleNamespace(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
    )
    sphere = CollisionModel._make_bounding_sphere(
        geometry, mesh_scale=np.array([0.001, 0.001, 0.001])
    )
    assert np.isinf(sphere.radius)


def test_broadphase_collision_update_fits_inner_budget() -> None:
    import time

    kin, collision = _collision()
    q = np.array([0.40, 0.1, -0.5, 0.2, 1.0, -0.1, 0.8, 0.0])
    collision.update(q, distance_threshold=0.05)
    samples = []
    for _ in range(40):
        t0 = time.perf_counter()
        collision.update(q, distance_threshold=0.05)
        samples.append((time.perf_counter() - t0) * 1000.0)
    p50 = float(np.median(samples))
    p95 = float(np.percentile(samples, 95))
    assert p50 <= 3.0
    assert p95 <= 5.0


def test_inner_tick_median_fits_200hz_budget(request) -> None:
    import time
    import uuid

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
    cfg = build_joint_ik_config(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
    cfg.native_shm_prefix = f"collision_timing_{uuid.uuid4().hex}"
    inner = JointIkController(RobotKinematics(), cfg)
    if inner._native is not None:
        request.addfinalizer(inner._native.shutdown)
    q = np.array([0.40, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
    inner.reset(q)
    twist = np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0])
    for _ in range(8):
        inner.update(twist, q_meas=inner.q_cmd, vel_ff=twist,
                     rail_exec_vel_m_s=float(inner.core.qdot_prev[0]))
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        inner.update(twist, q_meas=inner.q_cmd, vel_ff=twist,
                     rail_exec_vel_m_s=float(inner.core.qdot_prev[0]))
        samples.append((time.perf_counter() - t0) * 1000.0)
    p50 = float(np.median(samples))
    p95 = float(np.percentile(samples, 95))
    assert p50 <= 5.2
    assert p95 <= 6.5
