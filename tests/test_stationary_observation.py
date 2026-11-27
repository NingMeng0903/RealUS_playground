from __future__ import annotations

import numpy as np
import pytest

from perception.benchmark.stationary_observation import StationaryObservationGate


def _points() -> np.ndarray:
    points = np.zeros((67, 4), dtype=np.float64)
    points[:, :3] = np.arange(67 * 3, dtype=np.float64).reshape(67, 3) * 0.01
    points[:, 3] = 0.8
    return points


def _rotations() -> tuple[np.ndarray, np.ndarray]:
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 55, axis=0)
    agreement = np.zeros(55, dtype=np.float64)
    agreement[1:22] = 0.9
    agreement[25:] = 0.9
    return rotations, agreement


def _z_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_exact_and_small_noise_hold_against_last_anchor() -> None:
    base = _points()
    gate = StationaryObservationGate()
    assert not gate.matches(base)
    assert gate.commit(base)
    noisy = base.copy()
    noisy[:, :3] += 0.0002
    noisy[:, 3] += 0.02
    assert gate.matches(noisy)


def test_slow_motion_accumulates_from_anchor_without_moving_anchor() -> None:
    base = _points()
    gate = StationaryObservationGate()
    assert gate.commit(base)
    first = base.copy()
    first[:, 0] += 0.001
    assert gate.matches(first)
    second = base.copy()
    second[:, 0] += 0.0021
    assert not gate.matches(second)
    # The accepted anchor is still the original frame after the held match.
    assert gate.matches(base)


def test_single_limb_point_over_position_limit_releases() -> None:
    base = _points()
    gate = StationaryObservationGate()
    assert gate.commit(base)
    moved = base.copy()
    moved[40, 1] += 0.00201
    assert not gate.matches(moved)


def test_rgb_axial_rotation_releases_with_identical_positions() -> None:
    base = _points()
    rotations, agreement = _rotations()
    gate = StationaryObservationGate()
    assert gate.commit(base, rgb_rotations=rotations, rgb_agreement=agreement)
    one_degree = rotations.copy()
    one_degree[20] = _z_rotation(1.0)
    assert gate.matches(base, rgb_rotations=one_degree, rgb_agreement=agreement)
    three_degrees = rotations.copy()
    three_degrees[20] = _z_rotation(3.0)
    assert not gate.matches(base, rgb_rotations=three_degrees, rgb_agreement=agreement)


def test_root_and_face_rgb_agreement_are_ignored() -> None:
    base = _points()
    rotations, agreement = _rotations()
    gate = StationaryObservationGate()
    assert gate.commit(base, rgb_rotations=rotations, rgb_agreement=agreement)
    changed = agreement.copy()
    changed[[0, 22, 23, 24]] = 1.0
    changed[0] = 0.0
    assert gate.matches(base, rgb_rotations=rotations, rgb_agreement=changed)


def test_visibility_confidence_and_optional_presence_changes_release() -> None:
    base = _points()
    gate = StationaryObservationGate()
    assert gate.commit(base)
    missing = base.copy()
    missing[30, 3] = 0.0
    assert not gate.matches(missing)
    confidence = base.copy()
    confidence[30, 3] += 0.051
    assert not gate.matches(confidence)

    rotations, agreement = _rotations()
    rgb_gate = StationaryObservationGate()
    assert rgb_gate.commit(base, rgb_rotations=rotations, rgb_agreement=agreement)
    assert not rgb_gate.matches(base)
    assert not rgb_gate.matches(base, rgb_rotations=rotations)

    pelvis_gate = StationaryObservationGate()
    assert pelvis_gate.commit(base, rgb_pelvis_world_m=np.zeros(3))
    assert not pelvis_gate.matches(base)
    moved_pelvis = np.array([0.0021, 0.0, 0.0])
    assert not pelvis_gate.matches(base, rgb_pelvis_world_m=moved_pelvis)


def test_agreement_mask_and_weight_changes_release() -> None:
    base = _points()
    rotations, agreement = _rotations()
    gate = StationaryObservationGate()
    assert gate.commit(base, rgb_rotations=rotations, rgb_agreement=agreement)
    mask_change = agreement.copy()
    mask_change[20] = 0.0
    assert not gate.matches(base, rgb_rotations=rotations, rgb_agreement=mask_change)
    weight_change = agreement.copy()
    weight_change[20] += 0.051
    assert not gate.matches(base, rgb_rotations=rotations, rgb_agreement=weight_change)


def test_commit_and_snapshot_copy_arrays_and_matches_is_pure() -> None:
    base = _points()
    gate = StationaryObservationGate()
    assert gate.commit(base)
    snapshot = gate.snapshot()
    assert snapshot is not None
    assert not snapshot.points.flags.writeable
    with pytest.raises(ValueError):
        snapshot.points[0, 0] = 99.0
    base[0, 0] += 0.5
    assert gate.matches(_points())
    held = _points()
    held[1, 1] += 0.001
    assert gate.matches(held)
    assert gate.matches(_points())

    restored = StationaryObservationGate()
    restored.restore(snapshot)
    assert restored.matches(_points())
    restored.restore(None)
    assert not restored.matches(_points())


def test_weak_body25_does_not_commit_or_match() -> None:
    base = _points()
    base[:14, 3] = 0.0
    gate = StationaryObservationGate()
    assert not gate.commit(base)
    assert not gate.matches(base)


@pytest.mark.parametrize("kwargs", [
    {"position_tolerance_m": -1.0},
    {"position_tolerance_m": np.nan},
    {"rotation_tolerance_deg": -1.0},
    {"rotation_tolerance_deg": np.inf},
    {"min_confidence": -0.1},
    {"min_confidence": np.nan},
])
def test_bad_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StationaryObservationGate(**kwargs)


def test_bad_shapes_and_non_so3_rgb_are_rejected() -> None:
    gate = StationaryObservationGate()
    with pytest.raises(ValueError):
        gate.commit(np.zeros((66, 4)))
    with pytest.raises(ValueError):
        gate.commit(_points(), rgb_rotations=np.zeros((55, 3, 3)))
    with pytest.raises(ValueError):
        gate.commit(_points(), rgb_agreement=np.zeros(54))
    rotations, agreement = _rotations()
    rotations[20, 0, 0] = 2.0
    with pytest.raises(ValueError):
        gate.commit(_points(), rgb_rotations=rotations, rgb_agreement=agreement)
