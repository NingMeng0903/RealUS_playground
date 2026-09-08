"""Offline checks for rotation-minimizing phantom tool frames."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.DEMO.phathom_scanning.orientation import (
    frame_diagnostics,
    rotation_minimizing_frames,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    return value / np.linalg.norm(value)


def _body_rotvecs(frames: np.ndarray, initial: np.ndarray) -> np.ndarray:
    previous = np.concatenate((initial[None, :, :], frames[:-1]), axis=0)
    relative = np.einsum("nji,njk->nik", previous, frames)
    return Rotation.from_matrix(relative).as_rotvec()


def _curved_normals() -> np.ndarray:
    # The sequence bends along both in-plane directions, then reverses along
    # the S path.  The endpoint returns to the original normal.
    xy = np.asarray(
        [
            (-0.30, -0.18),
            (-0.16, -0.08),
            (0.00, 0.04),
            (0.19, 0.13),
            (0.31, 0.02),
            (0.17, -0.10),
            (-0.02, -0.15),
            (-0.19, -0.08),
            (-0.30, -0.18),
        ],
        dtype=float,
    )
    vectors = np.column_stack((xy, np.ones(len(xy))))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_constant_plane_preserves_initial_frame_without_spin() -> None:
    initial = Rotation.from_euler("xyz", [0.2, -0.3, 0.7]).as_matrix()
    # Input normals point outward; the probe tool +Z points inward.
    normals = np.repeat((-initial[:, 2])[None, :], 8, axis=0)

    frames = rotation_minimizing_frames(normals, initial)

    np.testing.assert_allclose(frames, np.repeat(initial[None], 8, axis=0), atol=1.0e-12)
    diagnostic = frame_diagnostics(frames, initial)
    assert diagnostic["max_step_rotation_deg"] < 1.0e-10
    assert diagnostic["max_tool_z_spin_deg"] < 1.0e-10
    json.dumps(diagnostic)


def test_curved_two_direction_s_path_has_continuous_z_alignment() -> None:
    initial = Rotation.from_euler("xyz", [-0.4, 0.25, -1.1]).as_matrix()
    normals = _curved_normals()
    frames = rotation_minimizing_frames(normals, initial)

    np.testing.assert_allclose(frames[:, :, 2], -normals, atol=1.0e-12)
    relative = _body_rotvecs(frames, initial)
    assert np.max(np.abs(relative[:, 2])) < 1.0e-10
    assert np.max(np.linalg.norm(relative, axis=1)) < np.pi
    # The return and the S-direction reversal do not create a yaw flip.
    assert np.max(np.linalg.norm(np.diff(frames, axis=0), axis=(1, 2))) < 0.5


def test_initial_tool_z_spin_is_retained_through_minimal_tilt() -> None:
    outward = _unit(np.array([0.25, -0.15, 0.956]))
    base = Rotation.from_euler("xyz", [0.4, -0.7, 0.2]).as_matrix()
    spin = Rotation.from_rotvec(np.array([0.0, 0.0, 1.37])).as_matrix()
    spun = base @ spin
    normals = np.repeat(outward[None, :], 5, axis=0)

    base_frames = rotation_minimizing_frames(normals, base)
    spun_frames = rotation_minimizing_frames(normals, spun)

    expected = base_frames[0] @ spin
    np.testing.assert_allclose(spun_frames[0], expected, atol=1.0e-12)
    np.testing.assert_allclose(spun_frames[-1], base_frames[-1] @ spin, atol=1.0e-12)
    np.testing.assert_allclose(spun_frames[:, :, 2], -normals, atol=1.0e-12)


def test_outward_normal_does_not_flip_already_inward_tool() -> None:
    # A common contact setup starts with tool +Z already pointing into the
    # top surface.  Aligning to -outward must leave the measured frame alone.
    initial = Rotation.from_euler("x", np.pi).as_matrix()
    outward = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), 6, axis=0)

    frames = rotation_minimizing_frames(outward, initial)

    np.testing.assert_allclose(frames, np.repeat(initial[None], 6, axis=0), atol=1.0e-12)
    diagnostic = frame_diagnostics(frames, initial)
    assert diagnostic["max_step_rotation_deg"] < 1.0e-10
    assert diagnostic["max_tool_z_spin_deg"] < 1.0e-10


def test_antiparallel_normals_use_deterministic_old_tool_x_flip() -> None:
    initial = np.eye(3)
    # Choose the outward signs so that the first frame already points into
    # the surface; the second frame then exercises the old-tool-X flip.
    normals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    frames_a = rotation_minimizing_frames(normals, initial)
    frames_b = rotation_minimizing_frames(normals, initial)
    np.testing.assert_allclose(frames_a, frames_b, atol=1.0e-12)
    np.testing.assert_allclose(frames_a[1], np.diag([1.0, -1.0, -1.0]), atol=1.0e-12)
    np.testing.assert_allclose(frames_a[3], initial, atol=1.0e-12)
    assert np.max(np.abs(_body_rotvecs(frames_a, initial)[:, 2])) < 1.0e-10


@pytest.mark.parametrize(
    "normals, initial, message",
    [
        (np.zeros((3, 2)), np.eye(3), "shape"),
        (np.asarray([[0.0, 0.0, 0.0]]), np.eye(3), "non-zero"),
        (np.asarray([[0.0, 0.0, 2.0]]), np.eye(3), "unit"),
        (np.asarray([[np.nan, 0.0, 1.0]]), np.eye(3), "finite"),
        (np.asarray([[0.0, 0.0, 1.0]]), np.diag([1.0, 1.0, -1.0]), "proper"),
    ],
)
def test_inputs_are_validated(normals, initial, message) -> None:
    with pytest.raises(ValueError, match=message):
        rotation_minimizing_frames(normals, initial)


def test_diagnostics_include_initial_and_adjacent_steps() -> None:
    initial = np.eye(3)
    normals = np.asarray([[0.0, 0.0, -1.0], [-0.1, 0.0, -1.0], [0.0, 0.0, -1.0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    frames = rotation_minimizing_frames(normals, initial)

    diagnostic = frame_diagnostics(frames, initial)
    assert diagnostic["n_frames"] == len(normals)
    assert len(diagnostic["step_rotation_deg"]) == len(normals)
    assert len(diagnostic["tool_z_spin_deg"]) == len(normals)
    assert diagnostic["initial_step_rotation_deg"] == pytest.approx(0.0)
    assert diagnostic["max_tool_z_spin_deg"] < 1.0e-10
    json.dumps(diagnostic)
