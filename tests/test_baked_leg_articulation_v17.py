"""Contract tests for the fixed, separable V17 leg-articulation response."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget import generic_lower_compile_v17 as generic
from projects.genesis_ue_sync.anatomy_retarget.baked_leg_articulation_v17 import (
    BakedLegArticulationV17,
)


def _pose(hip=0.0, knee=0.0, *, other=None):
    value = np.zeros((55, 3), dtype=np.float64)
    value[[1, 2], 0] = np.deg2rad(hip)
    value[[4, 5], 0] = np.deg2rad(knee)
    if other is not None:
        other = np.asarray(other, dtype=np.float64).reshape(-1)
        if other.size not in (4, 7):
            raise ValueError("test pose off-axis vector must have four or seven values")
        value[[1, 2], 1:] = np.deg2rad(other[:2])
        value[[4, 5], 1:] = np.deg2rad(other[2:4])
        if other.size == 7:
            value[[7, 8], :] = np.deg2rad(other[4:7])
    return value


def _fixture():
    hip_axis = np.asarray([-30.0, 0.0, 30.0], dtype=np.float64)
    knee_axis = np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    off_axis = np.asarray([4.0, 4.0, 3.0, 3.0], dtype=np.float64)
    values = np.zeros((len(hip_axis), len(knee_axis), 9, 2, 2, 3), dtype=np.float64)

    # The base grid is affine in the two flexion angles, which makes the
    # bilinear interpolation result independently checkable.  Variant offsets
    # are intentionally different for all four axes and both signs.
    for i, hip in enumerate(hip_axis):
        for j, knee in enumerate(knee_axis):
            for side in range(2):
                base_deg = np.asarray(
                    [
                        0.018 * hip + 0.009 * knee + 0.35 * side,
                        -0.011 * hip + 0.004 * knee - 0.20 * side,
                        0.006 * hip - 0.003 * knee + 0.15 * side,
                    ]
                )
                values[i, j, 0, side] = np.deg2rad(base_deg)
                for axis in range(4):
                    for sign_index, sign in enumerate((-1.0, 1.0)):
                        variant = 1 + 2 * axis + sign_index
                        increment_deg = np.asarray(
                            [
                                (0.18 + 0.04 * axis) * sign,
                                (-0.10 + 0.02 * axis) * sign,
                                (0.07 + 0.01 * side) * sign,
                            ]
                        )
                        values[i, j, variant, side] = values[i, j, 0, side] + np.deg2rad(
                            increment_deg
                        )

    report = {
        "method": "separable_bilinear_leg_articulation_v17",
        "simultaneous_off_axis_interactions_fitted": False,
        "anatomical_passed": False,
    }
    articulation = BakedLegArticulationV17(
        hip_axis,
        knee_axis,
        off_axis,
        values,
        np.zeros((2, 3), dtype=np.float64),
        report,
    )
    return articulation, values, hip_axis, knee_axis, off_axis


def _bilinear(values, hip_axis, knee_axis, hip, knee, variant, side):
    i = min(max(int(np.searchsorted(hip_axis, hip, side="right")) - 1, 0), len(hip_axis) - 2)
    j = min(max(int(np.searchsorted(knee_axis, knee, side="right")) - 1, 0), len(knee_axis) - 2)
    u = (hip - hip_axis[i]) / (hip_axis[i + 1] - hip_axis[i])
    v = (knee - knee_axis[j]) / (knee_axis[j + 1] - knee_axis[j])
    return (
        (1 - u) * (1 - v) * values[i, j, variant, side]
        + u * (1 - v) * values[i + 1, j, variant, side]
        + (1 - u) * v * values[i, j + 1, variant, side]
        + u * v * values[i + 1, j + 1, variant, side]
    )


def _expanded_fixture():
    """A seven-axis archive using the new normalized off-axis knots."""
    hip_axis = np.asarray([-30.0, 0.0, 30.0], dtype=np.float64)
    knee_axis = np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    off_axis = np.full(7, 90.0, dtype=np.float64)
    off_knots = np.asarray([-1.0, -0.5, -1 / 6, 0.0, 1 / 6, 0.5, 1.0])
    nonzero = off_knots[off_knots != 0.0]
    values = np.zeros(
        (len(hip_axis), len(knee_axis), 1 + 7 * len(nonzero), 2, 2, 3),
        dtype=np.float64,
    )
    for i, hip in enumerate(hip_axis):
        for j, knee in enumerate(knee_axis):
            base_deg = np.asarray(
                [0.015 * hip + 0.008 * knee, -0.010 * hip + 0.004 * knee,
                 0.005 * hip - 0.002 * knee]
            )
            for side in range(2):
                values[i, j, 0, side] = np.deg2rad(base_deg)
                for axis in range(7):
                    for knot_index, fraction in enumerate(nonzero):
                        variant = 1 + axis * len(nonzero) + knot_index
                        increment = np.deg2rad(
                            fraction * np.asarray(
                                [0.22 + 0.01 * axis, -0.13 + 0.01 * side, 0.08]
                            )
                        )
                        values[i, j, variant, side] = values[i, j, 0, side] + increment
    report = {
        "method": "separable_bilinear_leg_articulation_v17",
        "simultaneous_off_axis_interactions_fitted": False,
        "anatomical_passed": False,
    }
    articulation = BakedLegArticulationV17(
        hip_axis,
        knee_axis,
        off_axis,
        values,
        np.zeros((2, 3), dtype=np.float64),
        report,
        off_knots,
    )
    return articulation, values, hip_axis, knee_axis, off_axis, off_knots


def test_training_flexion_knots_replay_exactly_and_neutral_is_zero():
    articulation, values, hip_axis, knee_axis, _off_axis = _fixture()
    for hip in hip_axis:
        for knee in knee_axis:
            result = articulation.evaluate(_pose(hip, knee))
            expected = np.stack(
                [
                    _bilinear(values, hip_axis, knee_axis, hip, knee, 0, side)
                    for side in range(2)
                ]
            )
            if hip == 0.0 and knee == 0.0:
                np.testing.assert_array_equal(result, np.zeros((2, 2, 3)))
            else:
                np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0.0)

    np.testing.assert_array_equal(articulation.evaluate(_pose()), np.zeros((2, 2, 3)))


@pytest.mark.parametrize("axis,sign", [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1)])
def test_single_off_axis_knot_selects_the_recorded_signed_variant(axis, sign):
    articulation, values, hip_axis, knee_axis, off_axis = _fixture()
    # Use a non-neutral flexion knot so the explicit neutral fast path cannot
    # hide a wrong variant index.
    hip, knee = 30.0, 60.0
    other = np.zeros(4, dtype=np.float64)
    other[axis] = sign * off_axis[axis]
    result = articulation.evaluate(_pose(hip, knee, other=other))
    variant = 1 + 2 * axis + int(sign > 0)
    expected = np.stack(
        [values[-1, 1, variant, side] for side in range(2)]
    )
    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0.0)


def test_mixed_small_axes_are_the_documented_additive_approximation():
    articulation, values, hip_axis, knee_axis, off_axis = _fixture()
    hip, knee = 15.0, 30.0
    other = np.asarray([1.0, -2.0, 0.5, -1.0], dtype=np.float64)
    result = articulation.evaluate(_pose(hip, knee, other=other))
    expected_sides = []
    for side in range(2):
        base = _bilinear(values, hip_axis, knee_axis, hip, knee, 0, side)
        expected = base.copy()
        for axis, magnitude in enumerate(other):
            variant = 1 + 2 * axis + int(magnitude > 0)
            variant_value = _bilinear(values, hip_axis, knee_axis, hip, knee, variant, side)
            expected += (variant_value - base) * abs(magnitude) / off_axis[axis]
        expected_sides.append(expected)
    expected = np.stack(expected_sides)
    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0.0)
    assert articulation.report["simultaneous_off_axis_interactions_fitted"] is False


@pytest.mark.parametrize("axis", range(7))
def test_expanded_off_knots_replay_each_new_middle_node(axis):
    articulation, values, hip_axis, knee_axis, _off_axis, off_knots = _expanded_fixture()
    # +15 degrees is the first positive interior knot (+1/6 of the 90 degree
    # limit), so this checks interpolation over the new knot table rather than
    # the old endpoint-only linear response.
    other = np.zeros(7, dtype=np.float64)
    other[axis] = 15.0
    result = articulation.evaluate(_pose(15.0, 30.0, other=other))
    nonzero = off_knots[off_knots != 0.0]
    knot_index = int(np.flatnonzero(np.isclose(nonzero, 1 / 6))[0])
    variant = 1 + axis * len(nonzero) + knot_index
    expected = np.stack(
        [
            _bilinear(values, hip_axis, knee_axis, 15.0, 30.0, variant, side)
            for side in range(2)
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0.0)


def test_ankle_only_pose_has_a_nonzero_baked_response():
    articulation, _values, _hip_axis, _knee_axis, _off_axis, _off_knots = _expanded_fixture()
    pose = np.zeros((55, 3), dtype=np.float64)
    pose[7, 0] = np.deg2rad(15.0)  # left SMPL-X ankle x axis only
    result = articulation.evaluate(pose)
    assert np.linalg.norm(result[0]) > 0.0
    np.testing.assert_array_equal(result[1], np.zeros((2, 3)))


def test_equivalent_two_pi_axis_angle_is_normalized_before_lookup():
    articulation, _values, _hip_axis, _knee_axis, _off_axis, _off_knots = _expanded_fixture()
    pose = np.zeros((55, 3), dtype=np.float64)
    pose[1, 0] = np.deg2rad(15.0)
    pose[7, 0] = np.deg2rad(15.0)
    equivalent = pose.copy()
    equivalent[1, 0] += 2 * np.pi
    equivalent[7, 0] += 2 * np.pi
    np.testing.assert_allclose(
        articulation.evaluate(equivalent), articulation.evaluate(pose), atol=1e-14, rtol=0.0
    )


def test_expanded_off_knots_save_load_preserves_new_format(tmp_path):
    articulation, _values, _hip_axis, _knee_axis, _off_axis, _off_knots = _expanded_fixture()
    archive = tmp_path / "expanded_leg_articulation.npz"
    articulation.save(archive)
    loaded = BakedLegArticulationV17.load(archive)
    np.testing.assert_array_equal(loaded.off_knots, articulation.off_knots)
    probe = np.zeros((55, 3), dtype=np.float64)
    probe[7, 0] = np.deg2rad(15.0)
    np.testing.assert_array_equal(loaded.evaluate(probe), articulation.evaluate(probe))


def test_out_of_support_angles_raise_without_clipping():
    articulation, _values, _hip_axis, _knee_axis, off_axis = _fixture()
    with pytest.raises(ValueError, match="not clipped"):
        articulation.evaluate(_pose(30.01, 60.0))
    with pytest.raises(ValueError, match="not clipped"):
        articulation.evaluate(_pose(30.0, 60.0, other=[off_axis[0] + 0.01, 0, 0, 0]))


def test_save_load_preserves_grid_and_evaluation(tmp_path):
    articulation, _values, _hip_axis, _knee_axis, _off_axis = _fixture()
    archive = tmp_path / "leg_articulation.npz"
    articulation.save(archive)
    loaded = BakedLegArticulationV17.load(archive)
    for name in ("hip_axis", "knee_axis", "off_axis", "values", "head_centers_local"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(articulation, name))
    assert loaded.report == articulation.report
    probe = _pose(15.0, 30.0, other=[1.0, -2.0, 0.5, -1.0])
    np.testing.assert_array_equal(loaded.evaluate(probe), articulation.evaluate(probe))


class _RuntimeForEnvelope:
    def __init__(self):
        self.betas = np.linspace(-0.1, 0.1, 10)
        self.source_pack = SimpleNamespace(operator_runtime_digest="d" * 64)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text("{}\n", encoding="utf-8")


def _envelope_subject(articulation):
    runtime = _RuntimeForEnvelope()
    report = {
        "target_beta": np.zeros(10).tolist(),
        "source_reference_beta": runtime.betas.tolist(),
        "source_operator_digest": "d" * 64,
    }
    return generic.CompiledLowerSubjectV17(np.zeros(10), runtime, report, articulation)


def test_wrapper_persists_optional_articulation_and_rejects_hash_tampering(tmp_path, monkeypatch):
    articulation, _values, _hip_axis, _knee_axis, _off_axis = _fixture()
    subject = _envelope_subject(articulation)
    output = tmp_path / "compiled"
    subject.save(output)
    monkeypatch.setattr(generic, "load_compiled_subject", lambda _path: _RuntimeForEnvelope())

    loaded = generic.load_lower_subject(output)
    assert loaded.leg_articulation is not None
    np.testing.assert_array_equal(loaded.leg_articulation.values, articulation.values)

    archive = output / "leg_articulation.npz"
    payload = bytearray(archive.read_bytes())
    payload[-1] ^= 1
    archive.write_bytes(payload)
    with pytest.raises(ValueError, match="baked leg articulation hash"):
        generic.load_lower_subject(output)


def test_wrapper_pose_uses_baked_response_without_runtime_fit(monkeypatch):
    articulation, _values, _hip_axis, _knee_axis, _off_axis = _fixture()
    runtime = SimpleNamespace(
        target_inverse=np.eye(4, dtype=np.float64)[None],
        _lbs=lambda transforms: np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
    )
    subject = generic.CompiledLowerSubjectV17(
        np.zeros(10),
        runtime,
        {"target_beta": np.zeros(10).tolist()},
        articulation,
    )
    import projects.genesis_ue_sync.anatomy_retarget.lower_chain_pose_fit_v17 as pose_fit

    monkeypatch.setattr(
        pose_fit,
        "fit_leg_pose_v17",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fit called at runtime")),
    )
    monkeypatch.setattr(
        pose_fit,
        "articulated_leg_globals",
        lambda *_args, **_kwargs: np.eye(4, dtype=np.float64)[None],
    )
    pose = _pose(15.0, 30.0)
    result = subject.apply_pose(pose)
    np.testing.assert_array_equal(result, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
