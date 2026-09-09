"""Bounded contract tests for the coupled V17 leg response.

These tests exercise the representation layer only.  They use a tiny synthetic
field for runtime behavior, while the sample-domain checks cover the fixed
offline protocol.  No capture, SMPL-X model, or fitting job is needed.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.coupled_leg_articulation_v17 import (
    LEGACY_LOWER_DEG,
    LEGACY_UPPER_DEG,
    LOWER_DEG,
    SCALE_DEG,
    UPPER_DEG,
    CoupledLegArticulationV17,
    _features,
    _kernel,
    _source_supported,
    fixed_coupled_leg_samples_v17,
    interpolate_coupled_leg_values_v17,
)


def _pose55(*, left=None, right=None):
    """Make a pose from left/right hip-knee-ankle XYZ degree triplets."""
    pose = np.zeros((55, 3), dtype=np.float64)
    for side, values in ((0, left), (1, right)):
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64).reshape(3, 3)
        joints = (1, 4, 7) if side == 0 else (2, 5, 8)
        pose[list(joints)] = np.deg2rad(values)
    return pose


def _synthetic_articulation(*, width=1.0, with_kernel=False):
    """Build a smooth, small response with an explicit ankle coupling term."""
    centers = np.zeros((10, 9), dtype=np.float64)
    coefficients = np.zeros((10, 12), dtype=np.float64)
    if with_kernel:
        coefficients[:] = np.linspace(-2.0e-5, 2.0e-5, coefficients.size).reshape(
            coefficients.shape
        )
    affine = np.zeros((10, 12), dtype=np.float64)
    # [constant, hip xyz, knee xyz, ankle xyz].  The left ankle-x feature
    # changes the left hip-x correction, making the coupling observable.
    affine[7, 0] = 1.0e-2
    neutral_offset = _kernel(np.zeros((1, 9)), centers, width)[0] @ coefficients + affine[0]
    return CoupledLegArticulationV17(
        centers=centers,
        coefficients=coefficients,
        affine=affine,
        neutral_offset=neutral_offset,
        head_centers_local=np.zeros((2, 3), dtype=np.float64),
        report={"method": "synthetic-test", "anatomical_passed": False},
        kernel_inverse_width=width,
    )


def test_feature_order_keeps_left_and_right_hip_knee_ankle_xyz_groups():
    left = np.arange(1.0, 10.0).reshape(3, 3)
    right = np.arange(10.0, 19.0).reshape(3, 3)
    actual = _features(_pose55(left=left, right=right))
    np.testing.assert_allclose(actual[0], left.reshape(-1), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(actual[1], right.reshape(-1), atol=1e-12, rtol=0.0)


def test_fixed_samples_are_deterministic_subject_independent_and_in_domain():
    rows_a, rejected_a = fixed_coupled_leg_samples_v17()
    rows_b, rejected_b = fixed_coupled_leg_samples_v17()

    assert len(inspect.signature(fixed_coupled_leg_samples_v17).parameters) == 0
    assert rejected_a == rejected_b
    np.testing.assert_array_equal(rows_a, rows_b)
    assert rows_a.ndim == 2 and rows_a.shape[1] == 9
    assert np.isfinite(rows_a).all()
    assert np.unique(rows_a, axis=0).shape[0] == len(rows_a)
    assert np.array_equal(rows_a[0], np.zeros(9))
    assert np.all(rows_a >= LOWER_DEG - 1e-12)
    assert np.all(rows_a <= UPPER_DEG + 1e-12)
    assert all(_source_supported(row) for row in rows_a)
    assert np.all(np.linalg.norm(rows_a[:, 3:6], axis=1) <= 130.0 + 1e-8)
    assert np.all(np.linalg.norm(rows_a[:, 6:9], axis=1) <= 75.0 + 1e-8)
    # The expanded fixed protocol contains the new hip +60 and knee -30
    # samples; these are actual training centers, not a relaxed evaluator
    # bound.
    assert np.any(np.isclose(rows_a[:, 0], 60.0))
    assert np.any(np.isclose(rows_a[:, 3], -30.0))


def test_synthetic_field_save_load_is_exact_and_preserves_evaluation(tmp_path):
    articulation = _synthetic_articulation()
    archive = tmp_path / "coupled_leg_articulation.npz"
    articulation.save(archive)
    loaded = CoupledLegArticulationV17.load(archive)

    for name in (
        "centers",
        "coefficients",
        "affine",
        "neutral_offset",
        "head_centers_local",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(articulation, name))
    assert loaded.report == articulation.report
    probe = _pose55(left=np.asarray([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(loaded.evaluate(probe), articulation.evaluate(probe))


def test_width_two_archive_round_trip_preserves_kernel_and_evaluator(tmp_path):
    articulation = _synthetic_articulation(width=2.0, with_kernel=True)
    archive = tmp_path / "width_two.npz"
    articulation.save(archive)
    loaded = CoupledLegArticulationV17.load(archive)

    assert loaded.kernel_inverse_width == pytest.approx(2.0)
    probe = _pose55(
        left=np.asarray([[5.0, 0.0, 0.0], [15.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        right=np.asarray([[0.0, 5.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 10.0]]),
    )
    np.testing.assert_array_equal(loaded.evaluate(probe), articulation.evaluate(probe))


def test_legacy_width_one_archive_without_width_field_still_loads(tmp_path):
    articulation = _synthetic_articulation(with_kernel=True)
    complete = tmp_path / "complete.npz"
    legacy = tmp_path / "legacy_width_one.npz"
    articulation.save(complete)
    with np.load(complete, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files if key != "kernel_inverse_width"}
    np.savez_compressed(legacy, **payload)

    loaded = CoupledLegArticulationV17.load(legacy)
    assert loaded.kernel_inverse_width == pytest.approx(1.0)
    probe = _pose55(left=np.asarray([[5.0, 0.0, 0.0], [15.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(loaded.evaluate(probe), articulation.evaluate(probe))


def test_saved_support_report_controls_boundary_and_legacy_falls_back(tmp_path):
    pose_at_new_hip_sample = _pose55(
        left=np.asarray([[45.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )

    # An archive produced before the protocol expansion has no saved support
    # arrays and therefore retains the old hip upper bound of +30 degrees.
    legacy = _synthetic_articulation()
    np.testing.assert_array_equal(LEGACY_LOWER_DEG, np.asarray(legacy.report.get(
        "lower_support_deg", LEGACY_LOWER_DEG
    )))
    np.testing.assert_array_equal(LEGACY_UPPER_DEG, np.asarray(legacy.report.get(
        "upper_support_deg", LEGACY_UPPER_DEG
    )))
    with pytest.raises(ValueError, match="not clipped"):
        legacy.evaluate(pose_at_new_hip_sample)

    # A newly compiled archive records the expanded protocol.  Evaluating it
    # at +45 degrees succeeds without silently clipping the input.
    expanded = _synthetic_articulation()
    expanded.report["lower_support_deg"] = LOWER_DEG.tolist()
    expanded.report["upper_support_deg"] = UPPER_DEG.tolist()
    archive = tmp_path / "expanded_support.npz"
    expanded.save(archive)
    loaded = CoupledLegArticulationV17.load(archive)
    np.testing.assert_array_equal(loaded.report["lower_support_deg"], LOWER_DEG)
    np.testing.assert_array_equal(loaded.report["upper_support_deg"], UPPER_DEG)
    result = loaded.evaluate(pose_at_new_hip_sample)
    assert np.isfinite(result).all()


def test_small_affine_interpolation_is_finite_and_preserves_side_output_order():
    # Neutral plus one basis point for each of the nine normalized features is
    # the smallest full-rank synthetic KKT system.  The affine target makes
    # the expected response independently calculable at a new query.
    features = np.vstack([np.zeros(9), 0.2 * np.eye(9)])
    angles = features * SCALE_DEG
    affine = np.zeros((10, 12), dtype=np.float64)
    affine[1:] = np.arange(9 * 12, dtype=np.float64).reshape(9, 12) * 1.0e-5
    values = np.c_[np.ones(len(features)), features] @ affine
    field = interpolate_coupled_leg_values_v17(
        angles,
        values,
        np.zeros((2, 3), dtype=np.float64),
        {"method": "small-synthetic-interpolation"},
    )

    assert field.kernel_inverse_width == pytest.approx(2.0)
    assert np.isfinite(field.coefficients).all()
    assert field.report["training_reconstruction_max_deg"] < 1.0e-6

    query_features = np.asarray(
        [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    query = np.zeros((55, 3), dtype=np.float64)
    for row, joints in zip(query_features, ((1, 4, 7), (2, 5, 8))):
        query[list(joints)] = np.deg2rad(
            row.reshape(3, 3) * SCALE_DEG.reshape(3, 3)
        )
    raw_expected = np.c_[np.ones(2), query_features] @ affine
    expected = np.stack(
        [raw_expected[0, :6].reshape(2, 3), raw_expected[1, 6:].reshape(2, 3)]
    )
    np.testing.assert_allclose(field.evaluate(query), expected, atol=1.0e-10, rtol=0.0)


def test_repack_recovers_saved_sample_and_left_right_order_by_index(tmp_path, monkeypatch):
    from projects.genesis_ue_sync.anatomy_retarget.cli import (
        repack_coupled_interpolation_v17 as repack,
    )

    sample_count = 10
    centers = np.zeros((sample_count, 9), dtype=np.float64)
    centers[1:, 0] = np.arange(1.0, sample_count)
    records = []
    order = [7, 2, 9, 1, 5, 3, 8, 4, 6]
    for index in order:
        left = (index + np.arange(6) / 10.0).tolist()
        right = (100.0 + index + np.arange(6) / 10.0).tolist()
        records.append(
            {
                "sample": index,
                "fits": [
                    {"side": "L", "angles_deg": left},
                    {"side": "R", "angles_deg": right},
                ],
            }
        )
    old = CoupledLegArticulationV17(
        centers=centers,
        coefficients=np.zeros((sample_count, 12), dtype=np.float64),
        affine=np.zeros((10, 12), dtype=np.float64),
        neutral_offset=np.zeros(12, dtype=np.float64),
        head_centers_local=np.zeros((2, 3), dtype=np.float64),
        report={"fits": records, "method": "saved-old-fit"},
    )

    class FakeSubject:
        def __init__(self, articulation):
            self.leg_articulation = articulation
            self.report = {"base": True}
            self.saved_path = None
            self.pose_calls = []

        def save(self, path):
            self.saved_path = path
            path.mkdir(parents=True, exist_ok=True)

        def apply_pose(self, pose):
            self.pose_calls.append(np.asarray(pose).copy())
            return np.asarray([1.0, 2.0, 3.0])

    subject = FakeSubject(old)
    loaded = FakeSubject(old)
    loaded_calls = []

    def fake_load(path):
        loaded_calls.append(path)
        return subject if len(loaded_calls) == 1 else loaded

    captured = {}
    repacked = _synthetic_articulation(width=2.0)
    repacked.report["training_reconstruction_max_deg"] = 0.0

    def fake_interpolate(angles, values, centers_local, report):
        captured["angles"] = np.asarray(angles).copy()
        captured["values"] = np.asarray(values).copy()
        captured["centers_local"] = np.asarray(centers_local).copy()
        captured["report"] = dict(report)
        return repacked

    monkeypatch.setattr(repack, "load_lower_subject", fake_load)
    monkeypatch.setattr(repack, "interpolate_coupled_leg_values_v17", fake_interpolate)
    monkeypatch.setattr(
        repack,
        "fixed_lower_fit_poses",
        lambda: {"neutral": np.zeros((55, 3), dtype=np.float64)},
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "repacked"
    monkeypatch.setattr(
        sys,
        "argv",
        ["repack_coupled_interpolation_v17", "--compiled", str(compiled), "--output", str(output)],
    )

    repack.main()

    np.testing.assert_array_equal(captured["angles"], centers * SCALE_DEG)
    expected_values = np.zeros((sample_count, 12), dtype=np.float64)
    for record in records:
        expected_values[record["sample"]] = np.deg2rad(
            np.asarray(
                [record["fits"][0]["angles_deg"], record["fits"][1]["angles_deg"]]
            )
        ).reshape(12)
    np.testing.assert_array_equal(captured["values"], expected_values)
    assert [fit["side"] for fit in records[0]["fits"]] == ["L", "R"]
    assert captured["centers_local"].shape == (2, 3)
    assert subject.saved_path == output / "compiled"
    assert len(loaded.pose_calls) == 1
    assert (output / "report.json").exists()


@pytest.mark.parametrize(
    "archive_dir",
    (
        "v17_coupled_baked_213328_20260909_002",
        "v17_coupled_baked_213712_20260909_002",
        "v17_coupled_baked_beta_axis0_plus_20260909_002",
    ),
)
def test_existing_002_fit_cache_records_are_ordered_and_replay_without_clipping(archive_dir):
    archive = (
        Path("/home/camp/anatomy_retarget_outputs")
        / archive_dir
        / "compiled/leg_articulation.npz"
    )
    if not archive.exists():
        pytest.skip("optional local V17 archive is not present")
    field = CoupledLegArticulationV17.load(archive)
    records = field.report["fits"]
    sample_ids = [int(record["sample"]) for record in records]
    assert sample_ids == list(range(1, len(field.centers)))
    assert all([row["side"] for row in record["fits"]] == ["L", "R"] for record in records)

    # This is the exact immutable 9-angle cache key used by incremental
    # extension.  It must identify every saved fit once and preserve the
    # center row it came from.
    keys = [tuple(np.round(field.centers[index] * SCALE_DEG, 8)) for index in sample_ids]
    assert len(set(keys)) == len(keys)
    np.testing.assert_array_equal(field.centers[0], np.zeros(9))

    # Existing _002 archives use the pre-expansion report bounds even though
    # module constants now include +60 hip.  A repeated old-domain evaluation
    # is bit-identical, while +45 is rejected instead of clipped.
    old_pose = _pose55(left=np.asarray([[30.0, 0.0, 0.0], [90.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    reloaded = CoupledLegArticulationV17.load(archive)
    np.testing.assert_array_equal(field.evaluate(old_pose), reloaded.evaluate(old_pose))
    with pytest.raises(ValueError, match="not clipped"):
        field.evaluate(_pose55(left=np.asarray([[45.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])))


def test_neutral_is_zero_and_response_is_continuous_at_neutral():
    articulation = _synthetic_articulation()
    neutral = articulation.evaluate(np.zeros((55, 3), dtype=np.float64))
    np.testing.assert_allclose(neutral, 0.0, atol=1e-15, rtol=0.0)

    plus = articulation.evaluate(_pose55(left=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])))
    minus = articulation.evaluate(_pose55(left=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.01, 0.0, 0.0]])))
    np.testing.assert_allclose(plus, -minus, atol=1e-14, rtol=0.0)
    assert np.linalg.norm(plus - neutral) < 1e-5


def test_ankle_feature_changes_the_coupled_response():
    articulation = _synthetic_articulation()
    pose = _pose55(left=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    result = articulation.evaluate(pose)

    # The fitted correction values are radians, while the RBF/affine input
    # feature is the normalized degree value (10 / 35 for ankle-x).
    assert result[0, 0, 0] == pytest.approx(0.01 * 10.0 / SCALE_DEG[6], abs=1e-14)
    assert np.linalg.norm(result[0]) > 0.0
    np.testing.assert_array_equal(result[1], np.zeros((2, 3)))


def test_equivalent_two_pi_axis_angle_has_the_same_coupled_response():
    articulation = _synthetic_articulation()
    pose = _pose55(
        left=np.asarray([[5.0, 0.0, 0.0], [0.0, 15.0, 0.0], [10.0, 0.0, 0.0]]),
        right=np.asarray([[0.0, 0.0, 5.0], [0.0, -10.0, 0.0], [0.0, 0.0, 10.0]]),
    )
    equivalent = pose.copy()
    equivalent[[1, 4, 7]] += 2.0 * np.pi * np.asarray([[1.0, 0.0, 0.0]] * 3)
    equivalent[[2, 5, 8]] += 2.0 * np.pi * np.asarray([[0.0, 1.0, 0.0]] * 3)
    np.testing.assert_allclose(
        articulation.evaluate(equivalent), articulation.evaluate(pose), atol=1e-13, rtol=0.0
    )


def test_nonfinite_shape_and_unsupported_domain_are_rejected_without_clipping():
    articulation = _synthetic_articulation()
    with pytest.raises(ValueError, match="55 finite"):
        articulation.evaluate(np.zeros((54, 3), dtype=np.float64))
    nonfinite = np.zeros((55, 3), dtype=np.float64)
    nonfinite[1, 0] = np.nan
    with pytest.raises(ValueError, match="55 finite"):
        articulation.evaluate(nonfinite)

    # Component bounds are valid here, but the frozen source knee response
    # ball is not: sqrt(100^2 + 100^2) > 130 degrees.
    knee_norm = _pose55(left=np.asarray([[0.0, 0.0, 0.0], [0.0, 100.0, 100.0], [0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="not clipped"):
        articulation.evaluate(knee_norm)
    ankle_norm = _pose55(left=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 60.0, 60.0]]))
    with pytest.raises(ValueError, match="not clipped"):
        articulation.evaluate(ankle_norm)
    with pytest.raises(ValueError, match="not clipped"):
        articulation.evaluate(_pose55(left=np.asarray([[30.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])))

    # Every individual component is supported, but the complete normalized
    # nine-dimensional query lies beyond the fixed nearest-center envelope.
    far = _pose55(left=np.asarray([[30.0, 0.0, 0.0], [120.0, 0.0, 0.0], [60.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="too far"):
        articulation.evaluate(far)
