import numpy as np
import pytest

from work.leg_program_bench.learning import (
    common_policy_mixture,
    fit_evidence_pairs,
    fit_joint,
)


def _record(
    x,
    *,
    success,
    duration,
    error,
    status,
    program_id="direct",
    **extra,
):
    return {
        "x": x,
        "success": success,
        "duration": duration,
        "end_error_m": error,
        "status": status,
        "program_id": program_id,
        **extra,
    }


def test_joint_sampler_keeps_deadline_and_success_correlation():
    # Both models have the same success and short-duration marginals. Their
    # joint events differ: in model A success is fast; in model B success is
    # slow. Sampling complete records must preserve that distinction.
    records_a = [
        _record(0.0, success=True, duration=1.0, error=0.1, status="ok"),
        _record(0.0, success=False, duration=3.0, error=0.8, status="timeout"),
    ]
    records_b = [
        _record(0.0, success=True, duration=3.0, error=0.8, status="timeout"),
        _record(0.0, success=False, duration=1.0, error=0.1, status="ok"),
    ]
    context = {"x": 0.0, "program_id": "direct"}
    model_a = fit_joint(records_a, ["x"], k=2)
    model_b = fit_joint(records_b, ["x"], k=2)
    samples_a = model_a.sample(context, np.random.default_rng(11), n=6000)
    samples_b = model_b.sample(context, np.random.default_rng(11), n=6000)

    def deadline_success(samples):
        return np.mean([row["success"] and row["duration"] <= 2.0 for row in samples])

    assert abs(np.mean([row["success"] for row in samples_a]) - 0.5) < 0.04
    assert abs(np.mean([row["success"] for row in samples_b]) - 0.5) < 0.04
    assert abs(np.mean([row["duration"] <= 2.0 for row in samples_a]) - 0.5) < 0.04
    assert abs(np.mean([row["duration"] <= 2.0 for row in samples_b]) - 0.5) < 0.04
    assert deadline_success(samples_a) > 0.45
    assert deadline_success(samples_b) < 0.05


def test_common_policy_mixture_is_shared_across_opposite_models():
    mixture, lower, upper = common_policy_mixture(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(mixture, [0.5, 0.5], atol=1e-8)
    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(1.0)


def test_explicit_feature_keys_reject_missing_and_ignore_hidden_fields():
    records = [
        _record(0.0, success=True, duration=1.0, error=0.1, status="ok", U=999.0),
        _record(1.0, success=False, duration=2.0, error=0.6, status="lost", U=-999.0),
    ]
    model = fit_joint(records, ["x"], k=1)
    context = {"x": 0.0, "program_id": "direct", "U": -1e12, "V": 1e12}
    assert model.predict_success(context) == pytest.approx(1.0)
    sampled = model.sample(context, np.random.default_rng(3), n=1)[0]
    assert "U" not in sampled
    assert "V" not in sampled
    with pytest.raises(ValueError, match="missing requested feature"):
        fit_joint(records, ["x", "missing_feature"])
    with pytest.raises(ValueError, match="context is missing requested"):
        model.predict_success({"program_id": "direct"})


def test_duration_ratio_rescales_to_physical_context_nominal_duration():
    model = fit_joint(
        [
            _record(
                0.0,
                success=True,
                duration=5.0,
                error=0.1,
                status="ok",
                nominal_duration=10.0,
            )
        ],
        ["x"],
        k=1,
    )
    sample = model.sample(
        {"x": 0.0, "program_id": "direct", "nominal_duration": 20.0},
        np.random.default_rng(0),
    )[0]
    assert sample["duration"] == pytest.approx(10.0)
    assert sample["nominal_duration"] == pytest.approx(20.0)


def test_evidence_pairs_use_laplace_smoothing_and_ignore_extra_fields():
    table = fit_evidence_pairs(
        [
            {"y": "ambiguous", "q": "probe", "hidden": "u"},
            {"y": "ambiguous", "q": "probe", "hidden": "v"},
            {"y": "ambiguous", "q": "direct", "hidden": "w"},
        ],
        alpha=1.0,
    )
    assert table["ambiguous"]["probe"] == pytest.approx(3.0 / 5.0)
    assert table["ambiguous"]["direct"] == pytest.approx(2.0 / 5.0)
    assert table.probability("unseen", "probe") == pytest.approx(0.5)
    assert table.probability("ambiguous", "hidden") == pytest.approx(0.0)


def test_explicit_extra_outcomes_keep_physical_exit_joint_and_copy_values():
    exit_a = np.asarray([1.0, 2.0, 3.0])
    exit_b = [4.0, 5.0, 6.0]
    records = [
        _record(
            0.0,
            success=True,
            duration=1.0,
            error=0.1,
            status="ok",
            exit_position_local=exit_a,
            exit_rotation_local=[0.0, 0.0, 1.0],
            exit_velocity_local=np.asarray([0.1, 0.2, 0.3]),
            diagnostic_magnitude=np.float64(0.25),
        ),
        _record(
            0.0,
            success=False,
            duration=3.0,
            error=0.8,
            status="lost",
            exit_position_local=exit_b,
            exit_rotation_local=[0.0, 1.0, 0.0],
            exit_velocity_local=[0.4, 0.5, 0.6],
            diagnostic_magnitude=1.25,
        ),
    ]
    extra = (
        "exit_position_local",
        "exit_rotation_local",
        "exit_velocity_local",
        "diagnostic_magnitude",
    )
    model = fit_joint(records, ["x"], k=2, extra_outcome_keys=extra)
    samples = model.sample({"x": 0.0, "program_id": "direct"}, np.random.default_rng(7), n=40)

    expected = {
        (
            True,
            1.0,
            "ok",
            (1.0, 2.0, 3.0),
        ),
        (
            False,
            3.0,
            "lost",
            (4.0, 5.0, 6.0),
        ),
    }
    for sample in samples:
        key = (
            sample["success"],
            sample["duration"],
            sample["status"],
            tuple(sample["exit_position_local"]),
        )
        assert key in expected
        assert len(sample["exit_rotation_local"]) == 3
        assert len(sample["exit_velocity_local"]) == 3
        assert isinstance(sample["diagnostic_magnitude"], float)

    first = samples[0]
    first["exit_position_local"][0] = 999.0
    first["exit_rotation_local"].append(999.0)
    assert exit_a.tolist() == [1.0, 2.0, 3.0]
    assert exit_b == [4.0, 5.0, 6.0]
    second = model.sample({"x": 0.0, "program_id": "direct"}, np.random.default_rng(0))[0]
    assert len(second["exit_rotation_local"]) == 3
