from types import SimpleNamespace

import numpy as np
import pytest

from work.leg_program_bench import run_core_suite


class _FixedPosteriorModel:
    """Two-bundle fixture that makes nominal and robust choices disagree."""

    def __init__(self):
        self.rows = [
            {
                "object_id": "bundle-0",
                "stratum": "fixture",
                "near": -1.0,
                "probe": -1.0,
                "programs": {"fixture": {}},
            },
            {
                "object_id": "bundle-1",
                "stratum": "fixture",
                "near": 1.0,
                "probe": 1.0,
                "programs": {"fixture": {}},
            },
            # The dispatch test supplies the posterior explicitly, so these
            # rows intentionally do not expose any hidden generator labels.
        ]

    def weights(self, evidence):
        del evidence
        return np.asarray([0.8, 0.2], dtype=float)

    def candidate_weights(self, evidence, *, n_models, seed):
        del evidence, n_models, seed
        return SimpleNamespace(
            scenarios=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        )


def _run_policy(monkeypatch, method):
    def fixed_predict_tree(
        world,
        bundle,
        tree,
        *,
        release,
        hidden,
        chain,
        independent_rows=None,
        time_rows=None,
        rng=None,
    ):
        del world, release, hidden, chain, independent_rows, time_rows, rng
        winner = {
            "fixed_minus": "bundle-0",
            "fixed_plus": "bundle-1",
        }.get(tree.name)
        return {
            "success": winner == bundle["object_id"],
            "duration": 1.0,
            "program_calls": 1,
            "new_frames": 1,
        }

    monkeypatch.setattr(run_core_suite, "predict_tree", fixed_predict_tree)
    return run_core_suite.policy_predictions(
        world=SimpleNamespace(),
        model=_FixedPosteriorModel(),
        method=method,
        near=0.0,
        arm={"release": 0.0, "hidden": False},
        chain=("a", "f"),
        deadlines=np.asarray([1.0, 2.0]),
        seed=17,
        n_models=2,
    )


def test_joint_nominal_is_not_accidentally_dispatched_as_joint_robust(monkeypatch):
    nominal = _run_policy(monkeypatch, "joint_nominal")
    robust = _run_policy(monkeypatch, "joint_robust")
    finite_data = _run_policy(monkeypatch, "joint_n8")

    # The nominal posterior prefers bundle-0 with probability .8, so the
    # model-specific choice must be the first fixed tree at every deadline.
    for mixture in nominal["weights"]:
        assert mixture[0] == pytest.approx(1.0)
        assert mixture[1] == pytest.approx(0.0)
        assert sum(mixture) == pytest.approx(1.0)

    # The two candidate object scenarios make the fixed trees complementary;
    # max-min planning therefore has to mix them equally.  The finite-data
    # joint_n8 alias is also a robust dispatch, not the nominal branch.
    for result in (robust, finite_data):
        for mixture in result["weights"]:
            assert mixture[0] == pytest.approx(0.5)
            assert mixture[1] == pytest.approx(0.5)
            assert sum(mixture) == pytest.approx(1.0)

