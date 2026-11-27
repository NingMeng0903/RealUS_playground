"""Asset-free Gate 3 regressions for opportunity accounting and B1 marginals."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import block_diag

from reader import eval_reader as evaluation
from reader.baselines import B1_NoTopology_AbsXYZ
from reader.dynamic_factor_graph import _Association, _IDENTITY_MARGINAL


def _centers(*roles):
    values = np.full((6, 3), np.nan)
    for role in roles:
        values[role] = (role, 0., 0.)
    return values


def _frame(targets, centers):
    return SimpleNamespace(echo_targets=targets, centers_mm=centers, obs=SimpleNamespace(echoes=np.zeros((len(targets), 5))),
                           probe_pose=SimpleNamespace(dwell_s=0.), f_contact=2.5, q_hat=None, Sigma_q=None, dt=.1)


def _estimate(centers, *masks):
    identities = np.eye(64)[np.asarray(masks, dtype=int)]
    return SimpleNamespace(centers_mm=centers, role_probabilities=identities @ _IDENTITY_MARGINAL,
                           identity_probabilities=identities)


class _Reader:
    def __init__(self, online, history=None):
        self.online, self.history = online, history
        self.reset()

    def reset(self):
        self.index = 0
        self.replays = []

    def step(self, *_):
        estimate = self.online[self.index]
        self.index += 1
        return estimate

    def relabel_history(self):
        self.replays.append(self.index)
        return self.history[self.index]


def _risk(full=.05, b3=.15):
    return {"methods": {
        evaluation.FULL: {"acceptance_rate": .5, "conditional_failure": {"rate": full}, "cutoff_unbounded": False,
                          "miscoverage": {"rate": .15}, "false_certification": {"rate": .02}},
        "B3_NoDiscount": {"conditional_failure": {"rate": b3}},
    }}


def test_missing_rendered_center_cannot_pass_the_tracking_gate():
    full = evaluation.FULL
    b2 = "B2_NoPullback"
    cases = [
        # The double-role echo plus the repeated role yields two center opportunities, not three.
        # Full deliberately misidentifies the first slot but its finite role-0 center still counts.
        (_frame(((0, 1), (1,), ()), _centers(0, 1, 2)),
         _estimate(_centers(0, 2), 4, 2, 0), _estimate(_centers(0, 1, 2), 3, 2, 0)),
        (_frame(((1,),), _centers(1)), _estimate(_centers(), 2), _estimate(_centers(1), 2)),
        (_frame((), _centers()), _estimate(_centers()), _estimate(_centers())),
        (_frame(((),), _centers()), _estimate(_centers(), 0), _estimate(_centers(), 0)),
    ]
    cases[0][1].centers_mm[2, 0] += 3.
    for _ in range(8):
        cases.append((_frame(((0,),), _centers(0)), _estimate(_centers(0), 1), _estimate(_centers(0), 1)))
    for _, _, baseline in (cases[0], *cases[4:]):
        baseline.centers_mm[0, 0] += 1.
    tapes = [evaluation.Tape(f"tape:{i}", "synthetic", i) for i in range(len(cases))]
    episodes = [SimpleNamespace(frames=(f,), h_start_mm=-500., provenance={"geometry": {}}) for f, _, _ in cases]
    sim = SimpleNamespace(episode=lambda _source, seed, **_: episodes[seed])

    class TapeReader(_Reader):
        def __init__(self, column):
            self.column = column
            super().__init__(())

        def reset(self):
            super().reset()
            if hasattr(self, "tape_index"):
                self.online = (cases[self.tape_index][self.column],)
                self.tape_index += 1
            else:
                self.tape_index = 0

    readers = {method: TapeReader(1 if method == full else 2) for method in evaluation.METHODS}
    runs, metadata = evaluation._collect(readers, sim, tapes, "dynamic", 1)
    weights = np.ones((3, len(tapes)), dtype=int)
    weights[1] = 0  # A legal resample which omits every tape containing a missing prediction.
    weights[1, -1] = len(tapes)
    weights[2] = 0  # A legal empty resample: no rendered opportunities, not 0 mm.
    weights[2, 2] = len(tapes)
    dynamic, _ = evaluation._benchmark_report(runs, metadata, weights, "dynamic")
    assert set(dynamic["methods"]) == set(evaluation.METHODS)
    assert metadata[0]["rendered_frame_role_opportunities"] == 2
    assert metadata[0]["rendered_lumen_echoes"] == 2
    assert metadata[0]["rendered_echoes_by_contributor_count"] == [1, 1, 1, 0, 0, 0, 0]
    assert dynamic["acquisition"]["empty_echo_tapes"] == 1
    assert dynamic["acquisition"]["empty_lumen_tapes"] == 2
    metrics = dynamic["methods"][full]
    assert metrics["exact_mask_accuracy_by_true_contributor_count"]["2"]["accuracy"]["value"] == 0.
    assert dynamic["methods"][b2]["exact_mask_accuracy_by_true_contributor_count"]["2"]["accuracy"]["value"] == 1.
    assert metrics["tracking_centers_observed"] == 11
    assert metrics["tracking_centers_measured"] == 9
    assert metrics["tracking_centers_unmeasured"] == 2
    assert metrics["tracking_center_coverage"]["value"] == pytest.approx(9 / 11)
    assert metrics["tracking_conditional_finite_rmse_mm"]["value"] == 0.
    assert metrics["tracking_rmse_mm"] == {"value": None, "ci95": None, "defined_bootstrap_draws": 0}
    assert metrics["all_slice_role_conditional_finite_rmse_mm"]["value"] == pytest.approx(np.sqrt(9 / 10))
    assert metrics["all_slice_role_centers_unmeasured"] == 2
    assert dynamic["methods"][b2]["tracking_rmse_mm"]["value"] == pytest.approx(np.sqrt(9 / 11))

    comparison = dynamic["paired_full_minus_baseline"][b2]["common_center_rmse_wilcoxon"]
    assert comparison["p"] < .01
    assert comparison["episodes_total"] == 12
    assert comparison["episodes_without_opportunities"] == 2
    assert comparison["episodes_with_opportunities_but_no_common_prediction"] == 1
    assert comparison["episodes_with_partial_common_predictions"] == 1
    assert comparison["paired_episodes"] == comparison["matched_frame_role_centers"] == 9
    assert comparison["by_tape"][1] == {
        "episode_id": "tape:1", "opportunities": 1, "common": 0, "first_only": 0, "second_only": 1, "neither": 0,
    }

    def benchmark(metrics_by_reader):
        return {"methods": {method: {key: {"value": value} for key, value in entries.items()}
                            for method, entries in metrics_by_reader.items()}}

    benchmarks = {
        "offset": {"by_offset_mm": {"40": benchmark({full: {"role_macro_f1": .9},
                                                      "B0_Rigid27": {"role_macro_f1": .3},
                                                      "B1_NoTopology_AbsXYZ": {"role_macro_f1": .8}})}},
        "dynamic": dynamic,
        "retroactive": benchmark({full: {"wrong_to_right": .7, "right_to_wrong": .03},
                                  "B3_NoDiscount": {"wrong_to_right": .5}}),
        "station": benchmark({full: {"station_macro_f1": .9}, "B4_NoBeliefIPE": {"station_macro_f1": .8}}),
    }
    gate = evaluation._gate(SimpleNamespace(split="LOAO_OPEN3D"), benchmarks, _risk(), _risk())
    assert gate["checks"]["dynamic_vs_no_pullback_wilcoxon"]["status"] == "pass"
    tracking_gate = gate["checks"]["dynamic_tracking_rmse_mm"]
    assert tracking_gate["status"] == "insufficient_evidence"
    assert tracking_gate["value"] is None and tracking_gate["conditional_finite_rmse_mm"] == 0.
    assert tracking_gate["unmeasured_frame_roles"] == 2
    assert gate["status"] != "pass"


def test_capture_fraction_does_not_claim_roles_without_geometric_truth():
    frame = _frame(((0, 1),), _centers(0, 2))
    episode = SimpleNamespace(frames=(frame,))
    truth = evaluation._tape_truth(episode)
    acquisition = truth.acquisition()
    assert acquisition["rendered_frame_role_opportunities"] == 2
    assert acquisition["all_slice_truth_frame_role_opportunities"] == 2
    assert acquisition["rendered_frame_role_opportunities_without_finite_truth"] == 1
    assert acquisition["rendered_fraction_on_finite_slice_truth"] == .5
    measured = evaluation._measure(_Reader((_estimate(_centers(0, 1, 2), 3),)), episode, "dynamic", truth)
    assert_array_equal(np.isfinite(measured.squared_errors[0]), [True, False, False, False, False, False])


def test_retroactive_compares_replay_at_both_endpoints_on_all_prefix_slots():
    empty = _estimate(_centers())
    correct = _estimate(_centers(0, 1), 1, 2)
    wrong_second = _estimate(_centers(0, 1), 1, 1)
    frames = (_frame((), _centers()), _frame(((0,), (1,)), _centers(0, 1)),
              _frame((), _centers()), _frame((), _centers()))
    episode = SimpleNamespace(frames=frames)
    reader = _Reader((empty, correct, empty, empty), {2: (empty, wrong_second), 4: (empty, correct, empty, empty)})
    measured = evaluation._measure(reader, episode, "retroactive", evaluation._tape_truth(episode))
    assert reader.replays == [2, 4]
    assert_array_equal(measured.flips, [1, 1, 0, 1])
    assert_array_equal(measured.exact_flips, [1, 1, 0, 1])
    assert measured.prefix_echo_before and measured.prefix_echo_after


def test_b1_three_way_weighted_centroid_matches_dense_marginal_conditioning():
    rng = np.random.default_rng(12)
    reader = object.__new__(B1_NoTopology_AbsXYZ)
    reader._mean = rng.normal(size=(6, 3))
    initial_mean = reader._mean.copy()
    state_root = rng.normal(size=(6, 3, 3))
    reader._cov = state_root @ state_root.transpose(0, 2, 1) + np.eye(3)
    prior = block_diag(*reader._cov)
    roles = np.array([0, 2, 4, 5])
    contributors = (0, 2, 5)
    role_jacobians = rng.normal(size=(len(roles), 2, 3))
    pose_root = rng.normal(size=(2 * len(roles), 2))
    effective = pose_root @ pose_root.T + np.diag(np.linspace(.5, 1.2, len(pose_root)))
    transform = np.zeros((2, 2 * len(roles)))
    dense_H = np.zeros((2, 18))
    for role, w in zip(contributors, (.2, .3, .5), strict=True):
        idx = list(roles).index(role)
        transform[:, 2*idx:2*idx+2] = w * np.eye(2)
        dense_H[:, 3*role:3*role+3] = w * role_jacobians[idx]
    rho = .6
    residual = np.array([1.75, -2.5])
    association = _Association(0., ((), contributors), np.array([1]), transform, residual)
    cross = prior @ dense_H.T
    innovation = dense_H @ cross + transform @ effective @ transform.T / rho
    gain = np.linalg.solve(innovation, cross.T).T
    dense_cov = prior - gain @ cross.T
    dense_mean = initial_mean + (gain @ residual).reshape(6, 3)

    reader._condition_centroids(roles, role_jacobians, effective, association, rho)
    assert_allclose(reader._mean, dense_mean, rtol=1.e-12, atol=1.e-12)
    for role in range(6):
        ix = slice(3*role, 3*role+3)
        assert_allclose(reader._cov[role], dense_cov[ix, ix], rtol=1.e-12, atol=1.e-12)


def test_matched_conditional_advantage_cannot_pass_the_held_comparison():
    station = {"methods": {evaluation.FULL: {"station_macro_f1": {"value": .85}}}}
    gate = evaluation._gate(SimpleNamespace(split="FAMILY_HELD_OUT_ATLASES"), {"station": station},
                            _risk(), _risk(full=.10, b3=.05))
    assert gate["checks"]["matched_conditional_violation_advantage_vs_no_discount"]["status"] == "pass"
    assert gate["checks"]["held_source_conditional_violation"]["status"] == "pass"
    assert gate["checks"]["held_source_conditional_violation_advantage_vs_no_discount"]["status"] == "fail"
    assert gate["status"] == "fail"
    assert gate["engineering_controls"]["held_source_certification_acceptance"]["criterion"] == ">= 0.4"


def test_evaluator_rejects_schema_5_training_provenance(monkeypatch):
    monkeypatch.setattr(evaluation, "load_station_checkpoint", lambda *_: (None, None, {"schema_version": 5}))
    with pytest.raises(ValueError, match=r"evaluator schema 6.*requires refitting"):
        evaluation._load(SimpleNamespace(checkpoint="old.pt"), None, ())


def test_training_source_manifest_opens_only_allowlisted_pipeline_bytes(monkeypatch):
    root = Path(evaluation.__file__).resolve().parent.parent
    reader_names = ("__init__", "baselines", "belief_ipe", "dynamic_factor_graph", "eval_reader",
                    "joint_association", "split_prior", "station_classifier")
    sim_names = ("__init__", "anatomy_assets", "audit_training_support", "material_chart",
                 "metric_graph_gp", "physics_ultrasound_kernel", "smplx_dynamic_atlas")
    allowed = {f"{package}/{module}.py" for package, modules in (("reader", reader_names), ("sim", sim_names))
               for module in modules}
    open_path = Path.open
    seen = []

    def record_open(path, *args, **kwargs):
        relative = path.relative_to(root).as_posix()
        # Reject unexpected paths before opening them, including any holdout source.
        assert relative in allowed, f"training provenance opened unauthorized source {relative}"
        seen.append(relative)
        return open_path(path, *args, **kwargs)

    monkeypatch.delenv(evaluation.HOLDOUT_ENV, raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", record_open)
        manifest = evaluation._source_sha256()

    assert len(seen) == len(allowed)
    assert set(seen) == manifest.keys() == allowed
    assert manifest == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in allowed}
