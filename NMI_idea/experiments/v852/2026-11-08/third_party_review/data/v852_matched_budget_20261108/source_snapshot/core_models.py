"""Object-level posterior bundles for V8.5.2 local-to-chain experiments.

This module deliberately keeps the public input small.  A row is one complete
object/configuration and contains only observed evidence (``near``, ``probe``
and a public ``stratum``) plus single-program outcome records.  Hidden
generator labels, geometry, and terminal complete-chain labels are rejected
from the evidence interface.  Sampling returns a whole row in ``paired`` and
``full_history`` modes, so program outcomes retain their within-object
correlations.  ``independent`` is an explicit dependence ablation.

The bootstrap envelopes returned by :meth:`ObjectBundleModel.candidate_weights`
are empirical resampling scenarios.  They are not confidence sets or finite
sample guarantees.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog

from .learning import common_policy_mixture


_PUBLIC_EVIDENCE = frozenset(("near", "probe", "stratum"))
_HIDDEN_TOKENS = frozenset(("hidden", "true", "u", "v", "geometry", "generator", "oracle", "class"))


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite numeric value")
    return result


def _copy_json(value: Any, name: str = "value") -> Any:
    """Deep copy public records while rejecting non-serializable mutable values."""
    if isinstance(value, np.ndarray):
        return _copy_json(value.tolist(), name)
    if isinstance(value, np.generic):
        return _copy_json(value.item(), name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_copy_json(item, name) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item, f"{name}.{key}") for key, item in value.items()}
    raise ValueError(f"{name} contains an unsupported value type")


def _public_evidence(evidence: Mapping[str, Any], *, require_any: bool = True) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    for key in evidence:
        name = str(key).lower()
        # Match semantic tokens rather than substrings: ``stratum`` contains
        # the letter ``u`` but is a legitimate public context label.
        tokens = {token for token in name.replace("-", "_").split("_") if token}
        if tokens & _HIDDEN_TOKENS:
            raise ValueError(f"hidden or oracle evidence key is forbidden: {key!r}")
        if key not in _PUBLIC_EVIDENCE:
            raise ValueError(f"unknown evidence key {key!r}; only near/probe/stratum are public")
    result: dict[str, Any] = {}
    for key in ("near", "probe"):
        if key in evidence and evidence[key] is not None:
            result[key] = _finite_float(evidence[key], f"evidence[{key!r}]")
    if "stratum" in evidence and evidence["stratum"] is not None:
        value = evidence["stratum"]
        if isinstance(value, (list, dict, np.ndarray)):
            raise ValueError("evidence['stratum'] must be a scalar public label")
        result["stratum"] = str(value)
    if require_any and not result:
        raise ValueError("evidence must contain at least one public near/probe/stratum value")
    return result


def _normalise_weights(weights: Sequence[float], n: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) != n or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite, nonnegative, and match the row count")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must have positive mass")
    return values / total


@dataclass(frozen=True)
class CandidateWeightSet:
    """Empirical posterior scenarios from object-level bootstrap resampling."""

    scenarios: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict[str, Any]
    coupling_weights: dict[str, np.ndarray] = field(default_factory=dict)
    coupling_audit: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def weights(self) -> np.ndarray:
        """Alias used by callers that only need the scenario matrix."""
        return self.scenarios

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.scenarios, dtype=dtype)

    def __len__(self) -> int:
        return int(len(self.scenarios))


class ObjectBundleModel:
    """Posterior over complete object/configuration bundles.

    Parameters
    ----------
    rows:
        One row per independent object/configuration.  Required keys are
        ``object_id``, ``stratum``, ``near``, ``probe`` and ``programs``.
        Program values are copied as public outcome records; their internal
        fields are never used as evidence features.
    mode:
        ``paired`` preserves each near/probe association; ``shuffled``
        permutes near labels within stratum while keeping all program rows and
        probe labels attached to their objects; ``full_history`` uses a
        nearest-neighbour joint history model; ``independent`` is a deliberate
        program-dependence ablation and draws each program from a separate
        posterior row.
    bandwidth:
        Positive Gaussian-kernel bandwidth in standardized evidence units.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        mode: str = "paired",
        seed: int = 0,
        bandwidth: float = 1.0,
    ) -> None:
        if mode not in {"paired", "shuffled", "full_history", "independent"}:
            raise ValueError("mode must be paired, shuffled, full_history, or independent")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ValueError("rows must be a nonempty sequence")
        self.mode = str(mode)
        self.seed = int(seed)
        self.bandwidth = _finite_float(bandwidth, "bandwidth")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self._rng = np.random.default_rng(self.seed)
        original = self._validate_rows(rows)
        self._original_rows = tuple(original)
        if self.mode == "shuffled":
            self._rows, self._shuffle_audit = self._make_shuffled(original)
        else:
            self._rows = tuple(original)
            self._shuffle_audit = {
                "applied": False,
                "within_stratum": True,
                "program_rows_unchanged": True,
            }
        self._ids = tuple(row["object_id"] for row in self._rows)
        self._index = {object_id: index for index, object_id in enumerate(self._ids)}
        self._strata = tuple(str(row["stratum"]) for row in self._rows)
        self._program_names = tuple(
            sorted({name for row in self._rows for name in row["programs"]})
        )
        self._feature_names = ("near", "probe")
        self._features = np.asarray(
            [[float(row["near"]), float(row["probe"])] for row in self._rows],
            dtype=float,
        )
        self._means = np.mean(self._features, axis=0)
        scales = np.std(self._features, axis=0)
        self._scales = np.where(scales > 1.0e-12, scales, 1.0)
        self._standardized = (self._features - self._means) / self._scales
        self._stratum_counts = Counter(self._strata)
        self._near_sign = np.asarray([self._sign(row["near"]) for row in self._rows], dtype=int)
        self._probe_sign = np.asarray([self._sign(row["probe"]) for row in self._rows], dtype=int)
        # Coupling assignments and feature-only models are immutable from the
        # caller's point of view.  Keep the caches private and return copies
        # whenever a full row/audit is exposed; this avoids rebuilding large
        # program outcome records for every bootstrap envelope member.
        self._coupling_assignment_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        self._coupled_rows_cache: dict[str, tuple[dict[str, Any], ...]] = {}
        self._coupling_feature_models: dict[str, "ObjectBundleModel"] = {}
        self._summary = self._build_summary()

    @staticmethod
    def _sign(value: Any) -> int:
        numeric = _finite_float(value, "evidence sign")
        return int(np.sign(numeric))

    @staticmethod
    def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        required = {"object_id", "stratum", "near", "probe", "programs"}
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                raise TypeError(f"row {index} must be a mapping")
            missing = required - set(raw)
            if missing:
                raise ValueError(f"row {index} is missing required key(s): {sorted(missing)}")
            object_id = str(raw["object_id"])
            if not object_id:
                raise ValueError(f"row {index} has an empty object_id")
            if object_id in seen:
                raise ValueError(f"object_id must be unique: {object_id!r}")
            seen.add(object_id)
            stratum = str(raw["stratum"])
            programs = raw["programs"]
            if not isinstance(programs, Mapping) or not programs:
                raise ValueError(f"row {index} programs must be a nonempty mapping")
            public_programs = {
                str(name): _copy_json(value, f"row[{index}].programs[{name!r}]")
                for name, value in programs.items()
            }
            result.append(
                {
                    "object_id": object_id,
                    "stratum": stratum,
                    "near": _finite_float(raw["near"], f"row[{index}].near"),
                    "probe": _finite_float(raw["probe"], f"row[{index}].probe"),
                    "programs": public_programs,
                }
            )
        return result

    @staticmethod
    def _make_shuffled(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        rng = np.random.default_rng(0x5A17)
        copies = [deepcopy(dict(row)) for row in rows]
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for index, row in enumerate(copies):
            groups.setdefault(str(row["stratum"]), []).append(index)
        before = [float(row["near"]) for row in copies]
        permutation: dict[str, list[int]] = {}
        for stratum, indices in groups.items():
            values = [copies[index]["near"] for index in indices]
            order = rng.permutation(len(indices))
            permutation[stratum] = [indices[int(i)] for i in order]
            for destination, source_local in zip(indices, order):
                copies[destination]["near"] = values[int(source_local)]
        after = [float(row["near"]) for row in copies]
        program_unchanged = all(
            copies[index]["programs"] == rows[index]["programs"]
            for index in range(len(copies))
        )
        audit = {
            "applied": True,
            "within_stratum": True,
            "permutation": permutation,
            "near_multiset_by_stratum_preserved": all(
                sorted(before[index] for index in indices)
                == sorted(after[index] for index in indices)
                for indices in groups.values()
            ),
            "program_rows_unchanged": bool(program_unchanged),
            "probe_rows_unchanged": all(
                copies[index]["probe"] == rows[index]["probe"]
                for index in range(len(copies))
            ),
        }
        return tuple(copies), audit

    def _build_summary(self) -> dict[str, Any]:
        return {
            "schema": "v852_object_bundle_model_v1",
            "mode": self.mode,
            "n_objects": len(self._rows),
            "n_strata": len(self._stratum_counts),
            "strata": dict(self._stratum_counts),
            "programs": list(self._program_names),
            "evidence_features": list(self._feature_names),
            "hidden_features_used": [],
            "within_object_program_outcomes_preserved": self.mode != "independent",
            "posterior_structure": (
                "continuous_joint_rbf_knn" if self.mode == "full_history"
                else "laplace_P_sign_probe_given_sign_near_plus_probe_gaussian"
            ),
            "structure_discards_amplitude_without_probe": self.mode in {"paired", "shuffled", "independent"},
            "bootstrap_is_strict_confidence_set": False,
            "shuffle_audit": deepcopy(self._shuffle_audit),
        }

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(row) for row in self._rows)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def program_names(self) -> tuple[str, ...]:
        return self._program_names

    def summary(self) -> dict[str, Any]:
        return deepcopy(self._summary)

    def audit(self) -> dict[str, Any]:
        """Return immutable-copy diagnostics for protocol files."""
        return {
            **self.summary(),
            "n_unique_object_ids": len(self._index),
            "near_range": [float(np.min(self._features[:, 0])), float(np.max(self._features[:, 0]))],
            "probe_range": [float(np.min(self._features[:, 1])), float(np.max(self._features[:, 1]))],
            "row_hash_inputs": [row["object_id"] for row in self._rows],
        }

    def _candidate_indices(self, public: Mapping[str, Any]) -> np.ndarray:
        """Select a public stratum when it exists; unseen strata stay open."""
        if "stratum" not in public:
            return np.arange(len(self._rows), dtype=int)
        same = np.asarray([stratum == public["stratum"] for stratum in self._strata])
        # A held-out pose/shape is deliberately not assigned zero support just
        # because its public stratum label was absent from local training.
        return np.flatnonzero(same) if np.any(same) else np.arange(len(self._rows), dtype=int)

    def _continuous_weights(self, evidence: Mapping[str, Any]) -> np.ndarray:
        """Full-history joint RBF/KNN posterior over continuous visible features."""
        public = _public_evidence(evidence)
        distances = np.zeros(len(self._rows), dtype=float)
        observed = 0
        for column, key in enumerate(("near", "probe")):
            if key in public:
                query = float(public[key])
                distances += ((self._features[:, column] - query) / self._scales[column]) ** 2
                observed += 1
        if observed:
            distances /= float(observed)
        likelihood = np.exp(-0.5 * distances / (self.bandwidth ** 2))
        candidate = self._candidate_indices(public)
        mask = np.zeros(len(self._rows), dtype=bool)
        mask[candidate] = True
        likelihood *= mask
        if not np.isfinite(likelihood).all() or float(likelihood.sum()) <= 0:
            result = np.zeros(len(self._rows), dtype=float)
            result[candidate] = 1.0 / len(candidate)
            return result
        return likelihood / likelihood.sum()

    def _paired_weights(self, evidence: Mapping[str, Any]) -> np.ndarray:
        """Low-dimensional structural posterior using learned sign coupling.

        The near observation supplies the public category ``Y=sign(near)``.
        The probe category ``g=sign(probe)`` is learned from the observed rows
        with Laplace smoothing; it is never read from a hidden generator label.
        A continuous Gaussian likelihood is then applied only when a probe
        value is actually visible.  Thus paired mode intentionally discards
        amplitude information when only the near feature is available.
        """
        public = _public_evidence(evidence)
        candidate = self._candidate_indices(public)
        if "near" not in public and "probe" not in public:
            result = np.zeros(len(self._rows), dtype=float)
            result[candidate] = 1.0 / len(candidate)
            return result

        # Learn P(g | Y) on the candidate stratum, with Laplace smoothing.
        row_y = self._near_sign[candidate]
        row_g = self._probe_sign[candidate]
        categories = (-1, 0, 1)
        alpha = 1.0
        if "near" in public:
            y = self._sign(public["near"])
            subset = row_g[row_y == y]
            if len(subset) == 0:
                subset = row_g
        else:
            subset = row_g
        counts = {g: int(np.sum(subset == g)) for g in categories}
        denominator = float(len(subset) + alpha * len(categories))
        p_g = {g: (counts[g] + alpha) / denominator for g in categories}
        n_by_g = {g: max(1, int(np.sum(row_g == g))) for g in categories}
        result = np.zeros(len(self._rows), dtype=float)
        for local, index in enumerate(candidate):
            result[int(index)] = p_g[int(row_g[local])] / n_by_g[int(row_g[local])]

        if "probe" in public:
            query = float(public["probe"])
            # The raw probe is a visible observation; update the structural
            # group assignment with its Gaussian measurement likelihood.
            sigma = max(float(self._scales[1]), 1.0e-6) * self.bandwidth
            result[candidate] *= np.exp(-0.5 * ((self._features[candidate, 1] - query) / sigma) ** 2)
        total = float(result.sum())
        if total <= 0.0 or not np.isfinite(total):
            result[:] = 0.0
            result[candidate] = 1.0 / len(candidate)
            return result
        return result / total

    def weights(self, evidence: Mapping[str, Any] | None = None) -> np.ndarray:
        """Return an object posterior from only visible evidence.

        ``full_history`` uses the joint two-feature RBF/KNN distance.  The
        paired and shuffled modes use the explicit low-dimensional
        ``P(sign(probe) | sign(near))`` structure; shuffled changes only the
        public near/probe association while preserving each program row.
        """
        if evidence is None:
            evidence = {}
        if not evidence:
            return np.full(len(self._rows), 1.0 / len(self._rows))
        if self.mode == "full_history":
            return self._continuous_weights(evidence)
        return self._paired_weights(evidence)

    def _program_weight(self, row: Mapping[str, Any], evidence: Mapping[str, Any], program: str) -> float:
        # Independent-mode program draws use the same public evidence but only
        # records that contain the requested program.  No hidden generator key
        # is consulted.
        if program not in row["programs"]:
            return 0.0
        return 1.0

    def draw(
        self,
        evidence: Mapping[str, Any] | None,
        rng: Any,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample complete object bundles; preserve rows across programs."""
        if isinstance(n, bool) or int(n) != n or int(n) <= 0:
            raise ValueError("n must be a positive integer")
        if rng is None or not hasattr(rng, "choice"):
            raise TypeError("rng must provide choice()")
        posterior = self.weights(evidence or {})
        if self.mode != "independent":
            indices = np.asarray(rng.choice(len(self._rows), size=int(n), p=posterior), dtype=int).reshape(-1)
            return [deepcopy(self._rows[int(index)]) for index in indices]

        # Explicit ablation: each program is drawn independently from the
        # object posterior.  The synthetic row carries provenance so callers
        # cannot mistake it for a real same-object observation.
        outputs: list[dict[str, Any]] = []
        for sample in range(int(n)):
            programs: dict[str, Any] = {}
            selected: dict[str, str] = {}
            for program in self._program_names:
                candidates = [index for index, row in enumerate(self._rows) if program in row["programs"]]
                local = posterior[np.asarray(candidates, dtype=int)]
                local = local / local.sum()
                picked = int(rng.choice(candidates, p=local))
                programs[program] = deepcopy(self._rows[picked]["programs"][program])
                selected[program] = self._rows[picked]["object_id"]
            outputs.append({
                "object_id": f"independent_sample_{sample}",
                "stratum": str((evidence or {}).get("stratum", "unknown")),
                "near": (evidence or {}).get("near"),
                "probe": (evidence or {}).get("probe"),
                "programs": programs,
                "independent_source_object_ids": selected,
            })
        return outputs

    def _coupling_assignments(self, label: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Return only the counterfactual probe assignment and its audit.

        The assignment is the part needed by a posterior-weight calculation.
        It intentionally does not copy the potentially large ``programs``
        records.  A defensive array/audit copy is returned so private cache
        state cannot be mutated by callers.
        """
        if label not in {"positive", "negative"}:
            raise ValueError("coupling label must be positive or negative")
        cached = self._coupling_assignment_cache.get(label)
        if cached is not None:
            values, audit = cached
            return values.copy(), deepcopy(audit)

        assigned = np.asarray([float(row["probe"]) for row in self._rows], dtype=float)
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for index, row in enumerate(self._rows):
            groups.setdefault(str(row["stratum"]), []).append(index)
        for indices in groups.values():
            near_order = sorted(
                indices,
                key=lambda index: (self._sign(self._rows[index]["near"]), self._rows[index]["near"]),
            )
            probes = [self._rows[index]["probe"] for index in indices]
            probes.sort(key=lambda value: (self._sign(value), float(value)), reverse=label == "negative")
            for destination, probe in zip(near_order, probes):
                assigned[destination] = float(probe)
        audit = {
            "label": label,
            "generated_from_public_near_probe": True,
            "strict_confidence_set": False,
            "near_multiset_by_stratum_preserved": True,
            "probe_multiset_by_stratum_preserved": all(
                sorted(float(row["probe"]) for row in self._rows if row["stratum"] == stratum)
                == sorted(float(assigned[index]) for index in indices)
                for stratum, indices in groups.items()
            ),
            "program_rows_unchanged": True,
            "observed_sign_product_before": float(np.mean(self._near_sign * self._probe_sign)),
            "observed_sign_product_after": float(np.mean([
                self._sign(self._rows[index]["near"]) * self._sign(assigned[index])
                for index in range(len(self._rows))
            ])),
            "note": "Counterfactual public association scenario; not a hidden-V truth label.",
        }
        self._coupling_assignment_cache[label] = (assigned.copy(), deepcopy(audit))
        return assigned.copy(), deepcopy(audit)

    def _coupled_rows(self, label: str) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        """Create full public coupling rows without changing outcomes.

        ``positive`` and ``negative`` pairings preserve each stratum's near and
        probe value multisets and every complete program outcome row.  They are
        deliberately counterfactual association scenarios, not recovered
        hidden anatomy labels.  The full rows are cached privately and copied
        on return because downstream rollout needs the program records.
        """
        assigned, audit = self._coupling_assignments(label)
        cached_rows = self._coupled_rows_cache.get(label)
        if cached_rows is None:
            copies = [deepcopy(dict(row)) for row in self._rows]
            for index, probe in enumerate(assigned):
                copies[index]["probe"] = float(probe)
            cached_rows = tuple(copies)
            self._coupled_rows_cache[label] = cached_rows
        return tuple(deepcopy(row) for row in cached_rows), audit

    def _coupling_feature_model(self, label: str) -> "ObjectBundleModel":
        """Return a cached paired model containing only public features.

        Weight calculations never inspect program outcomes, so a temporary
        model can use a small sentinel program mapping.  Keeping this model
        separate from :meth:`_coupled_rows` ensures candidate-weight scenarios
        do not repeatedly copy frame captures or other execution records.
        """
        if label not in {"positive", "negative"}:
            raise ValueError("coupling label must be positive or negative")
        cached = self._coupling_feature_models.get(label)
        if cached is not None:
            return cached
        assigned, _ = self._coupling_assignments(label)
        rows = [
            {
                "object_id": f"{row['object_id']}__coupling_{label}_{index}",
                "stratum": str(row["stratum"]),
                "near": float(row["near"]),
                "probe": float(assigned[index]),
                "programs": {"__weights_only__": {}},
            }
            for index, row in enumerate(self._rows)
        ]
        model = ObjectBundleModel(rows, mode="paired", seed=self.seed, bandwidth=self.bandwidth)
        self._coupling_feature_models[label] = model
        return model

    def coupling_model(self, label: str) -> "ObjectBundleModel":
        """Return a paired model for a positive/negative public coupling scenario."""
        rows, _ = self._coupled_rows(label)
        return ObjectBundleModel(rows, mode="paired", seed=self.seed, bandwidth=self.bandwidth)

    def coupling_audit(self, label: str | None = None) -> dict[str, Any]:
        """Audit observed coupling and optional public counterfactual coupling."""
        labels = [label] if label is not None else ["positive", "negative"]
        scenarios: dict[str, Any] = {}
        for name in labels:
            _, audit = self._coupling_assignments(name)
            scenarios[name] = audit
        observed = float(np.mean(self._near_sign * self._probe_sign))
        return {
            "schema": "v852_public_coupling_audit_v1",
            "observed_sign_product": observed,
            "observed_positive_extreme": bool(observed >= 0.9),
            "observed_negative_extreme": bool(observed <= -0.9),
            "counterfactual_scenarios": scenarios,
        }

    def scenario_rows(self, evidence: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        """Return row-level public coupling scenarios for downstream planning.

        The returned ``positive``/``negative`` rows retain the same complete
        single-program outcome records and public feature marginals.  A caller
        can recompute a policy matrix against each scenario instead of treating
        a bootstrap envelope as an unearned partial-identification guarantee.
        """
        public = _public_evidence(evidence or {}, require_any=False)
        result: dict[str, dict[str, Any]] = {
            "observed": {
                "rows": self.rows,
                "weights": self.weights(public),
                "audit": {
                    "label": "observed",
                    "generated_from_public_near_probe": False,
                    "strict_confidence_set": False,
                },
            }
        }
        for label in ("positive", "negative"):
            rows, audit = self._coupled_rows(label)
            scenario = ObjectBundleModel(rows, mode="paired", seed=self.seed, bandwidth=self.bandwidth)
            result[label] = {
                "rows": scenario.rows,
                "weights": scenario.weights(public),
                "audit": audit,
            }
        return result

    def candidate_weights(
        self,
        evidence: Mapping[str, Any] | None,
        *,
        n_models: int = 9,
        seed: int | None = None,
    ) -> CandidateWeightSet:
        """Generate object-bootstrap posterior scenarios.

        Each scenario resamples objects with replacement before evaluating the
        same evidence kernel.  The pointwise envelope is descriptive and is
        explicitly *not* a confidence set.
        """
        if isinstance(n_models, bool) or int(n_models) != n_models or int(n_models) <= 0:
            raise ValueError("n_models must be a positive integer")
        public = _public_evidence(evidence or {}, require_any=False)
        base = self.weights(public)
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        scenarios = []
        for _ in range(int(n_models)):
            sample_indices = rng.choice(len(self._rows), size=len(self._rows), replace=True)
            # Bootstrap the public feature population, then evaluate the same
            # posterior structure on the resampled empirical model.  Program
            # rows are irrelevant to ``weights``; using a sentinel mapping
            # preserves every feature/statistical/RNG operation while avoiding
            # deep copies of frame captures and execution records.
            resampled_rows = []
            for position, original_index in enumerate(sample_indices):
                source = self._rows[int(original_index)]
                resampled_rows.append(
                    {
                        "object_id": f"{source['object_id']}__bootstrap_{position}",
                        "stratum": str(source["stratum"]),
                        "near": float(source["near"]),
                        "probe": float(source["probe"]),
                        "programs": {"__weights_only__": {}},
                    }
                )
            bootstrap_mode = "full_history" if self.mode == "full_history" else "paired"
            bootstrap_model = ObjectBundleModel(
                resampled_rows,
                mode=bootstrap_mode,
                seed=self.seed,
                bandwidth=self.bandwidth,
            )
            local = bootstrap_model.weights(public)
            mapped = np.zeros(len(self._rows), dtype=float)
            for picked, original_index in enumerate(sample_indices):
                mapped[int(original_index)] += float(local[picked])
            scenarios.append(mapped / mapped.sum())
        matrix = np.asarray(scenarios, dtype=float)
        coupling_weights: dict[str, np.ndarray] = {}
        coupling_audit: dict[str, dict[str, Any]] = {}
        for label in ("positive", "negative"):
            coupling_model = self._coupling_feature_model(label)
            coupling_weights[label] = coupling_model.weights(public)
            _, coupling_audit[label] = self._coupling_assignments(label)
        metadata = {
            "schema": "v852_empirical_object_bootstrap_envelope_v1",
            "n_models": int(n_models),
            "seed": self.seed if seed is None else int(seed),
            "mode": self.mode,
            "strict_confidence_set": False,
            "note": "Object bootstrap scenarios and pointwise envelope; no finite-sample coverage claim.",
            "point_estimate": base.tolist(),
            "coupling_scenarios": ["positive", "negative"],
            "coupling_note": "Positive/negative scenarios preserve public feature marginals and program rows; they are not hidden-state truth or confidence bounds.",
        }
        return CandidateWeightSet(
            matrix,
            np.min(matrix, axis=0),
            np.max(matrix, axis=0),
            metadata,
            coupling_weights=coupling_weights,
            coupling_audit=coupling_audit,
        )

    def posterior_update(
        self,
        weights: Sequence[float],
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        """Update an object posterior with a visible probe/program observation."""
        posterior = _normalise_weights(weights, len(self._rows))
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        allowed_observation = _PUBLIC_EVIDENCE | frozenset(("program_id", "success", "status", "duration"))
        for key in observation:
            name = str(key).lower()
            tokens = {token for token in name.replace("-", "_").split("_") if token}
            if tokens & _HIDDEN_TOKENS:
                raise ValueError(f"hidden or oracle observation key is forbidden: {key!r}")
            if key not in allowed_observation:
                raise ValueError(f"unknown observation key {key!r}")
        _public_evidence(
            {key: value for key, value in observation.items() if key in _PUBLIC_EVIDENCE},
            require_any=False,
        )
        likelihood = np.ones(len(self._rows), dtype=float)
        if "near" in observation or "probe" in observation:
            visible = {key: observation[key] for key in ("near", "probe") if key in observation}
            likelihood *= self._kernel_weights(visible)
            likelihood *= len(self._rows)
        if "stratum" in observation:
            public_stratum = str(observation["stratum"])
            likelihood *= np.asarray([1.0 if row["stratum"] == public_stratum else 1.0e-3 for row in self._rows])
        program = observation.get("program_id")
        if program is not None:
            program = str(program)
            success = observation.get("success", None)
            status = observation.get("status", None)
            duration = observation.get("duration", None)
            for index, row in enumerate(self._rows):
                record = row["programs"].get(program)
                if record is None:
                    likelihood[index] = 0.0
                    continue
                if success is not None:
                    # ``success`` may be an audit/independent label in newer
                    # execution records.  Policy-visible updates must use the
                    # public completion label when it is available, with the
                    # legacy success field as the compatibility fallback.
                    record_success = (
                        record.get("public_success")
                        if "public_success" in record
                        else record.get("success")
                    )
                    likelihood[index] *= 0.95 if bool(record_success) == bool(success) else 0.05
                if status is not None:
                    likelihood[index] *= 0.95 if str(record.get("status")) == str(status) else 0.05
                if duration is not None and "duration" in record:
                    sigma = max(1.0e-3, float(np.std([
                        float(r["programs"][program].get("duration", 0.0))
                        for r in self._rows if program in r["programs"]
                    ])) * 0.25)
                    delta = (float(record["duration"]) - _finite_float(duration, "observation.duration")) / sigma
                    likelihood[index] *= float(np.exp(-0.5 * delta * delta))
        updated = posterior * likelihood
        if not np.isfinite(updated).all() or updated.sum() <= 0:
            raise ValueError("observation has zero support in the object bundle model")
        return updated / updated.sum()

    def common_plan(self, A: np.ndarray, costs: Sequence[float] | None = None) -> dict[str, Any]:
        return common_plan(A, costs=costs)


def common_plan(A: np.ndarray, costs: Sequence[float] | None = None) -> dict[str, Any]:
    """Solve max-min policy selection with an optional lexicographic tie-break.

    The first LP maximizes the worst model value.  When ``costs`` is supplied,
    a second LP minimizes expected cost subject to retaining that optimal value
    (up to ``1e-9``).  Thus a cheaper policy cannot silently lower the robust
    completion guarantee.  ``lower`` and ``upper`` always describe the first
    value problem, independent of the cost tie-break.
    """
    values = np.asarray(A, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("A must be a nonempty 2-D model-by-policy matrix")
    if not np.isfinite(values).all():
        raise ValueError("A must contain only finite values")
    mixture, lower, upper = common_policy_mixture(values)
    tie_break: dict[str, Any] = {
        "enabled": costs is not None,
        "objective": "min_expected_cost_at_maxmin_value" if costs is not None else None,
        "costs": None,
        "expected_cost": None,
        "value_tolerance": 1.0e-9,
    }
    if costs is not None:
        cost_vector = np.asarray(costs, dtype=float).reshape(-1)
        if len(cost_vector) != values.shape[1] or not np.isfinite(cost_vector).all():
            raise ValueError("costs must be a finite vector with one value per policy")
        # Keep the original robust value as a hard lexicographic constraint.
        result = linprog(
            cost_vector,
            A_ub=-values,
            b_ub=np.full(values.shape[0], -(lower - 1.0e-9), dtype=float),
            A_eq=np.ones((1, values.shape[1]), dtype=float),
            b_eq=np.asarray([1.0]),
            bounds=[(0.0, None)] * values.shape[1],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"common policy cost tie-break failed: {result.message}")
        mixture = np.asarray(result.x, dtype=float)
        mixture[np.abs(mixture) < 1.0e-10] = 0.0
        mixture /= float(mixture.sum())
        tie_break["costs"] = cost_vector.tolist()
        tie_break["expected_cost"] = float(cost_vector @ mixture)
        tie_break["value_after_tie_break"] = float(np.min(values @ mixture))
    return {
        "weights": mixture,
        "lower": float(lower),
        "upper": float(upper),
        "n_models": int(np.asarray(A).shape[0]),
        "n_policies": int(np.asarray(A).shape[1]),
        "policy_is_common": True,
        "model_specific_oracle_not_used": True,
        "tie_break": tie_break,
    }


__all__ = ["CandidateWeightSet", "ObjectBundleModel", "common_plan"]
