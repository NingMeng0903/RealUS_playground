"""Small empirical models for program-level option outcomes.

The models in this module are deliberately empirical. They describe the
observed execution records supplied to fit_joint; they do not provide
confidence or safety guarantees and they do not infer unobserved variables.
Only the explicitly requested numeric feature keys, program_id, and the
recorded outcome fields are used.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog


_OUTCOME_KEYS = ("success", "duration", "end_error_m", "status", "program_id")
_MISSING = object()


def _as_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rng_index(rng: Any, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot sample an empty neighbor set")
    if rng is None:
        rng = np.random.default_rng()
    if hasattr(rng, "integers"):
        return int(rng.integers(0, size))
    if hasattr(rng, "randint"):
        return int(rng.randint(0, size))
    raise TypeError("rng must provide integers() or randint()")


def _validate_feature_keys(records: Sequence[Mapping[str, Any]], feature_keys: Sequence[str]) -> tuple[str, ...]:
    keys = tuple(str(key) for key in feature_keys)
    if not keys:
        raise ValueError("feature_keys must contain at least one explicit feature")
    if len(set(keys)) != len(keys):
        raise ValueError("feature_keys must be unique")
    if any(key in _OUTCOME_KEYS for key in keys):
        raise ValueError("feature_keys must contain numeric context features, not outcome keys")
    for index, record in enumerate(records):
        absent = [key for key in keys if key not in record]
        if absent:
            raise ValueError(f"record {index} is missing requested feature key(s): {absent}")
    return keys


def _validate_extra_outcome_keys(
    records: Sequence[Mapping[str, Any]],
    extra_outcome_keys: Sequence[str] | None,
    feature_keys: Sequence[str],
) -> tuple[str, ...]:
    if extra_outcome_keys is None:
        return ()
    if isinstance(extra_outcome_keys, (str, bytes)):
        raise ValueError("extra_outcome_keys must be a sequence of field names")
    keys = tuple(str(key) for key in extra_outcome_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("extra_outcome_keys must be unique")
    reserved = set(_OUTCOME_KEYS) | {"nominal_duration"}
    overlap = sorted((set(keys) & reserved) | (set(keys) & set(feature_keys)))
    if overlap:
        raise ValueError(f"extra_outcome_keys overlap reserved or feature keys: {overlap}")
    for index, record in enumerate(records):
        absent = [key for key in keys if key not in record]
        if absent:
            raise ValueError(f"record {index} is missing extra outcome key(s): {absent}")
    return keys


def _json_safe_copy(value: Any, *, name: str) -> Any:
    """Copy an explicitly allowed outcome into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return _json_safe_copy(value.tolist(), name=name)
    if isinstance(value, np.generic):
        return _json_safe_copy(value.item(), name=name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item, name=name) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} mapping keys must be strings")
            copied[key] = _json_safe_copy(item, name=f"{name}.{key}")
        return copied
    raise ValueError(f"{name} is not JSON-safe")


def _nominal_for_record(
    record: Mapping[str, Any],
    nominal_duration: Callable[[Mapping[str, Any]], Any] | float | None,
) -> float | None:
    value: Any = record.get("nominal_duration", _MISSING)
    if value is _MISSING and nominal_duration is not None:
        value = nominal_duration(record) if callable(nominal_duration) else nominal_duration
    if value is _MISSING or value is None:
        return None
    value = _as_float(value, name="nominal_duration")
    if value <= 0.0:
        raise ValueError("nominal_duration must be positive")
    return value


def _nominal_for_context(
    context: Mapping[str, Any],
    nominal_duration: Callable[[Mapping[str, Any]], Any] | float | None,
) -> float | None:
    value: Any = context.get("nominal_duration", _MISSING)
    if callable(value):
        value = value(context)
    elif value is _MISSING and nominal_duration is not None:
        value = nominal_duration(context) if callable(nominal_duration) else nominal_duration
    if value is _MISSING or value is None:
        return None
    value = _as_float(value, name="nominal_duration")
    if value <= 0.0:
        raise ValueError("nominal_duration must be positive")
    return value


@dataclass(frozen=True)
class _ProgramData:
    records: tuple[dict[str, Any], ...]
    features: np.ndarray
    duration_ratio: np.ndarray
    duration_ratio_valid: np.ndarray


class JointOutcomeModel:
    """Empirical KNN resampler for correlated program outcomes.

    A query is matched only to records for its program_id. The selected
    neighbor is resampled as one record, so status, success, duration, and
    end error retain their observed correlations. Fit-time means and scales
    are computed from training records only.
    """

    def __init__(
        self,
        *,
        feature_keys: tuple[str, ...],
        means: np.ndarray,
        scales: np.ndarray,
        programs: OrderedDict[str, _ProgramData],
        k: int,
        nominal_duration: Callable[[Mapping[str, Any]], Any] | float | None,
        extra_outcome_keys: tuple[str, ...],
        total_records: int,
    ) -> None:
        self.feature_keys = feature_keys
        self.means = np.asarray(means, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self._programs = programs
        self.k = int(k)
        self.nominal_duration = nominal_duration
        self.extra_outcome_keys = extra_outcome_keys
        self.metadata = {
            "kind": "empirical_knn_joint_outcome",
            "empirical": True,
            "formal_confidence_or_safety_guarantee": False,
            "feature_keys": list(feature_keys),
            "extra_outcome_keys": list(extra_outcome_keys),
            "program_ids": list(programs),
            "n_records": int(total_records),
            "k": int(k),
            "duration_scaling": bool(
                any(np.any(data.duration_ratio_valid) for data in programs.values())
            ),
        }

    def _query(self, context: Mapping[str, Any]) -> tuple[_ProgramData, list[int]]:
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        if "program_id" not in context:
            raise ValueError("context must contain program_id")
        program_id = str(context["program_id"])
        if program_id not in self._programs:
            raise KeyError(f"unknown program_id: {program_id!r}")
        missing = [key for key in self.feature_keys if key not in context]
        if missing:
            raise ValueError(f"context is missing requested feature key(s): {missing}")
        query = np.asarray(
            [_as_float(context[key], name=f"context[{key!r}]") for key in self.feature_keys],
            dtype=float,
        )
        query = (query - self.means) / self.scales
        data = self._programs[program_id]
        distances = np.linalg.norm(data.features - query[None, :], axis=1)
        order = sorted(range(len(distances)), key=lambda index: (float(distances[index]), index))
        return data, order[: min(self.k, len(order))]

    @staticmethod
    def _public_record(
        record: Mapping[str, Any],
        feature_keys: tuple[str, ...],
        extra_outcome_keys: tuple[str, ...],
        *,
        duration: float,
        nominal_duration: float | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {key: float(record[key]) for key in feature_keys}
        result.update(
            {
                "success": bool(record["success"]),
                "duration": float(duration),
                "end_error_m": float(record["end_error_m"]),
                "status": str(record["status"]),
                "program_id": str(record["program_id"]),
            }
        )
        if nominal_duration is not None:
            result["nominal_duration"] = float(nominal_duration)
        for key in extra_outcome_keys:
            result[key] = _json_safe_copy(record[key], name=key)
        return result

    def sample(
        self,
        context: Mapping[str, Any],
        rng: Any,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample complete observed outcomes for a physical context.

        The output is restricted to requested features and public outcome
        fields. Extra hidden fields in training records are never read or
        copied. If nominal durations are available, duration is sampled in
        ratio form and rescaled to the requested context nominal duration.
        """
        if isinstance(n, bool) or int(n) != n or int(n) < 0:
            raise ValueError("n must be a nonnegative integer")
        n = int(n)
        data, neighbors = self._query(context)
        target_nominal = _nominal_for_context(context, self.nominal_duration)
        outputs: list[dict[str, Any]] = []
        for _ in range(n):
            index = neighbors[_rng_index(rng, len(neighbors))]
            record = data.records[index]
            duration = float(record["duration"])
            if target_nominal is not None and bool(data.duration_ratio_valid[index]):
                duration = float(data.duration_ratio[index]) * target_nominal
            outputs.append(
                self._public_record(
                    record,
                    self.feature_keys,
                    self.extra_outcome_keys,
                    duration=duration,
                    nominal_duration=target_nominal,
                )
            )
        return outputs

    def predict_success(self, context: Mapping[str, Any]) -> float:
        """Return the empirical neighbor mean of the success indicator."""
        data, neighbors = self._query(context)
        return float(np.mean([bool(data.records[index]["success"]) for index in neighbors]))


def fit_joint(
    records: list[dict[str, Any]],
    feature_keys: list[str],
    k: int | None = None,
    nominal_duration: Callable[[Mapping[str, Any]], Any] | float | None = None,
    extra_outcome_keys: Sequence[str] | None = (),
) -> JointOutcomeModel:
    """Fit a deterministic empirical joint-outcome model.

    program_id partitions the KNN pools. Only feature_keys are used as numeric
    predictors; arbitrary extra record fields are ignored. The default
    neighbor count is min(24, n_records), capped by each program's pool.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    keys = _validate_feature_keys(records, feature_keys)
    extra_keys = _validate_extra_outcome_keys(records, extra_outcome_keys, keys)
    if k is not None and (isinstance(k, bool) or int(k) != k or int(k) <= 0):
        raise ValueError("k must be a positive integer or None")
    k_value = min(24, len(records)) if k is None else int(k)

    all_features = np.empty((len(records), len(keys)), dtype=float)
    prepared: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {row} must be a mapping")
        missing = [key for key in _OUTCOME_KEYS if key not in record]
        if missing:
            raise ValueError(f"record {row} is missing outcome key(s): {missing}")
        program_id = str(record["program_id"])
        if not program_id:
            raise ValueError(f"record {row} has an empty program_id")
        values = [_as_float(record[key], name=f"record[{row}][{key!r}]") for key in keys]
        all_features[row] = values
        duration = _as_float(record["duration"], name=f"record[{row}]['duration']")
        if duration < 0.0:
            raise ValueError("duration must be nonnegative")
        end_error = _as_float(record["end_error_m"], name=f"record[{row}]['end_error_m']")
        if end_error < 0.0:
            raise ValueError("end_error_m must be nonnegative")
        prepared_record = {
            **{key: float(record[key]) for key in keys},
            "success": bool(record["success"]),
            "duration": duration,
            "end_error_m": end_error,
            "status": str(record["status"]),
            "program_id": program_id,
            "_nominal_duration": _nominal_for_record(record, nominal_duration),
        }
        for key in extra_keys:
            prepared_record[key] = _json_safe_copy(record[key], name=f"record[{row}][{key!r}]")
        prepared.append(prepared_record)

    means = np.mean(all_features, axis=0)
    scales = np.std(all_features, axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    standardized = (all_features - means[None, :]) / scales[None, :]

    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for index, record in enumerate(prepared):
        grouped.setdefault(str(record["program_id"]), []).append(index)
    programs: OrderedDict[str, _ProgramData] = OrderedDict()
    for program_id, indices in grouped.items():
        rows = tuple(prepared[index] for index in indices)
        ratio = np.zeros(len(rows), dtype=float)
        valid = np.zeros(len(rows), dtype=bool)
        for local, record in enumerate(rows):
            nominal = record["_nominal_duration"]
            if nominal is not None:
                ratio[local] = float(record["duration"]) / nominal
                valid[local] = True
            else:
                ratio[local] = float(record["duration"])
        programs[program_id] = _ProgramData(
            records=rows,
            features=standardized[np.asarray(indices, dtype=int)].copy(),
            duration_ratio=ratio,
            duration_ratio_valid=valid,
        )
    return JointOutcomeModel(
        feature_keys=keys,
        means=means,
        scales=scales,
        programs=programs,
        k=k_value,
        nominal_duration=nominal_duration,
        extra_outcome_keys=extra_keys,
        total_records=len(records),
    )


class EvidencePairTable(dict):
    """Laplace-smoothed empirical table for P(q | y)."""

    def __init__(
        self,
        table: Mapping[Hashable, Mapping[Hashable, float]],
        q_categories: tuple[Hashable, ...],
        alpha: float,
    ) -> None:
        super().__init__({y: dict(distribution) for y, distribution in table.items()})
        self.q_categories = q_categories
        self.alpha = float(alpha)

    def distribution(self, y: Hashable) -> dict[Hashable, float]:
        if y in self:
            return dict(self[y])
        if not self.q_categories:
            return {}
        probability = 1.0 / len(self.q_categories)
        return {q: probability for q in self.q_categories}

    def probability(self, y: Hashable, q: Hashable) -> float:
        return float(self.distribution(y).get(q, 0.0))

    def predict(self, y: Hashable) -> dict[Hashable, float]:
        return self.distribution(y)


def fit_evidence_pairs(
    records: list[dict[str, Any]],
    alpha: float = 1.0,
) -> EvidencePairTable:
    """Fit an empirical Laplace-smoothed conditional table from only y and q."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    alpha = _as_float(alpha, name="alpha")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    counts: OrderedDict[Hashable, OrderedDict[Hashable, int]] = OrderedDict()
    q_categories: list[Hashable] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "y" not in record or "q" not in record:
            raise ValueError(f"record {index} must contain y and q")
        y, q = record["y"], record["q"]
        try:
            hash(y)
            hash(q)
        except TypeError as exc:
            raise ValueError("y and q categories must be hashable") from exc
        if q not in q_categories:
            q_categories.append(q)
        row = counts.setdefault(y, OrderedDict())
        row[q] = row.get(q, 0) + 1
    q_tuple = tuple(q_categories)
    table: OrderedDict[Hashable, dict[Hashable, float]] = OrderedDict()
    for y, row in counts.items():
        total = sum(row.values())
        denominator = total + alpha * len(q_tuple)
        table[y] = {
            q: (float(row.get(q, 0)) + alpha) / denominator
            for q in q_tuple
        }
    return EvidencePairTable(table, q_tuple, alpha)


def common_policy_mixture(A: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Solve max_lambda min_i A[i] @ lambda for one common policy mixture.

    Rows of A are models and columns are policies. The returned upper value is
    the simple global max(A) bound requested by the finite audit; it is not a
    per-model policy oracle.
    """
    values = np.asarray(A, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("A must be a nonempty 2-D model-by-policy matrix")
    if not np.isfinite(values).all():
        raise ValueError("A must contain only finite values")
    n_models, n_policies = values.shape
    objective = np.concatenate([np.zeros(n_policies), np.asarray([-1.0])])
    inequalities = np.hstack([-values, np.ones((n_models, 1), dtype=float)])
    equality = np.concatenate([np.ones(n_policies), np.asarray([0.0])])[None, :]
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(n_models, dtype=float),
        A_eq=equality,
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, None)] * n_policies + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"common policy LP failed: {result.message}")
    mixture = np.asarray(result.x[:n_policies], dtype=float)
    mixture[np.abs(mixture) < 1.0e-10] = 0.0
    mixture /= float(mixture.sum())
    lower = float(np.min(values @ mixture))
    upper = float(np.max(values))
    return mixture, lower, upper


__all__ = [
    "EvidencePairTable",
    "JointOutcomeModel",
    "common_policy_mixture",
    "fit_evidence_pairs",
    "fit_joint",
]
