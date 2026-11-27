"""Finite public-history policies for the paired Fisher experiment.

The experiment deliberately keeps the native transition catalogue at a hard
boundary.  ``TransitionCatalog`` contains checked native events and two action
descriptions: ``public_action`` is the measured-frame action that may cross
into this module, while ``private_action`` is retained by the runner for
acoustic rendering/evaluation.  No scenario, physical frame, hidden frame, or
terminal truth is imported here.

Four pieces are shared by all policy arms:

* the corrected closed-loop acoustic association model
  (``MixtureAddressMemory`` and its exact Gauss--Hermite outcome tree),
* the native prefix/validity catalogue,
* the terminal utility and public stop rule, and
* equal caps of two diagnostic opportunities and two terminal attempts.

``B_original_bank`` is the original task-value look-ahead evaluated on the
original bank.  ``B_plain_topK`` and ``B_fisher_topK`` use the same public
stage-1-valid source bank and equal stagewise K budget; Fisher changes only
the candidate shortlist, never the task objective.  ``B_logdet`` is the
separate pure log-determinant proposal with the same task-VOI admission gate.
The observed log-determinant reported after every real packet is computed
from the full association mixture posterior, rather than from a single
Gaussian or an expected pre-observation covariance.

This module is intentionally finite and auditable.  It does not claim global
optimality and has no task-time deadline.  Actual elapsed native time enters
the declared soft cost ``lambda_time`` only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# The reviewed sibling is the sole source for association, rendering helpers,
# and public continuation rules.  Import by path order so a caller that has
# already imported another experiment cannot silently shadow these modules.
SIBLING = Path(__file__).resolve().parents[1] / "ab_closed_loop_reviewed_20261111"
if str(SIBLING) not in sys.path:
    sys.path.insert(0, str(SIBLING))
from acoustics import (  # type: ignore  # noqa: E402
    MixtureAddressMemory,
    P_CLUTTER,
    ProbeAction,
    PublicObservation,
    condition,
    detection_probability,
    geometry_value,
    response_design,
    scan_probability,
    sensor_variance,
)
try:  # Optional vectorized public retry forecast supplied with this experiment.
    from forecast_fast import action_value as _fast_terminal_action_value  # type: ignore
except ImportError:  # pragma: no cover - fallback keeps the module standalone
    _fast_terminal_action_value = None


TIME_COST = 0.04
DEFAULT_TOP_K = 4
DIAGNOSTIC_CAP = 2
TERMINAL_CAP = 2
FALSE_STOP_CONFIDENCE = 0.65
QUALITY_THRESHOLD = 0.58
ORIGINAL_BASES = ("side_left", "side_right", "fork_minus", "fork_plus")
TERMINALS = ("left", "right")

# Public mechanics are deterministic for a layout and prior.  These caches
# are shared across paired arms and cases; private scenarios never enter the
# keys or values.
_SHARED_OUTCOME_CACHE: dict[tuple[Any, ...], tuple[dict[str, Any], ...]] = {}
_SHARED_TERMINAL_CACHE: dict[tuple[Any, ...], float] = {}


def _json_safe(value: Any) -> Any:
    """Return strict JSON values without erasing the public/private boundary."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _normalise_prefix(prefix: Any) -> tuple[str, ...]:
    if prefix is None or prefix == "" or prefix == "/":
        return ()
    if isinstance(prefix, str):
        return tuple(part for part in prefix.strip("/").split("/") if part)
    return tuple(str(part) for part in prefix)


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _action_from_mapping(payload: Mapping[str, Any] | ProbeAction | None,
                         *, action_id: str, default_kind: str | None = None,
                         default_duration: float = 0.15) -> ProbeAction:
    """Build a public/private action from the few names used by native workers."""

    if isinstance(payload, ProbeAction):
        return payload
    payload = {} if payload is None else dict(payload)
    xy = _first(payload, "xy_m", "xy", "position_m", "position", "goal_local",
                "goal", default=(0.0, 0.0, 0.0))
    if isinstance(xy, Mapping):
        xy = _first(xy, "xy_m", "xy", "position", default=(0.0, 0.0))
    values = list(xy) if isinstance(xy, (tuple, list, np.ndarray)) else [xy]
    while len(values) < 3:
        values.append(0.0)
    window = str(_first(payload, "window", "window_id", "aperture", default="diagnostic_mouth"))
    yaw = _first(payload, "yaw_rad", "tilt_rad", "yaw", "tilt", default=values[2])
    duration = _first(payload, "duration_s", "duration", "dwell_s", "dwell", default=default_duration)
    kind = str(_first(payload, "kind", "type", default=default_kind or
                       ("terminal" if action_id in TERMINALS else "probe")))
    return ProbeAction(window, (_coerce_float(values[0]), _coerce_float(values[1])),
                       _coerce_float(yaw), _coerce_float(duration, default_duration), kind)


@dataclass(frozen=True)
class Transition:
    """One checked native transition at a complete public prefix."""

    prefix: tuple[str, ...]
    action_id: str
    valid: bool
    duration_s: float
    elapsed_s: float
    route_id: str | None
    public_action: ProbeAction
    private_action: ProbeAction
    output: Mapping[str, Any] = field(default_factory=dict)
    reason: str = "checked"

    @property
    def next_prefix(self) -> tuple[str, ...]:
        return self.prefix + (self.action_id,)

    @property
    def kind(self) -> str:
        return "terminal" if self.action_id in TERMINALS or self.public_action.kind == "terminal" else "probe"

    def public_dict(self) -> dict[str, Any]:
        return {
            "prefix": list(self.prefix), "action_id": self.action_id,
            "valid": bool(self.valid), "duration_s": float(self.duration_s),
            "elapsed_s": float(self.elapsed_s), "route_id": self.route_id,
            "public_action": _json_safe(asdict(self.public_action)),
            "output": _json_safe(self.output), "reason": self.reason,
        }


def _row_to_transition(row: Mapping[str, Any], *, prefix_hint: Any = None,
                       action_hint: str | None = None) -> Transition:
    prefix = _normalise_prefix(_first(row, "prefix", "sequence", "state_prefix", "history",
                                      "key", default=prefix_hint))
    action_id = str(_first(row, "action_id", "capture", "action", "name",
                           default=action_hint))
    if not action_id or action_id == "None":
        raise ValueError("native transition is missing action_id")
    valid = bool(_first(row, "valid", "feasible", "exit_state_valid", default=False))
    duration = _coerce_float(_first(row, "duration_s", "actual_elapsed_s", "duration", "cost_s", default=0.0))
    output = _first(row, "output", "output_state", "next_state", "state_out", default={})
    if not isinstance(output, Mapping):
        output = {"value": _json_safe(output)}
    elapsed = _coerce_float(_first(row, "elapsed_s", "actual_exit_elapsed_s", "cumulative_elapsed_s", "clock_s",
                                   default=_first(output, "elapsed_s", "elapsed", default=duration)))
    route_id = _first(row, "route_id", "selected_route_id", "route", "native_route_id", default=None)
    public_payload = _first(row, "public_action", "action_public", "measured_action",
                            "public", default=None)
    if isinstance(public_payload, Mapping) and "planning_action" in public_payload:
        public_payload = public_payload["planning_action"]
    # Some native workers use one top-level action dict and then a private
    # action.  Treat that single dict as public only; no private data reaches
    # this module through any planner API.
    if isinstance(public_payload, Mapping) and "action" in public_payload and not any(
            key in public_payload for key in ("window", "xy_m", "goal_local", "yaw_rad")):
        public_payload = public_payload["action"]
    if bool(valid) and public_payload is None:
        raise ValueError("valid native transition is missing public.planning_action")
    private_payload = _first(row, "private_action", "action_private", "physical_action",
                             "actual_action", default=None)
    evaluation_payload = _first(row, "evaluation", "private", default=None)
    if private_payload is None and isinstance(evaluation_payload, Mapping):
        private_payload = _first(evaluation_payload, "true_surface_action", "physical_action",
                                 "action", default=None)
    if bool(valid) and private_payload is None:
        raise ValueError("valid native transition is missing evaluation.true_surface_action")
    default_kind = "terminal" if action_id in TERMINALS else "probe"
    public_action = _action_from_mapping(public_payload, action_id=action_id,
                                         default_kind=default_kind,
                                         default_duration=duration or 0.15)
    private_action = _action_from_mapping(private_payload, action_id=action_id,
                                          default_kind=default_kind,
                                          default_duration=public_action.duration_s)
    if duration <= 0.0:
        duration = float(public_action.duration_s)
    return Transition(prefix, action_id, valid, duration, elapsed, None if route_id is None else str(route_id),
                      public_action, private_action, dict(output), str(_first(row, "reason", default="checked")))


def _parse_transition_key(value: Any) -> tuple[tuple[str, ...], str | None]:
    """Decode native worker keys such as ``fork_minus>fork_plus|capture=left``."""

    text = str(value)
    if "|capture=" in text:
        prefix_text, action = text.split("|capture=", 1)
        prefix_text = "" if prefix_text == "START" else prefix_text
        separator = ">" if ">" in prefix_text else "/"
        prefix = tuple(part for part in prefix_text.split(separator) if part)
        return prefix, action or None
    return _normalise_prefix(value), None


def _iter_rows(payload: Any, *, prefix_hint: Any = None,
               action_hint: str | None = None) -> Iterable[Mapping[str, Any]]:
    """Extract rows from both flat and prefix-indexed worker JSON layouts."""

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if not isinstance(payload, Mapping):
        return
    for key in ("transitions", "records", "events", "rows", "entries"):
        if key in payload:
            value = payload[key]
            if isinstance(value, Mapping):
                for pref, actions in value.items():
                    if isinstance(actions, Mapping) and not any(k in actions for k in ("action_id", "valid", "feasible")):
                        for aid, item in actions.items():
                            if isinstance(item, Mapping):
                                row = dict(item)
                                parsed_prefix, parsed_action = _parse_transition_key(pref)
                                row.setdefault("prefix", parsed_prefix)
                                row.setdefault("action_id", parsed_action or aid)
                                yield row
                    elif isinstance(actions, Mapping):
                        row = dict(actions)
                        parsed_prefix, parsed_action = _parse_transition_key(pref)
                        row.setdefault("prefix", parsed_prefix)
                        row.setdefault("action_id", parsed_action)
                        yield row
            else:
                yield from _iter_rows(value, prefix_hint=prefix_hint, action_hint=action_hint)
            return
    if any(key in payload for key in ("action_id", "action", "capture", "name")):
        row = dict(payload)
        if prefix_hint is not None and not any(k in row for k in ("prefix", "state_prefix", "history", "key")):
            row["prefix"] = prefix_hint
        if action_hint is not None and not any(k in row for k in ("action_id", "action", "capture", "name")):
            row["action_id"] = action_hint
        yield row
        return
    # A compact worker form may be {"prefix": {"action": row, ...}, ...}.
    for key, value in payload.items():
        if key in {"metadata", "schema", "version", "fixture", "manifest"}:
            continue
        if isinstance(value, Mapping):
            if any(k in value for k in ("valid", "feasible", "duration_s", "output")):
                row = dict(value)
                row.setdefault("prefix", prefix_hint if prefix_hint is not None else "")
                row.setdefault("action_id", key)
                yield row
            else:
                yield from _iter_rows(value, prefix_hint=key, action_hint=None)


class TransitionCatalog:
    """Immutable checked native events keyed by the full actual prefix."""

    def __init__(self, transitions: Iterable[Transition], metadata: Mapping[str, Any] | None = None):
        self.metadata = dict(metadata or {})
        self._events: dict[tuple[tuple[str, ...], str], Transition] = {}
        for event in transitions:
            self._events[(event.prefix, event.action_id)] = event
        self._by_prefix: dict[tuple[str, ...], list[Transition]] = {}
        for event in self._events.values():
            self._by_prefix.setdefault(event.prefix, []).append(event)
        for rows in self._by_prefix.values():
            rows.sort(key=lambda event: (event.duration_s, event.action_id))
        self.cache_key = hashlib.sha256(json.dumps(
            [event.public_dict() for event in sorted(self._events.values(),
                                                     key=lambda row: (row.prefix, row.action_id))],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        declared_pair = tuple(self.metadata.get("fixed_pair", ()))
        calibrated_pair = self._calibrated_fixed_pair()
        self.fixed_pair = calibrated_pair or (declared_pair if len(declared_pair) == 2
                                              else self._infer_fixed_pair())
        self.shift_pair = tuple(self.metadata.get("shift_pair", self.metadata.get("fixed_pair_shift", ())))
        if len(self.shift_pair) != 2:
            self.shift_pair = ()

    @classmethod
    def from_json(cls, path: str | Path, *, index: int | None = None,
                  mode: str | None = None) -> "TransitionCatalog":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata: dict[str, Any] = {}
        if isinstance(payload, Mapping):
            metadata.update(dict(payload.get("metadata", {})))
            metadata.update({key: payload[key] for key in ("fixed_pair", "shift_pair", "fixed_pair_shift") if key in payload})
            # Allow a single file to carry both mirror layouts without making
            # the runner inspect private fields.
            fixtures = payload.get("fixtures")
            if isinstance(fixtures, list):
                chosen = None
                for item in fixtures:
                    if not isinstance(item, Mapping):
                        continue
                    if index is not None and item.get("index") == index:
                        if mode is None or item.get("mode") == mode:
                            chosen = item
                            break
                    if chosen is None and mode is not None and item.get("mode") == mode:
                        chosen = item
                if chosen is not None:
                    payload = chosen
                    metadata.update(dict(chosen.get("metadata", {})))
        transitions = []
        for row in _iter_rows(payload):
            try:
                transitions.append(_row_to_transition(row))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid native transition row in {path}: {exc}") from exc
        if not transitions:
            raise ValueError(f"no native transitions found in {path}")
        # Preserve obvious fixture-level fields when metadata was omitted.
        if isinstance(payload, Mapping):
            fixture = payload.get("fixture", {})
            if isinstance(fixture, Mapping):
                for key in ("fixed_pair", "shift_pair", "fixed_pair_shift"):
                    if key in fixture and key not in metadata:
                        metadata[key] = fixture[key]
        return cls(transitions, metadata)

    def _calibrated_fixed_pair(self) -> tuple[str, str] | None:
        candidates = [event.action_id for event in self._by_prefix.get((), ())
                      if event.valid and event.kind == "probe" and event.action_id.startswith("fork_")]
        complete: list[tuple[float, str, str]] = []
        for first in candidates:
            first_event = self.get((), first)
            if first_event is None or not first_event.valid:
                continue
            for second in candidates:
                if second == first:
                    continue
                second_event = self.get((first,), second)
                if second_event is None or not second_event.valid:
                    continue
                terminal_rows = self.options((first, second), kind="terminal")
                if len(terminal_rows) < 2:
                    continue
                worst = max(float(row.elapsed_s or (second_event.elapsed_s + row.duration_s))
                            for row in terminal_rows)
                complete.append((worst, first, second))
        if complete:
            _worst, first, second = min(complete, key=lambda item: (item[0], item[1], item[2]))
            return (first, second)
        return None

    def _infer_fixed_pair(self) -> tuple[str, str]:
        candidates = [event.action_id for event in self._by_prefix.get((), ())
                      if event.valid and event.kind == "probe" and event.action_id.startswith("fork_")]
        if "fork_plus" in candidates and "fork_minus" in candidates:
            return ("fork_plus", "fork_minus")
        if len(candidates) >= 2:
            return tuple(candidates[:2])  # type: ignore[return-value]
        return ("fork_minus", "fork_plus")

    @property
    def prefixes(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._by_prefix)

    @property
    def events(self) -> tuple[Transition, ...]:
        return tuple(self._events.values())

    def get(self, prefix: Sequence[str], action_id: str) -> Transition | None:
        return self._events.get((tuple(prefix), str(action_id)))

    def options(self, prefix: Sequence[str], *, kind: str | None = None,
                ids: Sequence[str] | None = None) -> tuple[Transition, ...]:
        allowed = None if ids is None else set(str(item) for item in ids)
        rows = self._by_prefix.get(tuple(prefix), ())
        return tuple(row for row in rows if row.valid and (kind is None or row.kind == kind)
                     and (allowed is None or row.action_id in allowed))

    def stage1_valid(self, *, original_only: bool = False) -> tuple[Transition, ...]:
        rows = self.options((), kind="probe")
        if original_only:
            rows = tuple(row for row in rows if _base_action_id(row.action_id) in ORIGINAL_BASES)
        return rows

    def original_ids(self) -> tuple[str, ...]:
        declared = self.metadata.get("original_candidate_ids", self.metadata.get("original_ids"))
        if isinstance(declared, (tuple, list)):
            ids = tuple(str(item) for item in declared)
            return tuple(item for item in ids if self.get((), item) is not None)
        return tuple(row.action_id for row in self.stage1_valid(original_only=True))

    def candidate_bank(self, *, top_k: int = DEFAULT_TOP_K,
                       common: bool = False) -> tuple[str, ...]:
        rows = self.stage1_valid(original_only=not common)
        rows = tuple(sorted(rows, key=lambda row: (row.duration_s, row.action_id)))
        if common:
            return tuple(row.action_id for row in rows)
        return tuple(row.action_id for row in rows[:max(0, int(top_k))])

    def public_manifest(self) -> dict[str, Any]:
        return {
            "metadata": _json_safe(self.metadata),
            "fixed_pair": list(self.fixed_pair),
            "shift_pair": list(self.shift_pair),
            "transition_count": len(self._events),
            "prefix_count": len(self._by_prefix),
            "stage1_valid": [row.public_dict() for row in self.stage1_valid()],
        }

    def public_view(self) -> "TransitionCatalog":
        """Return a planner-safe catalogue with private actions removed."""

        rows = [Transition(
            prefix=row.prefix, action_id=row.action_id, valid=row.valid,
            duration_s=row.duration_s, elapsed_s=row.elapsed_s, route_id=row.route_id,
            public_action=row.public_action, private_action=row.public_action,
            output=row.output, reason=row.reason)
                for row in self._events.values()]
        return TransitionCatalog(rows, self.metadata)


def _base_action_id(action_id: str) -> str:
    text = str(action_id)
    for sep in ("@", ":", "_"):
        if sep == "_" and text.startswith("fork_"):
            continue
        if sep in text:
            head, tail = text.rsplit(sep, 1)
            try:
                float(tail)
                return head
            except ValueError:
                pass
    return text


def public_stop(packet: PublicObservation) -> bool:
    return bool(packet.detected and packet.confidence >= FALSE_STOP_CONFIDENCE
                and packet.quality >= QUALITY_THRESHOLD)


def _event_for_prediction(action: ProbeAction, event_id: str, outcome: Mapping[str, Any]) -> PublicObservation:
    return PublicObservation(
        event_id=str(event_id), address="predicted_patch", window=action.window,
        elapsed_s=0.0, exit_state="predicted", detected=bool(outcome.get("detected", False)),
        confidence=_coerce_float(outcome.get("confidence"), 0.0),
        quality=_coerce_float(outcome.get("quality"), float("nan")) if outcome.get("detected") else float("nan"),
        center_offset_m=float("nan"), axis_angle_rad=float("nan"),
        candidate_type=str(outcome.get("candidate_type", "missing")),
    )


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(value))))))


def _predict_outcomes_order(action: ProbeAction, memory: MixtureAddressMemory,
                            *, order: int, informative: bool = True) -> tuple[dict[str, Any], ...]:
    """Public outcome tree with an explicit GH order for convergence audits."""

    if not informative:
        return ({"probability": 0.0, "detected": True, "candidate_type": "candidate",
                 "quality": 0.0, "confidence": 0.0},
                {"probability": 1.0, "detected": False, "candidate_type": "missing",
                 "quality": float("nan"), "confidence": 0.0})
    H = response_design(action)
    variance = sensor_variance(action)
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    weights = weights / math.sqrt(math.pi)
    outcomes: list[dict[str, Any]] = []
    p_real = float(detection_probability(action, informative=True))
    p_clutter = float((1.0 - p_real) * P_CLUTTER)
    p_missing = float((1.0 - p_real) * (1.0 - P_CLUTTER))
    for component_weight, component in zip(memory.weights, memory.components):
        mean, predictive_variance = component.predict(
            H, geometry_value_=geometry_value(action), observation_variance=variance)
        scale = math.sqrt(max(float(predictive_variance), 1.0e-12))
        for node, node_weight in zip(nodes, weights):
            quality = float(mean + math.sqrt(2.0) * scale * node)
            outcomes.append({"probability": float(component_weight * p_real * node_weight),
                             "detected": True, "candidate_type": "candidate",
                             "quality": quality,
                             "confidence": _sigmoid((quality - 0.34) / 0.14)})
        for node, node_weight in zip(nodes, weights):
            quality = float(0.12 + math.sqrt(2.0) * 0.08 * node)
            outcomes.append({"probability": float(component_weight * p_clutter * node_weight),
                             "detected": True, "candidate_type": "candidate",
                             "quality": quality,
                             "confidence": _sigmoid((quality - 0.34) / 0.14)})
        outcomes.append({"probability": float(component_weight * p_missing),
                         "detected": False, "candidate_type": "missing",
                         "quality": float("nan"), "confidence": 0.0})
    total = float(sum(item["probability"] for item in outcomes))
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("non-finite public outcome mass")
    outcomes[-1]["probability"] += 1.0 - total
    return tuple(outcomes)


def _belief_logdet(memory: MixtureAddressMemory) -> float:
    _mean, covariance = memory.posterior()
    covariance = np.asarray(covariance, dtype=float)
    covariance = covariance + np.eye(covariance.shape[0]) * 1.0e-15
    sign, value = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def observed_logdet_gain(before: MixtureAddressMemory, after: MixtureAddressMemory) -> float:
    """Logdet reduction for the *observed mixture posterior* (nats)."""

    before_value = _belief_logdet(before)
    after_value = _belief_logdet(after)
    return float(before_value - after_value) if math.isfinite(before_value) and math.isfinite(after_value) else float("nan")


def _belief_fingerprint(memory: MixtureAddressMemory) -> str:
    # ``snapshot()`` intentionally exposes only aggregate moments.  Aggregate
    # moments are insufficient for an association mixture: two mixtures can
    # share mean/covariance while having different future likelihoods.  Cache
    # keys therefore retain every public component's natural parameters and
    # association weight.
    payload = {
        "weights": np.asarray(memory.weights, dtype=float).tolist(),
        "components": [
            {"precision": np.asarray(component.precision, dtype=float).tolist(),
             "information": np.asarray(component.information, dtype=float).tolist(),
             "events": sorted(component.event_ids)}
            for component in memory.components
        ],
        "events": sorted(memory.event_ids),
    }
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PolicyConfig:
    """Shared finite protocol knobs; none is a task-time deadline."""

    method: str = "B_plain_topK"
    top_k: int = DEFAULT_TOP_K
    diagnostic_cap: int = DIAGNOSTIC_CAP
    terminal_cap: int = TERMINAL_CAP
    lambda_time: float = TIME_COST
    outcome_order: int = 9
    protocol_revision: str = "fisher_adaptive_event_v1"

    def __post_init__(self) -> None:
        allowed = {"A_greedy", "fixed_pair", "fixed_pair_shift", "B_original_bank",
                   "B_plain_topK", "B_logdet", "B_fisher_topK", "B_common_full"}
        if self.method not in allowed:
            raise ValueError(f"unknown policy method {self.method!r}")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be positive")
        if int(self.diagnostic_cap) != 2 or int(self.terminal_cap) != 2:
            raise ValueError("protocol fixes both caps at two for all arms")
        if int(self.outcome_order) != self.outcome_order or int(self.outcome_order) < 3:
            raise ValueError("outcome_order must be an integer >=3")
        if float(self.lambda_time) < 0.0 or not math.isfinite(float(self.lambda_time)):
            raise ValueError("lambda_time must be finite and nonnegative")


@dataclass(frozen=True)
class Decision:
    kind: str  # probe, commit, exit, no_valid_scan
    action_id: str | None
    score: float
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "action_id": self.action_id,
            "score": float(self.score), "reason": self.reason,
            "diagnostics": _json_safe(self.diagnostics),
        }


class AdaptivePlanner:
    """Public-history planner shared by all paired policy arms."""

    def __init__(self, catalog: TransitionCatalog, config: PolicyConfig):
        self.catalog = catalog
        self.config = config
        self.memory = MixtureAddressMemory()
        self.diagnostic_count = 0
        self.terminal_count = 0
        self.formal_started = False
        self.elapsed_s = 0.0
        self.previous_terminal: str | None = None
        self.events_seen: set[str] = set()
        self.observed_logdet: list[float] = []
        self.value_calls = 0
        self.fisher_calls = 0
        self.outcome_tree_calls = 0
        self._outcome_cache: dict[tuple[Any, ...], tuple[dict[str, Any], ...]] = {}
        self._value_cache: dict[tuple[Any, ...], float] = {}

    def _bank(self) -> tuple[str, ...]:
        if self.config.method == "B_common_full":
            return self.catalog.candidate_bank(top_k=self.config.top_k, common=True)
        if self.config.method == "B_original_bank":
            ids = self.catalog.original_ids()
            return tuple(ids[:self.config.top_k])
        # Plain/Fisher share the same stage-1-valid public source bank.  Their
        # stagewise K-shortlist ordering differs below (public duration versus
        # the task-Fisher comparator), while the task objective is identical.
        return self.catalog.candidate_bank(top_k=self.config.top_k, common=True)

    def bank_metadata(self) -> dict[str, Any]:
        bank = self._bank()
        return {
            "method": self.config.method, "candidate_bank": list(bank),
            "candidate_bank_size": len(bank), "top_k": int(self.config.top_k),
            "bank_rule": ("stage1_valid_common_source; stagewise_public_shortest_K"
                           if self.config.method == "B_plain_topK" else
                           "stage1_valid_common_source; stagewise_task_fisher_K"
                           if self.config.method == "B_fisher_topK" else
                           "stage1_valid_common_source; stagewise_logdet_K"
                           if self.config.method == "B_logdet" else
                           "stage1_valid_public_bank"),
            "candidate_bank_common": self.config.method in {"B_plain_topK", "B_logdet", "B_fisher_topK"},
        }

    def _terminal_stop_probability(self, action: ProbeAction, memory: MixtureAddressMemory) -> float:
        outcomes = self._outcomes(action, memory)
        return float(sum(float(item["probability"]) for item in outcomes
                         if bool(item.get("detected")) and
                         _coerce_float(item.get("confidence"), 0.0) >= FALSE_STOP_CONFIDENCE and
                         _coerce_float(item.get("quality"), -math.inf) >= QUALITY_THRESHOLD))

    def _outcomes(self, action: ProbeAction, memory: MixtureAddressMemory,
                  *, informative: bool = True) -> tuple[dict[str, Any], ...]:
        key = (self.catalog.cache_key, _belief_fingerprint(memory), repr(asdict(action)),
               bool(informative), int(self.config.outcome_order))
        shared = _SHARED_OUTCOME_CACHE.get(key)
        if shared is not None:
            self._outcome_cache[key] = shared
            return shared
        cached = self._outcome_cache.get(key)
        if cached is not None:
            return cached
        outcomes = _predict_outcomes_order(action, memory, order=self.config.outcome_order,
                                           informative=informative)
        self.outcome_tree_calls += 1
        total = float(sum(float(item.get("probability", 0.0)) for item in outcomes))
        if not math.isfinite(total) or abs(total - 1.0) > 1.0e-8:
            raise FloatingPointError(f"public outcome mass={total!r}")
        self._outcome_cache[key] = outcomes
        _SHARED_OUTCOME_CACHE[key] = outcomes
        return outcomes

    def _condition_prediction(self, memory: MixtureAddressMemory, action: ProbeAction,
                              outcome: Mapping[str, Any], event_id: str) -> MixtureAddressMemory:
        posterior = deepcopy(memory)
        condition(posterior, action, _event_for_prediction(action, event_id, outcome),
                  geometry_value_=geometry_value(action))
        return posterior

    def _terminal_options(self, prefix: Sequence[str], memory: MixtureAddressMemory) -> tuple[Transition, ...]:
        return self.catalog.options(prefix, kind="terminal")

    def _terminal_action_value(self, prefix: Sequence[str], memory: MixtureAddressMemory,
                               transition: Transition, attempts: int,
                               *, previous: str | None = None) -> float:
        """Public task value for one explicitly selected terminal action."""

        if attempts <= 0:
            return 0.0
        if _fast_terminal_action_value is not None:
            retry_options = ()
            if attempts > 1:
                retry_options = tuple((row.public_action, row.duration_s)
                                      for row in self._terminal_options(transition.next_prefix, memory))
            return float(_fast_terminal_action_value(
                transition.public_action, transition.duration_s, retry_options,
                memory, self.config.lambda_time, order=16))
        p_success = float(scan_probability(
            transition.public_action, memory,
            geometry_value_=geometry_value(transition.public_action), informative=True))
        continuation = 0.0
        if attempts > 1:
            for outcome in self._outcomes(transition.public_action, memory):
                probability = float(outcome.get("probability", 0.0))
                if probability < 1.0e-10:
                    continue
                stop = bool(outcome.get("detected")) and \
                    _coerce_float(outcome.get("confidence"), 0.0) >= FALSE_STOP_CONFIDENCE and \
                    _coerce_float(outcome.get("quality"), -math.inf) >= QUALITY_THRESHOLD
                if stop:
                    continue
                posterior = self._condition_prediction(
                    memory, transition.public_action, outcome,
                    f"forecast/{transition.action_id}/{attempts}")
                continuation += probability * self._terminal_utility(
                    transition.next_prefix, posterior, attempts - 1,
                    previous=transition.action_id)
        return float(min(1.0, p_success + continuation)
                     - self.config.lambda_time * transition.duration_s)

    def _terminal_utility(self, prefix: Sequence[str], memory: MixtureAddressMemory,
                          attempts: int, *, previous: str | None = None) -> float:
        """Expected public-model task value for up to ``attempts`` terminals.

        This is a planning forecast only.  It uses ``scan_probability`` and
        public predicted packet branches; no terminal truth, private action,
        or hidden frame is available here.
        """

        if attempts <= 0:
            return 0.0
        key = (self.catalog.cache_key, tuple(prefix), _belief_fingerprint(memory), int(attempts), previous,
               round(float(self.config.lambda_time), 12), int(self.config.outcome_order))
        shared = _SHARED_TERMINAL_CACHE.get(key)
        if shared is not None:
            self._value_cache[key] = shared
            return float(shared)
        if key in self._value_cache:
            return self._value_cache[key]
        options = self._terminal_options(prefix, memory)
        if not options:
            self._value_cache[key] = 0.0
            return 0.0
        values = [(self._terminal_action_value(prefix, memory, transition, attempts,
                                                previous=previous), transition.action_id)
                  for transition in options]
        value = max(values, key=lambda item: (item[0], item[1] != previous, item[1]))[0]
        self._value_cache[key] = float(value)
        _SHARED_TERMINAL_CACHE[key] = float(value)
        return float(value)

    @staticmethod
    def _normal_pdf(value: np.ndarray | float, mean: float, variance: float) -> np.ndarray:
        variance = max(float(variance), 1.0e-12)
        return np.exp(-0.5 * (np.asarray(value, dtype=float) - float(mean)) ** 2 / variance) \
            / math.sqrt(2.0 * math.pi * variance)

    def _observed_fisher_kappa(self, action: ProbeAction,
                               memory: MixtureAddressMemory) -> float:
        """Marginal-observation scalar Fisher information for one action.

        The public mixture is represented by its current aggregate predictive
        mean (the declared approximation in this protocol).  The target and
        clutter association branches are still explicitly marginalized in the
        denominator.  Missing packets carry zero response information.
        """

        H = response_design(action)
        R = max(float(sensor_variance(action)), 1.0e-12)
        mu, _predictive_variance = memory.predict(
            H, geometry_value_=geometry_value(action), observation_variance=R)
        p_target = float(detection_probability(action, informative=True))
        p_clutter = float((1.0 - p_target) * P_CLUTTER)
        if p_target <= 1.0e-15:
            return 0.0
        # A 33-node Hermite integral is materially tighter than the 5/3-node
        # outcome tree used for the planner's branch forecast.  The integral
        # itself is deterministic and included in every public diagnostic.
        nodes, weights = np.polynomial.hermite.hermgauss(33)
        weights = weights / math.sqrt(math.pi)

        target_y = float(mu) + math.sqrt(2.0 * R) * nodes
        phi_target = self._normal_pdf(target_y, mu, R)
        phi_clutter = self._normal_pdf(target_y, 0.12, 0.08 ** 2)
        denominator = p_target * phi_target + p_clutter * phi_clutter
        # Under target y~N(mu,R), score^2 = 2 z^2/R.  The association
        # responsibility r_T(y) is the only clutter correction; missing mass
        # has zero derivative.  This is equivalent to integrating
        # (pT phiT)^2/(pT phiT+pF phiF) * (y-mu)^2/R^2 over y, without
        # accidentally multiplying the target/clutter density twice.
        responsibility = np.divide(p_target * phi_target,
                                   np.maximum(denominator, 1.0e-300))
        kappa = (p_target / R) * float(np.sum(weights * (2.0 * nodes ** 2) * responsibility))
        return float(max(0.0, kappa))

    def _task_fisher_shortlist_score(self, prefix: Sequence[str], transition: Transition,
                                     memory: MixtureAddressMemory) -> tuple[float, dict[str, float]]:
        """Return GΔ and the logdet comparator for candidate shortlisting."""

        self.fisher_calls += 1
        terminals = {row.action_id: row for row in self._terminal_options(prefix, memory)}
        if "left" not in terminals or "right" not in terminals:
            return 0.0, {"fisher_kappa": 0.0, "fisher_logdet_gain": 0.0,
                         "task_fisher_score": 0.0}
        H = response_design(transition.public_action)
        H_left = response_design(terminals["left"].public_action)
        H_right = response_design(terminals["right"].public_action)
        d = H_left - H_right
        _mean, C = memory.posterior()
        kappa = self._observed_fisher_kappa(transition.public_action, memory)
        hch = float(H @ C @ H)
        dch = float(d @ C @ H)
        denominator = max(1.0 + kappa * hch, 1.0e-12)
        task_score = kappa * (dch ** 2) / denominator
        logdet_gain = math.log1p(max(0.0, kappa * hch))
        return float(task_score), {"fisher_kappa": float(kappa),
                                  "fisher_hch": hch, "fisher_dch": dch,
                                  "fisher_logdet_gain": float(logdet_gain),
                                  "task_fisher_score": float(task_score),
                                  "fisher_mixture_scope": "global_public_mixture_mean"}

    def _candidate_pool(self, prefix: Sequence[str]) -> tuple[Transition, ...]:
        return self.catalog.options(prefix, kind="probe", ids=self._bank())

    def _shortlist(self, prefix: Sequence[str], memory: MixtureAddressMemory) -> tuple[Transition, ...]:
        pool = self._candidate_pool(prefix)
        if self.config.method == "B_common_full":
            return pool
        if self.config.method == "B_fisher_topK":
            scored = [(self._task_fisher_shortlist_score(prefix, row, memory)[0], row)
                      for row in pool]
            scored.sort(key=lambda item: (-item[0], item[1].duration_s, item[1].action_id))
            return tuple(row for _score, row in scored[:self.config.top_k])
        if self.config.method == "B_logdet":
            scored = [(self._task_fisher_shortlist_score(prefix, row, memory)[1]["fisher_logdet_gain"], row)
                      for row in pool]
            scored.sort(key=lambda item: (-item[0], item[1].duration_s, item[1].action_id))
            return tuple(row for _score, row in scored[:self.config.top_k])
        # The plain control is pre-ordered by public checked duration, with a
        # stable action-id tie break and no information-based hand tuning.
        return tuple(sorted(pool, key=lambda row: (row.duration_s, row.action_id))[:self.config.top_k])

    def _diagnostic_score(self, prefix: Sequence[str], transition: Transition,
                          memory: MixtureAddressMemory, *, diagnostics_left: int,
                          terminals_left: int) -> tuple[float, dict[str, Any]]:
        self.value_calls += 1
        base_value = self._terminal_utility(prefix, memory, terminals_left, previous=self.previous_terminal)
        expected_after = 0.0
        branch_count = 0
        expected_logdet = 0.0
        for outcome in self._outcomes(transition.public_action, memory):
            probability = float(outcome.get("probability", 0.0))
            if probability < 1.0e-10:
                continue
            branch_count += 1
            posterior = self._condition_prediction(memory, transition.public_action, outcome,
                                                   f"forecast/value/{transition.action_id}/{diagnostics_left}")
            expected_after += probability * self._value_at(
                transition.next_prefix, posterior, diagnostics_left - 1, terminals_left)
            before = _belief_logdet(memory)
            after = _belief_logdet(posterior)
            if math.isfinite(before) and math.isfinite(after):
                expected_logdet += probability * (before - after)
        task_voi = expected_after - base_value - self.config.lambda_time * transition.duration_s
        fisher_gain, fisher_diagnostics = self._task_fisher_shortlist_score(prefix, transition, memory)
        # Fisher is a shortlist proposal only.  Both B arms use the identical
        # task VOI and soft elapsed-time objective after shortlisting.
        score = task_voi
        return float(score), {
            "task_voi": float(task_voi), "expected_after": float(expected_after),
            "commit_value_before": float(base_value), "expected_logdet_gain": float(expected_logdet),
            **fisher_diagnostics,
            "branches": int(branch_count), "duration_s": float(transition.duration_s),
        }

    def _value_at(self, prefix: Sequence[str], memory: MixtureAddressMemory,
                  diagnostics_left: int, terminals_left: int) -> float:
        if diagnostics_left < 0:
            return self._terminal_utility(prefix, memory, terminals_left, previous=self.previous_terminal)
        commit = self._terminal_utility(prefix, memory, terminals_left, previous=self.previous_terminal)
        if diagnostics_left == 0:
            return commit
        scored = [(self._diagnostic_score(prefix, transition, memory,
                                          diagnostics_left=diagnostics_left,
                                          terminals_left=terminals_left), transition)
                  for transition in self._shortlist(prefix, memory)]
        admitted = [(score, transition) for (score, _diagnostics), transition in scored
                    if score > 1.0e-10 or not self._terminal_options(prefix, memory)]
        if self.config.method == "B_logdet" and admitted:
            # Pure observed-mixture logdet proposal: task VOI is the common
            # admission/value test, while logdet chooses among admitted
            # actions.  This same rule is used in recursive forecasts.
            selected = max(admitted, key=lambda item: (
                self._task_fisher_shortlist_score(prefix, item[1], memory)[1]["fisher_logdet_gain"],
                -item[1].duration_s, item[1].action_id))
            best = max(commit, commit + selected[0])
        else:
            best = max([commit, *(commit + score for score, _ in admitted)])
        return float(best)

    def choose(self, prefix: Sequence[str]) -> Decision:
        prefix = tuple(prefix)
        terminals = self.catalog.options(prefix, kind="terminal")
        terminal_value = self._terminal_utility(prefix, self.memory,
                                                self.config.terminal_cap - self.terminal_count,
                                                previous=self.previous_terminal)
        # Once a formal terminal scan has begun, only the common checked
        # terminal retry remains admissible.  A diagnostic may not be inserted
        # between terminal attempts, so every arm has the same recovery budget.
        if self.formal_started:
            if not terminals:
                return Decision("no_valid_scan", None, 0.0, "no_valid_scan_after_formal_entry",
                                {**self.bank_metadata(), "candidate_count": 0,
                                 "scored_candidate_count": 0})
            values = [(self._terminal_action_value(
                           prefix, self.memory, row,
                           self.config.terminal_cap - self.terminal_count,
                           previous=self.previous_terminal), row)
                      for row in terminals]
            value, selected = max(values, key=lambda item: (item[0], item[1].action_id != self.previous_terminal, item[1].action_id))
            return Decision("commit", selected.action_id, value, "formal_retry",
                            {**self.bank_metadata(), "candidate_count": len(terminals),
                             "scored_candidate_count": len(values),
                             "formal_entry": True, "terminal_options": [row.action_id for row in terminals]})
        # The fixed pair is a mechanical control.  It does not use acoustic
        # information to reorder or shorten its two diagnostic events.
        if self.config.method in {"fixed_pair", "fixed_pair_shift"} and self.diagnostic_count < self.config.diagnostic_cap:
            pair = self.catalog.shift_pair if self.config.method == "fixed_pair_shift" else self.catalog.fixed_pair
            if self.diagnostic_count < len(pair):
                action_id = str(pair[self.diagnostic_count])
                transition = self.catalog.get(prefix, action_id)
                if transition is not None and transition.valid:
                    return Decision("probe", action_id, 0.0, "fixed_pair_next",
                                    {**self.bank_metadata(), "candidate_count": 1,
                                     "scored_candidate_count": 1, "diagnostic_index": self.diagnostic_count})
            # A checked pair can be unavailable on a fixture; do not invent a
            # route.  Fall through to the common terminal/exit admission.
        if self.config.method == "A_greedy":
            if not terminals:
                return Decision("no_valid_scan", None, 0.0, "no_valid_scan",
                                {**self.bank_metadata(), "candidate_count": 0, "scored_candidate_count": 0})
            values = [(float(scan_probability(row.public_action, self.memory,
                                              geometry_value_=geometry_value(row.public_action), informative=True)
                              - self.config.lambda_time * row.duration_s), row)
                      for row in terminals]
            value, selected = max(values, key=lambda item: (item[0], item[1].action_id != self.previous_terminal, item[1].action_id))
            return Decision("commit", selected.action_id, value, "greedy_terminal",
                            {**self.bank_metadata(), "candidate_count": len(terminals),
                             "scored_candidate_count": len(values), "predicted_success": float(scan_probability(selected.public_action, self.memory, geometry_value_=geometry_value(selected.public_action), informative=True)),
                             "terminal_options": [row.action_id for row in terminals]})
        if self.diagnostic_count >= self.config.diagnostic_cap or not self._bank():
            if terminals:
                values = [(self._terminal_action_value(
                               prefix, self.memory, row,
                               self.config.terminal_cap - self.terminal_count,
                               previous=self.previous_terminal), row)
                          for row in terminals]
                value, selected = max(values, key=lambda item: (item[0], item[1].action_id != self.previous_terminal, item[1].action_id))
                return Decision("commit", selected.action_id, value,
                                "diagnostic_cap_or_empty_bank",
                                {**self.bank_metadata(), "candidate_count": len(terminals),
                                 "scored_candidate_count": len(values), "diagnostic_cap_hit": self.diagnostic_count >= self.config.diagnostic_cap})
            return Decision("no_valid_scan", None, 0.0, "no_valid_scan",
                            {**self.bank_metadata(), "candidate_count": 0, "scored_candidate_count": 0,
                             "diagnostic_cap_hit": self.diagnostic_count >= self.config.diagnostic_cap})
        pool = self._candidate_pool(prefix)
        shortlist = self._shortlist(prefix, self.memory)
        scored = []
        for transition in shortlist:
            score, diagnostics = self._diagnostic_score(
                prefix, transition, self.memory,
                diagnostics_left=self.config.diagnostic_cap - self.diagnostic_count,
                terminals_left=self.config.terminal_cap - self.terminal_count)
            scored.append((score, transition.action_id, diagnostics))
        if scored:
            admitted = [item for item in scored if item[0] > 1.0e-10 or not terminals]
            if self.config.method == "B_logdet" and admitted:
                score, action_id, diagnostics = max(
                    admitted,
                    key=lambda item: (float(item[2].get("fisher_logdet_gain", 0.0)),
                                      -float(item[2].get("duration_s", 0.0)), item[1]))
            else:
                score, action_id, diagnostics = max(scored, key=lambda item: (item[0], item[1]))
            # A positive continuation score is required unless there is no
            # terminal continuation.  This is the original B soft-cost rule,
            # now applied after every interruptible segment.
            if (score > 1.0e-10 or not terminals) and (self.config.method != "B_logdet" or admitted):
                return Decision("probe", action_id, score, "task_voi",
                                {**self.bank_metadata(), "candidate_count": len(pool),
                                 "scored_candidate_count": len(scored),
                                 "shortlist": [row.action_id for row in shortlist],
                                 "best_probe": diagnostics,
                                 "terminal_value": terminal_value,
                                 "candidate_scores": [{"action_id": aid, "score": float(val), **_json_safe(diag)} for val, aid, diag in scored]})
        if terminals:
            values = [(self._terminal_action_value(
                           prefix, self.memory, row,
                           self.config.terminal_cap - self.terminal_count,
                           previous=self.previous_terminal), row)
                      for row in terminals]
            value, selected = max(values, key=lambda item: (item[0], item[1].action_id != self.previous_terminal, item[1].action_id))
            return Decision("commit", selected.action_id, value, "no_positive_continuation",
                            {**self.bank_metadata(), "candidate_count": len(terminals),
                             "scored_candidate_count": len(scored), "shortlist": [row.action_id for row in shortlist],
                             "terminal_value": float(value),
                             "candidate_scores": [{"action_id": aid, "score": float(val), **_json_safe(diag)} for val, aid, diag in scored]})
        if self.diagnostic_count < self.config.diagnostic_cap:
            return Decision("no_valid_scan", None, 0.0, "no_valid_scan",
                            {**self.bank_metadata(), "candidate_count": 0,
                             "scored_candidate_count": len(scored)})
        return Decision("no_valid_scan", None, 0.0, "diagnostic_cap_no_terminal",
                        {**self.bank_metadata(), "candidate_count": 0,
                         "scored_candidate_count": len(scored), "diagnostic_cap_hit": True})

    def observe(self, transition: Transition, packet: PublicObservation, *, event_id: str) -> dict[str, Any]:
        """Assimilate one public packet and return auditable public diagnostics."""

        before = deepcopy(self.memory)
        previous_logdet = _belief_logdet(self.memory)
        accepted = bool(condition(self.memory, transition.public_action, packet,
                                  geometry_value_=geometry_value(transition.public_action),
                                  event_id=event_id))
        gain = observed_logdet_gain(before, self.memory)
        self.observed_logdet.append(gain)
        self.events_seen.add(str(event_id))
        self.elapsed_s += float(transition.duration_s)
        if transition.kind == "probe":
            self.diagnostic_count += 1
        else:
            self.terminal_count += 1
            self.formal_started = True
            self.previous_terminal = transition.action_id
        posterior_mean, posterior_covariance = self.memory.posterior()
        return {
            "conditioned": accepted, "observed_logdet_before": previous_logdet,
            "observed_logdet_after": _belief_logdet(self.memory),
            "observed_logdet_gain": gain, "posterior_mean": posterior_mean.tolist(),
            "posterior_covariance": posterior_covariance.tolist(),
            "mixture_components": len(self.memory.components),
            "mixture_weights": self.memory.weights.tolist(),
        }


__all__ = [
    "AdaptivePlanner", "Decision", "PolicyConfig", "Transition", "TransitionCatalog",
    "TIME_COST", "DIAGNOSTIC_CAP", "TERMINAL_CAP", "FALSE_STOP_CONFIDENCE",
    "QUALITY_THRESHOLD", "public_stop", "observed_logdet_gain", "_json_safe",
]
