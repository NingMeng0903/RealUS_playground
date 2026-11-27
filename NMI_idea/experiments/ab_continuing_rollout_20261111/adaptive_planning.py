"""Public-history adaptive planning for the no-deadline stopping experiment.

This module deliberately has no access to a fixture scenario or to the
``NativeAdapter.physical_frame`` property.  A planner receives a checked
native continuation, turns its measured-frame exit pose into a public
``ProbeAction``, and updates :mod:`belief_model` only with a
``PublicObservation``.  The runner owns the private acoustic renderer and
keeps the resulting terminal truth in a separate ``privateeval`` record.

The implementation is intentionally finite and auditable.  It is a bounded
look-ahead policy (A, B_H1, B_H2, or the frozen pair), not a claim of global
optimality.  There is no task deadline: ``max_probe_frames`` is an
administrative fresh-acquisition watchdog and a guard against an unbounded
simulation, recorded as watchdog/right-censored when reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import ndtr, ndtri, owens_t
from scipy.stats import qmc

try:  # The experiment is also run as a stand-alone directory.
    from acoustics import (AXIS_TOLERANCE_RAD, CENTER_TOLERANCE_M, QUALITY_THRESHOLD,
                           YAW_SCALE_RAD, ProbeAction, PublicObservation)
    from belief_model import (Belief, CompleteOption, CompletePrediction, ModelParams,
                              new_belief, predict_complete_option)
    from fixture import SINGLE_DWELLS, local_pose, single_dwell_action
except ImportError:  # pragma: no cover - package import for test harnesses
    from .acoustics import (AXIS_TOLERANCE_RAD, CENTER_TOLERANCE_M, QUALITY_THRESHOLD,
                            YAW_SCALE_RAD, ProbeAction, PublicObservation)
    from .belief_model import (Belief, CompleteOption, CompletePrediction, ModelParams,
                               new_belief, predict_complete_option)
    from .fixture import SINGLE_DWELLS, local_pose, single_dwell_action


PROBE_BASES = ("side_left", "side_right", "fork_minus", "fork_plus")
TERMINAL_IDS = ("left", "right")
DEFAULT_PROBE_IDS = tuple(single_dwell_action(base, dwell)
                          for base in PROBE_BASES for dwell in SINGLE_DWELLS)

# These seeds are protocol metadata, not random draws from the test object.
TRAIN_SEED = 721501
CALIBRATION_SEED = 721502
TEST_SEED = 721503
NOISE_SEED = 721504


def _json_safe(value: Any) -> Any:
    """Make a log value strict-JSON serialisable without exposing truth."""
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
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict(json_safe=True))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _json_safe(vars(value))
    return value


def _event_fingerprint(event_id: str, action: ProbeAction, packet: PublicObservation) -> str:
    payload = {"event_id": str(event_id), "action": vars(action), "packet": vars(packet)}
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode()).hexdigest()


def _native_payload_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 1.0e-15]
    return float(-np.sum(values * np.log(values))) if len(values) else 0.0


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(value))))))


def _bivariate_standard_normal_cdf(h: np.ndarray, k: np.ndarray,
                                   rho: np.ndarray) -> np.ndarray:
    """Vectorized Phi2(h,k;rho), including the zero/degenerate limits."""
    h, k, rho = (np.asarray(value, dtype=float) for value in (h, k, rho))
    rho = np.clip(rho, -1.0, 1.0)
    result = np.empty(np.broadcast_shapes(h.shape, k.shape, rho.shape), dtype=float)
    h, k, rho = np.broadcast_arrays(h, k, rho)
    result[...] = np.nan
    ph, pk = ndtr(h), ndtr(k)
    near_one = rho >= 1.0 - 1.0e-12
    near_minus_one = rho <= -1.0 + 1.0e-12
    result[near_one] = ndtr(np.minimum(h[near_one], k[near_one]))
    result[near_minus_one] = np.maximum(0.0, ph[near_minus_one] + pk[near_minus_one] - 1.0)
    regular = ~(near_one | near_minus_one)
    if np.any(regular):
        hr, kr, rr = h[regular], k[regular], rho[regular]
        d = np.sqrt(np.maximum(1.0 - rr * rr, 1.0e-15))
        zero_h = np.abs(hr) <= 1.0e-12
        zero_k = np.abs(kr) <= 1.0e-12
        general = ~(zero_h | zero_k)
        values = np.empty_like(hr)
        if np.any(general):
            hg, kg, rg, dg = hr[general], kr[general], rr[general], d[general]
            values[general] = .5 * (ndtr(hg) + ndtr(kg)) \
                - owens_t(hg, (kg / hg - rg) / dg) \
                - owens_t(kg, (hg / kg - rg) / dg) \
                - .25 * (1.0 - np.sign(hg) * np.sign(kg))
        if np.any(zero_h & ~zero_k):
            values[zero_h & ~zero_k] = .5 * ndtr(kr[zero_h & ~zero_k]) \
                + owens_t(kr[zero_h & ~zero_k], rr[zero_h & ~zero_k] / d[zero_h & ~zero_k])
        if np.any(zero_k & ~zero_h):
            values[zero_k & ~zero_h] = .5 * ndtr(hr[zero_k & ~zero_h]) \
                + owens_t(hr[zero_k & ~zero_h], rr[zero_k & ~zero_h] / d[zero_k & ~zero_h])
        if np.any(zero_h & zero_k):
            values[zero_h & zero_k] = .25 + np.arcsin(rr[zero_h & zero_k]) / (2.0 * np.pi)
        result[regular] = values
    return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class PlannerConfig:
    """Frozen public planner parameters.

    ``max_probe_frames`` is explicitly a fresh-acquisition watchdog, never a deadline or a
    time-based stopping criterion.  ``pair_dwell`` is selected on development
    data and must be copied into the test invocation before the test stage.
    """

    arm: str = "B_H1"
    risk_gate: float = 0.90
    max_probe_frames: int = 8
    qmc_initial: int = 16
    qmc_top2: int = 32
    qmc_refine: int = 64
    qmc_unstable: int = 256
    h2_initial: int = 8
    h2_refine: int = 16
    h2_unstable: int = 32
    beam_width: int = 2
    lambda_time: float = 0.0
    pair_dwell: float = 0.30
    pair_order: tuple[str, str] | None = None
    qmc_seed: int = 721505
    protocol_revision: str = "adaptive_stopping_design_pending"

    def __post_init__(self):
        if self.arm not in {"A", "A_task", "A_info", "A_full", "B_H1", "B_H2",
                            "B_rollout", "pair_fixed"}:
            raise ValueError("arm must be A/A_task/A_info, B_H1, B_H2, B_rollout, or pair_fixed")
        if not 0.0 < float(self.risk_gate) < 1.0:
            raise ValueError("risk_gate must be in (0,1)")
        if int(self.max_probe_frames) != self.max_probe_frames or self.max_probe_frames < 1:
            raise ValueError("max_probe_frames must be a positive integer watchdog")
        for name in ("qmc_initial", "qmc_top2", "qmc_refine", "qmc_unstable",
                     "h2_initial", "h2_refine", "h2_unstable"):
            value = int(getattr(self, name))
            if value < 1 or value != getattr(self, name):
                raise ValueError(f"{name} must be a positive integer")
        if self.pair_dwell not in tuple(float(v) for v in SINGLE_DWELLS):
            raise ValueError("pair_dwell must be one of SINGLE_DWELLS")
        if self.pair_order is not None and len(self.pair_order) != 2:
            raise ValueError("pair_order must contain two fork bases")


@dataclass(frozen=True)
class ProbeCandidate:
    action_id: str
    event: Any
    action: ProbeAction
    feasible: bool
    duration_s: float
    # Optional public route proposal.  It is deliberately not a native event
    # or complete state certificate; selected actions still pass checked()
    # from the latest actual state before execution.
    predicted_output: Any | None = None
    prediction_status: str = "predicted_unknown"


@dataclass(frozen=True)
class TerminalOption:
    first_id: str
    second_id: str | None
    first_event: Any
    second_event: Any | None
    option: CompleteOption
    prediction: CompletePrediction
    first_action: ProbeAction
    second_action: ProbeAction | None

    @property
    def allowed(self) -> bool:
        return bool(self.prediction.admitted)

    @property
    def score(self) -> float:
        """Calibrated score used after passing the common risk gate."""
        return float(self.prediction.calibrated_p)


@dataclass(frozen=True)
class ProbeScore:
    candidate: ProbeCandidate
    value: float
    expected_allowed: float
    expected_raw: float
    qmc_n: int
    qmc_levels: tuple[int, ...]
    numerical_unstable: bool
    branches: int
    method: str
    expected_false: float = 0.0


@dataclass(frozen=True)
class PlanningDecision:
    kind: str  # "probe", "commit", "pair_probe", "exit", "undecided"
    action_id: str | None
    option: TerminalOption | None
    score: float
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def planning_decision_to_public_dict(decision: PlanningDecision) -> dict[str, Any]:
    """Serialize a zero-history decision without fixture/private fields.

    The cache is an optimization for chunked runner processes.  A cached
    decision is still checked against the current actual state before any
    physical execution; its unchecked event fields are intentionally omitted.
    """
    payload: dict[str, Any] = {
        "kind": decision.kind, "action_id": decision.action_id,
        "score": float(decision.score), "reason": decision.reason,
        "diagnostics": _json_safe(decision.diagnostics),
        "option": None,
    }
    if decision.option is not None:
        option = decision.option
        prediction = option.prediction
        payload["option"] = {
            "first_id": option.first_id, "second_id": option.second_id,
            "actions": [_json_safe(vars(action)) for action in option.option.actions],
            "max_retries": int(option.option.max_retries),
            "gh_order": int(option.option.gh_order),
            "qmc_seed": int(option.option.qmc_seed),
            "require_detected_for_success": bool(option.option.require_detected_for_success),
            "prediction": _json_safe(vars(prediction)),
        }
    return payload


def planning_decision_from_public_dict(payload: Mapping[str, Any]) -> PlanningDecision:
    """Reconstruct a cached public decision for a fresh planner instance."""
    option_payload = payload.get("option")
    option = None
    if isinstance(option_payload, Mapping):
        actions = tuple(ProbeAction(
            str(item["window"]), tuple(float(x) for x in item["xy_m"]),
            float(item["yaw_rad"]), float(item["duration_s"]),
            str(item.get("kind", "probe")))
            for item in option_payload.get("actions", ()))
        complete = CompleteOption(
            actions=actions,
            max_retries=int(option_payload.get("max_retries", len(actions))),
            gh_order=int(option_payload.get("gh_order", 33)),
            qmc_seed=int(option_payload.get("qmc_seed", 918273)),
            require_detected_for_success=bool(option_payload.get(
                "require_detected_for_success", False)),
        )
        prediction_payload = dict(option_payload.get("prediction", {}))
        if prediction_payload.get("calibration_interval") is not None:
            prediction_payload["calibration_interval"] = tuple(
                float(x) for x in prediction_payload["calibration_interval"])
        prediction = CompletePrediction(**prediction_payload)
        option = TerminalOption(
            str(option_payload["first_id"]),
            None if option_payload.get("second_id") is None else str(option_payload["second_id"]),
            None, None, complete, prediction, actions[0],
            None if len(actions) < 2 else actions[1])
    return PlanningDecision(
        str(payload.get("kind", "exit")),
        None if payload.get("action_id") is None else str(payload["action_id"]),
        option, float(payload.get("score", 0.0)), str(payload.get("reason", "cached")),
        dict(payload.get("diagnostics", {})),
    )


def public_action_from_event(event: Any, frame: np.ndarray) -> ProbeAction:
    """Construct the acoustic action from the actual measured-frame exit pose."""
    local = local_pose(frame, event.output.pose)
    meta = event.action
    kind = "terminal" if meta.get("kind") == "terminal" or event.action_id in TERMINAL_IDS else "probe"
    return ProbeAction(
        str(meta.get("window", "fork")),
        (float(local[0]), float(local[1])),
        float(local[2]),
        float(meta.get("dwell_s", 1.0 if kind == "terminal" else 0.15)),
        kind,
    )


def _belief_task_classes(belief: Belief, left: ProbeAction, right: ProbeAction) -> np.ndarray:
    """Posterior mass over public geometry classes 00/10/01/11.

    The class is a pair of terminal geometry *and noiseless-quality*
    viability bits.  It uses only the planner's Sobol particles, present-state
    HMM mass, and conditional Gaussian response posterior; no fixture truth is
    consulted.
    """
    # Keep this import private so the public planner does not require hidden
    # fixture fields.  The helper is a declared part of belief_model's public
    # particle approximation, not a truth accessor.
    try:
        from belief_model import _geometry_terms
    except ImportError:  # pragma: no cover
        from .belief_model import _geometry_terms
    # This is a posterior viability diagnostic, not a hypothetical next-frame
    # visibility transition.  Use the current mode/component mass, including
    # lost-mode mass, so A_info does not silently condition on present=true.
    modes = belief.mode_weights
    weights = belief.component_weights
    precisions = belief.precisions
    informations = belief.informations
    _, _, lquality, _, lgate = _geometry_terms(belief.geometry, left)
    _, _, rquality, _, rgate = _geometry_terms(belief.geometry, right)
    try:
        from belief_model import _response_stats
    except ImportError:  # pragma: no cover
        from .belief_model import _response_stats
    h_left = np.array((1., float(left.tilt_rad) / YAW_SCALE_RAD))
    h_right = np.array((1., float(right.tilt_rad) / YAW_SCALE_RAD))
    particle_count, mode_count, component_count = modes.shape[0], modes.shape[1], weights.shape[2]
    mean, covariance = _response_stats(
        precisions.reshape(-1, 2, 2), informations.reshape(-1, 2))
    mean = mean.reshape(particle_count, mode_count, component_count, 2)
    covariance = covariance.reshape(particle_count, mode_count, component_count, 2, 2)
    mean_l = lquality[:, None, None] + np.einsum("a,imja->imj", h_left, mean)
    mean_r = rquality[:, None, None] + np.einsum("a,imja->imj", h_right, mean)
    variance_l = np.einsum("a,imjab,b->imj", h_left, covariance, h_left)
    variance_r = np.einsum("a,imjab,b->imj", h_right, covariance, h_right)
    covariance_lr = np.einsum("a,imjab,b->imj", h_left, covariance, h_right)
    sd_l, sd_r = np.sqrt(np.maximum(variance_l, 1.0e-18)), np.sqrt(np.maximum(variance_r, 1.0e-18))
    z_l = (mean_l - QUALITY_THRESHOLD) / sd_l
    z_r = (mean_r - QUALITY_THRESHOLD) / sd_r
    rho = covariance_lr / np.maximum(sd_l * sd_r, 1.0e-18)
    p_l = ndtr(z_l) * lgate[:, None, None]
    p_r = ndtr(z_r) * rgate[:, None, None]
    p_lr = _bivariate_standard_normal_cdf(z_l, z_r, rho) \
        * (lgate & rgate)[:, None, None]
    # Deterministic limits avoid treating an almost-zero variance as a broad
    # Gaussian tail and make the class boundary reproducible.
    deterministic_l = variance_l <= 1.0e-14
    deterministic_r = variance_r <= 1.0e-14
    q_l = (mean_l >= QUALITY_THRESHOLD) & lgate[:, None, None]
    q_r = (mean_r >= QUALITY_THRESHOLD) & rgate[:, None, None]
    p_l = np.where(deterministic_l, q_l.astype(float), p_l)
    p_r = np.where(deterministic_r, q_r.astype(float), p_r)
    p_lr = np.where(deterministic_l & deterministic_r, (q_l & q_r).astype(float), p_lr)
    p_lr = np.where(deterministic_l & ~deterministic_r,
                    np.where(q_l, p_r, 0.0), p_lr)
    p_lr = np.where(~deterministic_l & deterministic_r,
                    np.where(q_r, p_l, 0.0), p_lr)
    mass = (belief.particle_weights[:, None, None] * modes[:, :, None] * weights)
    result = np.zeros(4, dtype=float)
    result[0] = np.sum(mass * (1.0 - p_l - p_r + p_lr))
    result[1] = np.sum(mass * (p_l - p_lr))
    result[2] = np.sum(mass * (p_r - p_lr))
    result[3] = np.sum(mass * p_lr)
    result = np.maximum(result, 0.0)
    total = float(result.sum())
    return result / total if total > 0.0 else np.full(4, 0.25)


def _make_predicted_packet(action: ProbeAction, event_id: str, outcome: Mapping[str, Any]) -> PublicObservation:
    return PublicObservation(
        event_id=str(event_id), address="predicted_public", window=action.window,
        elapsed_s=0.0, exit_state="predicted", detected=bool(outcome["detected"]),
        confidence=float(outcome["confidence"]), quality=float(outcome["quality"]),
        center_offset_m=float(outcome.get("center_offset_m", float("nan"))),
        axis_angle_rad=float(outcome.get("axis_angle_rad", float("nan"))),
        candidate_type=str(outcome["candidate_type"]),
    )


def _relabel_packet(packet: PublicObservation, event_id: str) -> PublicObservation:
    """Copy a hypothetical public packet with the belief event id it uses."""
    return PublicObservation(
        str(event_id), packet.address, packet.window, packet.elapsed_s,
        packet.exit_state, packet.detected, packet.confidence, packet.quality,
        packet.center_offset_m, packet.axis_angle_rad, packet.candidate_type)


def public_acceptance(packet: PublicObservation, *, quality_threshold: float = .58,
                      center_tolerance_m: float = CENTER_TOLERANCE_M,
                      axis_tolerance_rad: float = AXIS_TOLERANCE_RAD,
                      confidence_threshold: float = .65) -> bool:
    """Shared public verifier for every newly acquired packet."""
    return bool(packet.detected and np.isfinite(packet.quality)
                and np.isfinite(packet.center_offset_m)
                and np.isfinite(packet.axis_angle_rad)
                and packet.quality >= quality_threshold
                and abs(packet.center_offset_m) <= center_tolerance_m
                and abs(packet.axis_angle_rad) <= axis_tolerance_rad
                and packet.confidence >= confidence_threshold)


def _qmc_uniforms(n: int, seed: int) -> np.ndarray:
    if n & (n - 1) == 0:
        return qmc.Sobol(d=4, scramble=True, seed=int(seed)).random_base2(int(math.log2(n)))
    return qmc.Sobol(d=4, scramble=True, seed=int(seed)).random(int(n))


def public_outcome_branches(belief: Belief, action: ProbeAction, n: int, *, seed: int) -> tuple[tuple[float, PublicObservation], ...]:
    """Generate common-random public branches from a belief prediction.

    Every branch has weight ``1/n``.  A Sobol categorical draw selects either
    missing or one of the public mixture components; the second dimension is
    a quality quantile.  No target/clutter label is exposed in the packet.
    """
    if n < 1:
        raise ValueError("n must be positive")
    prediction = belief.public_predict(action)
    components = tuple(prediction.components)
    component_weights = np.asarray([max(0.0, float(c.weight)) for c in components], dtype=float)
    candidate_mass = float(np.sum(component_weights))
    if candidate_mass > 0.0:
        component_cdf = np.cumsum(component_weights / candidate_mass)
    else:
        component_cdf = np.empty(0)
    uniforms = _qmc_uniforms(n, seed)
    branches = []
    for index, (u_kind, u_quality, u_center, u_axis) in enumerate(uniforms):
        if float(u_kind) >= float(prediction.p_candidate) or not len(components):
            outcome = {"detected": False, "candidate_type": "missing",
                       "quality": float("nan"), "confidence": 0.0}
        else:
            component_index = int(np.searchsorted(component_cdf, float(u_kind) / max(prediction.p_candidate, 1.0e-12), side="right"))
            component_index = min(component_index, len(components) - 1)
            component = components[component_index]
            sd = math.sqrt(max(float(component.quality_variance), 1.0e-12))
            quality = float(component.quality_mean + sd * ndtri(float(np.clip(u_quality, 1.0e-8, 1.0 - 1.0e-8))))
            distribution = str(getattr(component, "feature_distribution",
                                       getattr(component, "distribution", "gaussian"))).lower()
            if "uniform" in distribution:
                # SENSOR_V2 clutter center/axis are uniform.  The marker is
                # internal to the model component; the packet remains only a
                # public candidate with no association label.
                half_center = float(getattr(belief.params, "clutter_center_halfwidth", .022))
                half_axis = float(getattr(belief.params, "clutter_axis_halfwidth", .35))
                center = float((2. * u_center - 1.) * half_center)
                axis = float((2. * u_axis - 1.) * half_axis)
            else:
                center_sd = math.sqrt(max(float(component.center_variance), 1.0e-12))
                axis_sd = math.sqrt(max(float(component.axis_variance), 1.0e-12))
                center = float(component.center_mean + center_sd * ndtri(float(np.clip(u_center, 1.0e-8, 1.0 - 1.0e-8))))
                axis = float(component.axis_mean + axis_sd * ndtri(float(np.clip(u_axis, 1.0e-8, 1.0 - 1.0e-8))))
            outcome = {"detected": True, "candidate_type": "candidate", "quality": quality,
                       "center_offset_m": center, "axis_angle_rad": axis,
                       "confidence": _sigmoid((quality - 0.34) / 0.14)}
        packet = _make_predicted_packet(action, f"qmc/{seed}/{index}", outcome)
        branches.append((1.0 / n, packet))
    return tuple(branches)


class AdaptivePlanner:
    """Public adaptive planner shared by all experiment arms."""

    def __init__(self, adapter: Any, measured_frame: np.ndarray, *, config: PlannerConfig | None = None,
                 belief: Belief | None = None, model_params: ModelParams | None = None,
                 initial_decision: PlanningDecision | None = None):
        self.adapter = adapter
        self.measured_frame = np.asarray(measured_frame, dtype=float).copy()
        self.config = config or PlannerConfig()
        if model_params is not None and abs(float(model_params.risk_gate) - self.config.risk_gate) > 1.0e-12:
            # The protocol gate is common to every arm; an artifact cannot
            # silently change it at test time.
            model_params = ModelParams.from_dict({**model_params.to_dict(),
                                                   "risk_gate": self.config.risk_gate})
        self.belief = belief or new_belief(params=model_params)
        self.state = adapter.start
        self.probe_count = 0
        self.fresh_count = 0
        self.terminal_count = 0
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        # A selected native candidate can be unavailable from the exact
        # actual state even when its public proposal was feasible.  Keep that
        # no-witness fact local to the complete state; it must not move the
        # state or poison the same action from a later actual state.
        self.blacklist: set[tuple[tuple, str]] = set()
        self.local_option_rejections = 0
        self.pending_retry: str | None = None
        self.pair_position = 0
        self.pairs_completed = 0
        self.recovery_state_key: tuple | None = None
        self.decision_rows: list[dict[str, Any]] = []
        self.prediction_cache: dict[tuple[str, str, int], tuple[tuple[float, PublicObservation], ...]] = {}
        self.check_cache: dict[tuple[tuple, str], ProbeCandidate] = {}
        self.planning_diagnostics: list[dict[str, Any]] = []
        self._last_prefix_event: str | None = None
        self.initial_decision = initial_decision
        self._native_counter_start = self._native_counter_snapshot()
        self._checked_event_start = len(getattr(self.adapter, "_checked_events", {}))
        self._planner_check_cache_hits = 0
        self.timing = {"choose_s": 0.0, "future_outcomes_s": 0.0,
                       "option_predict_s": 0.0, "belief_update_s": 0.0,
                       "native_selected_s": 0.0, "acquisition_s": 0.0}

    @property
    def arm(self) -> str:
        return self.config.arm

    def continuation_payload(self, *, private_present: bool | None = None) -> dict[str, Any]:
        """Serializable state for an exact unresolved continuation."""
        return {
            "state": self.state, "belief": self.belief.clone(),
            "probe_count": self.probe_count, "fresh_count": self.fresh_count,
            "terminal_count": self.terminal_count, "step_count": self.step_count,
            "history": list(self.history), "pair_order": self.config.pair_order,
            "pending_retry": self.pending_retry, "pair_position": self.pair_position,
            "pairs_completed": self.pairs_completed,
            "recovery_state_key": self.recovery_state_key,
            "decisions": list(self.decision_rows),
            "blacklist": [(list(state_key), action_id)
                          for state_key, action_id in sorted(self.blacklist, key=repr)],
            "local_option_rejections": self.local_option_rejections,
            "timing": dict(self.timing),
            "arm": self.arm, "max_probe_frames": self.config.max_probe_frames,
            "private_present": private_present,
        }

    def restore_continuation(self, payload: Mapping[str, Any]) -> None:
        self.state = payload["state"]
        self.belief = payload["belief"]
        self.probe_count = int(payload.get("probe_count", 0))
        self.fresh_count = int(payload.get("fresh_count", 0))
        self.terminal_count = int(payload.get("terminal_count", 0))
        self.step_count = int(payload.get("step_count", 0))
        self.history = list(payload.get("history", ()))
        self.pending_retry = payload.get("pending_retry")
        self.pair_position = int(payload.get("pair_position", 0))
        self.pairs_completed = int(payload.get("pairs_completed", 0))
        recovery_key = payload.get("recovery_state_key")
        self.recovery_state_key = None if recovery_key is None else tuple(recovery_key)
        self.decision_rows = list(payload.get("decisions", ()))
        self.blacklist = {(tuple(state_key), str(action_id))
                          for state_key, action_id in payload.get("blacklist", ())}
        self.local_option_rejections = int(payload.get("local_option_rejections", 0))
        if isinstance(payload.get("timing"), Mapping):
            self.timing.update({str(key): float(value)
                                for key, value in payload["timing"].items()})

    def _state_key(self) -> tuple:
        return tuple(self.state.key)

    def _native_counter_snapshot(self) -> dict[str, int]:
        return {name: int(getattr(self.adapter, name, 0)) for name in (
            "verification_calls", "execution_calls", "verification_count",
            "execution_count", "proposal_calls", "proposal_cache_hits",
            "proposal_dird_queries")}

    def _native_count_report(self) -> tuple[dict[str, int], dict[str, int]]:
        cumulative = self._native_counter_snapshot()
        delta = {name: int(cumulative[name] - self._native_counter_start.get(name, 0))
                  for name in cumulative}
        # checked() maintains a complete-state memory cache.  Its size grows
        # only on a new native planning trajectory, so the difference is a
        # direct count of newly materialised checks for this case.
        checked_cache_size = len(getattr(self.adapter, "_checked_events", {}))
        new_checked = max(0, checked_cache_size - self._checked_event_start)
        checked_cache_hits = max(0, delta.get("verification_calls", 0) - new_checked)
        execution_reuse = sum(
            1 for row in self.history
            if row.get("native_actual", {}).get("call_kind") == "execution_reuse")
        execution_native = sum(
            1 for row in self.history
            if row.get("native_actual", {}).get("call_kind") == "execution")
        delta.update({
            "new_checked_trajectories": int(new_checked),
            "checked_cache_hits": int(checked_cache_hits),
            "planner_check_cache_hits": int(self._planner_check_cache_hits),
            "execution_reuse_count": int(execution_reuse),
            "execution_native_count": int(execution_native),
            "cache_hits": int(checked_cache_hits + self._planner_check_cache_hits
                               + delta.get("proposal_cache_hits", 0) + execution_reuse),
            # A trajectory is real work when a new checked route is generated
            # or an execution has to replay native instead of reusing it.
            "realnative_count": int(new_checked + execution_native),
        })
        cumulative.update({
            "checked_cache_size": int(checked_cache_size),
            "proposal_cache_hits": int(cumulative.get("proposal_cache_hits", 0)),
        })
        return delta, cumulative

    @staticmethod
    def _blacklist_key(state: Any, action_id: str) -> tuple[tuple, str]:
        return tuple(state.key), str(action_id)

    def _is_blacklisted(self, state: Any, action_id: str) -> bool:
        return self._blacklist_key(state, action_id) in self.blacklist

    def _record_no_witness(self, state: Any, action_id: str, *, phase: str) -> None:
        key = self._blacklist_key(state, action_id)
        self.blacklist.add(key)
        self.recovery_state_key = key[0]
        self.history.append({
            "action_id": str(action_id), "status": "native_no_witness",
            "phase": str(phase), "state_key": list(key[0]),
            "blacklisted": True, "probe_count": self.probe_count,
            "fresh_count": self.fresh_count,
        })

    def _checked(self, action_id: str, state: Any | None = None) -> ProbeCandidate:
        state = self.state if state is None else state
        key = (tuple(state.key), str(action_id))
        if key in self.check_cache:
            self._planner_check_cache_hits += 1
            return self.check_cache[key]
        event = self.adapter.checked(state, action_id)
        action = public_action_from_event(event, self.measured_frame)
        candidate = ProbeCandidate(str(action_id), event, action, bool(event.feasible),
                                   float(event.duration_s), None, "native_checked")
        self.check_cache[key] = candidate
        return candidate

    def _predicted(self, action_id: str, state: Any | None = None) -> ProbeCandidate:
        """Public route/action prediction used during search.

        Search must not create 12 native trajectories per layer.  The fixture
        exposes its measured-frame goal and dwell as public action metadata;
        the actual adapter check is deferred until this candidate is selected
        for execution.  ``event=None`` is intentional and is logged as a
        predicted/unknown continuation rather than a native certificate.
        """
        state = self.state if state is None else state
        rig = getattr(self.adapter, "rig", None)
        if rig is None:
            # Test adapters can provide a small action map directly.
            meta = getattr(self.adapter, "action_metadata", {}).get(action_id)
            if meta is None:
                base = str(action_id).split("__", 1)[0]
                goals = {"left": (-.016, .018, .10), "right": (.016, .018, -.10),
                         "fork_minus": (.006, -.003, -.075), "fork_plus": (-.006, -.003, .075),
                         "side_left": (-.016, .013, .075), "side_right": (.016, .013, -.075)}
                if base not in goals:
                    raise KeyError(action_id)
                window = "left" if "left" in base else "right" if "right" in base else "fork"
                raw_dwell = str(action_id).rsplit("_", 1)[-1] if "single_dwell" in str(action_id) else "1.0"
                dwell = float(raw_dwell)
                goal, kind = goals[base], "terminal" if base in TERMINAL_IDS else "probe"
            else:
                window, goal, dwell, kind = meta["window"], meta["goal_local"], meta["dwell_s"], meta["kind"]
        else:
            canonical = rig.resolve_action_id(action_id)
            capture = rig.captures[canonical]
            window, goal, dwell, kind = capture.window, capture.goal, capture.dwell_s, capture.kind
            action_id = canonical
        action = ProbeAction(str(window), (float(goal[0]), float(goal[1])), float(goal[2]),
                             float(dwell), "terminal" if kind == "terminal" else "probe")

        # NativeAdapter.propose() is a public route/dIRD proxy only.  It must
        # not be confused with checked()/execute(), and the latter remains the
        # sole path for a selected physical action.  Small test adapters do
        # not need to implement this optional facade and retain the dwell-only
        # public fallback below.
        propose = getattr(self.adapter, "propose", None)
        if callable(propose):
            proposal = propose(state, str(action_id))
            proposal_action_id = str(getattr(proposal, "action_id", action_id))
            payload = getattr(proposal, "action", None)
            if isinstance(payload, Mapping):
                window = payload.get("window", window)
                goal = payload.get("goal_local", goal)
                dwell = payload.get("dwell_s", dwell)
                kind = payload.get("kind", kind)
                action = ProbeAction(str(window), (float(goal[0]), float(goal[1])),
                                     float(goal[2]), float(dwell),
                                     "terminal" if kind == "terminal" else "probe")
            available = bool(getattr(proposal, "available",
                                     getattr(proposal, "feasible", True)))
            duration = getattr(proposal, "duration_s", None)
            duration_s = float(dwell if duration is None else duration)
            predicted_output = getattr(proposal, "predicted_output",
                                       getattr(proposal, "output", None))
            status = str(getattr(proposal, "status", "predicted_unknown"))
            return ProbeCandidate(proposal_action_id, None, action, available,
                                  duration_s, predicted_output, status)

        # Dwell is the only action cost known before a native route is checked.
        # It is not used as a hard time constraint (lambda_time defaults zero).
        return ProbeCandidate(str(action_id), None, action, True, float(dwell),
                              None, "public_goal_geometry_only")

    def _probe_candidates(self, state: Any | None = None) -> tuple[ProbeCandidate, ...]:
        state = self.state if state is None else state
        # The complete 12-action layer is evaluated before any ranking, using
        # public geometric/kinematic predictions rather than native rollouts.
        values = [self._predicted(action_id, state) for action_id in DEFAULT_PROBE_IDS]
        return tuple(candidate for candidate in values
                     if candidate.feasible and not self._is_blacklisted(state, candidate.action_id))

    def _terminal_options(self, state: Any | None = None) -> tuple[TerminalOption, ...]:
        state = self.state if state is None else state
        options: list[TerminalOption] = []
        for first_id in TERMINAL_IDS:
            first = self._predicted(first_id, state)
            # A public route proposal is not a native certificate, but an
            # unavailable route cannot become an executable complete option.
            if not first.feasible or self._is_blacklisted(state, first.action_id):
                continue
            # A complete option is one selected window with one same-window
            # retry; switching windows is a new diagnostic/option decision.
            second_id = first_id
            second = (self._predicted(second_id, first.predicted_output or state)
                      if second_id is not None else None)
            if second is not None and (not second.feasible or
                                       self._is_blacklisted(first.predicted_output or state,
                                                            second.action_id)):
                second = None
            actions = (first.action,) if second is None else (first.action, second.action)
            retries = 1 if second is None else 2
            complete = CompleteOption(actions=actions, max_retries=retries, gh_order=33,
                                      require_detected_for_success=False)
            prediction = predict_complete_option(self.belief, complete)
            options.append(TerminalOption(first_id, None if second is None else second.action_id,
                                          first.event, None if second is None else second.event,
                                          complete, prediction, first.action,
                                          None if second is None else second.action))
        return tuple(options)

    def _best_terminal(self, options: Sequence[TerminalOption]) -> TerminalOption | None:
        allowed = [option for option in options if option.allowed]
        if not allowed:
            return None
        return max(allowed, key=lambda option: (option.score, option.prediction.raw_complete_p, option.first_id))

    def _prediction_branches(self, candidate: ProbeCandidate, n: int, *, depth: int, tag: str) -> tuple[tuple[float, PublicObservation], ...]:
        # Include the current public history in the key, so a branch is never
        # accidentally reused after a real observation changed the belief.
        history = tuple(sorted(self.belief.event_ids))
        # ``depth`` is hypothetical beam depth.  Real re-plans keep depth=0;
        # the actual probe count is a separate seed coordinate.  Omitting the
        # diagnostic tag makes 8/16/32/64/256 levels a nested common sequence.
        digest = hashlib.sha256(repr((history, candidate.action_id, depth)).encode()).hexdigest()
        key = (digest, candidate.action_id, int(n))
        if key not in self.prediction_cache:
            t0 = time.perf_counter()
            self.prediction_cache[key] = public_outcome_branches(
                self.belief, candidate.action, n,
                seed=int(self.config.qmc_seed + self.probe_count * 1009 + depth * 17
                         + int(digest[:8], 16)),
            )
            self.timing["future_outcomes_s"] += time.perf_counter() - t0
        return self.prediction_cache[key]

    def _branch_value(self, candidate: ProbeCandidate, packet: PublicObservation, *, branch_index: int,
                      depth: int, use_beam: bool) -> tuple[float, float, float]:
        posterior = self.belief.clone()
        event_id = f"plan/{self.arm}/{depth}/{candidate.action_id}/{branch_index}"
        posterior.update(event_id, candidate.action, PublicObservation(
            event_id, packet.address, packet.window, packet.elapsed_s, packet.exit_state,
            packet.detected, packet.confidence, packet.quality, packet.center_offset_m,
            packet.axis_angle_rad, packet.candidate_type))
        original = self.belief
        self.belief = posterior
        try:
            # The hypothetical probe has no checked native event yet.  Its
            # public action/observation is sufficient for belief and terminal
            # value prediction; the actual state is checked after selection.
            t_option = time.perf_counter()
            predicted_state = candidate.predicted_output or self.state
            options = self._terminal_options(predicted_state)
            self.timing["option_predict_s"] += time.perf_counter() - t_option
            # Accepted packets are handled as an absorbing mass by
            # _score_probe_at_level.  This branch is reached only for a public
            # rejection, so no extra same-action prediction is made here.
            best = self._best_terminal(options)
            if best is not None:
                allowed = float(best.score)
                raw = float(best.prediction.raw_complete_p)
            else:
                allowed = 0.0
                raw = max((float(option.prediction.raw_complete_p) for option in options), default=0.0)
            if use_beam and depth < 1 and best is None:
                # Beam-2 evaluates the best second diagnostic for this public
                # branch.  It is intentionally based on posterior packets,
                # never on a hidden scenario.
                second_scores = self._rank_probe_candidates(
                    self._probe_candidates(predicted_state), depth=depth + 1,
                    n=self.config.h2_initial, max_candidates=self.config.beam_width,
                    use_beam=False)
                if second_scores:
                    allowed = max(allowed, float(second_scores[0].value))
                    raw = max(raw, float(second_scores[0].expected_raw))
            return allowed, raw, 0.0
        finally:
            self.belief = original

    def _score_probe_at_level(self, candidate: ProbeCandidate, n: int, *, depth: int,
                              use_beam: bool, tag: str = "h1") -> ProbeScore:
        branches = self._prediction_branches(candidate, n, depth=depth, tag=tag)
        prediction = self.belief.public_predict(candidate.action)
        allowed_values, raw_values, false_values = [], [], []
        reject_mass = 0.0
        for branch_index, (_, packet) in enumerate(branches):
            branch_weight = 1.0 / max(len(branches), 1)
            if public_acceptance(
                    packet, quality_threshold=self.belief.params.public_quality_threshold,
                    center_tolerance_m=self.belief.params.center_tolerance_m,
                    axis_tolerance_rad=self.belief.params.axis_tolerance_rad,
                    confidence_threshold=self.belief.params.public_confidence_threshold):
                # Immediate correct/false acceptance is represented by the
                # analytic predictive masses below.  Do not posterior-update
                # or predict a future option for an absorbing branch.
                continue
            reject_mass += branch_weight
            allowed, raw, false = self._branch_value(candidate, packet, branch_index=branch_index,
                                                     depth=depth, use_beam=use_beam)
            allowed_values.append(branch_weight * allowed)
            raw_values.append(branch_weight * raw)
            false_values.append(branch_weight * false)
        # PacketPrediction's V2 masses are the source of immediate correct /
        # false acceptance probability.  QMC only estimates the conditional
        # public-reject posterior continuation and retains its reject mass.
        immediate_correct = float(getattr(prediction, "p_correct_accept", 0.0))
        immediate_false = float(getattr(prediction, "p_false_accept", 0.0))
        expected_allowed = immediate_correct + float(np.sum(allowed_values))
        expected_raw = immediate_correct + float(np.sum(raw_values))
        expected_false = immediate_false + float(np.sum(false_values))
        # lambda_time is a declared sensitivity only; native duration remains
        # in logs and is never used as a hard cutoff.
        value = expected_allowed - self.config.lambda_time * candidate.duration_s
        return ProbeScore(candidate, value, expected_allowed, expected_raw,
                          n, (n,), False, len(branches), tag, expected_false)

    def _rank_probe_candidates(self, candidates: Sequence[ProbeCandidate], *, depth: int,
                               n: int, max_candidates: int | None = None,
                               use_beam: bool = False) -> tuple[ProbeScore, ...]:
        if not candidates:
            return ()
        initial = [self._score_probe_at_level(candidate, n, depth=depth, use_beam=use_beam)
                   for candidate in candidates]
        initial.sort(key=lambda score: (score.value, score.expected_raw, score.candidate.action_id), reverse=True)
        top = initial[:2]
        refined: dict[str, ProbeScore] = {score.candidate.action_id: score for score in initial}
        levels: dict[str, list[int]] = {score.candidate.action_id: [n] for score in initial}
        unstable: dict[str, bool] = {score.candidate.action_id: False for score in initial}
        if len(top) >= 2:
            # H2 changes only whether a second hypothetical action is
            # expanded.  Its first real layer keeps the full H1 16→32→64→256
            # schedule; the nested beam layer uses the declared 8→16→32
            # schedule.  This avoids both silent top-level under-sampling and
            # an accidental 256-sample nested beam.
            nested = depth > 0
            levels_to_refine = ((self.config.h2_refine, self.config.h2_unstable)
                                if nested else (self.config.qmc_top2, self.config.qmc_refine))
            final_level = self.config.h2_unstable if nested else self.config.qmc_unstable
            final_trigger = self.config.h2_unstable if nested else self.config.qmc_refine
            for level in levels_to_refine:
                previous_ids = tuple(score.candidate.action_id for score in top)
                previous_values = {score.candidate.action_id: score.value for score in top}
                for score in top:
                    next_score = self._score_probe_at_level(score.candidate, level, depth=depth,
                                                            use_beam=use_beam,
                                                            tag="h2" if use_beam else "h1")
                    old = refined[score.candidate.action_id]
                    refined[score.candidate.action_id] = ProbeScore(
                        next_score.candidate, next_score.value, next_score.expected_allowed,
                        next_score.expected_raw, level, tuple(levels[score.candidate.action_id] + [level]),
                        old.numerical_unstable, next_score.branches, next_score.method,
                        next_score.expected_false)
                    levels[score.candidate.action_id].append(level)
                top = sorted((refined[s.candidate.action_id] for s in top),
                             key=lambda s: (s.value, s.expected_raw, s.candidate.action_id), reverse=True)
                winner_changed = top[0].candidate.action_id != previous_ids[0]
                max_value_change = max(abs(top[i].value - previous_values[previous_ids[i]])
                                       for i in range(min(2, len(top))))
                if winner_changed or max_value_change > 0.01:
                    # A final high-order pass is required before declaring a
                    # stable ranking.  H2 intentionally stops at 32.
                    if level == final_trigger:
                        for score in top:
                            final = self._score_probe_at_level(score.candidate, final_level, depth=depth,
                                                               use_beam=use_beam,
                                                               tag="h2_final" if use_beam else "h1_final")
                            refined[score.candidate.action_id] = ProbeScore(
                                final.candidate, final.value, final.expected_allowed,
                                final.expected_raw, final_level,
                                tuple(levels[score.candidate.action_id] + [final_level]),
                                True, final.branches, final.method, final.expected_false)
                            unstable[score.candidate.action_id] = True
                        break
        result = []
        for score in refined.values():
            result.append(ProbeScore(score.candidate, score.value, score.expected_allowed,
                                     score.expected_raw, score.qmc_n, score.qmc_levels,
                                     bool(score.numerical_unstable or unstable[score.candidate.action_id]),
                                     score.branches, score.method, score.expected_false))
        result.sort(key=lambda score: (score.value, score.expected_raw, score.candidate.action_id), reverse=True)
        return tuple(result if max_candidates is None else result[:max_candidates])

    def _info_gain_probe(self, candidates: Sequence[ProbeCandidate], *, depth: int) -> ProbeScore | None:
        if not candidates:
            return None
        left = self._predicted("left")
        right = self._predicted("right")
        prior = _belief_task_classes(self.belief, left.action, right.action)
        prior_entropy = _entropy(prior)
        ranked = []
        for candidate in candidates:
            outcomes = self._prediction_branches(candidate, self.config.qmc_initial, depth=depth, tag="A")
            expected_entropy = 0.0
            for branch_index, (weight, packet) in enumerate(outcomes):
                if public_acceptance(
                        packet, quality_threshold=self.belief.params.public_quality_threshold,
                        center_tolerance_m=self.belief.params.center_tolerance_m,
                        axis_tolerance_rad=self.belief.params.axis_tolerance_rad,
                        confidence_threshold=self.belief.params.public_confidence_threshold):
                    continue
                posterior = self.belief.clone()
                event_id = f"plan/A/{depth}/{candidate.action_id}/{branch_index}"
                posterior.update(event_id, candidate.action, _relabel_packet(packet, event_id))
                post_classes = _belief_task_classes(posterior, left.action, right.action)
                expected_entropy += float(weight) * _entropy(post_classes)
            value = prior_entropy - expected_entropy - self.config.lambda_time * candidate.duration_s
            ranked.append(ProbeScore(candidate, float(value), float(value), float(value),
                                     self.config.qmc_initial, (self.config.qmc_initial,), False,
                                     len(outcomes), "A_task_class_information_gain", 0.0))
        return max(ranked, key=lambda score: (score.value, score.candidate.action_id))

    def _task_probe(self, candidates: Sequence[ProbeCandidate], *, depth: int) -> ProbeScore | None:
        """A_task: maximize immediate correct completion probability.

        The public verifier may accept a probe, so this is a real completion
        option rather than a diagnostic-only proxy.  Information gain is used
        only as the deterministic tie-break, never as a way to weaken A.
        """
        if not candidates:
            return None
        # Compute the direct correct probability with the same shared-latent
        # fresh-frame predictor used for terminal options.  No public packet
        # is fabricated and no private truth is read.
        rows = []
        for candidate in candidates:
            prediction = self.belief.public_predict(candidate.action)
            direct = float(getattr(prediction, "p_correct_accept", 0.0))
            wrong = float(getattr(prediction, "p_false_accept", 0.0))
            rows.append(ProbeScore(candidate, direct, direct, direct,
                                   self.config.qmc_initial, (self.config.qmc_initial,),
                                   False, 1, "A_task_immediate_correct",
                                   wrong))
        # A separate information estimate for each tied score is expensive;
        # use the deterministic task-class IG only when scores are numerically
        # tied.  This keeps the baseline strong while retaining a reproducible
        # tie rule.
        best_value = max(row.value for row in rows)
        tied = [row for row in rows if abs(row.value - best_value) <= 1.0e-10]
        if len(tied) <= 1:
            return tied[0]
        ig_by_action: dict[str, float] = {}
        left = self._predicted("left")
        right = self._predicted("right")
        prior_entropy = _entropy(_belief_task_classes(self.belief, left.action, right.action))
        for row in tied:
            expected = 0.0
            for branch_index, (weight, packet) in enumerate(self._prediction_branches(
                    row.candidate, self.config.qmc_initial, depth=depth, tag="A_task_tie")):
                if public_acceptance(
                        packet, quality_threshold=self.belief.params.public_quality_threshold,
                        center_tolerance_m=self.belief.params.center_tolerance_m,
                        axis_tolerance_rad=self.belief.params.axis_tolerance_rad,
                        confidence_threshold=self.belief.params.public_confidence_threshold):
                    continue
                posterior = self.belief.clone()
                event_id = f"plan/A_task/{depth}/{row.candidate.action_id}/{branch_index}"
                posterior.update(event_id, row.candidate.action, _relabel_packet(packet, event_id))
                expected += float(weight) * _entropy(_belief_task_classes(posterior, left.action, right.action))
            ig_by_action[row.candidate.action_id] = prior_entropy - expected
        return max(tied, key=lambda row: (ig_by_action.get(row.candidate.action_id, -math.inf),
                                          row.candidate.action_id))

    def _task_probe_v4(self, candidates: Sequence[ProbeCandidate]) -> ProbeScore | None:
        """A_task_v4 immediate score with the shared cheap tie proxy."""
        if not candidates:
            return None
        try:
            from rollout_policy import task_action_scores
            correct, false, proxy = task_action_scores(
                self.belief, tuple(candidate.action for candidate in candidates))
        except (ImportError, AttributeError, TypeError, ValueError):
            rows = [self.belief.public_predict(candidate.action) for candidate in candidates]
            correct = np.asarray([float(getattr(row, "p_correct_accept", 0.0))
                                  for row in rows])
            false = np.asarray([float(getattr(row, "p_false_accept", 0.0))
                                for row in rows])
            proxy = np.zeros(len(candidates), dtype=float)
        best = float(np.max(correct)) if len(correct) else 0.0
        tied = np.flatnonzero(np.abs(correct - best) <= 1.0e-10)
        # Keep the external baseline and the rollout tail on the same
        # declared A_task_v4 tie rule: proxy, shorter dwell, canonical action
        # order (the candidate tuple is the canonical 12-action order).
        index = int(min(tied, key=lambda item: (
            -float(proxy[item]), float(candidates[item].action.duration_s), int(item))))
        candidate = candidates[index]
        return ProbeScore(candidate, float(correct[index]), float(correct[index]),
                          float(correct[index]), self.config.qmc_initial,
                          (self.config.qmc_initial,), False, 1,
                          "A_task_v4_immediate_correct_task_direction_tie",
                          float(false[index]))

    def choose(self) -> PlanningDecision:
        """Choose one public action from the current complete native state."""
        diagnostics: dict[str, Any] = {
            "risk_gate": self.config.risk_gate,
            "probe_frame": self.probe_count,
            "fresh_count": self.fresh_count,
            "no_deadline": True,
        }
        if self.fresh_count >= self.config.max_probe_frames:
            return PlanningDecision("exit", None, None, 0.0,
                                    "watchdog_probe_guard", diagnostics)
        if self.pending_retry is not None:
            return PlanningDecision("terminal_retry", self.pending_retry, None, 0.0,
                                    "admitted_option_pending_retry", diagnostics)

        recover = self.recovery_state_key == self._state_key()
        prescribed_pair = None
        if self.arm == "pair_fixed":
            pair_order = self.config.pair_order or ("fork_minus", "fork_plus")
            prescribed_pair = single_dwell_action(pair_order[self.pair_position],
                                                 self.config.pair_dwell)
            recover = recover or self._is_blacklisted(self.state, prescribed_pair)
            # Complete both prescribed members before the ordinary option gate.
            # A native failure allows the shared recovery rule without advancing
            # the prescribed pair's position.
            if not recover and (self.pair_position == 1 or self.pairs_completed == 0):
                return PlanningDecision("pair_probe", prescribed_pair, None, 0.0,
                                        "fixed_pair_prescribed_member", diagnostics)

        options = self._terminal_options(self.state)
        best = self._best_terminal(options)
        diagnostics["terminal_options"] = [self._option_log(option) for option in options]
        diagnostics["common_recovery"] = recover
        if best is not None:
            diagnostics["current_allowed_score"] = best.score
            return PlanningDecision("commit", best.first_id, best, best.score,
                                    "common_risk_gate_admitted", diagnostics)
        if prescribed_pair is not None and not recover:
            return PlanningDecision("pair_probe", prescribed_pair, None, 0.0,
                                    "fixed_pair_repeat_after_replan", diagnostics)

        candidates = self._probe_candidates(self.state)
        if not candidates:
            return PlanningDecision("exit", None, None, 0.0,
                                    "no_checked_probe", diagnostics)
        if recover:
            score = self._task_probe_v4(candidates)
            diagnostics["probe_score"] = self._probe_log(score)
            return PlanningDecision("probe", score.candidate.action_id, None, score.value,
                                    "common_A_task_recovery", diagnostics)
        if self.arm in {"A", "A_task", "A_info", "A_full"}:
            if self.arm == "A_info":
                score = self._info_gain_probe(candidates, depth=0)
            elif self.arm == "A_task":
                score = self._task_probe_v4(candidates)
            else:
                score = self._task_probe(candidates, depth=0)
            if score is None:
                return PlanningDecision("exit", None, None, 0.0, "no_checked_probe", diagnostics)
            diagnostics["probe_score"] = self._probe_log(score)
            return PlanningDecision("probe", score.candidate.action_id, None, score.value,
                                    "A_max_task_class_information_gain", diagnostics)

        if self.arm == "B_rollout":
            from rollout_policy import choose_one_layer
            choice = choose_one_layer(self, candidates)
            diagnostics["rollout"] = choice.diagnostics
            diagnostics["rollout_switch_reason"] = choice.reason
            diagnostics["rollout_switched"] = choice.switched
            return PlanningDecision("probe", choice.action_id, None, choice.score,
                                    "B_rollout_switch" if choice.switched
                                    else "B_rollout_fallback_A_task", diagnostics)

        use_beam = self.arm == "B_H2"
        scores = self._rank_probe_candidates(candidates, depth=0,
                                             n=self.config.qmc_initial,
                                             use_beam=use_beam)
        best_probe = scores[0] if scores else None
        if best_probe is not None:
            diagnostics["probe_score"] = self._probe_log(best_probe)
            diagnostics["all_probe_scores"] = [self._probe_log(score) for score in scores]
        if best_probe is not None:
            return PlanningDecision("probe", best_probe.candidate.action_id, None,
                                    best_probe.value, "B_expected_future_option", diagnostics)
        return PlanningDecision("exit", None, None, 0.0, "no_checked_action", diagnostics)

    @staticmethod
    def _option_log(option: TerminalOption) -> dict[str, Any]:
        prediction = option.prediction
        return {"first_id": option.first_id, "second_id": option.second_id,
                "prediction_scope": "local_two_attempt_option_score",
                "native_status": "predicted_unchecked" if option.first_event is None else "checked",
                "raw_complete_p": prediction.raw_complete_p,
                "calibrated_p": prediction.calibrated_p, "risk_score": prediction.risk_score,
                "admitted": prediction.admitted, "p_target_identity": prediction.p_target_identity,
                "p_geometry_gate": prediction.p_geometry_gate,
                "p_quality_gate": prediction.p_quality_gate,
                "p_joint_first_attempt": prediction.p_joint_first_attempt,
                "p_false_early_stop": prediction.p_false_early_stop,
                "p_public_success_stop": prediction.p_public_success_stop,
                "quadrature_order": prediction.quadrature_order,
                "particle_ess": prediction.particle_ess}

    @staticmethod
    def _probe_log(score: ProbeScore) -> dict[str, Any]:
        return {"action_id": score.candidate.action_id, "value": score.value,
                "expected_allowed": score.expected_allowed, "expected_raw": score.expected_raw,
                "expected_false": score.expected_false,
                "false_mass_scope": "immediate_public_accept_only",
                "prediction_scope": "local_two_attempt_option_score",
                "duration_s": score.candidate.duration_s, "qmc_n": score.qmc_n,
                "native_status": (score.candidate.prediction_status
                                  if score.candidate.event is None else "checked"),
                "qmc_levels": score.qmc_levels, "numerical_unstable": score.numerical_unstable,
                "branches": score.branches, "method": score.method}

    def observe(self, event_id: str, action: ProbeAction, packet: PublicObservation,
                *, event: Any | None = None, status: str = "executed") -> dict[str, Any]:
        """Assimilate one actual public packet and advance from actual state."""
        t_update = time.perf_counter()
        result = self.belief.update(event_id, action, packet)
        self.timing["belief_update_s"] += time.perf_counter() - t_update
        if event is not None:
            self.state = event.output
            self.fresh_count += 1
            if action.kind == "terminal":
                self.terminal_count += 1
            else:
                self.probe_count += 1
        self.step_count += 1
        row = {"event_id": str(event_id), "action_id": None if event is None else event.action_id,
               "action": _json_safe(vars(action)), "dwell_s": float(action.duration_s),
               "public": _json_safe(vars(packet)), "status": status,
               "belief_update": _json_safe(vars(result)), "belief": _json_safe(self.belief.snapshot()),
               "probe_count": self.probe_count, "fresh_count": self.fresh_count,
               "terminal_count": self.terminal_count, "native_elapsed_s": None if event is None else event.elapsed_s,
               "verification_count": None if event is None else event.verification_count,
               "execution_count": None if event is None else event.execution_count}
        if event is not None:
            actual = event.actual_output
            row["native_actual"] = {
                "entry_state_hash": _native_payload_hash({
                    "q": actual.get("entry_q"), "qdot": actual.get("entry_qdot"),
                    "snapshot": actual.get("entry_snapshot")}),
                "exit_state_hash": _native_payload_hash({
                    "q": actual.get("q"), "qdot": actual.get("qdot"),
                    "acceleration": actual.get("acceleration"),
                    "snapshot": actual.get("snapshot")}),
                "route_id": event.route_id,
                "pathref": list(event.trajectory_records),
                "call_kind": event.call_kind,
                "source": event.source,
            }
        self.history.append(row)
        return row

    def run(self, acquire: Any, *, max_steps: int | None = None,
            private_prefix: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
        """Run until public acceptance, legal exit, or the watchdog.

        ``acquire`` is a callable ``(event, state_before) -> (event_id,
        public_action, public_packet, private_eval)``.  The private value is
        returned only under the isolated ``privateeval`` key.
        """
        max_steps = int(max_steps or (self.config.max_probe_frames + 4))
        privateeval = list(private_prefix)
        reason = "undecided_watchdog"
        correct_complete = false_complete = abstain = watchdog = False
        while self.step_count < max_steps:
            t_choose = time.perf_counter()
            if self.step_count == 0 and self.initial_decision is not None:
                decision = self.initial_decision
                self.initial_decision = None
            else:
                decision = self.choose()
            choose_wall = time.perf_counter() - t_choose
            self.timing["choose_s"] += choose_wall
            self.decision_rows.append({
                "step": self.step_count, "decision": decision.kind,
                "action_id": decision.action_id, "score": decision.score,
                "reason": decision.reason, "diagnostics": _json_safe(decision.diagnostics),
                "choose_wall_s": choose_wall,
            })
            if decision.kind == "exit" or decision.action_id is None:
                reason = decision.reason
                watchdog = reason == "watchdog_probe_guard"
                abstain = not watchdog
                break

            action_id = decision.action_id
            state_before = self.state
            terminal = decision.kind in ("commit", "terminal_retry")
            if decision.kind == "commit" and decision.option is None:
                abstain = True
                reason = "missing_terminal_option"
                break
            phase = "terminal" if terminal else "probe"
            t_native = time.perf_counter()
            checked = None if self._is_blacklisted(state_before, action_id) else self._checked(
                action_id, state_before)
            if checked is None or not checked.feasible:
                self.timing["native_selected_s"] += time.perf_counter() - t_native
                self._record_no_witness(state_before, action_id, phase=phase + "_check")
                self.pending_retry = None
                reason = "native_no_witness_replan"
                continue
            event = self.adapter.execute(state_before, action_id)
            self.timing["native_selected_s"] += time.perf_counter() - t_native
            if not event.feasible:
                # A failed execution, unlike a hypothetical check, may already
                # have moved the robot. Retain its actual exit and withdraw.
                self.state = event.output
                self.history.append({
                    "action_id": action_id, "status": "native_execution_invalid",
                    "phase": phase, "state_key": list(state_before.key),
                    "exit_state_key": list(event.output.key), "fresh_count": self.fresh_count,
                })
                self.pending_retry = None
                abstain = True
                reason = "native_execution_invalid"
                break

            t_acquire = time.perf_counter()
            event_id, public_action, packet, private = acquire(event, state_before)
            self.timing["acquisition_s"] += time.perf_counter() - t_acquire
            privateeval.append(_json_safe(private))
            status = "terminal_executed" if terminal else (
                "pair_probe_executed" if decision.kind == "pair_probe" else "probe_executed")
            row = self.observe(event_id, public_action, packet, event=event, status=status)
            row["decision_index"] = len(self.decision_rows) - 1
            self.recovery_state_key = None
            if decision.kind == "pair_probe":
                self.pair_position ^= 1
                if self.pair_position == 0:
                    self.pairs_completed += 1

            public_accepted = public_acceptance(
                packet, quality_threshold=self.belief.params.public_quality_threshold,
                center_tolerance_m=self.belief.params.center_tolerance_m,
                axis_tolerance_rad=self.belief.params.axis_tolerance_rad,
                confidence_threshold=self.belief.params.public_confidence_threshold)
            private_correct = bool(private.get("correct", False))
            private_false = bool(private.get("false", False))
            row["public_accepted"] = public_accepted
            row["timing"] = {
                "choose_s": choose_wall, "native_selected_s": self.timing["native_selected_s"],
                "acquisition_s": self.timing["acquisition_s"],
                "belief_update_s": self.timing["belief_update_s"],
            }
            if public_accepted:
                self.pending_retry = None
                correct_complete = private_correct
                false_complete = private_false or not private_correct
                reason = "public_accepted_terminal" if terminal else "public_accepted_probe"
                break
            if terminal:
                self.pending_retry = decision.option.second_id if decision.kind == "commit" else None
                if self.pending_retry is None:
                    self.local_option_rejections += 1
                    reason = "terminal_option_rejected_replan"
        else:
            watchdog = True
            reason = "watchdog_step_guard"
        loss_active = False
        loss_episodes = 0
        reacquisition_frames = 0
        reacquisition_elapsed_s = []
        public_missing_count = 0
        loss_started_elapsed = None
        for item in privateeval:
            public = item.get("public", {})
            public_missing_count += int(not bool(public.get("detected", False)))
            present_before = bool(item.get("present_before", True))
            present_after = bool(item.get("present_after", present_before))
            if present_before and not present_after:
                loss_episodes += 1
                loss_active = True
                loss_started_elapsed = float(item.get("elapsed_s", 0.0))
            elif loss_active and bool(item.get("target_association", False)) and present_after:
                reacquisition_frames += 1
                reacquisition_elapsed_s.append(max(0.0, float(item.get("elapsed_s", 0.0))
                                                  - float(loss_started_elapsed or 0.0)))
                loss_active = False
        if not correct_complete and not false_complete and not abstain and not watchdog:
            abstain = True
        native_delta, native_cumulative = self._native_count_report()
        return {
            "arm": self.arm, "reason": reason, "correct_complete": bool(correct_complete),
            "false_complete": bool(false_complete), "abstain": bool(abstain),
            "watchdog": bool(watchdog), "probe_count": self.probe_count,
            "fresh_count": self.fresh_count, "terminal_count": self.terminal_count,
            "simtime": float(self.state.elapsed), "planningwall": 0.0,
            "recoveryloss": int(loss_episodes), "stops": int(correct_complete or false_complete),
            "reacquired": bool(reacquisition_frames),
            "reacquisition_frames": int(reacquisition_frames),
            "public_missing_count": int(public_missing_count),
            "reacquisition_elapsed_s": reacquisition_elapsed_s,
            "local_option_rejections": int(self.local_option_rejections),
            "blacklist": [{"state_key": list(state_key), "action_id": action_id}
                           for state_key, action_id in sorted(self.blacklist, key=repr)],
            "decisions": self.decision_rows, "events": self.history,
            "pairs_completed": self.pairs_completed, "pair_position": self.pair_position,
            "pending_retry": self.pending_retry,
            "timing": dict(self.timing),
            "privateeval": privateeval,
            "belief": _json_safe(self.belief.snapshot()),
            "native_counts": native_delta,
            "native_counts_cumulative": native_cumulative,
            "no_deadline": True, "deadline_s": None,
            "protocol": {"train_seed": TRAIN_SEED, "calibration_seed": CALIBRATION_SEED,
                          "test_seed": TEST_SEED, "noise_seed": NOISE_SEED,
                          "risk_gate": self.config.risk_gate,
                          "max_probe_frames_watchdog": self.config.max_probe_frames,
                          "guard_basis": "all_fresh_acquisitions",
                          "terminal_prediction_scope": "local_two_attempt_option_score",
                          "pair_dwell": self.config.pair_dwell,
                          "protocol_revision": self.config.protocol_revision},
        }


def load_model_params(path: str | Path | None) -> ModelParams:
    """Load a frozen belief artifact while tolerating common wrapper schemas."""
    if path is None:
        return ModelParams()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        for key in ("model_params", "params", "belief_model"):
            if isinstance(payload.get(key), Mapping):
                payload = payload[key]
                break
    return ModelParams.from_dict(payload)


__all__ = [
    "AdaptivePlanner", "PlannerConfig", "PlanningDecision", "ProbeCandidate", "ProbeScore",
    "TerminalOption", "DEFAULT_PROBE_IDS", "PROBE_BASES", "TERMINAL_IDS", "TRAIN_SEED",
    "CALIBRATION_SEED", "TEST_SEED", "NOISE_SEED", "load_model_params", "public_action_from_event",
    "public_outcome_branches", "public_acceptance", "planning_decision_to_public_dict",
    "planning_decision_from_public_dict",
]
