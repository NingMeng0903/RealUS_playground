"""One-layer paired rollout policy for the v4 continuing experiment.

The helper samples only from the public belief and returns aggregate S/F/U
diagnostics. Latent particle, mode, component and noise draws never cross the
planner boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import ndtri

from acoustics import (AXIS_TOLERANCE_RAD, CENTER_TOLERANCE_M, QUALITY_THRESHOLD,
                       ProbeAction, PublicObservation, YAW_SCALE_RAD,
                       _sigmoid)
from belief_model import (CompleteOption, _geometry_terms, _response_stats,
                          predict_complete_option)

N_SELECT = 16
HORIZON = 8
N_VALIDATE = 32
MAX_SIMULATED_PACKETS = 2048
MC_ALPHA = .05


@dataclass(frozen=True)
class RolloutChoice:
    action_id: str
    score: float
    baseline_action_id: str
    switched: bool
    reason: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Latent:
    geometry_index: int
    mode: int
    component: int
    response: np.ndarray
    uniforms: np.ndarray


def _task_direction_proxy(belief: Any, action: ProbeAction) -> float:
    """Cheap task-direction variance reduction used for A_task ties."""
    try:
        left = ProbeAction("left", (-.016, .018), .10, 1.0, "terminal")
        right = ProbeAction("right", (.016, .018), -.10, 1.0, "terminal")
        direction = left.as_feature() - right.as_feature()
        feature = action.as_feature()
        precisions = np.asarray(belief.precisions)
        informations = np.asarray(belief.informations)
        means, covariances = _response_stats(precisions, informations)
        masses = (np.asarray(belief.particle_weights)[:, None, None]
                  * np.asarray(belief.mode_weights)[:, :, None]
                  * np.asarray(belief.component_weights))
        _, _, _, support, _ = _geometry_terms(belief.geometry, action)
        next_present = np.asarray((1.0 - float(belief.params.present_to_lost),
                                   float(belief.params.lost_to_present)))
        target_mass = (float(belief.params.target_detect_probability)
                       * support.astype(float)[:, None, None]
                       * next_present[None, :, None])
        cross = np.einsum("...i,i->...", covariances @ feature, direction)
        variance = np.einsum("i,...ij,j->...", feature, covariances, feature)
        variance = variance + float(belief.params.sensor_variance(action))
        terms = masses * target_mass * (cross * cross) / np.maximum(variance, 1.0e-12)
        return float(np.sum(terms))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return 0.0


def task_action_scores(belief: Any, actions: Sequence[ProbeAction]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A_task correct/false masses and its cheap tie proxy."""
    try:
        predictions = tuple(belief.batched_acceptance(actions))
    except (AttributeError, TypeError, ValueError):
        predictions = tuple(belief.public_predict(action) for action in actions)
    correct = np.asarray([float(getattr(p, "p_correct_accept", 0.0))
                          for p in predictions], dtype=float)
    false = np.asarray([float(getattr(p, "p_false_accept", 0.0))
                        for p in predictions], dtype=float)
    proxy = np.asarray([_task_direction_proxy(belief, action) for action in actions], dtype=float)
    return correct, false, proxy


def choose_task_index(belief: Any, actions: Sequence[ProbeAction]) -> int:
    correct, _, proxy = task_action_scores(belief, actions)
    best = float(np.max(correct)) if len(correct) else 0.0
    tied = np.flatnonzero(np.abs(correct - best) <= 1.0e-10)
    if len(tied) == 0:
        return 0
    return int(min(
        tied,
        # Callers pass the canonical 12-action order; index is therefore the
        # final canonical-ID tie break after proxy and shorter dwell.
        key=lambda index: (-float(proxy[index]), float(actions[index].duration_s),
                           int(index))))


def _packet_accepts(packet: PublicObservation, params: Any) -> bool:
    return bool(packet.detected and np.isfinite(packet.quality)
                and np.isfinite(packet.center_offset_m)
                and np.isfinite(packet.axis_angle_rad)
                and packet.quality >= float(getattr(params, "public_quality_threshold", QUALITY_THRESHOLD))
                and abs(packet.center_offset_m) <= float(getattr(params, "center_tolerance_m", CENTER_TOLERANCE_M))
                and abs(packet.axis_angle_rad) <= float(getattr(params, "axis_tolerance_rad", AXIS_TOLERANCE_RAD))
                and packet.confidence >= float(getattr(params, "public_confidence_threshold", .65)))


def _draw_latents(belief: Any, n: int, horizon: int, seed: int) -> tuple[_Latent, ...]:
    rng = np.random.default_rng(int(seed))
    particle_weights = np.asarray(belief.particle_weights, dtype=float)
    particle_weights = particle_weights / max(float(particle_weights.sum()), 1.0e-12)
    latents = []
    for _ in range(int(n)):
        geometry_index = int(rng.choice(len(particle_weights), p=particle_weights))
        mode_weights = np.asarray(belief.mode_weights[geometry_index], dtype=float)
        mode_weights = mode_weights / max(float(mode_weights.sum()), 1.0e-12)
        mode = int(rng.choice(2, p=mode_weights))
        component_weights = np.asarray(belief.component_weights[geometry_index, mode], dtype=float)
        component_weights = component_weights / max(float(component_weights.sum()), 1.0e-12)
        component = int(rng.choice(len(component_weights), p=component_weights))
        mean, covariance = _response_stats(
            belief.precisions[geometry_index, mode, component][None],
            belief.informations[geometry_index, mode, component][None])
        response = rng.multivariate_normal(mean[0], covariance[0])
        latents.append(_Latent(geometry_index, mode, component, response,
                               rng.random((int(horizon), 8))))
    return tuple(latents)


def _advance_mode(mode: int, u: float, params: Any) -> int:
    if mode == 0:
        return 0 if float(u) < 1.0 - float(params.present_to_lost) else 1
    return 0 if float(u) < float(params.lost_to_present) else 1


def _sample_packet(belief: Any, action: ProbeAction, latent: _Latent,
                   step: int, event_id: str,
                   geometry_source: np.ndarray | None = None) -> tuple[PublicObservation, bool]:
    params = belief.params
    geometry = (np.asarray(geometry_source, dtype=float).reshape(1, -1)
                if geometry_source is not None else
                belief.geometry[latent.geometry_index:latent.geometry_index + 1])
    center, axis, quality, support, gate = _geometry_terms(geometry, action)
    center, axis, quality = float(center[0]), float(axis[0]), float(quality[0])
    supported = bool(support[0]) and latent.mode == 0
    target_probability = float(params.target_detect_probability) if supported else 0.0
    u = latent.uniforms[min(int(step), len(latent.uniforms) - 1)]
    target = float(u[1]) < target_probability
    # SENSOR_V2 draws clutter conditionally after target detection fails.
    # "target" has already been sampled, so multiplying by
    # "1-target_probability" here would double-count the branch.
    clutter_probability = float(params.clutter_probability)
    clutter = (not target) and float(u[2]) < clutter_probability
    detected = target or clutter
    if not detected:
        packet = PublicObservation(event_id, "rollout_public", action.window,
                                   float(action.duration_s),
                                   "rollout", False, 0.0, float("nan"), float("nan"),
                                   float("nan"), "missing")
        return packet, False
    if target:
        h = np.asarray((1.0, float(action.tilt_rad) / YAW_SCALE_RAD), dtype=float)
        quality_value = quality + float(h @ latent.response)
        quality_value += math.sqrt(max(float(params.sensor_variance(action)), 1.0e-12)) * float(ndtri(np.clip(u[3], 1.0e-8, 1.0 - 1.0e-8)))
        center_value = center + math.sqrt(max(float(params.center_variance(action)), 1.0e-12)) * float(ndtri(np.clip(u[4], 1.0e-8, 1.0 - 1.0e-8)))
        axis_value = axis + math.sqrt(max(float(params.axis_variance(action)), 1.0e-12)) * float(ndtri(np.clip(u[5], 1.0e-8, 1.0 - 1.0e-8)))
    else:
        quality_value = float(params.clutter_quality_mean) + float(params.clutter_quality_sd) * float(ndtri(np.clip(u[3], 1.0e-8, 1.0 - 1.0e-8)))
        center_value = (2.0 * float(u[4]) - 1.0) * float(params.clutter_center_halfwidth)
        axis_value = (2.0 * float(u[5]) - 1.0) * float(params.clutter_axis_halfwidth)
    packet = PublicObservation(
        event_id, "rollout_public", action.window, float(action.duration_s),
        "rollout", True,
        float(_sigmoid((quality_value - .34) / .14)), float(quality_value),
        float(center_value), float(axis_value), "candidate")
    accepted = _packet_accepts(packet, params)
    # This target/geometry result is internal to the rollout simulator only.
    correct = bool(accepted and target and bool(gate[0])
                   and quality_value >= float(getattr(params, "quality_threshold", QUALITY_THRESHOLD)))
    return packet, correct


def _stable_seed(root: int, *parts: Any) -> int:
    payload = json.dumps([int(root), *parts], sort_keys=True, default=str,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _state_key(state: Any) -> tuple[Any, ...]:
    return tuple(getattr(state, "key", ()))


def _action_from_proposal(proposal: Any, fallback: ProbeAction) -> ProbeAction:
    payload = getattr(proposal, "action", None)
    if not isinstance(payload, dict):
        return fallback
    try:
        goal = tuple(float(value) for value in
                     payload.get("goal_local", (*fallback.xy_m, fallback.yaw_rad)))
        xy = goal[:2] if len(goal) >= 2 else fallback.xy_m
        yaw = (payload.get("yaw_rad", goal[2])
               if len(goal) >= 3 else fallback.yaw_rad)
        return ProbeAction(
            str(payload.get("window", fallback.window)),
            tuple(float(value) for value in xy),
            float(yaw),
            float(payload.get("dwell_s", fallback.duration_s)),
            str(payload.get("kind", fallback.kind)),
        )
    except (TypeError, ValueError):
        return fallback


def _proposal(planner: Any, state: Any, action_id: str, fallback: ProbeAction,
              cache: dict[tuple[tuple[Any, ...], str, float],
                           tuple[bool, ProbeAction, Any, str]]) -> tuple[bool, ProbeAction, Any, str]:
    """Get an unchecked public route proposal; never call checked/execute."""
    key = (_state_key(state), str(action_id),
           float(getattr(state, "elapsed", 0.0)))
    cached = cache.get(key)
    if cached is not None:
        return cached
    propose = getattr(planner.adapter, "propose", None)
    if not callable(propose):
        # Tiny public test adapters have no mechanical proposal facade.  Keep
        # their deterministic state as a harmless public fallback.
        result = (True, fallback, state, "public_goal_geometry_only")
        cache[key] = result
        return result
    proposal = propose(state, str(action_id))
    available = bool(getattr(proposal, "available",
                              getattr(proposal, "feasible", False)))
    action = _action_from_proposal(proposal, fallback)
    output = getattr(proposal, "predicted_output",
                     getattr(proposal, "output", None))
    status = str(getattr(proposal, "status", "predicted_unknown"))
    result = (bool(available and output is not None), action, output, status)
    cache[key] = result
    return result


def _terminal_option(planner: Any, posterior: Any, state: Any,
                     terminal_actions: dict[str, ProbeAction],
                     proposal_cache: dict[tuple[tuple[Any, ...], str, float],
                                          tuple[bool, ProbeAction, Any, str]]):
    """Return the strongest currently admitted public terminal option."""
    options = []
    for action_id in ("left", "right"):
        first_ok, first_action, first_output, _ = _proposal(
            planner, state, action_id, terminal_actions[action_id], proposal_cache)
        if not first_ok:
            continue
        # Retry proposals follow the first predicted exit, exactly as the
        # external planner does; this remains an unchecked mechanical forecast.
        retry_state = first_output if first_output is not None else state
        second_ok, second_action, _, _ = _proposal(
            planner, retry_state, action_id, first_action, proposal_cache)
        actions = (first_action, second_action) if second_ok else (first_action,)
        option = CompleteOption(actions=actions, max_retries=len(actions),
                                gh_order=33, require_detected_for_success=False)
        prediction = predict_complete_option(posterior, option)
        if bool(prediction.admitted):
            options.append((float(prediction.calibrated_p), action_id, actions,
                            prediction))
    if not options:
        return None
    return max(options, key=lambda row: (row[0], row[3].raw_complete_p, row[1]))


def _available_task_actions(planner: Any, state: Any,
                            actions: Sequence[ProbeAction],
                            action_ids: Sequence[str],
                            proposal_cache: dict[tuple[tuple[Any, ...], str, float],
                                                 tuple[bool, ProbeAction, Any, str]]):
    available = []
    for action_id, fallback in zip(action_ids, actions):
        ok, action, _, _ = _proposal(
            planner, state, action_id, fallback, proposal_cache)
        if ok:
            available.append((str(action_id), action))
    return tuple(available)


def _full_task_library(planner: Any, candidates: Sequence[Any]) -> tuple[tuple[ProbeAction, ...], tuple[str, ...]]:
    """Recover the complete public 12-action library for later predicted exits."""
    try:
        from adaptive_planning import DEFAULT_PROBE_IDS
        action_ids = tuple(str(action_id) for action_id in DEFAULT_PROBE_IDS)
    except ImportError:  # pragma: no cover - package import fallback
        action_ids = tuple(str(candidate.action_id) for candidate in candidates)
    by_id = {str(candidate.action_id): candidate.action for candidate in candidates}
    actions = []
    selected_ids = []
    for action_id in action_ids:
        action = by_id.get(action_id)
        if action is None:
            try:
                action = planner._predicted(action_id, planner.state).action
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        actions.append(action)
        selected_ids.append(action_id)
    return tuple(actions), tuple(selected_ids)


def _simulate_episode(belief: Any, first_action: ProbeAction,
                      first_action_id: str, task_actions: Sequence[ProbeAction],
                      task_action_ids: Sequence[str], planner: Any,
                      latent: _Latent, horizon: int,
                      rollout_id: str) -> tuple[str, int]:
    """Simulate one public-only continuing trajectory.

    The sampled geometry/response latent is private to the evaluator.  The
    tail policy sees only packets, its filtered posterior, and predicted
    mechanical exits.
    """
    posterior = belief.clone()
    mode = int(latent.mode)
    packets = 0
    root_geometry = np.asarray(
        belief.geometry[latent.geometry_index], dtype=float).copy()
    state = getattr(planner, "state", None)
    proposal_cache: dict[tuple[tuple[Any, ...], str, float],
                         tuple[bool, ProbeAction, Any, str]] = {}
    terminal_actions = {
        "left": ProbeAction("left", (-.016, .018), .10, 1.0, "terminal"),
        "right": ProbeAction("right", (.016, .018), -.10, 1.0, "terminal"),
    }
    first_pending = True
    while packets < int(horizon):
        step = int(packets)
        if first_pending:
            first_pending = False
            ok, proposed_action, output, _ = _proposal(
                planner, state, first_action_id, first_action, proposal_cache)
            if not ok:
                return "F", packets
            mode = _advance_mode(
                mode, float(latent.uniforms[min(step, len(latent.uniforms) - 1), 0]),
                posterior.params)
            episode_latent = _Latent(
                latent.geometry_index, mode, latent.component,
                latent.response, latent.uniforms)
            packet, correct = _sample_packet(
                posterior, proposed_action, episode_latent, step,
                f"rollout/{rollout_id}/{step}/first", root_geometry)
            packets += 1
            state = output
            if _packet_accepts(packet, posterior.params):
                return ("S" if correct else "F"), packets
            posterior.update(packet.event_id, proposed_action, packet)
            continue
        # A_task_v4's local .90 gate is the same public decision rule used by
        # the external baseline.  A two-reject option returns here and keeps
        # the predicted exit/posterior rather than ending the mission.
        option = _terminal_option(
            planner, posterior, state, terminal_actions, proposal_cache)
        if option is not None:
            _, _, option_actions, _ = option
            for attempt, action in enumerate(option_actions[:2]):
                action_id = "left" if action.window == "left" else "right"
                ok, proposed_action, output, _ = _proposal(
                    planner, state, action_id, action, proposal_cache)
                if not ok:
                    return "F", packets
                # Visibility transitions happen before every fresh capture.
                capture_step = int(packets)
                mode = _advance_mode(
                    mode, float(latent.uniforms[min(capture_step, len(latent.uniforms) - 1), 0]),
                    posterior.params)
                episode_latent = _Latent(
                    latent.geometry_index, mode, latent.component,
                    latent.response, latent.uniforms)
                packet, correct = _sample_packet(
                    posterior, proposed_action, episode_latent, capture_step,
                    f"rollout/{rollout_id}/{capture_step}/terminal_{attempt}", root_geometry)
                packets += 1
                state = output
                if _packet_accepts(packet, posterior.params):
                    return ("S" if correct else "F"), packets
                posterior.update(packet.event_id, proposed_action, packet)
                if packets >= int(horizon):
                    return "U", packets
            continue

        available = _available_task_actions(
            planner, state, task_actions, task_action_ids, proposal_cache)
        if not available:
            # Operational withdrawal belongs to F in the selection counts;
            # only an otherwise-live endpoint at L=8 is unresolved.
            return "F", packets
        available_actions = tuple(row[1] for row in available)
        action_index = choose_task_index(posterior, available_actions)
        action_id, action = available[action_index]
        ok, proposed_action, output, _ = _proposal(
            planner, state, action_id, action, proposal_cache)
        if not ok:
            return "F", packets
        mode = _advance_mode(
            mode, float(latent.uniforms[min(step, len(latent.uniforms) - 1), 0]),
            posterior.params)
        episode_latent = _Latent(
            latent.geometry_index, mode, latent.component,
            latent.response, latent.uniforms)
        packet, correct = _sample_packet(
            posterior, proposed_action, episode_latent, step,
            f"rollout/{rollout_id}/{step}/probe", root_geometry)
        packets += 1
        state = output
        if _packet_accepts(packet, posterior.params):
            return ("S" if correct else "F"), packets
        posterior.update(packet.event_id, proposed_action, packet)
    return "U", packets


def _bounds(rate: float, n: int, alpha: float = MC_ALPHA) -> tuple[float, float]:
    half = math.sqrt(math.log(2.0 / max(alpha, 1.0e-12)) / (2.0 * max(n, 1)))
    return max(0.0, float(rate) - half), min(1.0, float(rate) + half)


def _summary(outcomes: Sequence[str], baseline: Sequence[str] | None = None,
             *, baseline_unresolved_success: bool = False) -> dict[str, Any]:
    n = max(len(outcomes), 1)
    counts = {key: int(sum(item == key for item in outcomes)) for key in ("S", "F", "U")}
    rates = {key: counts[key] / n for key in counts}
    bounds = {key: _bounds(rates[key], n) for key in counts}
    row: dict[str, Any] = {"n": len(outcomes), "counts": counts,
                           "rates": rates, "bounds": bounds}
    if baseline is not None:
        z = np.asarray([(1.0 if out == "S" else 0.0) -
                        (1.0 if (base == "S" or
                                 (baseline_unresolved_success and base == "U"))
                         else 0.0)
                        for out, base in zip(outcomes, baseline)], dtype=float)
        row["mean_z"] = float(z.mean()) if len(z) else 0.0
        row["z_positive"] = int(np.sum(z > 0.0))
        row["z_negative"] = int(np.sum(z < 0.0))
        row["z_tied"] = int(np.sum(z == 0.0))
    return row


def _selection_bounds(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """Protocol's finite-sample S/F/U bounds, not a confidence interval."""
    n = max(int(metrics.get("n", 0)), 1)
    counts = metrics.get("counts", {})
    return (float(counts.get("S", 0)) / n,
            float(counts.get("S", 0) + counts.get("U", 0)) / n)


def _exact_sign_p(positive: int, negative: int) -> float:
    discordant = int(positive + negative)
    if discordant <= 0 or positive <= 0:
        return 1.0
    return float(sum(math.comb(discordant, k) for k in range(positive, discordant + 1))
                 / (2.0 ** discordant))


def choose_one_layer(planner: Any, candidates: Sequence[Any], *, seed: int | None = None,
                     n_select: int = N_SELECT, horizon: int = HORIZON,
                     n_validate: int = N_VALIDATE,
                     max_packets: int = MAX_SIMULATED_PACKETS) -> RolloutChoice:
    """Screen 12 candidates against A_task and validate one screen winner."""
    if not candidates:
        return RolloutChoice("", 0.0, "", False, "no_public_candidates", {})
    actions = tuple(candidate.action for candidate in candidates)
    action_ids = tuple(str(candidate.action_id) for candidate in candidates)
    tail_actions, tail_action_ids = _full_task_library(planner, candidates)
    baseline_index = choose_task_index(planner.belief, actions)
    baseline = candidates[baseline_index]
    root_seed = int(seed if seed is not None else planner.config.qmc_seed)
    public_context = (tuple(getattr(planner.state, "key", ())),
                      tuple(sorted(getattr(planner.belief, "event_ids", ()))))
    selection_seed = _stable_seed(
        root_seed, "selection", public_context, int(n_select), int(horizon))
    validation_seed = _stable_seed(
        root_seed, "validation", public_context, int(n_validate), int(horizon))
    # Validation is a genuinely independent paired stream.  It never reuses
    # the N_select latents or packet channels, even when a winner is screened.
    screen_latents = _draw_latents(
        planner.belief, n_select, horizon, selection_seed)
    baseline_screen: list[str] = []
    packets_used = 0
    for index, latent in enumerate(screen_latents):
        outcome, packets = _simulate_episode(
            planner.belief, baseline.action, baseline.action_id, tail_actions,
            tail_action_ids, planner, latent, horizon,
            f"base/select/{index}")
        baseline_screen.append(outcome)
        packets_used += packets
    baseline_metrics = _summary(baseline_screen)
    baseline_lower, baseline_upper = _selection_bounds(baseline_metrics)
    # Exact output-preserving pruning must happen before any candidate
    # simulation.  The skipped candidates are intentionally absent from the
    # diagnostics rather than represented by fabricated scores.
    if math.isclose(baseline_upper, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        diagnostics = {
            "method": "B_rollout_one_layer",
            "prediction_scope": "local_rollout_S_F_U_not_mission_probability",
            "baseline_action_id": baseline.action_id,
            "N_select": int(n_select), "L": int(horizon),
            "N_validate": int(n_validate), "max_simulated_packets": int(max_packets),
            "baseline_screen": baseline_metrics,
            "screen_candidates": {baseline.action_id: {
                **baseline_metrics,
                "selection_lower": float(baseline_lower),
                "selection_upper": float(baseline_upper),
                "baseline_reused": True,
            }},
            "screen_winner": baseline.action_id,
            "screen_reason": "baseline_upper_one_pruned",
            "pruned_by_baseline_upper_one": True,
            "pruned_candidate_count": max(0, len(candidates) - 1),
            "pruned_candidate_ids": [str(candidate.action_id) for candidate in candidates
                                    if str(candidate.action_id) != str(baseline.action_id)],
            "selection_seed": int(selection_seed),
            "validation_seed": int(validation_seed),
            "validation_independent": True,
            "simulated_packets": int(packets_used),
            "mc_stability_gate": {
                "screen_lower_gt_baseline_upper": False,
                "validation_exact_p": None,
                "max_packets_respected": bool(packets_used <= max_packets),
            },
        }
        if packets_used > max_packets:
            raise RuntimeError(f"rollout packet budget exceeded: {packets_used}>{max_packets}")
        return RolloutChoice(
            baseline.action_id, float(baseline_lower), baseline.action_id, False,
            "baseline_upper_one_pruned", diagnostics)
    candidate_screen: dict[str, list[str]] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if str(candidate.action_id) == str(baseline.action_id):
            values = list(baseline_screen)
            candidate_screen[candidate.action_id] = values
            candidate_metrics[candidate.action_id] = _summary(values, baseline_screen)
            continue
        values: list[str] = []
        for index, latent in enumerate(screen_latents):
            outcome, packets = _simulate_episode(
                planner.belief, candidate.action, candidate.action_id, tail_actions,
                tail_action_ids, planner, latent, horizon,
                f"{candidate.action_id}/select/{index}")
            values.append(outcome)
            packets_used += packets
        candidate_screen[candidate.action_id] = values
        candidate_metrics[candidate.action_id] = _summary(values, baseline_screen)
    for candidate in candidates:
        lower, upper = _selection_bounds(candidate_metrics[candidate.action_id])
        candidate_metrics[candidate.action_id]["selection_lower"] = lower
        candidate_metrics[candidate.action_id]["selection_upper"] = upper
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate_metrics[candidate.action_id]["selection_lower"]),
            0 if str(candidate.action_id) == str(baseline.action_id) else 1,
            str(candidate.action_id)))
    winner = ordered[0]
    winner_lower = float(candidate_metrics[winner.action_id]["selection_lower"])
    screen_candidate = (winner if str(winner.action_id) != str(baseline.action_id)
                        and winner_lower > baseline_upper else None)
    diagnostics: dict[str, Any] = {
        "method": "B_rollout_one_layer",
        "prediction_scope": "local_rollout_S_F_U_not_mission_probability",
        "baseline_action_id": baseline.action_id,
        "N_select": int(n_select), "L": int(horizon), "N_validate": int(n_validate),
        "max_simulated_packets": int(max_packets),
        "baseline_screen": baseline_metrics,
        "screen_candidates": candidate_metrics,
        "screen_winner": None if screen_candidate is None else screen_candidate.action_id,
        "screen_reason": ("candidate_lower_gt_baseline_upper"
                          if screen_candidate is not None else "baseline_A_task_max_S_tie"),
        "selection_lower_baseline": float(baseline_lower),
        "selection_upper_baseline": float(baseline_upper),
        "pruned_by_baseline_upper_one": False,
        "pruned_candidate_count": 0,
        "selection_seed": int(selection_seed),
        "validation_seed": int(validation_seed),
        "validation_independent": True,
    }
    validated = None
    if screen_candidate is not None:
        validation_latents = _draw_latents(
            planner.belief, n_validate, horizon, validation_seed)
        validation_baseline: list[str] = []
        validation_candidate: list[str] = []
        for index, latent in enumerate(validation_latents):
            outcome, packets = _simulate_episode(
                planner.belief, baseline.action, baseline.action_id, tail_actions,
                tail_action_ids, planner, latent, horizon, f"base/validate/{index}")
            validation_baseline.append(outcome)
            packets_used += packets
            outcome, packets = _simulate_episode(
                planner.belief, screen_candidate.action, screen_candidate.action_id,
                tail_actions, tail_action_ids, planner, latent, horizon,
                f"{screen_candidate.action_id}/validate/{index}")
            validation_candidate.append(outcome)
            packets_used += packets
        validated = _summary(
            validation_candidate, validation_baseline,
            baseline_unresolved_success=True)
        p_value = _exact_sign_p(int(validated["z_positive"]),
                                int(validated["z_negative"]))
        validated["exact_p_one_sided"] = p_value
        validated["mean_z_positive"] = bool(validated.get("mean_z", 0.0) > 0.0)
        validated["switch_criterion"] = bool(
            validated["mean_z_positive"] and p_value <= .05)
    diagnostics["validation"] = validated
    diagnostics["simulated_packets"] = int(packets_used)
    diagnostics["mc_stability_gate"] = {
        "screen_lower_gt_baseline_upper": False if screen_candidate is None else bool(
            candidate_metrics[screen_candidate.action_id]["selection_lower"]
            > baseline_upper),
        "validation_exact_p": None if validated is None else validated.get("exact_p_one_sided"),
        "max_packets_respected": bool(packets_used <= max_packets),
        "hoeffding_bounds_diagnostic_only": True,
    }
    if packets_used > max_packets:
        raise RuntimeError(f"rollout packet budget exceeded: {packets_used}>{max_packets}")
    if not diagnostics["mc_stability_gate"]["screen_lower_gt_baseline_upper"]:
        return RolloutChoice(
            baseline.action_id, float(baseline_lower),
            baseline.action_id, False, "screen_lower_not_above_baseline_upper", diagnostics)
    if validated is not None and bool(validated.get("switch_criterion", False)):
        return RolloutChoice(
            screen_candidate.action_id, float(validated.get("mean_z", 0.0)),
            baseline.action_id, True, "validated_paired_meanZ_and_exact_sign_pass", diagnostics)
    return RolloutChoice(
        baseline.action_id, float(baseline_lower), baseline.action_id,
        False, "validation_meanZ_or_exact_sign_fail", diagnostics)


__all__ = ["RolloutChoice", "choose_one_layer", "choose_task_index",
           "task_action_scores", "N_SELECT", "HORIZON", "N_VALIDATE",
           "MAX_SIMULATED_PACKETS"]
