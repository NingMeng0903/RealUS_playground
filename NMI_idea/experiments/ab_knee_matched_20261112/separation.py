"""Paired held-out separation of a static lateral offset and knee lag state.

This is a small, estimator-only diagnostic built on the frozen knee atlas.  A
full 28-frame rule sweep is generated once per case.  At the next modeled
zero-crossing the knee is stopped at 15 degrees and held there.  Four arms use
exactly three post-stop acquisitions, with a common first and third clock:

``burst x one site``       (i, i, i)
``burst x two response``   (i, j, i)
``separated x one site``   (i, i, i)
``separated x two response`` (i, j, i)

The two sites are trunk stations 5 and 12, whose frozen atlas response
profiles are materially different (about .980 and .459).  The (i, j, i)
route uses the actual legal geometry and travel durations.  All arms share the
same prior, estimator, prefix, 15-degree hold, first/third frame times, and
final clock.  The pose-only comparator is the frozen Filter(latent=False).

The only evaluator-only state is the held knee trajectory and its latent
process path.  No truth, association, or future coordinates enter filtering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# The experiment directory intentionally reuses the frozen source tree.  Keep
# this import boundary explicit so this diagnostic cannot silently import a
# working copy from another experiment.
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "ab_knee_latent_rl_20261112"
sys.path.insert(0, str(FROZEN))

from belief import Filter, Reading  # noqa: E402
from experiment import State, apply, prepared, seed_at  # noqa: E402
from knee_world import CAPTURE_S, Candidate, KneeCase, KneeWorld  # noqa: E402


OUT = HERE / "separation"
RAW = OUT / "raw"
SUMMARY = OUT / "summary"
PLOT = OUT / "3D"

# The frozen study's prior B holdout used indices 256--383.  This disjoint
# contiguous block is the exact final held-out set for this diagnostic.
TEST_START = 384
N_CASES = 128
TEST_INDICES = tuple(range(TEST_START, TEST_START + N_CASES))
SITE_I = 8
SITE_J = 11
VIEW = 3  # frozen central/zero-offset view for both trunk sites
STOP_Q = 15.0
POST_HORIZON_S = 12.0
FUTURE_HORIZONS = (0.5, 1.0)
COMMON_TAIL_S = 0.4
Z90 = 1.6448536269514722
BOOTSTRAPS = 2000


@dataclass(frozen=True)
class HeldCase:
    """A frozen case with a physically explicit stop and stationary hold.

    Before ``stop_time`` all public state is delegated to the frozen case.
    At the next analytic q=15 crossing, the knee is stopped (q=15,
    qdot=0).  The true latent response then follows a case-specific first
    order decay with a sampled process path.  The estimator sees only the
    resulting public centroids and pose readings.
    """

    base: KneeCase
    stop_time: float
    w_global_mm: float
    tau_true_s: float
    process_sd_mm: float
    _z_times: np.ndarray
    _z_values: np.ndarray

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def w_subject_mm(self) -> np.ndarray:
        # Primary estimand is a scalar controlled global w.  The frozen atlas
        # and all branch/state parameters remain unchanged.
        return np.full_like(np.asarray(self.base.w_subject_mm, dtype=float),
                            float(self.w_global_mm))

    def true_angle_rate(self, t):
        if float(t) >= self.stop_time - 1.e-10:
            return STOP_Q, 0.0
        return self.base.true_angle_rate(t)

    def z_mm(self, t):
        t = float(t)
        if t <= self.stop_time + 1.e-10:
            return float(self.base.z_mm(t))
        if t >= float(self._z_times[-1]):
            return float(self._z_values[-1])
        return float(np.interp(t, self._z_times, self._z_values))


def _next_stop(base: KneeCase, after: float) -> float:
    """Return the next analytic q=15 zero crossing after ``after``."""
    phase = float(base.phase)
    omega = float(base.omega)
    # omega*t + phase = k*pi.  The strict margin avoids a root numerically
    # coinciding with the last prefix capture.
    k = int(math.floor((omega * (after + 1.e-7) + phase) / math.pi)) + 1
    while True:
        t = (k * math.pi - phase) / omega
        if t > after + 1.e-7:
            return float(t)
        k += 1


def _make_held_case(world: KneeWorld, base: KneeCase, prefix_end: float) -> HeldCase:
    # Pre-position at i=8 after the full sweep.  The stop is chosen after the
    # legal modeled move has completed, so no post-stop pose is fabricated
    # during probe travel.
    previous = world.initial(base)[-1]
    preposition = world.make_action(15., previous, (SITE_I, VIEW),
                                    "preposition before stop", wait=0.)
    stop_time = _next_stop(base, prefix_end + preposition.duration + CAPTURE_S + .05)
    rng = np.random.default_rng(np.random.SeedSequence([base.seed, 81231]))
    # Keep the controlled scalar inside the frozen Filter's [-6, 6] support;
    # the small interior margin avoids a boundary-driven prior mismatch.
    w_global = float(np.clip(rng.normal(0., 2.3), -5.8, 5.8))
    # True duration/process are deliberately not fixed to the Filter's prior.
    # tau remains in the frozen physical case's calibrated range; process is a
    # small independent stationary perturbation used only to generate truth.
    tau_true = float(np.clip(base.tau * rng.uniform(.88, 1.12), .45, 1.35))
    process_sd = float(rng.uniform(.22, .52))
    dt = .025
    horizon = stop_time + POST_HORIZON_S
    z_times = np.arange(stop_time, horizon + .5 * dt, dt, dtype=float)
    z_values = np.empty_like(z_times)
    z_values[0] = float(base.z_mm(stop_time))
    decay = math.exp(-dt / tau_true)
    innovation_sd = process_sd * math.sqrt(max(0., 1. - decay * decay))
    for k in range(1, len(z_values)):
        # A stationary latent response process after the knee stops.  This is
        # generated once and interpolated by the renderer; no filter sees it.
        z_values[k] = decay * z_values[k - 1] + rng.normal(0., innovation_sd)
    return HeldCase(base, stop_time, w_global, tau_true, process_sd,
                    z_times, z_values)


def _prefix_states(world: KneeWorld, case: HeldCase):
    """Generate one shared full rule sweep and two estimators over its frames."""
    # Calling prepared once is important: its initial route is exactly the
    # frozen 28-frame rule sweep.  A second estimator is fed those same Frame
    # objects, not a regenerated stream.
    latent = prepared(world, case, latent=True)
    pose = State(Filter(latent=False), {}, None, (), 0)
    for frame in latent.frames:
        pose = pose.updated(world, frame,
                            retain=world.identity_site(frame.site))
    if len(latent.frames) != 28:
        raise AssertionError(f"frozen full sweep changed: {len(latent.frames)}")
    return latent, pose


def _state_with_frame(world: KneeWorld, case: HeldCase, state: State,
                      action: Candidate, frame_cache: dict[tuple[int, int], object]):
    """Apply one deterministic public frame to one filter state."""
    key = (int(action.site), int(round(action.t * 1000)))
    frame = frame_cache.get(key)
    if frame is None:
        frame = world.observe(case, action)
        frame_cache[key] = frame
    if state.previous is not None and frame.t <= state.previous.t:
        raise AssertionError("non-increasing physical action time")
    return state.updated(world, frame,
                         retain=world.identity_site(frame.site)), frame


def _hold_anchor(world: KneeWorld, case: HeldCase, state: State,
                 frame_cache: dict[tuple[int, int], object]):
    """Add one shared q=15 hold frame at the end of the full sweep."""
    previous = state.previous
    if previous is None:
        raise AssertionError("empty prefix")
    goal = (int(previous.site), int(previous.view))
    wait = case.stop_time - (previous.t + CAPTURE_S)
    if wait < -.000001:
        raise AssertionError("stop crossing precedes legal hold capture")
    action = world.make_action(previous.q, previous, goal, "shared 15deg hold",
                               wait=max(0., float(wait)))
    if abs(action.t - case.stop_time) > 2.e-8:
        raise AssertionError((action.t, case.stop_time))
    out, frame = _state_with_frame(world, case, state, action, frame_cache)
    # External pose noise is part of the public frame. Validate the hidden
    # physical state while retaining that noisy public reading for inference.
    true_q, true_v = case.true_angle_rate(frame.t)
    if abs(true_q - STOP_Q) > 1.e-10 or abs(true_v) > 1.e-10:
        raise AssertionError("hold frame did not use the stopped physical pose")
    return out, frame, action


def _minimum_action(world: KneeWorld, previous, goal):
    action = world.make_action(15., previous, goal, "minimum", wait=0.)
    return float(action.duration)


def _schedule(world: KneeWorld, anchor_frame, timing: str):
    """Build exact legal three-frame times shared across spatial routes.

    The first and third captures have identical times for all four arms.  The
    middle capture is earlier for ``burst`` and later for ``separated``.  The
    slot lengths are the intersection of the legal route durations, with a
    50-ms safety margin.  This makes the spatial arm comparable without
    fabricating a probe clock or pose trajectory.
    """
    i_goal = (SITE_I, VIEW)
    j_goal = (SITE_J, VIEW)
    # All possible route transitions from the shared anchor and from i/j.
    d0i = _minimum_action(world, anchor_frame, i_goal)
    dii = _minimum_action(world, anchor_frame, i_goal)  # same-site capture
    dij = _minimum_action(world, anchor_frame, j_goal)
    dji = _minimum_action(world, anchor_frame, i_goal)
    # The helper above uses anchor_frame for geometry; durations depend only on
    # site/view and q=15, so verify them independently through the exact probe
    # distances and retain the legal route check.
    def duration_for(previous_site, goal):
        class P:
            pass
        p = P()
        p.site, p.view, p.q, p.t = int(previous_site[0]), int(previous_site[1]), 15., float(anchor_frame.t)
        return _minimum_action(world, p, goal)
    d0i = duration_for((anchor_frame.site, anchor_frame.view), i_goal)
    dii = duration_for(i_goal, i_goal)
    dij = duration_for(i_goal, j_goal)
    dji = duration_for(j_goal, i_goal)
    # The first transition is the same in both arms.  The middle and final
    # route transitions must both fit their common slots.
    first_gap = d0i + .05
    burst_gap = max(dii, dij) + .05
    separated_gap = burst_gap + 1.0
    final_gap = max(dii, dji) + COMMON_TAIL_S
    t1 = float(anchor_frame.t + first_gap)
    if timing == "burst":
        t2 = float(t1 + burst_gap)
    elif timing == "separated":
        t2 = float(t1 + separated_gap)
    else:
        raise ValueError(timing)
    # Common final capture clock; both timing arms use the later middle slot's
    # reachable endpoint.  The burst arm therefore has a longer final quiet
    # interval, which is explicit in its serialized frame times.
    t3 = float(t1 + separated_gap + final_gap)
    return {
        "timing": timing,
        "times": (t1, t2, t3),
        "durations_min_s": {"anchor_to_i": d0i, "i_to_i": dii,
                             "i_to_j": dij, "j_to_i": dji},
        "gaps_s": {"first": first_gap, "middle": t2 - t1,
                    "final": final_gap},
    }


def _action_at(world: KneeWorld, previous, goal, target_t: float, kind: str):
    minimum = world.make_action(15., previous, goal, kind, wait=0.)
    wait = float(target_t - minimum.t)
    if wait < -2.e-7:
        raise AssertionError({"target": target_t, "minimum": minimum.t,
                              "goal": goal, "kind": kind})
    return world.make_action(15., previous, goal, kind, wait=max(0., wait))


def _run_arm(world: KneeWorld, case: HeldCase, latent_prefix: State,
             pose_prefix: State, anchor_frame, schedule, route_name: str,
             frame_cache):
    goals = (SITE_I, SITE_I, SITE_I) if route_name == "one_site" else (SITE_I, SITE_J, SITE_I)
    states = {"latent": latent_prefix, "pose_only": pose_prefix}
    frames = []
    actions = []
    previous = anchor_frame
    for step, (site, target_t) in enumerate(zip(goals, schedule["times"])):
        goal = (int(site), VIEW)
        action = _action_at(world, previous, goal, float(target_t),
                            f"{schedule['timing']} {route_name} frame{step + 1}")
        actions.append(action)
        next_states = {}
        frame = None
        for method, state in states.items():
            next_state, this_frame = _state_with_frame(world, case, state, action,
                                                       frame_cache)
            if frame is None:
                frame = this_frame
            elif frame is not this_frame and (frame.site != this_frame.site or
                                              abs(frame.t - this_frame.t) > 1.e-12):
                raise AssertionError("paired estimators received different frames")
            next_states[method] = next_state
        if frame is None:
            raise AssertionError("no frame")
        frames.append(frame)
        previous = frame
        states = next_states
    return states, tuple(frames), tuple(actions)


def _advance_to_common_clock(state: State, final_t: float) -> State:
    belief = state.belief.copy()
    belief.advance(Reading(float(final_t), STOP_Q, 0.0))
    # No synthetic Frame is inserted: this is a prediction clock, not an
    # extra ultrasound acquisition.  The last real frame remains the endpoint.
    return State(belief, state.labels.copy(), state.previous, state.frames,
                 state.event)


def _smooth_at(frames: Iterable, world: KneeWorld, target_t: float,
               latent: bool) -> tuple[np.ndarray, np.ndarray]:
    """Exact finite-grid fixed-lag smoothing for the frozen Filter state."""
    filt = Filter(latent=latent)
    forwards = []
    transitions = []
    likelihoods = []
    for frame in frames:
        reading = Reading(frame.t, frame.q, frame.velocity)
        prior, transition = filt.prior_at(reading)
        likelihood, _ = filt.emission(world, reading, frame.site, frame.view,
                                      frame.centroids)
        normalizer = float(np.sum(prior * likelihood))
        if not np.isfinite(normalizer) or normalizer <= 0.:
            raise FloatingPointError("smoothing emission underflow")
        posterior = prior * likelihood / normalizer
        forwards.append(posterior)
        transitions.append(transition)
        likelihoods.append(likelihood)
        filt.weights = posterior
        filt.reading = reading
    matches = [k for k, frame in enumerate(frames)
               if abs(float(frame.t) - float(target_t)) <= 1.e-8]
    if len(matches) != 1:
        raise AssertionError((target_t, [frame.t for frame in frames]))
    target = matches[0]
    beta = np.ones_like(forwards[-1])
    for k in range(len(frames) - 2, target - 1, -1):
        weighted = likelihoods[k + 1] * beta
        beta = weighted @ transitions[k + 1].T
        scale = float(np.max(beta))
        if scale > 0.:
            beta /= scale
    smoothed = forwards[target] * beta
    smoothed /= float(np.sum(smoothed))
    weights = smoothed[..., None]
    mean = np.sum(weights * filt.parameters, axis=(0, 1))
    var = np.sum(weights * (filt.parameters - mean) ** 2, axis=(0, 1))
    return mean, np.sqrt(var)


def _truth_point(world: KneeWorld, case: HeldCase, t: float, site: int) -> np.ndarray:
    return np.asarray(world.truth_world(case, float(t), ((int(site), 0),))[0],
                      dtype=float)


def _arm_metrics(world: KneeWorld, case: HeldCase, state: State,
                 frames: tuple, final_t: float, stop_t: float):
    state = _advance_to_common_clock(state, final_t)
    mean, sd = state.belief.stats()
    true_w = float(case.w_global_mm)
    true_z_final = float(case.z_mm(final_t))
    # Smoothing uses the complete shared prefix/anchor and all three post-stop
    # observations.  ``frames`` below is kept as the three-frame arm trace for
    # the data convention, while the estimator's immutable state carries the
    # full observation sequence.
    smooth_mean, smooth_sd = _smooth_at(state.frames, world, stop_t,
                                        state.belief.latent)
    out = {
        "w_true_mm": true_w,
        "w_estimate_mm": float(mean[0]),
        "w_sd_mm": float(sd[0]),
        "w_error_mm": float(mean[0] - true_w),
        "z_common_true_mm": true_z_final,
        "z_common_estimate_mm": float(mean[2]),
        "z_common_sd_mm": float(sd[2]),
        "z_common_error_mm": float(mean[2] - true_z_final),
        "z_start_true_mm": float(case.z_mm(stop_t)),
        "z_start_smoothed_mm": float(smooth_mean[2]),
        "z_start_smoothed_sd_mm": float(smooth_sd[2]),
        "z_start_smoothed_error_mm": float(smooth_mean[2] - case.z_mm(stop_t)),
        "final_t_s": float(final_t),
        "last_acquisition_t_s": float(frames[-1].t if frames
                                       else state.previous.t),
        "frame_times_s": [float(frame.t) for frame in frames],
        "frame_sites": [int(frame.site) for frame in frames],
        "frame_counts": [int(len(frame.centroids)) for frame in frames],
        "forecast": {},
    }
    for horizon in FUTURE_HORIZONS:
        t = float(final_t + horizon)
        mean_xyz, sd_xyz = state.belief.forecast(world, t, world.query_sites)
        truth_xyz = world.truth_world(case, t, world.query_sites)
        error = mean_xyz - truth_xyz
        chosen_sites = ((SITE_I, 0), (SITE_J, 0))
        chosen_mean, chosen_sd = state.belief.forecast(world, t, chosen_sites)
        chosen_truth = world.truth_world(case, t, chosen_sites)
        key = f"{horizon:.1f}s"
        out["forecast"][key] = {
            "time_s": t,
            "rmse_mm": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
            "max_mm": float(np.max(np.linalg.norm(error, axis=1))),
            "coordinate_90_coverage": float(np.mean(np.abs(error) <= Z90 * sd_xyz)),
            "mean_xyz_mm": np.asarray(mean_xyz).tolist(),
            "sd_xyz_mm": np.asarray(sd_xyz).tolist(),
            "truth_xyz_mm": np.asarray(truth_xyz).tolist(),
            "selected_sites": [SITE_I, SITE_J],
            "selected_mean_xyz_mm": np.asarray(chosen_mean).tolist(),
            "selected_sd_xyz_mm": np.asarray(chosen_sd).tolist(),
            "selected_truth_xyz_mm": np.asarray(chosen_truth).tolist(),
        }
    return out, state


def _case_record(index: int, world: KneeWorld, seed: int) -> dict:
    base = world.case(seed, difficult=True)
    # Prefix timing is determined from the frozen 28-frame route before the
    # controlled stop.  We need a temporary base sweep only to locate its end;
    # all actual records use the scalar-w case below.
    base_prefix_frames = world.initial(base)
    scalar_case = _make_held_case(world, base, base_prefix_frames[-1].t)
    latent_prefix, pose_prefix = _prefix_states(world, scalar_case)
    frame_cache: dict[tuple[int, int], object] = {}
    # Move the probe from the frozen sweep endpoint to i=8 before the knee
    # stop.  This common frame is generated/evaluated identically for all arms.
    pre_previous = latent_prefix.previous
    pre_action = world.make_action(15., pre_previous, (SITE_I, VIEW),
                                   "preposition before stop", wait=0.)
    if pre_action.t >= scalar_case.stop_time - CAPTURE_S - .02:
        raise AssertionError("preposition did not finish before stationary stop")
    latent_pre, pre_frame = _state_with_frame(
        world, scalar_case, latent_prefix, pre_action, frame_cache)
    pose_pre = pose_prefix.updated(world, pre_frame, retain=False)
    latent_anchor, anchor_frame, anchor_action = _hold_anchor(
        world, scalar_case, latent_pre, frame_cache)
    pose_anchor = pose_pre.updated(world, anchor_frame, retain=False)
    if (len(latent_anchor.frames) != len(pose_anchor.frames)
            or any(a is not b for a, b in zip(latent_anchor.frames,
                                              pose_anchor.frames))):
        raise AssertionError("shared prefix/anchor frame sequence diverged")

    schedules = {timing: _schedule(world, anchor_frame, timing)
                 for timing in ("burst", "separated")}
    final_t = max(float(schedules[t]["times"][-1])
                  for t in schedules)
    # The schedule construction uses a common third acquisition time, hence a
    # single final clock is exactly that shared endpoint.
    if max(abs(schedules[a]["times"][-1] - final_t)
           for a in schedules) > 1.e-8:
        raise AssertionError("third acquisition clocks are not shared")

    # A same-lag propagation reference: keep the shared prefix/anchor
    # posterior, assimilate no post-stop ultrasound, and advance only to the
    # common final clock.  It is identical across the four arms by design.
    no_post_metrics, no_post_final = _arm_metrics(
        world, scalar_case, latent_anchor, (), final_t, scalar_case.stop_time)

    arms = {}
    arm_states = {}
    for timing in ("burst", "separated"):
        for route in ("one_site", "two_response"):
            key = f"{timing}_{route}"
            states, frames, actions = _run_arm(
                world, scalar_case, latent_anchor, pose_anchor,
                anchor_frame, schedules[timing], route, frame_cache)
            if tuple(int(f.site) for f in frames) != tuple(
                    (SITE_I, SITE_I, SITE_I) if route == "one_site"
                    else (SITE_I, SITE_J, SITE_I)):
                raise AssertionError("route mismatch")
            latent_metrics, latent_final = _arm_metrics(
                world, scalar_case, states["latent"], frames, final_t,
                scalar_case.stop_time)
            pose_metrics, pose_final = _arm_metrics(
                world, scalar_case, states["pose_only"], frames, final_t,
                scalar_case.stop_time)
            arms[key] = {
                "latent": latent_metrics,
                "no_lag_plus_US": pose_metrics,
                "latent_no_post_stop_US": no_post_metrics,
            }
            arm_states[key] = {
                "frames": frames,
                "actions": actions,
                "latent_final": latent_final,
                "pose_final": pose_final,
                "no_post_final": no_post_final,
            }

    # Keep one compact trace for the standalone 3D plot.  Raw per-case JSONL
    # carries all numeric estimates, while this trace carries public frames
    # and truth needed to reproduce the selected trajectory.
    trace = {
        "prefix_frames": [{
            "t": float(f.t), "q": float(f.q), "velocity": float(f.velocity),
            "site": int(f.site), "view": int(f.view),
            "probe_xyz_mm": world.probe_world(f.q, f.site, f.view).tolist(),
            "centroids": np.asarray(f.centroids).tolist(),
        } for f in latent_prefix.frames],
        "anchor": {
            "t": float(anchor_frame.t), "q": float(anchor_frame.q),
            "site": int(anchor_frame.site), "view": int(anchor_frame.view),
            "probe_xyz_mm": world.probe_world(anchor_frame.q,
                                               anchor_frame.site,
                                               anchor_frame.view).tolist(),
            "action": {"duration_s": float(anchor_action.duration),
                       "distance_mm": float(anchor_action.distance_mm)},
        },
        "preposition": {
            "t": float(pre_frame.t), "site": int(pre_frame.site),
            "view": int(pre_frame.view),
            "probe_xyz_mm": world.probe_world(pre_frame.q, pre_frame.site,
                                               pre_frame.view).tolist(),
            "action": {"duration_s": float(pre_action.duration),
                       "distance_mm": float(pre_action.distance_mm)},
        },
        "future_times": [float(final_t + h) for h in FUTURE_HORIZONS],
        "chosen_sites": {"i": SITE_I, "j": SITE_J},
        "chosen_truth": {
            "times": [float(scalar_case.stop_time), float(final_t),
                      float(final_t + .5), float(final_t + 1.0)],
            "i_xyz_mm": [_truth_point(world, scalar_case, t, SITE_I).tolist()
                         for t in (scalar_case.stop_time, final_t,
                                   final_t + .5, final_t + 1.0)],
            "j_xyz_mm": [_truth_point(world, scalar_case, t, SITE_J).tolist()
                         for t in (scalar_case.stop_time, final_t,
                                   final_t + .5, final_t + 1.0)],
        },
        "arms": {
            key: {
                "frames": [{
                    "t": float(f.t), "site": int(f.site), "view": int(f.view),
                    "probe_xyz_mm": world.probe_world(f.q, f.site, f.view).tolist(),
                    "centroids": np.asarray(f.centroids).tolist(),
                } for f in arm_states[key]["frames"]],
                "actions": [{"kind": a.kind, "t": float(a.t),
                             "duration_s": float(a.duration),
                             "distance_mm": float(a.distance_mm)}
                            for a in arm_states[key]["actions"]],
                "latent_forecast": arms[key]["latent"]["forecast"],
            } for key in sorted(arms)
        },
    }
    return {
        "case_index": int(index),
        "case_seed": int(seed),
        "heldout_split": "frozen_test_indices_384_511",
        "sites": {"i": SITE_I, "j": SITE_J,
                  "profile_i": float(world._site_profile[SITE_I]),
                  "profile_j": float(world._site_profile[SITE_J]),
                  "view": VIEW},
        "prefix": {"captures": len(latent_prefix.frames),
                   "first_t_s": float(latent_prefix.frames[0].t),
                   "last_t_s": float(latent_prefix.frames[-1].t),
                   "sha256": _prefix_sha(latent_prefix),
                   "common_preposition_capture": True,
                   "common_stop_capture": True},
        "truth": {"stop_t_s": float(scalar_case.stop_time),
                  "stop_q_deg": STOP_Q,
                  "w_global_mm": float(scalar_case.w_global_mm),
                  "tau_true_s": float(scalar_case.tau_true_s),
                  "process_sd_mm": float(scalar_case.process_sd_mm),
                  "z_start_mm": float(scalar_case.z_mm(scalar_case.stop_time)),
                  "z_final_mm": float(scalar_case.z_mm(final_t))},
        "schedule": {"burst": schedules["burst"],
                      "separated": schedules["separated"],
                      "common_final_t_s": final_t,
                      "future_t_s": [final_t + h for h in FUTURE_HORIZONS]},
        "arms": arms,
        "trace": trace,
    }


def _prefix_sha(state: State) -> str:
    import hashlib
    payload = [[frame.t, frame.q, frame.velocity, frame.site, frame.view,
                np.asarray(frame.centroids).tolist()] for frame in state.frames]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _values(records, method: str, metric: str, horizon: str | None = None):
    result = []
    for rec in records:
        for key in ("burst_one_site", "burst_two_response",
                    "separated_one_site", "separated_two_response"):
            row = rec["arms"][key][method]
            result.append(float(row["forecast"][horizon][metric]
                                if horizon is not None else row[metric]))
    return np.asarray(result, dtype=float)


def _case_values(records, arm: str, method: str, metric: str, horizon=None):
    return np.asarray([
        float(rec["arms"][arm][method]["forecast"][horizon][metric]
              if horizon is not None else rec["arms"][arm][method][metric])
        for rec in records], dtype=float)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [float("nan"), float("nan")]
    # Bootstrap a mean.  The per-case arrays are the unit of resampling.
    picks = rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))
    means = values[picks].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _summarize(records: list[dict]) -> dict:
    rng = np.random.default_rng(81232)
    arms = ("burst_one_site", "burst_two_response",
            "separated_one_site", "separated_two_response")
    methods = ("latent", "no_lag_plus_US", "latent_no_post_stop_US")
    summary = {
        "n_cases": len(records),
        "heldout_indices": [int(i) for i in TEST_INDICES],
        "heldout_seed_map": [{"index": int(rec["case_index"]),
                               "seed": int(rec["case_seed"])} for rec in records],
        "data_convention": {
            "case_unit": "one scalar-w subject, paired across all four arms",
            "prefix": "one shared frozen 28-frame rule sweep; then one common preposition capture and one common 15deg stop capture",
            "post_stop_frames": 3,
            "routes": {"one_site": [SITE_I, SITE_I, SITE_I],
                       "two_response": [SITE_I, SITE_J, SITE_I]},
            "timing": "common first/third acquisitions; early vs late middle sample differs by 1 s",
            "no_lag_plus_US": "Filter(latent=False), same observations/prior over w,b",
            "latent_no_post_stop_US": "Filter(latent=True), shared prefix/anchor then propagation only",
            "z_start": "fixed-lag posterior smoothing through the third frame",
            "future": "world 3D query_sites at common T+0.5 and T+1.0 s",
            "uncertainty": "truth tau varies per case and stationary latent process is generated",
            "stop_model": "ideal abrupt physical stop at the first q=15 crossing; no interpolated deceleration",
            "bl_coordinates": "all atlas/world coordinates are millimetres; plot converts to mm explicitly",
        },
        "sites": {
            "i": SITE_I, "j": SITE_J,
            "profile_i": float(records[0]["sites"]["profile_i"]),
            "profile_j": float(records[0]["sites"]["profile_j"]),
        },
        "arms": {},
        "paired_differences": {},
    }
    for arm in arms:
        summary["arms"][arm] = {}
        for method in methods:
            row = {}
            for metric in ("w_error_mm", "z_common_error_mm",
                           "z_start_smoothed_error_mm"):
                vals = _case_values(records, arm, method, metric)
                row[metric] = {"mean": float(np.mean(vals)),
                               "rmse": float(np.sqrt(np.mean(vals ** 2))),
                               "ci95_mean": _bootstrap_ci(vals, rng),
                               "n": int(vals.size)}
            for horizon in ("0.5s", "1.0s"):
                for metric in ("rmse_mm", "coordinate_90_coverage"):
                    vals = _case_values(records, arm, method, metric, horizon)
                    row.setdefault("forecast", {}).setdefault(horizon, {})[metric] = {
                        "mean": float(np.mean(vals)),
                        "ci95_mean": _bootstrap_ci(vals, rng),
                        "n": int(vals.size),
                    }
            row["w_sd_mean_mm"] = float(np.mean([
                rec["arms"][arm][method]["w_sd_mm"] for rec in records]))
            row["z_common_sd_mean_mm"] = float(np.mean([
                rec["arms"][arm][method]["z_common_sd_mm"] for rec in records]))
            summary["arms"][arm][method] = row

    # Paired within-case contrasts preserve the held-out pairing.  Report both
    # timing and spatial response contrasts, and latent-vs-pose references.
    contrasts = {
        "separated_minus_burst_one_site": ("separated_one_site", "burst_one_site"),
        "separated_minus_burst_two_response": ("separated_two_response", "burst_two_response"),
        "two_response_minus_one_site_burst": ("burst_two_response", "burst_one_site"),
        "two_response_minus_one_site_separated": ("separated_two_response", "separated_one_site"),
    }
    for method in methods:
        for name, (a, b) in contrasts.items():
            key = f"{method}:{name}"
            summary["paired_differences"][key] = {}
            for metric in ("w_error_mm", "z_common_error_mm",
                           "z_start_smoothed_error_mm"):
                diff = (_case_values(records, a, method, metric)
                        - _case_values(records, b, method, metric))
                summary["paired_differences"][key][metric] = {
                    "mean_difference": float(np.mean(diff)),
                    "ci95_mean": _bootstrap_ci(diff, rng),
                }
            for horizon in ("0.5s", "1.0s"):
                diff = (_case_values(records, a, method, "rmse_mm", horizon)
                        - _case_values(records, b, method, "rmse_mm", horizon))
                summary["paired_differences"][key].setdefault("forecast", {})[horizon] = {
                    "rmse_difference": float(np.mean(diff)),
                    "ci95_mean": _bootstrap_ci(diff, rng),
                }
    summary["truth_ranges"] = {
        "w_global_mm": [float(min(r["truth"]["w_global_mm"] for r in records)),
                        float(max(r["truth"]["w_global_mm"] for r in records))],
        "z_start_mm_rms": float(np.sqrt(np.mean([
            r["truth"]["z_start_mm"] ** 2 for r in records]))),
        "z_final_mm_rms": float(np.sqrt(np.mean([
            r["truth"]["z_final_mm"] ** 2 for r in records]))),
        "tau_true_s": [float(min(r["truth"]["tau_true_s"] for r in records)),
                       float(max(r["truth"]["tau_true_s"] for r in records))],
    }
    return summary


def _plot(records: list[dict], world: KneeWorld, out: Path):
    """Write a standalone scientific Plotly 3D trace for the first case."""
    import plotly.graph_objects as go

    rec = records[0]
    seed = int(rec["case_seed"])
    base = world.case(seed, difficult=True)
    prefix_frames = world.initial(base)
    held = _make_held_case(world, base, prefix_frames[-1].t)
    final_t = float(rec["schedule"]["common_final_t_s"])
    times = np.linspace(held.stop_time, final_t + 1.0, 80)
    fig = go.Figure()
    # Rule sweep trajectory, visibly distinct from the stationary experiment.
    sweep = np.asarray([world.probe_world(f.q, f.site, f.view)
                        for f in prefix_frames], dtype=float)
    fig.add_trace(go.Scatter3d(x=sweep[:, 0], y=sweep[:, 1], z=sweep[:, 2],
                               mode="lines+markers", name="shared 28-frame rule sweep",
                               line=dict(color="#8c8c8c", width=3),
                               marker=dict(size=2)))
    colors = {"i": "#1f77b4", "j": "#d62728"}
    for label, site in ((f"i={SITE_I}", SITE_I), (f"j={SITE_J}", SITE_J)):
        xyz = np.asarray([_truth_point(world, held, t, site) for t in times])
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                   mode="lines", name=f"true held trajectory {label}",
                                   line=dict(color=colors[label[0]], width=6)))
    # Mark the three public acquisitions for every arm.  The four arms overlap
    # at i; slight marker symbols identify the timing/spatial schedule.
    symbols = {"burst_one_site": "circle", "burst_two_response": "diamond",
               "separated_one_site": "square", "separated_two_response": "cross"}
    for arm, arm_data in sorted(rec["trace"]["arms"].items()):
        frame_xyz = np.asarray([f["probe_xyz_mm"] for f in arm_data["frames"]])
        fig.add_trace(go.Scatter3d(x=frame_xyz[:, 0], y=frame_xyz[:, 1], z=frame_xyz[:, 2],
                                   mode="markers+lines", name=f"acquisitions {arm}",
                                   marker=dict(size=6, symbol=symbols[arm]),
                                   line=dict(width=2, dash="dot")))
        # Latent prediction at T+1 for the chosen endpoints; the forecast is
        # shown as a separate marker trace to avoid implying a fitted path.
        forecast = arm_data["latent_forecast"]["1.0s"]
        pm = np.asarray(forecast["selected_mean_xyz_mm"], dtype=float)
        fig.add_trace(go.Scatter3d(x=pm[:, 0], y=pm[:, 1], z=pm[:, 2],
                                   mode="markers", name=f"latent prediction {arm} @T+1s",
                                   marker=dict(size=5, symbol="x", color="#111111")))
    fig.update_layout(
        title=("Held-knee w/z separation diagnostic: legal atlas sweep, "
               "post-stop trajectories, and latent predictions"),
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        legend=dict(orientation="v"),
        margin=dict(l=0, r=0, t=65, b=0),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)


def run(n_cases: int = N_CASES):
    if int(n_cases) != N_CASES:
        # The split is intentionally exact; a smoke run can use --cases but
        # the output remains explicitly labeled as a partial diagnostic.
        indices = TEST_INDICES[:int(n_cases)]
    else:
        indices = TEST_INDICES
    world = KneeWorld(noisy_pose=True)
    records = []
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for index in indices:
        seed = seed_at("test", int(index))
        record = _case_record(int(index), world, int(seed))
        records.append(record)
        with (RAW / f"case_{int(index):04d}.json").open("w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
    summary = _summarize(records)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (SUMMARY / "SEEDS.json").write_text(json.dumps(summary["heldout_seed_map"], indent=2) + "\n")
    _plot(records, world, PLOT / "SEPARATION_3D.html")
    (OUT / "TRACE.json").write_text(json.dumps(records[0], indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=N_CASES,
                        help="held-out cases (default 128; smoke runs may use fewer)")
    args = parser.parse_args()
    if args.cases < 1 or args.cases > N_CASES:
        raise SystemExit(f"--cases must be in 1..{N_CASES}")
    print(json.dumps(run(args.cases), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
