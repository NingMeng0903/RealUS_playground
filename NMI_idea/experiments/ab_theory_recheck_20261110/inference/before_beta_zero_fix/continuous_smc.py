"""Continuous likelihood-tempered resample--move SMC for the A/B recheck.

This module deliberately lives outside the original observation/planning code.
It treats a public frame as the only evidence and keeps a complete list of
accepted public frames.  A block is assimilated through a sequence of targets

    pi_beta(w) proportional pi_old(w) exp(beta * L_block(w)),

where ``pi_old`` already contains the full previous public history.  The
proposal is the blocked symmetric proposal from ``observation.environment``;
no truth, audit channels, task utility, or planner cache enters the target.

The generic engine is also used by the linear-Gaussian contract tests, so the
tests exercise the same tempering, resampling, and MH code as the geometric
wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import sys
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.special import logsumexp


LADDER_ROOT = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_module_ladder_20261110"
)
if str(LADDER_ROOT) not in sys.path:
    sys.path.insert(0, str(LADDER_ROOT))
from observation import environment as env  # noqa: E402  (fixed external API)


ArrayLike = np.ndarray
LogLike = Callable[[list], np.ndarray]
JointLogLike = Callable[[list], tuple[np.ndarray, np.ndarray]]
Prior = Callable[[object], float]
Proposal = Callable[[object, np.random.Generator, float], object]


def _normalise_weights(weights: Sequence[float], n: int) -> np.ndarray:
    """Validate and normalize a finite nonnegative weight vector."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite, nonnegative, and nonzero")
    return w / np.sum(w)


def raw_ess(weights: Sequence[float]) -> float:
    """Return the raw (pre-resampling) effective sample size."""
    w = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(np.square(w)))


def _cess(weights: np.ndarray, loglike: np.ndarray, delta: float) -> float:
    """Conditional ESS after adding ``delta * loglike`` to normalized weights."""
    if delta == 0.0:
        # Avoid the IEEE invalid operation 0 * (-inf) for particles that are
        # incompatible with a visible frame.  At zero temperature the
        # conditional ESS convention is N, independently of the incoming
        # (possibly nonuniform) weights.
        return float(len(weights))
    lw = np.full(weights.shape, -np.inf, dtype=float)
    positive = weights > 0
    lw[positive] = np.log(weights[positive])
    x1 = lw + delta * loglike
    # Conditional ESS uses the current normalized weights once in the
    # denominator, then multiplies by N.  Using 2*lw here would instead give
    # the ordinary post-update ESS and incorrectly plateau at .7N.
    x2 = lw + 2.0 * delta * loglike
    a = logsumexp(x1)
    b = logsumexp(x2)
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return float(len(weights) * np.exp(2.0 * a - b))


def adaptive_next_beta(
    beta: float,
    weights: Sequence[float],
    loglike: Sequence[float],
    *,
    target_cess: float,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Choose the largest next beta whose conditional ESS meets the target.

    ``weights`` are the normalized weights at the current beta.  The CESS is
    N at delta=0 by convention, even when the incoming weights are nonuniform;
    bisection is used only when beta=1 would fall below target.  A hard-zero
    likelihood support can make CESS discontinuous immediately to the right of
    zero; that case takes an explicit small support step instead of inventing a
    beta root.
    """
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    w = np.asarray(weights, dtype=float)
    ll = np.asarray(loglike, dtype=float)
    if w.ndim != 1 or ll.shape != w.shape:
        raise ValueError("weights and loglike must be one-dimensional and aligned")
    if np.any(np.isnan(ll)):
        raise ValueError("loglike contains NaN")
    remaining = 1.0 - beta
    if remaining <= tol:
        return 1.0, _cess(w, ll, remaining)
    full = _cess(w, ll, remaining)
    if full >= target_cess:
        return 1.0, full

    # Hard-zero likelihoods (e.g. a visible frame with the wrong contact
    # state) can make CESS jump discontinuously at beta=0.  If the positive
    # temperature support already contains fewer particles than the nominal
    # target, no bisection root exists; take a small explicit support step and
    # let the configured .6N resampling rule repair it.
    probe_delta = min(remaining, 1e-12)
    probe_cess = _cess(w, ll, probe_delta)
    if probe_cess < target_cess:
        support_step = min(remaining, max(1e-6, 1e-6 * remaining))
        return beta + support_step, _cess(w, ll, support_step)

    # Find the largest delta with CESS >= target.  A tiny numerical margin
    # prevents a stage from repeating the same beta at an ESS plateau.
    lo, hi = 0.0, remaining
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if _cess(w, ll, mid) >= target_cess:
            lo = mid
        else:
            hi = mid
    delta = max(lo, min(remaining, 1e-12))
    next_beta = min(1.0, beta + delta)
    if next_beta <= beta + tol:
        return beta, _cess(w, ll, 0.0)
    return next_beta, _cess(w, ll, next_beta - beta)


def _systematic_resample(
    particles: list,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list:
    """Systematic resampling; source particles are never mutated."""
    n = len(particles)
    marks = (rng.random() + np.arange(n)) / n
    indices = np.searchsorted(np.cumsum(weights), marks, side="right")
    indices = np.clip(indices, 0, n - 1)
    return [particles[int(i)] for i in indices]


def _finite_log_target(values: np.ndarray) -> np.ndarray:
    """Make target arithmetic robust to legal -inf prior/evidence values."""
    values = np.asarray(values, dtype=float)
    if np.any(np.isnan(values)):
        raise ValueError("log target contains NaN")
    return values


def _mcmc_move(
    particles: list,
    beta: float,
    block_loglike: LogLike,
    history_loglike: LogLike,
    logprior: Prior,
    propose: Proposal,
    rng: np.random.Generator,
    *,
    sweeps: int,
    scales: Sequence[float],
    joint_loglike: JointLogLike | None = None,
) -> tuple[list, float, int]:
    """Apply symmetric MH sweeps at the current tempered target."""
    if sweeps <= 0:
        return particles, 0.0, 0
    if joint_loglike is None:
        current_h = _finite_log_target(np.asarray(history_loglike(particles), float))
        current_b = _finite_log_target(np.asarray(block_loglike(particles), float))
    else:
        current_h, current_b = joint_loglike(particles)
        current_h = _finite_log_target(np.asarray(current_h, float))
        current_b = _finite_log_target(np.asarray(current_b, float))
    current_prior = np.asarray([logprior(p) for p in particles], dtype=float)
    current = _finite_log_target(current_prior + current_h + beta * current_b)
    accepted = 0
    n = len(particles)
    for sweep in range(sweeps):
        scale = float(scales[sweep % len(scales)])
        proposals = [propose(p, rng, scale) for p in particles]
        if joint_loglike is None:
            prop_h = _finite_log_target(np.asarray(history_loglike(proposals), float))
            prop_b = _finite_log_target(np.asarray(block_loglike(proposals), float))
        else:
            prop_h, prop_b = joint_loglike(proposals)
            prop_h = _finite_log_target(np.asarray(prop_h, float))
            prop_b = _finite_log_target(np.asarray(prop_b, float))
        prop_prior = np.asarray([logprior(p) for p in proposals], dtype=float)
        proposed = _finite_log_target(prop_prior + prop_h + beta * prop_b)
        with np.errstate(invalid="ignore"):
            delta = proposed - current
        # -inf - -inf is NaN; two incompatible states should simply reject.
        accept = (np.isfinite(proposed) & ~np.isfinite(current)) | (
            np.isfinite(proposed)
            & (np.isfinite(current) | (current == -np.inf))
            & (np.log(rng.random(n)) < np.minimum(delta, 0.0))
        )
        # A finite target is always accepted over -inf, including when the
        # expression above sees an infinite difference.
        accept |= np.isfinite(proposed) & (current == -np.inf)
        particles = [proposals[i] if bool(accept[i]) else particles[i] for i in range(n)]
        current = np.where(accept, proposed, current)
        accepted += int(np.sum(accept))
    return particles, accepted / float(n * sweeps), accepted


@dataclass
class TemperedUpdate:
    particles: list
    weights: np.ndarray
    diagnostics: dict
    committed: bool


def adaptive_tempered_resample_move(
    particles: Sequence,
    weights: Sequence[float],
    block_loglike: LogLike,
    history_loglike: LogLike,
    logprior: Prior,
    propose: Proposal,
    rng: np.random.Generator,
    *,
    target_fraction: float = 0.70,
    resample_fraction: float = 0.60,
    max_temper_steps: int = 20,
    mh_sweeps: int = 8,
    proposal_scales: Sequence[float] = (1.0, 0.5, 1.8),
    joint_loglike: JointLogLike | None = None,
) -> TemperedUpdate:
    """Perform one transactional adaptive tempered SMC update.

    The returned ``committed`` flag is false if beta=1 was not reached.  In
    that case the input particles and weights have not been mutated and the
    diagnostic contains the partial temperature path.  The caller must not
    append the block to its public history in that case.
    """
    current_particles = list(particles)
    n = len(current_particles)
    if n == 0:
        raise ValueError("particles must be nonempty")
    current_weights = _normalise_weights(weights, n)
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be in (0,1]")
    if not (0.0 < resample_fraction <= 1.0):
        raise ValueError("resample_fraction must be in (0,1]")
    if max_temper_steps <= 0 or mh_sweeps < 0:
        raise ValueError("max_temper_steps must be positive and mh_sweeps nonnegative")

    block_ll = _finite_log_target(np.asarray(block_loglike(current_particles), float))
    if block_ll.shape != (n,):
        raise ValueError(f"block_loglike must return ({n},), got {block_ll.shape}")
    if np.all(~np.isfinite(block_ll)):
        raise ValueError("all particles are incompatible with the new evidence")

    target_cess = target_fraction * n
    threshold = resample_fraction * n
    initial_ess = raw_ess(current_weights)
    # Block-start resampling follows the ordinary ESS threshold (.6N).  CESS
    # itself is N at beta=0, so a .7N CESS target does not justify a .7N
    # preemptive resample.
    initial_resampled = initial_ess + 1e-8 < threshold
    initial_move_acceptance = 0.0
    if initial_resampled:
        current_particles = _systematic_resample(current_particles, current_weights, rng)
        current_weights = np.full(n, 1.0 / n)
        # At beta zero the block contributes nothing.  Move against the old
        # full-history target to remove duplicate resampled particles.
        zero_block = lambda ps: np.zeros(len(ps), dtype=float)
        current_particles, initial_move_acceptance, _ = _mcmc_move(
            current_particles,
            0.0,
            zero_block,
            history_loglike,
            logprior,
            propose,
            rng,
            sweeps=mh_sweeps,
            scales=proposal_scales,
            joint_loglike=joint_loglike,
        )
        block_ll = _finite_log_target(np.asarray(block_loglike(current_particles), float))

    beta = 0.0
    temperature_path = [0.0]
    stage_records = []
    attempted_logz = 0.0
    total_accepted = 0
    total_proposals = 0
    finished = False

    for stage in range(max_temper_steps):
        pre_ess = raw_ess(current_weights)
        stage_target_cess = target_cess
        next_beta, predicted_cess = adaptive_next_beta(
            beta,
            current_weights,
            block_ll,
            target_cess=stage_target_cess,
        )
        if next_beta <= beta:
            # No positive temperature progress was numerically available.
            # Stop explicitly rather than pretending that an unfinished path
            # is a posterior.
            break
        delta_beta = next_beta - beta
        log_current_weights = np.full(current_weights.shape, -np.inf, dtype=float)
        positive_weights = current_weights > 0.0
        log_current_weights[positive_weights] = np.log(current_weights[positive_weights])
        lw = log_current_weights + delta_beta * block_ll
        increment = float(logsumexp(lw))
        if not np.isfinite(increment):
            break
        attempted_logz += increment
        lw -= increment
        current_weights = np.exp(lw)
        post_ess = raw_ess(current_weights)
        resampled = post_ess < threshold
        if resampled:
            current_particles = _systematic_resample(current_particles, current_weights, rng)
            current_weights = np.full(n, 1.0 / n)

        # A move kernel is applied at every stage.  It is target invariant even
        # when the weighted sample was not resampled; weights therefore remain
        # valid and old history remains in every MH target evaluation.
        current_particles, acc_rate, accepted = _mcmc_move(
            current_particles,
            next_beta,
            block_loglike,
            history_loglike,
            logprior,
            propose,
            rng,
            sweeps=mh_sweeps,
            scales=proposal_scales,
            joint_loglike=joint_loglike,
        )
        total_accepted += accepted
        total_proposals += n * mh_sweeps
        beta = next_beta
        temperature_path.append(float(beta))
        stage_records.append(
            {
                "stage": stage + 1,
                "beta": float(beta),
                "delta_beta": float(delta_beta),
                "pre_raw_ess": float(pre_ess),
                "post_raw_ess": float(post_ess),
                "cess": float(predicted_cess),
                "cess_target": float(stage_target_cess),
                "resampled": bool(resampled),
                "mh_acceptance": float(acc_rate),
                "logZ_increment": float(increment),
            }
        )
        # Do not relabel a near-end beta as beta=1: the remaining likelihood
        # increment must actually be applied.  The scheduler returns exactly
        # 1.0 only when it selects the complete remaining increment.
        if beta >= 1.0:
            finished = True
            break
        block_ll = _finite_log_target(np.asarray(block_loglike(current_particles), float))

    final_ess = raw_ess(current_weights)
    diagnostics = {
        "posterior_valid": bool(finished),
        "unfinished": bool(not finished),
        "beta_final": float(beta),
        "temperature_path": temperature_path,
        "stages": stage_records,
        "pre_raw_ess": float(initial_ess),
        "post_raw_ess": float(final_ess),
        "initial_resampled": bool(initial_resampled),
        # Kept as a compatibility alias for the first report; it now means
        # exactly the .6N block-start rule, not the old .7N preemption.
        "initial_preemptive_resample": bool(initial_resampled),
        "initial_mh_acceptance": float(initial_move_acceptance),
        "mh_acceptance": float(total_accepted / max(1, total_proposals)),
        "resample_count": int(sum(bool(s["resampled"]) for s in stage_records)),
        "attempted_logZ_increment": float(attempted_logz),
        "max_temper_steps": int(max_temper_steps),
        "cess_target": float(target_cess),
        "resample_threshold": float(threshold),
        "stage_count": int(len(stage_records)),
    }
    if not finished:
        # Transactional contract: return old state, not a partial-temperature
        # approximation.  The attempted state is intentionally discarded.
        diagnostics["posterior_valid"] = False
        diagnostics["rollback"] = True
        return TemperedUpdate(list(particles), np.asarray(weights, float) / np.sum(weights), diagnostics, False)
    diagnostics["rollback"] = False
    return TemperedUpdate(current_particles, current_weights, diagnostics, True)


def _pose_key(pose: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(pose, dtype=float).reshape(-1)
    return tuple(float(x) for x in arr)


class ContinuousBelief:
    """Public-frame posterior over the continuous geometric world model."""

    def __init__(
        self,
        particles: Sequence,
        weights: Sequence[float] | None = None,
        *,
        seed: int = 720282,
        cess_target: float = 0.70,
        resample_threshold: float = 0.60,
        max_temper_steps: int = 20,
        mh_sweeps: int = 8,
        prediction_backend: str = "numpy",
    ):
        self.particles = list(particles)
        if not self.particles:
            raise ValueError("particles must be nonempty")
        n = len(self.particles)
        self.weights = np.full(n, 1.0 / n) if weights is None else _normalise_weights(weights, n)
        self.rng = np.random.default_rng(seed)
        self.cess_target = float(cess_target)
        self.resample_threshold = float(resample_threshold)
        self.max_temper_steps = int(max_temper_steps)
        self.mh_sweeps = int(mh_sweeps)
        if prediction_backend not in ("numpy", "cuda"):
            raise ValueError("prediction_backend must be 'numpy' or 'cuda'")
        self.prediction_backend = prediction_backend
        self.history: list[tuple[dict, np.ndarray]] = []
        self.seen: set[str] = set()
        self.diagnostics: list[dict] = []
        self.logZ = 0.0

    @property
    def ess(self) -> float:
        return raw_ess(self.weights)

    @staticmethod
    def _prediction_for_pose(particles: list, pose: np.ndarray) -> dict:
        # One geometry calculation for this pose, reused by all frames in the
        # block/history group.  The returned prediction carries all particles.
        return env.predict_batch(particles, [pose])

    @classmethod
    def _prediction_for_poses(
        cls,
        particles: list,
        poses: Sequence[np.ndarray],
        *,
        backend: str = "numpy",
    ) -> dict:
        """Predict all unique poses in one call, preserving particle refs.

        ``env.predict_batch`` is already vectorized over actions.  Keeping one
        World-object list for the whole call avoids the previous full-history
        and block callbacks repeating geometry for the same particle set; no
        id-based cache is used, so Python object-id reuse cannot alias results.
        """
        pose_array = np.asarray(poses, dtype=float)
        if backend == "numpy":
            return env.predict_batch(particles, pose_array)
        if backend == "cuda":
            try:
                from .gpu_predictor import predict_batch as cuda_predict_batch
            except ImportError:
                from gpu_predictor import predict_batch as cuda_predict_batch

            return cuda_predict_batch(particles, pose_array)
        raise ValueError(f"unsupported prediction backend: {backend}")

    @classmethod
    def _loglike_frames(
        cls,
        frames_and_poses: Sequence[tuple[dict, np.ndarray]],
        particles: list,
        *,
        backend: str = "numpy",
    ) -> np.ndarray:
        if not frames_and_poses:
            return np.zeros(len(particles), dtype=float)
        grouped: dict[tuple[float, ...], list[dict]] = {}
        for frame, pose in frames_and_poses:
            grouped.setdefault(_pose_key(pose), []).append(frame)
        keys = list(grouped)
        poses = [np.asarray(key, dtype=float) for key in keys]
        if len(keys) == 1 and backend == "numpy":
            prediction = cls._prediction_for_pose(particles, poses[0])
        else:
            prediction = cls._prediction_for_poses(particles, poses, backend=backend)
        out = np.zeros(len(particles), dtype=float)
        for action_index, key in enumerate(keys):
            for frame in grouped[key]:
                out += env.log_likelihood_from_prediction(frame, prediction, action_index)
        return out

    @classmethod
    def _joint_loglikes(
        cls,
        history_frames_and_poses: Sequence[tuple[dict, np.ndarray]],
        block_frames_and_poses: Sequence[tuple[dict, np.ndarray]],
        particles: list,
        *,
        backend: str = "numpy",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate history and new-block LLs from one multi-pose prediction.

        The two returned vectors are kept separate because beta applies only
        to the new block.  Every unique pose across both sets is predicted once
        and then all five-frame (or historical) observations reuse that output.
        """
        if not history_frames_and_poses and not block_frames_and_poses:
            z = np.zeros(len(particles), dtype=float)
            return z.copy(), z
        grouped: dict[tuple[float, ...], list[tuple[int, dict]]] = {}
        for frame, pose in history_frames_and_poses:
            grouped.setdefault(_pose_key(pose), []).append((0, frame))
        for frame, pose in block_frames_and_poses:
            grouped.setdefault(_pose_key(pose), []).append((1, frame))
        keys = list(grouped)
        poses = [np.asarray(key, dtype=float) for key in keys]
        if len(keys) == 1 and backend == "numpy":
            prediction = cls._prediction_for_pose(particles, poses[0])
        else:
            prediction = cls._prediction_for_poses(particles, poses, backend=backend)
        history_ll = np.zeros(len(particles), dtype=float)
        block_ll = np.zeros(len(particles), dtype=float)
        for action_index, key in enumerate(keys):
            for kind, frame in grouped[key]:
                ll = env.log_likelihood_from_prediction(frame, prediction, action_index)
                if kind == 0:
                    history_ll += ll
                else:
                    block_ll += ll
        return history_ll, block_ll

    def _history_loglike(self, particles: list) -> np.ndarray:
        return self._loglike_frames(self.history, particles, backend=self.prediction_backend)

    def _unique_geometries(self) -> int:
        return len({p.points.tobytes() for p in self.particles})

    def update_block(self, publicframes: Iterable[dict], actual_pose: Sequence[float]) -> dict:
        """Assimilate one public frame block exactly once.

        Duplicate IDs are skipped.  A failed tempering path is returned with
        ``unfinished=True`` and the old posterior/history remain untouched.
        """
        pose = np.asarray(actual_pose, dtype=float).reshape(-1)
        if pose.shape != (3,) or np.any(~np.isfinite(pose)):
            raise ValueError("actual_pose must be a finite 3-vector")
        incoming = list(publicframes)
        candidate: list[dict] = []
        duplicate_ids: list[str] = []
        local_ids: set[str] = set()
        for frame in incoming:
            if not isinstance(frame, dict) or "frame_id" not in frame:
                raise ValueError("every public frame must be a dict with frame_id")
            fid = str(frame["frame_id"])
            if fid in self.seen or fid in local_ids:
                duplicate_ids.append(fid)
                continue
            local_ids.add(fid)
            candidate.append(dict(frame))
        base = {
            "block_size": len(incoming),
            "prediction_backend": self.prediction_backend,
            "accepted_frame_ids": [str(f["frame_id"]) for f in candidate],
            "duplicate_frame_ids": duplicate_ids,
            "skipped_duplicates": len(duplicate_ids),
            "history_frames_before": len(self.history),
        }
        if not candidate:
            d = {
                **base,
                "posterior_valid": True,
                "unfinished": False,
                "skipped": True,
                "temperature_path": [0.0],
                "stage_count": 0,
                "pre_raw_ess": self.ess,
                "post_raw_ess": self.ess,
                "unique_geometries": self._unique_geometries(),
                "unique_particles": self._unique_geometries(),
                "logZ": float(self.logZ),
            }
            self.diagnostics.append(d)
            return d

        block_pairs = [(frame, pose.copy()) for frame in candidate]
        block_loglike = lambda ps: self._loglike_frames(
            block_pairs, ps, backend=self.prediction_backend
        )
        joint_loglike = lambda ps: self._joint_loglikes(
            self.history,
            block_pairs,
            ps,
            backend=self.prediction_backend,
        )
        try:
            update = adaptive_tempered_resample_move(
                self.particles,
                self.weights,
                block_loglike,
                self._history_loglike,
                env._logprior,
                env._propose,
                self.rng,
                target_fraction=self.cess_target,
                resample_fraction=self.resample_threshold,
                max_temper_steps=self.max_temper_steps,
                mh_sweeps=self.mh_sweeps,
                joint_loglike=joint_loglike,
            )
        except ValueError as exc:
            # A hard all-particle support mismatch is a valid failed update,
            # not permission to floor likelihoods or drop the block.  Return a
            # complete diagnostic while leaving the old posterior untouched.
            if "incompatible with the new evidence" not in str(exc):
                raise
            d = {
                **base,
                "posterior_valid": False,
                "unfinished": True,
                "rollback": True,
                "failure": "all_particles_incompatible",
                "failure_message": str(exc),
                "temperature_path": [0.0],
                "beta_final": 0.0,
                "stage_count": 0,
                "pre_raw_ess": self.ess,
                "post_raw_ess": self.ess,
                "mh_acceptance": 0.0,
                "resample_count": 0,
                "attempted_logZ_increment": 0.0,
                "history_frames_after": len(self.history),
                "unique_geometries": self._unique_geometries(),
                "unique_particles": self._unique_geometries(),
                "logZ": float(self.logZ),
                "history_committed": False,
            }
            self.diagnostics.append(d)
            return d
        d = {**base, **update.diagnostics}
        d["history_frames_after"] = len(self.history) + (len(candidate) if update.committed else 0)
        d["unique_geometries"] = len({p.points.tobytes() for p in update.particles})
        d["unique_particles"] = d["unique_geometries"]
        d["logZ_before"] = float(self.logZ)
        if not update.committed:
            d["logZ"] = float(self.logZ)
            d["history_committed"] = False
            self.diagnostics.append(d)
            return d

        self.particles = list(update.particles)
        self.weights = np.asarray(update.weights, dtype=float)
        self.history.extend(block_pairs)
        self.seen.update(str(f["frame_id"]) for f in candidate)
        self.logZ += float(update.diagnostics["attempted_logZ_increment"])
        d["logZ_increment"] = float(update.diagnostics["attempted_logZ_increment"])
        d["logZ"] = float(self.logZ)
        d["history_committed"] = True
        self.diagnostics.append(d)
        return d


__all__ = [
    "ContinuousBelief",
    "TemperedUpdate",
    "adaptive_next_beta",
    "adaptive_tempered_resample_move",
    "raw_ess",
]
