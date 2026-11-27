"""Public-history-conditioned importance proposal for support recovery.

This module is intentionally separate from the frozen tempered SMC engine.  It
constructs a proper proposal density over the continuous world parameters and
the 16 discrete mode combinations using only public frames and poses.  The
importance weight is always

    log p_prior(x, mode) + log p(public_history | x, mode) - log q(x, mode).

There is no likelihood floor, hidden target/source label, audit field, old
particle weight, or task reward in this implementation.

The listed world coordinates contain 28 continuous values (2+10+10+2+2+2).
The requested ``30D`` interface is made explicit by two auxiliary standard
normal coordinates at the tail of the vector.  They are intentionally unused
by the renderer and remain prior-only; this preserves the requested 30D API
without inventing anatomical degrees of freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import expit, logsumexp
from scipy.linalg import solve_triangular


LADDER_ROOT = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_module_ladder_20261110"
)
if str(LADDER_ROOT) not in sys.path:
    sys.path.insert(0, str(LADDER_ROOT))
from observation import environment as env  # noqa: E402


CONTINUOUS_DIM = 30
WORLD_DIM = 28
EPSILON = 0.10
PATTERN_SEED = 720907
MODE_OPTIONS = tuple(
    np.asarray(bits, dtype=float).reshape(2, 2)
    for bits in product((-1.0, 1.0), repeat=4)
)
MODE_LOG_WEIGHT = -np.log(float(len(MODE_OPTIONS)))
LOG2PI = np.log(2.0 * np.pi)
CONTROL_X = np.asarray(env.CONTROL_X, dtype=float)
CHOL = np.asarray(env.CHOL, dtype=float)
LOGDET_CHOL = float(np.log(np.diag(CHOL)).sum())
FIELDS = tuple(env.FIELDS)

REG = slice(0, 2)
YC = slice(2, 12)
ZC = slice(12, 22)
DEPTH = slice(22, 24)
IDENTITY = slice(24, 26)
DETECT = slice(26, 28)
AUX = slice(28, 30)


def _as_vector(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (CONTINUOUS_DIM,) or np.any(~np.isfinite(arr)):
        raise ValueError(f"expected finite {CONTINUOUS_DIM}D vector")
    return arr


def canonical_mode(mode: np.ndarray | Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(mode, dtype=float).reshape(2, 2)
    if not np.all(np.isin(arr, (-1.0, 1.0))):
        raise ValueError("mode entries must be +/-1")
    out = tuple(float(v) for v in arr.reshape(-1))
    if out not in {tuple(m.reshape(-1)) for m in MODE_OPTIONS}:
        raise ValueError("mode is not one of the 16 prior combinations")
    return out


def mode_array(mode: tuple[float, ...]) -> np.ndarray:
    return np.asarray(mode, dtype=float).reshape(2, 2)


def vector_from_world(world, *, auxiliary: Sequence[float] | None = None) -> np.ndarray:
    """Pack the 28 rendered world coordinates plus two prior-only auxiliaries."""
    x = np.zeros(CONTINUOUS_DIM, dtype=float)
    x[REG] = world.registration
    x[YC] = np.asarray(world.y_controls).reshape(-1)
    x[ZC] = np.asarray(world.z_controls).reshape(-1)
    x[DEPTH] = world.depths
    x[IDENTITY] = world.identity_z
    x[DETECT] = world.detect_z
    if auxiliary is not None:
        x[AUX] = np.asarray(auxiliary, dtype=float).reshape(2)
    return x


def world_from_vector(x: Sequence[float], mode: np.ndarray | Sequence[float]):
    """Decode a vector into the old environment World without using truth."""
    arr = _as_vector(x)
    modes = mode_array(canonical_mode(mode))
    return env._world(
        arr[REG].copy(),
        modes,
        arr[YC].reshape(2, 5).copy(),
        arr[ZC].reshape(2, 5).copy(),
        arr[DEPTH].copy(),
        arr[IDENTITY].copy(),
        arr[DETECT].copy(),
    )


def _log_normal_diag(values: np.ndarray, sd: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    d = values.shape[-1]
    return -0.5 * np.sum((values / sd) ** 2, axis=-1) - d * np.log(sd) - 0.5 * d * LOG2PI


def _log_correlated_controls(values: np.ndarray, sd: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 5)
    white = np.linalg.solve(CHOL, (values / sd).T).T
    return -0.5 * np.sum(white**2, axis=-1) - 5.0 * np.log(sd) - LOGDET_CHOL - 2.5 * LOG2PI


def log_prior_continuous(x: Sequence[float] | np.ndarray) -> float | np.ndarray:
    """Normalized continuous prior density, including the two auxiliary dims."""
    arr = np.asarray(x, dtype=float)
    scalar = arr.ndim == 1
    arr = np.atleast_2d(arr)
    if arr.shape[-1] != CONTINUOUS_DIM:
        raise ValueError(f"expected last dimension {CONTINUOUS_DIM}")
    out = _log_normal_diag(arr[:, REG], 0.004)
    yc = arr[:, YC].reshape(-1, 2, 5)
    zc = arr[:, ZC].reshape(-1, 2, 5)
    out += np.sum(np.stack([_log_correlated_controls(yc[:, s], 0.003) for s in range(2)], axis=1), axis=1)
    out += np.sum(np.stack([_log_correlated_controls(zc[:, s], 0.0015) for s in range(2)], axis=1), axis=1)
    depths = arr[:, DEPTH]
    depth_ok = np.all((depths >= 0.025) & (depths <= 0.045), axis=1)
    out += np.where(depth_ok, -2.0 * np.log(0.020), -np.inf)
    out += _log_normal_diag(arr[:, IDENTITY], 1.0)
    out += _log_normal_diag(arr[:, DETECT], 1.0)
    out += _log_normal_diag(arr[:, AUX], 1.0)
    return float(out[0]) if scalar else out


def log_prior_vector(x: Sequence[float] | np.ndarray, mode) -> float | np.ndarray:
    """Normalized prior over one mode (mode probability is exactly 1/16)."""
    # The mode is discrete and therefore has no continuous Jacobian.
    canonical_mode(mode)
    value = log_prior_continuous(x) + MODE_LOG_WEIGHT
    return value


def sample_prior_vectors(n: int, seed: int = 720901) -> tuple[np.ndarray, np.ndarray]:
    """Draw the exact normalized prior parameterization used by sample_worlds."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    x = np.zeros((n, CONTINUOUS_DIM), dtype=float)
    x[:, REG] = rng.normal(0.0, 0.004, size=(n, 2))
    x[:, YC] = (rng.normal(size=(n, 2, 5)) @ CHOL.T * 0.003).reshape(n, 10)
    x[:, ZC] = (rng.normal(size=(n, 2, 5)) @ CHOL.T * 0.0015).reshape(n, 10)
    x[:, DEPTH] = rng.uniform(0.025, 0.045, size=(n, 2))
    x[:, IDENTITY] = rng.normal(size=(n, 2))
    x[:, DETECT] = rng.normal(size=(n, 2))
    x[:, AUX] = rng.normal(size=(n, 2))
    mode_indices = rng.integers(0, len(MODE_OPTIONS), size=n)
    modes = np.asarray([MODE_OPTIONS[i] for i in mode_indices], dtype=float)
    return x, modes


def _frame_pose_pairs(history: Iterable) -> list[tuple[dict, np.ndarray]]:
    """Normalize public history blocks or explicit ``(frame, pose)`` pairs."""
    pairs: list[tuple[dict, np.ndarray]] = []
    for item in history:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
            pairs.append((dict(item[0]), np.asarray(item[1], dtype=float).copy()))
            continue
        if isinstance(item, dict) and "public" in item and "pose" in item:
            pose = np.asarray(item["pose"], dtype=float).copy()
            pairs.extend((dict(frame), pose.copy()) for frame in item["public"])
            continue
        if isinstance(item, dict) and "frame" in item and "pose" in item:
            pairs.append((dict(item["frame"]), np.asarray(item["pose"], dtype=float).copy()))
            continue
        raise ValueError("history must contain (public_frame, pose) pairs or public blocks")
    return pairs


def _spline_basis(xobs: float) -> np.ndarray:
    basis = np.zeros((5,), dtype=float)
    for i in range(5):
        one = np.zeros(5, dtype=float)
        one[i] = 1.0
        basis[i] = float(CubicSpline(CONTROL_X, one, bc_type="natural")(xobs))
    return basis


def _identity_source_probability(identity_score: float) -> float:
    """Soft public-only source probability; never hard-assign a target."""
    score = float(np.clip(identity_score, 0.0, 1.0))
    return float(np.clip(expit(4.0 * (score - 0.5)), 0.05, 0.95))


def association_patterns(
    history_pairs: Sequence[tuple[dict, np.ndarray]],
    *,
    seed: int = PATTERN_SEED,
    count: int = 8,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Sample and aggregate soft source assignments from public identity only."""
    visible = [
        frame
        for frame, _pose in history_pairs
        if bool(frame.get("visible"))
        and all(frame.get(field) is not None for field in ("image_u_m", "image_depth_m", "identity_score"))
    ]
    if not visible:
        return ((tuple(), 1.0),)
    probs = np.asarray([_identity_source_probability(f["identity_score"]) for f in visible])
    rng = np.random.default_rng(seed)
    draws = [tuple(int(v) for v in (rng.random(len(probs)) >= probs)) for _ in range(count)]
    counts: dict[tuple[int, ...], int] = {}
    for pattern in draws:
        counts[pattern] = counts.get(pattern, 0) + 1
    total = float(sum(counts.values()))
    return tuple((pattern, count / total) for pattern, count in sorted(counts.items()))


def _base_covariance() -> np.ndarray:
    cov = np.zeros((CONTINUOUS_DIM, CONTINUOUS_DIM), dtype=float)
    cov[REG, REG] = np.eye(2) * 0.004**2
    yc_cov = 0.003**2 * (CHOL @ CHOL.T)
    zc_cov = 0.0015**2 * (CHOL @ CHOL.T)
    for source in range(2):
        ys = slice(2 + 5 * source, 2 + 5 * (source + 1))
        zs = slice(12 + 5 * source, 12 + 5 * (source + 1))
        cov[ys, ys] = yc_cov
        cov[zs, zs] = zc_cov
    cov[DEPTH, DEPTH] = np.eye(2) * 0.006**2
    cov[IDENTITY, IDENTITY] = np.eye(2)
    cov[DETECT, DETECT] = np.eye(2)
    cov[AUX, AUX] = np.eye(2)
    return cov


def _condition(mean: np.ndarray, covariance: np.ndarray, row: np.ndarray, observed: float, noise: float, offset: float = 0.0):
    innovation = float(observed - offset - row @ mean)
    variance = float(row @ covariance @ row + noise**2)
    if not np.isfinite(variance) or variance <= 0.0:
        return mean, covariance
    gain = covariance @ row / variance
    mean = mean + gain * innovation
    covariance = covariance - np.outer(gain, row @ covariance)
    covariance = (covariance + covariance.T) * 0.5
    covariance.flat[:: covariance.shape[0] + 1] += 1e-12
    return mean, covariance


def _component_gaussian(mode: tuple[float, ...], pattern: tuple[int, ...], visible_records) -> tuple[np.ndarray, np.ndarray]:
    modes = mode_array(mode)
    mean = np.zeros(CONTINUOUS_DIM, dtype=float)
    mean[DEPTH] = 0.035
    covariance = _base_covariance()
    for (frame, pose), source in zip(visible_records, pattern):
        pose = np.asarray(pose, dtype=float)
        yaw = float(pose[2])
        c, s = np.cos(yaw), np.sin(yaw)
        u = float(frame["image_u_m"])
        depth = float(frame["image_depth_m"])
        xobs = float(pose[0] + c * u)
        observed_y = float(pose[1] + s * u)
        basis = _spline_basis(xobs)
        yrow = np.zeros(CONTINUOUS_DIM, dtype=float)
        yrow[0] = -modes[source, 1] * 0.25
        yrow[1] = 1.0
        yrow[2 + 5 * source : 2 + 5 * (source + 1)] = basis
        yoffset = modes[source, 0] * 0.014 + modes[source, 1] * 0.25 * xobs
        mean, covariance = _condition(mean, covariance, yrow, observed_y, 0.0015, yoffset)

        zrow = np.zeros(CONTINUOUS_DIM, dtype=float)
        zrow[22 + source] = 1.0
        zrow[12 + 5 * source : 12 + 5 * (source + 1)] = basis
        mean, covariance = _condition(mean, covariance, zrow, depth, 0.001)

        # Identity is public evidence, not a hard target/source label.  Use a
        # noisy local linearization of the model's identity logistic only.
        score = float(np.clip(frame["identity_score"], 1e-5, 1.0 - 1e-5))
        intercept = 1.75 if source == 0 else 0.75
        identity_target = (float(np.log(score) - np.log1p(-score)) - intercept) / 0.35
        irow = np.zeros(CONTINUOUS_DIM, dtype=float)
        irow[24 + source] = 1.0
        mean, covariance = _condition(mean, covariance, irow, identity_target, 0.10 / 0.35)
    covariance = (covariance + covariance.T) * 0.5
    covariance.flat[:: covariance.shape[0] + 1] += 1e-10
    return mean, covariance


def _log_gaussian(x: np.ndarray, mean: np.ndarray, chol: np.ndarray, logdet: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    flat = np.atleast_2d(values)
    diff = flat - mean[None, :]
    white = solve_triangular(chol, diff.T, lower=True).T
    out = -0.5 * np.sum(white**2, axis=-1) - 0.5 * CONTINUOUS_DIM * LOG2PI - logdet
    return float(out[0]) if values.ndim == 1 else out


@dataclass(frozen=True)
class GaussianComponent:
    mode: tuple[float, ...]
    pattern: tuple[int, ...]
    weight: float
    mean: np.ndarray
    covariance: np.ndarray
    chol: np.ndarray
    logdet: float


@dataclass
class ImportanceResult:
    vectors: np.ndarray
    modes: np.ndarray
    worlds: list
    log_weights: np.ndarray
    weights: np.ndarray
    diagnostics: dict


class InformedProposal:
    """Proper public-history-conditioned mixture proposal."""

    def __init__(
        self,
        history: Iterable,
        *,
        epsilon: float = EPSILON,
        pattern_seed: int = PATTERN_SEED,
        pattern_count: int = 8,
    ):
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be in (0,1)")
        self.history = _frame_pose_pairs(history)
        self.epsilon = float(epsilon)
        self.pattern_seed = int(pattern_seed)
        self.pattern_count = int(pattern_count)
        self.visible_records = [
            (frame, pose)
            for frame, pose in self.history
            if bool(frame.get("visible"))
            and all(frame.get(field) is not None for field in ("image_u_m", "image_depth_m", "identity_score"))
        ]
        self.patterns = association_patterns(
            self.history, seed=self.pattern_seed, count=self.pattern_count
        )
        components: list[GaussianComponent] = []
        for mode in MODE_OPTIONS:
            mode_tuple = tuple(float(v) for v in mode.reshape(-1))
            for pattern, pattern_weight in self.patterns:
                mean, covariance = _component_gaussian(mode_tuple, pattern, self.visible_records)
                chol = np.linalg.cholesky(covariance)
                components.append(
                    GaussianComponent(
                        mode=mode_tuple,
                        pattern=tuple(pattern),
                        weight=float(pattern_weight / len(MODE_OPTIONS)),
                        mean=mean,
                        covariance=covariance,
                        chol=chol,
                        logdet=float(np.log(np.diag(chol)).sum()),
                    )
                )
        self.components = tuple(components)
        weights = np.asarray([component.weight for component in self.components], dtype=float)
        self.component_weights = weights / weights.sum()
        component_payload = [
            {"mode": c.mode, "pattern": c.pattern, "weight": c.weight}
            for c in self.components
        ]
        self.component_hash = hashlib.sha256(
            json.dumps(component_payload, sort_keys=True).encode()
        ).hexdigest()
        self.history_hash = hashlib.sha256(
            json.dumps(
                [
                    {"frame": frame, "pose": np.asarray(pose).tolist()}
                    for frame, pose in self.history
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    def log_prior(self, vectors: np.ndarray, modes: np.ndarray) -> np.ndarray:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        modes = np.asarray(modes, dtype=float).reshape(-1, 2, 2)
        if vectors.shape[0] != modes.shape[0]:
            raise ValueError("vectors and modes length mismatch")
        out = np.asarray(log_prior_continuous(vectors), dtype=float)
        for i, mode in enumerate(modes):
            try:
                canonical_mode(mode)
            except ValueError:
                out[i] = -np.inf
            else:
                out[i] += MODE_LOG_WEIGHT
        return out

    def log_q(self, vectors: np.ndarray, modes: np.ndarray) -> np.ndarray:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        modes = np.asarray(modes, dtype=float).reshape(-1, 2, 2)
        if vectors.shape[0] != modes.shape[0]:
            raise ValueError("vectors and modes length mismatch")
        lp = np.asarray(log_prior_continuous(vectors), dtype=float) + MODE_LOG_WEIGHT
        informed = np.full(len(vectors), -np.inf, dtype=float)
        for component in self.components:
            cmode = mode_array(component.mode)
            mask = np.all(modes == cmode[None, :, :], axis=(1, 2))
            if not np.any(mask):
                continue
            values = np.asarray(_log_gaussian(vectors[mask], component.mean, component.chol, component.logdet))
            informed[mask] = np.logaddexp(
                informed[mask], np.log(component.weight) + values
            )
        return np.logaddexp(np.log(self.epsilon) + lp, np.log1p(-self.epsilon) + informed)

    def sample(self, n: int, *, seed: int = 720908) -> tuple[np.ndarray, np.ndarray, list]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        vectors = np.zeros((n, CONTINUOUS_DIM), dtype=float)
        modes = np.zeros((n, 2, 2), dtype=float)
        prior_branch = rng.random(n) < self.epsilon
        prior_count = int(prior_branch.sum())
        if prior_count:
            prior_x, prior_modes = sample_prior_vectors(prior_count, seed=int(rng.integers(0, 2**32 - 1)))
            vectors[prior_branch] = prior_x
            modes[prior_branch] = prior_modes
        informed_count = n - prior_count
        if informed_count:
            indices = rng.choice(len(self.components), size=informed_count, p=self.component_weights)
            z = rng.normal(size=(informed_count, CONTINUOUS_DIM))
            for j, component_index in enumerate(indices):
                component = self.components[int(component_index)]
                vectors[np.flatnonzero(~prior_branch)[j]] = component.mean + component.chol @ z[j]
                modes[np.flatnonzero(~prior_branch)[j]] = mode_array(component.mode)
        worlds = [world_from_vector(vectors[i], modes[i]) for i in range(n)]
        return vectors, modes, worlds

    def log_likelihood_full(self, worlds: Sequence) -> np.ndarray:
        """Evaluate every public history frame with one multi-pose prediction."""
        if not worlds:
            return np.zeros(0, dtype=float)
        if not self.history:
            return np.zeros(len(worlds), dtype=float)
        grouped: dict[tuple[float, ...], list[dict]] = {}
        for frame, pose in self.history:
            grouped.setdefault(tuple(float(v) for v in pose), []).append(frame)
        poses = np.asarray(list(grouped), dtype=float)
        prediction = env.predict_batch(worlds, poses)
        out = np.zeros(len(worlds), dtype=float)
        for action_index, key in enumerate(grouped):
            for frame in grouped[key]:
                out += env.log_likelihood_from_prediction(frame, prediction, action_index)
        return out

    def importance_sample(self, n: int = 128, *, seed: int = 720908) -> ImportanceResult:
        vectors, modes, worlds = self.sample(n, seed=seed)
        log_q = self.log_q(vectors, modes)
        log_prior = self.log_prior(vectors, modes)
        log_likelihood = self.log_likelihood_full(worlds)
        log_weights = log_prior + log_likelihood - log_q
        finite = np.isfinite(log_weights)
        diagnostics = {
            "posterior_valid": bool(np.any(finite)),
            "unfinished": bool(not np.any(finite)),
            "failure": None if np.any(finite) else "zero_likelihood_support",
            "sample_count": int(n),
            "finite_logweight_count": int(finite.sum()),
            "finite_likelihood_count": int(np.isfinite(log_likelihood).sum()),
            "component_count": len(self.components),
            "pattern_count": len(self.patterns),
            "component_hash": self.component_hash,
            "history_hash": self.history_hash,
            "epsilon": self.epsilon,
            "evidence_estimate": None,
            "raw_ess": 0.0,
            "max_weight": 0.0,
            "log_q_min": float(np.nanmin(log_q)),
            "log_q_max": float(np.nanmax(log_q)),
        }
        if not np.any(finite):
            weights = np.zeros(n, dtype=float)
            return ImportanceResult(vectors, modes, worlds, log_weights, weights, diagnostics)
        normalizer = float(logsumexp(log_weights[finite]))
        weights = np.zeros(n, dtype=float)
        weights[finite] = np.exp(log_weights[finite] - normalizer)
        diagnostics["evidence_estimate"] = float(logsumexp(log_weights[finite]) - np.log(n))
        diagnostics["raw_ess"] = float(1.0 / np.sum(weights**2))
        diagnostics["max_weight"] = float(np.max(weights))
        diagnostics["posterior_valid"] = True
        return ImportanceResult(vectors, modes, worlds, log_weights, weights, diagnostics)


def build_informed_proposal(history: Iterable, **kwargs) -> InformedProposal:
    return InformedProposal(history, **kwargs)


__all__ = [
    "CONTINUOUS_DIM",
    "WORLD_DIM",
    "MODE_OPTIONS",
    "GaussianComponent",
    "ImportanceResult",
    "InformedProposal",
    "association_patterns",
    "build_informed_proposal",
    "canonical_mode",
    "log_prior_continuous",
    "log_prior_vector",
    "sample_prior_vectors",
    "vector_from_world",
    "world_from_vector",
]
