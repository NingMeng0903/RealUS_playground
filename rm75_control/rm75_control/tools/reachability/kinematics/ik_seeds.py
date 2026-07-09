"""Seed pools for the DLS IK solver.

7-DOF arms with wrist singularities and joint-limit walls have many disjoint
IK branches; DLS gets stuck in the basin nearest ``q_seed``. A small pool
covers the common branches (nominal posture, mirrored elbow, and low-discrepancy
random) at a fraction of the cost of full sampling-based IK.

Kept dependency-free (only numpy) so it can be pickled into worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# From configs/joint_admittance.yaml (the 7-DOF map's canonical rest posture
# for the RM75-6F; used by the controller's nullspace centering task).
DEFAULT_NOMINAL_DEG: tuple[float, ...] = (0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0)


@dataclass
class SeedPoolConfig:
    n_random: int = 8
    include_nominal: bool = True
    include_zeros: bool = True
    include_elbow_flip: bool = True
    random_seed: int = 0
    nominal_deg: tuple[float, ...] = field(default_factory=lambda: DEFAULT_NOMINAL_DEG)


def _halton(i: int, base: int) -> float:
    """Van der Corput / Halton element for index i in given base."""
    f = 1.0
    r = 0.0
    j = i
    while j > 0:
        f /= base
        r += f * (j % base)
        j //= base
    return r


def halton_matrix(n: int, dim: int, bases: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17)) -> np.ndarray:
    """(n, dim) Halton low-discrepancy sequence in [0, 1)."""
    if dim > len(bases):
        raise ValueError(f"need at least {dim} primes for {dim}-D Halton")
    out = np.zeros((n, dim), dtype=np.float64)
    for d in range(dim):
        b = bases[d]
        for i in range(n):
            out[i, d] = _halton(i + 1, b)
    return out


def build_seed_pool(
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    cfg: SeedPoolConfig | None = None,
) -> np.ndarray:
    """Return (S, 7) rad seeds inside ``[q_lower, q_upper]``.

    Order: [zeros?, nominal?, elbow_flip(nominal)?, halton(random)...]
    """
    cfg = cfg or SeedPoolConfig()
    n_dof = q_lower.size
    seeds: list[np.ndarray] = []

    if cfg.include_zeros:
        seeds.append(np.clip(np.zeros(n_dof), q_lower, q_upper))
    if cfg.include_nominal:
        nom = np.asarray(cfg.nominal_deg, dtype=np.float64) * (np.pi / 180.0)
        if nom.size != n_dof:
            raise ValueError(f"nominal_deg length {nom.size} != DOF {n_dof}")
        seeds.append(np.clip(nom, q_lower, q_upper))
        if cfg.include_elbow_flip:
            # flip joint_2 & joint_4 signs → the classic "elbow-down / mirror" branch
            flip = nom.copy()
            flip[1] = -flip[1]
            flip[3] = -flip[3]
            seeds.append(np.clip(flip, q_lower, q_upper))

    if cfg.n_random > 0:
        h = halton_matrix(cfg.n_random, n_dof)
        # offset by seed to keep the pool deterministic yet variable per call site
        h = (h + (cfg.random_seed * 0.618033988749895)) % 1.0
        rand = q_lower[None, :] + h * (q_upper - q_lower)[None, :]
        seeds.append(rand)

    return np.vstack([s.reshape(-1, n_dof) if s.ndim == 1 else s for s in seeds])
