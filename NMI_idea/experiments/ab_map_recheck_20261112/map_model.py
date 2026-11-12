"""Map-conditioned knee model used by the paired closed-loop recheck.

The frozen knee experiment uses a three-parameter atlas model ``(w0, b, z)``.
This module keeps that public interface and likelihood, while adding one
declared, low-dimensional material residual coefficient.  The fourth
coefficient is a smooth basis centred on the fork and nearby material
addresses; it is deliberately *not* one independent displacement per site.

``MapFilter`` is online until :meth:`MapFilter.freeze_map` is called.  A
frozen filter retains the static posterior mass from the common prefix and
only updates the conditional ``z``/identity distributions within each static
row.  The public world remains the nominal model in both arms; stress affects
only the realised case and observations.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import sys

import numpy as np


# The knee model is a frozen dependency for this experiment.  Keep this
# module importable both as a standalone file and when the driver changes its
# working directory to the new experiment directory.
HERE = Path(__file__).resolve().parent
FROZEN_KNEE = HERE.parent / "ab_knee_latent_rl_20261112"
if str(FROZEN_KNEE) not in sys.path:
    sys.path.insert(0, str(FROZEN_KNEE))

from belief import Filter as KneeFilter  # noqa: E402
from belief import Witness  # noqa: E402
from knee_world import (  # noqa: E402
    CAPTURE_S,
    DEPTH,
    TRAVEL_SPEED_MM_S,
    WIDTH,
    Candidate,
    Frame,
    KneeCase,
    World as KneeWorld,
)


# ---------------------------------------------------------------------------
# Public map constants
# ---------------------------------------------------------------------------

# Parameter order is [w0, daughter b, z, w_local].  The map has 315 static
# rows (9 x 7 x 5) and thirteen latent-z levels.  The static-row count is the
# budgeted map size; z remains the dynamic lag coordinate within each row.
MAP_W_AXIS = np.linspace(-6.0, 6.0, 9)
MAP_B_AXIS = np.linspace(-6.0, 6.0, 7)
MAP_LOCAL_AXIS = np.linspace(-4.0, 4.0, 5)
MAP_Z_AXIS = np.linspace(-6.0, 6.0, 13)
MAP_PARAMETER_ORDER = ("w0", "b", "z", "w_local")
MAP_STATIC_ROWS = len(MAP_W_AXIS) * len(MAP_B_AXIS) * len(MAP_LOCAL_AXIS)
MAP_ROWS = MAP_STATIC_ROWS * len(MAP_Z_AXIS)
if MAP_STATIC_ROWS > 315:  # Keep an accidental edit from violating the contract.
    raise RuntimeError("map static grid exceeds the 315-row contract")

# A local material residual is represented by a fixed Gaussian basis in the
# atlas's material address, not a separate latent w for every site.
LOCAL_BASIS_WIDTH_MM = 28.0
LOCAL_BASIS_BRANCH_SCALE = 0.68


def _static_grid():
    """Return static rows and their normalized prior masses."""
    rows = np.stack(
        np.meshgrid(MAP_W_AXIS, MAP_B_AXIS, MAP_LOCAL_AXIS, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    # Match the frozen knee prior convention while adding the declared local
    # coefficient.  The local prior is intentionally tighter than the two
    # broad atlas offsets.
    from belief import grid_prior  # imported from the frozen dependency

    static = (
        np.outer(grid_prior(MAP_W_AXIS, 2.8), grid_prior(MAP_B_AXIS, 2.7))
        .reshape(-1, 1)
        * grid_prior(MAP_LOCAL_AXIS, 2.5)[None, :]
    ).reshape(-1)
    return rows, static / static.sum()


MAP_STATIC_PARAMETERS, MAP_STATIC_PRIOR = _static_grid()


@dataclass(frozen=True)
class MapCase(KneeCase):
    """Knee case with the map coefficient and private stress controls.

    The extra fields are consumed only by :class:`MapWorld` while generating
    truth and observations.  They are not returned in ``Frame`` and are not
    used by the public likelihood or proposal methods.
    """

    w_local: float
    _stress_center_mm: float = 0.0
    _stress_width_mm: float = 1.0
    _stress_amplitude_mm: float = 0.0
    _miss_label: int = 0
    _miss_start_s: float = np.inf
    _miss_end_s: float = -np.inf
    _ghost_lateral_mm: float = 0.0
    _ghost_depth_mm: float = 0.0


class MapFilter(KneeFilter):
    """Frozen-knee filter with a 312-row map grid.

    Online updates use the inherited global normalizer exactly.  After
    ``freeze_map`` each future update computes a normalizer for every static
    row and restores the row mass captured at freeze time.  Witness tensors
    receive the same rowwise likelihood multiplier, so identities and map
    rows remain algebraically consistent.
    """

    parameter_order = MAP_PARAMETER_ORDER
    static_rows = MAP_STATIC_ROWS
    # ``map_rows`` is the contracted static map count; ``hypothesis_count``
    # includes the dynamic z coordinate used by the likelihood.
    map_rows = MAP_STATIC_ROWS
    hypothesis_count = MAP_ROWS

    def __init__(self, *, latent: bool = True):
        # Initialize inherited bookkeeping (reading/witness list) and then
        # replace only the grid arrays.  All inherited likelihood and motion
        # equations operate on the resulting fourth-column design.
        super().__init__(latent=latent)
        self.z = MAP_Z_AXIS.copy() if latent else np.zeros(1, dtype=float)
        self.w = MAP_W_AXIS.copy()
        self.b = MAP_B_AXIS.copy()
        self.local = MAP_LOCAL_AXIS.copy()
        static = MAP_STATIC_PARAMETERS
        self.parameters = np.empty((len(static), len(self.z), 4), dtype=float)
        self.parameters[..., 0] = static[:, None, 0]
        self.parameters[..., 1] = static[:, None, 1]
        self.parameters[..., 2] = self.z[None, :]
        self.parameters[..., 3] = static[:, None, 2]
        self.weights = MAP_STATIC_PRIOR[:, None] * self._z_prior()[None, :]
        self.frozen_marginal: np.ndarray | None = None

    def _z_prior(self):
        from belief import grid_prior

        return grid_prior(self.z, 1.5)

    def copy(self):
        """Use the frozen filter's replacement-based shallow copy contract.

        ``observe`` and ``advance`` replace posterior arrays and witness lists
        rather than editing them in place.  Keeping that inexpensive inherited
        copy is important because the planner forks many hypothetical states.
        The one map-only mutable object is copied explicitly so freezing a
        fork cannot alter the source filter.
        """
        other = super().copy()
        other.frozen_marginal = (
            None if self.frozen_marginal is None else self.frozen_marginal.copy()
        )
        return other

    def freeze_map(self):
        """Freeze static row masses in place and return ``self``.

        Calling this on ``filter.copy()`` therefore leaves the source filter
        untouched.  Prefix masses are copied exactly; no renormalization is
        done at the freeze boundary.
        """
        if self.frozen_marginal is None:
            self.frozen_marginal = np.asarray(self.weights.sum(axis=1), dtype=float).copy()
        return self

    def _row_normalizer(self, prior, likelihood):
        normalizer = np.sum(prior * likelihood, axis=1)
        if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
            raise FloatingPointError("observation outside identity model")
        return normalizer

    def _frozen_prior(self, prior):
        """Keep row masses exact while tolerating tiny transition drift."""
        if self.frozen_marginal is None:
            return prior
        row_mass = np.sum(prior, axis=1)
        if np.any(~np.isfinite(row_mass)):
            raise FloatingPointError("invalid frozen map transition")
        if np.any((self.frozen_marginal > 0.0) & (row_mass <= 0.0)):
            raise FloatingPointError("invalid frozen map transition")
        scale = np.divide(
            self.frozen_marginal, row_mass,
            out=np.zeros_like(self.frozen_marginal),
            where=row_mass > 0.0,
        )
        return prior * scale[:, None]

    def prior_at(self, reading):
        prior, transition = super().prior_at(reading)
        if self.frozen_marginal is not None:
            prior = self._frozen_prior(prior)
        return prior, transition

    def observe(self, world, reading, site, view, obs, *, event, retain, track=True):
        # Keep the online branch byte-for-byte in the frozen-knee algebra.
        if self.frozen_marginal is None:
            return super().observe(
                world, reading, site, view, obs,
                event=event, retain=retain, track=track,
            )

        obs = np.asarray(obs, dtype=float).reshape(-1, 2)
        prior, transition = self.prior_at(reading)
        likelihood, responsibilities = self.emission(world, reading, site, view, obs)
        # ``prior`` already carries each static row mass m.  First form the
        # row evidence r = m * E[L], then use E[L] as the conditional
        # normalizer.  Using r directly would multiply witness probabilities
        # by 1/m in low-mass rows.
        row_mass = np.sum(prior, axis=1)
        if np.any(~np.isfinite(row_mass)):
            raise FloatingPointError("invalid frozen map prior")
        row_normalizer = np.sum(prior * likelihood, axis=1)
        conditional_normalizer = np.divide(
            row_normalizer, row_mass,
            out=np.zeros_like(row_normalizer),
            where=row_mass > 0.0,
        )
        active = row_mass > 0.0
        if np.any(~np.isfinite(row_normalizer[active])) or np.any(
            conditional_normalizer[active] <= 0.0
        ):
            raise FloatingPointError("observation outside identity model")
        # row_normalizer already includes m, so the evidence is global sum(r).
        global_normalizer = float(np.sum(row_normalizer))
        if not np.isfinite(global_normalizer) or global_normalizer <= 0.0:
            raise FloatingPointError("observation outside identity model")

        # Conditional z posterior per static row, with exact frozen masses.
        # ``prior`` already contains m, so dividing by E[L]=r/m yields a
        # posterior whose row sum is m; multiplying by m again would square
        # the frozen mass.
        conditional = np.divide(
            prior * likelihood,
            conditional_normalizer[:, None],
            out=np.zeros_like(prior),
            where=conditional_normalizer[:, None] > 0.0,
        )
        self.weights = conditional
        self.reading = reading

        # This is L/(r/m) = m L/r, and applies to every Witness tensor.
        multiplier = np.divide(
            likelihood,
            conditional_normalizer[:, None],
            out=np.zeros_like(likelihood),
            where=conditional_normalizer[:, None] > 0.0,
        )
        if track:
            self.witnesses = [
                Witness(
                    w.event, w.site, w.ordinal, w.coordinate, w.key,
                    (w.joint @ transition) * multiplier[None, :, :],
                )
                for w in self.witnesses
            ]
        if retain:
            self.witnesses = self.witnesses + [
                Witness(
                    event, site, j, tuple(value), (event, j),
                    np.divide(
                        prior[None, :, :] * responsibilities[j],
                        conditional_normalizer[None, :, None],
                        out=np.zeros((3, len(self.parameters), len(self.z))),
                        where=conditional_normalizer[None, :, None] > 0.0,
                    ),
                )
                for j, value in enumerate(obs)
            ]
        # Return the scalar evidence expected by the frozen API.  The rowwise
        # values above are retained for the conditional posterior update.
        return global_normalizer

    def advance(self, reading, *, track=True):
        prior, transition = self.prior_at(reading)
        self.weights = prior
        self.reading = reading
        if track:
            self.witnesses = [
                Witness(w.event, w.site, w.ordinal, w.coordinate, w.key, w.joint @ transition)
                for w in self.witnesses
            ]


class MapWorld(KneeWorld):
    """Nominal/stress world sharing one public prediction model."""

    LOCAL_BASIS_WIDTH_MM = LOCAL_BASIS_WIDTH_MM
    LOCAL_BASIS_BRANCH_SCALE = LOCAL_BASIS_BRANCH_SCALE

    def __init__(self, geometry=None, *, noisy_pose=True, mode="nominal"):
        if mode not in ("nominal", "stress"):
            raise ValueError("mode must be 'nominal' or 'stress'")
        if geometry is None:
            super().__init__(noisy_pose=noisy_pose)
        else:
            super().__init__(geometry=geometry, noisy_pose=noisy_pose)
        self.mode = str(mode)

        # A single smooth fork-local basis supplies the fourth design column
        # for every pose and view.  It is shared by all sites and therefore
        # cannot act as an independent whole-leg material offset.
        s = np.asarray(self.scan.s, dtype=float)
        fork_s = float(s[self.fork_site])
        basis = np.exp(-0.5 * ((s - fork_s) / self.LOCAL_BASIS_WIDTH_MM) ** 2)
        self.local_basis_center_mm = fork_s
        self.local_basis = basis

        old_design = np.asarray(self.design)
        self.design = np.zeros(old_design.shape[:-1] + (4,), dtype=float)
        self.design[..., :3] = old_design
        self.design[:, :, 0, :, 3] = old_design[:, :, 0, :, 0] * basis[None, :, None]
        self.design[:, :, 1, :, 3] = (
            self.LOCAL_BASIS_BRANCH_SCALE
            * old_design[:, :, 1, :, 0]
            * basis[None, :, None]
        )

        # The fixed route gives a deterministic nominal fork-arrival anchor.
        # It depends only on geometry, not on a sampled case or future action.
        self.nominal_fork_arrival_s, self.nominal_route_end_s = self._fixed_route_times()

    def _fixed_route_times(self):
        q = 15.0
        t = 0.0
        previous = None
        fork_time = None
        for goal in self.initial_route:
            if previous is None:
                t = CAPTURE_S
                previous = Candidate(*goal, CAPTURE_S, 0.0, t, "fixed route")
            else:
                action = super().make_action(q, previous, goal, "fixed route")
                t = float(action.t)
                previous = action
            if goal[0] == self.fork_site and fork_time is None:
                fork_time = float(t)
        return float(t if fork_time is None else fork_time), float(t)

    @staticmethod
    def _map_case(base: KneeCase, *, w_local, stress_center, stress_width,
                  stress_amplitude, miss_label, miss_start, miss_end,
                  ghost_lateral=0.0, ghost_depth=0.0):
        values = [getattr(base, field.name) for field in fields(KneeCase)]
        return MapCase(
            *values,
            float(w_local), float(stress_center), float(stress_width),
            float(stress_amplitude), int(miss_label), float(miss_start), float(miss_end),
            float(ghost_lateral), float(ghost_depth),
        )

    def case(self, seed, *, difficult=True):
        base = super().case(seed, difficult=difficult)
        # Separate streams keep stress perturbations independent of the action
        # sequence and preserve nominal case generation exactly.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8021]))
        w_local = float(rng.normal(0.0, 1.15))
        if self.mode == "nominal":
            return self._map_case(
                base, w_local=w_local, stress_center=0.0, stress_width=1.0,
                stress_amplitude=0.0, miss_label=0, miss_start=np.inf, miss_end=-np.inf,
                ghost_lateral=0.0, ghost_depth=0.0,
            )

        center = float(self.local_basis_center_mm + rng.normal(0.0, 12.0))
        width = float(rng.uniform(14.0, 32.0))
        amplitude = float(rng.choice((-1.0, 1.0)) * rng.uniform(1.2, 2.8))
        profile = np.exp(-0.5 * ((np.asarray(self.scan.s) - center) / width) ** 2)
        shifted_w = np.asarray(base.w_subject_mm, dtype=float) + amplitude * profile

        # Keep the nominal fields and RNG exactly, replacing only the realised
        # material offsets with the unobserved smooth stress perturbation.
        base = KneeCase(
            base.seed, shifted_w, base.branch_offset_mm, base.phase,
            base.amplitude, base.omega, base.z0, base.tau, base.coupling,
            base.drift, base.drift_phase, base.pose_bias, base.dropout,
            base.clutter, base.ghost_label,
        )
        jitter = float(rng.normal(0.0, 0.32))
        miss_start = float(self.nominal_fork_arrival_s + jitter)
        if rng.random() >= 0.65:
            # A second fixed-clock burst puts suffix loss into the recheck
            # window after the nominal first sweep.  Its jitter and duration
            # use the same case-local streams as the fork-anchored burst.
            miss_start = float(self.nominal_route_end_s + 1.10 + jitter)
        miss_end = float(miss_start + rng.uniform(0.45, 0.90))
        miss_label = int(rng.integers(2))
        ghost_lateral = float(rng.normal(0.0, 2.4))
        ghost_depth = float(rng.normal(0.0, 1.0))
        return self._map_case(
            base, w_local=w_local, stress_center=center, stress_width=width,
            stress_amplitude=amplitude, miss_label=miss_label,
            miss_start=miss_start, miss_end=miss_end,
            ghost_lateral=ghost_lateral, ghost_depth=ghost_depth,
        )

    def truth_world(self, case, t, sites):
        sites = tuple(sites)
        q, _ = case.true_angle_rate(t)
        base, design = self.world_design(q, sites)
        params = np.array([
            [case.w_subject_mm[i], case.branch_offset_mm, case.z_mm(t), case.w_local]
            for i, _ in sites
        ])
        return base + np.einsum("nci,ni->nc", design, params)

    def _persistent_clutter(self, case, q, site, view, base):
        """A deterministic branch-near false structure for stress repeats."""
        if self.mode != "stress" or site not in self.branch_sites:
            return None
        # The decision and offset are fixed by case/site/view, so revisiting a
        # view sees the same false structure despite a different action time.
        rng = np.random.default_rng(np.random.SeedSequence([case.seed, 9311, site, view]))
        if rng.random() > 0.72:
            return None
        # Store one case-local displacement in a canonical material frame,
        # then project it through the actual view.  This keeps a repeat ghost
        # physically coherent across views instead of drawing image offsets
        # independently for every view.
        canonical_view = self.view_bank[site][0]
        canonical_pose = self._blend(self.scan.view_frames[:, site, canonical_view], q)
        canonical_lateral = canonical_pose[:, 1] / np.linalg.norm(canonical_pose[:, 1])
        canonical_inward = canonical_pose[:, 2] - canonical_lateral * (canonical_pose[:, 2] @ canonical_lateral)
        canonical_inward /= np.linalg.norm(canonical_inward)
        delta_world = (
            case._ghost_lateral_mm * canonical_lateral
            + case._ghost_depth_mm * canonical_inward
        )
        actual_pose = self._blend(self.scan.view_frames[:, site, view], q)
        actual_lateral = actual_pose[:, 1] / np.linalg.norm(actual_pose[:, 1])
        actual_inward = actual_pose[:, 2] - actual_lateral * (actual_pose[:, 2] @ actual_lateral)
        actual_inward /= np.linalg.norm(actual_inward)
        project = np.stack((actual_lateral, actual_inward))
        center = base[int(case.ghost_label)] + delta_world @ project.T
        return np.clip(center, [-WIDTH, DEPTH[0]], [WIDTH, DEPTH[1]])

    def observe(self, case, action, site=None, view=None):
        """Generate a frame with the frozen nominal formula plus stress only."""
        if isinstance(action, Candidate):
            t, site, view = action.t, action.site, action.view
        else:
            t = float(action)
        if site is None or view is None:
            raise ValueError("site and view are required for a map observation")

        tick = round(t * 1000)
        rng = np.random.default_rng(np.random.SeedSequence([case.seed, 141, tick, site, view]))
        pose_rng = np.random.default_rng(np.random.SeedSequence([case.seed, 147, tick]))
        q, speed = case.true_angle_rate(t)
        reading_q = q + (case.pose_bias + pose_rng.normal(0.0, 0.10) if self.noisy_pose else 0.0)
        reading_v = speed + (pose_rng.normal(0.0, 0.15) if self.noisy_pose else 0.0)

        base, rows = self.projection(q, site, view)
        offset = np.array([
            case.w_subject_mm[site], case.branch_offset_mm, case.z_mm(t), case.w_local,
        ])
        expected = base + rows @ offset
        values, labels = [], []
        pd = self.detection_probability(q, site, view) + 0.15 - case.dropout
        target_miss = (
            self.mode == "stress"
            and case._miss_start_s <= t <= case._miss_end_s
            and self.identity_site(site)
        )
        for label, center in enumerate(expected):
            # A brief target-specific physical-time miss is independent of
            # action count, planner choices, and view order.
            if target_miss and label == case._miss_label:
                continue
            if rng.random() >= pd:
                continue
            scale = 3.0 if rng.random() < 0.075 else 1.0
            observed = center + rng.normal(0.0, [1.0 * scale, 0.68])
            if abs(observed[0]) <= WIDTH and DEPTH[0] <= observed[1] <= DEPTH[1]:
                values.append(observed)
                labels.append(label)

        persistent = self._persistent_clutter(case, q, site, view, base)
        if persistent is not None:
            values.append(persistent)
            labels.append(-1)
        elif rng.random() < case.clutter * (1.0 if self.identity_site(site) else 0.5):
            if self.identity_site(site) and rng.random() < 0.78:
                center = base[case.ghost_label] + rng.normal(0.0, [2.9, 1.35])
                center = np.clip(center, [-WIDTH, DEPTH[0]], [WIDTH, DEPTH[1]])
            else:
                center = rng.uniform([-WIDTH, DEPTH[0]], [WIDTH, DEPTH[1]])
            values.append(center)
            labels.append(-1)
        if values:
            order = np.argsort(np.asarray(values)[:, 0], kind="stable")
            values = np.asarray(values)[order]
            labels = tuple(labels[i] for i in order)
        else:
            values, labels = np.zeros((0, 2)), ()
        return Frame(float(t), float(reading_q), float(reading_v), site, view, values, labels)


# Public aliases mirror the frozen knee module's import style.
Filter = MapFilter
World = MapWorld


__all__ = [
    "Filter", "MapFilter", "MapWorld", "World", "MapCase",
    "MAP_W_AXIS", "MAP_B_AXIS", "MAP_LOCAL_AXIS", "MAP_Z_AXIS",
    "MAP_PARAMETER_ORDER", "MAP_STATIC_ROWS", "MAP_ROWS",
    "LOCAL_BASIS_WIDTH_MM", "LOCAL_BASIS_BRANCH_SCALE",
]
