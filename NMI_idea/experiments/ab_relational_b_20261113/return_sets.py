"""Exact unordered multi-return observation tables.

The legacy sensor exposes only the nearest return.  This module keeps all
three unlabeled returns as a canonical multiset after projecting the 7 x 7
fine image bins onto a small coarse grid.  It deliberately treats repeated
coarse bins as repeated observations.  ``joint`` stores the mass for the
target return (kind zero), split equally over canonical slots when more than
one return occupies that slot's bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import math

import numpy as np
from numba import njit


FINE_X = 7
FINE_Z = 7
FINE_PIXELS = FINE_X * FINE_Z
MAX_KINDS = 3
CONFIDENCE = 0.87
THREE_ORDERS = np.asarray(list(permutations(range(3))), dtype=np.int64)

# The same representative coordinates used by the original nearest-return
# policy.  They are only used for the nested coarse nearest control.
X_REP = np.array([-10., -6., -2.75, 0., 2.75, 6., 10.], dtype=float)
Z_REP = np.array([-7., -3.5, -1.35, 0., 1.35, 3.5, 7.], dtype=float)


@dataclass(frozen=True)
class ReturnTables:
    """Public and nested-nearest observation kernels.

    Shapes are ``[clock, address, packet, hypothesis]`` for ``total`` and
    ``nearest_total``. ``joint`` is ``[clock, address, packet, slot,
    hypothesis]`` and contains target-kind-zero mass. ``nearest_correct`` is
    ``[clock, address, scalar_observation, hypothesis]`` and sums the target
    mass over packet slots.
    """

    total: np.ndarray
    joint: np.ndarray
    nearest_total: np.ndarray
    nearest_correct: np.ndarray
    bins: np.ndarray
    mapping: np.ndarray
    lookup: np.ndarray
    nearest_slot: np.ndarray
    nearest_packet: np.ndarray
    claimable: np.ndarray
    coordinates: np.ndarray
    grid: int


def _coarse_mapping(grid: int) -> tuple[np.ndarray, np.ndarray]:
    if grid not in (3, 5):
        raise ValueError("grid must be 3 or 5")
    mapping = np.empty(FINE_PIXELS, dtype=np.int64)
    x_groups: list[list[int]] = [[] for _ in range(grid)]
    z_groups: list[list[int]] = [[] for _ in range(3)]
    # Nested, symmetric cuts in the inherited physical centroid grid.
    x_group = (0, 0, 0, 1, 2, 2, 2) if grid == 3 else (0, 0, 1, 2, 3, 4, 4)
    z_group = (0, 0, 0, 1, 2, 2, 2)
    for i, group in enumerate(x_group):
        x_groups[group].append(i)
    for j, group in enumerate(z_group):
        z_groups[group].append(j)
    coordinates = np.empty((grid * 3, 2), dtype=float)
    for gx, xs in enumerate(x_groups):
        for gz, zs in enumerate(z_groups):
            index = gx * 3 + gz + 1
            coordinates[index - 1] = (np.mean(X_REP[xs]), np.mean(Z_REP[zs]))
            for x in xs:
                for z in zs:
                    mapping[x * FINE_Z + z] = index
    return mapping, coordinates


def _packets(n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate nondecreasing triples from zero through ``n_bins``."""
    values = list(product(range(n_bins + 1), repeat=3))
    canonical = sorted(set(tuple(sorted(row)) for row in values))
    bins = np.asarray(canonical, dtype=np.int64)
    lookup = np.full((n_bins + 1, n_bins + 1, n_bins + 1), -1, dtype=np.int64)
    ids = {tuple(row): packet for packet, row in enumerate(canonical)}
    for row in values:
        lookup[row] = ids[tuple(sorted(row))]
    return bins, lookup


@njit(cache=True)
def _expand(detect, cdf, mapping, bins, lookup):
    n_clock, n_address, n_hyp = detect.shape[:3]
    n_state = lookup.shape[0]
    total = np.zeros((n_clock, n_address, len(bins), n_hyp))
    joint = np.zeros((n_clock, n_address, len(bins), 3, n_hyp))
    for t in range(n_clock):
        for a in range(n_address):
            factors = np.zeros((3, n_state, n_hyp))
            for h in range(n_hyp):
                for kind in range(3):
                    present = detect[t, a, h, kind]
                    factors[kind, 0, h] = 1. - present
                    previous = 0.
                    for fine, coarse in enumerate(mapping):
                        current = cdf[t, a, h, kind, fine]
                        factors[kind, coarse, h] += present * max(0., current - previous)
                        previous = current
            for s0 in range(n_state):
                for s1 in range(n_state):
                    for s2 in range(n_state):
                        packet = lookup[s0, s1, s2]
                        for h in range(n_hyp):
                            mass = factors[0, s0, h] * factors[1, s1, h] * factors[2, s2, h]
                            total[t, a, packet, h] += mass
                        if s0 > 0:
                            duplicates = (int(bins[packet, 0] == s0) + int(bins[packet, 1] == s0)
                                          + int(bins[packet, 2] == s0))
                            for slot in range(3):
                                if bins[packet, slot] == s0:
                                    for h in range(n_hyp):
                                        mass = factors[0, s0, h] * factors[1, s1, h] * factors[2, s2, h]
                                        joint[t, a, packet, slot, h] += mass / duplicates
    return total, joint


def _nearest_indices(bins: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(len(coordinates) + 1, math.inf, dtype=float)
    distances[1:] = (coordinates[:, 0] / 4.) ** 2 + (coordinates[:, 1] / 3.) ** 2
    selected = np.full(len(bins), -1, dtype=np.int64)
    packet = np.zeros(len(bins), dtype=np.int64)
    k = len(coordinates)
    for i, row in enumerate(bins):
        visible = np.flatnonzero(row > 0)
        if not len(visible):
            continue
        # Distance first, bin ID as deterministic tie break.  Identical bins
        # remain a multiset and are represented by the first canonical slot.
        best = min((int(row[j]) for j in visible), key=lambda value: (distances[value], value))
        selected[i] = int(np.flatnonzero(row == best)[0])
        # Scalar code uses 0 for absent, 1..K for single and K+1..2K for
        # multiple.  ``best`` is already one-based.
        packet[i] = best + k * (len(visible) > 1)
    return selected, packet


def build_tables(detect: np.ndarray, cdf: np.ndarray, grid: int = 3) -> ReturnTables:
    """Construct exact unordered and nested-nearest kernels.

    Parameters use the legacy orientation ``detect[clock,address,h,kind]`` and
    ``cdf[clock,address,h,kind,fine_pixel]``.  The return layout is described
    by :class:`ReturnTables`.  No prior over hypotheses is needed, which keeps
    this kernel shared by MPC and B.
    """
    detect = np.asarray(detect, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    if detect.ndim != 4 or detect.shape[-1] != MAX_KINDS:
        raise ValueError(f"detect must have shape [clock,address,h,3], got {detect.shape}")
    expected = (*detect.shape[:3], MAX_KINDS, FINE_PIXELS)
    if cdf.shape != expected:
        raise ValueError(f"cdf must have shape {expected}, got {cdf.shape}")
    if np.any(detect < -1.e-12) or np.any(detect > 1. + 1.e-12):
        raise ValueError("detect probabilities must lie in [0,1]")
    mapping, coordinates = _coarse_mapping(grid)
    bins, lookup = _packets(int(mapping.max()))
    nearest_slot, nearest_packet = _nearest_indices(bins, coordinates)
    n_clock, n_address, n_hyp = detect.shape[:3]
    n_scalar = 1 + 2 * len(coordinates)
    total, joint = _expand(detect, cdf, mapping, bins, lookup)
    if not np.allclose(total.sum(axis=2), 1., atol=2.e-11, rtol=0.):
        raise ValueError("unordered packet kernel does not normalize")
    nearest_total = np.zeros((n_clock, n_address, n_scalar, n_hyp), dtype=float)
    nearest_correct = np.zeros_like(nearest_total)
    for p, scalar in enumerate(nearest_packet):
        nearest_total[:, :, scalar, :] += total[:, :, p, :]
        slot = nearest_slot[p]
        if slot >= 0:
            # The nested control only receives the nearest coarse bin.  A
            # target return in a farther bin must therefore not be counted as
            # a correct nearest observation.
            nearest_correct[:, :, scalar, :] += joint[:, :, p, slot, :]
    if not np.allclose(nearest_total.sum(axis=2), 1., atol=2.e-11, rtol=0.):
        raise ValueError("nested-nearest kernel does not normalize")
    claimable = np.any((total[..., None, :] > 1.e-260) &
                       (joint >= (CONFIDENCE - 1.e-12) * total[..., None, :]), axis=(3, 4))
    # claimable is [clock,address,packet] after reducing slot and hypothesis.
    return ReturnTables(total, joint, nearest_total, nearest_correct, bins, mapping,
                        lookup, nearest_slot, nearest_packet, claimable,
                        coordinates, grid)


@njit(cache=True)
def draw(detect: np.ndarray, cdf: np.ndarray, clock: int, address: int,
         truth_index: int, tape: np.ndarray, mapping: np.ndarray,
         lookup: np.ndarray) -> tuple[int, np.ndarray]:
    """Draw a canonical packet and private labelled-kind slot assignment.

    ``tape`` uses the legacy seven uniforms per physical clock.  Channels 0:3
    gate the three returns, channels 3:6 sample their fine-bin CDFs, and
    channel 6 privately orders equal coarse bins.  The public packet never
    exposes this order.  ``kinds`` is canonical-slot aligned and padded by
    ``-1`` for absence.
    """
    bins = np.zeros(MAX_KINDS, dtype=np.int64)
    kinds = np.full(MAX_KINDS, -1, dtype=np.int64)
    for kind in range(MAX_KINDS):
        if tape[clock, kind] < detect[clock, address, truth_index, kind]:
            fine = np.searchsorted(cdf[clock, address, truth_index, kind], tape[clock, kind + 3])
            if fine >= FINE_PIXELS:
                fine = FINE_PIXELS - 1
            bins[kind] = mapping[fine]
            kinds[kind] = kind
    # Sort public bins. Randomize evaluator-only order within positive ties.
    for i in range(1, MAX_KINDS):
        value = bins[i]
        value_kind = kinds[i]
        j = i - 1
        while j >= 0 and bins[j] > value:
            bins[j + 1] = bins[j]
            kinds[j + 1] = kinds[j]
            j -= 1
        bins[j + 1] = value
        kinds[j + 1] = value_kind
    private = tape[clock, 6]
    if bins[0] > 0 and bins[0] == bins[2]:
        kinds = kinds[THREE_ORDERS[min(5, int(private * 6.))]]
    elif private >= .5:
        for j in range(2):
            if bins[j] > 0 and bins[j] == bins[j + 1]:
                kinds[j], kinds[j + 1] = kinds[j + 1], kinds[j]
                break
    return int(lookup[bins[0], bins[1], bins[2]]), kinds


__all__ = ["ReturnTables", "build_tables", "draw"]
