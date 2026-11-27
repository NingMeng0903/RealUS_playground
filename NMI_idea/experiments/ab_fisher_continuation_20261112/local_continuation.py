"""Local, finite-state continuation benchmark on the checked 15-degree atlas.

This is a deliberately narrow mechanism study.  It reuses the frozen
``DynamicScan`` candidate table from ``ab_leg_dynamic_20261112`` and adds one
synthetic acoustic nuisance: a persistent preferred insonation-angle mode
shared by the two sites in an adjacent pair.  The nuisance is finite and
publicly learnable, so every planner below has an exact Bayes belief over the
same three modes.  The only observation exposed to a planner is one of

    miss, hit_q0, hit_q1, hit_q2

where ``q1`` and ``q2`` are usable.  A hit is not a hidden continuous score.
The full 50-mm aperture is retained (FOV ``[-25, 25]`` mm); candidate contact,
tilt, depth, and contact obliquity are read from the checked geometry and are
not replaced by a toy wall or an invented exit.

The executable benchmark has a common three-acquisition cap (an acquisition
cap, not a method-specific deadline) and reports cumulative completion at
shared acquisition counts.  ``B2`` and ``B3`` are exact finite-horizon
outcome-conditioned look-ahead policies; ``Fixed`` commits to a continuation
whose only branch is the public progress flag; ``A_reactive`` is posterior
one-step greedy; ``Fisher_MI`` is a tuned discrete-state expected-information
greedy policy; and ``Uniform`` is a width-independent uniform sensor control.
All arms see the same candidate banks, measured costs, hidden episode state,
and public outcome clock for a paired replay.

The module writes only ``local_study/`` beside this file.  It intentionally
does not claim a full-leg result: outputs are evidence for this local
persistent-window mechanism only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit, ndtr


HERE = Path(__file__).resolve().parent
DYNAMIC_DIR = HERE.parent / "ab_leg_dynamic_20261112"
GEOMETRY_PATH = DYNAMIC_DIR / "DYNAMIC_GEOMETRY_FROZEN.npz"
SAME_FIRST_PATH = DYNAMIC_DIR / "SAME_FIRST_WINDOW.json"
POSE_INDEX = 2
POSE_DEGREES = 15.0
FOV_HALF_WIDTH_MM = 25.0
FOV_TOTAL_MM = 2.0 * FOV_HALF_WIDTH_MM
MODE_ANGLES_DEG = np.asarray((-12.0, 0.0, 12.0), dtype=float)
MODE_PRIOR = np.asarray((0.30, 0.40, 0.30), dtype=float)
QUALITY_EDGES = np.asarray((0.50, 0.70), dtype=float)
QUALITY_SD = 0.11
USABLE_QUALITY_BINS = (1, 2)
# Outcome index 0 is miss and index 1 is hit_q0, so usable q1/q2 map to
# public outcome indices 2 and 3.
USABLE_OUTCOME_INDICES = (2, 3)
OUTCOME_LABELS = ("miss", "hit_q0", "hit_q1", "hit_q2")
N_OUTCOMES = len(OUTCOME_LABELS)
MAX_ACQUISITIONS = 3
GLOBAL_SEED = 2026111217
DEFAULT_DEV_SEEDS = tuple(range(8))
DEFAULT_HELDOUT_SEEDS = tuple(range(1000, 1024))
DEV_PAIR_INDICES = (0, 3, 6, 9, 12, 15, 18)
MI_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MISMATCH_GAIN_SD = 0.08
MISMATCH_QUALITY_SD = 0.025
MISMATCH_MODE_SD_DEG = 1.5
TIE_COST_TOLERANCE = 1.0e-10
METHODS = ("B2", "B3", "Fixed", "A_reactive", "Fisher_MI", "Uniform")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert NumPy values to strict JSON without hiding non-finite values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value cannot be written to JSON: {value!r}")
        return value
    return value


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(values)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def _normal_bins(mean: np.ndarray, sd: float = QUALITY_SD) -> np.ndarray:
    """Normal mass in q0/q1/q2 bins for arbitrary broadcastable means."""

    low = ndtr((QUALITY_EDGES[0] - mean) / sd)
    mid = ndtr((QUALITY_EDGES[1] - mean) / sd) - low
    high = 1.0 - ndtr((QUALITY_EDGES[1] - mean) / sd)
    result = np.stack((low, mid, high), axis=-1)
    return np.clip(result, 0.0, 1.0)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(values > 0.0, values * np.log2(values), 0.0)
    return -np.sum(terms, axis=-1)


@dataclass(frozen=True)
class Candidate:
    """One checked DynamicScan window at one station."""

    site: int
    candidate: int
    offset_mm: float
    tilt_deg: float
    depth_mm: float
    contact_obliquity_deg: float
    first_cost: float

    @property
    def action_id(self) -> str:
        return f"site{self.site}:window{self.candidate}"


class PairModel:
    """Geometry features, acoustic kernels, and checked transition costs."""

    def __init__(self, scan: Any, payload: Mapping[str, np.ndarray], source: int, target: int):
        self.scan = scan
        self.pose = POSE_INDEX
        self.source = int(source)
        self.target = int(target)
        self.sites = (self.source, self.target)
        self.fov_half_width_mm = FOV_HALF_WIDTH_MM
        views = np.asarray(payload["posed_views_m"])[self.pose]
        frames = np.asarray(payload["posed_frames"])[self.pose]
        trunk = np.asarray(payload["posed_vessel_m"])[self.pose]
        ray = trunk[:, None, :] - views
        depth = np.einsum("ikc,ikc->ik", ray, frames[:, :, :, 2]) * 1000.0
        ray_norm = np.linalg.norm(ray, axis=-1, keepdims=True)
        unit_ray = ray / np.maximum(ray_norm, 1.0e-12)
        inward = frames[:, :, :, 2]
        obliquity = np.degrees(
            np.arccos(np.clip(np.einsum("ikc,ikc->ik", unit_ray, inward), -1.0, 1.0))
        )
        self.depth_mm = depth
        self.obliquity_deg = obliquity
        self.tilts_deg = np.asarray(payload["view_tilts_deg"], dtype=float)
        self.offsets_mm = np.asarray(payload["view_offsets_mm"], dtype=float)
        self.candidates: dict[int, tuple[Candidate, ...]] = {}
        for site in self.sites:
            valid = np.flatnonzero(scan.eligible[self.pose, site]).astype(int)
            if valid.size == 0:
                raise ValueError(f"site {site} has no checked candidate windows")
            rows = []
            first = scan.costs(None, self.pose, [(site, int(candidate)) for candidate in valid])
            for index, candidate in enumerate(valid):
                depth_value = float(depth[site, candidate])
                if not math.isfinite(depth_value) or depth_value <= 0.0:
                    raise ValueError(f"non-positive checked depth at site {site}, window {candidate}")
                rows.append(
                    Candidate(
                        site=site,
                        candidate=int(candidate),
                        offset_mm=float(self.offsets_mm[candidate]),
                        tilt_deg=float(self.tilts_deg[candidate]),
                        depth_mm=depth_value,
                        contact_obliquity_deg=float(obliquity[site, candidate]),
                        first_cost=float(first[index]),
                    )
                )
            self.candidates[site] = tuple(rows)
        self._transition_costs: dict[tuple[int, int, int, int], float] = {}
        for old_site in self.sites:
            for old in self.candidates[old_site]:
                options = [
                    (new_site, new.candidate)
                    for new_site in self.sites
                    for new in self.candidates[new_site]
                ]
                values = scan.costs((self.pose, old_site, old.candidate), self.pose, options)
                for option, value in zip(options, values):
                    self._transition_costs[(old_site, old.candidate, option[0], option[1])] = float(value)
        self._nominal_kernels: dict[tuple[int, int], np.ndarray] = {}
        for site in self.sites:
            for candidate in self.candidates[site]:
                self._nominal_kernels[(site, candidate.candidate)] = self._kernel_for_action(
                    candidate, MODE_ANGLES_DEG, gain_shift=0.0, quality_shift=0.0, mode_shift=0.0
                )

    def actions(self, site: int) -> tuple[Candidate, ...]:
        return self.candidates[int(site)]

    def cost(self, previous: tuple[int, int] | None, action: Candidate) -> float:
        if previous is None:
            return float(action.first_cost)
        return float(self._transition_costs[(previous[0], previous[1], action.site, action.candidate)])

    def _kernel_for_action(
        self,
        action: Candidate,
        modes: np.ndarray,
        *,
        gain_shift: float,
        quality_shift: float,
        mode_shift: float,
    ) -> np.ndarray:
        """Return [mode, outcome] probabilities for one checked candidate.

        The explicit formulas are the synthetic local mechanism.  ``depth``
        and contact obliquity come from DynamicScan; ``tilt`` is the actual
        checked candidate tilt.  ``mode_shift``/gain/quality shifts are used
        only for the held-out truth generator, giving a modest unmodelled
        nuisance while leaving the planner's finite state catalogue exact.
        """

        modes = np.asarray(modes, dtype=float)
        effective_angle = action.tilt_deg + 0.35 * (action.contact_obliquity_deg - 50.0)
        mode_delta = effective_angle - (modes + mode_shift)
        alignment = np.exp(-0.5 * (mode_delta / 10.0) ** 2)
        phase = np.cos(np.deg2rad(mode_delta))
        depth = max(0.0, action.depth_mm)
        obliquity = action.contact_obliquity_deg
        logit = (
            -0.80
            + 1.10 * alignment
            + 0.55 * phase
            - 0.018 * depth
            - 0.010 * abs(obliquity - 50.0)
            + gain_shift
        )
        p_detect = expit(logit)
        quality_mean = (
            0.42
            + 0.25 * alignment
            + 0.18 * phase
            - 0.002 * depth
            - 0.0015 * abs(obliquity - 50.0)
            + quality_shift
        )
        quality = _normal_bins(quality_mean)
        result = np.zeros((modes.size, N_OUTCOMES), dtype=float)
        result[:, 0] = 1.0 - p_detect
        result[:, 1:] = p_detect[:, None] * quality
        result /= result.sum(axis=1, keepdims=True)
        return result

    def kernel(self, action: Candidate) -> np.ndarray:
        return self._nominal_kernels[(action.site, action.candidate)]

    def true_kernel(
        self,
        action: Candidate,
        mode: float,
        *,
        gain_shift: float,
        quality_shift: float,
        mode_shift: float,
    ) -> np.ndarray:
        return self._kernel_for_action(
            action,
            np.asarray((mode,), dtype=float),
            gain_shift=gain_shift,
            quality_shift=quality_shift,
            mode_shift=mode_shift,
        )[0]

    def expected_usable(self, action: Candidate, belief: np.ndarray) -> float:
        return float(np.asarray(belief) @ self.kernel(action)[:, USABLE_OUTCOME_INDICES].sum(axis=1))

    def mutual_information_bits(self, action: Candidate, belief: np.ndarray) -> float:
        kernels = self.kernel(action)
        belief = np.asarray(belief, dtype=float)
        predictive = belief @ kernels
        return float(_entropy(predictive) - belief @ _entropy(kernels))

    def geometry_record(self) -> dict[str, Any]:
        return {
            "source_station": self.source,
            "target_station": self.target,
            "pose_index": self.pose,
            "pose_degrees": POSE_DEGREES,
            "fov_total_mm": FOV_TOTAL_MM,
            "sites": {
                str(site): [
                    {
                        "candidate": c.candidate,
                        "offset_mm": c.offset_mm,
                        "tilt_deg": c.tilt_deg,
                        "depth_mm": c.depth_mm,
                        "contact_obliquity_deg": c.contact_obliquity_deg,
                        "first_cost": c.first_cost,
                    }
                    for c in self.candidates[site]
                ]
                for site in self.sites
            },
            "transition_cost_count": len(self._transition_costs),
            "depth_is_actual_dynamic_scan_feature": True,
            "contact_and_tilt_are_actual_checked_candidates": True,
        }


def _load_geometry() -> tuple[Any, dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Load the frozen scan and the audited common-bias pair gate."""

    if not GEOMETRY_PATH.exists() or not SAME_FIRST_PATH.exists():
        raise FileNotFoundError(f"missing checked geometry or SAME_FIRST_WINDOW: {GEOMETRY_PATH}")
    if str(DYNAMIC_DIR) not in sys.path:
        sys.path.insert(0, str(DYNAMIC_DIR))
    from dynamic_scan import DynamicScan  # type: ignore

    with np.load(GEOMETRY_PATH, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    scan = DynamicScan(payload)
    same = json.loads(SAME_FIRST_PATH.read_text())
    if int(same.get("pose_index", -1)) != POSE_INDEX or abs(float(same.get("pose_degrees", -1.0)) - POSE_DEGREES) > 1.0e-9:
        raise ValueError("SAME_FIRST_WINDOW is not the checked 15-degree gate")
    pairs = [row for row in same.get("all_adjacent_pairs", ()) if bool(row.get("included"))]
    if not pairs:
        raise ValueError("SAME_FIRST_WINDOW contains no supported pairs")
    for row in pairs:
        source, target = int(row["source_station"]), int(row["target_station"])
        if target != source + 1:
            raise ValueError(f"non-adjacent pair in SAME_FIRST_WINDOW: {source}, {target}")
        if not (source not in set(map(int, scan.branch_indices)) and target not in set(map(int, scan.branch_indices))):
            raise ValueError(f"branch station slipped into supported pair: {source}, {target}")
        if not np.any(scan.eligible[POSE_INDEX, source]) or not np.any(scan.eligible[POSE_INDEX, target]):
            raise ValueError(f"SAME_FIRST_WINDOW pair has no checked DynamicScan candidate: {source}, {target}")
    return scan, payload, pairs, same


def _pair_indices(n_pairs: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    development = tuple(index for index in DEV_PAIR_INDICES if index < n_pairs)
    heldout = tuple(index for index in range(n_pairs) if index not in development)
    if not development or not heldout:
        raise ValueError("need non-empty independent development and heldout pair groups")
    return development, heldout


def _posterior(belief: np.ndarray, likelihood: np.ndarray, outcome: int) -> np.ndarray:
    weights = np.asarray(belief, dtype=float) * np.asarray(likelihood, dtype=float)[:, int(outcome)]
    normalizer = float(weights.sum())
    if not normalizer > 0.0:
        raise FloatingPointError("zero-probability public outcome in finite Bayes update")
    return weights / normalizer


def _progress_after(
    pair: PairModel,
    progress: tuple[bool, bool],
    action: Candidate,
    outcome: int,
) -> tuple[bool, bool]:
    values = [bool(progress[0]), bool(progress[1])]
    if int(outcome) in USABLE_OUTCOME_INDICES:
        if action.site == pair.source:
            values[0] = True
        elif action.site == pair.target:
            values[1] = True
        else:  # pragma: no cover - checked candidate banks are pair-local
            raise ValueError(f"action site {action.site} is outside pair {pair.sites}")
    return bool(values[0]), bool(values[1])


def _site_for_progress(pair: PairModel, progress: tuple[bool, bool]) -> int:
    """The pair protocol is sequential: source first, then target."""

    if not progress[0]:
        return pair.source
    if not progress[1]:
        return pair.target
    return pair.target


def _options_for_progress(pair: PairModel, progress: tuple[bool, bool]) -> tuple[Candidate, ...]:
    if progress == (True, True):
        return ()
    return pair.actions(_site_for_progress(pair, progress))


def _choose_max_value(items: Sequence[tuple[float, float, Candidate, Any]]) -> tuple[float, float, Candidate, Any]:
    """Choose expected completion first, then lower expected cost, then ID."""

    best = items[0]
    for item in items[1:]:
        value, expected_cost, action, _ = item
        best_value, best_cost, best_action, _ = best
        if value > best_value + TIE_COST_TOLERANCE:
            best = item
        elif abs(value - best_value) <= TIE_COST_TOLERANCE:
            if expected_cost < best_cost - TIE_COST_TOLERANCE:
                best = item
            elif abs(expected_cost - best_cost) <= TIE_COST_TOLERANCE and action.action_id < best_action.action_id:
                best = item
    return best


class HorizonPlanner:
    """Exact finite-state Bayes horizon planner for one pair."""

    def __init__(self, pair: PairModel):
        self.pair = pair
        self.cache: dict[tuple[Any, ...], tuple[float, float, Candidate | None, dict[str, Any]]] = {}
        self.calls = 0

    @staticmethod
    def _belief_key(belief: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(belief, dtype=float), 12))

    def value(
        self,
        depth: int,
        progress: tuple[bool, bool],
        belief: np.ndarray,
        previous: tuple[int, int] | None,
    ) -> tuple[float, float, Candidate | None, dict[str, Any]]:
        self.calls += 1
        if progress == (True, True):
            return 1.0, 0.0, None, {"terminal": True, "outcome_actions": {}}
        if depth <= 0:
            return 0.0, 0.0, None, {"terminal": False, "outcome_actions": {}}
        key = (int(depth), progress, self._belief_key(belief), previous)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        options = _options_for_progress(self.pair, progress)
        if not options:
            return 0.0, 0.0, None, {"terminal": False, "outcome_actions": {}}
        rows: list[tuple[float, float, Candidate, dict[str, Any]]] = []
        for action in options:
            likelihood = self.pair.kernel(action)
            predictive = np.asarray(belief, dtype=float) @ likelihood
            value = 0.0
            expected_cost = self.pair.cost(previous, action)
            branch_rows = []
            for outcome, probability in enumerate(predictive):
                if probability <= 0.0:
                    continue
                next_belief = _posterior(belief, likelihood, outcome)
                next_progress = _progress_after(self.pair, progress, action, outcome)
                child_value, child_cost, child_action, _ = self.value(
                    depth - 1, next_progress, next_belief, (action.site, action.candidate)
                )
                value += float(probability) * child_value
                expected_cost += float(probability) * child_cost
                branch_rows.append(
                    {
                        "outcome": OUTCOME_LABELS[outcome],
                        "probability": float(probability),
                        "posterior": next_belief.tolist(),
                        "progress": list(next_progress),
                        "next_action": None if child_action is None else child_action.action_id,
                        "child_value": float(child_value),
                    }
                )
            rows.append((float(value), float(expected_cost), action, {"branches": branch_rows}))
        best = _choose_max_value(rows)
        result = (best[0], best[1], best[2], {"terminal": False, "outcome_actions": best[3]["branches"]})
        self.cache[key] = result
        return result


def _fixed_sequence(pair: PairModel) -> dict[str, Candidate]:
    """Select one source and one target window before held-out outcomes.

    The sequence then branches only on the public progress flag: a source miss
    retries the same source candidate; a source hit advances to the same target
    candidate.  Selection uses only the nominal finite prior and checked costs.
    """

    best: tuple[float, float, Candidate, Candidate] | None = None
    prior = MODE_PRIOR
    for source in pair.actions(pair.source):
        for target in pair.actions(pair.target):
            source_u = pair.kernel(source)[:, USABLE_OUTCOME_INDICES].sum(axis=1)
            target_u = pair.kernel(target)[:, USABLE_OUTCOME_INDICES].sum(axis=1)
            # Three shared acquisition slots, source retries before progress,
            # then target retries.  Conditional independence holds given mode.
            source_success = float(
                prior
                @ (
                    source_u * target_u * (2.0 - target_u)
                    + (1.0 - source_u) * source_u * target_u
                )
            )
            expected_cost = pair.cost(None, source) + pair.cost((source.site, source.candidate), source)
            expected_cost += pair.cost((source.site, source.candidate), target)
            score = source_success
            candidate_row = (score, expected_cost, source, target)
            if best is None or score > best[0] + TIE_COST_TOLERANCE or (
                abs(score - best[0]) <= TIE_COST_TOLERANCE and expected_cost < best[1]
            ):
                best = candidate_row
    assert best is not None
    return {"source": best[2], "target": best[3], "score": best[0], "expected_cost": best[1]}


def _select_action_fixed(pair: PairModel, progress: tuple[bool, bool], fixed: Mapping[str, Any]) -> Candidate:
    return fixed["source"] if not progress[0] else fixed["target"]


def _select_action_reactive(pair: PairModel, belief: np.ndarray, progress: tuple[bool, bool], previous: tuple[int, int] | None) -> tuple[Candidate, dict[str, Any]]:
    options = _options_for_progress(pair, progress)
    rows = []
    for action in options:
        usable = pair.expected_usable(action, belief)
        cost = pair.cost(previous, action)
        rows.append((usable - 1.0e-4 * cost, usable, cost, action))
    rows.sort(key=lambda row: (-row[0], row[2], row[3].action_id))
    score, usable, cost, action = rows[0]
    return action, {"greedy_score": float(score), "expected_usable": float(usable), "cost": float(cost)}


def _select_action_fisher(
    pair: PairModel,
    belief: np.ndarray,
    progress: tuple[bool, bool],
    previous: tuple[int, int] | None,
    mi_weight: float,
) -> tuple[Candidate, dict[str, Any]]:
    options = _options_for_progress(pair, progress)
    rows = []
    for action in options:
        usable = pair.expected_usable(action, belief)
        mi = pair.mutual_information_bits(action, belief)
        cost = pair.cost(previous, action)
        score = usable + float(mi_weight) * mi - 1.0e-4 * cost
        rows.append((score, usable, mi, cost, action))
    rows.sort(key=lambda row: (-row[0], row[3], row[4].action_id))
    score, usable, mi, cost, action = rows[0]
    return action, {
        "fisher_score": float(score),
        "expected_usable": float(usable),
        "mutual_information_bits": float(mi),
        "mi_weight": float(mi_weight),
        "cost": float(cost),
        "information_model": "exact public miss/quality-bin categorical MI over three persistent modes",
    }


def _select_action_uniform(
    pair: PairModel,
    progress: tuple[bool, bool],
    uniform_rng: np.random.Generator,
) -> tuple[Candidate, dict[str, Any]]:
    options = _options_for_progress(pair, progress)
    index = int(uniform_rng.integers(len(options)))
    action = options[index]
    return action, {
        "uniform_index": index,
        "uniform_bank_size": len(options),
        "sensor_control": "uniform over checked windows, independent of aperture width and outcome quality",
    }


def _episode_truth(pair_index: int, seed: int) -> tuple[int, float, float, float, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence((GLOBAL_SEED, int(pair_index), int(seed), 71)))
    mode = int(rng.choice(len(MODE_ANGLES_DEG), p=MODE_PRIOR))
    gain = float(np.clip(rng.normal(0.0, MISMATCH_GAIN_SD), -0.24, 0.24))
    quality = float(np.clip(rng.normal(0.0, MISMATCH_QUALITY_SD), -0.075, 0.075))
    mode_shift = float(np.clip(rng.normal(0.0, MISMATCH_MODE_SD_DEG), -4.5, 4.5))
    uniforms = rng.random(MAX_ACQUISITIONS)
    return mode, gain, quality, mode_shift, uniforms


def _sample_outcome(probabilities: np.ndarray, uniform: float) -> int:
    cdf = np.cumsum(np.asarray(probabilities, dtype=float))
    index = int(np.searchsorted(cdf, np.nextafter(float(uniform), 1.0), side="right"))
    return min(index, len(cdf) - 1)


def simulate_episode(
    pair: PairModel,
    pair_index: int,
    seed: int,
    method: str,
    *,
    mi_weight: float,
    max_acquisitions: int = MAX_ACQUISITIONS,
) -> dict[str, Any]:
    """Run one paired-replay arm using only public outcomes for decisions."""

    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    mode, gain_shift, quality_shift, mode_shift, uniforms = _episode_truth(pair_index, seed)
    belief = MODE_PRIOR.copy()
    progress = (False, False)
    previous: tuple[int, int] | None = None
    fixed = _fixed_sequence(pair) if method == "Fixed" else None
    uniform_rng = np.random.default_rng(np.random.SeedSequence((GLOBAL_SEED, pair_index, seed, 811)))
    planner = HorizonPlanner(pair) if method in {"B2", "B3"} else None
    trace: list[dict[str, Any]] = []
    total_cost = 0.0
    for step in range(int(max_acquisitions)):
        if progress == (True, True):
            break
        belief_before = belief.copy()
        progress_before = progress
        if method in {"B2", "B3"}:
            horizon = 2 if method == "B2" else 3
            value, expected_cost, action, planning = planner.value(
                min(horizon, int(max_acquisitions) - step), progress, belief, previous
            )
            if action is None:
                break
            action_info = {
                "expected_completion_value": float(value),
                "expected_cost": float(expected_cost),
                "horizon": min(horizon, int(max_acquisitions) - step),
                "exact_finite_bayes": True,
                "outcome_conditioned": True,
                "branch_plan": planning.get("outcome_actions", []),
                "cache_entries": len(planner.cache),
                "recursive_calls": planner.calls,
            }
            after_hit = [
                row for row in planning.get("outcome_actions", []) if row.get("outcome") in {"hit_q1", "hit_q2"}
            ]
            action_info["after_hit_next_site"] = any(
                row.get("next_action") is not None and f"site{pair.target}" in str(row.get("next_action"))
                for row in after_hit
            )
        elif method == "Fixed":
            action = _select_action_fixed(pair, progress, fixed)
            action_info = {
                "fixed_source_action": fixed["source"].action_id,
                "fixed_target_action": fixed["target"].action_id,
                "progress_only_branch": True,
                "fixed_nominal_score": float(fixed["score"]),
            }
        elif method == "A_reactive":
            action, action_info = _select_action_reactive(pair, belief, progress, previous)
            action_info["posterior_reactive"] = True
        elif method == "Fisher_MI":
            action, action_info = _select_action_fisher(pair, belief, progress, previous, mi_weight)
            action_info["posterior_reactive"] = True
            action_info["tuned_expected_information"] = True
        else:
            action, action_info = _select_action_uniform(pair, progress, uniform_rng)
        cost = pair.cost(previous, action)
        true_probabilities = pair.true_kernel(
            action,
            MODE_ANGLES_DEG[mode],
            gain_shift=gain_shift,
            quality_shift=quality_shift,
            mode_shift=mode_shift,
        )
        outcome = _sample_outcome(true_probabilities, uniforms[step])
        belief = _posterior(belief, pair.kernel(action), outcome)
        progress = _progress_after(pair, progress, action, outcome)
        hit = outcome != 0
        quality_bin = None if outcome == 0 else int(outcome - 1)
        usable = outcome in USABLE_OUTCOME_INDICES
        recovered = bool(not usable and step > 0)
        action_changed = bool(step > 0 and trace[-1]["action_id"] != action.action_id)
        replanned = bool(
            step > 0 and method in {"B2", "B3", "A_reactive", "Fisher_MI"}
        )
        total_cost += cost
        trace.append(
            {
                "acquisition_index": step + 1,
                "action_id": action.action_id,
                "site": int(action.site),
                "candidate": int(action.candidate),
                "offset_mm": float(action.offset_mm),
                "tilt_deg": float(action.tilt_deg),
                "depth_mm": float(action.depth_mm),
                "contact_obliquity_deg": float(action.contact_obliquity_deg),
                "transition_cost": float(cost),
                "current_exit": {"site": int(action.site), "candidate": int(action.candidate)},
                "outcome": OUTCOME_LABELS[outcome],
                "hit": bool(hit),
                "quality_bin": quality_bin,
                "usable": bool(usable),
                "correct": bool(usable),
                "wrong": False,
                "ordinary_completion_before": bool(progress_before == (True, True)),
                "ordinary_completion_after": bool(progress == (True, True)),
                "recovery": recovered,
                "replanned": replanned,
                "action_changed": action_changed,
                "progress_before": list(progress_before),
                "progress_after": list(progress),
                "belief_before": belief_before.tolist(),
                "belief_after": belief.tolist(),
                "planner": action_info,
            }
        )
        previous = (action.site, action.candidate)
    complete = progress == (True, True)
    return {
        "pair_index": int(pair_index),
        "source_station": int(pair.source),
        "target_station": int(pair.target),
        "seed": int(seed),
        "method": method,
        "fov_total_mm": FOV_TOTAL_MM,
        "max_acquisitions": int(max_acquisitions),
        "complete": bool(complete),
        "completion_acquisition": next((row["acquisition_index"] for row in trace if row["ordinary_completion_after"]), None),
        "acquisitions": len(trace),
        "total_cost": float(total_cost),
        "recovery_count": int(sum(row["recovery"] for row in trace)),
        "replanning_count": int(sum(row["replanned"] for row in trace)),
        "usable_hits": int(sum(row["usable"] for row in trace)),
        "trace": trace,
        "truth_audit": {
            "persistent_mode_index": int(mode),
            "persistent_mode_angle_deg": float(MODE_ANGLES_DEG[mode]),
            "unseen_gain_shift": float(gain_shift),
            "unseen_quality_shift": float(quality_shift),
            "unseen_mode_shift_deg": float(mode_shift),
            "not_available_to_policy": True,
        },
    }


def _tune_mi_weight(
    pairs: Sequence[PairModel],
    pair_indices: Sequence[int],
    seeds: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for weight in MI_WEIGHT_GRID:
        episodes = [
            simulate_episode(pairs[index], index, seed, "Fisher_MI", mi_weight=float(weight))
            for index in pair_indices
            for seed in seeds
        ]
        rows.append(
            {
                "mi_weight": float(weight),
                "episodes": len(episodes),
                "complete": int(sum(row["complete"] for row in episodes)),
                "completion_rate": float(np.mean([row["complete"] for row in episodes])),
                "mean_acquisitions": float(np.mean([row["acquisitions"] for row in episodes])),
                "mean_cost": float(np.mean([row["total_cost"] for row in episodes])),
                "mean_mutual_information_bits": float(
                    np.mean(
                        [
                            step["planner"].get("mutual_information_bits", 0.0)
                            for row in episodes
                            for step in row["trace"]
                            if "mutual_information_bits" in step["planner"]
                        ]
                    )
                ),
            }
        )
    selected = max(rows, key=lambda row: (row["completion_rate"], -row["mean_cost"], -row["mi_weight"]))
    return float(selected["mi_weight"]), rows


def _run_group(
    pairs: Sequence[PairModel],
    pair_indices: Sequence[int],
    seeds: Sequence[int],
    mi_weight: float,
) -> list[dict[str, Any]]:
    rows = []
    for pair_index in pair_indices:
        for seed in seeds:
            for method in METHODS:
                rows.append(
                    simulate_episode(
                        pairs[pair_index], pair_index, seed, method, mi_weight=mi_weight
                    )
                )
    return rows


def _model_matched_expected_completion(
    pair: PairModel,
    method: str,
    *,
    mi_weight: float,
    max_acquisitions: int = MAX_ACQUISITIONS,
) -> dict[str, Any]:
    """Integrate a policy exactly under its own nominal finite-state model.

    This is a deterministic branch sum, not a replay of the held-out seeds:
    all three persistent modes and every public categorical outcome are
    integrated through the policy's posterior updates.  It isolates planner
    approximation from the deliberately unseen truth mismatch used in the
    ordinary held-out replay.
    """

    if method not in {"B2", "B3", "Fixed", "A_reactive", "Fisher_MI"}:
        raise ValueError(f"model-matched expectation not defined for {method}")
    planner = HorizonPlanner(pair) if method in {"B2", "B3"} else None
    fixed = _fixed_sequence(pair) if method == "Fixed" else None
    calls = 0

    def exact_for_cap(cap: int) -> tuple[float, int]:
        cache: dict[tuple[Any, ...], float] = {}
        local_calls = 0

        def rec(
            step: int,
            progress: tuple[bool, bool],
            belief: np.ndarray,
            previous: tuple[int, int] | None,
        ) -> float:
            nonlocal local_calls
            local_calls += 1
            if progress == (True, True):
                return 1.0
            if step >= cap:
                return 0.0
            key = (step, progress, HorizonPlanner._belief_key(belief), previous)
            if key in cache:
                return cache[key]
            if method in {"B2", "B3"}:
                horizon = min(2 if method == "B2" else 3, cap - step)
                _, _, action, _ = planner.value(horizon, progress, belief, previous)
                assert action is not None
            elif method == "Fixed":
                action = _select_action_fixed(pair, progress, fixed)
            elif method == "A_reactive":
                action, _ = _select_action_reactive(pair, belief, progress, previous)
            else:
                action, _ = _select_action_fisher(pair, belief, progress, previous, mi_weight)
            likelihood = pair.kernel(action)
            predictive = belief @ likelihood
            total = 0.0
            for outcome, probability in enumerate(predictive):
                if probability <= 0.0:
                    continue
                posterior = _posterior(belief, likelihood, outcome)
                next_progress = _progress_after(pair, progress, action, outcome)
                total += float(probability) * rec(
                    step + 1,
                    next_progress,
                    posterior,
                    (action.site, action.candidate),
                )
            cache[key] = float(total)
            return float(total)

        value = float(rec(0, (False, False), MODE_PRIOR.copy(), None))
        return value, local_calls

    values: dict[str, float] = {}
    for cap in range(1, max_acquisitions + 1):
        values[str(cap)], local_calls = exact_for_cap(cap)
        calls += local_calls
    return {
        "pair": [pair.source, pair.target],
        "method": method,
        "max_acquisitions": int(max_acquisitions),
        "expected_completion": float(values[str(max_acquisitions)]),
        "expected_completion_by_shared_acquisition": values,
        "integration": "exact finite categorical branch sum under nominal model; no Monte Carlo",
        "persistent_modes_integrated": MODE_ANGLES_DEG.tolist(),
        "public_outcomes_integrated": list(OUTCOME_LABELS),
        "recursive_calls": int(calls),
        "planner_cache_entries": int(len(planner.cache)) if planner is not None else 0,
    }


def model_matched_expectations(
    pairs: Sequence[PairModel],
    pair_indices: Sequence[int],
    *,
    mi_weight: float,
    methods: Sequence[str] = ("B3", "A_reactive", "Fisher_MI", "Fixed"),
) -> dict[str, Any]:
    """Return exact held-out-pair integrated completion values for selected arms."""

    pair_rows = []
    for pair_index in pair_indices:
        for method in methods:
            row = _model_matched_expected_completion(
                pairs[pair_index], method, mi_weight=mi_weight, max_acquisitions=MAX_ACQUISITIONS
            )
            row["pair_index"] = int(pair_index)
            pair_rows.append(row)
    by_method: dict[str, Any] = {}
    for method in methods:
        selected = [row for row in pair_rows if row["method"] == method]
        by_method[method] = {
            "pairs": len(selected),
            "mean_expected_completion": float(np.mean([row["expected_completion"] for row in selected])),
            "mean_expected_completion_by_shared_acquisition": {
                str(k): float(
                    np.mean([row["expected_completion_by_shared_acquisition"][str(k)] for row in selected])
                )
                for k in range(1, MAX_ACQUISITIONS + 1)
            },
        }
    comparisons = {}
    if "B3" in by_method and "Fixed" in by_method:
        comparisons["B3_minus_Fixed"] = float(
            by_method["B3"]["mean_expected_completion"]
            - by_method["Fixed"]["mean_expected_completion"]
        )
    if "B3" in by_method and "A_reactive" in by_method:
        comparisons["B3_minus_A_reactive"] = float(
            by_method["B3"]["mean_expected_completion"]
            - by_method["A_reactive"]["mean_expected_completion"]
        )
    if "Fisher_MI" in by_method and "A_reactive" in by_method:
        comparisons["Fisher_MI_minus_A_reactive"] = float(
            by_method["Fisher_MI"]["mean_expected_completion"]
            - by_method["A_reactive"]["mean_expected_completion"]
        )
    return {
        "study": "model_matched_exact_expected_completion",
        "pair_indices": [int(index) for index in pair_indices],
        "methods": list(methods),
        "selected_mi_weight": float(mi_weight),
        "summary": by_method,
        "comparisons": comparisons,
        "pair_rows": pair_rows,
        "no_monte_carlo": True,
        "interpretation": "This is a nominal-model diagnostic; held-out replay remains the primary mismatch evidence.",
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, group: str, pair_indices: Sequence[int], seeds: Sequence[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "group": group,
        "pair_indices": [int(value) for value in pair_indices],
        "seeds": [int(value) for value in seeds],
        "episodes_per_method": len(pair_indices) * len(seeds),
        "methods": {},
    }
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        curve = []
        for k in range(1, MAX_ACQUISITIONS + 1):
            completed = sum(
                bool(any(step["ordinary_completion_after"] and step["acquisition_index"] <= k for step in row["trace"]))
                for row in selected
            )
            curve.append({"acquisitions": k, "complete": int(completed), "completion_rate": float(completed / max(1, len(selected)))})
        summary["methods"][method] = {
            "episodes": len(selected),
            "complete": int(sum(row["complete"] for row in selected)),
            "completion_rate": float(np.mean([row["complete"] for row in selected])),
            "mean_acquisitions": float(np.mean([row["acquisitions"] for row in selected])),
            "mean_cost": float(np.mean([row["total_cost"] for row in selected])),
            "mean_recovery_count": float(np.mean([row["recovery_count"] for row in selected])),
            "mean_replanning_count": float(np.mean([row["replanning_count"] for row in selected])),
            "cumulative_complete_by_shared_acquisition": curve,
        }
    comparisons = {}
    for left, right in (("B2", "Fixed"), ("B3", "Fixed"), ("B3", "A_reactive"), ("Fisher_MI", "A_reactive")):
        lhs = summary["methods"][left]
        rhs = summary["methods"][right]
        comparisons[f"{left}_minus_{right}"] = {
            "complete_count_difference": int(lhs["complete"] - rhs["complete"]),
            "completion_rate_difference": float(lhs["completion_rate"] - rhs["completion_rate"]),
            "shared_acquisition_curve_difference": [
                {
                    "acquisitions": int(a["acquisitions"]),
                    "complete_count_difference": int(a["complete"] - b["complete"]),
                    "completion_rate_difference": float(a["completion_rate"] - b["completion_rate"]),
                }
                for a, b in zip(
                    lhs["cumulative_complete_by_shared_acquisition"],
                    rhs["cumulative_complete_by_shared_acquisition"],
                )
            ],
            "interpretation": "paired descriptive evidence; no winner is assumed",
        }
    summary["comparisons"] = comparisons
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def run_study(
    *,
    output_dir: Path,
    development_seeds: Sequence[int] = DEFAULT_DEV_SEEDS,
    heldout_seeds: Sequence[int] = DEFAULT_HELDOUT_SEEDS,
    smoke: bool = False,
) -> dict[str, Any]:
    """Build geometry, preselect on development pairs, and run held-out A/B."""

    scan, payload, pair_rows, same_first = _load_geometry()
    pairs = [PairModel(scan, payload, int(row["source_station"]), int(row["target_station"])) for row in pair_rows]
    development_indices, heldout_indices = _pair_indices(len(pairs))
    if smoke:
        development_indices = development_indices[:2]
        heldout_indices = heldout_indices[:2]
        development_seeds = tuple(list(development_seeds)[:2])
        heldout_seeds = tuple(list(heldout_seeds)[:3])
    mi_weight, tuning = _tune_mi_weight(pairs, development_indices, development_seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "DEVELOPMENT_SELECTION.json",
        {
            "selection_rule": "preselect Fisher MI weight on development pairs/seeds only; highest completion, then lower checked cost, then lower weight",
            "development_pair_indices": list(development_indices),
            "development_pairs": [[pairs[index].source, pairs[index].target] for index in development_indices],
            "development_seeds": list(map(int, development_seeds)),
            "mi_weight_grid": list(MI_WEIGHT_GRID),
            "selected_mi_weight": mi_weight,
            "tuning_rows": tuning,
            "heldout_outcomes_not_used": True,
        },
    )
    development_rows = _run_group(pairs, development_indices, development_seeds, mi_weight)
    heldout_rows = _run_group(pairs, heldout_indices, heldout_seeds, mi_weight)
    with (output_dir / "DEVELOPMENT_RUNS.jsonl").open("w") as handle:
        for row in development_rows:
            handle.write(json.dumps(_json_safe(row), separators=(",", ":")) + "\n")
    with (output_dir / "HELDOUT_RUNS.jsonl").open("w") as handle:
        for row in heldout_rows:
            handle.write(json.dumps(_json_safe(row), separators=(",", ":")) + "\n")
    _write_json(output_dir / "PAIR_GEOMETRY.json", [pairs[index].geometry_record() for index in range(len(pairs))])
    _write_json(
        output_dir / "SUMMARY.json",
        {
            "study": "local_persistent_angular_window_continuation",
            "scope": "local synthetic mechanism benchmark, not a full-leg result",
            "geometry": str(GEOMETRY_PATH.resolve()),
            "geometry_sha256": _sha256(GEOMETRY_PATH),
            "same_first_window": str(SAME_FIRST_PATH.resolve()),
            "same_first_window_sha256": _sha256(SAME_FIRST_PATH),
            "pose_index": POSE_INDEX,
            "pose_degrees": POSE_DEGREES,
            "fov_total_mm": FOV_TOTAL_MM,
            "fov_bounds_mm": [-FOV_HALF_WIDTH_MM, FOV_HALF_WIDTH_MM],
            "fov_half_width_mm": FOV_HALF_WIDTH_MM,
            "included_pair_count": len(pairs),
            "all_included_pairs": [[pair.source, pair.target] for pair in pairs],
            "development_pair_indices": list(development_indices),
            "heldout_pair_indices": list(heldout_indices),
            "heldout_pairs": [[pairs[index].source, pairs[index].target] for index in heldout_indices],
            "mode_angles_deg": MODE_ANGLES_DEG.tolist(),
            "mode_prior": MODE_PRIOR.tolist(),
            "quality_edges": QUALITY_EDGES.tolist(),
            "usable_quality_bins": list(USABLE_QUALITY_BINS),
            "outcomes": list(OUTCOME_LABELS),
            "persistent_state_scope": "one mode shared across both sites of each pair",
            "truth_model_mismatch": {
                "gain_shift_sd": MISMATCH_GAIN_SD,
                "quality_shift_sd": MISMATCH_QUALITY_SD,
                "mode_shift_sd_deg": MISMATCH_MODE_SD_DEG,
                "planner_has_no_access": True,
            },
            "methods": {
                "B2": "exact finite-state Bayes horizon-2, outcome-conditioned, rolling replanning",
                "B3": "exact finite-state Bayes horizon-3, outcome-conditioned, rolling replanning",
                "Fixed": "fixed continuation; only public source/target progress flag branches",
                "A_reactive": "posterior expected-usable one-step greedy",
                "Fisher_MI": "posterior greedy with tuned optimistic discrete-state MI term",
                "Uniform": "uniform checked-window control independent of FOV width",
            },
            "max_acquisitions": MAX_ACQUISITIONS,
            "no_method_specific_deadline": True,
            "development": _summary(development_rows, group="development", pair_indices=development_indices, seeds=development_seeds),
            "heldout": _summary(heldout_rows, group="heldout", pair_indices=heldout_indices, seeds=heldout_seeds),
            "paired_environment": "same pair/seed hidden mode, mismatch, and public acquisition-clock uniforms across methods",
            "actual_ab_evidence_if_negative": True,
            "selected_mi_weight": mi_weight,
            "same_first_protocol_excerpt": {
                "study": same_first.get("study"),
                "common_bias_support_gate": same_first.get("support_gate"),
                "included_pair_count": same_first.get("included_pair_count"),
            },
        },
    )
    return {
        "output_dir": str(output_dir.resolve()),
        "included_pair_count": len(pairs),
        "development_pair_indices": development_indices,
        "heldout_pair_indices": heldout_indices,
        "selected_mi_weight": mi_weight,
        "development": _summary(development_rows, group="development", pair_indices=development_indices, seeds=development_seeds),
        "heldout": _summary(heldout_rows, group="heldout", pair_indices=heldout_indices, seeds=heldout_seeds),
    }


def write_model_matched_file(output_dir: Path) -> dict[str, Any]:
    """Write exact nominal-model expectations for the frozen held-out split."""

    scan, payload, pair_rows, _ = _load_geometry()
    pairs = [PairModel(scan, payload, int(row["source_station"]), int(row["target_station"])) for row in pair_rows]
    _, heldout_indices = _pair_indices(len(pairs))
    selection_path = output_dir / "DEVELOPMENT_SELECTION.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"run the ordinary study first: {selection_path}")
    selected_weight = float(json.loads(selection_path.read_text())["selected_mi_weight"])
    result = model_matched_expectations(pairs, heldout_indices, mi_weight=selected_weight)
    result.update(
        {
            "geometry": str(GEOMETRY_PATH.resolve()),
            "geometry_sha256": _sha256(GEOMETRY_PATH),
            "same_first_window": str(SAME_FIRST_PATH.resolve()),
            "heldout_pairs": [[pairs[index].source, pairs[index].target] for index in heldout_indices],
            "selected_weight_source": str(selection_path.resolve()),
            "truth_mismatch_in_replay": {
                "gain_shift_sd": MISMATCH_GAIN_SD,
                "quality_shift_sd": MISMATCH_QUALITY_SD,
                "mode_shift_sd_deg": MISMATCH_MODE_SD_DEG,
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "MODEL_MATCHED_EXPECTATIONS.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "local_study")
    parser.add_argument("--dev-seeds", type=int, default=len(DEFAULT_DEV_SEEDS))
    parser.add_argument("--heldout-seeds", type=int, default=len(DEFAULT_HELDOUT_SEEDS))
    parser.add_argument("--smoke", action="store_true", help="use two pairs and a few seeds for a quick verification")
    parser.add_argument(
        "--model-matched-only",
        action="store_true",
        help="write exact nominal-model expected completion for the frozen held-out split",
    )
    args = parser.parse_args(argv)
    if args.dev_seeds <= 0 or args.heldout_seeds <= 0:
        parser.error("seed counts must be positive")
    if args.model_matched_only:
        result = write_model_matched_file(args.out)
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return 0
    dev_seeds = tuple(range(args.dev_seeds))
    heldout_seeds = tuple(range(1000, 1000 + args.heldout_seeds))
    result = run_study(output_dir=args.out, development_seeds=dev_seeds, heldout_seeds=heldout_seeds, smoke=args.smoke)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

