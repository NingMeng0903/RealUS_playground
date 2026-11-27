"""Bounded rollout collection/evaluation for the frozen ProDMP recheck.

The implementation in this file is deliberately a thin wrapper around the
frozen ``ab_b_refinement_20261112`` study.  The frozen study owns geometry,
the common head planner, evidence gates, feasibility checks, and the motion
primitive.  This module only executes the already planned suffix and records a
bounded, public-feature candidate shortlist for each executed tail decision.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "ab_b_refinement_20261112"
# detection_branches.py imports the frozen refine_rechecks module by its public
# module name.  Put the frozen directory first and do not copy or modify either
# frozen source file.
if str(FROZEN) not in sys.path:
    sys.path.insert(0, str(FROZEN))
import detection_branches as branches  # noqa: E402
import refine_rechecks as study  # noqa: E402


# The aliases are the exact two public branches installed by the frozen
# detection_branches module's CLI.  Installing them in memory is equivalent to
# the frozen CLI setup and leaves the source study untouched.
study.ARMS.setdefault("locked_seen", study.ARMS["locked"])
study.ARMS.setdefault("B_seen", study.ARMS["B"])


K = 4
CAP_DEFAULT = 80
N_PREDICTIVE_DEFAULT = 8
N_FIRST_DEFAULT = 3
SWITCH_MARGIN_DEFAULT = 0.006
BEHAVIOR_ANCHOR_PROBABILITY = 0.60
BEHAVIOR_ALTERNATIVE_PROBABILITY = 0.40
POLICIES = ("rule", "immediate", "rl")


def _feature_names(n_targets: int = 32) -> tuple[str, ...]:
    names = [
        "phase",
        "remaining",
        "detected",
        "progress",
        "current_pose",
        "current_site",
    ]
    names.extend(f"count_{i}" for i in range(n_targets))
    names.extend(f"variance_{i}" for i in range(n_targets))
    names.extend(
        [
            "candidate_site",
            "candidate_view",
            "exact_movement_m",
            "score",
            "delta",
            "anchor",
        ]
    )
    return tuple(names)


# The frozen recheck has 32 target rows.  Expose names for fit_q/tests without
# making feature construction depend on a private model constant.
FEATURE_NAMES = _feature_names()
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class CandidateShortlist:
    """A deterministic, legal K-way suffix shortlist for one public state."""

    indices: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    scores: np.ndarray
    distances: np.ndarray
    anchor_index: int
    b_index: int
    anchor_choice: Any
    b_choice: Any


@dataclass
class ScheduledDecision:
    """A shortlist selected after a head and executed at the next tail step."""

    shortlist: CandidateShortlist
    selected_slot: int
    selected_index: int
    policy: str
    decision_number: int
    fisher_scores: np.ndarray | None = None
    behavior_propensity: float | None = None

    @property
    def switched(self) -> bool:
        return self.selected_index != self.shortlist.anchor_index


class RLPlanner(branches.DetectionPlanner):
    """Frozen detection planner with shortlist construction for suffixes.

    ``plan`` and the inherited ``choose`` method are intentionally unchanged.
    ``choose`` is called twice by :meth:`shortlist`: once for the exact
    ``locked_seen`` anchor and once for the exact ``B_seen`` choice.  This
    keeps the common head and all frozen evidence/feasibility behavior intact.
    """

    def _legal_options(self, program, previous, pose):
        arm = study.ARMS["B_seen"]
        options = np.asarray(program.options, dtype=int)
        legal = [
            int(index)
            for index in options
            if self._within_budget(previous, pose, int(index), np.inf, arm)
        ]
        return options, legal

    def fisher_proxy(self, pose, previous, belief, counts, shortlist):
        """Return the reliability-weighted directional Fisher proxy.

        The two calls deliberately use the same public state, candidate
        support, exact movement costs, and posterior score. Their difference
        therefore isolates only the ``Arm.fisher`` contribution.
        """

        valid = np.flatnonzero(shortlist.mask)
        result = np.full(K, np.nan, dtype=float)
        arm = study.ARMS["B_seen"]
        if not valid.size or not float(arm.fisher):
            return result
        indices = np.asarray(shortlist.indices[valid], dtype=int)
        with_fisher = np.asarray(
            self.score(pose, previous, belief, counts, arm, indices, exact=True),
            dtype=float,
        )
        no_fisher_arm = study.Arm(
            gate=arm.gate,
            mode=arm.mode,
            fisher=0.0,
            travel_scale=arm.travel_scale,
            learned_primitive=arm.learned_primitive,
            keep_prefix_witnesses=arm.keep_prefix_witnesses,
        )
        without_fisher = np.asarray(
            self.score(
                pose,
                previous,
                belief,
                counts,
                no_fisher_arm,
                indices,
                exact=True,
            ),
            dtype=float,
        )
        result[valid] = (with_fisher - without_fisher) / float(arm.fisher)
        return result

    def shortlist(
        self,
        program,
        pose,
        previous,
        belief,
        before,
        after,
        *,
        step: int,
        cap: int,
    ) -> CandidateShortlist | None:
        """Build the shared K<=4 candidate set after an executed head.

        The first row is always the exact locked-seen branch.  The original
        B-seen choice and the best posterior-score alternative are included
        whenever distinct; remaining slots are filled by highest-score legal
        near/cheaper proposals from ``program.options``.
        """

        if program is None:
            return None

        locked_arm = study.ARMS["locked_seen"]
        b_arm = study.ARMS["B_seen"]
        anchor_choice = self.choose(
            program, pose, previous, belief, before, after, locked_arm, False
        )
        b_choice = self.choose(
            program, pose, previous, belief, before, after, b_arm, False
        )

        options, legal = self._legal_options(program, previous, pose)
        if not legal:
            return None
        legal_set = set(legal)
        anchor_index = int(anchor_choice.index)
        b_index = int(b_choice.index)
        if anchor_index not in legal_set:
            # This should be impossible because the frozen suffix planner
            # filters by the same edge/budget contract.  Refuse to schedule a
            # candidate rather than silently executing an illegal action.
            return None

        legal_arr = np.asarray([int(i) for i in options if int(i) in legal_set], dtype=int)
        scores = np.asarray(
            self.score(pose, previous, belief, after, b_arm, legal_arr, exact=True),
            dtype=float,
        )
        score_by_index = {int(i): float(s) for i, s in zip(legal_arr, scores)}
        distance_by_index = {
            int(i): float(
                self.motion(
                    previous,
                    pose,
                    tuple(self.actions[pose][int(i)]),
                    b_arm.learned_primitive,
                ).length_m
            )
            for i in legal_arr
        }
        anchor_score = score_by_index[anchor_index]
        anchor_distance = distance_by_index[anchor_index]

        # Stable deterministic ranking: score first, then exact movement, then
        # frozen program order.  All proposals remain members of the frozen
        # program's legal support.
        order = sorted(
            (int(i) for i in legal_arr),
            key=lambda i: (-score_by_index[i], distance_by_index[i], i),
        )
        best_alternative = next((i for i in order if i != anchor_index), None)
        near_or_cheaper = [
            i
            for i in order
            if i != anchor_index and distance_by_index[i] <= anchor_distance + 1.0e-10
        ]

        selected: list[int] = [anchor_index]
        for candidate in (b_index, best_alternative):
            if candidate is not None and candidate in legal_set and candidate not in selected:
                selected.append(int(candidate))
                if len(selected) >= K:
                    break
        if len(selected) < K:
            for candidate in near_or_cheaper:
                if candidate not in selected:
                    selected.append(int(candidate))
                    if len(selected) >= K:
                        break
        if len(selected) < K:
            for candidate in order:
                if candidate not in selected:
                    selected.append(int(candidate))
                    if len(selected) >= K:
                        break

        selected = selected[:K]
        selected_arr = np.asarray(selected, dtype=int)
        selected_scores = np.asarray([score_by_index[i] for i in selected], dtype=float)
        selected_distances = np.asarray([distance_by_index[i] for i in selected], dtype=float)
        mask = np.zeros(K, dtype=bool)
        mask[: len(selected)] = True

        counts = np.asarray(after, dtype=float)
        _, variance, _, _ = self.public_book.status(belief)
        # The branch flag is public evidence stored by DetectionBook.  It is
        # intentionally not derived from eval-only labels.
        detected = bool(getattr(self.public_book, "observations_present", False))
        progress = bool(np.any(np.asarray(after) > np.asarray(before)))
        phase = float(np.clip((step + 1) / max(int(cap), 1), 0.0, 1.0))
        remaining = float(max(int(cap) - step - 1, 0) / max(int(cap), 1))
        pose_den = max(len(self.scan.angles) - 1, 1)
        site_den = max(int(self.scan.n) - 1, 1)
        n_views = int(self.scan.views.shape[2])
        view_den = max(n_views - 1, 1)
        current_site = int(previous[1]) if previous is not None else -1
        current_pose = float(np.clip(int(pose) / pose_den, 0.0, 1.0))
        current_site_value = float(np.clip(current_site / site_den, 0.0, 1.0))
        var_limit = max(float(self.model.var_limit), 1.0e-12)
        count_values = counts / max(float(study.VERIFY), 1.0)
        variance_values = np.asarray(variance, dtype=float) / var_limit
        rows: list[np.ndarray] = []
        for i, score, distance in zip(selected, selected_scores, selected_distances):
            site, view = map(int, self.actions[pose][int(i)])
            row = np.concatenate(
                [
                    np.asarray(
                        [
                            phase,
                            remaining,
                            float(detected),
                            float(progress),
                            current_pose,
                            current_site_value,
                        ],
                        dtype=float,
                    ),
                    count_values,
                    variance_values,
                    np.asarray(
                        [
                            float(site / site_den),
                            float(view / view_den),
                            float(distance),
                            float(score),
                            float(score - anchor_score),
                            float(int(i) == anchor_index),
                        ],
                        dtype=float,
                    ),
                ]
            )
            rows.append(row)
        features = np.zeros((K, len(rows[0]) if rows else FEATURE_DIM), dtype=np.float32)
        if rows:
            features[: len(rows)] = np.asarray(rows, dtype=np.float32)
        return CandidateShortlist(
            indices=np.pad(selected_arr, (0, K - len(selected_arr)), constant_values=-1),
            features=features,
            mask=mask,
            scores=np.pad(selected_scores, (0, K - len(selected_scores))),
            distances=np.pad(selected_distances, (0, K - len(selected_distances))),
            anchor_index=anchor_index,
            b_index=b_index,
            anchor_choice=anchor_choice,
            b_choice=b_choice,
        )


def _policy_action(
    shortlist: CandidateShortlist,
    *,
    mode: str,
    q_model: Any = None,
    episode_seed: int,
    decision_number: int,
    collection_behavior: str = "uniform",
    fisher_scores: np.ndarray | None = None,
) -> tuple[int, int]:
    """Return ``(slot, global-action-index)`` for one shortlist."""

    valid = np.flatnonzero(shortlist.mask)
    if len(valid) == 0:
        raise ValueError("empty legal suffix shortlist")
    anchor_slots = np.flatnonzero(shortlist.indices == shortlist.anchor_index)
    anchor_slot = int(anchor_slots[0]) if len(anchor_slots) else int(valid[0])

    if mode == "collect":
        if collection_behavior not in ("uniform", "fisher"):
            raise ValueError(f"unknown collection behavior {collection_behavior!r}")
        # This RNG is derived solely from the episode seed and decision index;
        # no train-case, checkpoint, or global RNG state participates.
        rng = np.random.default_rng(
            np.random.SeedSequence([int(episode_seed), 4242, int(decision_number)])
        )
        if float(rng.random()) < BEHAVIOR_ANCHOR_PROBABILITY:
            slot = anchor_slot
        else:
            slot = int(valid[int(rng.integers(len(valid)))])
        # Preserve the exact old uniform draw above. Fisher guidance is an
        # optional second-stage replacement only after that draw actually
        # selected a non-anchor. A failed .75 draw retains the original
        # uniform support exactly.
        if (
            collection_behavior == "fisher"
            and slot != anchor_slot
            and fisher_scores is not None
            and float(rng.random()) < 0.75
        ):
            nonanchor = valid[valid != anchor_slot]
            values = np.asarray(fisher_scores, dtype=float).reshape(-1)
            if len(values) >= K and len(nonanchor):
                finite_values = np.where(np.isfinite(values), values, -np.inf)
                # ``nonanchor`` is already stable shortlist order, so argmax
                # gives deterministic tie behavior independent of sorting.
                best = int(nonanchor[int(np.argmax(finite_values[nonanchor]))])
                if np.isfinite(finite_values[best]):
                    slot = best
        return slot, int(shortlist.indices[slot])

    if mode == "rule":
        return anchor_slot, int(shortlist.indices[anchor_slot])
    if mode == "immediate":
        b_slots = np.flatnonzero(shortlist.indices == shortlist.b_index)
        slot = int(b_slots[0]) if len(b_slots) else anchor_slot
        return slot, int(shortlist.indices[slot])
    if mode != "rl":
        raise ValueError(f"unknown rollout policy {mode!r}")
    if q_model is None:
        raise ValueError("rl evaluation requires a FrozenQ checkpoint")
    # Keep this call exactly in the public fit_q API requested by the study.
    predicted = np.asarray(q_model.predict(shortlist.features, shortlist.mask), dtype=float)
    predicted = predicted.reshape(-1)
    if len(predicted) != K:
        raise ValueError(f"FrozenQ.predict returned {len(predicted)} values, expected {K}")
    predicted = np.where(np.isfinite(predicted), predicted, -np.inf)
    best_slot = int(valid[np.argmax(predicted[valid])])
    if predicted[best_slot] >= predicted[anchor_slot] + 0.005:
        return best_slot, int(shortlist.indices[best_slot])
    return anchor_slot, int(shortlist.indices[anchor_slot])


def _behavior_propensity(
    shortlist: CandidateShortlist,
    selected_slot: int,
    *,
    collection_behavior: str,
    fisher_scores: np.ndarray | None,
) -> float:
    """Return the exact probability of the selected behavior action."""

    valid = np.flatnonzero(shortlist.mask)
    n_valid = max(len(valid), 1)
    anchor = int(np.flatnonzero(shortlist.indices == shortlist.anchor_index)[0])
    if collection_behavior == "uniform" or fisher_scores is None:
        return float(
            BEHAVIOR_ANCHOR_PROBABILITY * (selected_slot == anchor)
            + BEHAVIOR_ALTERNATIVE_PROBABILITY / n_valid
        )
    values = np.asarray(fisher_scores, dtype=float).reshape(-1)
    nonanchor = valid[valid != anchor]
    best = None
    if len(values) >= K and len(nonanchor):
        finite_values = np.where(np.isfinite(values), values, -np.inf)
        candidate = int(nonanchor[int(np.argmax(finite_values[nonanchor]))])
        if np.isfinite(finite_values[candidate]):
            best = candidate
    if selected_slot == anchor:
        # Fisher guidance is only consulted after the uniform branch selected
        # a non-anchor, so the anchor mass is unchanged.
        return float(BEHAVIOR_ANCHOR_PROBABILITY + BEHAVIOR_ALTERNATIVE_PROBABILITY / n_valid)
    if best is None:
        return float(BEHAVIOR_ALTERNATIVE_PROBABILITY / n_valid)
    if best is not None and selected_slot == best:
        return float(
            BEHAVIOR_ALTERNATIVE_PROBABILITY / n_valid * 0.25
            + BEHAVIOR_ALTERNATIVE_PROBABILITY * (n_valid - 1) / n_valid * 0.75
        )
    return float(BEHAVIOR_ALTERNATIVE_PROBABILITY / n_valid * 0.25)


def _jsonable(value: Any) -> Any:
    """Convert numpy/scalar/path values into strict JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_model_planner(
    *, n_predictive: int, n_first: int, switch_margin: float
) -> tuple[Any, Any, Any, Any]:
    geometry = study.DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(geometry) as data:
        scan = study.DynamicScan(data)
        support = study.build_support(scan, data)
    model = study.Recheck(scan, study.Edges(scan))
    planner = RLPlanner(
        model,
        travel_weight=1.4,
        n_predictive=n_predictive,
        n_first=n_first,
        switch_margin=switch_margin,
        predictive_z=2.0,
    )
    return scan, support, model, planner


def _initial_belief(scan):
    belief = np.zeros(scan.dim), scan.prior.copy()
    belief[1][-1, -1] = study.STRESS.slip_sd_mm**2
    return belief


def prepare_episode(scan, support, episode_seed: int):
    """Prepare deterministic truth/prefix state once for paired policies."""

    truth, rejects = study.draw_truth(scan, episode_seed, support)
    shared = scan.run(study.STRESS, episode_seed, "A", truth)
    return truth, rejects, shared["steps"]


def _tail_choice_payload(
    decision: ScheduledDecision,
    planner: RLPlanner,
    pose: int,
) -> dict[str, Any]:
    shortlist = decision.shortlist
    anchor_action = planner.actions[pose][shortlist.anchor_index]
    selected_action = planner.actions[pose][decision.selected_index]
    b_action = planner.actions[pose][shortlist.b_index]
    payload = {
        # These four fields exactly match the frozen step.tail_choice fields.
        "locked": list(map(int, planner.actions[pose][shortlist.anchor_choice.locked_index])),
        "usable": bool(shortlist.anchor_choice.usable),
        "gain": float(shortlist.anchor_choice.gain),
        "predicted_gain": float(shortlist.anchor_choice.predicted_gain),
        # Additional provenance is public and makes the selected suffix
        # unambiguous when the shortlist contains more than two proposals.
        "policy": decision.policy,
        "anchor": list(map(int, anchor_action)),
        "selected": list(map(int, selected_action)),
        "selected_slot": int(decision.selected_slot),
        "b_seen": list(map(int, b_action)),
        "candidate_indices": shortlist.indices.tolist(),
        "candidate_mask": shortlist.mask.tolist(),
        "changed_suffix": bool(decision.switched),
    }
    if decision.fisher_scores is not None:
        payload["fisher_scores"] = np.asarray(decision.fisher_scores, dtype=float).tolist()
    if decision.behavior_propensity is not None:
        payload["behavior_propensity"] = float(decision.behavior_propensity)
    return payload


def run_episode(
    planner: RLPlanner,
    scan: Any,
    support: Any,
    model: Any,
    *,
    episode_seed: int,
    mode: str,
    policy: str,
    q_model: Any,
    cap: int,
    reference_step=None,
    collection_behavior: str = "uniform",
    prepared_episode=None,
    tail_override=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Roll out a policy, optionally overriding one public tail choice.

    If supplied, ``tail_override`` is called as
    ``tail_override(shortlist, tuple(values), decision_number, default_slot)``
    after normal policy selection. It may return ``None`` or a valid shortlist
    slot; no private simulator state is passed to the callback.
    """

    if collection_behavior not in ("uniform", "fisher"):
        raise ValueError(f"unknown collection behavior {collection_behavior!r}")

    if prepared_episode is None:
        truth, rejects, prefix = prepare_episode(scan, support, episode_seed)
    else:
        truth, rejects, prefix = prepared_episode
    book = branches.DetectionBook(model, gate="predictive", predictive_z=2.0)
    belief = _initial_belief(scan)
    previous = None
    for item in prefix:
        pose, action = int(item["pose"]), tuple(item["action"])
        belief = book.assimilate(
            belief,
            pose,
            action,
            item["observation_mm"],
            item["eval_only_labels"],
        )
        previous = (pose, *action)

    counts, _, ready, _ = book.status(belief)
    initial = int(ready.sum())
    start = int(
        np.random.default_rng(np.random.SeedSequence([episode_seed, 81])).integers(
            len(scan.moving)
        )
    )
    scheduled: ScheduledDecision | None = None
    switched = False
    suffix_steps: list[dict[str, Any]] = []
    executed_decisions: list[dict[str, Any]] = []
    distance = 0.0
    decision_number = 0
    terminal_reason = "cap"

    for k in range(int(cap)):
        if np.all(ready):
            terminal_reason = "complete"
            break
        pose = scan.pose(study.STRESS, len(prefix) + k, start)
        next_pose = scan.pose(study.STRESS, len(prefix) + k + 1, start)
        plan = None
        decision = None
        is_tail = scheduled is not None
        if scheduled is not None:
            decision = scheduled
            scheduled = None
            action_index = int(decision.selected_index)
            action = tuple(planner.actions[pose][action_index])
        else:
            # locked_seen and B_seen share all Arm fields that affect the head
            # objective.  B_seen is used for the common head and the frozen
            # suffix program; only execution policy differs below.
            index, plan = planner.plan(
                book,
                pose,
                next_pose,
                previous,
                belief,
                counts,
                study.ARMS["B_seen"],
                np.inf,
                k + 1 == int(cap),
            )
            if index is None:
                terminal_reason = "failure"
                break
            action_index = int(index)
            action = tuple(planner.actions[pose][action_index])

        if not model.edge.allow(previous, (pose, *action)):
            raise RuntimeError("unchecked exit-to-contact edge")
        motion = planner.motion(
            previous,
            pose,
            action,
            study.ARMS["B_seen"].learned_primitive,
        )
        if reference_step is not None:
            reference_step(previous, (pose, *action), motion.skill)
        distance += motion.length_m
        values, labels = study.observe(
            model, pose, action, truth, episode_seed, k
        )
        before = np.asarray(counts, dtype=float).copy()
        belief = book.assimilate(belief, pose, action, values, labels)
        counts, _, ready, _ = book.status(belief)
        previous = (pose, *action)

        changed = bool(decision is not None and decision.switched)
        switched |= changed
        if decision is not None:
            # This list is intentionally appended only when the tail action is
            # actually executed.  A final scheduled tail that is skipped by
            # the online stop condition never becomes a transition row.
            executed_decisions.append(
                {
                    "step_index": len(suffix_steps),
                    "decision": decision,
                    "action": action,
                }
            )
        if plan is not None:
            shortlist = planner.shortlist(
                plan,
                next_pose,
                previous,
                belief,
                before,
                counts,
                step=k,
                cap=cap,
            )
            if shortlist is not None:
                selection_mode = "collect" if mode == "collect" else policy
                fisher_scores = None
                if mode == "collect" and collection_behavior == "fisher":
                    fisher_scores = planner.fisher_proxy(
                        next_pose, previous, belief, counts, shortlist
                    )
                selected_slot, selected_index = _policy_action(
                    shortlist,
                    mode=selection_mode,
                    q_model=q_model,
                    episode_seed=episode_seed,
                    decision_number=decision_number,
                    collection_behavior=collection_behavior,
                    fisher_scores=fisher_scores,
                )
                if tail_override is not None:
                    override = tail_override(
                        shortlist,
                        tuple(values),
                        decision_number,
                        selected_slot,
                    )
                    if override is not None:
                        if isinstance(override, (bool, np.bool_)) or not isinstance(
                            override, (int, np.integer)
                        ):
                            raise ValueError("tail_override must return None or an integer shortlist slot")
                        override_slot = int(override)
                        if (
                            override_slot < 0
                            or override_slot >= len(shortlist.mask)
                            or not bool(shortlist.mask[override_slot])
                        ):
                            raise ValueError(
                                f"tail_override returned illegal shortlist slot {override_slot}"
                            )
                        selected_slot = override_slot
                        selected_index = int(shortlist.indices[selected_slot])
                behavior_propensity = _behavior_propensity(
                    shortlist,
                    selected_slot,
                    collection_behavior=collection_behavior,
                    fisher_scores=fisher_scores,
                ) if mode == "collect" else None
                scheduled = ScheduledDecision(
                    shortlist=shortlist,
                    selected_slot=selected_slot,
                    selected_index=selected_index,
                    policy=("behavior" if mode == "collect" else policy),
                    decision_number=decision_number,
                    fisher_scores=fisher_scores,
                    behavior_propensity=behavior_propensity,
                )
                decision_number += 1

        step = {
            "pose": int(pose),
            "action": list(map(int, action)),
            "tail": bool(is_tail),
            "centroids_mm": list(map(float, values)),
            "eval_only_labels": list(map(int, labels)),
            "path_m": np.asarray(motion.points).tolist(),
            "path_skill": motion.skill,
            "length_m": float(motion.length_m),
            "cumulative_m": float(distance),
            "changed_suffix": bool(changed),
            "evidence_gains": np.flatnonzero(counts > before).tolist(),
            "n_ready": int(ready.sum()),
            "tail_choice": (
                None
                if decision is None
                else _tail_choice_payload(decision, planner, pose)
            ),
        }
        suffix_steps.append(step)

    if np.all(ready):
        terminal_reason = "complete"
    elif terminal_reason == "cap" and len(suffix_steps) < int(cap):
        terminal_reason = "failure"

    mean, covariance = belief
    _, variance, ready, inliers = book.status(belief)
    wrong = [
        i
        for i, events in enumerate(inliers)
        if len(events) >= study.VERIFY
        and not all(event[1] for event in events[-study.VERIFY :])
    ]
    error = model.query @ (mean - truth)
    stop = bool(np.all(ready))
    strict_complete = bool(
        stop and not wrong and np.max(np.abs(error)) <= study.BAND
    )
    run = {
        "seed": int(episode_seed),
        "episode_seed": int(episode_seed),
        "mode": mode,
        "policy": ("behavior" if mode == "collect" else policy),
        "collection_behavior": collection_behavior,
        "prefix_acquisitions": len(prefix),
        "initial_certified": initial,
        "certified": int(ready.sum()),
        "online_stop": stop,
        "complete": strict_complete,
        "strict_complete": strict_complete,
        "terminal_reason": terminal_reason,
        "incorrect_witnesses": wrong,
        "map_max_error_mm": float(np.max(np.abs(error))),
        "map_rmse_mm": float(np.sqrt(np.mean(error**2))),
        "max_band_mm": float(model.gamma * np.sqrt(np.max(variance))),
        "recheck_path_mm": float(1000.0 * distance),
        "actual_switch_count": int(sum(s["changed_suffix"] for s in suffix_steps)),
        "scheduled_decisions": int(decision_number),
        "executed_decisions": int(len(executed_decisions)),
        "truth_rejections": int(rejects),
        # Keep the frozen suffix convention in ``steps`` while also recording
        # the complete public main trace for audit/reporting consumers.
        "steps": suffix_steps,
        "prefix_steps": prefix,
        "full_steps": list(prefix) + suffix_steps,
    }
    return run, executed_decisions


def _transition_arrays(
    episodes: Iterable[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    cap: int,
) -> dict[str, np.ndarray]:
    """Build per-shard transition arrays with one cost accounting pass."""

    feature_rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_indices: list[int] = []
    episode_seeds: list[int] = []
    # Build all rows first so next_index can point to local shard rows.
    pending: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for run, decisions in episodes:
        pending.append((run, decisions))
        for item in decisions:
            decision = item["decision"]
            feature_rows.append(np.asarray(decision.shortlist.features, dtype=np.float32))
            masks.append(np.asarray(decision.shortlist.mask, dtype=bool))
            actions.append(int(decision.selected_slot))
            rewards.append(0.0)
            next_indices.append(-1)
            episode_seeds.append(int(run["episode_seed"]))

    offset = 0
    for run, decisions in pending:
        n = len(decisions)
        suffix_steps = run["steps"]
        for local, item in enumerate(decisions):
            row = offset + local
            step_index = int(item["step_index"])
            next_step = (
                int(decisions[local + 1]["step_index"])
                if local + 1 < n
                else len(suffix_steps)
            )
            # The first row owns the common-head prefix up to its first
            # executed tail; subsequent rows start at their tail.  Thus every
            # suffix acquisition/path is charged exactly once, including
            # failure/cap terminals, while transitions still represent
            # tail-to-next-tail segments.
            start_step = 0 if local == 0 else step_index
            end_step = next_step
            segment = suffix_steps[start_step:end_step]
            n_acquisitions = len(segment)
            meters = sum(float(s.get("length_m", 0.0)) for s in segment)
            reward = -0.002 * n_acquisitions - 0.05 * meters
            if local == n - 1 and bool(run["strict_complete"]):
                reward += 1.0
            rewards[row] = float(reward)
            next_indices[row] = offset + local + 1 if local + 1 < n else -1
        offset += n

    if feature_rows:
        features = np.stack(feature_rows).astype(np.float32, copy=False)
        mask = np.stack(masks).astype(bool, copy=False)
    else:
        features = np.zeros((0, K, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((0, K), dtype=bool)
    return {
        "features": features,
        "mask": mask,
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_index": np.asarray(next_indices, dtype=np.int64),
        "episode_seed": np.asarray(episode_seeds, dtype=np.int64),
    }


def _source_provenance() -> dict[str, Any]:
    geometry = study.DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz"
    files = [
        HERE / "rl_rollouts.py",
        FROZEN / "refine_rechecks.py",
        FROZEN / "detection_branches.py",
        FROZEN / "evidence.py",
    ]
    return {
        "geometry": str(geometry),
        "geometry_sha256": _sha(geometry),
        "source_sha256": {str(path.name): _sha(path) for path in files},
        "feature_names": list(FEATURE_NAMES),
        "shortlist_k": K,
    }


def _load_q(policy: str, checkpoint: str | Path | None):
    if policy != "rl":
        return None
    if checkpoint is None:
        raise ValueError("--policy rl requires --checkpoint (or a checkpoint path in --policy)")
    # The sibling fit_q.py is owned by the training worker and is intentionally
    # imported lazily so collect/rule/immediate smoke runs need no checkpoint.
    from fit_q import FrozenQ  # type: ignore

    return FrozenQ(checkpoint)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("collect", "evaluate"), default="collect")
    parser.add_argument("--seed", type=int, default=2411202619)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--policy", default="rule", help="rule, immediate, rl, or a rl checkpoint path")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--eval-arms",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="evaluation policies to write (default: rule immediate rl)",
    )
    parser.add_argument("--cap", type=int, default=CAP_DEFAULT)
    parser.add_argument("--n-predictive", "--n_predictive", dest="n_predictive", type=int, default=N_PREDICTIVE_DEFAULT)
    parser.add_argument("--n-first", "--n_first", dest="n_first", type=int, default=N_FIRST_DEFAULT)
    parser.add_argument("--switch-margin", "--switch_margin", dest="switch_margin", type=float, default=SWITCH_MARGIN_DEFAULT)
    parser.add_argument(
        "--behavior",
        choices=("uniform", "fisher"),
        default="uniform",
        help="collection behavior (evaluation policies are unchanged)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.cap <= 0:
        raise ValueError("--cap must be positive")
    if args.n < 0:
        raise ValueError("--n must be nonnegative")
    if args.mode == "collect":
        policy_for_config = "behavior"
        q_model = None
        policies = ["collect"]
    else:
        checkpoint = args.checkpoint
        policy_arg = str(args.policy)
        if policy_arg not in POLICIES:
            checkpoint = checkpoint or Path(policy_arg)
            policy_arg = "rl"
        # --eval-arms is the explicit multi-policy control requested by the
        # holdout runner.  A non-default --policy path still means rl only.
        policies = list(args.eval_arms)
        if policy_arg == "rl" and checkpoint is not None and str(args.policy) not in POLICIES:
            policies = ["rl"]
        policy_for_config = policy_arg
        q_model = None
        # Load one checkpoint once; rule/immediate never touch fit_q.
        if "rl" in policies:
            q_model = _load_q("rl", checkpoint)

    scan, support, model, planner = _make_model_planner(
        n_predictive=args.n_predictive,
        n_first=args.n_first,
        switch_margin=args.switch_margin,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = _source_provenance()
    config = {
        "out": str(args.out),
        "mode": args.mode,
        "seed": int(args.seed),
        "start": int(args.start),
        "n": int(args.n),
        "cap": int(args.cap),
        "policy": policy_for_config,
        "collection_behavior": args.behavior,
        "behavior": args.behavior,
        "eval_arms": policies if args.mode == "evaluate" else [],
        "n_predictive": int(args.n_predictive),
        "n_first": int(args.n_first),
        "switch_margin": float(args.switch_margin),
        "no_distance_cap": True,
        "behavior_anchor_probability": BEHAVIOR_ANCHOR_PROBABILITY,
        "behavior_uniform_probability": BEHAVIOR_ALTERNATIVE_PROBABILITY,
        "provenance": provenance,
    }
    (args.out / "CONFIG.json").write_text(json.dumps(_jsonable(config), indent=2) + "\n")

    all_transition_episodes: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    with (args.out / "RUNS.jsonl").open("w") as output:
        for index in range(args.start, args.start + args.n):
            episode_seed = int(
                np.random.SeedSequence([int(args.seed), int(index)]).generate_state(1)[0]
            )
            prepared_episode = prepare_episode(scan, support, episode_seed)
            for policy in policies:
                run, decisions = run_episode(
                    planner,
                    scan,
                    support,
                    model,
                    episode_seed=episode_seed,
                    mode=args.mode,
                    policy=("collect" if args.mode == "collect" else policy),
                    q_model=q_model if policy == "rl" else None,
                    cap=args.cap,
                    collection_behavior=args.behavior,
                    prepared_episode=prepared_episode,
                )
                run["provenance"] = provenance
                output.write(
                    json.dumps(_jsonable(run), separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
                output.flush()
                all_transition_episodes.append((run, decisions))

    if args.mode == "collect":
        arrays = _transition_arrays(all_transition_episodes, cap=args.cap)
        np.savez(args.out / "TRANSITIONS.npz", **arrays)


if __name__ == "__main__":
    main()

