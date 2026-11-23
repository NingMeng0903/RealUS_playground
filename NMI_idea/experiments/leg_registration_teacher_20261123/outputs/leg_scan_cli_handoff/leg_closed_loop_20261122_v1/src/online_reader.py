"""Incremental public-evidence reader for the online leg scanner.

Every anonymous contour is retained.  The reader uses only public atlas
geometry, echo quality, and the posterior emitted by the public observation
model.  Evaluator labels in frame tuples are preserved by the inherited
archive but are never read here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import motion


PUBLIC_HISTORY_NAMES = (
    "history_best_evidence", "history_best_posterior", "history_best_quality",
    "history_support", "history_conflict", "history_unknown",
    "history_candidate_density", "history_uncovered_fraction",
    "history_redundancy", "history_hypothesis_entropy",
    "history_cross_window_support",
)


def _angular_distance(first, second, sectors):
    return np.abs((first - second + sectors / 2.) % sectors - sectors / 2.)


class PublicHistoryReader(motion.Reader):
    """Rank all acquired candidates with a continuous, incremental score."""

    threshold = .55
    _support_radius = 3.05
    _support_angle = .75

    def __init__(self, *args):
        super().__init__(*args)
        self._geometric = []
        self._candidates = []
        self._region_candidates = defaultdict(list)
        self._station_index = defaultdict(list)
        self._supports = []
        self._cross_supports = []
        self._repetition = []
        self._choices = {}
        self._rankings = {}
        self._ranked_ids = {}
        self._summaries = {}
        self._explanations = {}

    # Public geometry -------------------------------------------------
    def _new_geometry(self, frame):
        station, sector, observed, _labels, quality, _region, posterior, unknown, _tick = frame
        observed = np.asarray(observed, dtype=float)
        posterior = np.asarray(posterior, dtype=float).reshape(-1)
        unknown = np.asarray(unknown, dtype=float).reshape(-1)
        if not len(observed):
            return np.empty((0, 2)), np.empty(0)
        world = np.asarray([self.absolute(station, sector, xy) for xy in observed], dtype=float)
        surface = self.model.surface
        location = int(np.clip(np.floor(float(station) + .5), 0, len(surface.y) - 1))
        prior = np.asarray(surface.priors[:, location, :5], dtype=float)
        valid = np.isfinite(prior).all(axis=-1)
        distance = np.linalg.norm(world[:, None, None] - prior[None], axis=-1)
        distance = np.where(valid[None], distance, 120.).min(axis=-1)
        target = distance[:, 0]
        rival = distance[:, 1:].min(axis=1) if distance.shape[1] > 1 else target + 20.
        absolute = np.exp(-.5 * (target / 17.) ** 2)
        margin = 1. / (1. + np.exp(np.clip((target - rival) / 5., -30., 30.)))
        relation = np.zeros(len(observed), dtype=float)
        if len(observed) > 1:
            # Anonymous contours that preserve a plausible local spacing give
            # a public relation cue.  The atlas distance terms above already
            # provide identity-independent target and rival scores.
            pair_distance = np.linalg.norm(world[:, None] - world[None, :], axis=-1)
            affinity = np.exp(-.5 * (pair_distance / 7.5) ** 2)
            affinity[np.diag_indices_from(affinity)] = 0.
            relation = np.clip(affinity.max(axis=1), 0., 1.)
        relative = .5 + .5 * np.tanh(2. * (relation - relation.mean()))
        q = float(np.clip(quality, 0., 1.))
        posterior = np.resize(np.clip(posterior, 0., 1.), len(observed))
        unknown = np.resize(np.clip(unknown, 0., 1.), len(observed))
        # Calibrate the local cue around the public posterior.  Atlas
        # proximity and margin remain useful soft geometry terms, while the
        # close-contour relation is deliberately tiny because merged ghosts
        # can otherwise make every nearby contour look mutually supported.
        score = (.62 * posterior + .08 * absolute + .06 * margin
                 + .02 * relation + .04 * q + .02 * (1. - unknown))
        return world, np.clip(score, 0., 1.)

    def _pair_support(self, new_ids, old_ids):
        """Update support for a new frame against nearby earlier frames."""
        if not new_ids or not old_ids:
            return
        new = [self._candidates[i] for i in new_ids]
        old = [self._candidates[i] for i in old_ids]
        nw = np.asarray([item["world"] for item in new])
        ow = np.asarray([item["world"] for item in old])
        ns = np.asarray([item["station"] for item in new])
        os = np.asarray([item["station"] for item in old])
        na = np.asarray([item["sector"] for item in new])
        oa = np.asarray([item["sector"] for item in old])
        ds = np.abs(os[:, None] - ns[None, :])
        da = _angular_distance(oa[:, None], na[None, :], motion.SECTORS)
        correlation = np.exp(-.5 * (ds / .55) ** 2 - .5 * (da / .75) ** 2)
        for row, cid in enumerate(old_ids):
            self._repetition[cid] = max(self._repetition[cid], float(correlation[row].max(initial=0.)))
        for col, cid in enumerate(new_ids):
            self._repetition[cid] = max(self._repetition[cid], float(correlation[:, col].max(initial=0.)))
        valid = ((ds >= .65) & (ds <= self._support_radius))
        valid |= (ds <= 2.05) & (da >= self._support_angle)
        if not np.any(valid):
            return
        distance = np.linalg.norm(ow[:, None] - nw[None], axis=-1)
        old_q = np.asarray([item["quality"] for item in old])[:, None]
        new_q = np.asarray([item["quality"] for item in new])[None, :]
        old_p = np.asarray([item["posterior"] for item in old])[:, None]
        new_p = np.asarray([item["posterior"] for item in new])[None, :]
        quality = .65 + .35 * np.minimum(old_q, new_q)
        posterior = .55 + .45 * np.sqrt(np.minimum(old_p, new_p))
        repeated = 1. - .72 * np.exp(-.5 * (ds / .55) ** 2 - .5 * (da / .75) ** 2)
        value = np.exp(-.5 * (distance / (4. + ds)) ** 2) * quality * posterior * repeated
        value = np.where(valid, value, 0.)
        old_max = value.max(axis=1, initial=0.)
        new_max = value.max(axis=0, initial=0.)
        for row, cid in enumerate(old_ids):
            self._supports[cid] = max(self._supports[cid], float(old_max[row]))
            mask = np.asarray([item["region"] != old[row]["region"] for item in new])
            self._cross_supports[cid] = max(self._cross_supports[cid], float(np.where(mask, value[row], 0.).max(initial=0.)))
        for col, cid in enumerate(new_ids):
            self._supports[cid] = max(self._supports[cid], float(new_max[col]))
            mask = np.asarray([item["region"] != new[col]["region"] for item in old])
            self._cross_supports[cid] = max(self._cross_supports[cid], float(np.where(mask, value[:, col], 0.).max(initial=0.)))

    def _append_frame(self, frame_id, frame):
        world, local_scores = self._new_geometry(frame)
        self._geometric.append((world, local_scores))
        if not len(world):
            return
        station, sector, _z, _labels, quality, region, posterior, unknown, tick = frame
        posterior = np.resize(np.clip(np.asarray(posterior, dtype=float).reshape(-1), 0., 1.), len(world))
        unknown = np.resize(np.clip(np.asarray(unknown, dtype=float).reshape(-1), 0., 1.), len(world))
        mixed_values = self.mixed_posterior[-1] if self.mixed_posterior else unknown
        mixed_values = np.resize(np.clip(np.asarray(mixed_values, dtype=float).reshape(-1), 0., 1.), len(world))
        quality = float(np.clip(quality, 0., 1.))
        bucket = int(np.floor(float(station)))
        old_ids = []
        for key in range(bucket - 4, bucket + 5):
            old_ids.extend(self._station_index.get(key, ()))
        new_ids = []
        for slot, (point, local) in enumerate(zip(world, local_scores)):
            cid = len(self._candidates)
            self._candidates.append({
                "frame": int(frame_id), "slot": int(slot), "region": int(region),
                "station": float(station), "sector": float(sector),
                "world": np.asarray(point, dtype=float), "local": float(local),
                "posterior": float(posterior[slot]), "unknown": float(unknown[slot]),
                "mixed": float(mixed_values[slot]),
                "quality": quality, "tick": None if tick is None else int(tick),
            })
            self._supports.append(0.)
            self._cross_supports.append(0.)
            self._repetition.append(0.)
            self._region_candidates[int(region)].append(cid)
            self._station_index[bucket].append(cid)
            new_ids.append(cid)
        self._pair_support(new_ids, old_ids)

    def push(self, s, a, z, labels, q, region, tick=None):
        # ``labels`` is intentionally passed through but never inspected.
        super().push(s, a, z, labels, q, region, tick)
        self._append_frame(len(self.frames) - 1, self.frames[-1])
        self._choices.clear()
        self._rankings.clear()
        self._ranked_ids.clear()
        self._summaries.clear()
        self._explanations.clear()

    # Ranking ---------------------------------------------------------
    def _evidence(self, cid):
        item = self._candidates[cid]
        value = item["local"] + .22 * self._supports[cid]
        ambiguity = .78 + .22 * (1. - item["unknown"])
        ambiguity *= .92 + .08 * (1. - item["mixed"])
        return float(np.clip(value * ambiguity, 0., 1.))

    def _ranking(self, region):
        ids = list(self._region_candidates.get(int(region), ()))
        if not ids:
            return np.empty(0, int), np.empty(0, int), np.empty(0, float)
        evidence = np.asarray([self._evidence(cid) for cid in ids])
        order = np.argsort(-evidence, kind="stable")
        return (np.asarray([self._candidates[ids[i]]["frame"] for i in order], dtype=int),
                np.asarray([self._candidates[ids[i]]["slot"] for i in order], dtype=int),
                evidence[order])

    def _summary(self, ids, evidence):
        if not ids:
            summary = np.zeros(len(PUBLIC_HISTORY_NAMES), dtype=np.float32)
            summary[7] = 1.
            return summary
        order = np.argsort(-evidence, kind="stable")
        ranked = [ids[int(i)] for i in order]
        best = float(evidence[order[0]])
        # Nearby echoes on one track form a single hypothesis cluster.  Only
        # a runner-up at a meaningfully different world point contributes to
        # the conflict signal.
        best_world = self._candidates[ranked[0]]["world"]
        runner = next((index for index in order[1:]
                       if np.linalg.norm(self._candidates[ids[int(index)]]["world"] - best_world) >= 5.), None)
        second = float(evidence[runner]) if runner is not None else 0.
        conflict = float(np.clip(1. - (best - second) / .35, 0., 1.)) if runner is not None else 0.
        shifted = np.clip(evidence - np.max(evidence), -40., 0.)
        probability = np.exp(shifted)
        probability /= max(float(probability.sum()), 1.e-12)
        entropy = float(-(probability * np.log(np.maximum(probability, 1.e-12))).sum()
                        / np.log(max(len(probability), 2)))
        quality = float(np.average([self._candidates[i]["quality"] for i in ranked], weights=probability))
        posterior = float(np.average([self._candidates[i]["posterior"] for i in ranked], weights=probability))
        unknown = float(np.average([self._candidates[i]["unknown"] for i in ranked], weights=probability))
        support = max(self._supports[i] for i in ranked)
        cross = max(self._cross_supports[i] for i in ranked)
        redundancy = float(np.average([self._repetition[i] for i in ranked], weights=probability))
        unique_views = {(round(self._candidates[i]["station"]),
                         round(self._candidates[i]["sector"])) for i in ranked}
        distinct_count = len(unique_views)
        return np.asarray((best, posterior, quality, support, conflict, unknown,
                           min(1., distinct_count / 8.), np.exp(-distinct_count / 3.),
                           redundancy, entropy, cross), dtype=np.float32)

    def details(self, region):
        region = int(region)
        if region in self._choices:
            return self._choices[region]
        ids = list(self._region_candidates.get(region, ()))
        ranking = self._ranking(region)
        self._rankings[region] = ranking
        if not ids:
            self._choices[region] = (None, 0.)
            self._ranked_ids[region] = []
            return self._choices[region]
        frame_ids, slots, evidence = ranking
        pair_to_id = {(self._candidates[i]["frame"], self._candidates[i]["slot"]): i for i in ids}
        ranked_ids = [pair_to_id[(int(frame), int(slot))] for frame, slot in zip(frame_ids, slots)]
        self._ranked_ids[region] = ranked_ids
        support = float(self._supports[ranked_ids[0]])
        top = (int(frame_ids[0]), int(slots[0]), float(evidence[0]))
        self._choices[region] = (top if evidence[0] >= self.threshold else None, support)
        return self._choices[region]

    def ranked(self, region):
        self.details(int(region))
        return self._rankings[int(region)]

    def public_summary(self, region):
        region = int(region)
        self.details(region)
        if region not in self._summaries:
            _frame_ids, _slots, evidence = self._rankings[region]
            self._summaries[region] = self._summary(self._ranked_ids[region], evidence)
        return self._summaries[int(region)].copy()

    def explanation(self, region):
        region = int(region)
        self.details(region)
        if region not in self._explanations:
            frame_ids, slots, evidence = self._rankings[region]
            ranked_ids = self._ranked_ids[region]
            summary = self.public_summary(region)
            self._explanations[region] = {
                "candidate_count": len(ranked_ids), "uncovered_fraction": float(summary[7]),
                "conflict": float(summary[4]), "unknown_fraction": float(summary[5]),
                "mixed_fraction": float(np.average([self._candidates[i]["mixed"] for i in ranked_ids])) if ranked_ids else 1.,
                "hypotheses": [
                    {"frame": int(frame), "slot": int(slot), "evidence": float(score),
                     "posterior": float(self._candidates[cid]["posterior"]),
                     "unknown": float(self._candidates[cid]["unknown"]),
                     "mixed": float(self._candidates[cid]["mixed"]),
                     "support": float(self._supports[cid])}
                    for frame, slot, score, cid in zip(frame_ids[:8], slots[:8], evidence[:8], ranked_ids[:8])
                ],
            }
        row = self._explanations[region]
        return {
            "candidate_count": int(row["candidate_count"]),
            "uncovered_fraction": float(row["uncovered_fraction"]),
            "conflict": float(row["conflict"]),
            "unknown_fraction": float(row["unknown_fraction"]),
            "mixed_fraction": float(row.get("mixed_fraction", 0.)),
            "hypotheses": [dict(item) for item in row["hypotheses"]],
        }

    def options(self, region):
        """Compatibility view of every candidate, without a posterior gate."""
        result = []
        for cid in self._region_candidates.get(int(region), ()):
            item = self._candidates[cid]
            result.append((item["frame"], item["slot"], item["station"], item["sector"],
                           item["world"].copy(), item["posterior"], item["quality"]))
        return result

    def select(self, region, threshold=None):
        """Select a deliverable candidate with an explicit output threshold."""
        region = int(region)
        choice, _support = self.details(region)
        threshold = self.threshold if threshold is None else float(threshold)
        if choice is not None and float(choice[2]) >= threshold:
            return choice
        frame_ids, slots, evidence = self._rankings.get(
            region, (np.empty(0, int), np.empty(0, int), np.empty(0, float)))
        if len(evidence) and float(evidence[0]) >= threshold:
            return int(frame_ids[0]), int(slots[0]), float(evidence[0])
        return None

    def fork(self):
        """Copy a reader for a branch without replaying old frames."""
        reader = type(self)(self.model, self.roles, self.history)
        reader.w = self.w.copy()
        reader.state = self.state
        reader.frames = self.frames.copy()
        reader.mixed_posterior = self.mixed_posterior.copy()
        reader._geometric = [(world, score) for world, score in self._geometric]
        reader._candidates = [item.copy() for item in self._candidates]
        reader._region_candidates = defaultdict(list, {k: list(v) for k, v in self._region_candidates.items()})
        reader._station_index = defaultdict(list, {k: list(v) for k, v in self._station_index.items()})
        reader._supports = list(self._supports)
        reader._cross_supports = list(self._cross_supports)
        reader._repetition = list(self._repetition)
        reader._choices = dict(self._choices)
        reader._rankings = {k: tuple(value.copy() for value in row) for k, row in self._rankings.items()}
        reader._ranked_ids = {k: list(value) for k, value in self._ranked_ids.items()}
        reader._summaries = {k: value.copy() for k, value in self._summaries.items()}
        reader._explanations = {k: {
            "candidate_count": row["candidate_count"], "uncovered_fraction": row["uncovered_fraction"],
            "conflict": row["conflict"], "unknown_fraction": row["unknown_fraction"],
            "mixed_fraction": row.get("mixed_fraction", 0.),
            "hypotheses": [dict(item) for item in row["hypotheses"]],
        } for k, row in self._explanations.items()}
        return reader
