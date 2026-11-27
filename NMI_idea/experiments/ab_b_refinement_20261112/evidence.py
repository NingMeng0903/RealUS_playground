"""Evidence bookkeeping for the refinement recheck.

The runner owns acquisition and planning; this module owns only the public
evidence history and the one robust posterior update performed for each frame.
The implementation deliberately follows :class:`certification.Recheck`'s
existing association and residual contracts so that the legacy gate is a
drop-in replacement for the original history-based implementation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

import certification


class EvidenceBook:
    """Maintain association-qualified witness events for a recheck model.

    Parameters
    ----------
    model:
        A ``certification.Recheck`` instance (or a compatible object exposing
        ``n``, ``loc``, ``model_cache``, ``worst_queries`` and ``var_limit``).
    gate:
        ``"legacy"`` keeps the original association-only witness gate.
        ``"predictive"`` additionally requires every assigned centroid to be
        inside the nominal predictive band computed from the *pre-update*
        belief.  Posterior residual screening remains in :meth:`status`.
    predictive_z:
        Nominal predictive standard-deviation multiplier for the predictive
        gate.  The default is two.
    """

    _GATES = frozenset(("legacy", "predictive"))

    def __init__(self, model, gate: str = "legacy", predictive_z: float = 2.0):
        if gate not in self._GATES:
            raise ValueError("gate must be 'legacy' or 'predictive'")
        z = float(predictive_z)
        if not np.isfinite(z) or z < 0.0:
            raise ValueError("predictive_z must be finite and nonnegative")
        self.model = model
        self.gate = gate
        self.predictive_z = z
        # The original Recheck stores one event list per target.  Keep that
        # shape and expose ``events`` as a compatibility alias for callers
        # that use the shorter name.
        self.history = [[] for _ in range(int(model.n))]
        self.events = self.history

    def copy(self) -> "EvidenceBook":
        """Return a cheap independent history copy.

        Event terms contain immutable-by-contract model rows and scalar
        observations.  Their enclosing term lists and per-target event lists
        are copied so appending or editing a copied history cannot mutate the
        source history.
        """

        result = type(self).__new__(type(self))
        result.model = self.model
        result.gate = self.gate
        result.predictive_z = self.predictive_z
        result.history = [
            [
                (list(terms), correct)
                for terms, correct in target_events
            ]
            for target_events in self.history
        ]
        result.events = result.history
        return result

    @staticmethod
    def _values(values: Iterable[float]) -> list[float]:
        """Normalize observations without making labels part of decisions."""

        if values is None:
            return []
        # ``update_set`` accepts a sequence; converting once also makes an
        # ndarray's truth value harmless and keeps recorded observations
        # independent of a caller-owned mutable container.
        return [float(value) for value in values]

    @staticmethod
    def _label_value(labels, index: int, target_label: int):
        """Return an offline correctness marker, or ``None`` if unavailable."""

        if labels is None:
            return None
        try:
            observed_label = labels[index]
        except (IndexError, KeyError, TypeError):
            return None
        try:
            return bool(observed_label == target_label)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _predictive_agrees(terms, mean, covariance, z: float) -> bool:
        """Check all assigned centroids against a nominal pre-update model."""

        for h, base, value in terms:
            # Keep the nominal SD exactly as Recheck.status does.  In
            # particular, do not use the stress mixture's 4 mm component for
            # this witness gate.
            variance = certification.SD ** 2 + h @ covariance @ h
            if abs(value - base - h @ mean) > z * np.sqrt(variance):
                return False
        return True

    def _record_evidence(
        self,
        site: int,
        values: Sequence[float],
        assoc: np.ndarray,
        joint: Sequence[int],
        joint_probability: float,
        labels,
        model_terms,
        pre_mean: np.ndarray,
        pre_covariance: np.ndarray,
    ) -> None:
        """Apply the original association gate and append accepted events."""

        for target_index, target_label in self.model.loc[site]:
            if not values:
                continue

            # This block intentionally mirrors certification.Recheck.evidence
            # line-for-line in its association decisions.  Labels are used
            # only for an optional offline correctness marker.
            if target_label == 2:
                if len(joint) != 2 or min(joint) < 0 or joint_probability < certification.CONF:
                    continue
                terms = [
                    (h, base, values[index])
                    for (h, base), index in zip(model_terms, joint)
                ]
                correct = None
                if labels is not None:
                    try:
                        correct = tuple(labels[index] for index in joint) == (0, 1)
                    except (IndexError, KeyError, TypeError, ValueError):
                        correct = None
            else:
                if target_label >= len(assoc):
                    continue
                # Keep the original argmax and confidence check.  No truth
                # labels can affect which centroid is selected.
                j = int(np.argmax(assoc[target_label]))
                if assoc[target_label, j] < certification.CONF:
                    continue
                h, base = model_terms[target_label]
                terms = [(h, base, values[j])]
                correct = self._label_value(labels, j, target_label)

            if self.gate == "predictive" and not self._predictive_agrees(
                terms, pre_mean, pre_covariance, self.predictive_z
            ):
                continue
            self.history[target_index].append((terms, correct))

    def assimilate(
        self,
        belief,
        pose,
        action,
        values,
        labels=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Robustly assimilate one frame and record any witness it supports.

        The update is performed exactly once for every frame, including a
        frame whose association witness is rejected by the predictive gate.
        ``labels`` are evaluation-only metadata and may be omitted.
        """

        mean, covariance = belief
        pre_mean = np.asarray(mean, dtype=float).copy()
        pre_covariance = np.asarray(covariance, dtype=float).copy()
        observations = self._values(values)
        action_key = tuple(action)
        model_terms = self.model.model_cache[(pose, *action_key)]
        # Use the same robust mixture and aperture as certification.Recheck.
        new_mean, new_covariance, assoc, joint, probability = certification.update_set(
            mean,
            covariance,
            model_terms,
            observations,
            certification.WIDTH,
            noise_mixture=certification.STRESS.noise_mixture,
            false_rate=certification.STRESS.clutter_rate,
        )
        self._record_evidence(
            int(action_key[0]),
            observations,
            assoc,
            joint,
            probability,
            labels,
            model_terms,
            pre_mean,
            pre_covariance,
        )
        return new_mean, new_covariance

    def status(self, belief):
        """Return ``(counts, V, ready, inliers)`` in the legacy shapes."""

        mean, covariance = belief
        _, variance = self.model.worst_queries(covariance)
        # This is deliberately the existing posterior residual check.  The
        # predictive gate only filters events at ingestion; it never changes
        # this status-time check.
        inliers = [
            [
                (terms, correct)
                for terms, correct in target_events
                if all(
                    abs(value - base - h @ mean)
                    <= 2.0 * np.sqrt(certification.SD ** 2 + h @ covariance @ h)
                    for h, base, value in terms
                )
            ]
            for target_events in self.history
        ]
        counts = np.array([min(certification.VERIFY, len(target)) for target in inliers], dtype=float)
        ready = (counts == certification.VERIFY) & (variance <= self.model.var_limit)
        return counts, variance, ready, inliers


__all__ = ["EvidenceBook"]

