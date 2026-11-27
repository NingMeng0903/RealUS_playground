"""Finite-belief scores for proposing an observation before an executable continuation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionValue:
    precommitted: float
    information: float
    perfect_information: float

    @property
    def score(self):
        return self.precommitted + self.information


def decision_value(prior, posterior_samples, payoff):
    """Columns are common feasible continuations, with payoffs conditional on each world."""
    precommitted = prior @ payoff
    fixed = int(np.argmax(precommitted))
    updated = posterior_samples @ payoff
    gain = float(np.mean(np.max(updated, axis=1) - updated[:, fixed]))
    ceiling = float(prior @ np.max(payoff, axis=1) - precommitted[fixed])
    return DecisionValue(float(precommitted[fixed]), gain, max(0., ceiling))


def acquisition_payoff(first_view, follow_up):
    """Approximate acquisition payoff from nominal view coverage, used for screening."""
    return first_view[:, None] + (1. - first_view[:, None]) * follow_up
