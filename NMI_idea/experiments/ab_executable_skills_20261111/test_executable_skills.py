import unittest

import numpy as np

from skill_value import (
    actual_statistics, accepted, observation_tables, path_statistics,
    plan, terminal_path, update,
)
from run_experiment import Episode, cases, packet_draws
from toy_world import Fixture, prior


def scalar_enumeration(path, fixture, belief):
    """Enumerate public packets without using the batched scorer."""
    if not path:
        return np.zeros(3)
    action, *remaining = path
    transition, emission, _, _, _ = observation_tables(action, fixture)
    predicted = belief @ transition
    totals = np.array([0., 0., 1.])
    for packet, probability in enumerate(predicted @ emission):
        if probability <= 0.:
            continue
        posterior, p_target = update(belief, fixture, action, packet)
        if accepted(action, packet, p_target):
            totals[:2] += probability * np.array([p_target, 1. - p_target])
        else:
            totals += probability * scalar_enumeration(remaining, fixture, posterior)
    return totals


class ExecutableSkillChecks(unittest.TestCase):
    def test_exact_path_matches_independent_public_packet_tree(self):
        fixture = Fixture("L", True)
        belief = prior()
        path = terminal_path("EL", "R", fixture)
        np.testing.assert_allclose(path_statistics(path, fixture, belief, belief),
                                   scalar_enumeration(path, fixture, belief), atol=1.e-12)

    def test_nominal_prediction_cannot_add_a_third_capture(self):
        fixture = Fixture("L", True)
        for start in ("C", "TL"):
            path = terminal_path(start, "L", fixture, already_captured_at_terminal=True)
            self.assertEqual(sum(a.dest == "TL" for a in path), 1)

    def test_shared_public_checkpoint_preserves_saved_cursor(self):
        case = cases()[7]
        uniforms = packet_draws(721619, 7, 3, 32)
        for arm in ("B_actual", "A_task", "A_fixed_goal", "B_nominal"):
            whole = Episode(case, arm, uniforms)
            split = Episode(case, arm, uniforms)
            whole.advance(32)
            for checkpoint in (1, 2, 8, 32):
                split.advance(checkpoint)
            self.assertEqual(whole.events, split.events)
            self.assertEqual(whole.programs, split.programs)
            self.assertEqual(whole.outcome, split.outcome)

    def test_actual_exit_score_and_no_lookahead_identity(self):
        belief = prior()
        for case in cases()[:8]:
            args = (case.start, case.fixture, tuple(belief), case.start[-1])
            self.assertEqual(plan("A_task", *args), plan("B_no_lookahead", *args))
            skill = plan("B_actual", *args)
            predicted = skill.statistics
            evaluated = actual_statistics(skill, case.fixture, belief)
            np.testing.assert_allclose(list(predicted.as_dict().values()),
                                       list(evaluated.as_dict().values()), atol=1.e-12)


if __name__ == "__main__":
    unittest.main()
