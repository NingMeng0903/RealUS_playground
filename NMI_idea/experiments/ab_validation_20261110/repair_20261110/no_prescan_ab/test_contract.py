"""Contract tests for the frozen no-prescan synthetic pilot."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import benchmark as b


class TestNoPrescanContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = b.PathBank()

    def test_public_missing_and_posterior(self):
        obs = b.Observation(False)
        self.assertIsNone(obs.public_dict()["image_u_m"])
        for a in range(len(b.ACTIONS)):
            p = b.posterior(b.PRIOR, b.pose_for_action(a), obs)
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(float(p.sum()), 1.0, places=12)

    def test_likelihood_quadrature_mass(self):
        for h in range(len(b.HYP)):
            for a in range(len(b.ACTIONS)):
                self.assertAlmostEqual(sum(w for w, _, _ in b.MODEL_NODES[h, a]), 1.0, places=12)

    def test_actual_dmp_peak_speed_contract(self):
        self.assertLessEqual(self.bank.max_nominal_speed, .0200000001)
        self.assertLess(self.bank.max_endpoint_error, .002)
        for route in self.bank.paths.values():
            yaw_rate = float(np.abs(np.diff(route["path"][:, 2])).max() / (route["duration"] / 20))
            self.assertLessEqual(yaw_rate, .5000000001)

    def test_information_permissions(self):
        causal = b.check_causal_noninterference(self.bank)
        self.assertTrue(all(v["pass_equal"] for v in causal.values()))
        prior = b.check_prior_sensitivity(self.bank)
        self.assertTrue(prior["actions_change"])
        self.assertGreater(prior["score_difference"], 1e-6)

    def test_independent_qualifier_and_paired_result(self):
        data = json.loads((Path(__file__).parent / "results.json").read_text())
        self.assertIn("not_la_contract", data)
        self.assertLess(data["policy_reference"]["absolute_value_error"], 1e-9)
        for cohort, n in (("test", 256), ("shift", 128)):
            aa = {r["world_id"]:r for r in data["rows"] if r["cohort"] == cohort and r["method"] == "A_one_step_greedy"}
            bb = {r["world_id"]:r for r in data["rows"] if r["cohort"] == cohort and r["method"] == "B_two_step_contingent"}
            self.assertEqual(len(aa), n)
            self.assertEqual(len(bb), n)
            self.assertTrue(any(r["false_accept"] for r in bb.values()))
            for i in aa:
                self.assertEqual([t["action_id"] for t in aa[i]["trace"]],
                                 [t["action_id"] for t in bb[i]["trace"]])
                self.assertEqual(aa[i]["success"], bb[i]["success"])


if __name__ == "__main__":
    unittest.main()
