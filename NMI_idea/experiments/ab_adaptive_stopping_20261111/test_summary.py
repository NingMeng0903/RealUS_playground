"""Focused tests for adaptive-stopping aggregation.

These are deliberately small subject-cluster tests.  They validate the
statistics contract, not the runner's mirrored execution implementation.
Run with ``python test_summary.py`` or ``pytest -q test_summary.py``.
"""

from __future__ import annotations

import unittest

from summarize_adaptive import (
    _arm_summary,
    _completeness,
    normalize_rows,
    paired_subject_bootstrap,
)


def _paired_rows():
    rows = []
    for object_id in ("obj0", "obj1"):
        for mirror in ("constrained_0", "constrained_1"):
            rows.extend([
                {"object_id": object_id, "mirror": mirror, "arm": "B_H1",
                 "correct_complete": object_id == "obj0", "false_complete": False,
                 "committed": True},
                {"object_id": object_id, "mirror": mirror, "arm": "A_task",
                 "correct_complete": object_id == "obj1", "false_complete": False,
                 "committed": True},
                {"object_id": object_id, "mirror": mirror, "arm": "complete_pair",
                 "correct_complete": False, "false_complete": False,
                 "committed": False},
            ])
    return normalize_rows(rows)


class SummaryContractTests(unittest.TestCase):
    def test_mirrors_are_execution_records_but_bootstrap_clusters_are_objects(self):
        rows = _paired_rows()
        result = paired_subject_bootstrap(rows, "B_H1", "A_task", seed=721505, samples=500)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["n_mirror_records_B"], 4)
        self.assertEqual(result["n_mirror_records_comparator"], 4)
        # Each object's two mirrors are averaged before the paired contrast.
        self.assertAlmostEqual(result["estimate"], 0.0)

        arm = _arm_summary([row for row in rows if row["arm"] == "B_H1"])
        self.assertEqual(arm["mirror_records"], 4)
        self.assertEqual(arm["object_clusters"], 2)
        self.assertEqual(arm["all_subject_denominator"], 2)

    def test_missing_primary_layout_is_explicitly_incomplete(self):
        rows = [row for row in _paired_rows()
                if not (row["object_id"] == "obj1" and row["arm"] == "complete_pair" and
                        row["mirror"] == "constrained_1")]
        contract = _completeness(
            rows,
            {"stage": "test", "subject_ids": ["obj0", "obj1"],
             "mirror_ids": ["constrained_0", "constrained_1"]},
            stage="test", expected_subjects=2, expected_mirrors=2,
            b_arm="B_H1", a_arm="A_task", pair_arm="complete_pair",
        )
        self.assertFalse(contract["complete"])
        self.assertTrue(any(item["object_id"] == "obj1" and
                            item["arm"] == "complete_pair" and
                            item["mirror"] == "constrained_1"
                            for item in contract["missing_records"]))

    def test_false_stop_after_qualification_is_not_success(self):
        row = normalize_rows([{
            "object_id": "obj0", "mirror": "constrained_0", "arm": "B_H1",
            "ever_qualified": True, "correct_complete": False,
            "false_complete": True, "committed": True,
        }])[0]
        self.assertTrue(row["everqualified"])
        self.assertTrue(row["false_complete"])
        self.assertFalse(row["correct_complete"])
        self.assertEqual(row["outcome"], "false_complete")

    def test_ci_direction_is_b_minus_comparator(self):
        rows = []
        for object_id in range(4):
            rows.extend([
                {"object_id": str(object_id), "mirror": "constrained_0",
                 "arm": "B_H1", "correct_complete": True},
                {"object_id": str(object_id), "mirror": "constrained_0",
                 "arm": "A_task", "correct_complete": False},
            ])
        rows = normalize_rows(rows)
        forward = paired_subject_bootstrap(rows, "B_H1", "A_task", seed=721505, samples=500)
        reverse = paired_subject_bootstrap(rows, "A_task", "B_H1", seed=721505, samples=500)
        self.assertGreater(forward["estimate"], 0.0)
        self.assertLess(reverse["estimate"], 0.0)
        self.assertGreater(forward["ci_lower"], 0.0)
        self.assertLess(reverse["ci_upper"], 0.0)


if __name__ == "__main__":
    unittest.main()

