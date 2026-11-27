"""Contract tests for the bounded development orchestration driver.

These tests do not launch run_adaptive.py or a native controller.  They check
the fixed matrix, disjoint paths, source guards, dry-run behavior, and
object-ID completeness used before any global freeze.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import study_driver as driver


def _source_manifest():
    return {"files": [{"path": "/runner.py", "sha256": "abc"}], "missing": []}


def _write_result(spec: driver.JobSpec, case_id: int, *, correct: bool, root: Path) -> None:
    path = (spec.job_root / "dev" / f"fixture_{spec.fixture_index:02d}_{spec.mode}"
            / spec.arm / f"case_{case_id:04d}" / "RESULT.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id, "fixture_index": spec.fixture_index,
        "mode": spec.mode, "arm": spec.arm,
        "correct_complete": correct, "false_complete": False,
        "probe_count": 2, "terminal_count": 1, "protocol": {"risk_gate": .90},
        "decisions": [], "source_manifest": _source_manifest(),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class DriverTests(unittest.TestCase):
    def test_fixed_matrix_has_disjoint_constrained_fixture_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            specs = driver.build_job_specs(temp, count=8)
            self.assertEqual(len(specs), 14)
            self.assertEqual(len({spec.job_root for spec in specs}), 14)
            self.assertEqual({spec.arm for spec in specs}, set(driver.ARMS))
            self.assertEqual({spec.fixture_index for spec in specs}, {0, 1})
            self.assertEqual({spec.mode for spec in specs}, {"constrained"})
            self.assertEqual({spec.dwell for spec in specs if spec.arm != "pair_fixed"}, {.30})
            self.assertEqual({spec.dwell for spec in specs if spec.arm == "pair_fixed"}, set(driver.DWELLS))
            command = driver.build_command(specs[0], python_executable=sys.executable)
            self.assertIn("--resume", command)
            self.assertIn("--max-probe-frames", command)
            self.assertEqual(command[command.index("--max-probe-frames") + 1], "8")

    def test_config_requires_exact_two_mirror_object_id_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = driver.build_job_specs(temp, count=2, arms=("A_task",),
                                          dwells=(.15,))[0]
            for fixture_index in (0, 1):
                mirror_spec = driver.JobSpec(spec.arm, spec.dwell, fixture_index,
                                             spec.mode, spec.count, spec.output_root)
                for case_id in (0, 1):
                    _write_result(mirror_spec, case_id, correct=(case_id == 0), root=Path(temp))
            baseline = _source_manifest()
            complete = driver.aggregate_config(spec, expected_count=2, baseline=baseline)
            self.assertTrue(complete["complete"])
            self.assertAlmostEqual(complete["correct_rate"], .5)
            self.assertEqual(complete["committed_n"], 4)
            self.assertAlmostEqual(complete["committed_fail_rate"], .5)
            self.assertEqual(complete["risk_status"], "risk_failed")
            self.assertIsNotNone(complete["committed_fail_rate_ci95_descriptive"])
            missing_path = (Path(temp) / spec.config_id / "fixture_01_constrained" /
                            "dev" / "fixture_01_constrained" / spec.arm /
                            "case_0001" / "RESULT.json")
            missing_path.unlink()
            incomplete = driver.aggregate_config(spec, expected_count=2, baseline=baseline)
            self.assertFalse(incomplete["complete"])
            self.assertTrue(incomplete["mirror_audit"]["1"]["missing_ids"])

    def test_source_hash_guard_compares_only_hash_maps(self):
        left = {"files": [{"path": "/a", "sha256": "x"}], "created_utc": "one", "missing": []}
        right = {"files": [{"path": "/a", "sha256": "x"}], "created_utc": "two", "missing": []}
        changed = {"files": [{"path": "/a", "sha256": "y"}], "missing": []}
        self.assertTrue(driver.source_manifests_equal(left, right))
        self.assertFalse(driver.source_manifests_equal(left, changed))

    def test_dry_run_writes_plan_without_starting_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = driver.run_jobs(temp, count=8, max_workers=2, dry_run=True,
                                      python_executable=sys.executable)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(len(summary["jobs"]), 14)
            self.assertTrue(all(item["status"] == "planned" for item in summary["jobs"]))
            self.assertTrue((Path(temp) / "DRIVER_SOURCE_START.json").exists())
            self.assertTrue((Path(temp) / "DRIVER_MANIFEST.json").exists())
            self.assertFalse(any((Path(temp) / "driver_logs").rglob("stdout.log"))
                             if (Path(temp) / "driver_logs").exists() else False)

    def test_selection_key_prefers_h1_on_equal_b_scores(self):
        h1 = {"arm": "B_H1", "dwell": .60, "correct_rate": .8, "mean_probe_count": 3}
        h2 = {"arm": "B_H2", "dwell": .15, "correct_rate": .8, "mean_probe_count": 3}
        self.assertLess(driver._selection_key(h1, family="B"), driver._selection_key(h2, family="B"))

    def test_pilot_never_reports_freeze_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            driver.run_jobs(temp, count=8, dry_run=True, python_executable=sys.executable)
            payload = driver.select_global_freeze(temp, count=8)
            self.assertFalse(payload["freeze_ready"])
            self.assertEqual(payload["reason"], "pilot_count_is_not_freeze_eligible")


if __name__ == "__main__":
    unittest.main()

