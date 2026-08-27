from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from perception.capture_flow import (
    _tail_error,
    capture_cmd,
    capture_env,
    is_capture_progress_line,
    try_start_smplx_capture,
)


class TestCaptureFlow(unittest.TestCase):
    def test_cmd_is_full_window8(self) -> None:
        repo = Path("/media/camp/EXT_DRIVE/RealUS_playground")
        cmd = capture_cmd(repo, run_name="20260827_000000")
        joined = " ".join(cmd)
        self.assertIn("run_smplx_capture.py", joined)
        self.assertIn("--write-debug-images", cmd)
        self.assertIn("--publish-genesis", cmd)
        self.assertIn("smplx_mesh", cmd)
        self.assertIn("20260827_000000", cmd)
        self.assertTrue(cmd[0].endswith("envs/genesis/bin/python"))

    def test_try_start_busy_and_cooldown(self) -> None:
        done = []

        def fake_run(**kwargs):
            time.sleep(0.05)
            result = mock.Mock()
            result.ok = True
            result.run_name = kwargs["run_name"]
            result.moment_dir = Path("/tmp") / kwargs["run_name"]
            result.log_path = result.moment_dir / "log"
            result.returncode = 0
            result.quality_rejection = None
            result.error = ""
            return result

        with mock.patch("perception.capture_flow.run_smplx_capture", side_effect=fake_run):
            first = try_start_smplx_capture(label="a", on_done=done.append)
            self.assertTrue(first.started)
            second = try_start_smplx_capture(label="b")
            self.assertFalse(second.started)
            self.assertEqual(second.reason, "busy")
            deadline = time.time() + 2.0
            while not done and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(done)
            third = try_start_smplx_capture(label="c", cooldown_s=30.0)
            self.assertFalse(third.started)
            self.assertEqual(third.reason, "cooldown")

    def test_progress_and_tail_error(self) -> None:
        self.assertTrue(is_capture_progress_line("INFO captured 12 hardware-sync groups"))
        self.assertTrue(is_capture_progress_line("-> [Optimize 3D Pose/12 frames]:  34.8s"))
        self.assertFalse(is_capture_progress_line("onnxruntime noise"))
        text = "INFO ok\nValueError: cam1 frame 435 violates undistorted/zero-distortion contract\n"
        self.assertIn("violates undistorted", _tail_error(text))

    def test_capture_env_blocks_user_site(self) -> None:
        env = capture_env(Path("/media/camp/EXT_DRIVE/RealUS_playground"))
        self.assertEqual(env.get("PYTHONNOUSERSITE"), "1")


if __name__ == "__main__":
    unittest.main()
