"""SHM dtype must stay byte-identical to rm75_state."""
from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.ingress.robot_state import (
    expected_shm_size,
    layout_dtype,
    pose6_to_T,
    slot_dtype,
    T_to_pose6,
)


class TestRobotStateDtype(unittest.TestCase):
    def test_itemsize_matches_upstream_definition(self) -> None:
        # Copied from rm75_control/.../state_relay.py — if this fails, SHM broke.
        header = np.dtype([("active", "<u8"), ("global_seq", "<u8"), ("session_id", "<u8")])
        slot = np.dtype(
            [
                ("seq", "<u8"),
                ("t_s", "<f8"),
                ("q_deg", "<f8", (7,)),
                ("pose", "<f8", (6,)),
                ("force", "<f8", (6,)),
                ("rail_m", "<f8"),
                ("ok", "u1"),
            ],
            align=True,
        )
        layout = np.dtype([("header", header), ("slots", slot, (2,))])
        self.assertEqual(slot_dtype().itemsize, slot.itemsize)
        self.assertEqual(layout_dtype().itemsize, layout.itemsize)
        self.assertEqual(expected_shm_size(), int(layout.itemsize))

    def test_optional_import_rm75_control(self) -> None:
        try:
            from rm75_control.control.admittance_common import state_relay
        except ImportError:
            self.skipTest("rm75_control not importable in this env")
        self.assertEqual(slot_dtype().itemsize, state_relay._SLOT_DTYPE.itemsize)
        self.assertEqual(layout_dtype().itemsize, state_relay._LAYOUT_DTYPE.itemsize)

    def test_attach_does_not_unlink_on_subscriber_exit(self) -> None:
        import os
        import subprocess
        import sys
        import textwrap
        from multiprocessing import shared_memory
        from pathlib import Path

        from multicam_calib.ingress.robot_state import SHM_SIZE

        name = "rm75_state_calib_test"
        src = str(Path(__file__).resolve().parents[1] / "src")
        shm = shared_memory.SharedMemory(name=name, create=True, size=SHM_SIZE)
        try:
            script = textwrap.dedent(
                f"""
                from multicam_calib.ingress.robot_state import RobotStateReader
                reader = RobotStateReader({name!r})
                reader.attach()
                reader.close()
                """
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.check_call([sys.executable, "-c", script], env=env)
            self.assertTrue(
                Path("/dev/shm").joinpath(name).exists(),
                "subscriber exit unlinked the publisher SHM name",
            )
        finally:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def test_pose6_roundtrip(self) -> None:
        pose = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2], dtype=np.float64)
        back = T_to_pose6(pose6_to_T(pose))
        np.testing.assert_allclose(back[:3], pose[:3], atol=1e-9)
        np.testing.assert_allclose(back[3:], pose[3:], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
