from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from multicam_calib.calib.orbbec_handeye import (
    calibrate_handeye_init,
    load_orbbec_color_intrinsics,
    load_orbbec_handeye_captures,
    payload_to_captures,
    save_orbbec_handeye_captures,
)
from multicam_calib.io.results import Intrinsics, load_joint_zero_offsets_deg
from multicam_calib.calib.pose_graph import se3_inv
from multicam_calib.calib.urdf_fk import UrdfFK
from multicam_calib.io.config import load_robot


def _T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


class TestLink7FromTcp(unittest.TestCase):
    def test_roundtrip_tcp(self) -> None:
        fk = UrdfFK(load_robot().wbc_urdf_path())
        T_l7_tcp = fk.T_link7_tcp()
        T_rt = _T(Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix(), [0.3, -0.2, 0.5])
        T_l7 = fk.T_railbase_link7(T_rt)
        np.testing.assert_allclose(T_l7 @ T_l7_tcp, T_rt, atol=1e-12)


class TestHandeyeInit(unittest.TestCase):
    def test_recovers_known_T_link7_cam(self) -> None:
        T_link7_cam = _T(Rotation.from_euler("xyz", [0.15, -0.4, 1.2]).as_matrix(), [0.04, -0.02, 0.08])
        T_rb_board = _T(Rotation.from_euler("xyz", [0.0, 0.0, 0.3]).as_matrix(), [0.2, 0.1, 0.05])
        T_g: list[np.ndarray] = []
        T_c: list[np.ndarray] = []
        for yaw in np.linspace(-0.8, 0.8, 8):
            for pitch in (-0.4, 0.35):
                T_l7 = _T(
                    Rotation.from_euler("xyz", [0.1, pitch, yaw]).as_matrix(),
                    [0.4 + 0.05 * yaw, 0.0, 0.35],
                )
                T_g.append(T_l7)
                T_c.append(se3_inv(T_l7 @ T_link7_cam) @ T_rb_board)
        T_hat = calibrate_handeye_init(T_g, T_c)
        np.testing.assert_allclose(T_hat[:3, 3], T_link7_cam[:3, 3], atol=5e-4)
        R_err = T_hat[:3, :3].T @ T_link7_cam[:3, :3]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))))
        self.assertLess(angle, 0.2)


class TestLoadOrbbecK(unittest.TestCase):
    def test_skips_saved_k_when_stream_size_differs(self) -> None:
        factory = Intrinsics(
            K=np.eye(3),
            dist=np.zeros(5),
            image_size=(999, 888),
            source="factory",
        )
        got = load_orbbec_color_intrinsics(factory=factory, image_size=(999, 888))
        self.assertEqual(tuple(got.image_size), (999, 888))
        self.assertIn(got.source, {"factory", "chessboard_scaled", "factory_scaled"})


class TestHandeyeCaptureRoundtrip(unittest.TestCase):
    def test_save_load_preserves_q_and_tags(self) -> None:
        import tempfile
        from pathlib import Path

        cap = {
            "n_tags": 2,
            "rail_m": 0.123,
            "q_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "image_size": [640, 480],
            "T_railbase_tcp": np.eye(4).tolist(),
            "detections": {
                10: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                11: [[9.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "caps.yaml"
            save_orbbec_handeye_captures([cap], path)
            got = load_orbbec_handeye_captures(path)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["n_tags"], 2)
        self.assertAlmostEqual(got[0]["rail_m"], 0.123)
        self.assertEqual(got[0]["q_deg"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertIn(10, got[0]["detections"])
        self.assertEqual(len(got[0]["detections"][10]), 4)

    def test_payload_accepts_legacy_list(self) -> None:
        rows = payload_to_captures(
            [
                {
                    "n_tags": 1,
                    "rail_m": 0.0,
                    "q_deg": [0.0] * 7,
                    "image_size": [640, 480],
                    "T_railbase_tcp": np.eye(4).tolist(),
                    "detections": {1: [[0, 0], [1, 0], [1, 1], [0, 1]]},
                }
            ]
        )
        self.assertEqual(len(rows), 1)


class TestStage2OffsetsInFk(unittest.TestCase):
    def test_loads_j6_and_moves_flange(self) -> None:
        fk = UrdfFK(load_robot().wbc_urdf_path())
        dq = load_joint_zero_offsets_deg(urdf_sha1=fk.sha1)
        self.assertAlmostEqual(float(dq[5]), -1.015, places=2)
        self.assertEqual(float(dq[6]), 0.0)
        q = np.zeros(7)
        rail = 0.05
        t0 = fk.fk(rail, np.deg2rad(q), None)
        t1 = fk.fk(rail, np.deg2rad(q), np.deg2rad(dq[:6]))
        dt = float(np.linalg.norm(t1[:3, 3] - t0[:3, 3]))
        self.assertGreater(dt, 0.001)
        self.assertLess(dt, 0.025)


if __name__ == "__main__":
    unittest.main()
