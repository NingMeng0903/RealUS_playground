"""NumPy URDF FK vs captured SHM ``T_railbase_tcp``."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from multicam_calib.calib.urdf_fk import UrdfFK
from multicam_calib.io.config import load_robot


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _sample_snaps() -> list[dict]:
    root = (
        _repo()
        / "camera_calibration"
        / "data"
        / "stage2_world"
        / "working"
        / "robot"
        / "samples"
    )
    if not root.is_dir():
        return []
    out = []
    for snap in sorted(root.glob("*/snapshot.json")):
        d = json.loads(snap.read_text())
        meta = d.get("metadata") or {}
        if meta.get("q_deg") is None or meta.get("T_railbase_tcp") is None:
            continue
        if meta.get("rail_m") is None:
            continue
        out.append(meta)
    return out


class TestUrdfFk(unittest.TestCase):
    def test_parses_link_7_to_tcp(self) -> None:
        cfg = load_robot()
        fk = UrdfFK(cfg.wbc_urdf_path())
        np.testing.assert_allclose(fk.link_7_to_tcp_xyz, [0.0, -0.01523, 0.12135], atol=1e-9)
        np.testing.assert_allclose(
            fk.link_7_to_tcp_rpy, [0.017732743, 0.870791073, -1.547861183], atol=1e-9
        )

    def test_matches_shm_translation_rotation_delta_constant(self) -> None:
        snaps = _sample_snaps()
        if len(snaps) < 8:
            self.skipTest("no robot samples with q_deg / T_railbase_tcp")
        cfg = load_robot()
        fk = UrdfFK(cfg.wbc_urdf_path())
        trans = []
        rot_deg = []
        R_deltas = []
        for meta in snaps:
            q = np.deg2rad(np.asarray(meta["q_deg"], dtype=np.float64).reshape(-1)[:7])
            T_fk = fk.fk(float(meta["rail_m"]), q)
            T_shm = np.asarray(meta["T_railbase_tcp"], dtype=np.float64).reshape(4, 4)
            trans.append(np.linalg.norm(T_fk[:3, 3] - T_shm[:3, 3]))
            R_d = T_fk[:3, :3].T @ T_shm[:3, :3]
            R_deltas.append(R_d)
            rot_deg.append(float(np.degrees(Rotation.from_matrix(R_d).magnitude())))
        self.assertLessEqual(max(trans), 1.1e-6, msg=f"max trans {max(trans)*1e6:.2f} µm")
        # Tool-frame constant: every sample must share the same R_fk^T R_shm.
        R0 = R_deltas[0]
        for R_d in R_deltas[1:]:
            err = float(np.degrees(Rotation.from_matrix(R0.T @ R_d).magnitude()))
            self.assertLess(err, 1e-4, msg=f"tool-frame delta not constant ({err:.4f} deg)")
        self.assertTrue(np.isfinite(np.mean(rot_deg)))


if __name__ == "__main__":
    unittest.main()
