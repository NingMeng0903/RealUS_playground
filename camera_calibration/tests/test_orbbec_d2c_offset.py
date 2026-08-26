from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from multicam_calib.calib.orbbec_d2c_offset import (
    apply_R_depth_to_color,
    fit_R_depth_to_color,
    load_R_depth_to_color,
    save_d2c_offset,
)


def _T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


class TestApplyAndYaml(unittest.TestCase):
    def test_missing_file_is_identity(self) -> None:
        R, meta = load_R_depth_to_color(Path("/no/such/orbbec_d2c_offset.yaml"))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        self.assertEqual(meta["source"], "identity")

    def test_apply_rotates_points(self) -> None:
        R = Rsc.from_euler("xyz", [0.0, np.deg2rad(2.0), 0.0]).as_matrix()
        xyz = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        out = apply_R_depth_to_color(xyz, R)
        np.testing.assert_allclose(out[0], (R @ xyz[0]), atol=1e-6)

    def test_yaml_roundtrip(self) -> None:
        Rgt = Rsc.from_euler("xyz", np.deg2rad([0.1, 2.0, -0.2])).as_matrix()
        from multicam_calib.calib.orbbec_d2c_offset import D2COffsetFit

        fit = D2COffsetFit(
            R=Rgt,
            R_wahba=Rgt,
            per_view_deg_before=[2.0],
            per_view_deg_after=[0.1],
            loo_deg=[0.12],
            views=["g01"],
            n_points=10,
            serial="TEST",
            ptp_rms_mm_before=3.0,
            ptp_rms_mm_after=1.0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "orbbec_d2c_offset.yaml"
            save_d2c_offset(fit, path)
            R, meta = load_R_depth_to_color(path)
        np.testing.assert_allclose(R, Rgt, atol=1e-8)
        self.assertEqual(meta["serial"], "TEST")


class TestFitFromBoardViews(unittest.TestCase):
    def test_recovers_known_depth_to_color_rotation(self) -> None:
        Rgt = Rsc.from_euler("xyz", np.deg2rad([0.12, 1.99, -0.23])).as_matrix()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, (yaw, pitch) in enumerate(
                [(-0.4, 0.3), (-0.1, 0.5), (0.2, 0.35), (0.5, 0.2), (-0.3, 0.6)]
            ):
                n_tag = Rsc.from_euler("xyz", [pitch, yaw, 0.0]).apply([0.0, 0.0, -1.0])
                n_tag = n_tag / np.linalg.norm(n_tag)
                t = np.array([0.02, -0.03, 0.45], dtype=np.float64)
                z_ax = n_tag
                x_ax = np.cross(np.array([0.0, 1.0, 0.0]), z_ax)
                x_ax /= np.linalg.norm(x_ax)
                y_ax = np.cross(z_ax, x_ax)
                T = _T(np.column_stack([x_ax, y_ax, z_ax]), t)
                uu, vv = np.meshgrid(np.linspace(-0.1, 0.1, 12), np.linspace(-0.08, 0.08, 10))
                pts_tag = np.stack([uu.ravel(), vv.ravel(), np.zeros(uu.size)], axis=1)
                xyz_color = (pts_tag @ T[:3, :3].T) + t
                xyz_depth = xyz_color @ Rgt  # x_color = R @ x_depth ⇒ x_depth = R.T @ x_color
                gdir = root / f"g{i + 1:02d}"
                gdir.mkdir()
                n_dep = Rgt.T @ n_tag
                if float(n_dep @ xyz_depth.mean(axis=0)) > 0.0:
                    n_dep = -n_dep
                np.savez(gdir / "cloud.npz", xyz_board=xyz_depth.astype(np.float32), T_cam_board=T)
                (gdir / "summary.json").write_text(
                    json.dumps(
                        {
                            "serial": "AY2MC31016E",
                            "n_tag_cam": n_tag.tolist(),
                            "n_depth_board_cam": n_dep.tolist(),
                            "T_cam_board": T.tolist(),
                        }
                    ),
                    encoding="utf-8",
                )
            fit = fit_R_depth_to_color(root)
        ang = float(np.rad2deg(Rsc.from_matrix(fit.R.T @ Rgt).magnitude()))
        self.assertLess(ang, 0.15)
        self.assertLess(fit.rms_after, 0.15)
        self.assertLess(fit.rms_after, fit.rms_before)


if __name__ == "__main__":
    unittest.main()
