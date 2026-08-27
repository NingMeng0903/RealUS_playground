from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multicam_calib.calib.orbbec_depth_ray import (
    apply_depth_ray_scale,
    fit_depth_ray,
    load_depth_ray_coeff,
    save_depth_ray,
)


class TestApplyAndYaml(unittest.TestCase):
    def test_missing_file_is_identity(self) -> None:
        c, meta = load_depth_ray_coeff(Path("/no/such/orbbec_depth_ray.yaml"))
        np.testing.assert_allclose(c, [1.0, 0.0, 0.0])
        self.assertEqual(meta["source"], "identity")

    def test_apply_stays_on_ray(self) -> None:
        xyz = np.array([[0.10, -0.05, 0.40], [0.00, 0.00, 0.50]], dtype=np.float32)
        out = apply_depth_ray_scale(xyz, np.array([1.01, -0.03, 0.002]))
        for a, b in zip(xyz, out):
            na = a / np.linalg.norm(a)
            nb = b / np.linalg.norm(b)
            np.testing.assert_allclose(na, nb, atol=1e-6)

    def test_yaml_roundtrip(self) -> None:
        from multicam_calib.calib.orbbec_depth_ray import DepthRayFit

        fit = DepthRayFit(
            coeff=np.array([1.007, -0.034, 0.003]),
            views=["g01"],
            per_view_deg_before=[1.7],
            per_view_deg_after=[0.2],
            loo_deg=[0.22],
            n_points=10,
            serial="TEST",
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "orbbec_depth_ray.yaml"
            save_depth_ray(fit, path)
            c, meta = load_depth_ray_coeff(path)
        np.testing.assert_allclose(c, fit.coeff)
        self.assertEqual(meta["serial"], "TEST")


class TestFitFromBoardViews(unittest.TestCase):
    def test_recovers_known_linear_scale(self) -> None:
        w_gt = np.array([1.008, -0.034, 0.003], dtype=np.float64)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(1)
            for i, (yaw, pitch) in enumerate([(-0.3, 0.4), (0.1, 0.5), (0.4, 0.25), (-0.2, 0.55)]):
                n = np.array(
                    [
                        np.sin(yaw) * np.cos(pitch),
                        np.sin(pitch),
                        -np.cos(yaw) * np.cos(pitch),
                    ],
                    dtype=np.float64,
                )
                n = n / np.linalg.norm(n)
                t = np.array([0.01, -0.02, 0.42])
                d = float(n @ t)
                uu, vv = np.meshgrid(np.linspace(-0.12, 0.12, 14), np.linspace(-0.08, 0.08, 10))
                # points on the tag plane, then apply inverse of the scale so fit recovers w_gt
                pts = np.stack([uu.ravel(), vv.ravel(), np.zeros(uu.size)], axis=1)
                # build a camera-frame plane basis
                z_ax = n
                x_ax = np.cross(np.array([0.0, 1.0, 0.0]), z_ax)
                x_ax /= np.linalg.norm(x_ax)
                y_ax = np.cross(z_ax, x_ax)
                xyz_true = (pts @ np.column_stack([x_ax, y_ax, z_ax]).T) + t
                # observed = true / mult(w_gt)  so apply(w_gt) returns to plane
                z = np.clip(xyz_true[:, 2], 1e-6, None)
                mult = w_gt[0] + w_gt[1] * (xyz_true[:, 0] / z) + w_gt[2] * (xyz_true[:, 1] / z)
                xyz_obs = xyz_true / mult.reshape(-1, 1)
                T = np.eye(4)
                T[:3, :3] = np.column_stack([x_ax, y_ax, z_ax])
                T[:3, 3] = t
                gdir = root / f"g{i + 1:02d}"
                gdir.mkdir()
                np.savez(gdir / "cloud.npz", xyz_board=xyz_obs.astype(np.float32), T_cam_board=T)
                (gdir / "summary.json").write_text(
                    json.dumps({"serial": "AY2MC31016E", "T_cam_board": T.tolist()}),
                    encoding="utf-8",
                )
            _ = rng
            fit = fit_depth_ray(root)
        np.testing.assert_allclose(fit.coeff, w_gt, atol=5e-3)
        self.assertLess(fit.rms_after, 0.15)


if __name__ == "__main__":
    unittest.main()
