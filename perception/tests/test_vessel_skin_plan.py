from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from perception.vessel_skin_plan import (
    bind_tpose_to_posed,
    consistent_along_polyline,
    extract_vessel_window,
    load_ready_plan,
    polyline_arclength,
    tool_axes_from_normal_tangent,
    write_plan_error,
    write_plan_json,
)
from perception.vessel_skin_plan import VesselPlan


class TestVesselSkinPlan(unittest.TestCase):
    def test_r_supfemv_window_is_10cm_on_right(self) -> None:
        n = 61
        t = np.linspace(0.0, 1.0, n)
        # SMPL: +X is left, so the right trunk sits at x < 0.
        pts = np.stack(
            [np.full(n, -0.12), np.zeros(n), 0.30 * t],
            axis=1,
        )
        window = extract_vessel_window(pts, h=t, window_m=0.10)
        length = float(polyline_arclength(window)[-1])
        self.assertAlmostEqual(length, 0.10, places=3)
        self.assertLess(float(np.mean(window[:, 0])), 0.0)

    def test_window_falls_back_to_mid_proximal_third(self) -> None:
        n = 51
        t = np.linspace(0.0, 1.0, n)
        pts = np.stack([np.full(n, -0.10), 0.40 * t, np.zeros(n)], axis=1)
        window = extract_vessel_window(pts, h=None, window_m=0.10)
        self.assertAlmostEqual(float(polyline_arclength(window)[-1]), 0.10, places=3)
        # Mid-proximal third of a 40 cm line is near s=0.133, so y ~ 0.08–0.18.
        self.assertLess(float(window[0, 1]), 0.20)
        self.assertGreater(float(window[-1, 1]), 0.08)

    def test_smplx_curve_rewrites_to_tcp_cartesian(self) -> None:
        from perception.vessel_skin_plan import load_T_smplx_from_tcp, smplx_poses_to_tcp

        T = load_T_smplx_from_tcp()
        pose_w = np.array([-0.11952878, 0.04011392, 0.47852174, 0.0, 0.0, 0.0])
        pose_t = smplx_poses_to_tcp(pose_w, T=T)
        back = T[:3, :3] @ pose_t[:3] + T[:3, 3]
        self.assertTrue(np.allclose(back, pose_w[:3], atol=1e-6))
        self.assertGreater(float(np.linalg.norm(pose_t[:3] - pose_w[:3])), 0.20)

    def test_consistent_along_polyline_flips_opposing_sample(self) -> None:
        nrm = np.array(
            [
                [-0.3, -0.2, -0.93],
                [-0.28, -0.22, -0.93],
                [0.24, 0.29, 0.93],
            ],
            dtype=float,
        )
        out = consistent_along_polyline(nrm)
        self.assertLess(float(out[2] @ nrm[2]), 0.0)
        self.assertGreater(float(out[2] @ out[1]), 0.0)

    def test_tool_axes_z_into_skin_y_along_path(self) -> None:
        # Outward +Z (thigh facing up) → tool +Z into skin (−world Z).
        R = tool_axes_from_normal_tangent(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        self.assertTrue(np.allclose(R[:, 2], [0.0, 0.0, -1.0], atol=1e-6))
        self.assertTrue(np.allclose(R[:, 1], [0.0, 1.0, 0.0], atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)
        self.assertLess(abs(float(R[:, 0] @ R[:, 1])), 1e-6)

    def test_barycentric_bind_tpose_to_posed(self) -> None:
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        posed = verts.copy()
        posed[:, 2] = 0.2
        tpose_pts = np.array([[0.25, 0.25, 0.0], [0.75, 0.25, 0.0]], dtype=float)
        world, _face, _bary = bind_tpose_to_posed(tpose_pts, verts, posed, faces)
        self.assertTrue(np.allclose(world[:, 2], 0.2, atol=1e-6))
        self.assertTrue(np.allclose(world[:, :2], tpose_pts[:, :2], atol=1e-5))

    def test_b_without_plan_is_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "smplx_outputs").mkdir()
            env = {"REALUS_SMPLX_OUTPUT_ROOT": str(root / "smplx_outputs")}
            with mock.patch.dict(os.environ, env, clear=False):
                plan, reason = load_ready_plan(repo=root)
            self.assertIsNone(plan)
            self.assertEqual(reason, "no capture")

    def test_error_plan_reports_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "smplx_outputs" / "20260831_000000"
            run.mkdir(parents=True)
            write_plan_error(run / "vessel_plan.json", "no R_SUPFEMV", run_name=run.name)
            env = {"REALUS_SMPLX_OUTPUT_ROOT": str(root / "smplx_outputs")}
            with mock.patch.dict(os.environ, env, clear=False):
                plan, reason = load_ready_plan(repo=root)
            self.assertIsNone(plan)
            self.assertEqual(reason, "no R_SUPFEMV")
            raw = json.loads((run / "vessel_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(raw["ok"])

    def test_b_not_blocked_by_capture_when_plan_exists(self) -> None:
        from peirastic.apps.vessel_scan import vessel_b_refuse_reason

        xyz = np.linspace([-0.12, 0.0, 0.0], [-0.12, 0.0, 0.10], 11)
        tan = np.repeat([[0.0, 0.0, 1.0]], 11, axis=0)
        nrm = np.repeat([[0.0, 1.0, 0.0]], 11, axis=0)
        poses = np.concatenate([xyz, np.zeros((11, 3))], axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "smplx_outputs" / "20260831_181435"
            run.mkdir(parents=True)
            plan = VesselPlan(
                path=run / "vessel_plan.json",
                label="R_SUPFEMV",
                run_name=run.name,
                world_xyz=xyz,
                scan_tangent=tan,
                skin_normals=nrm,
                scan_poses=poses,
                contact_pose=poses[0],
                tcp_poses=poses,
                tcp_contact=poses[0],
                window_m=0.10,
                standoff_m=0.05,
            )
            write_plan_json(plan)
            env = {"REALUS_SMPLX_OUTPUT_ROOT": str(root / "smplx_outputs")}
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("perception.capture_flow.capture_is_busy", return_value=True):
                    self.assertIsNone(vessel_b_refuse_reason(repo=root))


if __name__ == "__main__":
    unittest.main()
