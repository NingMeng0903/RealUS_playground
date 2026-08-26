"""Robot-world hand-eye: OpenCV convention lock + synthetic pipeline."""
from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from multicam_calib.board.apriltag_board import build_board_geometry
from multicam_calib.calib.pose_graph import se3_inv
from multicam_calib.calib.robot_world import (
    T_railbase_baselink,
    build_robot_world_export,
    calibrate_robot_world_handeye_init,
    fit_rail_axis,
    solve_robot_world,
    visual_slider_point_ref,
    world_axes_from_railbase,
)
from multicam_calib.calib.urdf_fk import UrdfFK
from multicam_calib.io.config import KinematicFitConfig, RobotConfig, load_board_ee, load_robot
from multicam_calib.io.results import ExtrinsicsSet, Intrinsics
from multicam_calib.recording.session import Sample, ViewDetections


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _rotz(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", np.deg2rad(deg)).as_matrix()


class TestHandEyeConvention(unittest.TestCase):
    def test_opencv_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        T_ref_rb = _T(_rotz(-90.0), [0.4, -0.2, 0.8])
        T_tcp_board = _T(Rotation.from_euler("xyz", [0.1, -0.05, 0.2]).as_matrix(), [0.0, 0.0, 0.08])
        T_boards = []
        T_tcps = []
        for i in range(8):
            rail = 0.1 * i
            wrist = Rotation.from_euler("xyz", [0.2 * np.sin(i), 0.15 * np.cos(i), 0.05 * i]).as_matrix()
            T_rt = T_railbase_baselink(rail) @ _T(wrist, [0.35, 0.0, 0.25])
            T_boards.append(T_ref_rb @ T_rt @ T_tcp_board)
            T_tcps.append(T_rt)
        T_rb_hat, T_tb_hat = calibrate_robot_world_handeye_init(T_boards, T_tcps)
        # Recovered transforms should reproduce T_ref_board.
        for Tb, Trt in zip(T_boards, T_tcps):
            pred = T_rb_hat @ Trt @ T_tb_hat
            err_t = np.linalg.norm(pred[:3, 3] - Tb[:3, 3])
            err_r = np.rad2deg(
                Rotation.from_matrix(pred[:3, :3].T @ Tb[:3, :3]).magnitude()
            )
            self.assertLess(err_t, 1e-6, msg=f"trans {err_t}")
            self.assertLess(err_r, 1e-4, msg=f"rot {err_r}")


class TestWorldAxesAndHeight(unittest.TestCase):
    def test_x_is_rail_axis_and_z_orthogonal(self) -> None:
        # rail_base +Y should become world +X after Rz(-90).
        T = _T(_rotz(-90.0), [1.0, 2.0, 0.5])
        x, y, z = world_axes_from_railbase(T)
        np.testing.assert_allclose(x, [1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(z, [0.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(float(x @ z), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.det(np.stack([x, y, z], axis=1))), 1.0, places=9)

    def test_baselink_z_is_height_constant(self) -> None:
        cfg = load_robot()
        T_ref_rb = _T(_rotz(-90.0), [0.5, 0.1, 1.2])
        x, y, z = world_axes_from_railbase(T_ref_rb)
        h = cfg.base_link_height_above_floor_m
        T_bl0 = T_ref_rb @ T_railbase_baselink(0.0, cfg.rail_y_origin_in_railbase_m)
        origin = T_bl0[:3, 3] - h * z
        basis_R = np.stack([x, y, z], axis=0)
        for rail in (0.0, 0.3, 0.8):
            T_bl = T_ref_rb @ T_railbase_baselink(rail, cfg.rail_y_origin_in_railbase_m)
            p_world = basis_R @ (T_bl[:3, 3] - origin)
            self.assertAlmostEqual(p_world[2], h, places=12)

    def test_export_base_pos_z(self) -> None:
        cfg = RobotConfig()
        T_ref_rb = _T(_rotz(-90.0), [0.2, -0.1, 0.9])
        x, y, z = world_axes_from_railbase(T_ref_rb)
        T_bl0 = T_ref_rb @ T_railbase_baselink(0.0, cfg.rail_y_origin_in_railbase_m)
        origin = T_bl0[:3, 3] - cfg.base_link_height_above_floor_m * z
        T_world_ref = np.eye(4)
        T_world_ref[:3, :3] = np.stack([x, y, z], axis=0)
        T_world_ref[:3, 3] = -T_world_ref[:3, :3] @ origin
        payload = build_robot_world_export(
            T_world_railbase=T_world_ref @ T_ref_rb,
            T_ref_railbase=T_ref_rb,
            T_tcp_board=np.eye(4),
            robot_cfg=cfg,
            diagnostics={},
        )
        self.assertAlmostEqual(payload["base_pos_m"][2], cfg.base_link_height_above_floor_m, places=12)

    def test_railbase_baselink_offset(self) -> None:
        T = T_railbase_baselink(0.0, (0.0, -0.4, 0.0))
        np.testing.assert_allclose(T[:3, 3], [0.0, -0.4, 0.0])
        T2 = T_railbase_baselink(0.8, (0.0, -0.4, 0.0))
        np.testing.assert_allclose(T2[:3, 3], [0.0, 0.4, 0.0])


class TestFitRailAxis(unittest.TestCase):
    def test_recovers_direction(self) -> None:
        d = np.array([0.0, 1.0, 0.0])
        pts = np.array([0.2 * i * d for i in range(6)])
        hat = fit_rail_axis(pts)
        self.assertAlmostEqual(abs(float(hat @ d)), 1.0, places=9)

    def test_slider_points_ignore_arm_motion(self) -> None:
        T_ref_rb = _T(_rotz(-90.0), [0.4, -0.2, 0.8])
        T_tcp_board = _T(Rotation.from_euler("xyz", [0.1, -0.05, 0.2]).as_matrix(), [0.0, 0.0, 0.08])
        pts = []
        for i in range(6):
            rail = 0.12 * i
            wrist = Rotation.from_euler("xyz", [0.4 * i, -0.3 * i, 0.2 * i]).as_matrix()
            T_rt = T_railbase_baselink(rail) @ _T(wrist, [0.35, 0.02, 0.22])
            T_board = T_ref_rb @ T_rt @ T_tcp_board
            pts.append(visual_slider_point_ref(T_board, T_rt, rail, T_tcp_board))
        hat = fit_rail_axis(np.stack(pts, axis=0))
        x, _, _ = world_axes_from_railbase(T_ref_rb)
        self.assertAlmostEqual(abs(float(hat @ x)), 1.0, places=9)


def _project(T_cam_board: np.ndarray, obj: np.ndarray, K: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    proj, _ = cv2.projectPoints(obj, rvec, T_cam_board[:3, 3], K, np.zeros(5))
    return proj.reshape(-1, 2)


class TestSyntheticFullSolve(unittest.TestCase):
    def test_ba_recovers_transforms(self) -> None:
        geom = build_board_geometry(load_board_ee())
        K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
        intr = {
            "cam1": Intrinsics(K=K, dist=np.zeros(5), image_size=(1920, 1080), source="test"),
            "cam2": Intrinsics(K=K, dist=np.zeros(5), image_size=(1920, 1080), source="test"),
        }
        T_ref_c1 = np.eye(4)
        T_ref_c2 = _T(_rotz(12.0), [0.25, 0.0, 0.0])
        stage1 = ExtrinsicsSet(reference="cam1", poses={"cam1": T_ref_c1, "cam2": T_ref_c2})

        T_ref_rb = _T(_rotz(-90.0), [0.3, -0.4, 1.0])
        T_tcp_board = _T(Rotation.from_euler("xyz", [0.08, -0.04, 0.15]).as_matrix(), [0.01, -0.02, 0.07])

        samples: list[Sample] = []
        for i in range(8):
            rail = 0.08 * i
            wrist = Rotation.from_euler("xyz", [0.25 * np.sin(i), 0.2 * np.cos(i), 0.04 * i]).as_matrix()
            T_rt = T_railbase_baselink(rail) @ _T(wrist, [0.30, 0.02, 0.22])
            T_ref_board = T_ref_rb @ T_rt @ T_tcp_board
            views = {}
            for alias, T_ref_cam in stage1.poses.items():
                T_cam_board = se3_inv(T_ref_cam) @ T_ref_board
                tags = {}
                # Inverse of EE board perm [1,2,3,0]: pupil = board[[3,0,1,2]].
                pupil_from_board = np.array([3, 0, 1, 2])
                for tid, corners in geom.corners_by_tag.items():
                    px = _project(T_cam_board, corners, K).reshape(4, 2)
                    tags[tid] = px[pupil_from_board]
                views[alias] = ViewDetections(alias=alias, tags=tags)
            samples.append(
                Sample(
                    index=i,
                    host_timestamp_ns=i,
                    views=views,
                    metadata={
                        "phase": "robot",
                        "rail_m": float(rail),
                        "T_railbase_tcp": T_rt.tolist(),
                    },
                )
            )

        out = solve_robot_world(
            samples, geom, intr, stage1, min_tags=7, robot_cfg=RobotConfig()
        )
        pred = out.T_ref_railbase @ T_railbase_baselink(0.0) @ se3_inv(T_railbase_baselink(0.0))
        err_t = np.linalg.norm(out.T_ref_railbase[:3, 3] - T_ref_rb[:3, 3])
        err_r = np.rad2deg(
            Rotation.from_matrix(out.T_ref_railbase[:3, :3].T @ T_ref_rb[:3, :3]).magnitude()
        )
        self.assertLess(err_t, 1e-3, msg=f"T_ref_railbase trans {err_t*1000:.2f} mm")
        self.assertLess(err_r, 0.1, msg=f"T_ref_railbase rot {err_r:.3f} deg")
        err_tb = np.linalg.norm(out.T_tcp_board[:3, 3] - T_tcp_board[:3, 3])
        self.assertLess(err_tb, 1e-3)
        self.assertLess(float(out.diagnostics["rail_axis_residual_deg"]), 0.5)
        self.assertFalse(out.diagnostics.get("joint_offsets_enabled", True))


class TestJointOffsetSolve(unittest.TestCase):
    def test_recovers_known_offsets(self) -> None:
        cfg = RobotConfig(
            kinematic_fit=KinematicFitConfig(joint_offsets=True, rail_span_min_m=0.05)
        )
        fk = UrdfFK(cfg.wbc_urdf_path())
        true_off = np.deg2rad([0.0, 0.0, 0.0, -0.30, 0.0, -0.80])
        q0 = np.deg2rad([17.5, -14.1, 64.8, 83.9, -58.4, 84.1, 10.0])

        geom = build_board_geometry(load_board_ee())
        K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
        intr = {
            "cam1": Intrinsics(K=K, dist=np.zeros(5), image_size=(1920, 1080), source="test"),
            "cam2": Intrinsics(K=K, dist=np.zeros(5), image_size=(1920, 1080), source="test"),
        }
        T_ref_c1 = np.eye(4)
        T_ref_c2 = _T(_rotz(12.0), [0.25, 0.0, 0.0])
        stage1 = ExtrinsicsSet(reference="cam1", poses={"cam1": T_ref_c1, "cam2": T_ref_c2})
        T_ref_rb = _T(_rotz(-90.0), [0.3, -0.4, 1.0])
        T_tcp_board = _T(Rotation.from_euler("xyz", [0.08, -0.04, 0.15]).as_matrix(), [0.01, -0.02, 0.07])

        samples: list[Sample] = []
        for i in range(12):
            rail = 0.05 * i + 0.04
            q_true = q0.copy()
            q_true[0] += 0.15 * np.sin(0.7 * i)
            q_true[1] += 0.12 * np.sin(i)
            q_true[3] += 0.10 * np.cos(i)
            q_true[4] += 0.18 * np.sin(1.3 * i)
            q_true[5] += 0.20 * np.cos(1.1 * i)
            T_rt = fk.fk(rail, q_true, np.zeros(6))
            q_meas = q_true.copy()
            q_meas[:6] = q_meas[:6] - true_off
            T_ref_board = T_ref_rb @ T_rt @ T_tcp_board
            views = {}
            for alias, T_ref_cam in stage1.poses.items():
                T_cam_board = se3_inv(T_ref_cam) @ T_ref_board
                tags = {}
                pupil_from_board = np.array([3, 0, 1, 2])
                for tid, corners in geom.corners_by_tag.items():
                    px = _project(T_cam_board, corners, K).reshape(4, 2)
                    tags[tid] = px[pupil_from_board]
                views[alias] = ViewDetections(alias=alias, tags=tags)
            samples.append(
                Sample(
                    index=i,
                    host_timestamp_ns=i,
                    views=views,
                    metadata={
                        "phase": "robot",
                        "rail_m": float(rail),
                        "q_deg": np.rad2deg(q_meas).tolist(),
                        "T_railbase_tcp": T_rt.tolist(),
                        "capture_group": "a" if i < 4 else "b",
                    },
                )
            )

        out = solve_robot_world(samples, geom, intr, stage1, min_tags=7, robot_cfg=cfg)
        self.assertTrue(out.diagnostics["joint_offsets_enabled"])
        got = np.asarray(out.joint_zero_offsets_deg, dtype=np.float64)
        np.testing.assert_allclose(got[3], np.rad2deg(true_off[3]), atol=0.15)
        self.assertLess(abs(got[5] - np.rad2deg(true_off[5])), 0.35)
        self.assertAlmostEqual(got[6], 0.0, places=9)
        self.assertLess(float(out.diagnostics["ba_rmse_px"]), 0.05)


class TestUrdfRailOrigin(unittest.TestCase):
    def test_wbc_and_viewer_share_minus_0p4(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        wbc = repo / "rm75_control/rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf"
        viewer = repo / "rm75_control/rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.slider.generated.urdf"
        self.assertTrue(wbc.is_file())
        self.assertTrue(viewer.is_file())
        import xml.etree.ElementTree as ET

        def origin_xyz(path: Path) -> list[float]:
            tree = ET.parse(path)
            for joint in tree.findall("joint"):
                if joint.get("name") == "rail_y":
                    xyz = (joint.find("origin").get("xyz") or "0 0 0").split()
                    return [float(v) for v in xyz]
            raise AssertionError(f"no rail_y in {path}")

        w = origin_xyz(wbc)
        v = origin_xyz(viewer)
        self.assertAlmostEqual(w[1], -0.4)
        self.assertAlmostEqual(v[1], -0.4)
        self.assertAlmostEqual(w[0], 0.0)
        self.assertAlmostEqual(v[0], 0.0)


if __name__ == "__main__":
    unittest.main()
