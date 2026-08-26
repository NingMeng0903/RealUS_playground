"""Robot/bed alignment must survive a UI restart without re-running hand-eye."""
from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.calib.robot_world import T_railbase_baselink, world_axes_from_railbase
from multicam_calib.io.config import RobotConfig
from multicam_calib.io.results import WorldMeta
from multicam_calib.recording.stage2_session import reconstruct_aligned_state_from_exports


class TestReconstructAlignedState(unittest.TestCase):
    def test_rebuilds_floor_and_bed_from_exports(self) -> None:
        T = np.eye(4)
        T[:3, :3] = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T[:3, 3] = [-0.4, -0.3, 1.9]
        rw = {
            "T_ref_railbase": T.tolist(),
            "T_tcp_board": np.eye(4).tolist(),
            "diagnostics": {"baselink_z_tilt_from_world_z_deg": 0.0},
        }
        meta = WorldMeta(
            origin_mode="bed_center_projected_to_floor",
            floor_plane_residual_mm=0.0,
            bed_height_m=0.284,
            bed_plane_residual_mm=3.36,
            bed_size_m=(0.0, 0.0),
            bed_center_world=[0.0, 0.0, 0.284],
            bed_center_on_floor=[0.0, 0.0, 0.0],
            corner_rects_xy=[],
            bed_outer_rect_xy=[],
            bed_rotation_deg=0.0,
            corner_fusion_std_mm=[],
            phases_completed=["robot", "bed"],
        )
        cfg = RobotConfig()
        state = reconstruct_aligned_state_from_exports(rw, meta, cfg)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state.floor_aligned)
        self.assertTrue(state.bed_aligned)
        self.assertAlmostEqual(state.bed_height_m or 0.0, 0.284, places=6)
        x, y, z = world_axes_from_railbase(T)
        np.testing.assert_allclose(state.x_axis, x, atol=1e-12)
        T_bl0 = T @ T_railbase_baselink(0.0, cfg.rail_y_origin_in_railbase_m)
        origin = T_bl0[:3, 3] - cfg.base_link_height_above_floor_m * z
        np.testing.assert_allclose(state.origin_tmp_ref, origin, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
