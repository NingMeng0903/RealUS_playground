from __future__ import annotations

import unittest

import cv2
import numpy as np

from multicam_calib.io.config import load_orbbec

from multicam_calib.calib.orbbec_rgbd import (
    PinholeModel,
    align_factory_pinhole_to_stream,
    build_undistort_maps,
    opencv_dist_from_orbbec,
    overlay_bgr,
    point_cloud_stats,
    preview_mosaic,
    remap_like,
    se3_from_orbbec_extrinsic,
    unproject_aligned_depth,
    warp_depth_to_color,
)


def _radial_model(w: int = 80, h: int = 60, k1: float = -0.25) -> PinholeModel:
    K = np.array([[50.0, 0.0, w / 2.0], [0.0, 50.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return PinholeModel(K=K, dist=dist, image_size=(w, h), source="test")


class TestOrbbecDistMap(unittest.TestCase):
    def test_opencv_dist_from_object(self) -> None:
        class _D:
            k1, k2, p1, p2, k3 = 0.1, 0.2, 0.0, 0.0, 0.01

        d = opencv_dist_from_orbbec(_D())
        self.assertEqual(d.shape[0], 8)
        self.assertAlmostEqual(d[0], 0.1)
        self.assertAlmostEqual(d[4], 0.01)

    def test_se3_from_rot_trans(self) -> None:
        class _E:
            rot = np.eye(3)
            transform = np.array([0.01, 0.02, 0.03])

        T = se3_from_orbbec_extrinsic(_E())
        np.testing.assert_allclose(T[:3, 3], [0.01, 0.02, 0.03])


class TestUndistortAlignment(unittest.TestCase):
    def test_undistort_rgb_only_moves_a_corner_mark(self) -> None:
        model = _radial_model()
        maps = build_undistort_maps(model)
        h, w = 60, 80
        color = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 1.0, dtype=np.float32)
        color[5, 5] = (0, 0, 255)
        depth[5, 5] = 1.5

        views = preview_mosaic(
            color,
            depth,
            maps,
            mode="undistort_rgb_only",
            min_depth_m=0.2,
            max_depth_m=3.0,
            overlay_alpha=0.5,
        )
        undist = views["color_undistorted"]
        # Strong barrel: the painted pixel must leave (5,5) after undistort.
        self.assertFalse(np.array_equal(undist[5, 5], color[5, 5]))

    def test_undistort_both_keeps_depth_and_color_on_same_grid(self) -> None:
        model = _radial_model()
        maps = build_undistort_maps(model)
        h, w = 60, 80
        color = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        color[8:12, 8:12] = (0, 255, 0)
        depth[8:12, 8:12] = 1.2
        both = remap_like(color, maps)
        depth_u = remap_like(depth, maps)
        # The remapped green blob and remapped depth blob share support.
        color_mask = both[:, :, 1] > 0
        depth_mask = depth_u > 0.5
        overlap = np.count_nonzero(color_mask & depth_mask)
        self.assertGreater(overlap, 0)
        # RGB-only remap vs untouched depth does not share that support as tightly.
        only_color = remap_like(color, maps)
        only_mask = only_color[:, :, 1] > 0
        overlap_broken = np.count_nonzero(only_mask & (depth > 0.5))
        self.assertLess(overlap_broken, overlap)

    def test_raw_overlay_same_shape(self) -> None:
        model = _radial_model()
        maps = build_undistort_maps(model)
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        depth = np.ones((60, 80), dtype=np.float32)
        views = preview_mosaic(
            color, depth, maps, mode="raw_d2c", min_depth_m=0.2, max_depth_m=3.0, overlay_alpha=0.4
        )
        self.assertEqual(views["overlay"].shape, color.shape)


class TestCloudStats(unittest.TestCase):
    def test_empty_fails(self) -> None:
        s = point_cloud_stats(np.zeros((0, 3), dtype=np.float32))
        self.assertFalse(s.ok)
        self.assertEqual(s.n_points, 0)

    def test_unproject_and_stats_pass(self) -> None:
        K = np.array([[80.0, 0.0, 40.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]])
        depth = np.full((60, 80), 1.0, dtype=np.float32)
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        color[:] = (10, 20, 30)
        xyz, rgb = unproject_aligned_depth(depth, K, color_bgr=color, stride=1, min_m=0.2, max_m=2.0)
        self.assertEqual(xyz.shape[1], 3)
        self.assertEqual(rgb.shape[0], xyz.shape[0])
        s = point_cloud_stats(xyz, min_m=0.2, max_m=2.0, min_valid=100, min_valid_frac=0.5)
        self.assertTrue(s.ok)
        self.assertAlmostEqual(s.z_median_m, 1.0, places=5)

    def test_overlay_resizes_depth(self) -> None:
        color = np.zeros((40, 80, 3), dtype=np.uint8)
        depth_vis = np.zeros((20, 40, 3), dtype=np.uint8)
        out = overlay_bgr(color, depth_vis, 0.5)
        self.assertEqual(out.shape, color.shape)


class TestOrbbecConfig(unittest.TestCase):
    def test_load_default_yaml(self) -> None:
        cfg = load_orbbec()
        self.assertEqual(cfg.align, "d2c_sw")
        self.assertEqual(cfg.depth_fps, 15)
        self.assertTrue(cfg.depth_flip_h)
        self.assertEqual(cfg.color_width, 640)
        self.assertEqual(cfg.color_height, 480)
        self.assertEqual(cfg.depth_width, 640)
        self.assertEqual(cfg.depth_height, 400)
        self.assertGreater(cfg.max_depth_m, cfg.min_depth_m)


class TestFactoryScaleAndWarp(unittest.TestCase):
    def test_scales_640_factory_k_to_1080(self) -> None:
        k = np.array([[456.0, 0.0, 328.0], [0.0, 456.0, 246.0], [0.0, 0.0, 1.0]])
        model = PinholeModel(K=k, dist=np.zeros(5), image_size=(1920, 1080), source="factory")
        out = align_factory_pinhole_to_stream(model, (1920, 1080))
        self.assertEqual(out.image_size, (1920, 1080))
        self.assertAlmostEqual(out.K[0, 0], 456.0 * 3.0)
        self.assertAlmostEqual(out.K[0, 2], 328.0 * 3.0)
        self.assertAlmostEqual(out.K[1, 2], 246.0 * 2.25)

    def test_warp_identity_same_size(self) -> None:
        k = np.array([[200.0, 0.0, 40.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])
        depth = np.zeros((40, 80), dtype=np.float32)
        depth[20, 40] = 1.5
        out = warp_depth_to_color(depth, k, (80, 40), k, None)
        self.assertEqual(out.shape, (40, 80))
        self.assertAlmostEqual(float(out[20, 40]), 1.5, places=5)

    def test_warp_scales_to_larger_color(self) -> None:
        dk = np.array([[200.0, 0.0, 20.0], [0.0, 200.0, 10.0], [0.0, 0.0, 1.0]])
        ck = np.array([[400.0, 0.0, 40.0], [0.0, 400.0, 20.0], [0.0, 0.0, 1.0]])
        depth = np.zeros((20, 40), dtype=np.float32)
        depth[10, 20] = 1.0
        out = warp_depth_to_color(depth, dk, (80, 40), ck, None)
        self.assertEqual(out.shape, (40, 80))
        self.assertGreater(float(out[20, 40]), 0.5)


class TestUndistortMapsFinite(unittest.TestCase):
    def test_maps_cover_image(self) -> None:
        model = _radial_model()
        maps = build_undistort_maps(model)
        self.assertEqual(maps.map1.shape[:2], (60, 80))
        src = np.full((60, 80, 3), 128, dtype=np.uint8)
        dst = remap_like(src, maps)
        self.assertEqual(dst.shape, src.shape)
        self.assertGreater(int(cv2.countNonZero(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY))), 0)


if __name__ == "__main__":
    unittest.main()
