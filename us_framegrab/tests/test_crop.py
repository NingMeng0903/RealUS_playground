"""Synthetic-frame tests for brightness auto-crop and apply_crop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from us_framegrab.config import clamp_cbox, load_config  # noqa: E402
from us_framegrab.crop import (  # noqa: E402
    apply_crop,
    detect_sector_extrema,
    get_cropping_param,
    longest_true_run,
)


class TestLongestTrueRun(unittest.TestCase):
    def test_interior_run(self) -> None:
        self.assertEqual(longest_true_run([False, True, True, True, False, True]), (1, 4))

    def test_trailing_run_wins(self) -> None:
        self.assertEqual(longest_true_run([True, False, True, True, True, True]), (2, 6))

    def test_empty(self) -> None:
        self.assertEqual(longest_true_run([]), (0, 0))


class TestClampCbox(unittest.TestCase):
    def test_min_span_and_bounds(self) -> None:
        self.assertEqual(clamp_cbox([10, 8, 10, 8], 100, 80), [10, 12, 10, 12])
        self.assertEqual(clamp_cbox([-5, 500, -2, 400], 100, 80), [0, 100, 0, 80])


class TestAutoCrop(unittest.TestCase):
    def test_finds_bright_sector(self) -> None:
        img = np.full((1080, 1920), 10, dtype=np.uint8)
        img[180:780, 620:1300] = 200
        init = [550, 1650, 150, 920]
        ok, box = get_cropping_param(img, init)
        self.assertTrue(ok)
        assert box is not None
        x0, x1, y0, y1 = box
        self.assertLessEqual(abs(x0 - 620), 4)
        self.assertLessEqual(abs(x1 - 1300), 4)
        self.assertLessEqual(abs(y0 - 180), 4)
        self.assertGreater(y1, 760)

    def test_all_dark_fails(self) -> None:
        img = np.full((1080, 1920), 5, dtype=np.uint8)
        ok, box = get_cropping_param(img, [550, 1650, 150, 920])
        self.assertFalse(ok)
        self.assertIsNone(box)

    def test_finds_dark_convex_fan(self) -> None:
        rng = np.random.default_rng(0)
        h, w = 1080, 1920
        img = np.full((h, w), 4, dtype=np.uint8)
        cy, cx = 90, 960
        ys, xs = np.ogrid[:h, :w]
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        radius = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dx, dy))
        sector = (radius > 90) & (radius < 690) & (np.abs(angle) < 36.0) & (ys > 160)
        speckle = rng.integers(22, 70, size=img.shape, dtype=np.uint8)
        img[sector] = speckle[sector]
        near = sector & (radius < 170)
        img[near] = np.clip(img[near].astype(np.int16) + 90, 0, 255).astype(np.uint8)
        init = [550, 1650, 150, 920]
        ok, box = get_cropping_param(img, init)
        self.assertTrue(ok, "convex fan should be detected")
        assert box is not None
        x0, x1, y0, y1 = box
        ys_s, xs_s = np.where(sector)
        self.assertLessEqual(x0, int(xs_s.min()) + 25)
        self.assertGreaterEqual(x1, int(xs_s.max()) - 25)
        self.assertLessEqual(y0, int(ys_s.min()) + 25)
        self.assertGreaterEqual(y1, int(ys_s.max()) - 25)
        self.assertGreater(x1 - x0, 500)
        self.assertGreater(y1 - y0, 350)
        # Tight AABB of the fan, not the init_cbox search window.
        self.assertLessEqual(abs(x0 - int(xs_s.min())), 20)
        self.assertLessEqual(abs(x1 - int(xs_s.max()) - 1), 20)
        self.assertLessEqual(abs(y0 - int(ys_s.min())), 20)
        self.assertLessEqual(abs(y1 - int(ys_s.max()) - 1), 20)
        self.assertLess(x1 - x0, init[1] - init[0] - 80)
        self.assertLess(y1 - y0, init[3] - init[2] - 40)

    def test_ignores_full_width_chrome(self) -> None:
        rng = np.random.default_rng(1)
        h, w = 1080, 1920
        img = np.full((h, w), 3, dtype=np.uint8)
        img[180:780, 620:1300] = rng.integers(22, 70, size=(600, 680), dtype=np.uint8)
        img[150:165, 550:1650] = 90
        img[900:920, 550:1650] = 90
        init = [550, 1650, 150, 920]
        ok, box = get_cropping_param(img, init)
        self.assertTrue(ok)
        assert box is not None
        x0, x1, y0, y1 = box
        self.assertGreater(y0, 165)
        self.assertLess(y1, 895)
        self.assertGreater(x0, 580)
        self.assertLess(x1, 1350)

    def test_hdmi_black_level_does_not_swallow_fan(self) -> None:
        """Legal-range HDMI black (~16) must not push y1 to the search floor."""
        rng = np.random.default_rng(2)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        cy, cx = 90, 960
        ys, xs = np.ogrid[:h, :w]
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        radius = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dx, dy))
        sector = (radius > 90) & (radius < 690) & (np.abs(angle) < 36.0) & (ys > 160)
        speckle = rng.integers(28, 90, size=img.shape, dtype=np.uint8)
        img[sector] = speckle[sector]
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        ys_s, xs_s = np.where(sector)
        x0, x1, y0, y1 = ext["aabb"]
        self.assertLessEqual(abs(x0 - int(xs_s.min())), 8)
        self.assertLessEqual(abs(x1 - (int(xs_s.max()) + 1)), 8)
        self.assertLessEqual(abs(y0 - int(ys_s.min())), 8)
        self.assertLessEqual(abs(y1 - (int(ys_s.max()) + 1)), 8)
        self.assertEqual(ext["left"][0], x0)
        self.assertEqual(ext["right"][0], x1 - 1)
        self.assertEqual(ext["top"][1], y0)
        self.assertEqual(ext["bottom"][1], y1 - 1)
        self.assertLess(y1, init[3] - 80)

    def test_recovers_hollow_convex_nearfield(self) -> None:
        """Black + bright rings at the apex must lift y0; flood-fill eats that neck."""
        rng = np.random.default_rng(3)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        cy, cx = 90, 960
        ys, xs = np.ogrid[:h, :w]
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        radius = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dx, dy))
        sector = (radius > 90) & (radius < 690) & (np.abs(angle) < 36.0) & (ys > 160)
        body = sector & (ys >= 260)
        img[body] = rng.integers(30, 90, size=int(body.sum()), dtype=np.uint8).reshape(-1)
        # Hollow apex: only concentric bright rings (SonoScape near-field).
        for r0 in (110.0, 130.0, 150.0):
            ring = sector & (ys < 260) & (np.abs(radius - r0) <= 5.0)
            img[ring] = 210
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        ring_ys = np.where(sector & (ys < 260) & (img > 180))[0]
        self.assertGreater(ring_ys.size, 0)
        self.assertLessEqual(y0, int(ring_ys.min()) + 8)
        self.assertGreater(y1, 700)
        self.assertLess(y0, 200)
        self.assertLess(y0, 260 - 40)

    def test_apex_above_init_cbox_with_dark_gap(self) -> None:
        """3C-A: ring crest sits above init_cbox y0; dark gap then tissue."""
        rng = np.random.default_rng(4)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        cy, cx = 40, 960
        ys, xs = np.ogrid[:h, :w]
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        radius = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dx, dy))
        sector = (radius > 80) & (radius < 720) & (np.abs(angle) < 38.0) & (ys > 110)
        body = sector & (ys >= 260)
        img[body] = rng.integers(30, 90, size=int(body.sum()), dtype=np.uint8).reshape(-1)
        # Probe-face rings, including rows above init_cbox[2]=150.
        for r0 in (95.0, 115.0, 140.0):
            ring = sector & (ys < 220) & (np.abs(radius - r0) <= 4.0)
            img[ring] = 200
        # Compact "S" icon above the fan — must not become y0.
        img[100:108, 952:968] = 255
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        ring_ys = np.where(sector & (ys < 220) & (img > 180))[0]
        self.assertGreater(ring_ys.size, 0)
        self.assertLess(y0, 150, "apex must be allowed above init_cbox")
        self.assertGreater(y0, 108, "must skip the S icon")
        self.assertLessEqual(y0, int(ring_ys.min()) + 8)
        self.assertGreater(y1, 700)
        self.assertLess(x0, 700)
        self.assertGreater(x1, 1200)

    def test_upper_band_across_large_dark_gap(self) -> None:
        """Kidney / 17 cm: tissue starts mid-fan; y0 is the upper-band corners."""
        rng = np.random.default_rng(5)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        img[20:80, 200:1700] = 52  # gray title bar
        cy, cx = 40, 960
        ys, xs = np.ogrid[:h, :w]
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        radius = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dx, dy))
        sector = (radius > 80) & (radius < 780) & (np.abs(angle) < 40.0) & (ys > 110)
        body = sector & (ys >= 520)
        img[body] = rng.integers(30, 90, size=int(body.sum()), dtype=np.uint8).reshape(-1)
        for r0 in (90.0, 110.0, 130.0):
            ring = sector & (ys < 180) & (np.abs(radius - r0) <= 3.0)
            img[ring] = 200
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        ring_ys = np.where(sector & (ys < 180) & (img > 180))[0]
        self.assertGreater(ring_ys.size, 0)
        self.assertLess(y0, int(0.25 * h), "probe-face stays in the top quarter")
        self.assertNotEqual(y0, 520)
        self.assertLess(y0, 200)
        self.assertLessEqual(y0, int(ring_ys.min()) + 8)
        self.assertGreater(y1, 750)
        self.assertIn("band_left", ext)
        self.assertIn("band_right", ext)
        self.assertLess(ext["band_left"][0], ext["band_right"][0])

    def test_linear_does_not_lift_to_ruler_ticks(self) -> None:
        """L741-style rectangle: keep the scan top, ignore HDMI ruler ticks above it."""
        rng = np.random.default_rng(6)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        img[200:780, 700:1450] = rng.integers(22, 70, size=(580, 750), dtype=np.uint8)
        # Near-field bright bands — the real linear top.
        img[200:210, 700:1450] = 190
        # Ruler ticks above the scan (the false "upper band").
        for x in range(720, 1430, 36):
            img[152:168, x : x + 3] = 230
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        self.assertGreaterEqual(y0, 190)
        self.assertLess(y0, 220)
        self.assertGreater(x0, 680)
        self.assertLess(x1, 1470)
        self.assertGreater(y1, 760)

    def test_low_gain_linear_is_not_just_the_bright_bands(self) -> None:
        """Low dyn: dark speckle must stay in the box; do not collapse to the top bands."""
        rng = np.random.default_rng(7)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        img[200:780, 700:1450] = rng.integers(18, 30, size=(580, 750), dtype=np.uint8)
        img[200:214, 700:1450] = 170
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        self.assertLessEqual(abs(y0 - 200), 6)
        self.assertGreater(y1, 740)
        self.assertGreater(y1 - y0, 500)
        self.assertLessEqual(abs(x0 - 700), 6)
        self.assertLessEqual(abs(x1 - 1450), 6)

    def test_cfm_hairline_at_edge_does_not_eat_black(self) -> None:
        """CFM ROI flush to the linear edge must not expand the crop into bezel."""
        rng = np.random.default_rng(8)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        img[200:780, 700:1450] = rng.integers(28, 80, size=(580, 750), dtype=np.uint8)
        img[200:214, 700:1450] = 180
        # 2 px CFM line on the true right edge + a 3 px glow in the black.
        img[260:720, 1448:1450] = 210
        img[260:720, 1450:1453] = 42
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        self.assertLessEqual(abs(x1 - 1450), 3)
        self.assertLessEqual(abs(x0 - 700), 3)
        self.assertGreater(y1, 740)
        self.assertLess(y0, 220)

    def test_linear_ignores_s_icon_left_of_band(self) -> None:
        """SonoScape S marker left of the scan must not become x0."""
        rng = np.random.default_rng(9)
        h, w = 1080, 1920
        img = np.full((h, w), 16, dtype=np.uint8)
        img[200:780, 700:1450] = rng.integers(22, 70, size=(580, 750), dtype=np.uint8)
        img[200:212, 700:1450] = 190
        img[196:224, 660:692] = 240
        init = [550, 1650, 150, 920]
        ext = detect_sector_extrema(img, init)
        self.assertIsNotNone(ext)
        assert ext is not None
        x0, x1, y0, y1 = ext["aabb"]
        self.assertGreaterEqual(x0, 696)
        self.assertLessEqual(abs(x0 - 700), 4)
        self.assertLessEqual(abs(x1 - 1450), 4)
        self.assertGreaterEqual(y0, 196)
        self.assertLess(y0, 216)
        self.assertGreater(y1, 760)


class TestApplyCrop(unittest.TestCase):
    def test_gray_hflip(self) -> None:
        frame = np.arange(20, dtype=np.uint8).reshape(4, 5)
        cropped = apply_crop(frame, [1, 4, 1, 3], color=False, hflip=True)
        self.assertEqual(cropped.shape, (2, 3))
        np.testing.assert_array_equal(cropped[0], frame[1, 1:4][::-1])


class TestLoadConfig(unittest.TestCase):
    def test_package_yaml(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
        self.assertEqual(cfg.pub_bind, "tcp://127.0.0.1:17359")
        self.assertEqual(cfg.camera_name, "us_img")
        self.assertEqual(cfg.machine, "sonoscape_e2")
        self.assertEqual(len(cfg.final_cbox), 4)
        self.assertLess(cfg.final_cbox[0], cfg.final_cbox[1])
        self.assertLess(cfg.final_cbox[2], cfg.final_cbox[3])


if __name__ == "__main__":
    unittest.main()
