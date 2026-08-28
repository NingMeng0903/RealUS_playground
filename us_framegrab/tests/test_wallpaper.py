"""Wallpaper store, locked Default theme, and 9:16 crop geometry."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from us_framegrab.wallpaper import (  # noqa: E402
    DEFAULT_ID,
    WallpaperStore,
    cover_source_box,
    crop_bgr,
    default_wallpaper_dir,
    largest_portrait_box,
    list_image_paths,
    migrate_wallpapers,
    move_portrait_box,
    resize_portrait_box,
    ui_wallpaper_dir,
)


class TestPortraitBox(unittest.TestCase):
    def test_largest_is_9_by_16_on_1080p(self) -> None:
        x0, x1, y0, y1 = largest_portrait_box(1920, 1080)
        self.assertEqual(y1 - y0, 1080)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 9 / 16, places=2)

    def test_move_stays_inside(self) -> None:
        box = largest_portrait_box(400, 800)
        moved = move_portrait_box(box, -1000, -1000, 400, 800)
        self.assertEqual(moved[0], 0)
        self.assertEqual(moved[2], 0)
        moved = move_portrait_box(box, 1000, 1000, 400, 800)
        self.assertEqual(moved[1], 400)
        self.assertLessEqual(moved[3], 800)

    def test_resize_keeps_aspect(self) -> None:
        box = largest_portrait_box(800, 800)
        smaller = resize_portrait_box(box, "e", -200, 0, 800, 800)
        w, h = smaller[1] - smaller[0], smaller[3] - smaller[2]
        self.assertAlmostEqual(w / h, 9 / 16, places=2)
        self.assertLess(w, box[1] - box[0])

    def test_crop_extracts_box(self) -> None:
        img = np.zeros((80, 90, 3), dtype=np.uint8)
        img[10:42, 9:27] = (0, 40, 200)
        cut = crop_bgr(img, [9, 27, 10, 42])
        self.assertEqual(cut.shape, (32, 18, 3))
        self.assertEqual(int(cut[0, 0, 2]), 200)


class TestCoverSource(unittest.TestCase):
    def test_covers_wide_pane(self) -> None:
        sx, sy, sw, sh = cover_source_box(90, 160, 340, 200, align="bottom")
        self.assertAlmostEqual(sw / sh, 340 / 200, places=5)
        self.assertGreaterEqual(sx, -1e-6)
        self.assertAlmostEqual(sy + sh, 160, places=5)

    def test_lists_jpg_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"x")
            (root / "b.PNG").write_bytes(b"x")
            (root / "note.txt").write_bytes(b"x")
            names = {path.name.lower() for path in list_image_paths(root)}
            self.assertEqual(names, {"a.jpg", "b.png"})


class TestWallpaperStore(unittest.TestCase):
    def test_default_dir_under_ui(self) -> None:
        dest = ui_wallpaper_dir()
        self.assertEqual(dest.name, "wallpapers")
        self.assertEqual(dest.parent.name, "ui")
        self.assertEqual(default_wallpaper_dir().resolve(), dest.resolve())

    def test_migrate_copies_legacy_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "old"
            dest = Path(tmp) / "new"
            legacy.mkdir()
            (legacy / "index.yaml").write_text("active: default\nthemes: []\n", encoding="utf-8")
            (legacy / "wp_abc.jpg").write_bytes(b"img")
            migrate_wallpapers(dest, legacy)
            self.assertTrue((dest / "wp_abc.jpg").is_file())
            self.assertTrue((dest / "index.yaml").is_file())

    def test_default_cannot_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WallpaperStore(Path(tmp))
            self.assertEqual([t.id for t in store.list_all()], [DEFAULT_ID])
            self.assertFalse(store.delete_theme(DEFAULT_ID))
            self.assertEqual(store.active_id, DEFAULT_ID)
            self.assertIsNone(store.load_bgr(DEFAULT_ID))

    def test_add_preview_load_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WallpaperStore(Path(tmp))
            canvas = np.zeros((160, 90, 3), dtype=np.uint8)
            canvas[:] = (30, 80, 160)
            theme = store.add_theme("Kidney", canvas)
            self.assertEqual(theme.name, "Kidney")
            self.assertEqual(store.active_id, DEFAULT_ID)
            loaded = store.load_bgr(theme.id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.shape[1] / loaded.shape[0], 90 / 160)
            thumb = store.load_thumb_bgr(theme.id)
            self.assertIsNotNone(thumb)
            store.set_active(theme.id)
            self.assertEqual(store.active_id, theme.id)
            again = WallpaperStore(Path(tmp))
            self.assertEqual(again.active_id, theme.id)
            self.assertEqual(again.themes[0].name, "Kidney")
            self.assertTrue(again.delete_theme(theme.id))
            self.assertEqual(again.active_id, DEFAULT_ID)
            self.assertEqual(len(again.themes), 0)
            self.assertFalse((Path(tmp) / theme.image).exists())
            self.assertFalse((Path(tmp) / theme.thumb).exists())
            self.assertEqual(list(Path(tmp).glob("wp_*")), [])


if __name__ == "__main__":
    unittest.main()
