"""Machine crop presets load into the session and YAML."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from us_framegrab.config import FrameGrabConfig  # noqa: E402
from us_framegrab.presets import SONOSCAPE_E2, get_preset, list_presets  # noqa: E402
from us_framegrab.runtime import FrameGrabSession  # noqa: E402


class TestPresets(unittest.TestCase):
    def test_sonoscape_e2_is_the_only_entry(self) -> None:
        names = [(p.id, p.name) for p in list_presets()]
        self.assertEqual(names, [("sonoscape_e2", "SonoScape E2")])
        self.assertEqual(get_preset("SonoScape E2"), SONOSCAPE_E2)
        self.assertIsNone(get_preset("missing"))

    def test_apply_loads_search_box_crop_and_resolution(self) -> None:
        cfg = FrameGrabConfig(
            path=Path("."),
            machine="other",
            init_cbox=[0, 100, 0, 100],
            final_cbox=[10, 20, 10, 20],
            frame_width=640,
            frame_height=480,
            compressed_quality=50,
            hflip=True,
            color=True,
        )
        session = FrameGrabSession(cfg, auto_crop_on_startup=False)
        self.assertTrue(session.apply_machine_preset("sonoscape_e2"))
        snap = session.snapshot()
        self.assertEqual(cfg.machine, "sonoscape_e2")
        self.assertEqual(cfg.init_cbox, [550, 1650, 150, 920])
        self.assertEqual(snap.cbox, [559, 1611, 115, 920])
        self.assertEqual(snap.image_size, (1920, 1080))
        self.assertEqual(snap.jpeg_quality, 80)
        self.assertFalse(snap.hflip)
        self.assertFalse(snap.color)

    def test_apply_unknown_is_false(self) -> None:
        session = FrameGrabSession(FrameGrabConfig(path=Path(".")), auto_crop_on_startup=False)
        before = session.snapshot().cbox
        self.assertFalse(session.apply_machine_preset("nope"))
        self.assertEqual(session.snapshot().cbox, before)

    def test_yaml_roundtrip_keeps_machine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            cfg = FrameGrabConfig(path=path, machine="sonoscape_e2")
            cfg.save()
            text = path.read_text(encoding="utf-8")
            self.assertIn("machine: sonoscape_e2", text)


if __name__ == "__main__":
    unittest.main()
