from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from multicam_calib.devices.orbbec import (
    _FIRST_GEN_PIDS,
    _params_from_v1,
    _prefer_v4l2,
    _resolve_sdk_root,
    _sdk_mismatch_hint,
    _sdk_no_match,
    diagnose_orbbec_usb,
    list_orbbec_v4l_nodes,
    pinhole_guess_v4l,
)
from multicam_calib.io.config import load_orbbec


class TestSdkMatchHints(unittest.TestCase):
    def test_no_device_is_mismatch(self) -> None:
        self.assertTrue(_sdk_no_match(RuntimeError("No device found")))
        self.assertTrue(_sdk_no_match(RuntimeError("Orbbec serial 'x' not found (0 device(s))")))
        self.assertFalse(_sdk_no_match(RuntimeError("libusb Access denied")))

    def test_first_gen_hint_names_v4l_not_udev(self) -> None:
        hint = _sdk_mismatch_hint(["0614", "0511"])
        self.assertIn("第一代", hint)
        self.assertIn("V4L2", hint)
        self.assertNotIn("udev 后拔插", hint)
        self.assertTrue(_FIRST_GEN_PIDS & {"0511", "0614"})

    def test_other_pid_does_not_claim_gemini_f(self) -> None:
        hint = _sdk_mismatch_hint(["0635"])
        self.assertNotIn("Astra 3D Camera", hint)


class TestPinholeGuess(unittest.TestCase):
    def test_size_and_source(self) -> None:
        model = pinhole_guess_v4l(640, 480)
        self.assertEqual(model.image_size, (640, 480))
        self.assertEqual(model.source, "v4l2_guess")
        self.assertAlmostEqual(model.K[0, 2], 320.0)
        self.assertAlmostEqual(model.K[1, 2], 240.0)
        self.assertGreater(model.K[0, 0], 100.0)
        np.testing.assert_allclose(model.dist, 0.0)


class TestV4lListing(unittest.TestCase):
    def test_diagnose_mentions_first_gen_when_usb_present(self) -> None:
        text = diagnose_orbbec_usb()
        if "PID 0614" not in text and "PID 0511" not in text:
            self.skipTest("Orbbec USB not attached")
        self.assertIn("write=ok", text)
        self.assertIn("V4L", text)
        self.assertNotIn("装 udev 后拔插", text)

    def test_list_rgb_index0(self) -> None:
        nodes = list_orbbec_v4l_nodes()
        if not nodes:
            self.skipTest("no Orbbec V4L nodes")
        idx0 = [n for n in nodes if n.get("index") == "0"]
        self.assertTrue(idx0)
        self.assertTrue(idx0[0]["node"].startswith("/dev/video"))
        self.assertTrue(_prefer_v4l2())


class TestV1Params(unittest.TestCase):
    def test_params_from_factory_dict(self) -> None:
        raw = {
            "serial": "AY2MC31016E",
            "model": "SV1301S_U3",
            "color": {
                "K": [[456.0, 0.0, 328.0], [0.0, 456.0, 246.0], [0.0, 0.0, 1.0]],
                "dist": [0.05, -0.06, 0.0, 0.0, 0.0],
                "image_size": [640, 480],
                "source": "factory",
            },
            "T_color_depth": np.eye(4).tolist(),
        }
        p = _params_from_v1(raw)
        self.assertEqual(p.serial, "AY2MC31016E")
        self.assertAlmostEqual(p.color.K[0, 0], 456.0)
        self.assertEqual(p.color.source, "factory")
        self.assertIsNotNone(p.T_color_depth)

    def test_sdk_root_and_v1_python_exist(self) -> None:
        cfg = load_orbbec()
        root = _resolve_sdk_root(cfg)
        self.assertTrue(root.exists(), msg=str(root))
        self.assertTrue(Path(cfg.v1_python).is_file(), msg=cfg.v1_python)


if __name__ == "__main__":
    unittest.main()
