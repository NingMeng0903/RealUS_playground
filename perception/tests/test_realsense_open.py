from __future__ import annotations

import unittest

from realsense_open import usb_is_superspeed


class TestUsbIsSuperspeed(unittest.TestCase):
    def test_usb3_and_4(self) -> None:
        self.assertTrue(usb_is_superspeed("3.2"))
        self.assertTrue(usb_is_superspeed("3.0"))
        self.assertTrue(usb_is_superspeed("4"))

    def test_usb2_is_not(self) -> None:
        self.assertFalse(usb_is_superspeed("2.1"))
        self.assertFalse(usb_is_superspeed("2.0"))
        self.assertFalse(usb_is_superspeed("?"))


if __name__ == "__main__":
    unittest.main()
