"""The standalone ultrasound entrypoint prepares observer resources first."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


_RUN_UI = Path(__file__).resolve().parents[1] / "scripts" / "run_ui.py"


def _load_run_ui():
    # The script intentionally re-execs once to disable the user site.  The
    # test imports it in-process, so provide that already-established state.
    with patch.dict(os.environ, {"PYTHONNOUSERSITE": "1"}):
        spec = importlib.util.spec_from_file_location("test_run_ui", _RUN_UI)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


class TestObserverBootstrap(unittest.TestCase):
    def test_help_prepares_resources_without_starting_camera(self) -> None:
        module = _load_run_ui()
        events: list[str] = []

        with patch.object(
            module, "_prepare_observer_process", lambda: events.append("prepare")
        ), self.assertRaises(SystemExit):
            module.main(["--help"])

        self.assertEqual(events, ["prepare"])
        # ``--help`` exits during argument parsing, before config/UI/runtime
        # imports, so a camera worker cannot have been started by this path.
        source = _RUN_UI.read_text(encoding="utf-8")
        main_source = source[source.index("def main(") :]
        prepare_idx = main_source.index("_prepare_observer_process()")
        self.assertLess(prepare_idx, main_source.index("from us_framegrab.config"))
        self.assertLess(prepare_idx, main_source.index("from PyQt5"))
        self.assertLess(prepare_idx, main_source.index("from us_framegrab.runtime"))


if __name__ == "__main__":
    unittest.main()
