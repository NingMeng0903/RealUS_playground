"""Import ``rm75_control.tools.reachability.*`` without executing package ``__init__``.

Top-level ``rm75_control`` pulls ``Robotic_Arm``; IRD playground only needs map IO + viz.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_rm75_namespace(rm75_control_root: str | Path | None = None) -> Path:
    """Register a namespace package for ``rm75_control`` if not already importable cleanly.

    Returns the inner package directory ``.../rm75_control/rm75_control``.
    """
    if rm75_control_root is None:
        env = Path(__file__).resolve().parents[3] / "rm75_control"
        rm75_control_root = Path(
            __import__("os").environ.get("RM75_CONTROL_ROOT", str(env))
        )
    root = Path(rm75_control_root).resolve()
    pkg_dir = root / "rm75_control"
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"rm75_control package dir missing: {pkg_dir}")

    existing = sys.modules.get("rm75_control")
    if existing is not None and getattr(existing, "__path__", None):
        return pkg_dir

    # Prefer namespace stub over the real __init__ (Robotic_Arm).
    pkg = types.ModuleType("rm75_control")
    pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    sys.modules["rm75_control"] = pkg
    return pkg_dir
