#!/usr/bin/env python3
"""Emit wbc_build_id.hpp without importing the full Python control stack."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_NATIVE = Path(__file__).resolve().parent
_PKG = _NATIVE.parents[1] / "rm75_control" / "control" / "joint_admittance_8dof" / "wbc_rt"
_SOURCES = (
    _NATIVE / "include" / "wbc_rt" / "protocol.hpp",
    _NATIVE / "include" / "wbc_rt" / "inner.hpp",
    _NATIVE / "include" / "wbc_rt" / "types.hpp",
    _NATIVE / "src" / "inner.cpp",
    _NATIVE / "src" / "main.cpp",
    _PKG / "protocol.py",
)


def combined_hash() -> str:
    items = {}
    root = _NATIVE.parents[1]
    for path in _SOURCES:
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = path.name
        items[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    blob = "".join(f"{k}={v}\n" for k, v in sorted(items.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


dest = Path(sys.argv[1] if len(sys.argv) > 1 else "wbc_build_id.hpp")
dest.write_text(
    f'#pragma once\n#define WBC_SRC_HASH "{combined_hash()}"\n',
    encoding="utf-8",
)
