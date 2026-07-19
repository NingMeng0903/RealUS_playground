"""Resolve capability-map directories."""

from __future__ import annotations

from pathlib import Path


def resolve_map_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        raise NotADirectoryError(p)
    if not (p / "manifest.yaml").exists():
        raise FileNotFoundError(f"missing manifest.yaml under {p}")
    return p
