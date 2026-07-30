"""In-place progress bar: single-line ``\\r`` rewrite on stdout."""

from __future__ import annotations

import sys

_last_filled: dict[str, int] = {}
_width_pad = 72


def stage_progress(label: str, step: int, total: int, *, width: int = 36) -> None:
    total = max(int(total), 1)
    step = min(max(int(step), 0), total)
    pct = int(100 * step / total)
    filled = int(width * step / total)
    done = step >= total

    if not done and filled == _last_filled.get(label, -1):
        return
    _last_filled[label] = filled

    bar = "#" * filled + "-" * (width - filled)
    line = f"  {label} [{bar}] {pct:3d}%"
    # Pad so a shorter line fully overwrites a longer previous one.
    payload = f"\r{line:<{_width_pad}}"
    if done:
        payload += "\n"
        _last_filled.pop(label, None)
    sys.stdout.write(payload)
    sys.stdout.flush()


def close_progress() -> None:
    _last_filled.clear()
    # If a bar was mid-line, finish the row.
    sys.stdout.write("\n")
    sys.stdout.flush()
