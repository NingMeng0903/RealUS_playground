"""Shared thread pool for CPU-bound multiview preprocessing.

cv2/numpy resize, crop and transpose release the GIL, so mapping per-view
preprocessing across threads gives near-linear speedup for synchronized views.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_MAX_WORKERS = int(os.environ.get("DWPOSE_PREPROCESS_THREADS", "8"))
_EXECUTOR: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, _MAX_WORKERS), thread_name_prefix="mv-pre")
    return _EXECUTOR


def thread_map(fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
    """Apply ``fn`` to each item across threads, preserving input order.

    Falls back to a serial loop for 0/1 items to avoid pool overhead.
    """
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    return list(_executor().map(fn, items))


__all__ = ["thread_map"]
