"""Process entry point for the non real-time joint-admittance logger.

This module deliberately keeps the process bootstrap tiny.  The formatter
remains on ``_TickLogger`` so the CSV schema has one source of truth; the
child creates a formatter-only instance without starting another child.
"""

from __future__ import annotations


class _RawFlag:
    """Lock-free one-bit status shared with the logger sidecar."""

    def __init__(self, raw_value) -> None:
        self._raw_value = raw_value

    def set(self) -> None:
        self._raw_value.value = 1

    def is_set(self) -> bool:
        return bool(self._raw_value.value)


def run_tick_logger_process(
    path,
    request_queue,
    stop_event,
    failed_event,
    dropped_count,
    header,
    flush_s,
    file_buffer,
    verbose_json,
    phi_source,
    phi_sha8,
) -> None:
    """Run the CSV/JSON formatter and file sink in a spawned process."""

    from ..admittance_common.observer_runtime import prepare_observer_process
    prepare_observer_process()
    # Formatting imports and work belong to this lower-priority process.
    from .loop import _TickLogger

    worker = _TickLogger.__new__(_TickLogger)
    worker._q = request_queue
    worker._stop = stop_event
    worker._closed = stop_event
    worker._failed = failed_event
    worker._dropped_shared = dropped_count
    worker._dropped_local = 0
    worker._is_child = True
    worker._verbose_json = bool(verbose_json)
    worker._phi_source = str(phi_source or "")
    worker._phi_sha8 = str(phi_sha8 or "")
    worker._prev_arm_send_ns = 0
    worker._prev_q_send_arm = None
    worker.dropped = 0
    worker._HEADER = header
    worker._FLUSH_S = float(flush_s)
    worker._FILE_BUFFER = int(file_buffer)
    worker._run(path)
