"""Shared-memory helpers: avoid resource_tracker unlink warnings on exit."""

from __future__ import annotations

from multiprocessing import resource_tracker, shared_memory


def _patch_shm_resource_tracker() -> None:
    """Do not track shared_memory in the stdlib tracker (we manage lifecycle explicitly)."""
    if getattr(resource_tracker, "_rm75_no_track_shm", False):
        return
    _orig_register = resource_tracker.register
    _orig_unregister = resource_tracker.unregister

    def register(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_unregister(name, rtype)

    resource_tracker.register = register
    resource_tracker.unregister = unregister
    resource_tracker._rm75_no_track_shm = True


_patch_shm_resource_tracker()


def _posix_unlink(name: str) -> None:
    try:
        from multiprocessing.shared_memory import _posixshmem

        _posixshmem.shm_unlink(name)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _unregister(name: str) -> None:
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass


def create_named_shm(name: str, size: int) -> shared_memory.SharedMemory:
    """Create (or replace) a named segment; owner must call close_named_shm to destroy."""
    try:
        old = attach_named_shm(name)
        old.close()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    _posix_unlink(name)
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    _unregister(shm._name)
    return shm


def attach_named_shm(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment; unregister so process exit does not unlink it."""
    shm = shared_memory.SharedMemory(name=name, create=False)
    _unregister(shm._name)
    return shm


def close_attached_shm(shm: shared_memory.SharedMemory | None) -> None:
    """Close a subscriber handle without destroying the segment.

    Callers must drop every ``numpy`` / ``memoryview`` that referenced
    ``shm.buf`` *before* calling this; otherwise CPython raises
    ``BufferError: cannot close exported pointers exist`` (also seen in
    ``SharedMemory.__del__`` during exception teardown).
    """
    if shm is None:
        return
    try:
        shm.close()
    except (OSError, BufferError):
        pass


def close_named_shm(shm: shared_memory.SharedMemory | None) -> None:
    """Close and destroy a segment created by this process (via create_named_shm)."""
    if shm is None:
        return
    name = getattr(shm, "_name", None)
    try:
        shm.close()
    except (OSError, BufferError):
        pass
    if name:
        _posix_unlink(name)
