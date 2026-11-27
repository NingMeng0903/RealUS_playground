"""Persistent CPU parallelism for independent body/hand triangulation parts."""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import weakref
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np


def _worker_initializer() -> None:
    """Restore the RealUS package path before ProcessPool unpickling.

    The official HybrIK loader prepends a bundled ``pytorch3d`` directory that
    also contains a top-level ``projects`` package.  A broad pytest process
    can inherit that path ordering, and ProcessPool unpickles
    ``projects.genesis_ue_sync`` before the task wrapper runs.  Fix the path
    and discard only a shadowing ``projects`` module during worker startup.
    """
    src_root = Path(__file__).resolve().parents[4]
    src_text = str(src_root)
    sys.path[:] = [entry for entry in sys.path if entry != src_text]
    sys.path.insert(0, src_text)
    projects = sys.modules.get("projects")
    if projects is not None:
        locations = [str(path) for path in getattr(projects, "__path__", ())]
        if not any(location.startswith(src_text) for location in locations):
            sys.modules.pop("projects", None)


def _warmup_worker() -> bool:
    """Import the CPU triangulation helper before the first timed frame."""
    from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
        _triangulate_part,
    )

    return callable(_triangulate_part)


def _compact_part_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    """Keep the same selected-hypothesis diagnostics as no-debug persistence.

    Compact before sending the result through the process pipe.  Selection
    and geometry gates have already run with every candidate present.
    """
    compact = dict(diag)
    details = compact.get("joint_details")
    if isinstance(details, list):
        compact_details = []
        for raw_detail in details:
            if not isinstance(raw_detail, dict):
                compact_details.append(raw_detail)
                continue
            detail = dict(raw_detail)
            candidates = detail.get("candidate_hypotheses")
            if isinstance(candidates, list):
                selected = [candidate for candidate in candidates
                            if isinstance(candidate, dict) and candidate.get("selected")]
                if selected:
                    detail["candidate_hypotheses"] = selected
                else:
                    detail.pop("candidate_hypotheses", None)
            compact_details.append(detail)
        compact["joint_details"] = compact_details
    return compact


def _triangulate_part_worker(
    payload: tuple[Any, ...], compact_diagnostics: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Spawn-safe wrapper around the existing exact triangulation helper."""
    from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
        _triangulate_part,
    )

    points, diagnostics = _triangulate_part(*payload)
    if compact_diagnostics:
        diagnostics = _compact_part_diagnostics(diagnostics)
    return points, diagnostics


class BodyHandTriangulationJob:
    """One submitted frame, with CPU timing independent of result collection."""

    def __init__(
        self,
        futures: dict[str, Future],
        diagnostics: dict[str, Any] | None,
        started: float,
    ):
        self._futures = futures
        self._diagnostics = diagnostics
        self._started = started
        self._completion_lock = Lock()
        self._completed = Event()
        self._remaining = len(futures)
        self._elapsed_s: float | None = None
        job_ref = weakref.ref(self)

        def completed(future: Future) -> None:
            # Future retains callbacks after completion.  Avoid retaining
            # this job and all frame diagnostics in a callback reference cycle.
            job = job_ref()
            if job is not None:
                job._part_completed(future)

        for future in futures.values():
            future.add_done_callback(completed)

    def _part_completed(self, _future: Future) -> None:
        # ProcessPool marks a future complete only after receiving and
        # unpickling its result, so this includes worker result IPC.
        finished = time.perf_counter()
        with self._completion_lock:
            elapsed = finished - self._started
            self._elapsed_s = max(self._elapsed_s or 0.0, elapsed)
            self._remaining -= 1
            if self._remaining == 0:
                self._completed.set()

    @property
    def elapsed_s(self) -> float | None:
        """Submission-to-completion wall time; available after all parts finish."""
        return self._elapsed_s if self._completed.is_set() else None

    def result(self) -> dict[str, np.ndarray]:
        parts = {field: future.result() for field, future in self._futures.items()}
        # A waiter can wake just before Future invokes its callback.  Waiting
        # for that callback prevents a transient missing elapsed_s value.
        self._completed.wait()
        body, body_diag = parts["body25"]
        empty_hand = np.zeros((21, 4), dtype=np.float32)
        handl, handl_diag = parts.get("handl", (empty_hand.copy(), {"disabled": True}))
        handr, handr_diag = parts.get("handr", (empty_hand.copy(), {"disabled": True}))
        if self._diagnostics is not None:
            self._diagnostics.update({"body25": body_diag, "handl": handl_diag, "handr": handr_diag})
        return {"keypoints3d": body, "handl3d": handl, "handr3d": handr}


class BodyHandTriangulator:
    """Reuse spawned CPU workers for Body25 and two hand triangulations.

    The worker count is deliberately small because each part already uses
    NumPy/SVD internally.  ``spawn`` keeps this pool safe to use after CUDA
    has been initialized in the parent process.
    """

    def __init__(self, workers: int = 3):
        workers = int(workers)
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self._executor: ProcessPoolExecutor | None = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            # Apply the same path repair in the parent before spawn inherits
            # its import state; this also lets the child import the
            # initializer module itself when another test prepended a
            # shadowing pytorch3d/projects directory.
            _worker_initializer()
            context = mp.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=context,
                initializer=_worker_initializer,
            )
        return self._executor

    def preload(self) -> "BodyHandTriangulator":
        """Start all workers and wait for one short CPU task from each."""
        executor = self._ensure_executor()
        futures = [executor.submit(_warmup_worker) for _ in range(self.workers)]
        for future in futures:
            future.result()
        return self

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "BodyHandTriangulator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def triangulate(
        self,
        annots_by_cam: dict[str, dict[str, np.ndarray]],
        camera_ids: list[str],
        P_all: np.ndarray,
        tri_cfg: Any | None,
        include_hands: bool = True,
        diagnostics: dict[str, Any] | None = None,
        observation_meta_by_cam: dict[str, dict[str, Any]] | None = None,
        compact_diagnostics: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return the same parts and diagnostics as the serial helper."""
        return self.submit(
            annots_by_cam, camera_ids, P_all, tri_cfg,
            include_hands=include_hands, diagnostics=diagnostics,
            observation_meta_by_cam=observation_meta_by_cam,
            compact_diagnostics=compact_diagnostics,
        ).result()

    def submit(
        self,
        annots_by_cam: dict[str, dict[str, np.ndarray]],
        camera_ids: list[str],
        P_all: np.ndarray,
        tri_cfg: Any | None,
        include_hands: bool = True,
        diagnostics: dict[str, Any] | None = None,
        observation_meta_by_cam: dict[str, dict[str, Any]] | None = None,
        compact_diagnostics: bool = False,
    ) -> BodyHandTriangulationJob:
        """Submit CPU parts without waiting, allowing independent GPU work."""
        started = time.perf_counter()
        if tri_cfg is None:
            from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import (
                TriangulationConfig,
            )

            tri_cfg = TriangulationConfig()

        executor = self._ensure_executor()
        common = (annots_by_cam, camera_ids, P_all, tri_cfg, observation_meta_by_cam)
        fields = ["keypoints", "handl2d", "handr2d"]
        futures = {"body25": executor.submit(
            _triangulate_part_worker,
            (common[0], common[1], common[2], fields[0], common[3], common[4]),
            compact_diagnostics,
        )}
        if include_hands:
            futures["handl"] = executor.submit(
                _triangulate_part_worker,
                (common[0], common[1], common[2], fields[1], common[3], common[4]),
                compact_diagnostics,
            )
            futures["handr"] = executor.submit(
                _triangulate_part_worker,
                (common[0], common[1], common[2], fields[2], common[3], common[4]),
                compact_diagnostics,
            )
        return BodyHandTriangulationJob(futures, diagnostics, started)
