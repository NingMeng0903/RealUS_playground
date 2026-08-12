from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.posture_planner import (
    PlanComputation,
    PlannerStatus,
    PosturePlanner,
    PosturePlanningRequest,
)


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_submit_is_nonblocking_and_obsolete_result_is_not_published() -> None:
    release = threading.Event()
    entered = threading.Event()

    def calculate(value: int) -> np.ndarray:
        if value == 1:
            entered.set()
            assert release.wait(1.0)
        return np.array([value], dtype=float)

    planner = PosturePlanner(calculate, stale_after_s=10.0)
    try:
        t0 = time.monotonic()
        first = planner.submit(1)
        assert time.monotonic() - t0 < 0.05
        assert entered.wait(1.0)
        second = planner.submit(2)
        release.set()
        assert planner.wait_for(second, timeout_s=1.0)
        latest = planner.latest()
        assert latest.sequence == second
        assert latest.value is not None
        assert latest.value.tolist() == [2.0]
        assert first < second
    finally:
        planner.shutdown()


def test_latest_value_is_isolated_and_numpy_arrays_are_read_only() -> None:
    planner = PosturePlanner(lambda _: {"q": np.arange(3.0)}, stale_after_s=10.0)
    try:
        sequence = planner.submit(object())
        assert planner.wait_for(sequence)
        snap = planner.latest()
        assert snap.status is PlannerStatus.VALID
        with pytest.raises(TypeError):
            snap.value["new"] = 1  # type: ignore[index]
        with pytest.raises(ValueError):
            snap.value["q"][0] = 99.0  # type: ignore[index]
        assert planner.latest().value["q"].tolist() == [0.0, 1.0, 2.0]  # type: ignore[index]
    finally:
        planner.shutdown()


def test_stale_and_invalid_results_fade_last_good_value_smoothly() -> None:
    clock = FakeClock()

    def calculate(value: int):
        if value < 0:
            return PlanComputation.invalid("synthetic rejection")
        return np.array([value])

    planner = PosturePlanner(
        calculate,
        stale_after_s=1.0,
        fade_after_s=2.0,
        clock=clock,
    )
    try:
        sequence = planner.submit(7)
        assert planner.wait_for(sequence)
        clock.value = 2.0
        stale = planner.latest()
        assert stale.status is PlannerStatus.STALE
        assert stale.confidence == pytest.approx(0.5)
        assert stale.value.tolist() == [7]

        sequence = planner.submit(-1)
        assert planner.wait_for(sequence)
        invalid0 = planner.latest()
        assert invalid0.status is PlannerStatus.INVALID
        assert invalid0.value.tolist() == [7]
        assert invalid0.confidence == pytest.approx(0.5)
        clock.value = 2.5
        invalid1 = planner.latest()
        assert invalid1.status is PlannerStatus.INVALID
        assert 0.0 < invalid1.confidence < invalid0.confidence
    finally:
        planner.shutdown()


def test_shutdown_is_idempotent_and_rejects_new_work() -> None:
    planner = PosturePlanner(lambda x: x)
    assert planner.shutdown()
    assert planner.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        planner.submit(1)


def test_generic_submit_packages_state_reference_and_optional_horizon() -> None:
    seen = []

    def calculate(request):
        seen.append(request)
        return "guide"

    planner = PosturePlanner(calculate)
    try:
        sequence = planner.submit(
            robot_state={"q": [0.0]},
            current_task_reference={"twist": [0.0]},
            optional_reference_horizon="future",
        )
        assert planner.wait_for(sequence, timeout_s=1.0)
        assert isinstance(seen[0], PosturePlanningRequest)
        assert seen[0].optional_reference_horizon == "future"
    finally:
        planner.shutdown()
