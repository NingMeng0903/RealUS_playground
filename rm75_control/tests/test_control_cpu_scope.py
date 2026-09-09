"""Repeated controller runs must not pin later background children to the loop CPU."""
import json
import os
import subprocess
import sys

import pytest

from rm75_control.control.joint_admittance_8dof import loop


@pytest.mark.parametrize("fail_in_run", [False, True])
def test_run_restores_background_mask_before_spawning_next_child(fail_in_run):
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux CPU affinity required")
    original = set(os.sched_getaffinity(0))
    candidates = sorted(original - {2, 3, 4, 5})
    if len(candidates) < 3:
        pytest.skip("three non-production test CPUs required")
    background, control = set(candidates[:2]), candidates[2]
    try:
        os.sched_setaffinity(0, background)
        for _ in range(2):
            try:
                with loop._control_cpu_scope(control) as pinned:
                    assert pinned
                    assert set(os.sched_getaffinity(0)) == {control}
                    if fail_in_run:
                        raise RuntimeError("fake runner failure")
            except RuntimeError as exc:
                assert str(exc) == "fake runner failure"
            assert set(os.sched_getaffinity(0)) == background
            child = subprocess.check_output(
                [sys.executable, "-c",
                 "import json, os; print(json.dumps(sorted(os.sched_getaffinity(0))))"],
                text=True, timeout=5,
            )
            assert set(json.loads(child)) == background
    finally:
        os.sched_setaffinity(0, original)


def test_no_pin_when_incoming_affinity_cannot_be_saved(monkeypatch):
    def unavailable(_pid):
        raise OSError("affinity unavailable")

    monkeypatch.setattr(loop.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(
        loop.os, "sched_setaffinity",
        lambda *_: pytest.fail("cannot leave an unrestorable control pin"),
    )
    with loop._control_cpu_scope(6) as pinned:
        assert not pinned
