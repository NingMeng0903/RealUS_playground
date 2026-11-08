"""Focused causal-contract tests for ``core_execution``.

The fake world keeps these tests fast and makes the information channels
explicit.  The final test instantiates the real native SMPL-X world as a small
geometry smoke check; no robot or B-mode claim is made here.
"""

from __future__ import annotations

from types import SimpleNamespace as Namespace

import numpy as np
import pytest

from leg_program_bench.core_execution import (
    BundleExecutor,
    CoreWorld,
    ProgramExecutor,
    SPANS,
    Tree,
    execute_tree,
    predict_tree,
)
from leg_program_bench.simulation import ServoState, Spec


class FakeAtlas:
    centerline_points = np.asarray([[s, 0.0, 0.0] for s in np.linspace(0.0, 1.0, 101)])
    centerline_progress = np.linspace(0.0, 1.0, 101)


class FakeWorld:
    """Minimal public plant for wrapper and bundle tests."""

    def __init__(self) -> None:
        self.profile = {"lag": 0.08}
        self.level = "fake"
        self.spec = Spec("fake-object", 1, 1, -1)
        self.atlas = FakeAtlas()
        self.digest = "fake-world-digest"
        self.transition_cache = {}
        self.core_transition_cache = {}

    @property
    def true_points(self):
        raise AssertionError("predict_tree must not read hidden true_points")

    def new_state(self, s: float, mode: int = 0) -> ServoState:
        return ServoState(
            position=np.asarray([float(s), 0.01 * float(mode), 0.0]),
            rotation=np.eye(3),
            time=0.0,
            velocity=np.zeros(3),
        )

    def reference_at(self, s: float, mode: int = 0):
        return np.asarray([float(s), 0.01 * float(mode), 0.0]), np.eye(3)

    def public_context(self, start: float, end: float, mode: int):
        return {"path_length": abs(float(end) - float(start)), "path_turn": 0.0}


def _bundle(near: float) -> dict:
    programs = {}
    for name in ("a", "b", "f", "probe"):
        for mode in (-1, 0, 1):
            # Make the mode choice observable in duration, while keeping every
            # public row valid and the f/probe records fresh-capable.
            duration = 1.0 + (0.6 if mode == 1 else 0.0)
            programs[f"{name}:{mode}"] = {
                "program": name,
                "mode": mode,
                "success": True,
                "public_success": True,
                "independent_success": True,
                "service_duration": duration,
                # Keep the nominal reference fixed so the measured service
                # duration remains observable through BundleExecutor's
                # nominal-duration rescaling.
                "nominal_duration": 1.0,
                "exit_position_local": [0.0, 0.0, 0.0],
                "exit_rotation_local": [0.0, 0.0, 0.0],
                "exit_velocity_local": [0.0, 0.0, 0.0],
                "frame_count": 1,
                "fresh_frames": 1 if name == "f" else 0,
            }
    return {"object_id": "fake-object", "near": near, "probe": near, "programs": programs}


@pytest.fixture
def instant_move(monkeypatch):
    """Make only entry motion instantaneous; program wrapper logic remains real."""

    def move(self, start, mode, max_time=20.0):
        target, rotation = self.world.reference_at(start, mode)
        self.state.position = target.copy()
        self.state.rotation = rotation.copy()
        return True

    monkeypatch.setattr(ProgramExecutor, "move", move)


def test_late_release_does_not_insert_an_automatic_delay_penalty(monkeypatch):
    calls = []

    def fake_call(self, name, mode=0, *, fresh_after=0.0):
        # This is a causal wrapper fake: the first selected mode is allowed to
        # fail and execute_tree must issue a real recovery call.
        before = self.state.time
        self.state.time += 0.2
        success = not (name == "a" and mode == -1)
        row = {
            "program": name,
            "mode": mode,
            "success": success,
            "public_success": success,
            "independent_success": success,
            "duration": 0.2,
            "service_duration": 0.2,
            "fresh_frames": 1 if name == "f" else 0,
            "frame_count": 1,
            "nominal_duration": 0.2,
            "fresh_after_seen": fresh_after,
            "start_time": before,
            "end_time": self.state.time,
        }
        self.calls.append(row)
        calls.append(row)
        return row, []

    monkeypatch.setattr(ProgramExecutor, "call", fake_call)
    monkeypatch.setattr(ProgramExecutor, "wait", lambda self, duration: setattr(self.state, "time", self.state.time + duration))
    world = FakeWorld()
    tree = Tree("react_test", "react", default_mode=-1, near_mapping=1, recover=True)

    early = execute_tree(world, 1.0, tree, release=0.0, hidden=False, chain=("a", "f"), seed=3)
    calls.clear()
    late = execute_tree(world, 1.0, tree, release=0.8, hidden=False, chain=("a", "f"), seed=3)

    # The react tree scans immediately while the packet is still hidden behind
    # the gate; release is not implemented as duration += delay.
    assert late["duration"] > early["duration"]
    assert late["duration"] < early["duration"] + 0.8
    assert late["decisions"][0]["near_visible"] is False
    assert late["success"]


def test_fresh_capture_time_is_passed_to_recheck_and_old_packet_is_not_fresh(monkeypatch):
    seen = []

    def fake_call(self, name, mode=0, *, fresh_after=0.0):
        before = self.state.time
        self.state.time += 0.1
        seen.append((name, fresh_after, before))
        row = {
            "program": name,
            "mode": mode,
            "success": True,
            "public_success": True,
            "independent_success": True,
            "duration": 0.1,
            "service_duration": 0.1,
            "fresh_frames": 1 if name == "f" else 0,
            "frame_count": 1,
            "nominal_duration": 0.1,
        }
        self.calls.append(row)
        return row, []

    monkeypatch.setattr(ProgramExecutor, "call", fake_call)
    world = FakeWorld()
    tree = Tree("fixed_plus", "fixed", default_mode=1, recover=True)
    result = execute_tree(world, 1.0, tree, release=0.0, hidden=False, chain=("a", "f"), seed=4)
    assert result["success"]
    f_entries = [row for row in seen if row[0] == "f"]
    assert len(f_entries) == 1
    assert f_entries[0][1] == pytest.approx(f_entries[0][2])
    assert result["fresh_frames"] == 1


def test_hidden_prediction_does_not_react_to_bundle_near_or_true_points(monkeypatch):
    monkeypatch.setattr(
        ProgramExecutor,
        "move",
        lambda self, start, mode, max_time=20.0: True,
    )
    world = FakeWorld()
    tree = Tree("react_test", "react", default_mode=-1, near_mapping=1, recover=True)
    positive = predict_tree(world, _bundle(1.0), tree, hidden=True, chain=("a", "b", "f"))
    negative = predict_tree(world, _bundle(-1.0), tree, hidden=True, chain=("a", "b", "f"))
    assert positive == negative

    visible_positive = predict_tree(world, _bundle(1.0), tree, hidden=False, chain=("a", "b", "f"))
    visible_negative = predict_tree(world, _bundle(-1.0), tree, hidden=False, chain=("a", "b", "f"))
    assert visible_positive["duration"] != visible_negative["duration"]


def test_bundle_executor_preserves_one_object_bundle_across_calls(monkeypatch):
    monkeypatch.setattr(ProgramExecutor, "move", lambda self, start, mode, max_time=20.0: True)
    world = FakeWorld()
    bundle = _bundle(1.0)
    executor = BundleExecutor(world, bundle)
    first = executor.call("a", 1)
    second = executor.call("b", 1)
    assert executor.bundle is bundle
    assert first["program"] == "a"
    assert second["program"] == "b"
    assert executor.source_s == pytest.approx(SPANS["b"][1])
    assert executor.exit["program"] == "b"
    assert len(executor.calls) == 2


def test_failed_branch_recovery_is_a_real_second_program_call(monkeypatch):
    calls = []

    def fake_call(self, name, mode=0, *, fresh_after=0.0):
        self.state.time += 0.1
        success = not (name == "a" and mode == -1)
        row = {
            "program": name,
            "mode": mode,
            "success": success,
            "public_success": success,
            "independent_success": success,
            "duration": 0.1,
            "service_duration": 0.1,
            "fresh_frames": 1 if name == "f" else 0,
            "frame_count": 1,
            "nominal_duration": 0.1,
        }
        self.calls.append(row)
        calls.append((name, mode))
        return row, []

    monkeypatch.setattr(ProgramExecutor, "call", fake_call)
    world = FakeWorld()
    tree = Tree("fixed_minus", "fixed", default_mode=-1, recover=True)
    result = execute_tree(world, 1.0, tree, release=0.0, hidden=False, chain=("a", "f"), seed=5)
    assert result["success"]
    assert calls[:3] == [("a", -1), ("a", 1), ("f", 0)]
    assert result["recoveries"] == 1
    assert result["program_calls"] == 3


def _audited_fake_row(
    name: str,
    mode: int,
    *,
    public_success: bool,
    independent_success: bool,
    duration: float = 0.1,
) -> dict:
    """Return both public feedback and private scoring labels explicitly.

    The regression tests below intentionally make these labels disagree.  A
    policy may branch only on ``public_success``; the independent/audit label
    is used for the final score and must not alter the executed trace.
    """

    return {
        "program": name,
        "mode": mode,
        "success": bool(independent_success),
        "public_success": bool(public_success),
        "independent_success": bool(independent_success),
        "status": "completed" if public_success else "coverage_gap",
        "duration": duration,
        "service_duration": duration,
        "fresh_frames": 1 if name == "f" else 0,
        "frame_count": 1,
        "nominal_duration": duration,
    }


def test_private_audit_label_does_not_control_execute_trace(monkeypatch):
    """Flipping independent scoring must not change action, timing, or branch."""

    def run_with_audit(independent_success: bool):
        seen = []

        def fake_call(self, name, mode=0, *, fresh_after=0.0):
            self.state.time += 0.1
            row = _audited_fake_row(
                name,
                mode,
                public_success=True,
                independent_success=(independent_success if name == "a" else True),
            )
            self.calls.append(row)
            seen.append((name, mode))
            return row, []

        monkeypatch.setattr(ProgramExecutor, "call", fake_call)
        world = FakeWorld()
        tree = Tree("audit_flip", "fixed", default_mode=1, recover=True)
        result = execute_tree(world, 1.0, tree, release=0.0, hidden=False, chain=("a", "f"), seed=41)
        return result, seen

    audited_bad, bad_trace = run_with_audit(False)
    audited_good, good_trace = run_with_audit(True)

    assert bad_trace == good_trace == [("a", 1), ("f", 0)]
    assert audited_bad["decisions"] == audited_good["decisions"]
    assert audited_bad["duration"] == pytest.approx(audited_good["duration"])
    assert audited_bad["recoveries"] == audited_good["recoveries"] == 0
    assert audited_bad["success"] is False
    assert audited_good["success"] is True


def test_execute_public_feedback_controls_recovery_not_private_audit(monkeypatch):
    """A public failure recovers even when the independent audit says true."""

    seen = []

    def fake_call(self, name, mode=0, *, fresh_after=0.0):
        self.state.time += 0.1
        public_success = not (name == "a" and mode == -1)
        row = _audited_fake_row(
            name,
            mode,
            public_success=public_success,
            independent_success=True,
        )
        self.calls.append(row)
        seen.append((name, mode))
        return row, []

    monkeypatch.setattr(ProgramExecutor, "call", fake_call)
    world = FakeWorld()
    tree = Tree("public_recovery", "fixed", default_mode=-1, recover=True)
    result = execute_tree(world, 1.0, tree, release=0.0, hidden=False, chain=("a", "f"), seed=42)

    assert seen == [("a", -1), ("a", 1), ("f", 0)]
    assert result["recoveries"] == 1
    assert result["program_calls"] == 3
    assert result["success"] is True


def test_predict_public_feedback_and_private_audit_are_separate(monkeypatch):
    """Prediction uses public feedback for recovery but private audit for score."""

    def run_prediction(public_failure: bool):
        seen = []

        def fake_bundle_call(self, name, mode):
            self.time += 0.1
            public_success = not (public_failure and name == "a" and mode == -1)
            independent_success = not (not public_failure and name == "a")
            row = _audited_fake_row(
                name,
                mode,
                public_success=public_success,
                independent_success=independent_success,
            )
            self.calls.append(row)
            seen.append((name, mode))
            return row

        monkeypatch.setattr(BundleExecutor, "call", fake_bundle_call)
        world = FakeWorld()
        bundle = _bundle(1.0)
        tree = Tree("predict_separation", "fixed", default_mode=-1, recover=True)
        result = predict_tree(world, bundle, tree, hidden=False, chain=("a", "f"))
        return result, seen

    # Public success with private/audit failure: no recovery, continue to f,
    # but the final task score remains false.
    scored_bad, no_recovery_trace = run_prediction(public_failure=False)
    assert no_recovery_trace == [("a", -1), ("f", 0)]
    assert scored_bad["recoveries"] == 0
    assert scored_bad["success"] is False

    # Public failure with private/audit success: recovery is executed.
    recovered, recovery_trace = run_prediction(public_failure=True)
    assert recovery_trace == [("a", -1), ("a", 1), ("f", 0)]
    assert recovered["recoveries"] == 1
    assert recovered["success"] is True


def test_native_smplx_world_smoke_uses_real_atlas_geometry():
    from leg_program_bench.smplx_scene import SmplxAtlasLeg

    atlas = SmplxAtlasLeg.load()
    world = CoreWorld(atlas, Spec("smoke", 17, 1, -1, knee_deg=60.0, width_scale=1.0, length_scale=1.0))
    point, rotation = world.reference_at(0.72, 1)
    assert point.shape == (3,)
    assert rotation.shape == (3, 3)
    assert np.isfinite(point).all()
    assert np.isfinite(world.true_points).all()
    assert world.digest

