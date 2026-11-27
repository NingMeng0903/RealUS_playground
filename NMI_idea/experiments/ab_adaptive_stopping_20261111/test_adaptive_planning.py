"""Fast public-contract tests for adaptive_planning.

Native replay is covered by ``test_native_adapter.py`` and remains opt-in.  The
tests here use a tiny deterministic adapter so planner boundary violations are
cheap to catch before a multi-hour development run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from acoustics import ProbeAction, PublicObservation
from adaptive_planning import (AdaptivePlanner, PlannerConfig, public_action_from_event,
                               public_outcome_branches)
from belief_model import ModelParams, new_belief
from fixture import SINGLE_DWELLS, single_dwell_action
from run_adaptive import _event_id
from train_calibrate import render_sensor_v2
from acoustics import make_scenario


@dataclass(frozen=True)
class FakeState:
    key: tuple = ()
    elapsed: float = 0.0
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    native: object = None
    valid: bool = True


class FakeAdapter:
    def __init__(self):
        self.start = FakeState()
        self.measured_frame = np.eye(4)
        self.verification_calls = self.execution_calls = 0
        self.verification_count = self.execution_count = 0
        self._events = {}

    def _event(self, state, action_id):
        terminal = action_id in ("left", "right")
        if "single_dwell" in action_id:
            dwell = float(action_id.rsplit("_", 1)[-1])
        else:
            dwell = 1.0
        window = "left" if action_id in ("left", "side_left") or action_id.startswith("side_left") else "right" if action_id in ("right", "side_right") or action_id.startswith("side_right") else "fork"
        goal = {"left": (-.016, .018, .10), "right": (.016, .018, -.10),
                "fork_minus": (.006, -.003, -.075), "fork_plus": (-.006, -.003, .075),
                "side_left": (-.016, .013, .075), "side_right": (.016, .013, -.075)}
        base = action_id.split("__", 1)[0]
        local = goal.get(base, goal.get(action_id, (0.0, 0.018, 0.0)))
        pose = np.eye(4)
        pose[0, 3], pose[1, 3] = local[:2]
        state_out = FakeState(tuple(state.key) + (action_id,), state.elapsed + 0.1 + dwell, pose)
        action = {"id": action_id, "base": base, "kind": "terminal" if terminal else "probe",
                  "window": window, "goal_local": list(local), "dwell_s": dwell,
                  "interruptible": not terminal}
        return SimpleNamespace(action_id=action_id, action=action, feasible=True, output=state_out,
                               actual_output={"pose": pose}, elapsed_s=state_out.elapsed,
                               duration_s=0.1 + dwell, margin=1.0, route_id="fake",
                               call_kind="verification", source="fake", verification_calls=self.verification_calls,
                               execution_calls=self.execution_calls, verification_count=self.verification_count,
                               execution_count=self.execution_count, trajectory_records=(), result={})

    def checked(self, state, action_id):
        self.verification_calls += 1
        self.verification_count += 1
        event = self._event(state, action_id)
        self._events[(tuple(state.key), action_id)] = event
        return event

    def execute(self, state, action_id):
        self.execution_calls += 1
        self.execution_count += 1
        return self._events.get((tuple(state.key), action_id), self._event(state, action_id))


def tiny_params():
    return ModelParams(particle_count=8, max_components=1, particle_seed=721501)


def tiny_config(arm="B_H1"):
    return PlannerConfig(arm=arm, qmc_initial=2, qmc_top2=2, qmc_refine=2,
                         qmc_unstable=2, h2_initial=2, h2_refine=2,
                         h2_unstable=2, max_probe_frames=3)


def test_public_action_and_qmc_never_expose_hidden_truth():
    adapter = FakeAdapter()
    event = adapter.checked(adapter.start, single_dwell_action("fork_minus", SINGLE_DWELLS[0]))
    action = public_action_from_event(event, adapter.measured_frame)
    assert isinstance(action, ProbeAction)
    branches = public_outcome_branches(new_belief(params=tiny_params()), action, 2, seed=99)
    assert len(branches) == 2
    for _, packet in branches:
        assert not hasattr(packet, "hidden")
        assert packet.candidate_type in {"candidate", "missing"}
        if packet.detected:
            assert np.isfinite(packet.center_offset_m)
            assert np.isfinite(packet.axis_angle_rad)
        else:
            assert packet.center_offset_m != packet.center_offset_m


def test_every_actual_public_packet_updates_belief_and_duplicates_are_safe():
    adapter = FakeAdapter()
    planner = AdaptivePlanner(adapter, np.eye(4), config=tiny_config(), model_params=tiny_params())
    event = adapter.checked(adapter.start, "left")
    packet = PublicObservation("e0", "public", "left", 1.0, "checked", False, 0.0,
                               float("nan"), float("nan"), float("nan"), "missing")
    action = public_action_from_event(event, adapter.measured_frame)
    planner.observe("e0", action, packet, event=event)
    duplicate = planner.belief.update("e0", action, packet)
    assert duplicate.duplicate
    assert "e0" in planner.belief.event_ids
    assert planner.fresh_count == 1


def test_all_arms_share_probe_permission_and_no_deadline_symbol():
    for arm in ("A", "B_H1", "B_H2", "pair_fixed"):
        adapter = FakeAdapter()
        planner = AdaptivePlanner(adapter, np.eye(4), config=tiny_config(arm), model_params=tiny_params())
        candidates = planner._probe_candidates()
        assert len(candidates) == 12
        assert planner.config.max_probe_frames == 3  # watchdog, not wall time
    source = Path(__file__).with_name("adaptive_planning.py").read_text(encoding="utf-8")
    assert "if state.elapsed > deadline" not in source
    assert '"no_deadline": True' in source


def test_same_prefix_has_same_public_event_identity():
    adapter_a, adapter_b = FakeAdapter(), FakeAdapter()
    event_a = adapter_a.checked(adapter_a.start, "left")
    event_b = adapter_b.checked(adapter_b.start, "left")
    assert tuple(event_a.output.key) == tuple(event_b.output.key)
    # The runner's event-id convention is prefix-derived; this test checks the
    # underlying property used by paired arms without invoking native replay.
    assert event_a.action_id == event_b.action_id
    assert np.allclose(event_a.output.pose, event_b.output.pose)
    event_id_a = _event_id(0, "constrained", adapter_a.start, event_a)
    event_id_b = _event_id(0, "constrained", adapter_b.start, event_b)
    assert event_id_a == event_id_b
    scenario = make_scenario(721503, "left_leg:test:0000")
    action = public_action_from_event(event_a, np.eye(4))
    packet_a, _ = render_sensor_v2(scenario, action, event_id=event_id_a, seed=721504)
    packet_b, _ = render_sensor_v2(scenario, action, event_id=event_id_b, seed=721504)
    for key, value in vars(packet_a).items():
        other = getattr(packet_b, key)
        if isinstance(value, float) and np.isnan(value):
            assert isinstance(other, float) and np.isnan(other)
        else:
            assert value == other


def test_all_arms_execute_one_public_missing_packet_without_gt_access():
    """A/A_info/A_task/B_H1/B_H2/pair share the reject update boundary."""
    from adaptive_planning import AdaptivePlanner

    for arm in ("A_task", "A_info", "B_H1", "B_H2", "pair_fixed"):
        adapter = FakeAdapter()
        planner = AdaptivePlanner(adapter, np.eye(4), config=tiny_config(arm), model_params=tiny_params())

        def acquire(event, state):
            action = public_action_from_event(event, adapter.measured_frame)
            event_id = f"actual/{arm}/{state.key!r}/{event.action_id}"
            packet = PublicObservation(event_id, "public", action.window, event.elapsed_s,
                                       "checked", False, 0.0, float("nan"), float("nan"),
                                       float("nan"), "missing")
            private = {"correct": False, "false": False, "hidden_kind": "missing",
                       "present_before": True, "present_after": True,
                       "target_association": False}
            return event_id, action, packet, private

        result = planner.run(acquire, max_steps=1)
        assert result["fresh_count"] == 1
        assert result["false_complete"] is False
        assert adapter.verification_calls == adapter.execution_calls == 1


def test_probe_public_acceptance_is_absorbing_for_an_arm():
    adapter = FakeAdapter()
    planner = AdaptivePlanner(adapter, np.eye(4), config=tiny_config("B_H1"), model_params=tiny_params())

    def acquire(event, state):
        action = public_action_from_event(event, adapter.measured_frame)
        event_id = f"accept/{state.key!r}/{event.action_id}"
        packet = PublicObservation(event_id, "public", action.window, event.elapsed_s,
                                   "checked", True, .99, .90, 0.0, 0.0, "candidate")
        return event_id, action, packet, {"correct": True, "false": False,
                                         "hidden_kind": "target", "present_before": True,
                                         "present_after": True, "target_association": True}

    result = planner.run(acquire, max_steps=4)
    assert result["reason"] == "public_accepted_probe"
    assert result["correct_complete"] is True
    assert result["fresh_count"] == 1


def test_uniform_clutter_feature_marker_stays_public_and_bounded():
    from types import SimpleNamespace
    from adaptive_planning import public_outcome_branches

    class StubBelief:
        params = SimpleNamespace(clutter_center_halfwidth=.022, clutter_axis_halfwidth=.35)

        def public_predict(self, _action):
            component = SimpleNamespace(weight=.90, quality_mean=.60,
                                        quality_variance=.12 ** 2, center_mean=0.0,
                                        center_variance=.022 ** 2 / 3., axis_mean=0.0,
                                        axis_variance=.35 ** 2 / 3.,
                                        feature_distribution="uniform")
            return SimpleNamespace(p_candidate=.90, p_missing=.10,
                                   components=(component,))

    action = ProbeAction("fork", (0.0, 0.0), 0.0, .30, "probe")
    branches = public_outcome_branches(StubBelief(), action, 64, seed=721504)
    candidates = [packet for _, packet in branches if packet.detected]
    assert candidates
    assert all(abs(packet.center_offset_m) <= .022 for packet in candidates)
    assert all(abs(packet.axis_angle_rad) <= .35 for packet in candidates)
    assert all(not hasattr(packet, "feature_distribution") for packet in candidates)


def test_guard_eighth_update_gets_common_terminal_gate_for_every_arm():
    """The eighth public probe may release a terminal option, but no ninth probe."""
    admitted = SimpleNamespace(first_id="left", second_id=None, score=.95)
    for arm in ("A", "A_task", "A_info", "A_full", "B_H1", "B_H2", "pair_fixed"):
        adapter = FakeAdapter()
        planner = AdaptivePlanner(
            adapter, np.eye(4), config=replace(tiny_config(arm), max_probe_frames=8),
            model_params=tiny_params())
        planner.probe_count = 8
        planner._terminal_options = lambda state=None: ()
        planner._best_terminal = lambda options: admitted

        def acquire(event, state):
            action = public_action_from_event(event, adapter.measured_frame)
            event_id = f"guard-accept/{arm}/{state.key!r}/{event.action_id}"
            packet = PublicObservation(event_id, "public", action.window, event.elapsed_s,
                                       "checked", True, .99, .90, 0.0, 0.0, "candidate")
            return event_id, action, packet, {
                "correct": True, "false": False, "present_before": True,
                "present_after": True, "target_association": True,
            }

        result = planner.run(acquire, max_steps=1)
        assert result["correct_complete"] is True
        assert result["watchdog"] is False
        assert result["probe_count"] == 8
        assert result["terminal_count"] == 1
        assert adapter.execution_calls == 1


def test_guard_eighth_update_without_admission_is_watchdog_not_ninth_probe():
    for arm in ("A", "A_task", "A_info", "A_full", "B_H1", "B_H2", "pair_fixed"):
        adapter = FakeAdapter()
        planner = AdaptivePlanner(
            adapter, np.eye(4), config=replace(tiny_config(arm), max_probe_frames=8),
            model_params=tiny_params())
        planner.probe_count = 8
        planner._terminal_options = lambda state=None: ()

        def never_acquire(event, state):  # pragma: no cover - guard must prevent this
            raise AssertionError("watchdog boundary executed an extra acquisition")

        result = planner.run(never_acquire, max_steps=1)
        assert result["watchdog"] is True
        assert result["abstain"] is False
        assert result["fresh_count"] == 0
        assert result["probe_count"] == 8
        assert result["decisions"][0]["reason"] == "watchdog_probe_guard"

