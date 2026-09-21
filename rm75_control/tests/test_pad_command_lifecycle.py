"""Exercise handheld lifecycle hooks through the real runner, without hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof import loop
from rm75_control.control.joint_admittance_8dof.command_lifecycle import CommandLifecycleTransaction


class Owner:
    pending_rotation_base_tcp = np.eye(3)

    def __init__(self, events, fault):
        self.events, self.fault = events, fault
        self.aborts, self.commits, self.reviews = [], [], []

    def command_constraints(self):
        return {"rocking_axis_base": [0, 1, 0], "rocking_bounds": (-.1, .1)}

    def review(self, final_tool, *, now_s, facts):
        self.events.append("review")
        self.reviews.append(final_tool.copy())
        return self.fault != "review"

    def started(self):
        self.events.append("started")
        return self.fault != "started"

    def commit(self, final_tool, *, now_s, facts):
        self.events.append("outer_commit")
        self.commits.append((final_tool.copy(), facts))

    def abort(self, reason, *, definitely_not_sent=False, facts=None, latch=None):
        self.events.append("abort")
        self.aborts.append((reason, definitely_not_sent, facts))


def run_fixture(monkeypatch, fault="none", *, owned=True, phases=1):
    events = []
    owner = Owner(events, fault)
    if fault == 'arm_dispatch_expired':
        owner.arm_dispatch_time = lambda now_s: False
    owner.inner_periods = []
    modeled = np.array([.004, 0, .002, 0, .04, 0])
    clock = [10.]
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(loop.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(loop, "_wait_until", lambda _: clock.__setitem__(0, clock[0] + .005))
    wd = SimpleNamespace(fired=False, start=lambda: None, arm=lambda: None,
                         beat=lambda: None, stop=lambda: None)
    if fault in ('scheduled', 'stop_in_window'):
        owner.dispatch_wait_s = lambda now_s: .008
        def wait(deadline):
            events.append('dispatch_wait')
            clock[0] = max(clock[0], deadline)
            if fault == 'stop_in_window': wd.fired = True
        monkeypatch.setattr(loop, '_wait_until', wait)
    monkeypatch.setattr(loop, "Watchdog", lambda *args: wd)
    monkeypatch.setattr(loop, "_rail_execution_velocity_estimate", lambda *args, **kwargs: None)
    monkeypatch.setattr(loop, "_record_execution_observation", lambda *args: None)
    snap = SimpleNamespace(q_deg=np.zeros(7), pose=np.zeros(6), t_s=10., seq=1, force_raw=np.zeros(6))
    def read():
        if fault in ('slow_ack', 'timeout'):
            snap.t_s = clock[0]
            snap.seq += 1
        return snap
    observer = SimpleNamespace(wait_first_pose=lambda **kwargs: np.zeros(6), read=read)
    robot = SimpleNamespace(rm_set_arm_slow_stop=lambda: events.append("stop"))
    rail = SimpleNamespace(enabled=True, measured_m=0., config=SimpleNamespace(poll_hz=50),
        reserve_target_m=lambda *args, **kwargs: events.append("reserve") or True,
        abort_reservation=lambda: events.append("rail_abort"),
        hold_current=lambda: events.append("hold"))
    def rail_commit(**kwargs):
        if owned and fault != "hot_swap":
            assert not kwargs.get("acknowledge")
        events.append("rail_send")
        if fault == "slow_ack": clock[0] += .015
        if fault == "stop_after_rail": wd.fired = True
        if fault == "rail_exception": raise OSError("rail transport")
        return fault != "partial"
    rail.commit_reservation = rail_commit
    # Reservation guard is tested separately in test_final_qpik_send_chain.
    monkeypatch.setattr(loop, "_reserve_rail_target", lambda *args, **kwargs: (events.append("reserve") or True, ""))
    def send(*args):
        events.append("arm_send")
        if fault == "unknown": raise TimeoutError("arm outcome unknown")
        if fault == "stop_after_arm": wd.fired = True
    monkeypatch.setattr(loop, "_send_joint_canfd_cmd", send)
    step = loop.JointIkStep(q_send=np.zeros(8), qdot=np.zeros(8), twist_base=modeled,
        sigma_min=1., manip=1., slack_norm=0., n_cbf_active=0, follow_err_rad=0.)
    step.v_tcp_estimated = modeled.copy()
    coast_ticks = [0]
    if fault == "timeout":
        step.fallback_reason = "native_timeout_coast"
    if fault == "publication_infeasible":
        step.fallback_reason = "publication_infeasible"
        step.fallback_level = "stop"
    cfg = loop.JointIkConfig()
    cfg.rt_disable_gc = False
    inner = SimpleNamespace(cfg=cfg, q_cmd=np.zeros(8), execution_observer=None,
        kin=SimpleNamespace(fk_pose=lambda q: np.zeros(6), jacobian=lambda q: np.zeros((6, 8))),
        rail_observer=SimpleNamespace(_initialized=False), last_v_r_ref=0.,
        limits=SimpleNamespace(q_lower=np.full(8, -1.), q_upper=np.ones(8)),
        _direct_joint_ptp=False, _plan_drives_rail=False,
        set_direct_joint_ptp=lambda x: None, set_plan_drives_rail=lambda x: None,
        commit_publication=lambda q: events.append("inner_commit"),
        abort_publication=lambda: events.append("inner_abort"))
    def update(twist, *args, **kwargs):
        events.append("ik")
        owner.inner_periods.append((args[0], kwargs['dt_wall_s']))
        if owned and fault != "hot_swap": assert kwargs["rocking_bounds"] == (-.1, .1)
        assert kwargs["auto_commit"] is False
        if fault == "timeout":
            coast_ticks[0] += 1
            step.fallback_reason = "native_timeout_coast" if coast_ticks[0] == 1 else ""
        if fault == "publication_infeasible":
            coast_ticks[0] += 1
            if coast_ticks[0] == 1:
                step.fallback_reason = "publication_infeasible"
                step.fallback_level = "stop"
            else:
                step.fallback_reason = ""
                step.fallback_level = "none"
        return step
    inner.update = update
    def sample(*args, **kwargs):
        events.append("sample")
        if fault == "sample": raise ValueError("candidate preparation")
        return modeled
    outer = SimpleNamespace(sample=sample)
    if owned and fault != "hot_swap": outer.command_lifecycle = owner
    tick_number = [0]
    def tick(*args):
        tick_number[0] += 1
        if fault == "hot_swap":
            if tick_number[0] == 1: outer.command_lifecycle = owner
            elif tick_number[0] == 2: del outer.command_lifecycle
        if fault == "callback": raise ValueError("after publish")
    result = loop.run_joint_admittance_phases(
        SimpleNamespace(robot=robot),
        [loop.Phase(outer=outer, duration_s=.06 if fault == "slow_ack" else .015 if fault in ("hot_swap", "timeout", "publication_infeasible") else .005, on_tick=tick) for _ in range(phases)],
        inner, state_bus=SimpleNamespace(observer=observer), rail_bridge=rail,
        preserve_controller_state=True, verbose=False,
    )
    return owner, events, result, modeled


def test_real_runner_commits_only_after_both_transports(monkeypatch):
    owner, events, result, modeled = run_fixture(monkeypatch)
    assert result.stop_reason == ""
    assert events[:7] == ["sample", "ik", "reserve", "review", "started", "arm_send", "rail_send"]
    assert events.index("outer_commit") > events.index("rail_send")
    assert events.index("inner_commit") > events.index("rail_send")
    assert not owner.aborts
    np.testing.assert_array_equal(owner.commits[0][0], modeled)
    assert owner.commits[0][1]["arm"] == owner.commits[0][1]["rail"] == "sent"


@pytest.mark.parametrize("fault", ["review", "started", "sample"])
def test_real_runner_rejects_without_sends_or_commits(monkeypatch, fault):
    owner, events, result, _ = run_fixture(monkeypatch, fault)
    assert result.stop_reason
    assert "arm_send" not in events and "rail_send" not in events
    assert "inner_commit" not in events and not owner.commits
    assert len(owner.aborts) == 1 and owner.aborts[0][1]


def test_native_timeout_coast_skips_unpublished_tick_and_continues(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, "timeout")
    assert result.stop_reason == ""
    assert owner.aborts and owner.aborts[0][0] == "native_timeout_coast"
    assert owner.commits
    assert "arm_send" in events and "rail_send" in events


def test_pad_inner_hold_is_only_publication_infeasible():
    step = loop.JointIkStep(q_send=np.zeros(8), qdot=np.zeros(8), twist_base=np.zeros(6),
        sigma_min=1., manip=1., slack_norm=0., n_cbf_active=0, follow_err_rad=0.)
    step.fallback_reason = "publication_infeasible"
    assert loop.pad_inner_should_hold_without_send(step)
    step.fallback_reason = "native_timeout"
    assert not loop.pad_inner_should_hold_without_send(step)
    step.fallback_reason = "native_timeout_coast"
    assert not loop.pad_inner_should_hold_without_send(step)


def test_publication_infeasible_holds_unpublished_tick_and_continues(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, "publication_infeasible")
    assert result.stop_reason == ""
    assert owner.aborts and owner.aborts[0][0] == "publication_infeasible"
    assert owner.commits
    assert "arm_send" in events and "rail_send" in events
    assert "stop" not in events


@pytest.mark.parametrize("fault,arm,rail", [
    ("unknown", "unknown", "not_committed"),
    ("partial", "sent", "failed_or_unknown"),
    ("rail_exception", "sent", "failed_or_unknown"),
    ("stop_after_arm", "sent", "not_committed"),
    ("stop_after_rail", "sent", "acknowledged_then_stopped"),
])
def test_real_runner_partial_or_unknown_never_commits(monkeypatch, fault, arm, rail):
    owner, events, result, _ = run_fixture(monkeypatch, fault)
    assert result.stop_reason
    assert "inner_commit" not in events and not owner.commits
    assert len(owner.aborts) == 1
    assert owner.aborts[0][1] is False
    assert owner.aborts[0][2] == {"arm": arm, "rail": rail}
    assert "stop" in events and "rail_abort" in events


def test_real_runner_phase_transition_does_not_reuse_candidate(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, phases=2)
    assert result.stop_reason == ""
    assert len(owner.reviews) == len(owner.commits) == 2
    assert not owner.aborts


def test_real_runner_post_commit_exception_keeps_committed_candidate(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, "callback")
    assert result.stop_reason.startswith("pad_exception")
    assert len(owner.commits) == 1 and not owner.aborts
    assert "stop" in events


def test_legacy_runner_does_not_require_lifecycle(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, owned=False)
    assert result.stop_reason == ""
    assert events.index("inner_commit") > events.index("rail_send")
    assert not owner.reviews and not owner.commits and not owner.aborts


def test_transaction_rejects_reuse_and_aborts_on_owner_failure():
    owner = Owner([], "none")
    tx = CommandLifecycleTransaction(owner)
    tx.begin_candidate()
    with pytest.raises(RuntimeError): tx.begin_candidate()
    tx.abort("phase_exit")
    with pytest.raises(RuntimeError): tx.started()
    with pytest.raises(RuntimeError): tx.commit(now_s=1, rail_enabled=False)
    assert owner.aborts == [("phase_exit", True, None)]


def test_real_runner_hot_binds_pad_owner_within_one_phase(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, "hot_swap")
    assert result.stop_reason == ""
    assert len(owner.reviews) == len(owner.commits) == 1
    assert not owner.aborts
    # First and third ticks run without a lifecycle, the middle tick has one.
    assert events.count("arm_send") == 3
    assert events.index("review") > events.index("inner_commit")


def test_slow_rail_ack_keeps_five_millisecond_inner(monkeypatch):
    owner, _, result, _ = run_fixture(monkeypatch, 'slow_ack')
    assert not result.stop_reason
    assert len(owner.inner_periods) >= 2
    assert max(wall for _, wall in owner.inner_periods) > .015
    for integration, _wall in owner.inner_periods:
        assert integration == pytest.approx(0.005)


def test_runner_waits_for_certified_window_before_review_and_send(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, 'scheduled')
    assert not result.stop_reason
    assert events.index('ik') < events.index('dispatch_wait') < events.index('review')
    assert events.index('review') < events.index('arm_send')
    assert len(owner.commits) == 1


def test_stop_during_dispatch_wait_cannot_send_or_commit(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, 'stop_in_window')
    assert result.stop_reason == 'pad_stop_before_dispatch'
    assert 'arm_send' not in events and 'rail_send' not in events
    assert not owner.commits and len(owner.aborts) == 1


def test_delay_after_review_cannot_dispatch_outside_certified_window(monkeypatch):
    owner, events, result, _ = run_fixture(monkeypatch, 'arm_dispatch_expired')
    assert result.stop_reason == 'pad_arm_dispatch_expired'
    assert 'arm_send' not in events and 'rail_send' not in events
    assert not owner.commits and owner.aborts[0][1] is True
