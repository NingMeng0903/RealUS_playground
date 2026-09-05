"""Behavioral tests for daemon mode/DOF boundary handoff."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from peirastic.core.ipc import Cmd, Status
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof import daemon
from rm75_control.control.joint_admittance_8dof.loop import LoopResult, Phase


class _Hub:
    def __init__(self, polls=None):
        self.events: list[tuple[str, dict]] = []
        self.polls = list(polls or [])
        self.stop = False
        self.motion = SimpleNamespace(publish=lambda **kw: None)

    def publish(self, **kwargs):
        self.events.append(("publish", dict(kwargs)))

    def poll(self):
        return self.polls.pop(0) if self.polls else None

    def ack(self, seq):
        self.events.append(("ack", {"seq": int(seq)}))

    def request_stop(self):
        self.stop = True

    def clear_stop(self):
        self.stop = False

    def should_stop(self):
        return bool(self.stop)


class _Panel:
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def event(self, level, msg):
        self.events.append((str(level), str(msg)))

    def update(self, **kwargs):
        del kwargs

    def maybe_draw(self):
        pass


class _Twist:
    def read(self):
        return {
            "stamp": 0.0,
            "hz": float("nan"),
            "connected": False,
            "r3": False,
            "twist": np.zeros(6),
        }


class _Outer:
    last_err_mm = float("nan")

    def set_origin(self, pose, **kwargs):
        del pose, kwargs


def _service_for_boundary() -> daemon.ControllerService:
    svc = object.__new__(daemon.ControllerService)
    svc.raw = {"timing": {"dt_ms": 5.0}}
    svc.log_csv = None
    svc.estop = daemon.EstopBus()
    svc.panel = _Panel()
    svc.hub = _Hub()
    svc.twist = _Twist()
    svc.inner = SimpleNamespace(
        q_cmd=np.zeros(8),
        core=SimpleNamespace(qdot_prev=np.zeros(8)),
    )
    svc.ctx = SimpleNamespace(dof=8)
    svc.kin = SimpleNamespace()
    svc.force_observer = None
    svc.mode = Mode.SERVO_TWIST_HOLD
    svc.ticks = 0
    svc._stop = False
    svc._pending = ModeRequest(Mode.TRACK_CARTESIAN, {"label": "queued"})
    svc._pending_commanded = True
    svc._pending_install_seq = 41
    svc._pending_dof = (7, 40)
    svc._dof = 8
    svc._live = None
    svc._mode_t0 = 0.0
    svc._finite_duration = None
    svc._cmd_seq = 40
    svc._runner_started = False
    svc._dof_boundary_open = False
    svc._fault_sm = "RUNNING"
    svc._fault_epoch = 0
    svc._rail_hist = []
    svc._rail_t = []
    svc.tcp_name = None
    svc.force_observer_error = ""
    return svc


def test_pending_dof_runs_transition_hold_then_installs_queued_mode(monkeypatch):
    svc = _service_for_boundary()
    compiled_modes: list[Mode] = []
    entered: list[Mode] = []

    def fake_compile(ctx, req, *, raw, twist_read, dt):
        del ctx, raw, twist_read, dt
        compiled_modes.append(req.mode)
        return Phase(
            outer=_Outer(),
            label=str(req.payload.get("label") or req.mode.name),
            on_enter=lambda: entered.append(req.mode),
        )

    def commit(_bus, _rail):
        svc._pending_dof = None
        svc._dof = 7
        svc.ctx.dof = 7
        svc.hub.events.append(("commit", {"dof": 7}))
        return True

    runner_calls: list[bool] = []

    def fake_runner(_sess, phases, _inner, **kwargs):
        runner_calls.append(bool(kwargs["preserve_controller_state"]))
        phase = phases[0]
        if phase.on_enter is not None:
            phase.on_enter()
        step = SimpleNamespace(q_send=np.zeros(8), slack_norm=0.0)
        kwargs["on_step"]("test", 0.0, step, np.zeros(6), np.zeros(3), 0.0)
        if len(runner_calls) == 1:
            assert kwargs["stop_check"]()
        else:
            svc._stop = True
        return LoopResult(1, 0.005, 0.0, False)

    monkeypatch.setattr(daemon, "compile_request", fake_compile)
    monkeypatch.setattr(daemon, "run_joint_admittance_phases", fake_runner)
    svc._commit_pending_dof = commit

    svc.run(SimpleNamespace(robot=None), None, None)

    assert runner_calls == [False, True]
    assert compiled_modes[0] == Mode.SERVO_TWIST_HOLD
    assert compiled_modes[-1] == Mode.TRACK_CARTESIAN
    assert entered[-1] == Mode.TRACK_CARTESIAN
    install = [event for kind, event in svc.hub.events if kind == "publish" and "install_seq" in event]
    assert install and install[-1]["install_seq"] == 41
    commit_i = next(i for i, event in enumerate(svc.hub.events) if event[0] == "commit")
    install_i = next(i for i, event in enumerate(svc.hub.events) if event[0] == "publish" and event[1].get("install_seq") == 41)
    assert commit_i < install_i


def test_transition_hold_polls_estop_and_does_not_commit(monkeypatch):
    svc = _service_for_boundary()
    compiled_modes: list[Mode] = []

    def fake_compile(ctx, req, *, raw, twist_read, dt):
        del ctx, raw, twist_read, dt
        compiled_modes.append(req.mode)
        return Phase(outer=_Outer(), label=req.mode.name)

    commit_called = []

    def commit(_bus, _rail):
        commit_called.append(True)
        return True

    def fake_runner(_sess, phases, _inner, **kwargs):
        if phases[0].on_enter is not None:
            phases[0].on_enter()
        svc.hub.polls = [(Cmd.ESTOP, 99, None)]
        step = SimpleNamespace(q_send=np.zeros(8), slack_norm=0.0)
        kwargs["on_step"]("test", 0.0, step, np.zeros(6), np.zeros(3), 0.0)
        assert svc.estop.tripped
        svc._stop = True
        return LoopResult(1, 0.005, 0.0, False)

    monkeypatch.setattr(daemon, "compile_request", fake_compile)
    monkeypatch.setattr(daemon, "run_joint_admittance_phases", fake_runner)
    svc._commit_pending_dof = commit

    svc.run(SimpleNamespace(robot=None), None, None)

    assert compiled_modes == [Mode.SERVO_TWIST_HOLD]
    assert commit_called == []
    assert svc.estop.tripped


def test_hardware_fault_cancels_pending_dof_and_mode() -> None:
    svc = _service_for_boundary()
    robot_events: list[str] = []

    class _Robot:
        def rm_set_arm_slow_stop(self):
            robot_events.append("arm_slow_stop")

    svc._trip_hardware(None, "uncertified_brake", robot=_Robot())

    assert svc.estop.tripped
    assert svc._pending_dof is None
    assert svc._pending is None
    assert svc._pending_commanded is False
    assert svc._pending_install_seq is None
    assert svc._dof_boundary_open is False
    assert robot_events == ["arm_slow_stop"]
    failures = [
        event
        for kind, event in svc.hub.events
        if kind == "publish" and event.get("status") == Status.ERROR
    ]
    assert failures
    assert any(event.get("dof_done_seq") == 40 for event in failures)
    assert any(event.get("done_seq") == 41 for event in failures)


def test_prepublication_stop_uses_coordinated_brake_without_estop() -> None:
    events: list[str] = []

    class _Rail:
        enabled = True

        def hold_current(self):
            events.append("rail_hold")

    class _Robot:
        def rm_set_arm_slow_stop(self):
            events.append("arm_slow_stop")

    daemon.ControllerService._coordinated_brake(_Rail(), robot=_Robot())
    assert events == ["rail_hold", "arm_slow_stop"]
