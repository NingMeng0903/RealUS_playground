"""Behavioral tests for daemon mode/DOF boundary handoff."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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


class _ContactGate:
    """Small deterministic stand-in for the hybrid contact progress gate."""

    def __init__(self) -> None:
        self.started = False
        self.elapsed_s = 0.0


class _GateController:
    def __init__(self) -> None:
        self.contact_present = False
        self.v_force_cmd_z = 0.012


class _GatedOuter(_Outer):
    def __init__(self, gate: _ContactGate, controller: _GateController) -> None:
        self.contact_gate = gate
        self.controller = controller
        self.begin_calls = 0
        self.origin_times: list[float | None] = []

    def set_origin(self, pose, *, t_s=None):
        del pose
        self.origin_times.append(None if t_s is None else float(t_s))

    def begin_hybrid_episode(self, applied_twist, current_pose):
        del applied_twist, current_pose
        self.begin_calls += 1


def _hot_swap_service() -> daemon.ControllerService:
    """Make the daemon runner usable without constructing robot kinematics."""

    svc = _service_for_boundary()
    svc.inner.kin = SimpleNamespace(jacobian=lambda _q: np.zeros((6, 8)))
    svc.inner.begin_hybrid_episode = lambda *_args: None
    svc._pending = ModeRequest(Mode.SERVO_TWIST, {"label": "servo"})
    svc._pending_commanded = True
    svc._pending_install_seq = None
    svc._pending_dof = None
    return svc


def _publish_events(svc):
    return [event for kind, event in svc.hub.events if kind == "publish"]


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
    svc._compile_fault = None
    svc._rail_hist = []
    svc._rail_t = []
    svc.tcp_name = None
    svc.force_observer_error = ""
    return svc


def test_pending_dof_runs_transition_hold_then_installs_queued_mode(monkeypatch):
    svc = _service_for_boundary()
    force_samples = []
    svc.force_observer = object()
    svc._state_relay = SimpleNamespace(set_f_ext=lambda f: force_samples.append(f.copy()))
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
        kwargs["on_step"]("test", 0.0, step, np.zeros(6), np.arange(6.0), 0.0)
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
    assert len(force_samples) == 2
    np.testing.assert_array_equal(force_samples[0], np.arange(6.0))
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


def test_panel_event_does_not_flush_on_caller() -> None:
    from peirastic.core.panel import Panel

    panel = Panel(enabled=True)
    panel.event("STATE", "csv /tmp/example.csv")
    assert "[STATE]" in panel.last_frame
    panel._out_q.put_nowait(None)


def test_compile_fault_stays_until_commanded_success() -> None:
    svc = _service_for_boundary()
    req = ModeRequest(Mode.SERVO_TWIST, {})
    svc._cmd_seq = 9
    svc._note_compile_fault(req, RuntimeError("bad compile"), commanded=True)
    assert svc._compile_fault["seq"] == 9
    assert svc._hold_compile_fault(False) is True
    errs = [
        event
        for kind, event in svc.hub.events
        if kind == "publish" and event.get("status") == Status.ERROR
    ]
    assert errs and errs[-1]["err_code"] == 1
    assert svc._hold_compile_fault(True) is False
    svc._clear_compile_fault()
    assert svc._hold_compile_fault(False) is False


def test_gated_hybrid_hot_install_uses_contact_elapsed_and_keeps_proxy(
    monkeypatch,
):
    """A seek must not consume the scan clock or rebuild the hybrid phase."""

    svc = _hot_swap_service()
    old_wait = object()
    new_wait = object()
    compiled: list[tuple[Mode, dict]] = []
    gated: list[_GatedOuter] = []

    def fake_compile(ctx, req, *, raw, twist_read, dt):
        del ctx, raw, twist_read, dt
        compiled.append((req.mode, dict(req.payload)))
        if req.mode == Mode.TRACK_HYBRID:
            outer = _GatedOuter(_ContactGate(), _GateController())
            gated.append(outer)
            return Phase(
                outer=outer,
                label="hfpc",
                duration_s=2.0,
                # A replacement phase must not leave this callable on the
                # already-running proxy; the gate owns completion instead.
                wait_until=new_wait,
            )
        if req.mode == Mode.SERVO_TWIST:
            return Phase(outer=_Outer(), label="servo", wait_until=old_wait)
        return Phase(outer=_Outer(), label="servo_hold")

    def fake_runner(_sess, phases, _inner, **kwargs):
        phase = phases[0]
        step = SimpleNamespace(q_send=np.zeros(8), slack_norm=0.0)
        if phase.on_enter is not None:
            phase.on_enter()

        # The current phase is already a SERVO proxy. Install HFPC from a
        # nonzero reference time while staying inside the same runner.
        assert phase.wait_until is None
        svc.hub.polls = [
            (
                Cmd.SET_MODE,
                77,
                ModeRequest(
                    Mode.TRACK_HYBRID,
                    {"label": "hfpc", "duration_s": 2.0},
                ),
            )
        ]
        kwargs["on_step"](
            "servo",
            1.25,
            step,
            np.zeros(6),
            np.zeros(6),
            1.25,
        )
        gate_outer = gated[0]
        assert svc._mode_t0 == pytest.approx(1.25)
        assert svc._finite_duration == pytest.approx(2.0)
        assert gate_outer.origin_times == [pytest.approx(1.25)]
        assert gate_outer.begin_calls == 1
        # Neither the old proxy wait_until nor the new Phase wait_until is
        # copied into the long-lived proxy wrapper.
        assert phase.wait_until is None

        def done_events():
            return [
                event
                for event in _publish_events(svc)
                if event.get("status") == Status.DONE
            ]

        # The seek has lasted longer than the two-second scan duration, but
        # contact has not started, so elapsed_s is still zero and no DONE is
        # allowed. This would fail if daemon used t_ref - mode_t0 here.
        kwargs["on_step"](
            "hfpc",
            8.0,
            step,
            np.zeros(6),
            np.zeros(6),
            8.0,
        )
        assert done_events() == []
        assert any(
            event.get("msg", "").startswith("hfpc:approach")
            for event in _publish_events(svc)
        )

        # Once contact starts, elapsed_s still has to reach the full scan
        # duration; the wall/reference time remains irrelevant to completion.
        gate_outer.contact_gate.started = True
        gate_outer.contact_gate.elapsed_s = 1.5
        gate_outer.controller.contact_present = True
        gate_outer.controller.v_force_cmd_z = -0.004
        kwargs["on_step"](
            "hfpc",
            12.0,
            step,
            np.zeros(6),
            np.zeros(6),
            12.0,
        )
        assert done_events() == []
        assert any(
            event.get("msg", "").startswith("hfpc:tracking")
            and "contact=1" in event.get("msg", "")
            and "vz_cmd=-4.0" in event.get("msg", "")
            for event in _publish_events(svc)
        )

        gate_outer.contact_gate.elapsed_s = 2.0
        kwargs["on_step"](
            "hfpc",
            20.0,
            step,
            np.zeros(6),
            np.zeros(6),
            20.0,
        )
        assert len(done_events()) == 1
        assert gate_outer.begin_calls == 1
        assert sum(mode == Mode.TRACK_HYBRID for mode, _ in compiled) == 1
        svc._stop = True
        return LoopResult(4, 0.02, 0.0, False)

    monkeypatch.setattr(daemon, "compile_request", fake_compile)
    monkeypatch.setattr(daemon, "run_joint_admittance_phases", fake_runner)

    svc.run(SimpleNamespace(robot=None), None, None)

    assert sum(mode == Mode.TRACK_HYBRID for mode, _ in compiled) == 1
    assert gated[0].begin_calls == 1


@pytest.mark.parametrize("task_mode", [Mode.TRACK_CARTESIAN, Mode.TRACK_HYBRID])
def test_nongated_finite_hot_install_uses_mode_t0(monkeypatch, task_mode):
    """Finite velocity tasks without a gate retain the mode-time fallback."""

    svc = _hot_swap_service()
    compiled: list[Mode] = []
    install_t0: list[float] = []

    def fake_compile(ctx, req, *, raw, twist_read, dt):
        del ctx, raw, twist_read, dt
        compiled.append(req.mode)
        if req.mode == task_mode:
            return Phase(outer=_Outer(), label="finite", duration_s=2.0)
        if req.mode == Mode.SERVO_TWIST:
            return Phase(outer=_Outer(), label="servo")
        return Phase(outer=_Outer(), label="servo_hold")

    def fake_runner(_sess, phases, _inner, **kwargs):
        phase = phases[0]
        step = SimpleNamespace(q_send=np.zeros(8), slack_norm=0.0)
        if phase.on_enter is not None:
            phase.on_enter()
        svc.hub.polls = [
            (
                Cmd.SET_MODE,
                88,
                ModeRequest(task_mode, {"label": "finite", "duration_s": 2.0}),
            )
        ]
        kwargs["on_step"](
            "servo",
            1.25,
            step,
            np.zeros(6),
            np.zeros(6),
            1.25,
        )
        install_t0.append(float(svc._mode_t0))
        assert svc._finite_duration == pytest.approx(2.0)

        def done_events():
            return [
                event
                for event in _publish_events(svc)
                if event.get("status") == Status.DONE
            ]

        # 1.75 seconds after installation: not complete yet.
        kwargs["on_step"](
            "finite",
            3.0,
            step,
            np.zeros(6),
            np.zeros(6),
            3.0,
        )
        assert done_events() == []

        # The reference clock, rather than wall time before installation,
        # reaches the two-second finite duration here.
        kwargs["on_step"](
            "finite",
            3.30,
            step,
            np.zeros(6),
            np.zeros(6),
            3.30,
        )
        assert len(done_events()) == 1
        svc._stop = True
        return LoopResult(3, 0.015, 0.0, False)

    monkeypatch.setattr(daemon, "compile_request", fake_compile)
    monkeypatch.setattr(daemon, "run_joint_admittance_phases", fake_runner)

    svc.run(SimpleNamespace(robot=None), None, None)

    assert install_t0 == [pytest.approx(1.25)]
    assert compiled.count(task_mode) == 1
