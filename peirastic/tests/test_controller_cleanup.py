"""Controller resources must be released before the SDK session disconnects."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.realman8dof import daemon


def test_force_relay_never_labels_missing_or_invalid_observer_as_zero():
    samples = []
    svc = object.__new__(daemon.ControllerService)
    svc._state_relay = SimpleNamespace(set_f_ext=lambda f: samples.append(f.copy()))
    svc.force_observer = None
    svc._publish_force_sample(np.zeros(6))
    svc.force_observer = object()
    svc._publish_force_sample(np.full(6, np.nan))
    svc._publish_force_sample(np.zeros(3))
    assert not samples
    svc._publish_force_sample(np.arange(6.0))
    np.testing.assert_array_equal(samples[0], np.arange(6.0))


def test_close_releases_inner_and_restores_only_owned_signals(monkeypatch):
    calls = []
    svc = object.__new__(daemon.ControllerService)
    svc.inner = SimpleNamespace(close=lambda: calls.append("inner"))
    svc.hub = SimpleNamespace(close=lambda: calls.append("hub"))
    svc.twist = SimpleNamespace(close=lambda: calls.append("twist"))
    owned, previous, newer = object(), object(), object()
    handlers = {daemon.signal.SIGINT: owned, daemon.signal.SIGTERM: newer}
    svc._signal_handler = owned
    svc._previous_signals = dict.fromkeys(handlers, previous)
    monkeypatch.setattr(daemon.signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(daemon.signal, "signal", handlers.__setitem__)
    svc.close()
    svc.close()
    assert calls == ["inner", "hub", "twist"]
    assert handlers[daemon.signal.SIGINT] is previous
    assert handlers[daemon.signal.SIGTERM] is newer


def test_close_releases_ipc_even_if_inner_close_fails():
    calls = []
    svc = object.__new__(daemon.ControllerService)

    def fail():
        raise RuntimeError("native cleanup failed")

    svc.inner = SimpleNamespace(close=fail)
    svc.hub = SimpleNamespace(close=lambda: calls.append("hub"))
    svc.twist = SimpleNamespace(close=lambda: calls.append("twist"))
    with pytest.raises(RuntimeError, match="native cleanup failed"):
        svc.close()
    assert calls == ["hub", "twist"]


@pytest.mark.parametrize("failure", ["bus_start", "relay_start", "run"])
def test_service_startup_and_run_failures_stop_streams_before_sdk(monkeypatch, tmp_path, failure):
    calls = []

    def step(name):
        calls.append(name)
        if name == failure:
            raise RuntimeError(name)

    @contextmanager
    def session(**kwargs):
        try:
            yield SimpleNamespace(robot=object(), ip="test")
        finally:
            step("sdk_close")

    svc = SimpleNamespace(
        inner=SimpleNamespace(kin=object()),
        close=lambda: step("service_close"),
        panel=SimpleNamespace(event=lambda *args: None),
        tcp_name="test", force_observer=None,
        run=lambda *args: step("run"),
    )
    rail = SimpleNamespace(enabled=False, stop=lambda: step("rail_stop"))
    bus = SimpleNamespace(start=lambda: step("bus_start"), stop=lambda: step("bus_stop"))
    relay = SimpleNamespace(start=lambda: step("relay_start"), stop=lambda: step("relay_stop"))
    monkeypatch.setattr(daemon, "load_yaml", lambda p: {})
    monkeypatch.setattr(daemon, "parse_rail_servo_config", lambda raw: None)
    monkeypatch.setattr(daemon, "RailServoBridge", lambda cfg: rail)
    monkeypatch.setattr(daemon, "parse_state_relay_config", lambda raw: SimpleNamespace(enabled=True, name="test", hz=200))
    monkeypatch.setattr(daemon, "RobotSession", session)
    monkeypatch.setattr(daemon, "ControllerService", lambda *args, **kwargs: svc)
    monkeypatch.setattr(daemon, "RobotStateBus", lambda *args, **kwargs: bus)
    monkeypatch.setattr(daemon, "StateRelayPublisher", lambda *args, **kwargs: relay)
    with pytest.raises(RuntimeError, match=failure):
        daemon.run_service(tmp_path / "controller.yaml")
    assert calls.index("bus_stop") < calls.index("sdk_close")
    assert calls.index("service_close") < calls.index("sdk_close")
    if failure != "bus_start":
        assert calls.index("relay_stop") < calls.index("bus_stop")
    assert "rail_stop" in calls
