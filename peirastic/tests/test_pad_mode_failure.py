"""Gamepad mode failure reconciliation; no IPC/robot process started."""
from itertools import chain, repeat
from types import SimpleNamespace
import time

import pytest
from peirastic.apps import gamepad
from peirastic.apps.gamepad import PadHybridExitReporter, _install_mode, pad_hybrid_active
from peirastic.api.codes import ERR_CONTROLLER, ERR_STOPPED, OK
from peirastic.core.ipc import CommandClient, IPC_ABI_MAGIC, IPC_ABI_VERSION, Status
from peirastic.core.modes import Mode, ModeRequest


@pytest.mark.parametrize('status', [Status.ERROR, Status.STOPPED, Status.ESTOP, Status.DONE])
def test_failed_pad_label_is_not_an_active_hybrid(status):
    assert not pad_hybrid_active(dict(mode=int(Mode.TRACK_HYBRID),status=int(status),msg='pad_final_review_rejected'))


def test_active_pad_label_keeps_current_recorder():
    assert pad_hybrid_active(dict(mode=int(Mode.TRACK_HYBRID),status=int(Status.RUNNING),msg='track_hybrid_pad_ultrapoc'))


def test_install_error_exposes_controller_fault_reason():
    client = SimpleNamespace(set_mode=lambda req: 7,wait_installed=lambda seq,mode: 1,
        snapshot=lambda: dict(msg='qpik_fault:stop:publication_infeasible'))
    with pytest.raises(RuntimeError,match='publication_infeasible'):
        _install_mode(client,ModeRequest(Mode.TRACK_HYBRID,{}))


def _telemetry(status=Status.RUNNING, mode=Mode.TRACK_HYBRID, **extra):
    return dict(
        status=int(status), mode=int(mode),
        msg='track_hybrid_pad_ultrapoc', err_code=0,
        **extra,
    )


@pytest.mark.parametrize('status', [Status.ERROR, Status.STOPPED, Status.ESTOP, Status.DONE])
def test_exit_warning_once_even_before_first_running_poll(status):
    reporter = PadHybridExitReporter()
    reporter.installed(hybrid=True)
    tel = _telemetry(status)
    tel.update(msg='qpik_fault:stop:publication_infeasible', err_code=1)
    warning = reporter.observe(tel)
    assert 'exited unexpectedly' in warning
    assert f'status={status.name}({int(status)})' in warning
    assert 'mode=TRACK_HYBRID(4)' in warning
    assert "msg='qpik_fault:stop:publication_infeasible' err_code=1" in warning
    assert reporter.observe(tel) is None
    reporter.installed(hybrid=True)
    assert reporter.observe(tel) is not None


def test_running_hybrid_observation_arms_exit_reporting():
    reporter = PadHybridExitReporter()
    assert reporter.observe(_telemetry()) is None
    assert reporter.observe(_telemetry(Status.ERROR)) is not None


def test_intentional_l3_exit_does_not_warn():
    reporter = PadHybridExitReporter()
    reporter.installed(hybrid=True)
    assert reporter.observe(_telemetry()) is None
    reporter.installed(hybrid=False)
    assert reporter.observe(_telemetry(mode=Mode.SERVO_TWIST_HOLD)) is None


def test_idle_overwrite_reports_only_observed_telemetry():
    reporter = PadHybridExitReporter()
    reporter.installed(hybrid=True)
    tel = _telemetry(mode=Mode.SERVO_TWIST_HOLD)
    tel.update(msg='idle', err_code=0)
    warning = reporter.observe(tel)
    assert 'status=RUNNING(1) mode=SERVO_TWIST_HOLD(2)' in warning
    assert "msg='idle' err_code=0" in warning
    assert 'exit reason unavailable in current telemetry' in warning
    assert 'qpik_fault' not in warning


@pytest.mark.parametrize(
    'status,err_code,estop,expected',
    [
        (Status.ERROR, 1, False, 1),
        (Status.ERROR, 0, False, ERR_CONTROLLER),
        (Status.ERROR, -12, False, -12),
        (Status.STOPPED, 0, False, ERR_STOPPED),
        (Status.ESTOP, -6, True, ERR_STOPPED),
        (Status.RUNNING, 0, True, ERR_STOPPED),
        (Status.RUNNING, 0, False, OK),
        (Status.DONE, 0, False, OK),
    ],
)
def test_install_ack_checks_current_failure_before_success(status, err_code, estop, expected):
    def snapshot():
        tel = _telemetry(status, abi_magic=IPC_ABI_MAGIC,
                         abi_version=IPC_ABI_VERSION, t_mono=time.monotonic(),
                         install_seq=7, ack_seq=7, estop=estop)
        tel['err_code'] = err_code
        return tel

    client = SimpleNamespace(snapshot=snapshot)
    assert CommandClient.wait_installed(client, 7, Mode.TRACK_HYBRID) == expected


def test_install_wait_ignores_error_from_previous_request():
    samples = iter([
        dict(status=Status.ERROR, install_seq=6, ack_seq=6, err_code=1),
        dict(status=Status.RUNNING, install_seq=7, ack_seq=7, err_code=0),
    ])

    def snapshot():
        return dict(mode=int(Mode.TRACK_HYBRID), abi_magic=IPC_ABI_MAGIC,
                    abi_version=IPC_ABI_VERSION, t_mono=time.monotonic(),
                    **next(samples))

    client = SimpleNamespace(snapshot=snapshot)
    assert CommandClient.wait_installed(client, 7, Mode.TRACK_HYBRID) == OK


@pytest.mark.parametrize(
    'failure_status,install_result,return_code,warning_count',
    [
        (Status.ERROR, OK, 0, 1),
        (Status.ESTOP, OK, 130, 1),
        (Status.ERROR, 1, 0, 0),
    ],
)
def test_quiet_gamepad_reports_immediate_hybrid_failure(
    monkeypatch, capsys, failure_status, install_result, return_code, warning_count,
):
    """Exercise the actual L3 install and loop, with no pad, IPC or robot."""
    tel = _telemetry(mode=Mode.SERVO_TWIST)
    clicks = chain([True], repeat(False))  # L3, then no further clicks.
    src = SimpleNamespace(
        start=lambda: None, close=lambda: None,
        snapshot=lambda: dict(
            connected=True, armed=True, twist=[0.] * 6, axes=[0.] * 6,
            buttons=[0.] * 16, hz=100., l3=True, r3=False,
            l3_edge=next(clicks), r3_edge=False,
        ),
    )

    def set_mode(req):
        tel['mode'] = int(req.mode)
        return 7

    def wait_installed(seq, mode):
        if mode == Mode.TRACK_HYBRID:
            # The install succeeded, but the controller fails before the
            # first ordinary telemetry poll after the acknowledgement.
            tel.update(status=int(failure_status), err_code=1,
                       msg='qpik_fault:stop:publication_infeasible')
            return install_result
        return OK

    client = SimpleNamespace(
        snapshot=lambda: dict(tel), set_mode=set_mode, wait_installed=wait_installed,
        stop=lambda: None, close=lambda: None,
    )
    sleeps = iter([False, True])

    def sleep(_):
        if next(sleeps):
            raise KeyboardInterrupt

    monkeypatch.setattr(gamepad.sys, 'argv', ['gamepad', '--no-capture-y', '--no-vessel-b'])
    monkeypatch.setattr(gamepad, 'CommandClient', lambda **_: client)
    monkeypatch.setattr(gamepad, 'TwistBus', lambda **_: SimpleNamespace(write=lambda *a, **kw: None, close=lambda: None))
    monkeypatch.setattr(gamepad, 'GamepadTwistSource', lambda **_: src)
    monkeypatch.setattr(gamepad, 'program_is_running', lambda: False)
    monkeypatch.setattr(gamepad, 'pad_hybrid_payload', lambda *a, **kw: {})
    monkeypatch.setattr(gamepad, 'desired_z_n', lambda: 5.)
    monkeypatch.setattr(gamepad.time, 'sleep', sleep)
    assert gamepad.main() == return_code
    output = capsys.readouterr().out
    assert '[STATE] ready' in output
    assert output.count('exited unexpectedly') == warning_count
    assert 'publication_infeasible' in output
    if install_result != OK:
        assert '[MODE] TRACK_HYBRID install failed' in output
        assert '[MODE] TRACK_HYBRID pad+force' not in output
