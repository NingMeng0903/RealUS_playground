"""Gamepad mode failure reconciliation; no IPC/robot process started."""
from types import SimpleNamespace
import pytest
from peirastic.apps.gamepad import _install_mode, pad_hybrid_active
from peirastic.core.ipc import Status
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
