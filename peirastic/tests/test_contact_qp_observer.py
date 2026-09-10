import json

from peirastic.contact_qp.types import ContactObservation
from peirastic.realman8dof.force.contact_observer import ConfidenceSubscriber


class Socket:
    def __init__(self, messages):
        self.messages = messages
        self.poll_timeouts = []

    def poll(self, timeout):
        self.poll_timeouts.append(timeout)
        return len(self.messages)

    def recv_multipart(self):
        return self.messages.pop(0)


def packet(seq):
    obs = ContactObservation(seq, "source", 1+seq*.04, 1.2+seq*.04,
                             [.1, .2, .3], [True]*3, "reg", "win")
    return [b"contact_qp_features_v1", json.dumps(obs.to_dict()).encode()]


def test_control_ingress_is_bounded_and_never_waits_for_image_work():
    socket = Socket([packet(i) for i in range(12)])
    sub = ConfidenceSubscriber(socket=socket)
    assert sub.snapshot().frame_seq == 3
    assert len(socket.messages) == 8
    assert set(socket.poll_timeouts) == {0}
    assert sub.snapshot().frame_seq == 7


def test_bad_envelopes_do_not_erase_last_valid_snapshot_or_refresh_its_age():
    socket = Socket([packet(1), [b"contact_qp_features_v1", b"{}"], packet(0)])
    sub = ConfidenceSubscriber(socket=socket)
    obs = sub.snapshot()
    assert obs.frame_seq == 1 and sub.rejected == 1
    assert not obs.fresh(3., .3)
