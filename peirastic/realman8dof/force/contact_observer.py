"""Nonblocking, bounded snapshot ingress. Image processing lives in its own process."""
from __future__ import annotations

import json

from peirastic.contact_qp.features import LatestObservation
from peirastic.contact_qp.types import ContactObservation


class ConfidenceSubscriber:
    def __init__(self, endpoint="tcp://127.0.0.1:17361", *, socket=None):
        self._context = None
        if socket is None:
            import zmq
            self._context = zmq.Context()
            socket = self._context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVHWM, 2)
            socket.setsockopt(zmq.SUBSCRIBE, b"contact_qp_features_v1")
            socket.connect(endpoint)
        self.socket = socket
        self.latest = LatestObservation()
        self.rejected = 0
        self.last_error = ""

    def snapshot(self):
        # At most four small JSON messages; never wait or decode B-mode here.
        for _ in range(4):
            if not self.socket.poll(0):
                break
            parts = self.socket.recv_multipart()
            try:
                if len(parts) != 2 or parts[0] != b"contact_qp_features_v1" or len(parts[1]) > 8192:
                    raise ValueError("invalid feature envelope")
                self.latest.accept(ContactObservation.from_dict(json.loads(parts[1])))
            except (ValueError, TypeError, KeyError) as exc:
                self.rejected += 1
                self.last_error = str(exc)
        return self.latest.observation

    def close(self):
        self.socket.close()
        if self._context is not None:
            self._context.term()
