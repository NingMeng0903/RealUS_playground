"""Worker pacing against a simulated clock/transport; no sockets or hardware."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.apps import contact_qp_features as worker
from peirastic.contact_qp.types import ContactObservation


class WorkerHarness:
    def __init__(self, monkeypatch, durations, arrivals, interrupt_at=None):
        self.now = 1.
        self.durations = iter(durations)
        self.pending = list(enumerate(arrivals))
        self.interrupt_at = interrupt_at
        self.starts = []
        self.processed = []
        self.sent = []
        self.closed = []
        self.terminated = False
        self.sub = SimpleNamespace(setsockopt=lambda *a: None, connect=lambda *a: None,
            poll=self.poll, recv_multipart=self.receive, close=lambda: self.closed.append('sub'))
        self.pub = SimpleNamespace(setsockopt=lambda *a: None, bind=lambda *a: None,
            send_multipart=lambda parts, **kw: self.sent.append(json.loads(parts[1])),
            close=lambda: self.closed.append('pub'))
        context = SimpleNamespace(socket=lambda kind: self.sub if kind == 1 else self.pub,
                                  term=self.terminate)
        monkeypatch.setitem(sys.modules, 'zmq', SimpleNamespace(Context=lambda: context,
            SUB=1, PUB=2, RCVHWM=3, SUBSCRIBE=4, SNDHWM=5, LINGER=6, NOBLOCK=7,
            Again=type('Again', (Exception,), {})))
        monkeypatch.setitem(sys.modules, 'rm75_control.control.admittance_common.cpu_resources',
                            SimpleNamespace(prepare_background_cpus=lambda: []))
        monkeypatch.setattr(worker, 'time', SimpleNamespace(monotonic=lambda: self.now,
                                                          perf_counter=lambda: self.now))
        monkeypatch.setattr(worker, 'process_parts', self.process)

    def poll(self, timeout_ms):
        end = self.now + timeout_ms / 1000.
        arrival = self.pending[0][1] if self.pending else float('inf')
        wake = max(self.now, min(end, arrival))
        if self.interrupt_at is not None and wake >= self.interrupt_at:
            self.now = self.interrupt_at
            raise KeyboardInterrupt
        self.now = wake
        return arrival <= self.now

    def receive(self):
        seq, capture = self.pending.pop(0)
        assert capture <= self.now
        return [b'topic', json.dumps(dict(seq=seq, capture=capture)).encode(), b'jpeg']

    def process(self, parts, extractor, received):
        meta = json.loads(parts[1])
        self.starts.append(self.now)
        self.processed.append(meta)
        self.now += next(self.durations)
        return ContactObservation(meta['seq'], 'camera:instance',
            meta['capture'] - extractor.config.effective_delay_s, received,
            [.8, .6, .7], [True] * 3, 'registration', extractor.config.window_version)

    def terminate(self):
        self.terminated = True

    def run(self, frames, max_hz=20):
        worker.main(['--max-frames', str(frames), '--max-hz', str(max_hz)])
        assert self.closed == ['sub', 'pub'] and self.terminated


@pytest.mark.parametrize('durations', [
    [.035] * 5,
    [.080] * 5,
    [.080, .010, .035, .080, .010],
])
def test_start_intervals_include_compute_and_never_catch_up(monkeypatch, durations):
    arrivals = [1. + i * .001 for i in range(501)]
    harness = WorkerHarness(monkeypatch, durations, arrivals)
    harness.run(len(durations))
    np.testing.assert_allclose(np.diff(harness.starts), np.maximum(.050, durations[:-1]), atol=.001)
    assert np.all(np.diff(harness.starts) >= .050 - 1e-12)
    for start, meta, payload in zip(harness.starts, harness.processed, harness.sent):
        assert meta['capture'] == max(t for t in arrivals if t <= start)
        assert payload['effective_time_s'] == meta['capture'] - worker.FeatureConfig().effective_delay_s
        assert payload['received_time_s'] == start
    np.testing.assert_allclose([p['processing_s'] for p in harness.sent], durations, atol=1e-12)


def test_drains_frames_arriving_during_wait_and_at_deadline(monkeypatch):
    harness = WorkerHarness(monkeypatch, [.035, .035], [1., 1.01, 1.04, 1.049, 1.05])
    harness.run(2)
    assert [p['seq'] for p in harness.processed] == [0, 4]
    assert harness.starts == pytest.approx([1., 1.05], abs=.001)


def test_zero_rate_cap_starts_immediately_after_compute(monkeypatch):
    harness = WorkerHarness(monkeypatch, [.035] * 3, [1. + i * .001 for i in range(200)])
    harness.run(3, max_hz=0)
    np.testing.assert_allclose(np.diff(harness.starts), [.035, .035], atol=1e-12)


@pytest.mark.parametrize('interrupt_at,arrivals', [
    (1.02, []),  # idle input poll
    (1.045, [1., 1.01, 1.04]),  # waiting for next processing deadline
])
def test_interrupt_closes_transport_while_idle_or_rate_limited(monkeypatch, interrupt_at, arrivals):
    harness = WorkerHarness(monkeypatch, [.035], arrivals, interrupt_at)
    harness.run(0)
    assert len(harness.sent) == bool(arrivals)


def test_interrupt_during_processing_closes_transport(monkeypatch):
    harness = WorkerHarness(monkeypatch, [], [1.])
    def interrupted(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(worker, 'process_parts', interrupted)
    harness.run(0)
    assert not harness.sent
