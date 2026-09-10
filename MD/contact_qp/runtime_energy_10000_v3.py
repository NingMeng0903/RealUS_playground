"""Reproduce the hardware-free 10,000-interval characterization; no acoustic model."""
import json
import time
import tracemalloc

import numpy as np

from peirastic.contact_qp.energy import EnergyLedger, PortBounds
from peirastic.contact_qp.runtime_energy import RuntimeEnergy

x = np.array([1., 0, 0, 0, 0, 0])
v = .001 * x
r = RuntimeEnergy(EnergyLedger(1, 2, .1, PortBounds(calibration_version="physical-v1")),
                  command_budget_enforced=True)


def obs(t, identity):
    return r.observe_port(-x, v, source_t_s=t, source_id=identity, now_s=t,
                          valid=True, time_aligned=True, physical_w_checked=True,
                          reference="tcp_tool", calibration_version="physical-v1")


obs(1., 0)
times = []
retained = 0
tracemalloc.start()
begin = time.perf_counter()
for i in range(10000):
    t = 1. + i * .005
    end = 1. + (i + 1) * .005
    start = time.perf_counter_ns()
    c = r.snapshot(now_s=t, hold_s=.005, wrench_environment=-x, source_t_s=t)
    assert r.reserve(i, c, v, now_s=t)
    r.publication_started(i)
    assert obs(end, i + 1)
    events = r.drain_events()
    retained += len(events["runtime"]) + len(events["ledger"])
    times.append((time.perf_counter_ns() - start) / 1e6)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
assert abs(r.ledger.balance_j - .95) < 1e-10
assert r.ledger.reserved_j < 1e-12
print(json.dumps(dict(intervals=10000, wall_s=time.perf_counter() - begin,
                      p50_ms=float(np.percentile(times, 50)),
                      p99_ms=float(np.percentile(times, 99)), max_ms=max(times),
                      balance_j=r.ledger.balance_j, reserved_j=r.ledger.reserved_j,
                      drained_events=retained,
                      retained_events=len(r.events) + len(r.ledger.events),
                      memory_current_bytes=current, memory_peak_bytes=peak,
                      profiling="tracemalloc enabled; software characterization, not RT certification"),
                 indent=2))
