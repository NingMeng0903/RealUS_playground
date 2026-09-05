"""Causal, observation-only model of the actual position/velocity send paths.

Arm inputs are position targets, rail inputs are worker-written velocities.
There is deliberately no controller-output API: fitting/validation does not
silently enable compensation. Transport delay uses monotonic send timestamps.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ActuatorModel:
    delay_s: float
    tau_s: float
    gain: float = 1.0

    def __post_init__(self):
        if not np.isfinite([self.delay_s, self.tau_s, self.gain]).all():
            raise ValueError("execution model parameters must be finite")
        if self.delay_s < 0 or self.tau_s < 0 or self.gain <= 0:
            raise ValueError("execution model requires delay/tau >= 0 and gain > 0")


class _Channel:
    def __init__(self, model, initial, now):
        self.model, self.origin = model, float(initial)
        self.value, self.target, self.time = float(initial), float(initial), float(now)
        self.queue = deque()
        self.last_sent = float("-inf")

    def send(self, now, value):
        if now < self.last_sent:
            raise ValueError("command time moved backwards")
        if not np.isfinite([now, value]).all():
            raise ValueError("command must be finite")
        self.last_sent = float(now)
        self.queue.append((float(now) + self.model.delay_s, float(value)))

    def _advance(self, now):
        dt = max(0.0, now - self.time)
        target = self.origin + self.model.gain * (self.target - self.origin)
        self.value = (target if self.model.tau_s <= 1e-12 else
                      target + (self.value - target) * np.exp(-dt / self.model.tau_s))
        self.time = float(now)

    def sample(self, now):
        if now < self.time:
            raise ValueError("observation time moved backwards; start a new episode")
        while self.queue and self.queue[0][0] <= now + 1e-12:
            at, target = self.queue.popleft()
            self._advance(min(now, max(at, self.time)))
            self.target = target
        self._advance(float(now))
        return float(self.value)


class ExecutionObserver:
    """Independent scalar lag models, locally mapped through the measured J."""
    mode = "observe"

    def __init__(self, arm, rail, *, model_hash="", validated=False):
        if len(arm) != 7:
            raise ValueError("execution model needs seven arm position channels")
        self.arm_models, self.rail_model = tuple(arm), rail
        self.model_hash, self.validated = str(model_hash), bool(validated)
        self.channels = None
        self.last_time = None
        self.last_arm = None
        self.last_rail_seq = 0

    @classmethod
    def from_file(cls, path):
        data = Path(path).read_bytes()
        raw = json.loads(data)
        if raw.get("schema_version") != 1 or raw.get("mode", "observe") != "observe":
            raise ValueError("only schema_version=1, mode=observe is supported")
        if not raw.get("provenance"):
            raise ValueError("execution model must identify its independent fitting data")
        return cls([ActuatorModel(**v) for v in raw["arm_position"]],
                   ActuatorModel(**raw["rail_velocity"]),
                   model_hash=hashlib.sha256(data).hexdigest(),
                   validated=raw.get("validated", False))

    def reset(self, now, q_measured, rail_velocity=0.0):
        q = np.asarray(q_measured, dtype=float).reshape(8)
        if not np.isfinite(q).all():
            raise ValueError("observer seed must be measured and finite")
        self.channels = [_Channel(m, q[i + 1], now) for i, m in enumerate(self.arm_models)]
        self.channels.append(_Channel(self.rail_model, rail_velocity, now))
        self.last_time, self.last_arm = float(now), q[1:].copy()
        self.last_rail_seq = 0

    def record_arm_send(self, now, q_position):
        if self.channels is None:
            return
        q = np.asarray(q_position, dtype=float).reshape(8)
        for channel, value in zip(self.channels[:7], q[1:]):
            channel.send(now, value)

    def record_rail_write(self, now, command_seq, velocity):
        if self.channels is None or command_seq <= self.last_rail_seq:
            return
        self.channels[7].send(now, velocity)
        self.last_rail_seq = int(command_seq)

    def sample(self, now, jacobian):
        if self.channels is None or now <= self.last_time:
            return np.full(6, np.nan)
        values = np.array([c.sample(now) for c in self.channels])
        qdot = np.r_[values[7], (values[:7] - self.last_arm) / (now - self.last_time)]
        self.last_arm, self.last_time = values[:7], float(now)
        return np.asarray(jacobian, dtype=float).reshape(6, 8) @ qdot
