"""Compensated external wrench from rolling pose/force buffer + phi."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, lfilter, lfilter_zi

from rm75_control.force.compensation import regressor as fid
from rm75_control.force.compensation.paths import CONFIG_FORCE, PHI_JSON


@dataclass
class ForceObserverConfig:
    phi_path: Path = PHI_JSON
    phi_source: str = "phi_recommended"
    force_sensor: Path = CONFIG_FORCE
    fc_hz: float = 2.5
    buffer_s: float = 4.0
    min_samples: int = 35
    use_inertia: bool = False
    poll_hz: float = 100.0
    # Causal online estimator (Keemink 2018 G2: keep filter order low and the
    # cutoff high to avoid the phase lag that destabilises the marginally passive
    # virtual-inertia model). Order 2 Butterworth realised as a persistent biquad.
    causal_fc_hz: float = 6.0
    causal_order: int = 2
    causal_history: int = 5


@dataclass
class ForceSampleBuffer:
    max_len: int
    t: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)
    force: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.t = deque(maxlen=self.max_len)
        self.pose = deque(maxlen=self.max_len)
        self.force = deque(maxlen=self.max_len)

    def append(self, t_s: float, pose6: np.ndarray, force6: np.ndarray) -> None:
        self.t.append(t_s)
        self.pose.append(np.asarray(pose6, dtype=float))
        self.force.append(np.asarray(force6, dtype=float))

    def __len__(self) -> int:
        return len(self.t)


class CompensatedForceObserver:
    def __init__(self, cfg: ForceObserverConfig) -> None:
        self._fid = fid
        self.cfg = cfg
        self.phi = self._load_phi(cfg.phi_path, cfg.phi_source)
        self.frame = fid.FrameConfig.from_yaml(cfg.force_sensor)
        max_len = max(cfg.min_samples + 5, int(cfg.buffer_s * cfg.poll_hz) + 5)
        self.buf = ForceSampleBuffer(max_len=max_len)

        # --- causal online estimator state (O(1) per tick) ---
        k = max(2, int(cfg.causal_history))
        self._pose_ring: deque = deque(maxlen=k)
        self._t_ring: deque = deque(maxlen=k)
        self._n_updates = 0
        fs = float(cfg.poll_hz)
        wn = min(float(cfg.causal_fc_hz) / (0.5 * fs), 0.99)
        self._lpf_b, self._lpf_a = butter(int(cfg.causal_order), wn, btype="low")
        self._lpf_zi_unit = lfilter_zi(self._lpf_b, self._lpf_a)  # (order,)
        self._lpf_zi: np.ndarray | None = None  # (order, 6), lazily warm-started
        self._f_ext_last = np.zeros(6, dtype=float)
        # Compensated but UNfiltered wrench from the latest update(): the
        # Dimeas instability index must see the 5.8-20 Hz band the 6 Hz
        # control LPF removes (feed this to the index, f_ext_filt to control).
        self.f_ext_raw_last = np.zeros(6, dtype=float)

    @staticmethod
    def _load_phi(path: Path, source: str) -> np.ndarray:
        data = json.loads(path.read_text())
        if source not in data:
            raise SystemExit(f"Key '{source}' not in {path}")
        return np.array([data[source][k] for k in fid.PHI_NAMES])

    def append(self, t_s: float, pose6: np.ndarray, force_raw: np.ndarray) -> None:
        self.buf.append(t_s, pose6, force_raw)

    def ready(self) -> bool:
        return len(self.buf) >= self.cfg.min_samples

    def latest_wrench(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return (signed_filtered_raw, f_ext).

        Return (signed_filtered_raw, f_ext) in the link_7 / sensor frame.
        """
        if not self.ready():
            return None
        t = np.asarray(self.buf.t)
        pose = np.asarray(self.buf.pose)
        force = np.asarray(self.buf.force)
        W, Y = self._fid.build_dataset(
            pose, force, t, self.frame, fc=self.cfg.fc_hz, use_inertia=self.cfg.use_inertia
        )
        k = len(t) - 1
        sl = slice(6 * k, 6 * k + 6)
        raw_show = Y[sl].copy()
        f_ext = (Y[sl] - W[sl] @ self.phi).reshape(6)
        return raw_show, f_ext

    def update(
        self,
        t_s: float,
        regressor_pose: np.ndarray,
        force_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Causal link_7-frame external wrench (before ``wrench_link7_to_tcp``)."""
        self._pose_ring.append(np.asarray(regressor_pose, dtype=float).reshape(6).copy())
        self._t_ring.append(float(t_s))
        self._n_updates += 1

        poses = np.asarray(self._pose_ring, dtype=float)
        times = np.asarray(self._t_ring, dtype=float)
        W_row, _g_s = self._fid.regressor_row_causal(
            poses, times, self.frame, use_inertia=self.cfg.use_inertia
        )

        signed = self._fid.apply_sign(
            np.asarray(force_raw, dtype=float), self.frame.force_sign
        )
        f_ext_raw = signed - W_row @ self.phi  # (6,)
        self.f_ext_raw_last = f_ext_raw.copy()

        if self._lpf_zi is None:
            # Warm-start each channel at its first value → no startup transient.
            self._lpf_zi = np.outer(self._lpf_zi_unit, f_ext_raw)
        f_ext_filt, self._lpf_zi = lfilter(
            self._lpf_b, self._lpf_a, f_ext_raw[None, :], axis=0, zi=self._lpf_zi
        )
        f_ext_filt = f_ext_filt.reshape(6)
        self._f_ext_last = f_ext_filt
        return signed, f_ext_filt

    def ready_causal(self) -> bool:
        """Warm-up gate for the causal path (filter settled + history filled)."""
        return self._n_updates >= self.cfg.min_samples

    @property
    def n_samples(self) -> int:
        """Number of causal update() calls seen (for warm-up progress messages)."""
        return self._n_updates

    @classmethod
    def from_yaml(cls, raw: dict) -> CompensatedForceObserver:
        f = raw.get("force", {})
        fc_cfg = float(yaml.safe_load(CONFIG_FORCE.read_text()).get("filtfilt_cutoff_hz", 2.5))
        fc_hz = float(f.get("fc_hz", fc_cfg))
        timing = raw.get("timing", {})
        dt_ms = float(timing.get("dt_ms", 10.0))
        rp = raw.get("realtime_push", {})
        cycle = int(rp.get("cycle", max(1, int(round(dt_ms / 5.0)))))
        poll_hz = 1000.0 / (cycle * 5.0)
        return cls(
            ForceObserverConfig(
                phi_path=PHI_JSON,
                phi_source=str(f.get("phi_source", "phi_recommended")),
                fc_hz=fc_hz,
                buffer_s=float(f.get("buffer_s", 4.0)),
                min_samples=int(f.get("min_samples", 35)),
                use_inertia=bool(f.get("use_inertia", False)),
                poll_hz=poll_hz,
                causal_fc_hz=float(f.get("causal_fc_hz", 6.0)),
                causal_order=int(f.get("causal_order", 2)),
                causal_history=int(f.get("causal_history", 5)),
            )
        )
