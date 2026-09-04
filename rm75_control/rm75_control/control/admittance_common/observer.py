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
    use_dynamic_kinematics: bool = False
    use_rotational_inertia: bool = False
    dynamic_kinematics_mode: str = "off"
    delay_s: float = 0.0
    rail_stationary_only: bool = True
    poll_hz: float = 100.0
    # Certificate 1: 20 Hz 1st-order (~7.9 ms at 3 Hz) instead of 10 Hz
    # 2nd-order (~23.1 ms).  Hardware crossover check is deferred to contact.
    causal_fc_hz: float = 20.0
    causal_order: int = 1
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
        self.f_ext_dynamic_candidate = np.zeros(6, dtype=float)
        self._arm_obs = None
        self._delay_ring = None
        self._kin = None

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
        *,
        q_meas: np.ndarray | None = None,
        qdot_sdk: np.ndarray | None = None,
        rail_locked: bool = True,
        sensor_age_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Causal link_7-frame external wrench (before ``wrench_link7_to_tcp``).

        ``t_s`` must be the snapshot receive time (``snap.t_s``), not loop
        ``now - total_t0``. Control output is gravity-only unless mode is
        ``apply`` and gates pass. ``observe`` stores a dynamic candidate only.
        """
        self._pose_ring.append(np.asarray(regressor_pose, dtype=float).reshape(6).copy())
        self._t_ring.append(float(t_s))
        self._n_updates += 1

        signed = self._fid.apply_sign(
            np.asarray(force_raw, dtype=float), self.frame.force_sign
        )
        use_dyn = bool(self.cfg.use_dynamic_kinematics)
        use_rot = bool(self.cfg.use_rotational_inertia)
        mode = str(self.cfg.dynamic_kinematics_mode)
        stale = sensor_age_s is not None and float(sensor_age_s) > 0.015
        rail_ok = (not self.cfg.rail_stationary_only) or bool(rail_locked)
        apply_dyn = (
            use_dyn
            and mode == "apply"
            and rail_ok
            and not stale
            and q_meas is not None
        )

        W_grav, W_dyn = self._rows(t_s, q_meas, qdot_sdk, rail_locked, use_dyn, use_rot)
        f_grav = signed - W_grav @ self.phi
        f_dyn = signed - W_dyn @ self.phi
        self.f_ext_dynamic_candidate = f_dyn.copy()
        residual = f_dyn if apply_dyn else f_grav
        self.f_ext_raw_last = residual.copy()

        if self._lpf_zi is None:
            self._lpf_zi = np.outer(self._lpf_zi_unit, residual)
        f_ext_filt, self._lpf_zi = lfilter(
            self._lpf_b, self._lpf_a, residual[None, :], axis=0, zi=self._lpf_zi
        )
        f_ext_filt = f_ext_filt.reshape(6)
        self._f_ext_last = f_ext_filt
        return signed, f_ext_filt

    def _rows(self, t_s, q_meas, qdot_sdk, rail_locked, use_dyn, use_rot):
        from rm75_control.force.compensation.v2.regressor_v2 import regressor_row_v2

        poses = np.asarray(self._pose_ring, dtype=float)
        times = np.asarray(self._t_ring, dtype=float)
        W_legacy, g_s = self._fid.regressor_row_causal(
            poses, times, self.frame, use_inertia=False
        )
        W_grav = regressor_row_v2(
            np.zeros(3), g_s, np.zeros(3), np.zeros(3),
            use_dynamic_kinematics=False, use_rotational_inertia=False,
        )
        if not use_dyn or q_meas is None:
            return W_grav, W_legacy if self.cfg.use_inertia else W_grav
        q8, qd8, qdd8, var8 = self._joint_state(t_s, q_meas, qdot_sdk, rail_locked)
        if not np.isfinite(var8[1:]).all() or float(np.max(var8[1:])) > 50.0:
            return W_grav, W_grav
        delay = float(self.cfg.delay_s)
        if abs(delay) > 1e-6:
            q8, qd8, qdd8 = self._delayed(t_s - delay, q8, qd8, qdd8)
        a, w, al, g = self._classical(q8, qd8, qdd8)
        W_dyn = regressor_row_v2(
            a, g, w, al,
            use_dynamic_kinematics=True,
            use_rotational_inertia=use_rot,
        )
        return W_grav, W_dyn

    def _joint_state(self, t_s, q_meas, qdot_sdk, rail_locked):
        from rm75_control.force.compensation.v2.joint_observer import ArmJointObserver

        if self._arm_obs is None:
            self._arm_obs = ArmJointObserver(rail_locked=bool(rail_locked))
        self._arm_obs.rail_locked = bool(rail_locked)
        q = np.asarray(q_meas, dtype=float).reshape(-1)
        rail = float(q[0]) if q.size == 8 else 0.0
        q_arm = q[1:8] if q.size == 8 else q[:7]
        qd = None
        if qdot_sdk is not None:
            qd = np.asarray(qdot_sdk, dtype=float).reshape(-1)
            if qd.size == 8:
                qd = qd[1:8]
        return self._arm_obs.step(t_s, q_arm, qd, rail_q=rail)

    def _delayed(self, t_query, q, qd, qdd):
        from rm75_control.force.compensation.v2.joint_observer import DelayRing

        if self._delay_ring is None:
            self._delay_ring = DelayRing()
        self._delay_ring.push(self._t_ring[-1], q, qd, qdd)
        return self._delay_ring.at(t_query)

    def _classical(self, q8, qd8, qdd8):
        from rm75_control.control.joint_admittance_8dof.model import shared_robot_kinematics

        if self._kin is None:
            self._kin = shared_robot_kinematics()
        mot = self._kin.frame_classical_motion(q8, qd8, qdd8, "link_7")
        g = self._kin.gravity_link7(q8, np.asarray(self.frame.gravity_base, dtype=float))
        return mot.linear_acceleration, mot.angular_velocity, mot.angular_acceleration, g

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
        from rm75_control.force.compensation.v2.flags import resolve_online_flags

        mode, use_dyn, use_rot = resolve_online_flags(f)
        return cls(
            ForceObserverConfig(
                phi_path=PHI_JSON,
                phi_source=str(f.get("phi_source", "phi_recommended")),
                fc_hz=fc_hz,
                buffer_s=float(f.get("buffer_s", 4.0)),
                min_samples=int(f.get("min_samples", 35)),
                use_inertia=bool(f.get("use_inertia", False)),
                use_dynamic_kinematics=use_dyn,
                use_rotational_inertia=use_rot,
                dynamic_kinematics_mode=mode.value,
                delay_s=float(f.get("delay_online_effective_s", f.get("delay_s", 0.0))),
                rail_stationary_only=bool(f.get("rail_stationary_only", True)),
                poll_hz=poll_hz,
                causal_fc_hz=float(f.get("causal_fc_hz", 20.0)),
                causal_order=int(f.get("causal_order", 1)),
                causal_history=int(f.get("causal_history", 5)),
            )
        )
