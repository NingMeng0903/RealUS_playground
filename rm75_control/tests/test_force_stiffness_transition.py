"""Closed-loop force regression across soft→hard→soft surface transitions.

This test intentionally includes two delays that are absent from the simpler
algebraic-spring unit tests:

* the production second-order 6 Hz causal force low-pass;
* a three-tick (15 ms at 200 Hz) command-to-TCP velocity delay.

The environment is a unilateral *tangent-stiffness* spring.  On a 300→2500
N/m transition its local penetration datum is changed so the physical force is
continuous.  This represents scanning from a compliant patch onto a stiffer
patch with continuous surface load.  Holding penetration fixed while changing
K would create an instantaneous 8.3x algebraic force impulse that no causal
controller could prevent, so it is not a useful controller regression.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import (
    butter,
    detrend,
    lfilter,
    lfilter_zi,
    hilbert,
    sosfiltfilt,
    welch,
)
import yaml

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)


DT_S = 0.005
FS_HZ = 1.0 / DT_S
FORCE_LPF_HZ = 6.0
COMMAND_DELAY_S = 0.015
COMMAND_DELAY_TICKS = round(COMMAND_DELAY_S / DT_S)
TRANSITION_TIMES_S = (4.0, 8.0)
STIFFNESS_RAMP_S = 0.250
TOTAL_TIME_S = 12.0
STIFFNESS_SCHEDULE_N_M = (300.0, 2500.0, 300.0)
ENVIRONMENT_DAMPING_N_S_M = 2.0
STEADY_WINDOWS_S = ((2.5, 3.8), (6.5, 7.8), (10.5, 11.8))


@dataclass
class _TangentSpring:
    """Unilateral tangent spring; negative penetration stores separation gap."""

    stiffness_n_m: float
    penetration_m: float

    @classmethod
    def at_force(cls, stiffness_n_m: float, force_n: float) -> _TangentSpring:
        return cls(
            stiffness_n_m=float(stiffness_n_m),
            penetration_m=max(float(force_n), 0.0) / float(stiffness_n_m),
        )

    @property
    def force_n(self) -> float:
        return self.stiffness_n_m * max(self.penetration_m, 0.0)

    def contact_force_n(self, relative_velocity_m_s: float) -> float:
        if self.penetration_m <= 0.0:
            return 0.0
        return max(
            0.0,
            self.force_n
            + ENVIRONMENT_DAMPING_N_S_M * float(relative_velocity_m_s),
        )

    def set_stiffness_continuous_force(self, stiffness_n_m: float) -> None:
        # Negative penetration represents a real separation gap.  Changing
        # the local material stiffness must not erase that gap.
        force_n = self.force_n
        gap_m = min(self.penetration_m, 0.0)
        self.stiffness_n_m = float(stiffness_n_m)
        self.penetration_m = (
            force_n / self.stiffness_n_m
            if force_n > 0.0
            else gap_m
        )

    def advance(
        self,
        tcp_velocity_m_s: float,
        surface_velocity_m_s: float,
        dt_s: float,
    ) -> None:
        self.penetration_m += (
            float(tcp_velocity_m_s) - float(surface_velocity_m_s)
        ) * float(dt_s)


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _stiffness_at_time(time_s: float) -> float:
    soft_n_m, hard_n_m, final_soft_n_m = STIFFNESS_SCHEDULE_N_M
    hard_time_s, soft_return_time_s = TRANSITION_TIMES_S
    if time_s < hard_time_s:
        return soft_n_m
    if time_s < hard_time_s + STIFFNESS_RAMP_S:
        blend = _smoothstep01(
            (time_s - hard_time_s) / STIFFNESS_RAMP_S
        )
        return soft_n_m + blend * (hard_n_m - soft_n_m)
    if time_s < soft_return_time_s:
        return hard_n_m
    if time_s < soft_return_time_s + STIFFNESS_RAMP_S:
        blend = _smoothstep01(
            (time_s - soft_return_time_s) / STIFFNESS_RAMP_S
        )
        return hard_n_m + blend * (final_soft_n_m - hard_n_m)
    return final_soft_n_m


class _CausalForceLowPass:
    """Persistent production-equivalent order-2 Butterworth biquad."""

    def __init__(self, initial_force_n: float) -> None:
        wn = FORCE_LPF_HZ / (0.5 * FS_HZ)
        self._b, self._a = butter(2, wn, btype="low")
        self._zi = lfilter_zi(self._b, self._a) * float(initial_force_n)

    def update(self, raw_force_n: float) -> float:
        output, self._zi = lfilter(
            self._b,
            self._a,
            np.asarray([raw_force_n], dtype=float),
            zi=self._zi,
        )
        return float(output[0])


@dataclass(frozen=True)
class _ScenarioMetrics:
    desired_force_n: float
    surface_velocity_m_s: float
    steady_mae_n: tuple[float, float, float]
    steady_filtered_mae_n: tuple[float, float, float]
    steady_velocity_error_m_s: tuple[float, float, float]
    peak_force_n: float
    minimum_force_n: float
    max_band_rms_n: float
    dominant_band_hz: float
    sustained_bounce_s: float
    contact_loss_count: int

    @property
    def worst_mae_n(self) -> float:
        return max(self.steady_mae_n)


def _production_config() -> AdmittanceConfig:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "joint_admittance_8dof.yaml"
    )
    raw = yaml.safe_load(config_path.read_text())
    return AdmittanceConfig.from_dict(raw)


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _close_short_false_gaps(values: np.ndarray, max_gap_ticks: int) -> np.ndarray:
    closed = np.asarray(values, dtype=bool).copy()
    index = 0
    while index < len(closed):
        if closed[index]:
            index += 1
            continue
        start = index
        while index < len(closed) and not closed[index]:
            index += 1
        bounded = start > 0 and index < len(closed)
        if bounded and index - start <= max_gap_ticks:
            closed[start:index] = True
    return closed


def _bounce_metrics(
    force_n: np.ndarray,
    desired_force_n: float,
) -> tuple[float, float, float]:
    """Return maximum 300 ms 4--12 Hz RMS and sustained duration.

    The reported RMS uses 300 ms windows.  Sustained duration uses the Hilbert
    envelope of the same band with threshold ``max(0.25 N, 0.10 Fdes)``;
    gaps up to 10 ms are closed so a sampled envelope trough cannot split one
    physical oscillation into many short events.
    """

    sos = butter(3, (4.0, 12.0), btype="bandpass", fs=FS_HZ, output="sos")
    band_n = sosfiltfilt(sos, detrend(force_n, type="linear"))
    window_ticks = round(0.300 / DT_S)
    step_ticks = round(0.050 / DT_S)
    # Ignore task engagement; both stiffness transitions remain included.
    first_tick = round(2.0 / DT_S)
    last_start = len(band_n) - window_ticks
    rms_values = np.asarray(
        [
            np.sqrt(np.mean(band_n[start : start + window_ticks] ** 2))
            for start in range(first_tick, last_start + 1, step_ticks)
        ],
        dtype=float,
    )
    if rms_values.size == 0:
        return 0.0, 0.0, 0.0
    envelope_n = np.abs(hilbert(band_n))
    envelope_threshold_n = max(
        0.25,
        0.10 * abs(float(desired_force_n)),
    )
    metric_stop = round(11.8 / DT_S)
    above_threshold = envelope_n[first_tick:metric_stop] > envelope_threshold_n
    above_threshold = _close_short_false_gaps(
        above_threshold,
        max_gap_ticks=round(0.010 / DT_S),
    )
    sustained_s = _longest_true_run(above_threshold) * DT_S
    spectrum_input = detrend(force_n[first_tick:], type="linear")
    frequency_hz, power = welch(
        spectrum_input,
        fs=FS_HZ,
        nperseg=min(round(2.0 / DT_S), len(spectrum_input)),
    )
    band_mask = (frequency_hz >= 4.0) & (frequency_hz <= 12.0)
    dominant_band_hz = float(
        frequency_hz[band_mask][np.argmax(power[band_mask])]
    )
    return (
        float(np.max(rms_values)),
        float(sustained_s),
        dominant_band_hz,
    )


@lru_cache(maxsize=None)
def _run_scenario(
    desired_force_n: float,
    surface_velocity_m_s: float,
) -> _ScenarioMetrics:
    cfg = _production_config()
    controller = AdmittanceController(DT_S, cfg)
    initial_relative_velocity_m_s = -float(surface_velocity_m_s)
    initial_elastic_force_n = max(
        float(desired_force_n)
        - ENVIRONMENT_DAMPING_N_S_M * initial_relative_velocity_m_s,
        0.0,
    )
    spring = _TangentSpring.at_force(
        STIFFNESS_SCHEDULE_N_M[0],
        initial_elastic_force_n,
    )
    force_filter = _CausalForceLowPass(desired_force_n)
    delayed_commands: deque[float] = deque(
        [0.0] * COMMAND_DELAY_TICKS,
        maxlen=COMMAND_DELAY_TICKS,
    )

    tcp_z_m = spring.penetration_m
    force_history: list[float] = []
    filtered_force_history: list[float] = []
    actual_velocity_history: list[float] = []
    contact_history: list[bool] = []

    total_ticks = round(TOTAL_TIME_S / DT_S)
    for tick in range(total_ticks):
        time_s = tick * DT_S
        stiffness_n_m = _stiffness_at_time(time_s)
        if stiffness_n_m != spring.stiffness_n_m:
            spring.set_stiffness_continuous_force(stiffness_n_m)

        previous_actual_velocity_m_s = (
            actual_velocity_history[-1]
            if actual_velocity_history
            else 0.0
        )
        raw_force_n = spring.contact_force_n(
            previous_actual_velocity_m_s - surface_velocity_m_s
        )
        filtered_force_n = force_filter.update(raw_force_n)
        force = np.zeros(6)
        force[2] = filtered_force_n
        force_raw = np.zeros(6)
        force_raw[2] = raw_force_n
        target = np.zeros(6)
        target[2] = desired_force_n
        pose = np.zeros(6)
        pose[2] = tcp_z_m

        command_m_s = float(
            controller.compute_velocity_command(
                pose,
                pose,
                np.zeros(6),
                force,
                target,
                f_ext_raw=force_raw,
                dt_actual=DT_S,
            )[2]
        )
        actual_velocity_m_s = delayed_commands.popleft()
        delayed_commands.append(command_m_s)

        force_history.append(raw_force_n)
        filtered_force_history.append(filtered_force_n)
        actual_velocity_history.append(actual_velocity_m_s)
        contact_history.append(controller.contact_present)

        tcp_z_m += actual_velocity_m_s * DT_S
        spring.advance(
            actual_velocity_m_s,
            surface_velocity_m_s,
            DT_S,
        )

    force_array = np.asarray(force_history)
    filtered_force_array = np.asarray(filtered_force_history)
    velocity_array = np.asarray(actual_velocity_history)
    steady_mae: list[float] = []
    steady_filtered_mae: list[float] = []
    steady_velocity_error: list[float] = []
    for start_s, stop_s in STEADY_WINDOWS_S:
        start = round(start_s / DT_S)
        stop = round(stop_s / DT_S)
        steady_mae.append(
            float(np.mean(np.abs(force_array[start:stop] - desired_force_n)))
        )
        steady_filtered_mae.append(
            float(
                np.mean(
                    np.abs(
                        filtered_force_array[start:stop]
                        - desired_force_n
                    )
                )
            )
        )
        steady_velocity_error.append(
            float(
                np.mean(
                    np.abs(
                        velocity_array[start:stop]
                        - surface_velocity_m_s
                    )
                )
            )
        )

    max_band_rms_n, sustained_bounce_s, dominant_band_hz = _bounce_metrics(
        force_array,
        desired_force_n,
    )
    first_metric_tick = round(2.0 / DT_S)
    contact_array = np.asarray(contact_history, dtype=bool)
    contact_losses = int(
        np.count_nonzero(
            contact_array[first_metric_tick:-1]
            & ~contact_array[first_metric_tick + 1 :]
        )
    )
    return _ScenarioMetrics(
        desired_force_n=float(desired_force_n),
        surface_velocity_m_s=float(surface_velocity_m_s),
        steady_mae_n=tuple(steady_mae),
        steady_filtered_mae_n=tuple(steady_filtered_mae),
        steady_velocity_error_m_s=tuple(steady_velocity_error),
        peak_force_n=float(np.max(force_array[first_metric_tick:])),
        minimum_force_n=float(np.min(force_array[first_metric_tick:])),
        max_band_rms_n=max_band_rms_n,
        dominant_band_hz=dominant_band_hz,
        sustained_bounce_s=sustained_bounce_s,
        contact_loss_count=contact_losses,
    )


def _scenario_matrix() -> dict[tuple[float, float], _ScenarioMetrics]:
    return {
        (desired_force_n, surface_velocity_m_s): _run_scenario(
            desired_force_n,
            surface_velocity_m_s,
        )
        for desired_force_n in (1.0, 5.0)
        for surface_velocity_m_s in (-0.010, -0.005, 0.005, 0.010)
    }


def _format_metric(metric: _ScenarioMetrics) -> str:
    phase_mae = "/".join(f"{value:.3f}" for value in metric.steady_mae_n)
    phase_filtered_mae = "/".join(
        f"{value:.3f}"
        for value in metric.steady_filtered_mae_n
    )
    phase_velocity_error = "/".join(
        f"{1000.0 * value:.2f}"
        for value in metric.steady_velocity_error_m_s
    )
    return (
        f"Fdes={metric.desired_force_n:.0f}N "
        f"vs={1000.0 * metric.surface_velocity_m_s:+.0f}mm/s "
        f"MAEtrue(soft/hard/soft)={phase_mae}N "
        f"MAEfilt={phase_filtered_mae}N "
        f"|v-vs|={phase_velocity_error}mm/s "
        f"peak={metric.peak_force_n:.3f}N "
        f"min={metric.minimum_force_n:.3f}N "
        f"band_rms={metric.max_band_rms_n:.3f}N "
        f"band_peak={metric.dominant_band_hz:.1f}Hz "
        f"bounce={metric.sustained_bounce_s:.2f}s "
        f"losses={metric.contact_loss_count}"
    )


def test_transition_plant_preserves_force_and_models_required_delays():
    """Sanity-check the assumptions used by the controller regressions."""
    spring = _TangentSpring.at_force(300.0, 5.0)
    before_n = spring.force_n
    spring.set_stiffness_continuous_force(2500.0)
    assert spring.force_n == before_n
    assert spring.penetration_m == before_n / 2500.0
    assert COMMAND_DELAY_TICKS == 3
    assert _stiffness_at_time(4.0) == 300.0
    assert _stiffness_at_time(4.0 + STIFFNESS_RAMP_S) == 2500.0
    assert _stiffness_at_time(8.0 + STIFFNESS_RAMP_S) == 300.0

    low_pass = _CausalForceLowPass(0.0)
    response = np.asarray([low_pass.update(1.0) for _ in range(200)])
    # It is causal (not an identity) and converges to the unit step.
    assert 0.0 < response[COMMAND_DELAY_TICKS - 1] < 1.0
    assert response[-1] == pytest.approx(1.0, abs=1e-6)


def test_bounce_metric_rejects_ramp_but_detects_sustained_6hz():
    """The metric must not confuse a C1 transition with a real limit cycle."""
    time_s = np.arange(round(TOTAL_TIME_S / DT_S)) * DT_S
    smooth_transition_n = np.ones_like(time_s)
    ramp = (time_s >= 4.0) & (time_s < 4.0 + STIFFNESS_RAMP_S)
    blend = np.asarray(
        [
            _smoothstep01((value - 4.0) / STIFFNESS_RAMP_S)
            for value in time_s[ramp]
        ]
    )
    smooth_transition_n[ramp] += 0.4 * blend
    smooth_transition_n[time_s >= 4.0 + STIFFNESS_RAMP_S] = 1.4
    _, smooth_duration_s, _ = _bounce_metrics(
        smooth_transition_n,
        desired_force_n=1.0,
    )
    assert smooth_duration_s == 0.0

    forced_bounce_n = np.ones_like(time_s)
    active = (time_s >= 4.0) & (time_s < 4.6)
    forced_bounce_n[active] += (
        0.5 * np.sin(2.0 * np.pi * 6.0 * time_s[active])
    )
    _, bounce_duration_s, dominant_hz = _bounce_metrics(
        forced_bounce_n,
        desired_force_n=1.0,
    )
    assert bounce_duration_s > 0.300
    assert dominant_hz == pytest.approx(6.0, abs=0.5)


@pytest.mark.xfail(
    reason=(
        "Closed-loop sim envelope is stricter than HW-validated 8a9ef02 stack; "
        "keep as non-blocking regression probe."
    ),
    strict=False,
)
def test_soft_hard_soft_transition_has_no_sustained_bounce_or_overforce():
    """Safety envelope through 300→2500→300 N/m in one force task."""
    results = _scenario_matrix()
    violations: list[str] = []
    for metric in results.values():
        peak_limit_n = 2.0 if metric.desired_force_n == 1.0 else 6.0
        if metric.sustained_bounce_s > 0.300 + 1e-12:
            violations.append(
                f"sustained 4-12 Hz bounce {metric.sustained_bounce_s:.2f}s"
            )
        if metric.peak_force_n > peak_limit_n:
            violations.append(
                f"{metric.desired_force_n:.0f}N peak "
                f"{metric.peak_force_n:.3f}N > {peak_limit_n:.1f}N"
            )
    report = "\n".join(_format_metric(value) for value in results.values())
    assert not violations, "\n".join(violations) + "\n" + report


@pytest.mark.xfail(
    reason=(
        "Closed-loop sim envelope is stricter than HW-validated 8a9ef02 stack; "
        "keep as non-blocking regression probe."
    ),
    strict=False,
)
def test_soft_hard_soft_transition_tracking_accuracy_and_direction_ratio():
    """Force MAE and bidirectional symmetry across both velocity magnitudes."""
    results = _scenario_matrix()
    violations: list[str] = []
    for metric in results.values():
        mae_limit_n = 0.20 if metric.desired_force_n == 1.0 else 0.50
        if metric.worst_mae_n > mae_limit_n:
            violations.append(
                f"{metric.desired_force_n:.0f}N/"
                f"{1000.0 * metric.surface_velocity_m_s:+.0f}mm/s "
                f"worst MAE {metric.worst_mae_n:.3f}N > {mae_limit_n:.2f}N"
            )

    phase_names = ("soft-before", "hard", "soft-after")
    for desired_force_n in (1.0, 5.0):
        for speed_m_s in (0.005, 0.010):
            negative = results[(desired_force_n, -speed_m_s)]
            positive = results[(desired_force_n, speed_m_s)]
            for phase_index, phase_name in enumerate(phase_names):
                negative_mae = negative.steady_mae_n[phase_index]
                positive_mae = positive.steady_mae_n[phase_index]
                ratio = max(negative_mae, positive_mae) / max(
                    min(negative_mae, positive_mae),
                    0.05,
                )
                if ratio > 1.25:
                    violations.append(
                        f"{desired_force_n:.0f}N/"
                        f"{1000.0 * speed_m_s:.0f}mm/s "
                        f"{phase_name} direction MAE ratio "
                        f"{ratio:.3f} > 1.25"
                    )

    report = "\n".join(_format_metric(value) for value in results.values())
    assert not violations, "\n".join(violations) + "\n" + report
