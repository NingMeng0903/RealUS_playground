"""Independent configuration for the handheld scan-style outer."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.delayed_kf import KfConfig
from peirastic.contact_qp.runtime_config import calibrated_geometry, load_study_config


def default_pad_scan_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'config/contact_qp/pad_scan.yaml'


def pad_visual_force_gate(force_n, start_n, end_n):
    """Smoothstep γ(F) on the pad band. Same plant as the v8 linear gate.

    Endpoints match clip((end-F)/(end-start), 0, 1); the cubic has a zero
    derivative at both ends so ordinary Fz ripple does not square-wave Ω.
    """
    if not math.isfinite(force_n):
        return 0.
    start = float(start_n)
    end = float(end_n)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('pad force gate requires finite start < end')
    t = float(np.clip((end - float(force_n)) / (end - start), 0., 1.))
    return float(t * t * (3. - 2. * t))


def load_pad_scan_config(path=None) -> dict:
    config = deepcopy(load_study_config(default_pad_scan_config_path() if path is None else path))
    geometry, _ = calibrated_geometry(config)
    feature = config['feature']
    fc = FeatureConfig(**feature['config'])
    kf = KfConfig(**config['image_kf'])
    if (feature['window_version'] != fc.window_version
            or feature['region_layout_version'] != fc.region_layout_version
            or kf.channels != fc.region_count or kf.observation_kind != 'regions'):
        raise ValueError('pad scan feature layout and KF channels must agree')
    if fc.width != 145:
        raise ValueError('pad scan requires full 145-column echo evidence')
    pad = config['pad']
    pad.setdefault('normal_allocation_margin_m_s', .0001)
    # A published command may arrive at a jittered interval. The original
    # name incorrectly suggested the lifecycle paced or delayed dispatch.
    pad.pop('minimum_publication_period_s', None)
    pad.setdefault('control_hold_s', .005)
    if (pad.get('policy') != 'pad_scan_v1'
            or config['qp'].get('allocation_policy') != 'delay_kf_cop_v1'):
        raise ValueError('pad scan requires pad_scan_v1 and delay_kf_cop_v1')
    pad.setdefault('alpha_min', .12)
    for name in ('normal_max_m_s', 'normal_allocation_margin_m_s', 'max_visual_omega_rad_s', 'angular_max_rad_s',
                 'angular_acceleration_rad_s2', 'cop_min_force_n', 'cop_max_m',
                 'repair_speed_cap_m_s', 'repair_pause_after_s', 'recover_hold_s',
                 'alpha_ramp_down_s', 'alpha_ramp_up_s', 'max_image_age_s',
                 'max_feedback_age_s', 'command_max_age_s', 'hard_stop_n',
                 'control_hold_s', 'force_gate_start_n', 'force_gate_end_n'):
        value = pad[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'pad.{name} must be finite and positive')
    if (pad['max_visual_omega_rad_s'] > pad['angular_max_rad_s']
            or pad['cop_max_m'] > geometry.half_length_m
            or pad['force_gate_start_n'] >= pad['force_gate_end_n']
            or pad['force_gate_end_n'] >= pad['hard_stop_n']
            or pad['normal_allocation_margin_m_s'] >= pad['normal_max_m_s']
            or pad['control_hold_s'] != .005
            or pad['control_hold_s'] >= pad['command_max_age_s']):
        raise ValueError('inconsistent pad scan limits')
    alpha_min = pad['alpha_min']
    if (isinstance(alpha_min, bool) or not isinstance(alpha_min, (float, int))
            or not math.isfinite(alpha_min) or not 0. < float(alpha_min) < 1.):
        raise ValueError('pad.alpha_min must be in (0, 1)')
    pad['alpha_min'] = float(alpha_min)
    if pad['hard_stop_n'] != 6. or pad.get('hard_stop_policy') != 'fresh_immediate_v1':
        raise ValueError('pad scan requires an immediate fresh 6 N stop')
    if not isinstance(config.get('feature_endpoint'), str) or not config['feature_endpoint']:
        raise ValueError('pad scan feature endpoint is required')
    evidence = pad.setdefault('evidence', {})
    evidence.setdefault('echo_moment_compress', 0.)
    evidence.setdefault('echo_moment_cubic_knee', 0.)
    evidence.setdefault('force_step_release_n', evidence.get('force_step_max_n', .8))
    evidence.setdefault('force_unstable_hold_s', 0.)
    compress = evidence['echo_moment_compress']
    if isinstance(compress, bool) or not isinstance(compress, (float, int)) or not math.isfinite(compress) or compress < 0:
        raise ValueError('pad.evidence.echo_moment_compress must be finite and non-negative')
    knee = evidence['echo_moment_cubic_knee']
    if isinstance(knee, bool) or not isinstance(knee, (float, int)) or not math.isfinite(knee) or knee < 0:
        raise ValueError('pad.evidence.echo_moment_cubic_knee must be finite and non-negative')
    release = evidence['force_step_release_n']
    if isinstance(release, bool) or not isinstance(release, (float, int)) or not math.isfinite(release) or release <= 0:
        raise ValueError('pad.evidence.force_step_release_n must be finite and positive')
    hold_s = evidence['force_unstable_hold_s']
    if isinstance(hold_s, bool) or not isinstance(hold_s, (float, int)) or not math.isfinite(hold_s) or hold_s < 0:
        raise ValueError('pad.evidence.force_unstable_hold_s must be finite and non-negative')
    if evidence['force_step_release_n'] > evidence['force_step_max_n']:
        raise ValueError('force_step_release_n cannot exceed force_step_max_n')
    pad.setdefault('visual_hold_s', 0.)
    pad.setdefault('visual_hold_omega_rad_s', .03)
    pad.setdefault('static_stick_m_s', .01)
    pad.setdefault('static_visual_omega_rad_s', .04)
    pad.setdefault('static_commit_s', .80)
    pad.setdefault('static_committed_omega_rad_s', .12)
    pad.setdefault('relative_spread_floor', .12)
    for name in ('visual_hold_s', 'visual_hold_omega_rad_s', 'static_stick_m_s',
                 'static_visual_omega_rad_s', 'static_commit_s', 'static_committed_omega_rad_s',
                 'relative_spread_floor'):
        value = pad[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'pad.{name} must be finite and non-negative')
    pad['static_committed_omega_rad_s'] = min(
        float(pad['static_committed_omega_rad_s']), float(pad['max_visual_omega_rad_s']))
    pad['static_visual_omega_rad_s'] = min(
        float(pad['static_visual_omega_rad_s']), float(pad['static_committed_omega_rad_s']))
    # The pad block is authoritative; copies used by the shared solver are
    # resolved here so the startup snapshot describes the executed limits.
    qp = config['qp']
    for name in ('max_visual_omega_rad_s', 'cop_min_force_n', 'cop_max_m', 'max_image_age_s', 'recover_hold_s'):
        qp[name] = pad[name]
    # Shared v8 policy stays at 4/4.5/6. Handheld γ uses pad.force_gate_* so a
    # wider pad band does not retune the scan DifferentialRepairConfig lock.
    qp.setdefault('differential_repair', {}).update(hard_stop_n=pad['hard_stop_n'])
    qp.update(alpha_min=pad['alpha_min'], alpha_hold=max(pad['alpha_min'], float(qp.get('alpha_hold') or pad['alpha_min'])),
              pad_alpha_min=pad['alpha_min'], pad_max_visual_omega_rad_s=pad['max_visual_omega_rad_s'],
              visual_rocking_jerk_scale=1.,
              pad_omega_step_rad_s=0.,
              pad_relative_spread_floor=float(pad['relative_spread_floor']),
              pad_visual_hold_s=float(pad['visual_hold_s']),
              pad_visual_hold_omega_rad_s=float(pad['visual_hold_omega_rad_s']),
              visual_hold_while_open=False, visual_hold_omega_rad_s=None,
              visual_hold_omega_s=None, escalation_omega_rad_s=None)
    config['motion'] = dict(normal_max_m_s=pad['normal_max_m_s'], seek_m_s=pad['normal_max_m_s'])
    return config
