"""Independent configuration for the handheld scan-style outer."""
from copy import deepcopy
import math
from pathlib import Path

from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.delayed_kf import KfConfig
from peirastic.contact_qp.runtime_config import calibrated_geometry, load_study_config


def default_pad_scan_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'config/contact_qp/pad_scan.yaml'


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
    if pad['hard_stop_n'] != 6. or pad.get('hard_stop_policy') != 'fresh_immediate_v1':
        raise ValueError('pad scan requires an immediate fresh 6 N stop')
    if not isinstance(config.get('feature_endpoint'), str) or not config['feature_endpoint']:
        raise ValueError('pad scan feature endpoint is required')
    # The pad block is authoritative; copies used by the shared solver are
    # resolved here so the startup snapshot describes the executed limits.
    qp = config['qp']
    for name in ('max_visual_omega_rad_s', 'cop_min_force_n', 'cop_max_m', 'max_image_age_s', 'recover_hold_s'):
        qp[name] = pad[name]
    qp.setdefault('differential_repair', {}).update(
        nominal_force_n=pad['force_gate_start_n'], soft_force_n=pad['force_gate_end_n'],
        hard_stop_n=pad['hard_stop_n'])
    qp.update(alpha_min=.25, visual_rocking_jerk_scale=1.,
              pad_omega_step_rad_s=0., pad_visual_hold_s=0.,
              visual_hold_while_open=False, visual_hold_omega_rad_s=None,
              visual_hold_omega_s=None, escalation_omega_rad_s=None)
    config['motion'] = dict(normal_max_m_s=pad['normal_max_m_s'], seek_m_s=pad['normal_max_m_s'])
    return config
