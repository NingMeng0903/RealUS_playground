"""Independent, validated gamepad contact configuration (no transport)."""
from copy import deepcopy
from pathlib import Path
import math

from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.runtime_config import load_study_config, calibrated_geometry

DEFAULT_PAD_CONTACT_CONFIG = Path(__file__).resolve().parents[2] / 'config/contact_qp/pad_contact.yaml'


def load_pad_contact_config(source=None):
    config = deepcopy(load_study_config(source or DEFAULT_PAD_CONTACT_CONFIG))
    geometry, _ = calibrated_geometry(config)
    feature = config['feature']
    fc = FeatureConfig(**feature['config'])
    if fc.window_version != feature['window_version'] or fc.region_layout_version != feature['region_layout_version']:
        raise ValueError('pad feature configuration does not match its declared versions')
    pad = config['pad']
    # Older independent pad files inherit the publication timing policy.
    pad.setdefault('minimum_publication_period_s', .005)
    pad.setdefault('dispatch_policy', 'full_command_lifetime_v1')
    # Old files inherit the corrected policy; obsolete lead/window values
    # must not silently reinstate the 2 ms installation failure.
    pad.pop('dispatch_lead_s', None)
    pad.pop('dispatch_window_s', None)
    if pad.get('policy') != 'pad_contact_v1':
        raise ValueError('unsupported pad contact policy')
    if pad['dispatch_policy'] != 'full_command_lifetime_v1':
        raise ValueError('unsupported pad dispatch policy')
    for name in ('normal_max_m_s', 'max_visual_omega_rad_s', 'angular_max_rad_s',
                 'angular_acceleration_rad_s2', 'cop_min_force_n', 'cop_max_m',
                 'repair_speed_cap_m_s', 'repair_pause_after_s', 'recover_hold_s',
                 'alpha_ramp_down_s', 'alpha_ramp_up_s', 'max_image_age_s',
                 'max_feedback_age_s', 'command_max_age_s', 'hard_stop_n',
                 'minimum_publication_period_s'):
        value = pad[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'pad.{name} must be finite and positive')
    pad.setdefault('hard_stop_persist_s', .15)
    pad.setdefault('hard_stop_latch_n', 8.)
    for name in ('hard_stop_persist_s', 'hard_stop_latch_n'):
        value = pad[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'pad.{name} must be finite and positive')
    if pad['max_visual_omega_rad_s'] > pad['angular_max_rad_s']:
        raise ValueError('visual request exceeds final angular limit')
    if pad['minimum_publication_period_s'] >= pad['command_max_age_s']:
        raise ValueError('pad minimum publication period exceeds command lifetime')
    if pad['cop_max_m'] > geometry.half_length_m or pad['hard_stop_n'] != 6.:
        raise ValueError('invalid pad COP aperture or supervisory force limit')
    if pad['hard_stop_latch_n'] <= pad['hard_stop_n']:
        raise ValueError('pad latch force must exceed the 6 N supervisor')
    if pad['hard_stop_persist_s'] > pad['command_max_age_s'] * 4:
        raise ValueError('pad force supervisor persist exceeds four command lifetimes')
    return config
