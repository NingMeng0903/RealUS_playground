"""Offline study configuration checks. Validation never opens a transport."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_study_config(source):
    if isinstance(source,(str,Path)):
        import yaml
        source=yaml.safe_load(Path(source).expanduser().read_text())
    if not isinstance(source,dict):raise ValueError('study configuration must be a mapping')
    return dict(source)


def calibrated_geometry(config):
    from peirastic.contact_qp.types import ProbeGeometry
    raw=dict(config.get('geometry') or {})
    convention=raw.pop('face_normal_convention',None)
    if convention not in ('into_contact','outward'):
        raise ValueError('geometry.face_normal_convention must be into_contact or outward')
    geometry=ProbeGeometry(**raw)
    if not geometry.verified or geometry.calibration_version in ('unverified','synthetic_v1'):
        raise ValueError('active requires an explicitly verified real geometry revision')
    normal=geometry.T_tcp_face[:3,2].copy()
    if convention=='outward':normal=-normal
    return geometry,normal


def compression_force(config,wrench_environment_tool):
    """Physical FACE normal-load diagnostic, not the original 4 N task scalar.

    The controller and QP force_n retain normal_sign * original f_ext_z. This
    face projection may differ for a tilted installation and never replaces it.
    """
    _,normal=calibrated_geometry(config)
    wrench=np.asarray(wrench_environment_tool,dtype=float)
    if wrench.shape!=(6,) or not np.isfinite(wrench).all():
        raise ValueError('finite compensated environment-on-tool wrench required')
    return float(-normal @ wrench[:3])


def _validate_effective_tasks(config, mode, feature):
    """Construct only parameter/ledger objects, with the runtime merge order.

    Active velocity/acceleration/angle limits come from the original controller
    and replace these user QP fields. They are deliberately not interpreted as
    extra configurable limits here. No solver, worker, IPC, or robot is started.
    """
    if mode == 'baseline':return
    from peirastic.contact_qp.qp import QpConfig
    settings=dict(config.get('qp') or {})
    if mode=='shadow':settings.setdefault('quality_policy_version','unverified_real_study_policy')
    required=('config','window_version','registration_version','c_min','quality_policy_version')
    if all(feature.get(key) is not None for key in required):
        from peirastic.contact_qp.features import FeatureConfig
        fc=FeatureConfig(**feature['config'])
        if fc.window_version!=feature['window_version']:
            raise ValueError('feature window version does not match declared worker configuration')
        settings.update(c_min=feature['c_min'],quality_policy_version=feature['quality_policy_version'],
                        lateral_windows=fc.lateral_windows)
    if mode=='active':
        for key in ('max_velocity','max_acceleration','angle_limit_rad'):
            settings.pop(key,None)  # overwritten by checked original nominal limits
    qp_config=QpConfig(**settings)
    if mode!='active':return
    enabled=config.get('energy_constraint_enabled',False)
    if type(enabled) is not bool:raise ValueError('energy_constraint_enabled must be boolean')
    if qp_config.differential_repair.permission_mode=='continuous':
        if qp_config.allocation_policy!='differential_repair_v8' or not qp_config.enable_visual:
            raise ValueError('continuous visual repair requires the enabled v8 visual task')
        if not enabled or feature.get('required') is not True:
            raise ValueError('continuous visual repair requires command energy admission and required image feedback')
    energy=config.get('energy') or {}
    if not isinstance(energy,dict):raise ValueError('energy must be a mapping')
    task_source=energy.get('task_power_source','none')
    if task_source not in ('none','nominal_command') or (task_source!='none' and not enabled):
        raise ValueError('nominal task power requires enabled logical command budget')
    grace=float(feature.get('dropout_grace_s',0.))
    if (grace>0 or feature.get('dropout_policy','stop')=='pause_visual') and (feature.get('required') is not True or not qp_config.enable_visual
            or qp_config.allocation_policy!='differential_repair_v8'
            or qp_config.differential_repair.permission_mode!='continuous'
            or not enabled or grace>qp_config.max_image_age_s):
        raise ValueError('dropout policy requires required continuous visual feedback, command budget, and grace <= max_image_age_s')
    if not energy:
        if enabled:raise ValueError('enabled command energy requires an explicit single-tank configuration')
        return
    allowed={'initial_j','capacity_j','stopping_reserve_j','measurement_bounds','constraint',
             'max_measurement_age_s','max_rail_interval_s','settlement_port','wrench_convention',
             'max_command_interval_s','task_power_source'}
    unknown=set(energy)-allowed
    if unknown:raise ValueError('unknown energy fields: '+', '.join(sorted(unknown)))
    if enabled:
        from peirastic.contact_qp.command_budget import CommandBudget
        CommandBudget(energy['initial_j'],energy['capacity_j'],energy['stopping_reserve_j'],
            settlement_port=energy.get('settlement_port'),wrench_convention=energy.get('wrench_convention'),
            max_command_interval_s=energy.get('max_command_interval_s'),constraint=energy.get('constraint'),
            task_power_source=task_source)
        from peirastic.contact_qp.energy import PortBounds
        from peirastic.contact_qp.port_alignment import MeasuredPortAligner
        bounds=PortBounds(**dict(energy.get('measurement_bounds') or {}),verified=False)
        MeasuredPortAligner(calibration_version=bounds.calibration_version,
            max_source_interval_s=bounds.max_sample_interval_s,
            max_rail_interval_s=energy.get('max_rail_interval_s',bounds.max_sample_interval_s),
            max_wait_s=energy.get('max_measurement_age_s',config['source']['max_age_s']))
        return
    from peirastic.contact_qp.energy import EnergyLedger,PortBounds
    from peirastic.contact_qp.port_constraint import PortEnergyConstraint
    from peirastic.contact_qp.runtime_energy import RuntimeEnergy
    from peirastic.contact_qp.port_alignment import MeasuredPortAligner
    bounds=PortBounds(**dict(energy.get('measurement_bounds') or {}),verified=False)
    ledger=EnergyLedger(energy['initial_j'],energy['capacity_j'],energy['stopping_reserve_j'],bounds)
    runtime=RuntimeEnergy(ledger,command_budget_enforced=enabled,
        max_measurement_age_s=energy.get('max_measurement_age_s',config['source']['max_age_s']))
    parameters=dict(energy.get('constraint') or {})
    runtime.bind_dissipation(parameters)
    # Validate static constraint shapes/numbers even when only monitoring now;
    # these are the same parameters used if command admission is enabled.
    if parameters.get('assurance','command_model') not in ('command_model','monitor'):
        raise ValueError('runtime command budgets cannot certify declared physical bounds')
    PortEnergyConstraint(wrench_environment=np.zeros(6),available_j=ledger.available_j,
                         hold_s=.005,**parameters)
    MeasuredPortAligner(calibration_version=bounds.calibration_version,
        max_source_interval_s=bounds.max_sample_interval_s,
        max_rail_interval_s=energy.get('max_rail_interval_s',bounds.max_sample_interval_s),
        max_wait_s=runtime.max_measurement_age_s)


def validate_study_config(source):
    config=load_study_config(source)
    mode=config.get('mode','baseline')
    if mode not in ('baseline','shadow','active'):raise ValueError('invalid study mode')
    missing=[]
    geometry=config.get('geometry') or {}
    for key in ('half_length_m','T_tcp_face','image_x_sign','calibration_version','face_normal_convention'):
        if geometry.get(key) is None:missing.append('geometry.'+key)
    if geometry.get('verified') is not True:missing.append('geometry.verified=true')
    feature=config.get('feature') or {}
    if type(feature.get('required',False)) is not bool:
        raise ValueError('feature.required must be boolean')
    dropout_policy=feature.get('dropout_policy','stop')
    if dropout_policy not in ('stop','pause_visual') or (dropout_policy!='stop' and mode!='active'):
        raise ValueError('feature.dropout_policy must be stop or active pause_visual')
    grace=feature.get('dropout_grace_s',0.)
    if isinstance(grace,(bool,np.bool_)):
        raise ValueError('feature.dropout_grace_s must be numeric')
    grace=float(grace)
    if not np.isfinite(grace) or grace<0 or (grace>0 and mode!='active'):
        raise ValueError('feature.dropout_grace_s must be finite, nonnegative, and active-only')
    for key in ('config','window_version','registration_version','quality_policy_version','c_min'):
        if feature.get(key) is None:missing.append('feature.'+key)
    if feature.get('verified') is not True:missing.append('feature.verified=true')
    if not config.get('feature_endpoint'):missing.append('feature_endpoint')
    feature_values=feature.get('config') or {}
    if isinstance(feature_values,dict) and feature_values.get('algorithm_version')=='randomwalk_welleweerd2020_v3' and feature.get('c_min') is not None:
        from peirastic.contact_qp.features import FeatureConfig
        if feature['c_min'] != FeatureConfig(**feature_values).low_confidence_threshold:
            raise ValueError('v3 feature.c_min must equal FeatureConfig.low_confidence_threshold')
    if not missing:
        from peirastic.contact_qp.features import FeatureConfig
        g,_=calibrated_geometry(config)
        fc=FeatureConfig(**feature['config'])
        if fc.window_version!=feature['window_version']:
            raise ValueError('feature.window_version does not match the worker configuration')
        if feature.get('registration') is not None:
            from peirastic.contact_qp.features import registration_revision
            registration=feature['registration']
            if not isinstance(registration,dict):raise ValueError('feature.registration must be a mapping')
            if registration_revision(fc,**registration)!=feature['registration_version']:
                raise ValueError('feature.registration_version does not match source/crop/flip registration')
        if fc.calibration_version!=g.calibration_version or fc.image_x_sign!=g.image_x_sign:
            raise ValueError('feature/probe calibration or image axis mismatch')
        threshold=feature['c_min']
        if isinstance(threshold,bool) or not np.isfinite(threshold) or not 0<threshold<=1:
            raise ValueError('feature.c_min must be a declared threshold in (0,1]')

    if mode == 'active':
        source = config.get('source') or {}
        for key in ('period_s', 'jitter_fraction', 'max_age_s'):
            if source.get(key) is None: missing.append('source.' + key)
        if config.get('force_axis_monotonicity_confirmed') is not True:
            missing.append('force_axis_monotonicity_confirmed=true')
        if not any(key.startswith('source.') for key in missing):
            from peirastic.contact_qp.runtime_source import SourceClock
            SourceClock(**source)
    if mode=='active' and missing:
        raise ValueError('active calibration missing: '+', '.join(missing))
    _validate_effective_tasks(config,mode,feature)
    return dict(mode=mode,configuration_valid=True,
                command_authority='unchanged_baseline' if mode!='active' else 'outer_qp_original_ik',
                calibration_status='unverified' if missing else 'declared',
                missing_active_calibration=missing,
                physical_port_assurance='unverified',hardware_connected=False,
                command_energy_budget_enabled=bool(config.get('energy_constraint_enabled',False)),
                differential_repair_revision=((config.get('qp') or {}).get('differential_repair') or {}).get('revision','v8r2_bounded_episode'),
                repair_permission_mode=((config.get('qp') or {}).get('differential_repair') or {}).get('permission_mode','bounded_episode'),
                settlement_port=(config.get('energy') or {}).get('settlement_port'),
                task_power_source=(config.get('energy') or {}).get('task_power_source','none'),
                energy_assurance='two_port_command_model' if (config.get('energy') or {}).get('task_power_source')=='nominal_command' else 'command_model',
                image_dropout_grace_s=grace,
                image_dropout_policy=dropout_policy,
                physical_w_checked=config.get('physical_w_checked') is True,
                allocation_policy=(config.get('qp') or {}).get('allocation_policy','legacy_v7'),
                force_limit_assurance='measured_policy_not_prediction' if (config.get('qp') or {}).get('allocation_policy')=='differential_repair_v8' else 'legacy')
