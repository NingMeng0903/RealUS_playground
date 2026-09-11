"""Static bad configurations must fail before a scan can construct hardware."""
from dataclasses import asdict
import numpy as np
import pytest
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.runtime_config import validate_study_config


def active_config():
    fc=FeatureConfig(calibration_version='test-fixture')
    return dict(mode='active',feature_endpoint='inproc://not-opened',
        geometry=dict(half_length_m=.025,T_tcp_face=np.eye(4).tolist(),image_x_sign=1,
            calibration_version=fc.calibration_version,verified=True,face_normal_convention='into_contact'),
        feature=dict(config=asdict(fc),window_version=fc.window_version,registration_version='test',
            c_min=.5,quality_policy_version='test',verified=True),
        source=dict(period_s=.005,jitter_fraction=.1,max_age_s=.02),
        force_axis_monotonicity_confirmed=True,physical_w_checked=True,
        energy_constraint_enabled=True,
        energy=dict(initial_j=.2,capacity_j=1.,stopping_reserve_j=.01,
            settlement_port='logical_final_model',wrench_convention='negative_control_raw_tcp_v1',
            max_command_interval_s=.05))


@pytest.mark.parametrize('qp',[{'force_target_n':5},{'unknown':1},{'slack_weight':float('nan')},{'max_velocity':[1,2]}])
def test_shadow_rejects_effective_bad_qp(qp):
    with pytest.raises((ValueError,TypeError)):
        validate_study_config(dict(mode='shadow',qp=qp))


def test_feature_overrides_and_active_ignored_limits_match_runtime():
    config=active_config();config['qp']=dict(c_min=-1,quality_policy_version='',max_velocity=[0],
                                          max_acceleration=[0],angle_limit_rad=-1)
    assert validate_study_config(config)['configuration_valid']
    config['mode']='shadow'
    with pytest.raises(ValueError):validate_study_config(config)


@pytest.mark.parametrize('energy',[
    dict(initial_j=-1,capacity_j=1,stopping_reserve_j=0),
    dict(initial_j=.2,capacity_j=1,stopping_reserve_j=0,unknown=1),
    dict(initial_j=.2,capacity_j=1,stopping_reserve_j=0,constraint={'tracking_error':[1]}),
    dict(initial_j=.2,capacity_j=1,stopping_reserve_j=0,constraint={'beta':0}),
    dict(initial_j=.2,capacity_j=1,stopping_reserve_j=0,measurement_bounds={'unknown':1}),
])
def test_active_bad_energy_rejected_without_transport(energy):
    config=active_config();config['energy'].update(energy)
    with pytest.raises((ValueError,TypeError)):validate_study_config(config)


def test_baseline_does_not_interpret_unused_qp_or_energy():
    assert validate_study_config(dict(mode='baseline',qp={'unknown':1},energy={'unknown':1}))['configuration_valid']


@pytest.mark.parametrize('field',['settlement_port','wrench_convention','max_command_interval_s'])
def test_enabled_energy_requires_each_explicit_logical_contract_field(field):
    config=active_config();del config['energy'][field]
    with pytest.raises((ValueError,TypeError)):validate_study_config(config)
