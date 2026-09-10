"""Versioned visual allocation policy, distinct from mechanical guarantees."""
from dataclasses import replace
import numpy as np
import pytest
from peirastic.contact_qp.qp import ContactQp,QpConfig
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.contact_qp.types import ProbeGeometry
from peirastic.tests.test_contact_qp_solver import datum


def config(**kw):
    return QpConfig(allocation_policy='differential_repair_v8',max_acceleration=np.full(6,100.),**kw)


def solve(quality=(.4,.8,1.),nominal=None,**kw):
    if nominal is None:nominal=np.array([.005,0.,.004,0.,0.,0.])
    path=nominal.copy();path[[2,4]]=0.
    data=datum(quality,nominal_twist=nominal,path_twist=path,previous_twist=nominal,**kw)
    return ContactQp(config()).solve(data),data


def test_common_loading_cannot_pay_differential_task():
    result,data=solve()
    assert result.qp_twist is not None
    assert abs(result.qp_twist[4])>1e-3
    assert result.diagnostics['differential_request_m_s']>0
    assert result.diagnostics['differential_achieved_m_s']>0
    # Same geometric asymmetry, mirrored image: mirrored rocking sign.
    mirrored,_=solve((1.,.8,.4))
    assert mirrored.qp_twist[4]==pytest.approx(-result.qp_twist[4],abs=1e-7)
    # Physical image-axis reversal also reverses the commanded rocking.
    flipped=ContactQp(config()).solve(replace(data,geometry=ProbeGeometry.synthetic(image_x_sign=-1)))
    assert flipped.qp_twist[4]==pytest.approx(-result.qp_twist[4],abs=1e-7)


def test_sufficient_correct_nominal_rocking_not_requested_as_new_increment():
    first,data=solve();sign=np.sign(first.qp_twist[4])
    nominal=data.nominal_twist.copy();nominal[4]=sign*.08
    # No alpha/image-loss interference in the exact nominal-optimum check.
    cfg=config(enable_progress_loss=False,keep_speed_m_s=0.)
    result=ContactQp(cfg).solve(replace(data,nominal_twist=nominal,previous_twist=nominal))
    assert result.qp_twist is not None
    assert result.qp_twist[4]==pytest.approx(nominal[4],abs=1e-7)


def test_wrong_nominal_is_corrected_towards_total_requested_direction():
    good,data=solve();nominal=data.nominal_twist.copy();nominal[4]=-np.sign(good.qp_twist[4])*.01
    result=ContactQp(config()).solve(replace(data,nominal_twist=nominal,previous_twist=nominal))
    assert result.qp_twist is not None
    assert abs(result.qp_twist[4]-good.qp_twist[4])<abs(nominal[4]-good.qp_twist[4])


def test_force_gate_closes_entire_acoustic_task_without_hiding_loss():
    policy=DifferentialRepairConfig()
    assert [policy.force_gate(f) for f in (3.,4.,4.25,4.5,5.)]==[1.,1.,.5,0.,0.]
    nominal=np.array([.005,0.,-.003,0.,-.01,0.])
    result,data=solve(nominal=nominal,force_n=4.6)
    assert result.qp_twist is not None
    assert not result.diagnostics['visual_rows_active']
    assert result.diagnostics['acquisition_loss']>0
    assert result.diagnostics['alpha_preferred']<1.
    assert result.qp_twist[2]==pytest.approx(nominal[2],abs=1e-7)
    assert result.qp_twist[4]==pytest.approx(nominal[4],abs=1e-7)
    assert result.diagnostics['differential_request_m_s']==0.


def test_same_slack_differential_cost_remains_when_alpha_is_stopped():
    policy=DifferentialRepairConfig();basis=np.zeros((6,3));basis[2,0]=.002;basis[4,1]=.1;basis[0,2]=.005
    visual=np.zeros((2,6));visual[:,2]=1.;visual[:,4]=[-.0155,.0155]
    h,g,rows,upper,facts=policy.terms(scaled_basis=basis,visual_rows=visual,endpoint_rows=visual,
        nominal_twist=np.zeros(6),deficits=np.array([.5,0.]),repair_speed=.002,normal_scale=.002,
        force_n=4.,alpha_preferred=.6,progress_weight=1.)
    # Differential inequality contains no alpha/common-vn coefficient.
    assert rows[0,2]==0 and rows[0,0]==0
    assert h[5,5]>=policy.differential_weight


def test_policy_configuration_rejects_unknown_or_changed_force_levels():
    with pytest.raises(ValueError):QpConfig(allocation_policy='not_versioned')
    with pytest.raises(ValueError):QpConfig(differential_repair={'soft_force_n':5.})
    assert QpConfig().allocation_policy=='legacy_v7'


def test_full_port_single_budget_limits_consuming_tilt_not_all_motion():
    from peirastic.contact_qp.port_constraint import PortEnergyConstraint
    unconstrained,data=solve();omega=unconstrained.qp_twist[4]
    wrench=np.zeros(6);wrench[4]=-np.sign(omega)
    energy=PortEnergyConstraint(wrench_environment=wrench,available_j=0.,hold_s=data.dt_s)
    limited=ContactQp(config()).solve(replace(data,energy=energy))
    assert limited.qp_twist is not None
    assert energy.admissible(limited.qp_twist,tolerance_w=1e-8,velocity_tolerance=1e-8)
    assert np.sign(omega)*limited.qp_twist[4]<=1e-7
    assert limited.diagnostics['differential_shortfall_m_s']>0
    assert limited.qp_twist[2]>0  # zero-power normal motion is not forbidden
    funded=ContactQp(config()).solve(replace(data,energy=replace(energy,available_j=.001)))
    assert funded.qp_twist is not None and np.sign(omega)*funded.qp_twist[4]>1e-3


def test_fresh_raw_six_newtons_stops_before_nominal_candidate(monkeypatch):
    from peirastic.tests.test_contact_qp_active import make_active
    active,pose,clock,sink=make_active(monkeypatch)
    active.solver=ContactQp(replace(active.solver.config,allocation_policy='differential_repair_v8'))
    initial=active.controller.last_v_cmd.copy()
    try:
        with pytest.raises(RuntimeError,match='6 N'):
            active.sample(0.,pose,np.array([0.,0.,4.,0.,0.,0.]),dt_actual=.005,
                f_ext_raw=np.array([0.,0.,6.,0.,0.,0.]),wrench_source_id='test-source',
                wrench_source_time_s=clock[0],wrench_source_wall_time_ns=1)
        np.testing.assert_array_equal(active.controller.last_v_cmd,initial)
        assert not active._nominal_pending and active.reference_time_s==0.
        events=[r for r in sink.records if r['event']=='force_supervisor_stop']
        assert len(events)==1 and events[0]['raw_original_force_n']==6.
    finally:active.close()


def test_real_v8_profile_reports_disabled_energy_and_policy_without_hardware():
    from pathlib import Path
    from peirastic.contact_qp.runtime_config import validate_study_config
    path=Path(__file__).parents[1]/'config/contact_qp/active_probe50_v8.yaml'
    facts=validate_study_config(path)
    assert facts['allocation_policy']=='differential_repair_v8'
    assert facts['command_energy_budget_enabled'] is False
    assert facts['force_limit_assurance']=='measured_policy_not_prediction'
    assert facts['hardware_connected'] is False


def test_v8_capability_rejects_old_active_service_without_command():
    import uuid
    from peirastic.core.ipc import CommandClient,CommandHub
    from peirastic.core.capabilities import CapabilityAdvertisement,study_capabilities,SOURCE_TIMEBASE_CAPABILITY,DIFFERENTIAL_REPAIR_CAPABILITY
    hub=CommandHub(prefix='v8_test_'+uuid.uuid4().hex+'_')
    client=CommandClient(prefix=hub.ctl_name.removesuffix('peirastic_ctl_v2'))
    advertisement=CapabilityAdvertisement(hub,{'contact_qp.active_v1',SOURCE_TIMEBASE_CAPABILITY,'contact_qp.differential_repair_v8'})
    try:
        profile=dict(mode='active',source={'timebase':'variable_step_bilinear_v1'},qp={'allocation_policy':'differential_repair_v8'})
        required=study_capabilities(profile)
        assert required[-1]==DIFFERENTIAL_REPAIR_CAPABILITY
        with pytest.raises(RuntimeError):
            for capability in required:client.require_capability(capability)
        assert client.snapshot()['cmd_seq']==0
        assert DIFFERENTIAL_REPAIR_CAPABILITY not in study_capabilities(dict(mode='active'))
    finally:advertisement.close();client.close();hub.close()
