"""Exercise real active transactions without any robot or image receiver."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from peirastic.contact_qp.execution import ProposalDeferred
from peirastic.contact_qp.types import ContactStatus
from peirastic.realman8dof.modes.contact_active import ContactQpOuter
from peirastic.tests.test_contact_qp_runtime import make_outer, MemorySink
from peirastic.tests.test_contact_qp_runtime_config import continuous_config
from peirastic.tests.test_contact_qp_solver import observation


def fixture(monkeypatch):
    config=continuous_config()
    config.update(execution_policy='continuous_recovery_v1')
    config['source'].update(timebase='variable_step_bilinear_v1',max_age_s=.015,
        max_interval_s=.015,gap_policy='lease_fresh_foh_v1')
    config['qp']['solver_policy']='bounded_retry_v1'
    config['qp']['differential_repair']['quality_objective']='deficit_only_v1'
    config['feature']['dropout_policy']='pause_visual'
    config['energy'].update(initial_j=.1,capacity_j=.15,stopping_reserve_j=.05,
                            task_power_source='nominal_command')
    baseline,pose=make_outer();sink=MemorySink();clock=[10.]
    monkeypatch.setattr('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:clock[0])
    active=ContactQpOuter(baseline,config,sink=sink,
        feature_receiver=SimpleNamespace(observation=None,close=lambda **kw:None))
    active.set_origin(pose)
    active.features.observation=replace(observation((.9,.9,.95),now=clock[0]),
        registration_version=active.registration_version,window_version=active.feature_config.window_version,
        calibration_version=active.feature_config.calibration_version)
    return active,pose,clock,sink


def sample(active,pose,clock,*,source=None,dt=.005,force=4.):
    stamp=clock[0] if source is None else source
    wrench=np.array([0.,0.,force,0.,0.,0.])
    return active.sample(0.,pose,wrench,f_ext_raw=wrench,dt_actual=dt,
        wrench_source_id='A',wrench_source_time_s=stamp,wrench_source_wall_time_ns=int(stamp*1e6))


def commit(active,command,clock):
    active.rocking_constraints()
    # This detached fixture bypasses IK; explicitly declare mechanics-only.
    assert active.publication_review(active.pending_id,command,now_s=clock[0],
        facts=dict(rocking_policy_tier=4,rocking_limited=True))
    assert active.publication_started(active.pending_id)
    active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})


def test_actual_reference_interval_and_refusal_have_no_catch_up(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    commit(active,sample(active,pose,clock,dt=1/179),clock)
    assert active.reference_time_s==pytest.approx(1/179)
    before=active.reference_time_s
    clock[0]+=.006
    sample(active,pose,clock,dt=.006)
    clock[0]+=.012  # failed computation, no device send
    active.publication_abort('test numerical delay',definitely_not_sent=True)
    clock[0]+=.002
    command=sample(active,pose,clock,dt=.014)
    assert active.pending_reference_dt==pytest.approx(.002)
    commit(active,command,clock)
    assert active.reference_time_s-before==pytest.approx(.002)
    assert active.command_budget.balance_j>=.05


def test_numeric_deferral_consumes_measurement_once_and_keeps_original_lease(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    commit(active,sample(active,pose,clock),clock)
    old=active.command_budget.active;reference=active.reference_time_s
    previous=active._previous.copy();angle_history=active._rocking.omega_base.copy()
    solve=active.solver.solve
    def refuse(data,**kwargs):
        result=solve(data,**kwargs)
        return replace(result,status=ContactStatus.DEFERRED,qp_twist=None,
                       diagnostics={'reason':'solver_attempts_exhausted'})
    monkeypatch.setattr(active.solver,'solve',refuse)
    clock[0]+=.005
    count=active.controller._measurement_updates
    with pytest.raises(ProposalDeferred):sample(active,pose,clock)
    assert active.controller._measurement_updates==count+1
    assert not active._nominal_pending and active.command_budget.pending is None
    clock[0]+=.001
    assert active.retry_unsent_publication()
    assert active.command_budget.active is old and old.expires_s==pytest.approx(10.05)
    assert active.waiting_for_retry_source(10.005,now_s=clock[0])
    assert active.controller._measurement_updates==count+1
    assert active.reference_time_s==reference
    np.testing.assert_array_equal(active._previous,previous)
    np.testing.assert_array_equal(active._rocking.omega_base,angle_history)
    clock[0]=10.05
    with pytest.raises(RuntimeError,match='lease expired'):
        active.waiting_for_retry_source(10.005,now_s=clock[0])


def test_observer_to_qp_age_expiry_defers_after_nominal_measurement(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    commit(active,sample(active,pose,clock),clock)
    clock[0]=10.006
    active.prepare_source('A',10.005,10005000,now_s=clock[0])
    clock[0]=10.021
    count=active.controller._measurement_updates
    with pytest.raises(ProposalDeferred,match='wrench_source_expired'):
        sample(active,pose,clock,source=10.005)
    assert active.controller._measurement_updates==count+1
    clock[0]+=.0001
    assert active.retry_unsent_publication()
    assert active.command_budget.active.expires_s==pytest.approx(10.05)


def test_unleased_gap_starts_epoch_instead_of_stopping(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    active.take_source_epoch_reset()
    active.prepare_source('A',10.000,10000000,now_s=clock[0])
    clock[0]=10.030
    step=active.prepare_source('A',10.029,10029000,now_s=clock[0])
    assert not step.gap_recovered
    assert step.source_t_s==pytest.approx(10.029)
    assert active.take_source_epoch_reset()
    assert any(r['event']=='source_epoch_reset' for r in sink.records)


def test_original_dual_lease_admits_gap_and_raw_6n_still_stops(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    commit(active,sample(active,pose,clock),clock)
    clock[0]=10.030
    step=active.prepare_source('A',10.029,10029000,now_s=clock[0])
    assert step.gap_recovered and step.gap_context.lease_id==1
    count=active.controller._measurement_updates
    with pytest.raises(RuntimeError,match='6 N'):
        sample(active,pose,clock,source=10.029,force=6.)
    assert active.reference_time_s==pytest.approx(.005)
    assert active.controller._measurement_updates==count  # hard stop precedes proposal


def test_missing_native_rocking_facts_cannot_disable_new_policy(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    command=sample(active,pose,clock)
    assert not active.publication_review(active.pending_id,command,now_s=clock[0])
    assert active._publication_rejection_reason=='final_rocking_interval_violation'


def test_stale_ingress_after_success_has_no_stale_dispatch_marker(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    commit(active,sample(active,pose,clock),clock)
    old=active.command_budget.active
    assert active.pending_id is None and active._dispatch_time_s is None
    clock[0]=10.022
    with pytest.raises(ProposalDeferred,match='awaiting_fresh_force'):
        active.prepare_source('A',10.005,10005000,now_s=clock[0])
    active.publication_abort('awaiting_fresh_force',definitely_not_sent=True)
    clock[0]+=.0001
    assert active.retry_unsent_publication()
    assert active.command_budget.active is old and old.expires_s==pytest.approx(10.05)
    assert [r for r in sink.records if r['event']=='publication_aborted'][-1]['control_id'] is None


def test_both_healthy_disable_extra_visual_requests_and_preserve_nominal(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    sample(active,pose,clock)
    diagnostics=active.pending_result.diagnostics
    assert not diagnostics['visual_rows_active']
    assert diagnostics.get('differential_request_m_s',0.)==0
    assert diagnostics['alpha_target']==1.


def test_final_rocking_review_uses_existing_inner_tolerance(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    command=sample(active,pose,clock)
    assert active.publication_review(active.pending_id,command,now_s=clock[0],
        facts=dict(rocking_policy_tier=1,rocking_lower_rad_s=command[4]-.001,
                   rocking_upper_rad_s=command[4]-1e-7,rocking_tolerance_rad_s=1e-5))


def test_transport_time_audit_exposes_jerk_miss_without_relabeling_success():
    from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
    smooth=RockingSmoothing(1.,1.,.1,1.)
    smooth.seed(np.zeros(3),1.)
    smooth.acceleration_base[1]=.5
    # A command valid at a 5ms proposal interval can have a different measured
    # acceleration at a 10ms publication interval. Do not claim jerk tier1.
    final=np.zeros(6);final[4]=.5*.005
    audit=smooth.publication_audit(final,np.eye(3),1.01,.005,1,1e-8)
    assert audit['timing_limited'] and audit['actual_policy_tier']==2
    assert abs(audit['actual_jerk_rad_s3'])>smooth.jerk_limit
    assert smooth.time_s==1.


def test_failed_input_nested_immutable_diagnostics_remain_machine_readable():
    from peirastic.contact_qp.qp import QpResult
    from peirastic.contact_qp.types import TwistConstraints
    from peirastic.realman8dof.modes.contact_recording import json_value
    result=QpResult(None,0.,np.zeros(2),TwistConstraints(),ContactStatus.DEFERRED,
        {'qp_input':{'force_n':4.,'nominal_twist':np.zeros(6)},
         'numeric_attempts':[{'status':'max_iter','iterations':30}]})
    saved=json_value(dict(result.diagnostics))
    assert saved['qp_input']['force_n']==4.
    assert saved['qp_input']['nominal_twist']==[0.]*6
    assert saved['numeric_attempts'][0]['iterations']==30


def test_numeric_replay_wire_retains_signed_infinity_nan_and_empty_matrix():
    import json
    from peirastic.contact_qp.numeric_record import encode, decode
    from peirastic.realman8dof.modes.contact_recording import json_value
    payload={'H':np.eye(2), 'l':np.array([-np.inf,0.]), 'u':np.array([np.inf,1.]),
             'bad':float('nan'), 'empty':np.empty((0,6))}
    decoded=decode(json.loads(json.dumps(json_value(encode(payload)),allow_nan=False)))
    for key in ('H','l','u','empty'):np.testing.assert_array_equal(decoded[key],payload[key])
    assert np.isnan(decoded['bad'])
