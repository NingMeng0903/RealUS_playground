"""Logical held-pair accounting: no measured or predicted recovery credit."""
from dataclasses import replace
import numpy as np
import pytest

from peirastic.contact_qp.command_budget import CommandBudget
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.tests.test_contact_qp_solver import datum


def budget(**kwargs):
    config=dict(initial_j=.6,capacity_j=.8,stopping_reserve_j=.05,max_command_interval_s=.05,
        settlement_port='logical_final_model',wrench_convention='negative_control_raw_tcp_v1')
    config.update(kwargs)
    return CommandBudget(**config)


def candidate(tank,seq,t,*,power=-1.):
    v=np.ones(6)*.1
    raw=np.ones(6)*(-power/.6)
    snap=tank.snapshot(now_s=t,wrench_control_raw=raw,rotation_base_tcp=np.eye(3))
    assert tank.reserve(seq,snap,v,now_s=t)
    return v


def commit(tank,seq,v,t):
    tank.publication_started(seq)
    tank.commit(seq,v,now_s=t,rotation_base_tcp=np.eye(3),dual_success=True)


def test_frozen_pair_partition_and_review_overlap_do_not_rewrite_history():
    one=budget();v=candidate(one,1,0.);commit(one,1,v,0.)
    split=budget();v=candidate(split,1,0.);commit(split,1,v,0.)
    one.advance(.03)
    for t in (.01,.02,.03):split.advance(t)
    assert one.balance_j==pytest.approx(.57)
    assert split.balance_j==pytest.approx(one.balance_j)
    # New positive-power wrench has not been published and earns no credit.
    new=candidate(split,2,.035,power=2.)
    assert split.balance_j==pytest.approx(.565)
    commit(split,2,new,.04)
    assert split.balance_j==pytest.approx(.56)
    assert split.active.expires_s==pytest.approx(.085)  # review expiry, not commit+.05
    split.advance(.045)
    assert split.balance_j==pytest.approx(.57)


def test_old_and_new_liability_overlap_and_known_no_send_keeps_old_hold():
    tank=budget();v=candidate(tank,1,0.);commit(tank,1,v,0.)
    candidate(tank,2,.01)
    assert tank.reserved_j==pytest.approx(.09)
    assert tank.reject_new_only(2,definitely_not_sent=True,now_s=.02)
    assert tank.balance_j==pytest.approx(.58)
    assert tank.reserved_j==pytest.approx(.03)
    tank.advance(.03)
    assert tank.balance_j==pytest.approx(.57)


@pytest.mark.parametrize('started,known',[(False,False),(True,False),(True,True)])
def test_partial_unknown_fences_commands_and_retains_liability(started,known):
    tank=budget();candidate(tank,1,0.)
    if started:tank.publication_started(1)
    assert not tank.reject_new_only(1,definitely_not_sent=known,now_s=.01)
    assert tank.reserved_j==pytest.approx(.05)
    assert tank.balance_j==.6
    with pytest.raises(ValueError):candidate(tank,2,.02)
    tank.stop(now_s=.03)
    assert tank.reserved_j==pytest.approx(.05)


@pytest.mark.parametrize('change',['payload','rotation','dual','expiry'])
def test_final_pair_commit_is_immutable_and_cannot_revive_expiry(change):
    tank=budget();v=candidate(tank,1,0.);tank.publication_started(1)
    rotation=np.eye(3)
    if change=='payload':v=v.copy();v[5]+=.001
    if change=='rotation':rotation=np.diag([-1.,-1.,1.])
    with pytest.raises(ValueError):tank.commit(1,v,now_s=.051 if change=='expiry' else .01,
        rotation_base_tcp=rotation,dual_success=change!='dual')
    assert tank.latched_reason
    assert tank.reserved_j==pytest.approx(.05)


def test_expiry_stops_only_logical_output_and_never_lower_clips_balance():
    tank=budget();v=candidate(tank,1,0.);commit(tank,1,v,0.)
    assert not tank.advance(.06)
    assert tank.balance_j==pytest.approx(.55)
    assert tank.active is None and tank.facts['actual_tail']=='unknown'
    with pytest.raises(ValueError):candidate(tank,2,.07)
    recovering=budget();v=candidate(recovering,1,0.,power=10.);commit(recovering,1,v,0.)
    recovering.advance(.04)
    assert recovering.balance_j==.8


def test_duplicate_reverse_and_overflow_do_not_refund():
    tank=budget();v=candidate(tank,1,0.);commit(tank,1,v,0.);tank.advance(.01)
    for t in (.01,0.):
        with pytest.raises(ValueError):tank.advance(t)
        assert tank.balance_j==pytest.approx(.59)
    fresh=budget()
    snap=fresh.snapshot(now_s=0.,wrench_control_raw=np.full(6,1e308),rotation_base_tcp=np.eye(3))
    assert not fresh.reserve(1,snap,np.full(6,-1e308),now_s=0.)
    assert fresh.balance_j==.6 and fresh.reserved_j==0.


def test_independent_hold_can_cover_but_not_shorten_qp_step():
    energy=PortEnergyConstraint(np.zeros(6),.1,.05)
    assert datum(energy=energy).dt_s==.005
    with pytest.raises(ValueError,match='cover'):datum(energy=replace(energy,hold_s=.001))


@pytest.mark.parametrize('field',['damping','tracking_error','wrench_error','wrench_rate'])
def test_first_version_refuses_unmodeled_error_or_damping(field):
    parameters={field:np.ones(6)}
    if field=='damping':parameters.update(contact_map=np.eye(6),contact_speed_bound=np.ones(6))
    with pytest.raises(ValueError):budget(constraint=parameters)


@pytest.mark.parametrize('field',['initial_j','capacity_j','stopping_reserve_j','max_command_interval_s'])
def test_boolean_budget_fields_are_not_numeric(field):
    with pytest.raises(ValueError):budget(**{field:True})


def test_invalid_time_latches_without_refunding_pending():
    tank=budget();candidate(tank,1,1.)
    with pytest.raises(ValueError):tank.reject_new_only(1,definitely_not_sent=True,now_s=float('nan'))
    assert tank.pending is not None and tank.reserved_j==pytest.approx(.05)
    with pytest.raises(ValueError):candidate(tank,2,2.)


def test_new_profile_validation_and_old_daemon_cannot_accept_it():
    import uuid
    from pathlib import Path
    from peirastic.contact_qp.runtime_config import validate_study_config,load_study_config
    from peirastic.core.capabilities import (CapabilityAdvertisement,study_capabilities,
        SOURCE_TIMEBASE_CAPABILITY,DIFFERENTIAL_REPAIR_CAPABILITY,
        CONFIDENCE_BALANCE_CAPABILITY,LOGICAL_COMMAND_BUDGET_CAPABILITY)
    from peirastic.core.ipc import CommandHub,CommandClient
    path=Path(__file__).parents[1]/'config/contact_qp/active_probe50_v8r3_tank.yaml'
    config=load_study_config(path);facts=validate_study_config(config)
    assert facts['command_energy_budget_enabled'] and not facts['physical_w_checked']
    assert facts['settlement_port']=='logical_final_model'
    assert facts['differential_repair_revision']=='v8r3_confidence_balance'
    requirements=study_capabilities(config)
    assert CONFIDENCE_BALANCE_CAPABILITY in requirements
    assert LOGICAL_COMMAND_BUDGET_CAPABILITY in requirements
    hub=CommandHub(prefix='r3_tank_'+uuid.uuid4().hex+'_')
    client=CommandClient(prefix=hub.ctl_name.removesuffix('peirastic_ctl_v2'))
    advert=CapabilityAdvertisement(hub,{'contact_qp.active_v1',SOURCE_TIMEBASE_CAPABILITY,DIFFERENTIAL_REPAIR_CAPABILITY})
    try:
        for capability in (CONFIDENCE_BALANCE_CAPABILITY,LOGICAL_COMMAND_BUDGET_CAPABILITY):
            with pytest.raises(RuntimeError):client.require_capability(capability)
        assert client.snapshot()['cmd_seq']==0
    finally:advert.close();client.close();hub.close()
