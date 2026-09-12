"""Actual outer adapter routes logical accounting without measured credit."""
from types import SimpleNamespace
import numpy as np
import pytest

from peirastic.tests.test_contact_qp_active import make_active, propose
from peirastic.tests.test_contact_qp_runtime import make_outer, MemorySink
from peirastic.realman8dof.modes.contact_active import ContactQpOuter


def enabled(monkeypatch):
    template,_,clock,_=make_active(monkeypatch)
    config=template.config.copy();template.close()
    config.update(energy_constraint_enabled=True,physical_w_checked=False,
        energy=dict(initial_j=.6,capacity_j=.8,stopping_reserve_j=.05,
            settlement_port='logical_final_model',wrench_convention='negative_control_raw_tcp_v1',
            max_command_interval_s=.05))
    outer,pose=make_outer();sink=MemorySink()
    active=ContactQpOuter(outer,config,sink=sink,
        feature_receiver=SimpleNamespace(observation=None,close=lambda **kw:None))
    active.set_origin(pose,t_s=0.)
    return active,pose,clock,sink


def test_real_enabled_adapter_commits_final_pair_and_drains_events(monkeypatch):
    active,pose,clock,sink=enabled(monkeypatch)
    try:
        command=propose(active,pose,clock)
        row=active.command_power_constraints()
        rotation=active.pending_rotation_base_tcp
        wrench=active.pending_result.energy_certificate.wrench_environment
        np.testing.assert_allclose(row['command_power_wrench_base'],
            np.r_[rotation @ wrench[:3],rotation @ wrench[3:]])
        bound=active.pending_result.energy_certificate
        assert row['command_power_min_w']==pytest.approx(-(bound.task_power_w+bound.beta*bound.available_j/bound.hold_s))
        assert active.energy is None  # no second spendable or monitoring tank
        assert active.pending_result.energy_certificate.hold_s==.05
        assert active.pending_dt==.005
        assert active.publication_review(active.pending_id,command,now_s=clock[0])
        active.publication_started(active.pending_id)
        clock[0]+=.001
        active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
        np.testing.assert_array_equal(active.command_budget.active.wrench,[0.,0.,-4.,0.,0.,0.])
        assert active.command_budget.active.expires_s==pytest.approx(10.05)
        assert active.command_budget.events==[]
        assert any(r['event']=='logical_command_energy' for r in sink.records)
    finally:active.close()


@pytest.mark.parametrize('failure',['changed','partial','late'])
def test_real_enabled_publication_failures_latch_without_committing_nominal(monkeypatch,failure):
    active,pose,clock,_=enabled(monkeypatch)
    try:
        initial=active.reference_time_s
        command=propose(active,pose,clock)
        assert active.publication_review(active.pending_id,command,now_s=clock[0])
        active.publication_started(active.pending_id)
        if failure=='partial':active.publication_abort('partial',facts={'arm':'sent','rail':'failed'})
        else:
            if failure=='changed':command=command.copy();command[4]+=.001
            if failure=='late':clock[0]+=.051  # held-pair transport deadline, not an outer task diagnostic
            with pytest.raises(ValueError):active.publication_commit(active.pending_id,command,
                now_s=clock[0],facts={'arm':'sent','rail':'sent'})
            active.publication_abort(failure)
        assert active.command_budget.latched_reason
        assert active.command_budget.pending is not None
        assert active.reference_time_s==initial
    finally:active.close()


def test_review_checks_current_source_age_and_stop_keeps_physical_tail_unknown(monkeypatch):
    active,pose,clock,_=enabled(monkeypatch)
    try:
        command=active.sample(0.,pose,np.array([0.,0.,4.,0.,0.,0.]),dt_actual=.005,
            f_ext_raw=np.array([0.,0.,4.,0.,0.,0.]),wrench_source_id='source',
            wrench_source_time_s=clock[0]-.019,wrench_source_wall_time_ns=1)
        clock[0]+=.002
        assert not active.publication_review(active.pending_id,command,now_s=clock[0])
        active.record_stop('review refused')
        assert active.command_budget.facts['actual_tail']=='unknown'
        assert active.command_budget.latched_reason
    finally:active.close()


def test_measured_reverse_work_is_logged_without_crediting_logical_tank(monkeypatch):
    from peirastic.contact_qp.energy import PortInterval
    active,pose,clock,sink=enabled(monkeypatch)
    try:
        captured={}
        interval=PortInterval(9.99,10.,np.ones(6),np.ones(6),np.ones(6),np.ones(6))
        def update(**kwargs):captured.update(kwargs);return [(interval,{'test':'aligned'})]
        active._port_aligner=SimpleNamespace(update=update,drain_events=lambda:[])
        before=active.command_budget.balance_j
        active.observe_measured_port(SimpleNamespace(t_s=10.,q_deg=np.zeros(7),ok=True),None,
            np.arange(6.),None,now_s=10.,source_id='new-measured-sample')
        np.testing.assert_array_equal(captured['wrench_tcp'],-np.arange(6.))
        assert active.command_budget.balance_j==before
        event=sink.records[-1]
        assert event['event']=='nonspendable_measured_port' and not event['budget_balance_changed']
        assert not event['physical_certified']
    finally:active.close()


def test_recorded_003_dispatch_crosses_task_diagnostic_but_has_fresh_force(monkeypatch):
    """Recorded source/sample/review/transport timing, without sending devices."""
    active,pose,clock,sink=enabled(monkeypatch)
    try:
        active.source_clock.max_age_s=.015
        source=7505.899251706
        clock[0]=source+.003339425
        command=active.sample(0.,pose,np.array([0.,0.,3.89967,0.,0.,0.]),dt_actual=.007526652,
            f_ext_raw=np.array([.66268,3.98062,3.88292,.10827,-.01458,.04264]),
            wrench_source_id='recorded-003',wrench_source_time_s=source,wrench_source_wall_time_ns=1)
        # Final transport in the failed run completed after created+10ms, but
        # before source+15ms. The same final-model energy budget still applies.
        clock[0]=7505.913572727
        assert clock[0]>active.pending_result.hard_constraints.valid_until_s
        assert active.publication_review(active.pending_id,command,now_s=clock[0])
        assert active.publication_started(active.pending_id)
        clock[0]=7505.913880019
        active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
        assert active.command_budget.latched_reason is None
        assert active.command_budget.active is not None
        assert active.reference_time_s>0
        assert sink.records[-1]['event']=='publication'
    finally:active.close()


def test_force_can_expire_between_review_and_dispatch_without_any_send(monkeypatch):
    active,pose,clock,sink=enabled(monkeypatch)
    try:
        command=propose(active,pose,clock)
        assert active.publication_review(active.pending_id,command,now_s=clock[0])
        clock[0]+=active.source_clock.max_age_s+.001
        assert not active.publication_started(active.pending_id)
        active.publication_abort('dispatch rejected',definitely_not_sent=True)
        assert active.command_budget.pending is None
        assert active.command_budget.active is None
        assert active.command_budget.latched_reason is None
        assert active.reference_time_s==0
        assert any(r.get('event')=='publication_dispatch_rejected' for r in sink.records)
    finally:active.close()


def test_fresh_dispatch_can_complete_after_source_deadline_within_reserved_epoch(monkeypatch):
    active,pose,clock,_=enabled(monkeypatch)
    try:
        command=propose(active,pose,clock)
        assert active.publication_review(active.pending_id,command,now_s=clock[0])
        assert active.publication_started(active.pending_id)
        clock[0]+=.025
        active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
        assert active.command_budget.active is not None
        assert active.command_budget.latched_reason is None
    finally:active.close()
