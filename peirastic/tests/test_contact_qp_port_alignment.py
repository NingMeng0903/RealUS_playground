from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from peirastic.contact_qp.energy import EnergyLedger,PortBounds,PortInterval,ExposureSegment
from peirastic.contact_qp.runtime_energy import RuntimeEnergy
from peirastic.contact_qp.port_alignment import MeasuredPortAligner

X=np.array([1.,0,0,0,0,0]);ZERO=np.zeros(6)
KIN=NS(jacobian=lambda q:np.array([[1,1,0,0,0,0,0,0],*[np.zeros(8) for _ in range(5)]]),
       fk_pose=lambda q:np.zeros(6))


def interval(a,b,v=X,w=ZERO,**kw):
    return PortInterval(a,b,w,w,v,v,calibration_version='test',time_aligned=True,**kw)


def runtime():
    return RuntimeEnergy(EnergyLedger(.03,1.,0.,PortBounds(calibration_version='test')),
                         command_budget_enforced=False,max_measurement_age_s=.1)


def test_zero_port_work_still_spends_same_tank_dissipation():
    rt=runtime();rt.bind_dissipation(dict(contact_map=[X],damping=[2.]))
    assert rt.observe_interval(interval(1.,1.01),now_s=1.01,physical_w_checked=True)
    assert rt.ledger.balance_j==pytest.approx(.01)
    fact=rt.ledger.events[-1]
    assert fact['port_output_work_upper_j']==0 and fact['dissipation_work_j']==pytest.approx(.02)
    assert fact['total_charge_j']==pytest.approx(.02)
    assert rt.observe_interval(interval(1.01,1.02),now_s=1.02,physical_w_checked=True)
    assert rt.ledger.balance_j==pytest.approx(-.01)  # no floor replenishment
    assert not rt.facts['certified']
    with pytest.raises(ValueError,match='differ'):
        rt.bind_dissipation(dict(contact_map=[X],damping=[1.]))


@pytest.mark.parametrize('bad',[-1,float('inf'),float('nan'),True])
def test_invalid_dissipation_rejected_before_mutation(bad):
    with pytest.raises(ValueError):interval(1,1.01,dissipation_work_j=bad)


def test_d_zero_exact_old_charge_and_positive_recovery():
    rt=runtime();rt.bind_dissipation({})
    assert rt.observe_interval(interval(1,1.01,w=X),now_s=1.01,physical_w_checked=True)
    assert rt.ledger.balance_j==pytest.approx(.04)
    assert rt.ledger.events[-1]['dissipation_work_j']==0
    old=rt.ledger.balance_j
    assert not rt.observe_interval(interval(1,1.01,w=X),now_s=1.02,physical_w_checked=True)
    assert rt.ledger.balance_j==old


def test_mean_velocity_dissipation_is_not_certified_upper_bound():
    rt=runtime();rt.bind_dissipation(dict(contact_map=[X],damping=[2.]))
    # An unresolved out-and-back could have positive D although net V is zero.
    assert rt.observe_interval(interval(1,1.01,v=ZERO),now_s=1.01,physical_w_checked=True)
    assert rt.events[-1]['dissipation_work_j']==0
    assert rt.events[-1]['estimated'] and not rt.events[-1]['certified']
    assert 'not_upper_bound' in rt.events[-1]['dissipation_estimator']
    # The conditional core can debit an independently validated full-interval bound.
    bounded=replace(interval(2,2.01,v=ZERO),dissipation_work_j=.02)
    assert rt.ledger.settle(bounded,now_s=2.01)
    assert rt.ledger.events[-1]['dissipation_work_j']==.02


def feed(a,t,rt,rq,seq,*,now=None,q=0.,source='udp',w=-X):
    return a.update(source_id=source,source_t_s=t,arm_q_rad=[q,0,0,0,0,0,0],wrench_tcp=w,
        source_valid=True,rail_feedback=NS(valid=True,sample_mono_s=rt,position_m=rq,motion_seq=seq),
        kin=KIN,now_s=t if now is None else now)


def aligner(**kw):return MeasuredPortAligner(calibration_version='test',max_wait_s=.1,**kw)


def test_async_measured_brackets_settle_once_without_extrapolation():
    a=aligner();rt=runtime()
    assert feed(a,1.,.995,0.,1)==[]
    assert feed(a,1.01,1.005,.01,2,q=.01)==[]  # no right support for 1.01 yet
    pieces=feed(a,1.01,1.015,.02,3,q=.01,now=1.016)
    assert len(pieces)==2  # rail knot splits the original W/arm interval
    assert [(p.start_s,p.end_s) for p,_ in pieces]==[(1.,1.005),(1.005,1.01)]
    for port,provenance in pieces:
        np.testing.assert_allclose(port.velocity_start,2*X,atol=1e-12)
        for end in ('rail_start','rail_end'):
            assert sum(provenance[end]['weights'])==pytest.approx(1)
            assert min(provenance[end]['weights'])>=0
        assert rt.observe_interval(port,now_s=1.016,physical_w_checked=True,provenance=provenance)
    assert rt.ledger.balance_j==pytest.approx(.01)
    assert rt.monitor_status=='monitor_estimated'
    assert feed(a,1.01,1.015,.02,3,q=.01,now=1.017)==[]


def test_missing_rail_support_never_refunds_exposure():
    a=aligner();rt=runtime();rt.ledger.reserve(1,[ExposureSegment(1,1.01,1.)])
    feed(a,1.,.995,0.,1)
    assert feed(a,1.01,1.005,.01,2,q=.01)==[]
    assert feed(a,1.12,1.115,.02,3,q=.02)==[]
    assert rt.ledger.reserved_j==pytest.approx(.01)
    assert any(e['reason']=='rail_bracket_timeout' for e in a.events)


def test_source_epoch_and_rail_gap_do_not_bridge():
    a=aligner(max_rail_interval_s=.01)
    feed(a,1.,.999,0.,1)
    assert feed(a,1.005,1.004,.01,2,source='new')==[]
    assert feed(a,1.01,1.009,.02,3,source='new')==[]  # no right bracket yet
    assert feed(a,1.01,1.04,.03,4,source='new',now=1.04)==[]  # gap cannot be interpolated


def test_active_measurement_ingress_reaches_same_ledger_without_commands():
    from peirastic.realman8dof.modes.contact_active import ContactQpOuter
    from peirastic.tests.test_contact_qp_runtime import MemorySink
    active=object.__new__(ContactQpOuter)
    active.energy=runtime();active.energy.bind_dissipation(dict(contact_map=[X],damping=[.5]))
    active._port_aligner=aligner();active._physical_w_checked=True;active.sink=MemorySink()
    def ingress(t,rail_t,rail_q,seq,armq,now):
        active.observe_measured_port(NS(t_s=t,q_deg=np.rad2deg([armq,0,0,0,0,0,0]),ok=True),
            NS(valid=True,sample_mono_s=rail_t,position_m=rail_q,motion_seq=seq),-X,KIN,
            now_s=now,source_id='udp')
    ingress(1.,.995,0.,1,0.,1.)
    ingress(1.01,1.005,.01,2,.01,1.01)
    assert active.energy.ledger.balance_j==.03
    ingress(1.01,1.015,.02,3,.01,1.016)
    # V=2, output=2W, D=.5*2^2=2W for .01s, one ledger pays .04J.
    assert active.energy.ledger.balance_j==pytest.approx(-.01)
    facts=[row for row in active.sink.records if row['event']=='energy_measurement']
    assert facts and active.energy.monitor_status=='monitor_estimated'
    assert all(row['event']!='publication' for row in active.sink.records)


def test_dissipation_total_overflow_does_not_release_reservation():
    ledger=EnergyLedger(1.,2.,0.,PortBounds(calibration_version='test'))
    ledger.reserve(1,[ExposureSegment(1,1.01,1)])
    huge=interval(1,1.01,v=X,w=-1e308*X,dissipation_work_j=1.79e308)
    assert not ledger.settle(huge,now_s=1.01)
    assert ledger.balance_j==1. and ledger.reserved_j>0


@pytest.mark.parametrize('bad_rail',[
    None,NS(valid=False,sample_mono_s=1.005,position_m=.005,motion_seq=2),
    NS(valid=True,sample_mono_s=float('nan'),position_m=.005,motion_seq=2),
    NS(valid=True,sample_mono_s=.8,position_m=.005,motion_seq=2),
])
def test_known_invalid_rail_sample_fences_interpolation_and_liabilities(bad_rail):
    a=aligner();rt=runtime();rt.ledger.reserve(1,[ExposureSegment(1.,1.01,1.)])
    feed(a,1.,1.,0.,1)
    assert a.update(source_id='udp',source_t_s=1.005,arm_q_rad=[0]*7,wrench_tcp=-X,
                    source_valid=True,rail_feedback=bad_rail,kin=KIN,now_s=1.005)==[]
    assert not a.rail and not a.pending
    assert feed(a,1.01,1.01,.01,3)==[]
    assert rt.ledger.reserved_j==pytest.approx(.01)
    # Recovery begins a fresh fully supported interval, never the pre-gap one.
    result=feed(a,1.015,1.015,.015,4)
    assert result and all(p.start_s>=1.01 for p,_ in result)


def test_normal_repeated_valid_rail_sample_does_not_fence_history():
    a=aligner();feed(a,1.,.995,0.,1)
    assert feed(a,1.005,.995,0.,1)==[]
    assert len(a.rail)==1 and len(a.pending)==1
    result=feed(a,1.01,1.01,.015,2)
    assert result and result[0][0].start_s==1.
