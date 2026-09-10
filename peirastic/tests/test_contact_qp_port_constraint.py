from dataclasses import replace
import numpy as np
import pytest
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.qp import ContactQp,QpConfig,QpInput
from peirastic.contact_qp.types import ProbeGeometry,ContactObservation,ContactStatus,TwistConstraints

CFG=QpConfig(enable_aperture=False,enable_force_priority=False,max_acceleration=np.full(6,1000.))
PATH=np.array([0.,.02,0.,0.,0.,0.])


def data(energy=None,nominal=None,path=None,quality=.8,**kwargs):
    b=PATH if path is None else np.array(path)
    return QpInput(ProbeGeometry.synthetic(),b if nominal is None else nominal,b,4.,.005,1.,
        ContactObservation(0,'camera',1.,1.,[quality]*3,[True]*3,'reg','window'),
        energy=energy,**kwargs)


def budget(wrench,available=0.,**kwargs):
    return PortEnergyConstraint(wrench,available,.005,**kwargs)


def test_full_six_dimensional_net_power_cancellation_is_not_charged_per_axis():
    b=np.array([0.,.02,0.,.1,0.,0.]);w=np.array([0.,-5.,0.,1.,0.,0.])
    energy=budget(w)
    result=ContactQp(CFG).solve(data(energy,path=b))
    assert result.success and result.alpha==1.
    np.testing.assert_array_equal(result.qp_twist,b)
    assert energy.lower_power_w(b)==pytest.approx(0.,abs=1e-14)


@pytest.mark.parametrize('path,wrench', [(PATH,[0,-5,0,0,0,0]),([0,0,0,.1,0,0],[0,0,0,-1,0,0])])
def test_scan_or_planned_attitude_energy_jointly_reduces_alpha(path,wrench):
    energy=budget(wrench,.00025)
    result=ContactQp(CFG).solve(data(energy,path=path))
    assert result.success
    assert result.alpha==pytest.approx(.5,abs=1e-7)
    assert result.final_velocity_admissible(result.qp_twist)
    assert not result.final_velocity_admissible(np.array(path))
    assert result.energy_certificate is energy
    assert result.diagnostics['energy_margin_work_j']==pytest.approx(0.,abs=1e-9)


def test_environment_compression_sign_and_low_budget_find_a_smaller_feasible_action():
    nominal=PATH.copy();nominal[2]=.001
    energy=budget([0,0,-4,0,0,0],.00001)
    result=ContactQp(CFG).solve(data(energy,nominal=nominal))
    assert result.success and result.qp_twist[2]==pytest.approx(.0005,abs=1e-8)
    assert result.alpha==pytest.approx(1.)
    assert energy.lower_power_w([0,0,.001,0,0,0])<0
    assert energy.lower_power_w([0,0,-.001,0,0,0])>0


def test_sufficient_budget_preserves_same_visual_problem_optimum():
    solver=ContactQp(CFG);base=data(quality=.2)
    without=solver.solve(base)
    energy=budget([0,-1,-4,0,.2,0],10.,wrench_error=np.ones(6)*.001)
    with_energy=solver.solve(replace(base,energy=energy))
    assert without.success and with_energy.success
    np.testing.assert_allclose(with_energy.qp_twist,without.qp_twist,atol=1e-8,rtol=0)


def test_nonzero_tracking_error_cost_and_actual_damping_bound_are_enforced():
    pc=np.eye(6)[[2]];eta=np.zeros(6);eta[2]=.0001
    energy=budget([0,0,-4,0,0,0],.00001,tracking_error=eta,contact_map=pc,
                  damping=[1.],contact_speed_bound=[.002],assurance='declared_bound',bounds_version='test-v1')
    nominal=PATH.copy();nominal[2]=.001
    result=ContactQp(CFG).solve(data(energy,nominal=nominal))
    assert result.success
    assert result.qp_twist[2] < .0004
    a,lo,hi=energy.velocity_rows()
    np.testing.assert_allclose(hi,[.0019])
    assert any('energy_contact_speed' in label for label in result.hard_constraints.labels)
    assert energy.tracking_cost_w==pytest.approx(.0004002)


def test_zero_damping_does_not_add_contact_speed_limit():
    energy=budget(np.zeros(6),contact_map=np.eye(6)[[1]],damping=[0.],contact_speed_bound=[0.])
    result=ContactQp(CFG).solve(data(energy))
    assert result.success and result.alpha==1.
    assert len(energy.velocity_rows()[0])==0


def test_unachievable_tracking_bound_or_budget_is_task_failure_not_fake_zero():
    eta=np.ones(6)*.01
    bound=budget(np.zeros(6),tracking_error=eta,contact_map=np.eye(6)[[2]],damping=[1.],contact_speed_bound=[.001])
    result=ContactQp(CFG).solve(data(bound))
    assert result.status==ContactStatus.TASK_INFEASIBLE and result.qp_twist is None
    # Even zero command spends unavoidable tracking uncertainty. Prevent
    # unlimited retracting energy recovery by keeping all velocities zero.
    energy=budget([0,0,-4,0,0,0],tracking_error=eta)
    mechanics=TwistConstraints(np.eye(6),np.zeros(6),np.zeros(6),valid_until_s=1.01)
    result=ContactQp(CFG).solve(data(energy,mechanical=mechanics))
    assert result.status==ContactStatus.TASK_INFEASIBLE
    assert result.diagnostics['reason']=='energy_budget_infeasible'


def test_prefix_budget_recomputes_true_absolute_values_not_solver_epigraphs():
    energy=budget([0,-2,-4,0,.5,0],.0002,beta=.8,wrench_error=np.ones(6)*.01,
                  wrench_rate=np.ones(6)*.1,contact_map=np.eye(6)[[2]],damping=[2.],contact_speed_bound=[.01])
    nominal=PATH.copy();nominal[2]=.001
    result=ContactQp(CFG).solve(data(energy,nominal=nominal))
    assert result.success and result.final_velocity_admissible(result.qp_twist)
    for tau in np.linspace(0,.005,101):
        assert energy.available_j+energy.lower_work_j(result.qp_twist,tau)>=-1e-10
    assert energy.margin_work_j(result.qp_twist)==pytest.approx(energy.lower_work_j(result.qp_twist,.005)+.8*.0002)
    assert result.diagnostics['energy_margin_power_w']==pytest.approx(energy.margin_power_w(result.qp_twist))


@pytest.mark.parametrize('kwargs',[dict(available_j=-1),dict(hold_s=0),dict(beta=1.1),
    dict(wrench_error=[-1]*6),dict(wrench_rate=[np.inf]*6),dict(tracking_error=[np.nan]*6),
    dict(available_j=1e308,hold_s=1e-308),dict(frame='tcp_base'),dict(assurance='declared_bound')])
def test_invalid_or_overflowing_energy_bounds_cannot_create_credit(kwargs):
    values=dict(wrench_environment=np.zeros(6),available_j=1.,hold_s=.005);values.update(kwargs)
    with pytest.raises(ValueError):PortEnergyConstraint(**values)


@pytest.mark.parametrize('diagnostics',[False,True])
def test_path_subspace_conflict_is_not_full_mechanical_infeasibility(diagnostics):
    b=np.array([0.,-.0001,0,0,0,0]);previous=np.array([0.,.01,0,0,0,0])
    result=ContactQp(QpConfig(enable_visual=False)).solve(data(path=b,previous_twist=previous),interval_diagnostics=diagnostics)
    assert result.status==ContactStatus.TASK_INFEASIBLE
    assert result.diagnostics['reason']=='motion_subspace_conflict'


def test_pure_budget_snapshot_is_immutable_and_solver_does_not_spend_it():
    energy=budget([0,-5,0,0,0,0],.00025)
    solver=ContactQp(CFG)
    for _ in range(3):
        result=solver.solve(data(energy));assert result.success
        assert energy.available_j==.00025
    with pytest.raises(ValueError):energy.wrench_environment[0]=2.
    assert not result.final_velocity_admissible(result.qp_twist,now_s=2.)


def test_zero_tracking_error_reproduces_command_model_and_monitor_never_claims_verified():
    energy=budget([0,-5,0,0,0,0],.00025)
    declared=replace(energy,assurance='declared_bound',bounds_version='ideal-test',tracking_error=np.zeros(6))
    first=ContactQp(CFG).solve(data(energy));second=ContactQp(CFG).solve(data(declared))
    np.testing.assert_array_equal(first.qp_twist,second.qp_twist)
    monitor=ContactQp(CFG).solve(data(replace(energy,assurance='monitor')))
    assert monitor.diagnostics['energy_assurance']=='monitor'
    assert monitor.diagnostics['port_verified'] is False


def test_budget_hold_must_be_the_execution_hold_and_other_bounds_are_strict():
    with pytest.raises(ValueError,match='hold'):
        data(replace(budget(np.zeros(6)),hold_s=.004))
    with pytest.raises(ValueError):budget(np.zeros(6),contact_map=np.ones((1,5)))
    with pytest.raises(ValueError):budget(np.zeros(6),contact_map=np.ones((1,6)),damping=[-1],contact_speed_bound=[1])


def test_energy_does_not_hide_a_true_six_dimensional_mechanical_contradiction():
    mechanics=TwistConstraints(np.eye(6)[[2,2]],[.001,-np.inf],[np.inf,-.001],valid_until_s=1.01)
    result=ContactQp(CFG).solve(data(budget(np.zeros(6)),mechanical=mechanics))
    assert result.status==ContactStatus.MECHANICAL_INFEASIBLE


def test_feasibility_diagnostic_failure_is_solver_failure_not_a_physical_claim(monkeypatch):
    from peirastic.contact_qp import qp
    def failed(*args):raise RuntimeError('test_lp_numeric_failure')
    monkeypatch.setattr(qp,'_linear_feasible',failed)
    b=np.array([0.,-.0001,0,0,0,0]);previous=np.array([0.,.01,0,0,0,0])
    result=ContactQp(QpConfig(enable_visual=False)).solve(data(path=b,previous_twist=previous))
    assert result.status==ContactStatus.SOLVER_FAILED
    assert result.diagnostics['reason']=='test_lp_numeric_failure'


def test_actual_bounded_tracking_and_wrench_variation_obey_every_prefix():
    eta=np.array([0.,.0001,.00002,0.,0.,0.])
    energy=budget([0,-2,-4,0,0,0],.00015,beta=.9,
                  wrench_error=np.ones(6)*.01,wrench_rate=np.ones(6)*.3,
                  tracking_error=eta,contact_map=np.eye(6)[[2]],damping=[5.],
                  contact_speed_bound=[.002],assurance='declared_bound',bounds_version='prefix-test')
    nominal=PATH.copy();nominal[2]=.001
    result=ContactQp(CFG).solve(data(energy,nominal=nominal))
    assert result.success
    v=result.qp_twist;actual_work=0.;step=energy.hold_s/1000
    for i in range(1000):
        midpoint=(i+.5)*step
        error=eta*np.cos(midpoint/energy.hold_s*3.)
        actual=v+error
        wrench=energy.wrench_environment-(energy.wrench_error+midpoint*energy.wrench_rate)*np.sign(actual)
        local=energy.contact_map @ actual
        assert np.all(abs(local)<=energy.contact_speed_bound+1e-10)
        actual_work+=step*float(wrench @ actual-energy.damping @ local**2)
        prefix=(i+1)*step
        assert actual_work >= energy.lower_work_j(v,prefix)-1e-10
        assert energy.available_j+actual_work>=-1e-10


def test_energy_100000_ticks_transparent_without_touching_budget_or_input_history():
    solver=ContactQp(CFG)
    energy=budget([0,-1,-4,0,.2,0],10.,wrench_error=np.ones(6)*.001)
    template=data(energy)
    previous=template.previous_twist.copy()
    for i in range(100000):
        nominal=PATH.copy();nominal[[2,4]]=[.0001*np.sin(i*.01),.003*np.cos(i*.007)]
        current=replace(template,nominal_twist=nominal)
        result=solver.solve(current)
        assert result.success and result.alpha==1.
        np.testing.assert_array_equal(result.qp_twist,nominal)
        assert result.diagnostics['transparent']
    np.testing.assert_array_equal(template.previous_twist,previous)
    assert energy.available_j==10.


@pytest.mark.parametrize('name',['available_j','hold_s','beta'])
@pytest.mark.parametrize('value',[True,False,np.bool_(True)])
def test_boolean_budget_scalars_are_not_energy_or_time(name,value):
    values=dict(wrench_environment=np.zeros(6),available_j=1.,hold_s=.005,beta=1.)
    values[name]=value
    with pytest.raises(ValueError,match='boolean'):PortEnergyConstraint(**values)


def test_prefix_overflow_never_becomes_returned_energy_credit():
    energy=budget(np.full(6,1e308))
    with pytest.raises(ValueError,match='nonfinite prefix'):energy.lower_work_j(np.ones(6),.005)
    assert energy.lower_work_j(np.ones(6),0.)==0.
    assert energy.lower_power_w(np.ones(6))==-np.inf


@pytest.mark.parametrize('now',[np.nan,np.inf,-np.inf,0.,.999])
def test_final_publication_time_must_be_finite_and_not_before_snapshot(now):
    result=ContactQp(CFG).solve(data(budget([0,-5,0,0,0,0],.00025)))
    assert not result.final_velocity_admissible(result.qp_twist,now_s=now)


def test_publication_age_expands_wrench_and_tracking_cost_for_full_hold():
    eta=np.zeros(6);eta[1]=.0001
    rates=np.zeros(6);rates[1]=100.
    energy=budget([0,-5,0,0,0,0],.00025,wrench_rate=rates,tracking_error=eta)
    result=ContactQp(CFG).solve(data(energy))
    assert result.success and result.final_velocity_admissible(result.qp_twist,now_s=1.)
    assert not result.final_velocity_admissible(result.qp_twist,now_s=1.007)
    aged=energy.aged(.007)
    assert aged.hold_s==energy.hold_s
    assert aged.wrench_error[1]==pytest.approx(.7)
    expected_loss=.7*(abs(result.qp_twist[1])+eta[1])
    assert energy.margin_power_w(result.qp_twist)-aged.margin_power_w(result.qp_twist)==pytest.approx(expected_loss)


def test_overflowing_linear_projection_cannot_hide_final_velocity_violation():
    mechanical=TwistConstraints(np.full((1,6),1e308),[-np.inf],[np.inf],valid_until_s=1.01)
    result=ContactQp(CFG).solve(data(budget(np.zeros(6)),mechanical=mechanical))
    assert result.success
    assert result.hard_constraints.violation(np.ones(6)*10)==np.inf
    assert not result.final_velocity_admissible(np.ones(6)*10,now_s=1.)
