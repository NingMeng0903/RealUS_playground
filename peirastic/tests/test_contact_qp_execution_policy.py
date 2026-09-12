"""Numerical execution-policy contracts, without hardware or transport mocks."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.execution import QualityIntervals, QualityProgress
from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
from rm75_control.control.admittance_common.reference import MotionReference


def observation(frame, quality, *, source='camera', time_s=None):
    return SimpleNamespace(source_id=source, frame_seq=frame,
        effective_time_s=frame/30 if time_s is None else time_s,
        quality=np.asarray(quality,dtype=float))


def test_progress_uses_each_frame_once_and_refusal_does_not_commit_alpha():
    progress=QualityProgress(ramp_s=.4,c_min=.8)
    weak=observation(1,[0.,.9,1.])
    target,first=progress.preview(weak,.005)
    assert target==.25 and first==pytest.approx(.990625)
    for _ in range(40):
        assert progress.preview(weak,.005)==pytest.approx((target,first))
        assert progress.alpha==1.
    # Contradictory payload under the same image identity is not new evidence.
    same_identity=observation(1,[1.,1.,1.])
    assert progress.preview(same_identity,.005)==pytest.approx((target,first))
    progress.commit(first)
    assert progress.preview(weak,.005)[1]==pytest.approx(.98125)
    assert progress.preview(observation(2,[1.,1.,1.]),.005)[0]==1.


@pytest.mark.parametrize('ramp_s',[.2,.4,.8])
def test_progress_positive_floor_and_existing_ramp_set_attack_and_release(ramp_s):
    progress=QualityProgress(ramp_s=ramp_s,c_min=.8)
    dt=ramp_s/80
    weak=observation(1,[0.,1.,0.])
    previous=1.
    for _ in range(80):
        target,value=progress.preview(weak,dt)
        assert target==.25 and .25<=value<=1.
        assert abs(value-previous)<=.75*dt/ramp_s+1e-14
        progress.commit(value);previous=value
    assert progress.alpha==pytest.approx(.25,abs=2e-14)
    healthy=observation(2,[.81,.4,.99])
    for _ in range(80):
        target,value=progress.preview(healthy,dt)
        assert target==1. and 0<=value-previous<=.75*dt/ramp_s+1e-14
        progress.commit(value);previous=value
    assert progress.alpha==pytest.approx(1.,abs=2e-14)


def test_healthy_asymmetric_sides_do_not_slow_progress_and_dropout_is_not_recovery():
    progress=QualityProgress(ramp_s=.4,c_min=.8)
    assert progress.preview(observation(1,[.8,.1,1.]),.01)==(1.,1.)
    assert progress.preview(observation(2,[1.,.1,.8]),.01)==(1.,1.)
    assert progress.preview(observation(3,[.4,1.,1.]),.01)[0]==pytest.approx(.625)
    target,value=progress.preview(None,.01)
    assert target==.25 and value<1.
    progress.commit(value)
    assert progress.preview(None,.01)[1]<value
    # A fresh healthy frame requests recovery with the same slew limit.
    assert progress.preview(observation(4,[1.,1.,1.]),.01)[1]==pytest.approx(1.)


def test_quality_interval_annotations_do_not_disable_progress_or_count_control_ticks():
    events=[]
    monitor=QualityIntervals(.8,.03,.2,lambda event,**facts:events.append((event,facts)))
    annotated=QualityProgress(.4,.8);plain=QualityProgress(.4,.8)
    first=observation(1,[.5,1.,1.])
    for tick in range(61):
        # Repeated observation across many control ticks must not qualify as
        # repeated evidence of failed repair. New frame 2 does qualify.
        obs=first if tick<60 else observation(2,[.51,1.,1.])
        monitor.observe(obs,tick*.005,{'force_gate'} if tick==20 else set())
        proposal=annotated.preview(obs,.005)
        assert proposal==plain.preview(obs,.005)
        annotated.commit(proposal[1]);plain.commit(proposal[1])
    marks=[facts for event,facts in events if event=='quality_no_improvement']
    assert len(marks)==1 and marks[0]['diagnostic_only'] is True
    assert marks[0]['reasons']==['force_gate']
    assert annotated.alpha==plain.alpha and annotated.alpha>=.25
    monitor.observe(None,.31,())
    monitor.add_reasons({'image_resumed'})
    monitor.observe(observation(3,[.9,.9,.9]),.4,())
    end=next(facts for event,facts in events if event=='quality_interval_end')
    assert end['unique_frames']==2 and end['no_improvement_marked'] is True
    assert end['end_reason']=='quality_recovered'
    assert set(end['reasons'])=={'force_gate','image_unavailable','image_resumed'}


def test_improving_weak_quality_is_not_marked_and_scan_close_is_diagnostic():
    events=[]
    monitor=QualityIntervals(.8,.03,.2,lambda event,**facts:events.append((event,facts)))
    monitor.observe(observation(1,[.4,1.,.4]),0.,())
    monitor.observe(observation(2,[.6,1.,.6]),.3,())
    monitor.close(.4)
    assert not any(event=='quality_no_improvement' for event,_ in events)
    ends=[facts for event,facts in events if event=='quality_interval_end']
    assert len(ends)==2 and all(facts['end_reason']=='scan_ended' for facts in ends)


def test_final_rocking_history_is_rotated_into_current_tool_frame():
    smooth=RockingSmoothing(.8,2.,.04,1.)
    smooth.seed([.1,-.2,.3],1.)
    final_rotation=Rotation.from_euler('z',90,degrees=True).as_matrix()
    final_tool=np.array([0.,0.,0.,.04,.12,-.08])
    smooth.commit(final_tool,final_rotation,1.02)
    current_rotation=Rotation.from_euler('xyz',[20,-15,35],degrees=True).as_matrix()
    axis,bounds,facts=smooth.preview(current_rotation,1.03,.005)
    expected_base=final_rotation@final_tool[3:]
    expected_local=current_rotation.T@expected_base
    expected_acc_local=current_rotation.T@((expected_base-np.array([.1,-.2,.3]))/.02)
    np.testing.assert_allclose(axis,current_rotation[:,1],atol=1e-15)
    assert facts['previous_rad_s']==pytest.approx(expected_local[1])
    assert facts['previous_acceleration_rad_s2']==pytest.approx(expected_acc_local[1])
    assert facts['elapsed_s']==pytest.approx(.01)
    assert (bounds[2]+bounds[3])/2==pytest.approx(expected_local[1])


def test_rocking_preview_and_rejection_do_not_mutate_published_history():
    smooth=RockingSmoothing(.8,2.,.04,1.)
    smooth.seed([0.,.1,0.],1.)
    omega=smooth.omega_base.copy();acceleration=smooth.acceleration_base.copy()
    for now in [1.005,1.01,1.015]:
        _,_,facts=smooth.preview(np.eye(3),now,.005)
        assert facts['elapsed_s']==pytest.approx(now-1.)
        assert smooth.time_s==1.
        np.testing.assert_array_equal(smooth.omega_base,omega)
        np.testing.assert_array_equal(smooth.acceleration_base,acceleration)
    # A different final publication, rather than an earlier candidate, owns
    # the new watermark and the measured command-to-command acceleration.
    smooth.commit([0.,0.,0.,0.,.104,0.],np.eye(3),1.02)
    assert smooth.time_s==1.02
    assert smooth.omega_base[1]==pytest.approx(.104)
    assert smooth.acceleration_base[1]==pytest.approx(.2)


@pytest.mark.parametrize('inertia,damping,tau,jerk',[(.04,1.,.04,50.),(.08,1.,.08,25.),(.04,2.,.02,100.)])
def test_rocking_jerk_comes_from_existing_inertia_and_damping(inertia,damping,tau,jerk):
    smooth=RockingSmoothing(.8,2.,inertia,damping)
    assert smooth.tau_s==pytest.approx(tau)
    assert smooth.jerk_limit==pytest.approx(jerk)
    smooth.seed(np.zeros(3),0.)
    _,bounds,_=smooth.preview(np.eye(3),.01,.005)
    # From rest the jerk envelope is tighter than the acceleration envelope.
    assert bounds[5]==pytest.approx(jerk*.01**2)
    assert bounds[5]<bounds[3]


def test_mechanical_override_above_amax_keeps_tier_one_intersection_empty():
    smooth=RockingSmoothing(2.,2.,.04,1.)
    smooth.seed(np.zeros(3),0.)
    smooth.commit([0.,0.,0.,0.,1.,0.],np.eye(3),.01)
    _,bounds,facts=smooth.preview(np.eye(3),.02,.005)
    assert facts['previous_acceleration_rad_s2']==pytest.approx(100.)
    lower=max(bounds[::2]);upper=min(bounds[1::2])
    assert lower>upper  # Genuine conflict: it must trigger an explicit tier decision.
    certificate=dict(rocking_policy_tier=1,rocking_lower_rad_s=lower,rocking_upper_rad_s=upper)
    assert not smooth.review([0.,0.,0.,0.,1.,0.],certificate,1e-9)


def test_confidence_flip_force_gate_and_image_resume_have_bounded_final_increments():
    """Exercise envelopes with a scalar admissible publication, not a robot QP.

    The numerical demand is discontinuous; frame and force updates arrive at
    different rates. One proposal is refused to check publication-owned time.
    """
    smooth=RockingSmoothing(.5,1.,.08,1.)
    smooth.seed(np.zeros(3),0.)
    now=0.;last_time=0.;last_w=last_a=0.;largest_demand_step=0.;largest_final_step=0.;previous_demand=0.
    for tick in range(240):
        dt=[.005,1/179,1/150][tick%3];now+=dt
        side_difference=.8 if tick<70 or tick>=150 else -.8
        force_gate=1. if tick<90 or tick>=110 else .2
        image_available=not 120<=tick<150
        demand=.2*side_difference*force_gate if image_available else 0.
        largest_demand_step=max(largest_demand_step,abs(demand-previous_demand));previous_demand=demand
        _,bounds,facts=smooth.preview(np.eye(3),now,dt)
        lower=max(bounds[::2]);upper=min(bounds[1::2]);assert lower<=upper
        candidate=float(np.clip(demand,lower,upper))
        if tick==27:
            assert smooth.time_s==last_time
            continue
        elapsed=now-last_time;acceleration=(candidate-last_w)/elapsed
        assert abs(candidate)<=.5+1e-12
        assert abs(acceleration)<=1.+1e-12
        assert abs(acceleration-last_a)<=smooth.jerk_limit*elapsed+1e-11
        assert facts['elapsed_s']==pytest.approx(elapsed)
        facts.update(rocking_policy_tier=1,rocking_lower_rad_s=lower,rocking_upper_rad_s=upper)
        command=np.array([0.,0.,0.,0.,candidate,0.])
        assert smooth.review(command,facts,1e-12)
        smooth.commit(command,np.eye(3),now)
        largest_final_step=max(largest_final_step,abs(candidate-last_w))
        last_time=now;last_w=candidate;last_a=acceleration
    assert largest_demand_step>=.32-1e-12
    assert largest_final_step<.012


class LinearFiveMillimetreReference:
    duration_s=8.
    ramp=0.

    def sample(self,time_s):
        pose=np.zeros(6);pose[0]=.005*np.clip(time_s,0.,self.duration_s)
        return MotionReference(pose,np.zeros(6),time_s)


@pytest.mark.parametrize('hz',[150,179,200])
def test_actual_interval_reference_delivers_five_mm_s_at_variable_control_rates(hz):
    reference=FiniteIntervalReference(LinearFiveMillimetreReference())
    reference.set_origin(np.zeros(6),t_s=3.)
    accepted=3.;dt=1/hz;distance=0.
    for tick in range(8*hz):
        motion,h_ref=reference.candidate(accepted,dt)
        assert h_ref==pytest.approx(dt,abs=2e-12)
        assert motion.vel_ff[0]==pytest.approx(.005,abs=2e-12)
        distance+=motion.vel_ff[0]*dt
        accepted=reference.commit_time(accepted,1.,dt)
        if tick+1==4*hz:
            assert accepted==pytest.approx(7.,abs=2e-12)
    assert accepted==pytest.approx(11.,abs=2e-12)
    assert distance==pytest.approx(.04,abs=2e-12)
    assert reference.sample(accepted,dt).pose_d[0]==pytest.approx(.04)


def test_waits_refusals_and_zero_alpha_never_advance_or_create_reference_debt():
    reference=FiniteIntervalReference(LinearFiveMillimetreReference())
    reference.set_origin(np.zeros(6),t_s=10.)
    accepted=reference.commit_time(10.,1.,.01)
    before=reference.sample(accepted,.005).pose_d.copy()
    for _ in range(300):
        # Preview itself has no accepted clock. A refused/held publication
        # cannot advance even when actual wall time continues for seconds.
        motion,_=reference.candidate(accepted,.02)
        np.testing.assert_array_equal(motion.pose_d,before)
        assert reference.commit_time(accepted,0.,.02)==accepted
    resumed=reference.commit_time(accepted,1.,1/150)
    assert resumed-accepted==pytest.approx(1/150)
    assert reference.sample(resumed,1/150).pose_d[0]-before[0]==pytest.approx(.005/150)
