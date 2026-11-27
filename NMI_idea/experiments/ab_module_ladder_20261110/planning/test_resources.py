"""Budget, timing and private-observation contracts without robot processes."""
from types import SimpleNamespace
import numpy as np
import pytest
import system as s


class Robot:
    q_seed=np.zeros(8);vmax=np.ones(8);queries=0
    def endpoint(self,pose,q):return dict(found=True,q_kin=np.zeros(8),proxy=0.)
    def costs(self,source,poses,q,use_cap=True):
        return np.full(len(poses),.6),[self.endpoint(p,q) for p in poses]


def test_last_action_never_expands_an_unavailable_future(monkeypatch):
    particles=s.env.sample_prior(720281,8)
    planner=s.Planner(particles,'B_two_step',Robot())
    monkeypatch.setattr(s.env,'sample_block_from_prediction',lambda *a,**k:pytest.fail('no seventh action'))
    d=planner.decide(np.zeros(3),[],np.ones(8)/8,np.zeros(8),remaining_actions=1,remaining_time=5.)
    assert d.diagnostics['kind']=='myopic'


def test_no_time_or_no_actions_returns_no_motion():
    planner=s.Planner(s.env.sample_prior(720281,2),'B_two_step',Robot())
    assert planner.decide(np.zeros(3),[],np.ones(2)/2,np.zeros(8),remaining_actions=0).pose is None
    assert planner.decide(np.zeros(3),[],np.ones(2)/2,np.zeros(8),remaining_time=0.).pose is None


def test_fixed_second_cannot_use_new_observation_to_replan_when_unaffordable(monkeypatch):
    planner=s.Planner(s.env.sample_prior(720281,2),'fixed_two',Robot())
    planner.pending=np.ones(3)*.01
    monkeypatch.setattr(planner,'terminal',lambda *a,**k:pytest.fail('fixed plan cannot change with new history'))
    d=planner.decide(np.zeros(3),[],np.ones(2)/2,np.zeros(8),remaining_time=.3)
    assert d.pose is None and d.value==.3


def test_unaffordable_root_does_not_expand_future(monkeypatch):
    planner=s.Planner(s.env.sample_prior(720281,4),'B_two_step',Robot())
    monkeypatch.setattr(s.env,'sample_block_from_prediction',lambda *a,**k:pytest.fail('unaffordable future'))
    d=planner.decide(np.zeros(3),[],np.ones(4)/4,np.zeros(8),remaining_time=.3)
    assert d.pose is None


@pytest.mark.parametrize('seconds_per_update,expected',[(12.,False),(.01,True)])
def test_public_acceptance_must_finish_before_deadline(monkeypatch,seconds_per_update,expected):
    clock=[0.]
    class Belief:
        def __init__(self,*a,**k):self.weights=np.ones(1)
        def update(self,frame,pose):
            assert frame['frame_id'].startswith('acq-')
            clock[0]+=seconds_per_update
    class Planner:
        def __init__(self,*args,**kw):
            assert len(args)==3  # particles,method,robot; no private world/index
            self.response=SimpleNamespace(geometry_queries=0)
        def decide(self,*args,**kw):
            assert kw['remaining_actions']==6 and kw['remaining_time']==60.
            return s.Decision(np.zeros(3),0.,{})
    def render(*a,start_time=0.,**k):
        public=[dict(frame_id=f'private-42-{i}',time_s=start_time+offset,visible=True,
            contact=True,visible_length_m=.02,image_u_m=0.,identity_score=.95)
            for i,offset in enumerate(s.env.FRAME_OFFSETS)]
        audit=[dict(frame_id=f['frame_id'],time_s=f['time_s'],passed=True) for f in public]
        return public,audit
    monkeypatch.setattr(s,'Planner',Planner);monkeypatch.setattr(s.env,'Belief',Belief)
    monkeypatch.setattr(s.env,'render_block',render)
    monkeypatch.setattr(s.time,'perf_counter',lambda:clock[0])
    r=s.run_episode(object(),[object()],'B_two_step',Robot(),'private-42')
    assert r['correct_public_stop'] is expected
    assert r['rmt_s']<=60.
    assert r['events'][0]['timely_public_decision'] is expected
