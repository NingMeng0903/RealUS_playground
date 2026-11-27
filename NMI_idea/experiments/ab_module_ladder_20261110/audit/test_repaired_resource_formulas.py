"""Independent probability/resource algebra fixtures, not robot performance."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'planning'))
import system as s

ROOT_POSES=np.array([[.01,0,0],[.02,0,0]])
NEXT_POSES=np.array([[.03,0,0],[.04,0,0]])
W=np.array([.6,.4])
C1=np.array([[.2,.1],[.1,.1]])
P1=np.array([[.7,.2],[.1,.2]])
C2=np.array([[.9,.6],[.1,.3]])

class FixtureRobot:
    q_seed=np.zeros(8)
    vmax=np.ones(8)
    queries=0
    def costs(self,source,poses,q,use_cap=True):
        cost=[{.01:2.,.02:3.,.03:4.,.04:6.}[round(float(p[0]),2)] for p in poses]
        return np.array(cost),[dict(found=True,q_kin=np.zeros(8),proxy=float(p[0])) for p in poses]

def configured(monkeypatch,method):
    planner=s.Planner([object(),object()],method,FixtureRobot())
    planner.candidates=lambda *args:ROOT_POSES.copy()
    planner.forecast_candidates=lambda *args:NEXT_POSES.copy()
    planner.response=SimpleNamespace(predict=lambda poses:{'is_root':poses[0,0]<.03})
    monkeypatch.setattr(s,'quality_arrays',lambda pred:(C1,P1,np.ones_like(C1,bool)) if pred['is_root']
        else (C2,C2,np.ones_like(C2,bool)))
    return planner

def enumerate_fixed_rmt(first,second,R):
    c1=[2.,3.][first];c2=[4.,6.][second]
    total=0.
    for h,w in enumerate(W):
        p_correct=C1[h,first]
        p_wrong=P1[h,first]-p_correct
        p_continue=1-P1[h,first]
        second_loss=(C2[h,second]*(c1+c2)+(1-C2[h,second])*R) if c1+c2<=R else R
        total+=w*(p_correct*c1+p_wrong*R+p_continue*second_loss)
    return total

def test_terminal_is_censored_time_not_time_plus_failure_penalty(monkeypatch):
    planner=configured(monkeypatch,'A_myopic')
    value,action,data=planner.terminal(np.zeros(3),[],W,np.zeros(8),remaining_time=20.)
    expected=np.array([sum(W[h]*(C1[h,a]*c+(1-C1[h,a])*20.) for h in range(2))
        for a,c in enumerate([2.,3.])])
    np.testing.assert_allclose(data['values'],expected,rtol=0,atol=1e-12)
    assert value==pytest.approx(expected.min())

def test_fixed_second_value_matches_enumerated_correct_wrong_missing_branches(monkeypatch):
    planner=configured(monkeypatch,'fixed_two')
    got=planner.decide(np.zeros(3),[],W,np.zeros(8),remaining_actions=2,remaining_time=20.)
    values=np.array([[enumerate_fixed_rmt(a,b,20.) for b in range(2)] for a in range(2)])
    a,b=np.unravel_index(np.argmin(values),values.shape)
    assert got.diagnostics['kind']=='fixed_two'
    assert got.value==pytest.approx(values[a,b],abs=1e-12)
    np.testing.assert_array_equal(got.pose,ROOT_POSES[a])
    np.testing.assert_array_equal(got.second,NEXT_POSES[b])

def test_unaffordable_second_has_no_fictional_success_gain(monkeypatch):
    planner=configured(monkeypatch,'fixed_two')
    got=planner.decide(np.zeros(3),[],W,np.zeros(8),remaining_actions=2,remaining_time=5.)
    expected=np.array([sum(W[h]*(C1[h,a]*c+(1-C1[h,a])*5.) for h in range(2))
        for a,c in enumerate([2.,3.])])
    assert got.value==pytest.approx(expected.min())
    assert got.diagnostics['kind']=='direct'
    assert got.second is None and planner.pending is None

def test_pending_fixed_action_ignores_changed_public_posterior(monkeypatch):
    planner=configured(monkeypatch,'fixed_two')
    planner.pending=NEXT_POSES[0].copy()
    planner.terminal=lambda *args,**kwargs:pytest.fail('fixed action must not adapt to image')
    got=planner.decide(np.zeros(3),[{'unread':'changed public image'}],np.array([0.,1.]),np.zeros(8),
        remaining_actions=1,remaining_time=5.)
    np.testing.assert_array_equal(got.pose,NEXT_POSES[0])
    assert got.diagnostics['kind']=='committed_fixed_second'

def test_empty_action_budget_has_remaining_failure_value(monkeypatch):
    planner=configured(monkeypatch,'B_two_step')
    got=planner.decide(np.zeros(3),[],W,np.zeros(8),remaining_actions=0,remaining_time=7.)
    assert got.pose is None and got.value==7.
