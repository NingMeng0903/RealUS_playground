"""Causal interface counterexamples; recorded failing next QP inputs are absent."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from peirastic.contact_qp.qp import ContactQp,QpConfig,QpInput
from peirastic.contact_qp.types import ProbeGeometry
from peirastic.tests.test_contact_qp_active import make_active,propose


def test_recorded_004_model_residual_must_not_be_outer_slew_history():
    # Actual published 3357 values; hold nominal fixed for this counterexample.
    # This does not claim to reconstruct the unlogged failing input of 3358.
    command=np.array([.0008574774171818153,.004604096967433394,-.001106080811972955,
                      -.0010776662902027819,.009030374238297358,-.00025448135422153045])
    model=np.array([.0018870292478143209,.009673986235770118,-.0013931276165965688,
                    -.001077669286099677,.009030372701525175,-.00025448861491262514])
    path=command.copy();path[[2,4]]=0.
    inp=QpInput(ProbeGeometry.synthetic(),command,path,4.0832,.005,1.,previous_twist=model)
    qp=ContactQp(QpConfig())
    bad=qp.solve(inp)
    assert bad.qp_twist is None and bad.diagnostics['reason']=='motion_subspace_conflict'
    good=qp.solve(replace(inp,previous_twist=command))
    assert good.qp_twist is not None
    assert good.diagnostics['max_hard_violation']<=qp.config.feasibility_tolerance


def test_actual_slew_interval_does_not_change_hold_or_relative_force_rows():
    # Baseline normal slew of .8 m/s² over 6ms, future command hold 5ms.
    nominal=np.array([.002,0.,.0048,0.,0.,0.]);path=nominal.copy();path[2]=0.
    data=QpInput(ProbeGeometry.synthetic(),nominal,path,3.,.005,1.)
    qp=ContactQp(QpConfig())
    bad=qp.solve(data)
    assert bad.qp_twist is None
    assert bad.diagnostics['reason']=='relative_baseline_rows_conflict_with_mechanical_admission'
    good=qp.solve(replace(data,acceleration_dt_s=.006))
    assert good.qp_twist is not None and good.qp_twist[2]>=.0048-1e-8
    assert good.hard_constraints.valid_until_s==pytest.approx(1.01)
    assert good.diagnostics['command_hold_s']==.005
    assert good.diagnostics['command_slew_dt_s']==.006


def test_active_success_keeps_outer_history_separate_rotates_and_abort_preserves(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        command=propose(active,pose,clock);old_rotation=active.pending_rotation_base_tcp.copy()
        model=command.copy();model[1]+=.003
        active.publication_commit(active.pending_id,model,now_s=clock[0]+.001)
        np.testing.assert_array_equal(active._previous,command)
        # Nominal command integration uses the same accepted outer command.
        np.testing.assert_array_equal(active.controller.last_v_cmd,command)
        captured=[];solve=active.solver.solve
        active.solver.solve=lambda data:(captured.append(data) or solve(data))
        moved=pose.copy();moved[3:]=Rotation.from_matrix(old_rotation @ Rotation.from_rotvec([0,0,.01]).as_matrix()).as_euler(active.controller.cfg.euler_order)
        clock[0]+=.005
        propose(active,moved,clock)
        transform=active.pending_rotation_base_tcp.T @ old_rotation
        np.testing.assert_allclose(captured[-1].previous_twist,np.r_[transform@command[:3],transform@command[3:]],atol=1e-14)
        active.publication_abort('not sent',definitely_not_sent=True)
        np.testing.assert_array_equal(active._previous,command)
        np.testing.assert_array_equal(active._previous_rotation,old_rotation)
    finally:active.close()


def test_expired_review_remains_rejected_with_exact_timing_reason(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        command=propose(active,pose,clock)
        assert not active.publication_review(active.pending_id,command,now_s=clock[0]+.011)
        record=sink.records[-1]
        assert record['event']=='publication_review_rejected'
        assert record['reason']=='certificate_expired'
        assert record['valid_until_s']==pytest.approx(clock[0]+.01)
        assert active.reference_time_s==0.
    finally:active.close()


def test_normal_payload_residual_cannot_rebase_next_nominal_command(monkeypatch):
    # Two identical measured histories and accepted outer actions, but different
    # payload-model rail compensation. Neither nominal state nor the next outer
    # request may change solely because of that model residual.
    one,pose,clock_one,_=make_active(monkeypatch)
    two,_,clock,_=make_active(monkeypatch)
    def sample(active):
        return active.sample(0.,pose,np.zeros(6),f_ext_raw=np.zeros(6),dt_actual=.005,
            wrench_source_id='test-source',wrench_source_time_s=clock[0],wrench_source_wall_time_ns=1)
    try:
        for cycle in range(4):
            first=sample(one);second=sample(two)
            np.testing.assert_allclose(first,second,atol=1e-12,rtol=0)
            if cycle==0:assert first[2]==pytest.approx(.001,abs=1e-8)
            compensated=second.copy();compensated[2]+=.001
            one.publication_commit(one.pending_id,first,now_s=clock[0]+.001)
            two.publication_commit(two.pending_id,compensated,now_s=clock[0]+.001)
            np.testing.assert_allclose(one.controller.last_v_cmd,two.controller.last_v_cmd,atol=1e-12)
            np.testing.assert_allclose(two.controller.last_v_cmd,second,atol=1e-12)
            assert two.controller._u_force_slewed==pytest.approx(one.controller._u_force_slewed)
            clock[0]+=.005
        before=two.controller.last_v_cmd.copy();sample(two)
        two.publication_abort('rejected',definitely_not_sent=True)
        np.testing.assert_array_equal(two.controller.last_v_cmd,before)
    finally:one.close();two.close()


def test_normal_rebase_counterexample_keeps_force_priority_unchanged():
    path=np.array([.002,0.,0.,0.,0.,0.]);previous=path.copy();previous[2]=.001
    qp=ContactQp(QpConfig())
    # Wrong .002 model history permits nominal .006; actual outer slew stops at .005.
    wrong_nominal=path.copy();wrong_nominal[2]=.006
    data=QpInput(ProbeGeometry.synthetic(),wrong_nominal,path,3.,.005,1.,previous_twist=previous)
    assert qp.solve(data).diagnostics['reason']=='relative_baseline_rows_conflict_with_mechanical_admission'
    correct_nominal=path.copy();correct_nominal[2]=.005
    good=qp.solve(replace(data,nominal_twist=correct_nominal))
    assert good.qp_twist is not None and good.qp_twist[2]>=.005-1e-8
