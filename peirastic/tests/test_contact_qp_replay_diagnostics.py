"""Offline diagnostic fixture checks; no physical repair/force claims."""
from dataclasses import asdict
import json
import numpy as np
import pytest
from scripts.evaluate_contact_qp_replay import evaluate_case,main,profile_config


def frame(quality=(.2,.8,.95),valid=(True,True,True)):
    return dict(frame_id='example:12',source_file='example.h5',frame_index=12,
                profiles={key:dict(quality=list(quality),valid=list(valid)) for key in ('v1','v2','v3')})


def test_no_image_baseline_is_transparent_not_invalid_image_slowdown():
    result=evaluate_case(frame(),'nominal_no_image',4.,'visual_isolation')
    assert result['alpha']==1. and result['success']
    np.testing.assert_array_equal(result['qp_twist_tool'],[0,.02,0,0,0,0])
    assert result['physical_motion_observed'] is False and result['repair_success_measured'] is None


def test_v3_left_bad_right_good_requests_left_compression():
    result=evaluate_case(frame(),'v3',4.,'visual_isolation')
    assert result['success'] and result['max_hard_violation']<=1e-8
    assert result['visual_requests_m_s'][0]>0 and result['visual_requests_m_s'][1]<0
    assert result['added_local_velocity_lcr_m_s'][0]>result['added_local_velocity_lcr_m_s'][2]
    assert result['v3_task_reference_low_window_repair_intent'][0]
    assert result['visual_satisfied_with_slack']


@pytest.mark.parametrize('force',[3.7,4.3])
def test_force_priority_is_respected_for_declared_recovery_nominal(force):
    result=evaluate_case(frame(),'v3',force,'force_recovery')
    assert result['success'] and result['force_sign_reliable']
    assert np.max(result['force_priority_added_endpoint_residual'])<=1e-8
    assert np.sign(result['nominal_twist_tool'][2])==np.sign(4-force)


def test_unknown_window_is_reported_not_counted_as_verified_repair():
    result=evaluate_case(frame(valid=(False,True,True)),'v3',4.,'visual_isolation')
    assert not result['image_valid'] and not result['v3_task_reference_valid']
    assert result['v3_task_reference_low_window_repair_intent'] is None


def test_cli_three_frame_fixture_streaming_reports_and_external_configs(tmp_path):
    data=tmp_path/'features.jsonl';output=tmp_path/'output'
    configs={key:asdict(profile_config(key,{})) for key in ('v1','v2','v3')}
    (tmp_path/'feature_configs.json').write_text(json.dumps(configs))
    data.write_text(''.join(json.dumps(frame(q))+'\n' for q in ((.2,.8,.95),(.95,.8,.2),(.9,.9,.9))))
    assert main(['--input',str(data),'--output',str(output),'--forces','4','--nominal-modes','isolated'])==0
    report=json.loads((output/'summary.json').read_text())
    rows=[json.loads(line) for line in (output/'qp_diagnostics.jsonl').read_text().splitlines()]
    assert report['frame_count']==3 and report['case_count']==len(rows)==12
    assert len(report['groups'])==4 and len(report['groups_by_source']['example.h5'])==4
    assert report['feature_configs']==json.loads(json.dumps(configs))
    assert all(not row['v3_task_reference_is_ground_truth'] for row in rows)
