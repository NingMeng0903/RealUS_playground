"""Audit provenance/causality, independent of large external acquisition files."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

path=Path(__file__).parents[2]/'scripts/replay_delay_outer.py'
spec=importlib.util.spec_from_file_location('logged_state_audit',path)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_publication_history_requires_two_successes_strictly_before_decision(tmp_path):
    rows=[]
    for identifier in (1,2,3):
        rows.extend([
            dict(event='control_sample',control_id=identifier,proposal_time_s=float(identifier),rotation_base_tcp=np.eye(3).tolist()),
            dict(event='publication_transport_result',control_id=identifier,publication_time_s=identifier+.01),
            dict(event='publication',control_id=identifier,success=True,facts=dict(arm='sent',rail='sent'),
                 final_command_model_tool=[.001,0,0,0,.01*identifier,0])])
    # A rejected/partial publication must never seed the successful history.
    rows.extend([dict(event='control_sample',control_id=4,proposal_time_s=4.,rotation_base_tcp=np.eye(3).tolist()),
        dict(event='publication_transport_result',control_id=4,publication_time_s=4.01),
        dict(event='publication',control_id=4,success=True,facts=dict(arm='sent',rail='failed'),
             final_command_model_tool=[0,0,0,0,9,0])])
    path=tmp_path/'log.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    _,publications,_,_,success=audit.read_log(path)
    times=np.array([p['time_s'] for p in publications])
    assert success=={1,2,3}
    assert audit.old_history_at(publications,times,2.01) is None
    selected=audit.old_history_at(publications,times,2.02)
    assert selected['control_id']==2
    np.testing.assert_allclose(selected['acceleration_base'],[0,.01,0])
    assert audit.old_history_at(publications,times,5.)['control_id']==3


def test_effective_feedback_is_residual_not_zero_or_scaled_scan():
    ff=np.array([.003,.004,.2,.005,.3,.006])
    class Ref:
        def candidate(self,t,dt):
            assert t==.7 and dt==.006
            return SimpleNamespace(vel_ff=ff),dt
    scan=np.array([.004,.006,0,.008,0,.009])
    sample=dict(reference_s=.7,reference_interval_s=.006,h_ref_s=.006,
        rotation_base_tcp=np.eye(3),fusion_components={'scan_requested_tool':scan})
    feedback,path=audit.split_path(sample,Ref())
    np.testing.assert_allclose(path,[.003,.004,0,.005,0,.006])
    np.testing.assert_allclose(feedback,[.001,.002,0,.003,0,.003])
    np.testing.assert_allclose(feedback+path,scan)
    np.testing.assert_allclose(feedback+.5*path,[.0025,.004,0,.0055,0,.006])


def test_worker_receipt_is_required_and_capture_time_does_not_authorize_image():
    from peirastic.contact_qp.features import FeatureConfig
    fc=FeatureConfig(calibration_version='fixture')
    data=dict(frame_seq=np.array([1,2]),effective_s=np.array([1.,1.1]),
        source_id=np.array(['source','source']),quality_lcr=np.ones((2,3))*.9,
        region_edges=fc.region_edges,region_confidence=np.ones((2,fc.region_count))*.7,
        region_valid=np.ones((2,fc.region_count),dtype=bool),
        valid_lr=np.ones((2,2),dtype=bool),confidence_lr=np.array([[.5,.9],[.6,.9]]))
    images,missing=audit.observations(data,{(1,1.):1.2},
        {'feature':{'registration_version':'fixture'}},fc)
    assert missing==1
    assert len(images)==1
    assert images[0].received_time_s==1.2
    assert images[0].effective_time_s==1.


def test_schema_does_not_claim_end_to_end_controller_replay():
    assert audit.SCHEMA=='frozen_logged_state_regional_one_step_qp_audit_v2'
    text=' '.join(audit.ASSUMPTIONS)
    for phrase in ('not an end-to-end','not reconstructed','zero angular state',
                   'not a separately logged feedback','No image improvement'):
        assert phrase in text
