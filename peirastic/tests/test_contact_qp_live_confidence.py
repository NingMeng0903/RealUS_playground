"""Live registration and preflight regression, without hardware or sockets."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.apps.contact_qp_features import process_parts
from peirastic.apps.contact_qp_check import observation_problem, wait_for_features
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, LatestObservation
from peirastic.contact_qp.runtime_config import load_study_config, validate_study_config


def config():
    return load_study_config(Path(__file__).parents[1]/'config/contact_qp/active_probe50_v8r3_tank.yaml')


def frame(monkeypatch, *, instance='first', crop=(753,1417,154,880), hflip=False):
    import sys
    # Nonblank scanlines; this test targets metadata/registration, not image quality.
    image=np.random.default_rng(19).integers(20,200,(24,32),dtype=np.uint8)
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(IMREAD_GRAYSCALE=0,imdecode=lambda *args:image))
    cfg=config()
    meta=dict(source_id='realus.us_framegrab',publisher_instance_id=instance,
        frame_index=0,capture_monotonic_ns=2_000_000_000,clock_domain='host_monotonic',
        crop_box=list(crop),hflip=hflip)
    extractor=FeatureExtractor(FeatureConfig(**cfg['feature']['config']))
    return process_parts([b'topic',json.dumps(meta).encode(),b'jpeg'],extractor,2.01)


def test_publisher_restart_keeps_registration_but_resets_frame_history(monkeypatch):
    cfg=config()
    validate_study_config(cfg)
    first=frame(monkeypatch)
    restarted=frame(monkeypatch,instance='restarted')
    assert first.registration_version==restarted.registration_version==cfg['feature']['registration_version']
    assert first.source_id!=restarted.source_id
    history=LatestObservation()
    assert history.accept(first) and history.accept(restarted)
    assert history.generation==1
    assert observation_problem(restarted,cfg,now_s=2.02) is None


@pytest.mark.parametrize('change',[{'crop':(754,1418,154,880)},{'hflip':True}])
def test_changed_crop_or_flip_still_rejected(monkeypatch,change):
    assert 'version mismatch' in observation_problem(frame(monkeypatch,**change),config(),now_s=2.02)


def test_registration_manifest_cannot_silently_disagree_with_hash():
    cfg=config()
    cfg['feature']['registration']['crop_box'][0]+=1
    with pytest.raises(ValueError,match='source/crop/flip'):
        validate_study_config(cfg)


def test_preflight_needs_three_distinct_good_frames_and_times_out_if_missing(monkeypatch):
    import peirastic.apps.contact_qp_check as check
    obs=frame(monkeypatch)
    clock=[2.02]
    monkeypatch.setattr(check.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(check.time,'sleep',lambda delta:clock.__setitem__(0,clock[0]+delta))
    sequence=iter([obs,obs,replace(obs,frame_seq=1),replace(obs,frame_seq=2)])
    subscriber=SimpleNamespace(snapshot=lambda:next(sequence),last_error='')
    result=wait_for_features(config(),timeout_s=.1,subscriber=subscriber)
    assert result['distinct_frames']==3
    assert clock[0]==pytest.approx(2.05)
    missing=SimpleNamespace(snapshot=lambda:None,last_error='')
    with pytest.raises(RuntimeError,match='no confidence messages'):
        wait_for_features(config(),timeout_s=.03,subscriber=missing)


def test_required_feedback_rejects_missing_input_explicitly(monkeypatch):
    from peirastic.tests.test_contact_qp_active import make_active,propose
    active,pose,clock,sink=make_active(monkeypatch)
    active.config['feature']['required']=True
    try:
        with pytest.raises(RuntimeError,match='confidence feedback missing'):
            propose(active,pose,clock)
        assert active.reference_time_s==0
        assert any(r.get('event')=='required_image_feedback_rejected' for r in sink.records)
    finally:active.close()
