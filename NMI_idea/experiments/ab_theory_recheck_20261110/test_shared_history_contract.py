import json
from pathlib import Path
import numpy as np
import pytest
import shared_history_experiment as study

def test_anchor_uses_public_frames_and_leaves_pose_unchanged():
    pose=np.array([.01,.02,.2])
    blocks=[dict(pose=pose,public=[dict(visible=True,contact=True,
        image_u_m=.003,identity_score=.9)],audit_for_evaluation_only={'target':False})]
    first=study.public_anchor(blocks)
    blocks[0]['audit_for_evaluation_only']={'target':True,'true_pose':[100,100,100]}
    np.testing.assert_array_equal(study.public_anchor(blocks),first)
    np.testing.assert_array_equal(pose,[.01,.02,.2])

def test_no_visible_frame_has_fixed_public_fallback():
    blocks=[dict(pose=[.03,.03,.8],public=[dict(visible=False,contact=True)],
                 audit_for_evaluation_only={'target':[.03,.03,.8]})]
    np.testing.assert_array_equal(study.public_anchor(blocks),np.zeros(3))

def test_release_rejects_an_approved_empty_manifest(tmp_path):
    release=tmp_path/'release.json';release.write_text(json.dumps({'files':{}}))
    with pytest.raises(ValueError,match='omits required'):
        study.verify_release(release,study.sha(release))

def test_summaries_retain_failed_objects_and_label_score_denominator():
    common=dict(method='m',prefix=5,correct_public_stop=False,wrong_public_stop=False,
                inference_s=1.)
    rows=[dict(common,valid_posterior=True,predictive={'mean_capped_NLL':7.}),
          dict(common,valid_posterior=False)]
    result=study.summaries([{'condition':'iid','rows':rows}])['iid/m/prefix5']
    assert result['n']==2 and result['posterior_invalid_n']==1
    assert result['predictive_scored_n']==1
    assert result['mean_predictive_NLL_valid_only']==7.

def test_unfinished_posterior_is_not_used_or_missing_diagnostic(monkeypatch,tmp_path):
    class Incomplete:
        def __init__(self,*args,**kwargs):pass
        def update_block(self,*args):
            return dict(unfinished=True,posterior_valid=False,beta_final=.4,
                        temperature_path=[0.,.2,.4])
    monkeypatch.setattr(study,'METHODS',('tempered128',))
    monkeypatch.setattr(study,'TemperedBelief',Incomplete)
    world=study.env.sample_worlds(720990,1)[0]
    result=study.run_case(world,study.env.sample_prior(720991,128),
                          (0,'iid','contract'),tmp_path)
    rows=[r for r in result['rows'] if r['method']=='tempered128']
    assert len(rows)==2
    for row in rows:
        assert not row['valid_posterior'] and not row['correct_public_stop']
        assert row['steps'][0]['diagnostics']['temperature_path']==[0.,.2,.4]
        assert row['steps'][-1]['status']=='failed'
