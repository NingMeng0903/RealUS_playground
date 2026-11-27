"""Independent read-only split/score audit and fresh actual QP subset."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
D = HERE.parent/'dird'
BASE = HERE.parents[1]
SOURCE = Path('/media/camp/EXT_DRIVE/RealUS_playground/ird_playground')
sys.path.insert(0, str(D))
from common import clean_json
from qp_replay import QPReplayBridge
from ird_playground.neural.signed_field import ReachabilitySDF


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    sealed = read(D/'SEALED_MANIFEST.json')
    assert all(sha(p) == h for p,h in sealed['files_sha256'].items())
    olddir = SOURCE/'data/checkpoints/rm4d_signed_current_robot_20261108'
    old = read(olddir/'train_report.json'); new = read(D/'candidate/train_report.json')
    canonical = SOURCE/'data/ird/current_robot_20261108/canonical.npz'
    raw = np.load(canonical, allow_pickle=False)
    source = raw['source_pose_id'].astype(np.int64); boundary = raw['boundary_id'].astype(np.int64)
    groups = np.empty(len(source), np.int64); mask = source >= 0
    uniq, inverse = np.unique(source[mask], return_inverse=True); groups[mask] = inverse
    count = len(uniq); left = (source < 0) & (boundary >= 0)
    uniq, inverse = np.unique(boundary[left], return_inverse=True); groups[left] = count+inverse
    count += len(uniq); remaining = np.flatnonzero((source < 0) & (boundary < 0))
    groups[remaining] = np.arange(count, count+len(remaining)); count += len(remaining)
    order = np.arange(count); np.random.default_rng(20261128).shuffle(order)
    assignment = np.zeros(count, np.int8); start=0
    names = ['train','selection','zero_calibration','safety_calibration','test']
    for label, fraction in enumerate([.15,.05,.05,.1], 1):
        size = int(np.floor(count*fraction)); assignment[order[start:start+size]] = label; start += size
    row_owner = assignment[groups]
    splits = {name:np.flatnonzero(row_owner==i) for i,name in enumerate(names)}
    fingerprint = hashlib.sha256()
    for name in sorted(splits):
        fingerprint.update(name.encode('ascii')); fingerprint.update(source[splits[name]].astype('<i8').tobytes())
    digest = fingerprint.hexdigest(); counts = {k:len(v) for k,v in splits.items()}
    assert counts == old['split_counts'] == new['split_counts']
    assert digest == old['split_fingerprint'] == new['split_fingerprint']
    # Independent check of both nonnegative source ownership and boundary ownership.
    overlaps = []
    for i, first in enumerate(names):
        for second in names[i+1:]:
            for kind, ids in [('source',source),('boundary',boundary)]:
                x=ids[splits[first]]; y=ids[splits[second]]
                n=len(np.intersect1d(x[x>=0],y[y>=0]))
                if n: overlaps.append([kind,first,second,n])
    assert not overlaps
    bridge = QPReplayBridge('native','urdf')
    audits = read(D/'independent_model_audit.json'); global_data=np.load(D/'fresh_global_audit.npz')
    model_records={}
    for name, directory in [('old',olddir),('new',D/'candidate')]:
        conf=read(directory/'conformal.json'); checkpoint=directory/'selected.pt'
        assert sha(checkpoint)==conf['checkpoint_sha256']
        assert abs(conf['zero_bias']+conf['safety_threshold']-conf['m_safe'])<1e-15
        field=ReachabilitySDF.load(checkpoint,device='cpu',expected_robot=bridge.assets.spec)
        score=field.score_np(global_data['flange_canonical']); truth=global_data['reachable']>.5
        assert np.max(np.abs(score-np.asarray(audits['models'][name]['global']['scores'])))<1e-6
        classified=score>=conf['zero_bias']; accepted=score>=conf['m_safe']
        model_records[name]={'checkpoint_sha256':sha(checkpoint),'raw_threshold':conf['m_safe'],
            'balanced_accuracy':float(.5*(classified[truth].mean()+(~classified[~truth]).mean())),
            'false_accept_count':int(accepted[~truth].sum()),'negative_count':int((~truth).sum())}
    manifest=bridge.manifest()
    frozen=np.load(BASE/'dird/trajectory_paths.npz')
    seed0=np.asarray(read(BASE/'dird/nominal_context.json')['q_seed'])
    seed4=np.asarray(read(D/'initialization_sensitivity_seed.json')['q_seed'])
    records=[]
    for label,index,seed in [('branch0_path0',0,seed0),('branch4_path0',0,seed4),('branch4_anomaly_path1',1,seed4)]:
        result=bridge.query_pose_path(frozen['no_dird_T_tcp_railbase'][index],seed,
            speed_m_s=.02,settle_s=2.,provenance='independent phase2 fresh frozen QP subset',return_trace=True)
        times=np.array([x['native_model_t_mono'] for x in result['trace']])
        assert np.max(np.abs(times-(1000+np.arange(len(times))*.005)))<1e-10
        record={k:v for k,v in result.items() if k not in ['trace','exit_state']}
        record['label']=label; record['exit_state_valid_flag']=result['exit_state']['valid']
        if result['failures']:
            bad=result['failures'][0]; last=result['trace'][-1]
            assert bad['rejected_before_ideal_plant_integration']
            assert np.array_equal(last['q'],bad['q_meas_before'])
            assert np.array_equal(result['q_exit'],bad['q_meas_before'])
            rejected=False
            try:
                bridge.query_pose_path(frozen['no_dird_T_tcp_railbase'][0],result['q_exit'],initial_state=result['exit_state'],return_trace=False)
            except ValueError:
                rejected=True
            assert rejected
            record['invalid_state_continuation_rejected']=rejected
            record['last_valid_q_preserved']=True
        records.append(record)
        print(label,'steps',result['n_steps'],'failures',len(result['failures']),'p95_mm',1000*result['position_error_p95_m'],flush=True)
    result={'sealed_manifest_all_hashes_match':True,'split_counts':counts,'split_fingerprint':digest,
        'source_and_boundary_group_overlaps':overlaps,'fresh_global_model_rescoring':model_records,
        'fresh_native_QP_subset':records,'robot_contract':manifest['robot_contract'],
        'TCP_cache_comparison':{k:manifest[k] for k in ['cached_tool_name','cache_contract_translation_difference_m','cache_contract_rotation_difference_rad','live_active_tcp_confirmed']},
        'effective_backend':manifest['backend'],'effective_tcp_mode':manifest['tcp_mode'],
        'fingerprint':bridge._state_fingerprint(),'sealed_manifest_sha256':sha(D/'SEALED_MANIFEST.json')}
    (HERE/'dird_phase2_checks.json').write_text(json.dumps(clean_json(result),indent=2))
    print('PHASE2 INDEPENDENT CHECKS COMPLETE',flush=True)


if __name__=='__main__':main()
