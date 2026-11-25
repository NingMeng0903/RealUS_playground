"""Verify exported frozen inputs and independently recount the paired results."""

from argparse import ArgumentParser
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        return json.load(f)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def teacher_evidence(exports):
    path = exports/'teacher_pairs.json.gz'
    if path.exists():
        return read(path)
    groups = {}
    for stage in ('teach','teach_cal'):
        rows = []
        for file in sorted((exports/'shards'/stage).glob('*.json.gz')):
            p = read(file)
            for search in p['search']:
                assert all(b['outcome']['completed_legal'] and b['execution']['start'] == p['prefix']
                           for b in search['branches'])
            assert p['proof']['completed_legal'] and p['proof_continue']['completed_legal']
            assert p['proof_execution']['start'] == p['proof_continue_execution']['start'] == p['prefix']
            rows.append({k:p[k] for k in ('case','layout_id','version','target_region','before_last',
                                         'actions','chosen','utility_delta','proof','proof_continue')})
        groups[stage] = rows
    if all(groups.values()):
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as file, gzip.GzipFile(fileobj=file,mode='wb',filename='',mtime=0) as zipped:
            zipped.write(json.dumps(groups,separators=(',',':')).encode())
        temp.replace(path)
    return groups


def recount_teacher(exports, config):
    result = {}
    summary = read(exports/'teacher_summary.json')
    version = sha(exports/'teacher_freeze.json')
    for stage,group in teacher_evidence(exports).items():
        assert len({r['layout_id'] for r in group}) == len(group) == config['counts'][stage]
        assert all(r['version'] == version for r in group)
        strict = [(r['proof_continue']['strict'],r['proof']['strict']) for r in group]
        counts = {'prefixes':len(group), 'selected_program':sum(r['chosen']>0 for r in group),
                  'proof_rescue':sum(not a and b for a,b in strict),
                  'proof_loss':sum(a and not b for a,b in strict),
                  'proof_strict':sum(b for _,b in strict), 'continue_strict':sum(a for a,_ in strict)}
        assert all(counts[k] == summary[stage][k] for k in counts)
        result[stage] = counts
    return result


def recount_student_comparison(rows, config, summary):
    bands = ('proof_0','proof_1')
    profile = np.asarray([r['case']['profile'] for r in rows])
    rng = np.random.default_rng(config['bootstrap_seed'])
    strata = [np.flatnonzero(profile==name) for name in ('ordinary','stress')]
    sampled = np.column_stack([rng.choice(ix,size=(8000,len(ix)),replace=True) for ix in strata if len(ix)])
    checked = {}
    for baseline in ('v13','v14_terminal','v15_online_reference'):
        key = f'v15_student-vs-{baseline}'
        truth = np.array([[(r['tapes'][b][baseline]['selected_ids'],r['tapes'][b]['v15_student']['selected_ids'])
                           for b in bands] for r in rows])
        strict = (truth==0).all(axis=-1).astype(int)
        wrong = (truth>0).any(axis=-1).astype(int)
        checked[key] = {}
        for metric,raw in (('strict',strict),('wrong_branch',wrong)):
            delta = np.diff(raw,axis=-1)[:,:,0]
            d = delta.mean(axis=1)*100
            stats = summary['pairs']['proof_only'][key][metric]
            assert (stats['increase'],stats['decrease']) == ((delta>0).sum(),(delta<0).sum())
            np.testing.assert_allclose(stats['point_pp'],d.mean(),atol=1.e-10)
            np.testing.assert_allclose(stats['ci95_pp'],np.quantile(d[sampled].mean(axis=1),(.025,.975)),atol=1.e-10)
            checked[key][metric] = stats
    return checked


def verify(exports, models):
    cfg = read(exports/'protocol.json')
    source = Path(__file__).resolve().parents[1]
    assert {n:sha(source/n) for n in cfg['source_sha256']} == cfg['source_sha256']
    assert sha(exports/'gpu_preflight.json') == cfg['gpu_preflight_sha256']
    assert sha(exports/'gpu_driver.txt') == cfg['gpu_driver_sha256']
    assert sha(exports/'data_freeze.json') == cfg['data_freeze_sha256']
    reader, teacher, student, test = (read(exports/f'{part}_freeze.json') for part in ('reader','teacher','student','test'))
    assert teacher['reader'] == reader['reader_sha256'] == sha(models/'reader.json')
    assert teacher['continuation_student'] == cfg['frozen_continuation_student_sha256']
    assert student['teacher'] == sha(exports/'teacher_freeze.json')
    assert teacher['protocol'] == test['protocol'] == sha(exports/'protocol.json')
    assert test['student'] == sha(exports/'student_freeze.json')
    assert student['models'] == test['models'] == {n:sha(models/n) for n in test['models']}
    rows = read(exports/'blind_paired.json.gz')
    summary = read(exports/'results.json')
    assert summary['test_freeze_sha256'] == sha(exports/'test_freeze.json')
    assert len({r['layout_id'] for r in rows}) == len(rows) == summary['layouts'] == cfg['counts']['blind']
    assert summary['verified_layout_split_counts'] == cfg['counts']
    assert all(r['version'] == summary['test_freeze_sha256'] for r in rows)
    bands, arms = cfg['blind_offsets'], cfg['blind_arms']
    checked = {}
    for b in bands:
        checked[b] = {}
        for row in rows:
            panel = row['tapes'][b]
            for a in ('v14_terminal','v15_terminal'):
                for k in ('acquisition_sha256','echo_candidates','capture'):
                    assert panel[a][k] == panel['v13'][k]
            assert panel['v15_terminal']['selected_ids'][:3] == panel['v13']['selected_ids'][:3]
        for a in arms:
            values = [r['tapes'][b][a] for r in rows]
            labels = np.asarray([r['selected_ids'] for r in values])
            outcome = {'strict':int((labels==0).all(axis=1).sum()),'wrong':int((labels>0).any(axis=1).sum()),
                       'r_correct':(labels==0).sum(axis=0).tolist(), 'r_wrong':(labels>0).sum(axis=0).tolist()}
            assert all(v['completed_legal'] and v['execution']['clock_preserved'] for v in values)
            assert all(outcome[k] == summary['bands'][b][a][k] for k in outcome)
            checked[b][a] = outcome
    pairs = read(exports/'reader_evidence_pairs.json')
    c = Counter((p['region'],name,b['before']==0,b['after']==0) for p in pairs['pairs'] for name,b in p['arms'].items())
    for r in range(4,7):
        for a in ('v13','v15'):
            v = pairs['by_region'][str(r)][a]
            assert (v['rescued'],v['lost']) == (c[r,a,False,True],c[r,a,True,False])
    return {'layout_count':len(rows),'paired_tapes':list(bands),'matched_early_regions_and_candidates':True,
            'checked_full_scans':len(rows)*len(bands)*3,'old_evidence_pairs':len(pairs['pairs']),
            'recounted_teacher':recount_teacher(exports,cfg),'recounted':checked,
            'recounted_student_proof':recount_student_comparison(rows,cfg,summary)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',type=Path,default=root/'run')
    parser.add_argument('--models',type=Path)
    args = parser.parse_args()
    result = verify(args.outputs,args.models or args.outputs/'models')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
