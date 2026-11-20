"""Verify frozen checksums and invariants of the paired scan traces."""

from itertools import product
import json

from evaluate import CONDITIONS, freeze_config, hash_path, summarize
from train import HERE, OUT, read_rows


def json_data(name):
    return json.loads((OUT/name).read_text())


def main():
    frozen = json_data('freeze.json')
    assert frozen == freeze_config()
    rows = read_rows(OUT/'heldout.jsonl.gz')
    keys = {(r['seed'], r['profile']) for r in rows}
    assert len(rows) == len(keys) and keys == set(product(range(*frozen['test_anatomies']), CONDITIONS))
    assert summarize(rows) == json_data('heldout_summary.json')
    for row in rows:
        assert row['deadline']-row['prefix_tick'] == 260
        for arm in row['arms'].values():
            assert arm['end_speed'] < 1.e-7 and arm['end_tick'] <= row['deadline']
            assert not arm['correct'] or arm['strict_at_cine']
            assert not arm['strict_at_cine'] or arm['ever_after_prefix']
            assert not (arm['correct'] and arm['wrong'])
            if arm['accepted_tick'] is not None:
                assert any(s['arrival'] < arm['accepted_tick'] <= s['depart_after_cine']
                           for s in arm['station_log'])
            for visit in arm['interventions']:
                assert max(map(abs, visit['from_v'])) > 1.e-7
                assert visit['query_tick']+visit['new_motion_frames']+18 <= row['deadline']
        fork = row['same_exit']
        if fork:
            assert row['prefix_tick'] < fork['tick'] < row['deadline']
            assert set([fork['fresh_choice'], fork['stale_choice'], 'continue']) <= set(fork['alternatives'])
    witnesses = json_data('heldout_witnesses.json')
    assert {r['seed'] for r in witnesses['anatomies']} == {r['seed'] for r in rows}
    assert witnesses['summary']['reachable'] == len({r['seed'] for r in rows})
    later = json_data('residual_freeze.json')
    assert later['software_sha256'] == hash_path(HERE/'residual.py')
    assert later['policy_sha256'] == hash_path(OUT/'residual_policy.joblib')
    assert len({r['seed'] for r in rows} & set(range(*later['test_range']))) == 0
    other = read_rows(OUT/'residual_holdout.jsonl.gz')
    assert {(r['seed'], r['profile']) for r in other} == set(product(range(*later['test_range']), CONDITIONS))
    for row in other:
        b = row['baseline_index']
        if b is None:
            assert row['hand'] == row['residual'] == row['zero_margin']
        else:
            assert row['hand'] == row['outcomes'][b]
            assert row['residual'] == row['outcomes'][row['chosen_index']]
            assert all(o['end_tick'] <= row['deadline'] for o in row['outcomes'])
    print(f'Validated {len(rows)} primary and {len(other)} secondary paired scans.')


if __name__ == '__main__':
    main()
