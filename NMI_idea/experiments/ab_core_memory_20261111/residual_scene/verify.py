"""Verify the frozen residual-scene artifacts and independently check conditioning."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from scene import (BUDGET, DEV_SEED, DWELL, FEATURES, HERE, MAX_PROBES, NOMINAL,
                   PROBES, START, TERMINALS, TEST_SEED, Model, PublicTree,
                   digest, evaluate_split, fit_public_quality, geometry, read)


def verify_conditioning(duration):
    model = Model(np.array([.01, -.02]), np.array([[.07**2, .001], [.001, .24**2]]), .035**2)
    nominal = geometry(NOMINAL)
    actions = [1, 2, 2]
    y = np.array([-.2, -.1, -.12])
    design = FEATURES[actions]
    prior = model.covariance
    marginal = design @ prior @ design.T + model.noise_variance * np.eye(len(y))
    gain = np.linalg.solve(marginal, design @ prior).T
    post_mean = model.mean + gain @ (y - nominal['base'][0, actions] - design @ model.mean)
    post_cov = prior - gain @ design @ prior
    action = TERMINALS[-1]
    feat = FEATURES[action]
    for reset in (False, True):
        tree = PublicTree(model, nominal, duration, reset=reset)
        natural, counts = tree.natural, (0, 0)
        for probe, observation in zip(actions, y):
            natural, counts = tree.update(natural, counts, probe, observation)
        np.testing.assert_allclose(tree.covariance(counts), post_cov, atol=1e-13)
        np.testing.assert_allclose(natural @ tree.covariance(counts), post_mean, atol=1e-13)
        expected_mean = post_mean[0] + model.mean[1] if reset else feat @ post_mean
        expected_var = post_cov[0, 0] + prior[1, 1] if reset else feat @ post_cov @ feat
        p = ndtr((nominal['base'][0, action] + expected_mean) / np.sqrt(expected_var + model.noise_variance))
        np.testing.assert_allclose(tree.predict(natural, counts)[0][-1], p, atol=1e-13)


def verify_rows(result, duration):
    common = {}
    for arm, methods in result['rows'].items():
        for method, rows in methods.items():
            summary = result['arms'][arm][method]
            assert summary['n'] == len(rows) == result['n']
            assert summary['passes'] == sum(row['pass_'] for row in rows)
            for obj, row in enumerate(rows):
                *probes, terminal = row['actions']
                assert len(probes) == len(row['public_probe_y']) <= MAX_PROBES
                assert all(a in PROBES for a in probes) and terminal in TERMINALS
                actions = [START, *row['actions']]
                cost = sum(duration[a, b] + DWELL[b] for a, b in zip(actions, actions[1:]))
                assert abs(cost - row['time_s']) < 1e-12 and cost <= BUDGET
                assert row['pass_'] == (row['physical_pass'] and row['acoustic_pass'])
                visits = {a: 0 for a in PROBES}
                for a, y in zip(probes, row['public_probe_y']):
                    key = obj, a, visits[a]
                    assert y == common.setdefault(key, y)
                    visits[a] += 1
    for a, b in zip(result['rows']['persistent']['h4'], result['rows']['resampled']['h4']):
        assert all(a[key] == b[key] for key in ('actions', 'public_probe_y', 'joint_probability', 'time_s'))
    for a, b in zip(result['rows']['resampled']['h1'], result['rows']['resampled']['reset_h4']):
        assert a['actions'] == b['actions'] and len(b['actions']) == 1


def verify(out, replay):
    protocol = read(out/'PROTOCOL.json')
    for name, sha in protocol['source_sha256'].items():
        assert digest(Path(name)) == sha, name
    for name, sha in protocol['pretest_artifact_sha256'].items():
        assert digest(out/name) == sha, name
    robot = read(out/'ROBOT.json')
    assert len(robot['endpoint_witnesses']) == 5
    assert all(w['found'] and w['check']['valid'] and w['q_solution'][0] == robot['fixed_rail_m']
               for w in robot['endpoint_witnesses'])
    assert max(robot['prodmp_max_endpoint_errors']) < 1e-10
    duration = np.array(robot['prodmp_motion_s'])
    verify_conditioning(duration)
    with np.load(out/'TRAIN_PUBLIC.npz') as train:
        assert set(train.files) == {'public_y', 'poses'}
        fitted, _ = fit_public_quality(train['public_y'], geometry(NOMINAL)['base'][0])
    model = Model.from_file(out/'MODEL.json')
    assert model.serialized() == fitted.serialized()
    predictions = read(out/'PUBLIC_PREDICTIONS.json')
    for group in ('persistent_model', 'reset_aware_model'):
        for horizon in ('h1', 'h2', 'h3', 'h4'):
            assert predictions['25'][group]['root'][horizon]['chosen_action'] == predictions['41'][group]['root'][horizon]['chosen_action']
    for filename, seed in (('DEV.json', DEV_SEED), ('TEST.json', TEST_SEED)):
        result = read(out/filename)
        assert result['seed'] == seed
        if filename == 'TEST.json':
            assert result.pop('frozen_protocol_sha256') == digest(out/'PROTOCOL.json')
        verify_rows(result, duration)
        if replay:
            assert result == evaluate_split(seed, result['n'], model, duration)
    print('Verified input hashes, endpoints, independent Gaussian conditioning, public-only fit, paired action logs and budget.'
          + (' Dev and test replayed exactly.' if replay else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    verify(args.output, args.replay)
