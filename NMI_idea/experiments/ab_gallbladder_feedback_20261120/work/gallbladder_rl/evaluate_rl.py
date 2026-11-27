"""Paired, anatomy-held-out evaluation of the public recurrent policies.

This command requires the original simulator assets and trained checkpoints.
It never substitutes archived CSV outcomes or a toy scene for a new rollout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .env import GallbladderRLEnv, asset_fingerprints, require_assets


ARMS = ('legacy_hand', 'model', 'rl', 'hybrid', 'hybrid_frozen')
SOURCE = Path(__file__).resolve().parent


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.glob('*.py'))}


def reject_training_overlap(checkpoint, test_seeds):
    ranges = checkpoint.get('training_seed_ranges')
    if not ranges:
        ranges = [checkpoint['training_seed_range']]
    for item in ranges:
        overlap = sorted(s for s in test_seeds if int(item['first']) <= s <= int(item['last_used']))
        if overlap:
            raise ValueError(f'Test anatomy overlaps a training/curriculum ancestor: {overlap[:8]}')


def load_policy(path, *, planner_features, test_seeds, decision_frames):
    import torch
    from .recurrent_ppo import RecurrentActorCritic
    from .train_rl import CHECKPOINT_VERSION

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != CHECKPOINT_VERSION:
        raise ValueError('Unsupported checkpoint schema')
    config = checkpoint['config']
    expected = {'planner_features': planner_features, 'freeze_belief': False,
                'decision_frames': decision_frames,
                'observation_dim': GallbladderRLEnv.observation_dim,
                'privileged_dim': GallbladderRLEnv.privileged_dim}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Checkpoint {key}={config.get(key)!r}; evaluation requires {value!r}')
    # An outcome is not comparable if an unreported environment/network change
    # occurred after training. Keep an explicit new experiment for migrations.
    for name in ('env.py', 'recurrent_ppo.py'):
        actual = hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
        if checkpoint['source_info']['file_sha256'].get(name) != actual:
            raise ValueError(f'Checkpoint was produced with a different {name}')
    reject_training_overlap(checkpoint, test_seeds)
    model = RecurrentActorCritic(config['observation_dim'], config['privileged_dim'], config['hidden'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    metadata = {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'training_seed_ranges': checkpoint.get('training_seed_ranges', [checkpoint['training_seed_range']]),
                'interactions': checkpoint['interactions'], 'config': config}
    return model, metadata


def rollout(arm, seed, condition, stream, decision_frames, policies):
    # Every arm builds its own deterministic scene/tape with the same keys.
    # No outcome or hidden state from another arm is passed into this run.
    env = GallbladderRLEnv(planner_features=arm.startswith('hybrid'),
                          freeze_belief=arm == 'hybrid_frozen',
                          decision_frames=decision_frames, reset_mode='prefix')
    if arm == 'legacy_hand':
        env._load()
        scanner = env._api.Scanner(seed, condition, stream)
        outcome = scanner.scan(False, 'heuristic')
        return outcome
    obs, info = env.reset(seed, condition, stream)
    actor = None if arm == 'model' else policies['hybrid' if arm.startswith('hybrid') else 'rl']
    hidden = None if actor is None else actor.burn_in(info['warmup'])
    while True:
        if actor is None:
            action = env.model_action()
        else:
            # Inference deliberately does NOT pass info['privileged'] to act().
            action, _, _, hidden = actor.act(obs, hidden, deterministic=True)
        obs, _, done, info = env.step(action)
        if done:
            return info['outcome']


def paired_interval(rows, arm, reference, replicates=12000):
    by = {(r['seed'], r['condition'], r['arm']): int(r['correct']) for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    differences = []
    for seed in seeds:
        differences.append([by[seed, condition, arm] - by[seed, condition, reference]
                            for condition in ('wide', 'stress')])
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20261120)
    picks = rng.integers(len(seeds), size=(replicates, len(seeds)))
    estimates = differences.mean(axis=1)[picks].mean(axis=1)
    return {'arm': arm, 'reference': reference, 'anatomy_clusters': len(seeds),
            'correct_difference_pp': float(100*differences.mean()),
            'clustered_95ci_pp': (100*np.percentile(estimates, [2.5, 97.5])).tolist(),
            'wins': int((differences > 0).sum()), 'losses': int((differences < 0).sum()),
            'bootstrap_replicates': replicates, 'bootstrap_seed': 20261120}


def summarize(rows, arms):
    summary = {}
    for arm in arms:
        subset = [r for r in rows if r['arm'] == arm]
        summary[arm] = {'n': len(subset), 'correct': sum(r['correct'] for r in subset),
                        'wrong': sum(r['wrong'] for r in subset),
                        'ever_strict_including_prefix': sum(r['ever'] for r in subset),
                        'strict_at_confirmation_cine': sum(r['strict_at_cine'] for r in subset),
                        'mean_seconds_after_prefix': float(np.mean([r['seconds'] for r in subset])),
                        'mean_travel_after_prefix_mm': float(np.mean([r['travel'] for r in subset]))}
    comparisons = []
    for reference in ('model', 'legacy_hand'):
        if reference in arms:
            for arm in arms:
                if arm != reference:
                    comparisons.append(paired_interval(rows, arm, reference))
    if 'hybrid' in arms and 'hybrid_frozen' in arms:
        comparisons.append(paired_interval(rows, 'hybrid', 'hybrid_frozen'))
    return {'arms': summary, 'paired_comparisons': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=['model', 'rl'])
    parser.add_argument('--policy-checkpoint', type=Path)
    parser.add_argument('--hybrid-checkpoint', type=Path)
    parser.add_argument('--seed-start', type=int, default=58720000)
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--stream', type=int, default=50000)
    parser.add_argument('--decision-frames', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count < 1 or args.decision_frames < 1 or args.seed_start < 0 or args.stream < 0:
        parser.error('count and decision-frames must be positive; seed and stream nonnegative')
    if len(args.arms) != len(set(args.arms)):
        parser.error('Duplicate arms are not allowed')
    if 'rl' in args.arms and args.policy_checkpoint is None:
        parser.error('--policy-checkpoint is required for rl')
    if any(a.startswith('hybrid') for a in args.arms) and args.hybrid_checkpoint is None:
        parser.error('--hybrid-checkpoint is required for hybrid arms')
    if args.output.exists():
        parser.error('Use a new output directory; previous outcomes must not be overwritten')
    # Fail before any output files are made if original anatomy is unavailable.
    require_assets()
    seeds = set(range(args.seed_start, args.seed_start+args.count))
    policies, checkpoint_metadata = {}, {}
    for name, path in (('rl', args.policy_checkpoint), ('hybrid', args.hybrid_checkpoint)):
        if path is not None:
            policies[name], checkpoint_metadata[name] = load_policy(
                path, planner_features=name == 'hybrid', test_seeds=seeds,
                decision_frames=args.decision_frames)
    args.output.mkdir(parents=True)
    manifest = {'base_commit': '101325037dcb9c899dd4553245f5aa505a9caab9',
                'kind': 'new_asset_backed_simulator_rollouts', 'start': 'after_shared_costal_sweep',
                'test_seed_range_inclusive': [min(seeds), max(seeds)],
                'noise_streams': [args.stream, args.stream+1],
                'arms': args.arms, 'decision_frames': args.decision_frames,
                'deterministic_actor': True, 'observation_kind': 'simulated_detector_readings_not_bmode',
                'common_executor_arms': [a for a in args.arms if a != 'legacy_hand'],
                'legacy_hand_note': 'unchanged source, different legacy action/decision interface',
                'frozen_note': 'freeze discrete organ/liver posterior throughout environment after prefix; images and continuous fit still update',
                'checkpoints': checkpoint_metadata, 'assets': asset_fingerprints(),
                'source_sha256': source_hashes()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = []
    with (args.output/'rollouts.jsonl').open('x') as detailed:
        for seed in sorted(seeds):
            for offset, condition in enumerate(('wide', 'stress')):
                for arm in args.arms:
                    outcome = rollout(arm, seed, condition, args.stream+offset,
                                      args.decision_frames, policies)
                    key = {'seed': seed, 'condition': condition, 'stream': args.stream+offset, 'arm': arm}
                    detailed.write(json.dumps(dict(key, outcome=outcome), allow_nan=False)+'\n')
                    detailed.flush()
                    row = dict(key, **{name: outcome[name] for name in
                        ('correct', 'wrong', 'ever', 'strict_at_cine', 'seconds', 'travel', 'end_tick', 'end_speed')})
                    rows.append(row)
            print(json.dumps({'completed_anatomy': seed, 'completed_paired_cases': len(rows)//len(args.arms)}), flush=True)
    with (args.output/'paired.csv').open('x', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(rows, args.arms)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
