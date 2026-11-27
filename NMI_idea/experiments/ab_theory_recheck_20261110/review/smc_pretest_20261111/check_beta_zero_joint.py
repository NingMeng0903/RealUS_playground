"""Independent finite fixture for beta=0 with a hard-zero new block.

This script writes audit-only output. It never opens the locked cohort.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import warnings

import numpy as np

ROOT = Path('/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_theory_recheck_20261110')
sys.path.insert(0, str(ROOT))
from inference import continuous_smc as smc


def run():
    particles = [0, 1, 2, 3]
    weights = np.array([.8, .1, .05, .05])

    def block(ps):
        return np.array([-np.inf if p == 0 else 0. for p in ps])

    def history(ps):
        return np.zeros(len(ps))

    def joint(ps):
        return history(ps), block(ps)

    source = Path(smc.__file__)
    result = {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'weights': weights.tolist(),
        'initial_ESS': smc.raw_ess(weights),
        'resample_threshold': .6 * len(particles),
        'block_loglike': ['-inf', 0, 0, 0],
        'cases': {},
    }
    for name, callback in [('separate_callbacks', None), ('joint_callbacks', joint)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                out = smc.adaptive_tempered_resample_move(
                    particles, weights, block, history, lambda p: 0.,
                    lambda p, rng, scale: p, np.random.default_rng(0),
                    mh_sweeps=1, joint_loglike=callback,
                )
                record = dict(committed=out.committed, particles=out.particles,
                              weights=out.weights.tolist(), diagnostics=out.diagnostics)
            except Exception as error:
                record = dict(error_type=type(error).__name__, error=str(error))
            record['warnings'] = [str(item.message) for item in caught]
        result['cases'][name] = record
    result['equivalent_committed_outputs'] = all(
        item.get('committed', False) for item in result['cases'].values()
    ) and result['cases']['separate_callbacks']['particles'] == result['cases']['joint_callbacks']['particles']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-source', type=Path)
    args = parser.parse_args()
    if args.module_source is not None:
        spec = importlib.util.spec_from_file_location('smc_preserved_audit_fixture', args.module_source)
        smc = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = smc
        spec.loader.exec_module(smc)
    result = run()
    # Never overwrite evidence from the first failed sealed-source fixture.
    name = 'BETA_ZERO_JOINT_' + result['source_sha256'][:12] + '.json'
    target = Path(__file__).parent / name
    if target.exists():
        raise SystemExit(f'Audit artifact already exists: {target}')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
