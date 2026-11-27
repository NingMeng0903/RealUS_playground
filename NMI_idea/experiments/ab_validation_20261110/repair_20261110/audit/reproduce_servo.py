"""Fresh real-loop rerun redirected entirely into the independent audit folder."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent
WORKER = REPAIR / 'servo'
DEST = HERE / 'servo_fresh'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ['causal_no_ruckig_config.json', 'causal_module_c_bypass_config.json']:
        shutil.copyfile(WORKER / name, DEST / name)
    script = WORKER / 'run_repaired_servo.py'
    spec = importlib.util.spec_from_file_location('audit_repaired_servo', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = DEST
    module.main()
    comparisons = {}
    fresh = json.loads((DEST / 'metrics.json').read_text())
    original = json.loads((WORKER / 'metrics.json').read_text())
    for row in fresh['rows']:
        name = row['id']
        target = next(x for x in original['rows'] if x['id'] == name)
        a = np.load(DEST / f'{name}.npz'); b = np.load(WORKER / f'{name}.npz')
        arrays = {}
        for key in a.files:
            same_nan = bool(np.array_equal(np.isnan(a[key]), np.isnan(b[key])))
            finite = np.isfinite(a[key]) & np.isfinite(b[key])
            error = float(np.max(np.abs(a[key][finite] - b[key][finite]))) if finite.any() else 0.
            assert same_nan and error < 1e-10, (name, key, error)
            arrays[key] = {'max_abs_difference': error, 'same_missing_mask': same_nan}
        assert row['ruckig_constructor_calls'] == target['ruckig_constructor_calls']
        comparisons[name] = {'arrays': arrays, 'visible_fraction': row['visible_fraction'],
            'ruckig_constructor_calls': row['ruckig_constructor_calls'],
            'truth_geometry_u_abs_mean_mm': row['truth_geometry_u_abs_mean_mm']}
    result = {'fresh_real_loop_rows': len(fresh['rows']), 'comparisons': comparisons,
              'fresh_missing_noninterference': fresh['missing'],
              'harness_reviewed_and_reexecuted': str(script), 'harness_sha256': digest(script),
              'source_provenance': fresh['provenance'],
              'scope': 'Probe-pose loop simulation on actual production geometry; no robot joint QP, contact physics or ultrasound images.'}
    (HERE / 'servo_reproduction.json').write_text(json.dumps(result, indent=2))
    print('INDEPENDENT FULL SERVO REPRODUCTION PASSED', len(comparisons), flush=True)


if __name__ == '__main__':
    main()
