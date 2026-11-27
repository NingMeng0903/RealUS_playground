#!/usr/bin/env python3
"""Independent pure-Python audit. Does not import the witness renderer/policy."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'mask_recovery_witness.py'
RESULT = ROOT / 'MASK_RECOVERY_WITNESS.json'
report = json.loads(RESULT.read_text())
rows = report['results']
candidates = (-5, 0, 5, 10, 15, 20, 25, 30)

def image_hash(u, has_branch):
    packed = bytearray()
    acc = 0
    bit_count = 0
    for row in range(177):
        z = -11 + row * 0.125
        for column in range(177):
            y = -11 + column * 0.125
            main = y*y + z*z <= 2.6**2
            branch = False
            if has_branch and u >= 14:
                center_y = min(8, .82*(u-14))
                branch = (y-center_y)**2 + z*z <= 1.6**2
            acc = (acc << 1) | int(main or branch)
            bit_count += 1
            if bit_count == 8:
                packed.append(acc)
                acc = bit_count = 0
    if bit_count:
        packed.append(acc << (8-bit_count))
    return hashlib.sha256(packed).hexdigest()

hashes = {(branch, u): image_hash(u, branch) for branch in (False, True) for u in range(-10, 36, 5)}
assert len(rows) == 24
assert len({(r['condition'], r['policy'], r['private_slip_mm']) for r in rows}) == 24
assert all(hashes[True, x+5] == hashes[True, x-5] for x in (-5, 0, 5))
assert all(hashes[False, x+5] == hashes[False, x-5] for x in candidates)
distinguishing = [x for x in candidates if hashes[True, x+5] != hashes[True, x-5]]
assert distinguishing == [15, 20, 25]
assert min(distinguishing, key=lambda x: (abs(x), x)) == report['first_discriminating_grid_position_mm'] == 15

counts = {}
matched_masks = 0
for r in rows:
    has_branch = r['condition'] != 'uniform_tube_symmetry'
    slip = r['private_slip_mm']
    modes = [slip] if r['condition'] == 'inner_registration_revealed' else [-5, 5]
    previous = 0
    travel = 0
    for x, h in r['observations']:
        assert h == hashes[has_branch, x-slip], (r['condition'], r['policy'], x)
        matched_masks += 1
        modes = [mode for mode in modes if hashes[has_branch, x-mode] == h]
        travel += abs(x-previous)
        previous = x
    if 'remaining_slips_mm' in r:
        assert modes == r['remaining_slips_mm']
    assert travel == r['total_travel_mm']
    chosen = r['return_body_x_mm']
    error = None if chosen is None else abs(chosen-slip)
    assert error == r['axial_error_mm']
    status = 'abstain' if chosen is None else 'correct' if error <= 2 else 'wrong'
    assert status == r['status']
    key = f"{r['condition']}/{r['policy']}"
    counts.setdefault(key, Counter())[status] += 1
    if r['policy'] == 'active_mask_recovery' and r['condition'] == 'tracked_branch':
        assert [x for x, _ in r['observations']] == [0, 15, slip]
    if r['policy'] == 'classical_fixed_sweep' and r['condition'] == 'tracked_branch':
        assert [x for x, _ in r['observations']] == [0, 5, 10, 15, slip]

# In this closed, exact two-mode model, a negative branch observation at x=15
# in the +5 world is informative; no extra private registration is required.
assert hashes[True, 10] == hashes[True, 0]
assert hashes[True, 20] != hashes[True, 0]

rotation = report['known_external_pose']['rotation']
max_orthonormal_error = max(abs(sum(rotation[k][i]*rotation[k][j] for k in range(3)) - int(i == j)) for i in range(3) for j in range(3))
assert max_orthonormal_error < 1e-12
assert report['fixed_short_axis_view_normal_world'] == [rotation[i][0] for i in range(3)]

out = {
    'audit_scope': 'Independent raster hash reconstruction, posterior filtering, counts, travel, target-error, sequence and public rigid-transform arithmetic. No import of witness helpers; no native/ultrasound experiment rerun.',
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'result_sha256': hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    'rows_checked': len(rows),
    'observation_hashes_independently_matched': matched_masks,
    'distinguishing_candidate_positions_mm': distinguishing,
    'active_first_probe_mm': 15,
    'counts': {k: dict(v) for k,v in counts.items()},
    'max_rotation_orthonormal_error': max_orthonormal_error,
    'tracked_branch_capture_counts_including_target': {'active': 3, 'fixed_sweep': 5},
    'tracked_branch_total_travel_mm_by_slip': {'-5': {'active': 35, 'fixed_sweep': 35}, '+5': {'active': 25, 'fixed_sweep': 25}},
    'conclusion': 'The frozen landmark recovery mechanism works in the ideal two-mode mask model. Full-history classical sweep matches accuracy and path. Exact template-guided active matching uses fewer captures; it is itself a classical same-information baseline. No full B-specific or sensor-noise/native/clinical advantage is demonstrated.'
}
(ROOT / 'ASTRA_MASK_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
