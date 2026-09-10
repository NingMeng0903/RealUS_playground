# Historical coordinator dependency snapshot

This supplemental archive preserves the exact Python coordinator, its energy/geometry/types dependency closure, package initializers, and complete original execution/energy tests before removal from the active tree. The full energy test file includes the two branch-helper tests later removed from the active suite; active ledger tests remain in production. The parent archive manifest is unchanged.

This is historical research evidence, not the active publication adapter and not hardware acceptance. The active implementation keeps EnergyLedger and RuntimeEnergy.

Reproduce from this directory using an environment with NumPy and pytest:

```bash
cd source
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. /media/camp/EXT_DRIVE/envs/genesis/bin/python -m pytest peirastic/tests/test_contact_qp_execution.py peirastic/tests/test_contact_qp_energy.py -q
```

Run in a separate process rooted at `source` so its `peirastic` imports resolve to this snapshot. This dependency snapshot may also accompany the parent native archive; it does not recreate the full withdrawn native build environment.
