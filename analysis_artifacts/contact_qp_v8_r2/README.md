# Bounded repair attempt revision

The first candidate's frozen failures remain in `../contact_qp_v8/`; this directory does not replace them. This revision follows the observed persistent-shadow failures. Seeds 801/802 are new validation units, not independent acceptance of acoustic efficacy after tuning.

The main comparison uses 5 scenarios × 2 seeds × 2 policies. The optional old-seed regression uses shadow only, seeds 701/702, and writes separate `regression_metrics.json`. Both reuse `peirastic.apps.contact_qp_experiment.run_case` with a local settings override, the same nominal controller, finite-area delayed plant and image feature chain. There is no hardware or controller service connection.

Commands, from the repository root with the supplied project environment:

```bash
python analysis_artifacts/contact_qp_v8_r2/compare.py --freeze
python analysis_artifacts/contact_qp_v8_r2/compare.py --workers 2
python analysis_artifacts/contact_qp_v8_r2/compare.py --regression --workers 2
python analysis_artifacts/contact_qp_v8_r2/energy_sweep.py
python analysis_artifacts/contact_qp_v8_r2/report.py
```

Freezing and execution refuse to overwrite existing manifests/results. For a reproduction, copy the three Python scripts into a new empty sibling artifact directory, retain the same code/configuration and run that directory's scripts. The `PYTHONPATH` must include both the repository and `rm75_control`; use the supplied rm75 environment and single-thread BLAS settings. Source hashes include every Python module in `peirastic/contact_qp`, including the repair episode module, and are compared before/after execution.

The unchanged 5 mm/s left/right gap scenarios previously remained coupled with image quality above 0.8 under the nominal controller. They cannot establish acoustic improvement under an active repair demand. Persistent shadow is deliberately unrepairable. Stopping its repeated repair attempts while completing with gaps establishes bounded behavior in this synthetic setting, not acoustic repair efficacy.
