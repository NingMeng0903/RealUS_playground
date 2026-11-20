# Gallbladder motion feedback and PSDP

This simulation experiment compares fixed-station scanning, feedback during probe motion, multimodal view selection, and backward policy fitting. The primary study has 80 paired trials and the replication has 64. Rule-based motion feedback increased correct stopped delivery from 45 to 54 in the primary study and from 42 to 45 in the replication. See [the report](REPORT.md) and [the paired results](reference_results/).

## Setup

Run from this directory in the full RealUS_playground checkout. Use Python 3.10 and the package versions in `requirements.txt`. The simulator reuses the repository's material atlas interpolation and ProDMP basis.

```bash
python3 -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Full simulation needs the original registered torso geometry, licensed SMPL-X male model, and generated torso atlas. Resolve the original inputs from the project's simulation setup. Set these variables to the absolute paths on your machine, or place the files at the shown paths under `local_assets/`. The [asset manifest](reference_results/asset_inputs.json) records the SHA256 of each input used for the published results.

| Environment variable | Default filename in `local_assets/` |
| --- | --- |
| `REALUS_GB_ANATOMY` | `registered_torso.npz` |
| `REALUS_GB_SMPLX` | `SMPLX_MALE.npz` |
| `REALUS_GB_ATLAS` | `thorax_atlas.npz` |

## Reproduce

```bash
python3 -B work/gallbladder_feedback/train.py reverse --workers 16
python3 -B work/gallbladder_feedback/train.py student --workers 18
python3 -B work/gallbladder_feedback/evaluate.py gate --workers 18
python3 -B work/gallbladder_feedback/evaluate.py test --workers 22
python3 -B work/gallbladder_feedback/check_curriculum.py
python3 -B work/gallbladder_feedback/check_witnesses.py
python3 -B work/gallbladder_feedback/audit_modes.py
python3 -B work/gallbladder_feedback/residual.py fit --workers 20
python3 -B work/gallbladder_feedback/residual.py test --workers 20
python3 -B work/gallbladder_feedback/verify_results.py
```

Training writes the model weights, manifests, and replay data to `work/gallbladder_feedback/results/`. Archived summaries, seed ranges, and the original training run's source and model hashes are in `reference_results/`. Reproduction generates fresh source hashes for the portable file layout. The experiment exercises a simulated surface chart and detector.
