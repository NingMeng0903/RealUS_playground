# Inverse-B motion-bank reproducibility record

This record covers the completed mechanics-only build.  It does not authorize
or imply a regenerated bank after the recorded build.

## Frozen inputs and outputs

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `work/inverse_b/motion.py` (current builder) | 27245 | `1a106c001bd1c74ef34374ab862a593def8f75fbb3c5b6fff9cf0ec0091df099` |
| `work/full_b/motion_bank.py` | 40840 | `cebd4db066aea9807401ebb575050a2f99bb94997447e93c311f1a9433b0a9d2` |
| `work/full_b/motion/BANK_MAIN.json` | 5479199 | `a29cad4b5fe37760093e312c5be7a140d7905252c51fe42905908b9fad339e47` |
| `work/RealUS_playground/NMI_idea/experiments/ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz` | 445068 | `e61ff1a05ca957674d3b146dc225b96cce61286094f0e1da052b5874c8219180` |
| `BANK.json` | 70699431 | `2c9f17ddbb0504cef7a5008d3cc9239e7bec56dbb64d048e42d62359c2d22875` |
| `PREFLIGHT.json` | 852346 | `21a1151c7ab6808df8b567d4502c49d6f0084f1655edaebd62de0b3915ff11bb` |
| `REJECTIONS.json` | 8163 | `c099a4bdb5753d581ec0a8167e203cb9a893476510796f2ac86111fff7fe90cb` |
| `BUILD_REPORT.json` | 1106 | `63f9fe94cb108972d8560952c2a83fe39e8e8c089c54f783faa0493800c5b651` |

The recorded build used `trace_mode=joint`, one BLAS/OpenMP thread, 90 nodes,
2,196 edges, 90 holds, 2,028 added directed edges, and a 1.5 s edge duration.
The recorded wall/process time in `BUILD_REPORT.json` is 531.826 s.  The bank
contains 60 new nodes at angles `-9,-6,-3,3,6,9` degrees; the inherited
`-12,0,12` degree views and all inherited IDs remain unchanged.

## Checks run after the build

The final bank was checked without regeneration:

1. Every inherited node and edge (30 and 168 respectively) compares equal to
   the corresponding object in `BANK_MAIN.json`, including scalar JSON types.
2. New-node count is 60; all new nodes pass endpoint and dIRD proxy checks and
   contain `target_robot`, `atlas_frame`, and view-construction provenance.
3. All 2,196 edges have duration 1.5 s, 76 sampled q/tip rows, and sampled
   joint-rate fraction at most one.
4. An explicit-repo loader test copied `BANK_MAIN.json` to an arbitrary
   temporary directory and called `load_base_builder(temp_base, --repo)`.  It
   resolved both the inherited builder and `CapabilityEvaluator` to the
   checkout under `--repo`, not the mounted archive.

The loader test command was run with the pinned runtime:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python - <<'PY'
# import motion.py, copy BANK_MAIN.json to a temporary arbitrary path,
# call load_base_builder(temp_path, repo), and assert source/module origins
# are both under repo/NMI_idea.
PY
```

The final bank was not rewritten after these checks.
