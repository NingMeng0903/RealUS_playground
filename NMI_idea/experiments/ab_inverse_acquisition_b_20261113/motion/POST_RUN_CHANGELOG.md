# Post-run builder change log

The frozen bank was generated before the current builder was changed.  The
screen-time source snapshot is preserved as `motion_at_screen.py`.

| Snapshot | SHA-256 | Meaning |
| --- | --- | --- |
| `motion_at_screen.py` | `4d639fed33637afc38129f221d8188a29e682f05dc001d9eedb824e112ae8b64` | Exact builder source used for screen-time identity |
| `../motion.py` | `1a106c001bd1c74ef34374ab862a593def8f75fbb3c5b6fff9cf0ec0091df099` | Current reproducibility-oriented builder |

The only post-run source change is import resolution: the current builder
preloads the dIRD capability package from the explicit `--repo` checkout so a
checkout-local builder cannot silently resolve that package from the mounted
archive when `--base` is in an arbitrary temporary directory.  It does not
change the bank construction, node/edge generation, sampled traces, dIRD
selection, motion dynamics, or online controller.  The frozen `BANK.json` and
all outcome data remain untouched.

Independent verification:

```text
sha256(motion_at_screen.py) = 4d639fed33637afc38129f221d8188a29e682f05dc001d9eedb824e112ae8b64
sha256(motion.py)           = 1a106c001bd1c74ef34374ab862a593def8f75fbb3c5b6fff9cf0ec0091df099
diff scope                  = imports + load_base_builder capability preload only
```

The exact content diff is archived in `motion_postrun.diff`.
