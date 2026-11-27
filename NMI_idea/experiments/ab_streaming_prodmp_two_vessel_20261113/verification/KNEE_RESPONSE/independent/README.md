# Independent knee-response implementation checks

The audit independently reconstructs the anonymous two-role likelihood and
the joint prediction over static offset, OU motion, slope, and visibility.
It covers packets with zero, one, or two returns, predictions through 20
exposures, and the marginal retained across visits. A singleton zero slope
and a full unexcited slope lattice both recover the original OU posterior
and association up to floating-point rounding. The ordinary pre-change OU
scan and the updated shared scan produced identical outputs for one seeded
case, including all 201 sampled joint, FK, and chart states.

Pilot case 9000 and frozen response case 0 and null case 3 each have five
calibration passes and six held-out executions. The replay recomputes FK
and kinematic checks for the 6,633 robot samples along all 33 stored paths.
It independently reproduces 1,353 exposure packets, task scores, and
costs. The three slope-aware held-out executions per case had a maximum
association replay error of 2.11e-15. Corresponding fixed-angle
and varied-angle visits share their disturbance tapes, and the two
estimators start the held-out scan at the same joints with the same first
packet. Every replayed robot sample passed the configured kinematic checks.

The pilot rerun traced 21 acquisition scores and 28 in-segment feedback
attempts. All 18 accepted replacement suffixes preserved their initial
lateral state and segment endpoint time. The next poses of all 18
replacements appeared in the executed path. Evaluator label and physical
position changes altered the outcome while leaving controller decisions and
paths identical. Altering the private dropout, noise, and OU tape from
exposure 14 left the earlier records and path through that exposure
identical. The programmed body-angle schedule stayed fixed.

Run the scripts from this directory in the benchmark's Python environment.
Set `experiment` to the implementation directory, `among` to the Among_US
checkout, and `repair` to the frozen repair assets. Set `response` and
`null` to the respective frozen archive directories containing
`CASES.json.xz` and `TRACES.npz`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python audit_filter.py --experiment "$experiment"
python audit_legacy.py --experiment "$experiment" --among-us "$among" --repair "$repair"
python audit_pilot.py --experiment "$experiment" --among-us "$among" --repair "$repair" --pilot "$response" --case 0
python audit_pilot.py --experiment "$experiment" --among-us "$among" --repair "$repair" --pilot "$null" --case 3
```

The causal rerun also takes `--pilot "$pilot" --case 9000 --causal` with
the archived development case. `AUDIT.json` records the observed checks.
