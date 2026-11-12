# Full-episode B versus replanning MPC

This study implements the full comparison requested after the [earlier B experiments](../ab_b_mpc_validation_20261113/README.md). It compares direct execution of the shared MPC with a two- or three-acquisition public-history option followed by that **same MPC to task termination**. It uses fresh synthetic ultrasound centroids and actual sampled kinematic costs from a checked robot graph. The original moving-Y data remain a separate historical experiment.

**Result.** The frozen B admission rule never switched from MPC in the independent cohort. Forcing the selected options exercised actual observation branches, early handoffs and recovery, without a reliable improvement at equivalent motion. A separate active cost diagnostic met the all-case sampled-motion margins and found one additional correct daughter completion alongside one additional wrong confirmation. Single-trunk outcomes tied.

[Protocol](PROTOCOL.md) · [Freeze and source hashes](FREEZE.json) · [Independent audits](INDEPENDENT_REVIEW.md) · [Archive hashes](DATA_MANIFEST.json)

## What executes

The two policies share the same public start, centroid likelihood, Bayesian updates, confirmation rule and robot paths. The paired tapes are indexed by physical acquisition clock and echo channel. The same geometry, internal motion, calibration offset and persistent false return remain in the case throughout. The sensor reports a discretized nearest unlabeled centroid and a multiple-return flag. A real trajectory's private label is evaluated only after the public claim.

The baseline uses a fixed two-capture plan during each receding optimization, with first-packet Bayesian updates and a resource-aware terminal potential. It replans after every actual observation. Prior claim correctness is tracked as joint probability mass per hypothesis. An option may choose a checked opposite view, return to the reference after a miss, or hand off early after a claim or concentrated displacement belief. Handoff retains the belief, prior-claim mass, configuration, acquisition clock, progress and unused resources. Candidate programs and the no-option baseline are compared on paired posterior episodes to final success, wrong claims and sampled costs. The admission rule uses independent model simulations. Its screening quantities are model estimates.

The main bank contains 30 robot nodes and 168 directed or hold edges. dIRD ranks endpoint proposals. Sampled FK, joint limits, speed and configured robot/probe self-collision check the traces. Equivalent proposals share a registered probe pose and differ in joint realization. Under the common caps, both tasks have complete mechanical routes with opposing tilts and alternate realizations. The daughter start pair has a small difference in reachable acoustic suffixes. The trunk start pair has the same acoustic support. Actual benchmark execution follows the endpoint, arrival time and integrated costs of a checked sampled trace. The benchmark does not integrate the native controller or tissue contact dynamics during a case.

## Frozen evaluation

There were 192 independent cases per task and initial-information condition. The same case seeds are reused across these conditions. Success requires **fresh correct identity at both required sites and completion of the checked scan**. Each entry is correct completions / any wrong confirmations.

| Task and initial information | MPC | Guarded B | Unguarded option |
| --- | ---: | ---: | ---: |
| Daughter, coarse prior | 48 / 20 | 48 / 20 | 52 / 18 |
| Daughter, common reference | 57 / 18 | 57 / 18 | 57 / 29 |
| Single trunk, coarse prior | 160 / 10 | 160 / 10 | 162 / 10 |
| Single trunk, common reference | 162 / 10 | 162 / 10 | 159 / 9 |

Guarded B, the one-observation selector and the fixed-program selector all handed off immediately in this cohort. Their sampled paths and outcomes are identical to MPC. The empirical paired intervals for identical traces are degenerate. For zero harmful discordances in 192 cases, the exact one-sided 97.5% upper bound on their probability is 1.90%. Identical observations do not establish population equality.

Unguarded B executes options on every case. Its best completion contrast is +4/192 in daughter with the coarse prior, from 20 wins and 16 losses. The paired 95% interval is −4.17 to +8.33 points, with +5.72 mm and +9.36° mean sampled motion. All four unguarded conditions fail the sampled-motion equivalence test. On daughter with the common reference it adds 11 wrong-confirmation cases. The recorded standalone planning CPU for guarded selection averages about 2.4–5.1 s by condition. The acoustic clock includes the selected motion duration. CPU accounting is separate and does not simulate tissue motion during planning.

[Frozen paired tables](data/confirmation/paired.json) · [Per-case records](data/confirmation/cases.jsonl.xz) · [Public-model forecasts](data/confirmation/plans.json.xz)

## Active cost diagnostic on a new cohort

For an active comparison, the independent cost-only diagnostic admits the best predicted short option within two-sided predicted motion margins. It drops the success and risk admission screens and uses new paired cases. Real execution costs are tested separately.

| Task, common reference | MPC correct / wrong | Cost-screened option correct / wrong | Activations | Different routes |
| --- | ---: | ---: | ---: | ---: |
| Daughter | 48 / 20 | 49 / 21 | 51/192 | 15/192 |
| Single trunk | 158 / 11 | 158 / 11 | 21/192 | 4/192 |

The simultaneous all-case intervals for sampled TCP distance, rotation and duration pass the ±2 mm, ±2° and ±.10 s margins in both tasks. Daughter adds one correct completion and one wrong confirmation. The exact conservative upper bound on harmful discordance is 2.87%, above the 2% gate. The triggered subsets alone do not establish equivalent cost. The sampled-motion-plus-measured-CPU ledger also fails time equivalence. The frozen comparison and this diagnostic each fail the efficacy gate.

[Cost diagnostic protocol](COST_MATCH_PROTOCOL.md) · [Paired tables and triggered subsets](data/cost_diagnostic/paired.json) · [Raw records](data/cost_diagnostic/cases.jsonl.xz)

## Mechanism checks

- In 128 independently generated calibration cases, the 256-atom opposite view reduced trunk displacement-sign Brier score by 0.0167 relative to a repeat. The paired interval is [−0.0307, −0.00475]. The daughter interval crosses zero. This measures the sign of displacement at material address 5. It does not measure which complete continuation is better. Joint sequence prediction scores better than independent marginals on these fixed sequences. The 768-atom check improves effective support and leaves the guarded policy inactive in its small development sensitivity. [Calibration](data/calibration_summary.json) · [768-atom sensitivity](data/sensitivity768/paired.json)
- The 16-case diagnostic repeats each legal program under eight acoustic tapes. A retrospective program selected using one half of the tapes gains one daughter completion in 64 other draws and zero trunk completions. This is a finite-library opportunity diagnostic using labels unavailable online. [Headroom](data/headroom/paired.json)
- A separate speed-derated graph uses the shortest of two actually checked 1.5 s and 2.5 s profiles under a shared rate cap. It exercises action-dependent arrival clocks. Guarded B still does not activate in the 12-case development sensitivity. [Timing sensitivity](data/timing_sensitivity/paired.json)

## Reproduce

Run from this study directory. Python 3.10, NumPy, SciPy and Numba suffice for paired analysis and simulation with the archived graph. Regenerating the dIRD and FK bank with `motion_bank.py` also requires the original robot model and dIRD checkpoint from the project runtime.

```bash
repo_root="$(git rev-parse --show-toplevel)"
replay_dir="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
xz -dc motion/BANK_MAIN.json.xz > "$replay_dir/BANK_MAIN.json"
mkdir "$replay_dir/frozen"
cp data/confirmation/settings.json "$replay_dir/frozen/"
xz -dc data/confirmation/cases.jsonl.xz > "$replay_dir/frozen/cases.jsonl"
python audit.py "$replay_dir/frozen" --motion "$replay_dir/BANK_MAIN.json"
diff -u data/confirmation/paired.json "$replay_dir/frozen/paired.json"
python selfcheck.py --repo "$repo_root" --motion "$replay_dir/BANK_MAIN.json"

python run.py --repo "$repo_root" --motion "$replay_dir/BANK_MAIN.json" \
  --out "$replay_dir/rerun" --split confirmation --n 192 --workers 16 \
  --model-size 256 --screen 24 --validate 48 --initial coarse anchor \
  --arms MPC B one fixed unguarded
```

The replay regenerates acoustic cases and policy decisions deterministically. Measured CPU times depend on the machine. Run `cost_match.py` with the decompressed frozen `plans.json` for the new cost-only cohort. `DATA_MANIFEST.json` gives hashes of committed and decompressed files.

**Evidence boundary.** This is a two-task simulation on one nominal atlas surface with internal motion. The policy uses quantized centroid observations. Physical contact, B-mode appearance, environment collision and a measured real-time deployment loop remain outside this evidence. The tested options and admissible graph do not establish a correct-completion advantage over replanning MPC.
