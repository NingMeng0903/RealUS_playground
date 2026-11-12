# Independent review and supervision

## ASTRA MAX

ASTRA redesigned the target as the expected paired change in correct completion after a checked continuation and the unchanged full MPC suffix. It distinguished shared hard caps, per-state predicted cost matching and cohort-average cost allocation. It reviewed candidate completeness, mechanical witnesses, independent ranking and evaluation tapes, and the interpretation of the same-tape hindsight ceiling. ASTRA supported a single controlled expansion to a two-command status-contingent prefix. In this library the second endpoint is specified per candidate, and public progress decides whether that command is executed. ASTRA supported stopping expansion after the second opportunity test.

ASTRA verified the freeze and the recorded source hashes. It checked that the learned differences use public features and offline paired terminal labels. The public features compress the posterior. The second-command emission descriptors use marginal predictions from the first decision state. The dual price half margins are a fitting target. The stored deterministic development choice exceeds the half-margin target for at least one metric in three of four conditions. The actual prospective cost comparison determines equivalence.

The final theory review finds no support for the prespecified matched-cost contextual B claim. The small positive fixed-route result is a secondary prospective finding. All 13 completion discordances in that comparison arise when direct MPC initially remains at the previously accepted station. The fixed rule advances immediately to the final station and uses the signed tilt only if unresolved. The result identifies a scheduling weakness of the shared finite-horizon MPC in this simulated task. It does not establish that context prediction, opposing-view evidence or alternate robot realization caused the gain.

## SOL MAX

SOL independently audited prospective admissibility, paired disturbance reuse, outcome accounting, the shared MPC suffix, and the split between frozen predictors and evaluator labels. It independently regenerated all 24 pilot public states, verified 97 concrete witnesses, replayed 484 paired terminal paths, regenerated 1,970 packets, and checked 288 traces against the prior MPC and one-command simulator. In the expanded diagnostic it audited the first four latent cases, both tasks and checkpoints, with eight tapes each. That check covered 447 concrete witnesses, 3,704 candidate/tape paths, 15,182 packets and 384 archived trace comparisons. It verified both the second-command branch and immediate handoff after a public claim.

SOL independently recomputed the single and expanded repeat-based conclusions from the raw records and recommended stopping candidate expansion. It verified the frozen ridge normal equations for 21,438 supervised state-action rows from 192 independent cases, the development dual objectives and decisions, and the disjoint case seeds. It distinguished exact per-state timing matching on the 1.5 s/12-tape grid from equivalence of actual sampled cohort costs.

In the final cohort, SOL verified the frozen source and input hashes, all 2,048 public state records, all serialized scores and deployed choices, all 10,240 arm paths and their 49,817 logged edges, and all 20 paired contrasts. It independently checked fresh completion, wrong-return subtypes, actual resources, exact harmful-discordance bounds and simultaneous cost intervals. A separate acoustic and controller replay used 20 spread-out latent indices, covering 80 public states, 2,235 geometric witnesses, 1,671 independently generated packets, 400 complete arm trajectories and 259 archived MPC or one-command parity comparisons. It checked the handoff and terminal state hashes.

Run the two independent checks from the study directory after extracting the prior motion bank.

```bash
repo_root="$(git rev-parse --show-toplevel)"
check_dir="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
xz -dc "$repo_root/NMI_idea/experiments/ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz" > "$check_dir/BANK_MAIN.json"
python independent/audit_summary.py --study "$PWD" --motion "$check_dir/BANK_MAIN.json"
python independent/audit_replay.py confirmation --study "$PWD" --repo "$repo_root" --motion "$check_dir/BANK_MAIN.json" \
  --first-cases 12 --extra-cases 63 127 191 255 319 383 447 511
```
