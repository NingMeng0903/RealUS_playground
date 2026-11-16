# Stage-A search data: corrected experiment and interim review

Subsequent experiment: [public-history teacher, revisable evidence, and fresh paired results](ADAPTIVE_RESULTS.md).

Date: 2026-11-17. Status: **development diagnostic, not a learned-B result**.

## Why the earlier pilot was stopped

The 1.5/2.5 mm out-and-back sweeps in `pilot.py` did not enact the intended
exploration teacher. The inherited image field spans 50 mm laterally. The old
ProDMP data filter in `ab_recovery_training_20261115/run.py:shape_examples`
accepted a segment after one locally correct packet, without requiring a
correctly completed continuation. Thus neither the sweep pilot nor the old
motion fit demonstrates recovery from true loss or identity confusion.

The original V9.4 draft already requested student-induced failure states,
multistart constrained teachers, multimodal paths, and complete rollout
assessment. The corrected `PROTOCOL.md` restores that objective.

## Mechanism checked first

The inherited two-vessel generator rotates the relationship of the two tracks
through a central transition. Outside that region, the two structural models
can again give the same **unordered** two-point image while requiring different
identity assignments. For example, at nominal chart station 24 mm the two
models' unordered predicted centroids coincide, while the requested target
centroids differ by about 19.3 mm. A downstream lateral image alone cannot
resolve the connection history. A revisit to the transition region is a
plausible diagnostic program, but must earn back its execution cost.

`revisit_gate.py` first tested a cheap fixed one/two-block revisit from the
same public recovery checkpoint, each followed by the same 10 Hz feedback
ProDMP and informative MPC. In 32 training worlds, 21 were eligible. With
MPC included as incumbent, independent suffix replicas gave success
`131/168` for MPC and `127/168` for the selected fixed-revisit arm; wrong
acceptances were zero, and mean TCP distance was 43.65 versus 43.98 mm.
This is a negative screen for the **fixed revisit**, not a proof that no useful
continuous program exists. Eight suffix replicas per world are correlated
repeats, not 168 independent anatomies.

## Continuous privileged CEM screen

`oracle_cem.py` searched two one-second continuous relative ProDMP blocks:
`Δs ∈ [-6,7.8] mm`, `Δx ∈ [-8.5,8.5] mm`. Each candidate used the frozen
kinematic gates, 10 Hz public image updates and within-block reconditioning.
A public identity/centering condition could hand off after block one. Every
candidate then ran the same complete informative MPC tail. Search used two
future-noise replicas, and verification used eight fresh paired replicas.
Four CEM modes, population eight, two iterations, and the original MPC
incumbent were used. All 16 independent worlds were selected by a public
recovery trigger; none had a prior wrong acceptance. The world truth was used
only by the **offline candidate selector and evaluator**, not by the executed
filter, servo or MPC.

| Development outcome | Full MPC | World-oracle CEM + same MPC tail |
|---|---:|---:|
| Strict correct completion | 102/128 | 110/128 |
| Worlds with better / worse / tied paired success | — | 7 / 2 / 7 |
| Wrong acceptance | 0/128 | 0/128 |
| Mean executed TCP distance | 45.63 mm | 42.58 mm |
| Mean executed turn | 7.67° | 7.07° |
| Mean simulated duration | 6.30 s | 6.52 s |

The 128 suffix rollouts represent 16 independent worlds. The apparent
+6.25-percentage-point completion difference is **not** a performance claim
for B: a different program was selected after inspecting each hidden world;
CEM also had a broader continuous action family than the old MPC, and the
mean duration increased by 0.22 s rather than meeting the old ±0.10 s
cost-equivalence starting margin. Only these two-vessel synthetic centroid
scenes were used. The sample is small and this is development data.

For 14 worlds in which CEM selected a program, a post-hoc first-block-only
ablation on the same eight verification replicas gave 90/112 MPC,
91/112 first-block-only, and 98/112 full selected program completions. This
suggests some benefit from a second observation/decision, but it was
not separately optimized or tested on new worlds; the paired 14-world
bootstrap interval for the full-minus-first success difference crosses zero.

## What the data can teach and what remains untested

An offline true-world CEM trajectory may teach **feasible motion shape** and
supply a privileged value target. It cannot be copied as a unique first-action
label when two hidden worlds share the same public history and require
opposite directions. The next gate is a public-history teacher: compatible
worlds share one root action, later action choices branch only on simulated
public images, and value is averaged over hidden worlds. The same continuous
candidate library must be offered to an MPC comparator before assigning any
increment to B's decision mechanism.

Only full correct completion with no uncorrected wrong acceptance and valid
executed commands should qualify as a positive trajectory demonstration.
Store other branches, including local reappearance followed by final failure,
for value and stop models. Preserve different successful modes. Then let the
current student generate its own error states and recollect corrections;
training on fixed small offsets misses this distribution. This follows the
mechanisms of [DART](https://proceedings.mlr.press/v78/laskey17a.html),
[DAgger](https://proceedings.mlr.press/v15/ross11a.html), and
[IntervenGen](https://ieeexplore.ieee.org/abstract/document/10801523).
Continuous candidate search itself follows established
[iCEM](https://proceedings.mlr.press/v155/pinneri21a.html), while offline
clairvoyant data for a public policy has a precedent in
[Adaptive Information Gathering via Imitation Learning](https://arxiv.org/abs/1705.07834).

The current `Controller` permanently credits the first accepted packet in a
coverage bin and permanently fails strict success after an earlier wrong
acceptance. Therefore **correction after a confident wrong association is not
yet a valid test**. It needs a public-evidence-based retract/reverify ledger,
with historical wrong events and final uncorrected wrong delivery both
reported. The present synthetic `World` also needs a broader, physically
motivated family of persistent occlusions, similar neighboring structures,
registration error, and recoverable versus unrecoverable windows. Those
changes belong in a separate mechanism test and must be shared with all
comparators. No new ProDMP weights or deployable B were trained in this run.

## Reproduction

```bash
python -B NMI_idea/experiments/ab_search_embedding_a_20261117/revisit_gate.py \
  --cases 32 --search-replicas 2 --verify-replicas 8 \
  --out NMI_idea/experiments/ab_search_embedding_a_20261117/results/revisit_gate_train32_incumbent.json
python -B NMI_idea/experiments/ab_search_embedding_a_20261117/oracle_cem.py \
  --cases 16 --population 8 --iterations 2 --search-replicas 2 \
  --verify-replicas 8 \
  --out NMI_idea/experiments/ab_search_embedding_a_20261117/results/oracle_cem_dev16
```

Frozen inputs: `ab_recovery_training_20261115/world.py` SHA-256
`ff50b8fc2f865c7a58f4271b9f6595747f04a610524dcb0f4e3aede87c739dcb`,
`controller.py` SHA-256
`513febd2d93e7aec35c7f563872f9ac7f621fe4730257426807c3ddb7b4ff4d6`,
`results/prodmp_weights.npz` SHA-256
`5c1c4cb17f46b87ebe1f2773292d1bd446f7e177b7d0323e492ae667d5e62b4c`.
The raw per-candidate outcomes and final selected/MPC public trajectories are
in `results/oracle_cem_dev16/oracle_cem_cases.jsonl.gz`; its `summary.json`
and first-block ablation are alongside it.
