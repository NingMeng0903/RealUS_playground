# All-return association and continuation B on shared MPC

This development study follows the [continuation-specific B benchmark](../ab_continuation_b_20261113/README.md). It tests a shared observation upgrade and then a B selector built on the upgraded baseline. The selector changes actual routes in the held-out cases. Neither the existing-frame B nor the paid-probe diagnostic meets the frozen continuation screen.

[Protocol](PROTOCOL.md) · [Freeze](FREEZE.json) · [Independent review](INDEPENDENT_REVIEW.md) · [Archive hashes](DATA_MANIFEST.json)

## Shared observation and controller

The original world draws up to three fine-grid returns. The new interface preserves the unordered multiset after coarsening each return, including counts and coincident bins. The matched nearest interface sees the same packet projected to a nearest bin and a multiple-return flag. The legacy control keeps the original 7×7 nearest packet and archived controller.

The exact finite kernel marginalizes labeled assignments of the visible and absent returns. Each indistinguishable slot gets an equal share of target mass for its bin. For observed packet \(o\), target identity is chosen using \(j^*=\arg\max_j b^\top C_j(o)\). The association and the 0.87 claim threshold use the public posterior, and private kind enters the evaluator only. A claim updates prior-correctness mass with \(C_{j^*}\). No separate likelihood multiplies the packet's deterministic relative-position features.

The all-return planner retains the archived two-capture horizon, motion costs, wrong-identity penalty, terminal potential and replan-after-packet behavior. It adapts the predictive observation and claim terms to the new alphabet. The same ten first-packet quantiles partition each alphabet in its own canonical order. The fixed-path comparison below isolates association without a planning change. All policies use the same checked motion graph and resource caps.

## Sensor comparison and fixed-path identity

Each row shows correct complete scans / cases with a wrong identity. Cases are paired across interfaces. The sensitivity cohort uses different latent seeds from the primary cohort.

| Task and grid | Legacy 7×7 nearest | Same-grid nearest | All-return MPC |
| --- | ---: | ---: | ---: |
| Daughter, 3×3, 96 cases | 16 / 17 | 0 / 0 | 24 / 12 |
| Single trunk, 3×3, 96 cases | 81 / 3 | 82 / 5 | 79 / 6 |
| Daughter, 5×3, 32 cases | 13 / 3 | 3 / 0 | 9 / 5 |
| Single trunk, 5×3, 32 cases | 27 / 1 | 26 / 1 | 28 / 0 |

The primary daughter contrast against legacy is +8/96, with 16 wins and 8 losses and a paired 95% interval of −1.04 to +18.75 points. The finer daughter sensitivity gives −4/32. The coarsened nearest daughter control often abstains at the shared identity threshold, so its +24/96 contrast measures recovery from that specific compression. Measured motion-equivalence intervals and wrong-identity discordances are available for every pair of interfaces.

On the exact same legacy trajectories and clocks, the primary daughter results are 52 correct and 25 wrong accepted frames with legacy versus 59 and 18 with all returns. Fourteen all-return correct accepts occur at frames where the legacy nearest selector picked a different return. The fixed-path comparison concerns target-eligible frames on the reference path. Those accepts are not counts of completed cases. In the 5×3 sensitivity, daughter accepted frames change from 33 correct and 3 wrong to 38 and 6. The result supports the mechanism that a discarded target return can be recovered. Its gain over legacy depends on the representation and case distribution.

[Primary paired sensor comparison](data/sensor/summary.json) · [5×3 sensitivity](data/sensitivity/summary.json) · [Identical-path identity audit](data/fixed_path_sensor/summary.json)

## B on the all-return baseline

After two commands of common all-return MPC, B can use the latest packet to select direct MPC, a checked centered next-site acquisition, or a checked tilted next-site acquisition. A forced command is followed by the exact same all-return MPC with the real exit belief, clock, robot node, task progress and remaining resources. A separate paid diagnostic executes one checked same-site change of view or a hold, then scores continuations at the actual exit. A progress advance at that probe hands off immediately.

The two public rankers predict paired terminal completion differences. One sees the projection of the same packet. The other adds the multiset, relative geometry and set-conditioned summaries. Both get common pre-packet belief and robot/task state. Evidence-by-action terms allow their ranking of alternative routes to change. The downstream controller always sees the complete packet in every arm. An incumbent score of zero and a training-only fixed route provide comparisons. All coefficients and source hashes were frozen after 64 training cases and before the 64 new cases. Both tasks reuse the same latent case indices. Each selected route also runs on a fresh tape outside the 16 repeat tapes used for opportunity evaluation.

| Held-out task and decision | Paired no-probe all-return MPC | Relation B | Nearest-input B at same exit | Fixed route at same exit |
| --- | ---: | ---: | ---: | ---: |
| Daughter, existing frame | 13 / 6 | 12 / 3 | 13 / 3 | 14 / 5 |
| Single trunk, existing frame | 56 / 3 | 57 / 2 | 57 / 2 | 56 / 3 |
| Daughter, one paid probe | 15 / 1 | 15 / 6 | 15 / 6 | 14 / 5 |
| Single trunk, one paid probe | 51 / 7 | 46 / 8 | 46 / 8 | 46 / 8 |

Existing-frame daughter B selects 28 non-MPC continuations and disagrees with the nearest-input scorer on nine cases. Its repeat-averaged completion difference is −0.88 points versus direct MPC and −1.46 points versus the nearest-input scorer. The latter paired interval is [−3.03, −0.29]. The trunk scorers choose identically on all 64 cases, including the five overrides. That trunk gain is shared by both evidence representations.

At the daughter probe exit, B gains +1.17 points over probe-only MPC in the repeat comparison, with a paired interval of [−1.27, +3.81], and −0.39 points against the same-packet nearest scorer. The complete paid intervention has −2.05 points repeat-average gain over matched direct MPC. On the fresh tape, completion ties and there are five B-only wrong confirmations. Mean additional sampled TCP distance, rotation and time are 9.51 mm, 14.27 degrees and 0.35 s. The paid trunk branch has no alternative beyond MPC at the exit under this library, and its complete paid intervention has −11.33 points repeat-average gain. The same-tape paid comparison against free relational selection also favors the free intervention in both tasks by point estimate.

Split-repeat opportunity after the existing frame is −0.20 points for daughter and +0.20 for trunk. At the paid exit it is +1.86 and 0.00. These label-assisted comparisons choose on one noise half and evaluate on the other, then swap. They measure achieved selection under finite repeats. The daughter paid-exit interval reaches zero. None justifies developing an active probe trigger in this pilot.

[All held-out contrasts](data/heldout/summary.json) · [Held-out public scores, paired tapes and executed routes](data/heldout/cases.jsonl.xz)

## Reproduce

Run from this directory with Python 3.10, NumPy, SciPy and Numba. The legacy files and motion bank are verified by hash. The relative-position evidence is coarsely quantized. These simulations use one nominal anatomical chart and checked sampled robot motion. CPU is logged separately from the acoustic clock. The pilot tests identity and route relevance with synthetic returns. It provides neither real-time nor tissue-contact evidence, and 64 cases cannot certify the previous 2% harmful-discordance bound.

```bash
repo_root="$(git rev-parse --show-toplevel)"
replay="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
xz -dc "$repo_root/NMI_idea/experiments/ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz" > "$replay/BANK_MAIN.json"
python bench.py --repo "$repo_root" --motion "$replay/BANK_MAIN.json" --out "$replay/sensor" --split sensor --n 96
python fixed_path.py --repo "$repo_root" --motion "$replay/BANK_MAIN.json" --sensor data/sensor --out "$replay/fixed"
python route_b.py --repo "$repo_root" --motion "$replay/BANK_MAIN.json" --out "$replay/train" --split train --n 64
python route_b.py --repo "$repo_root" --motion "$replay/BANK_MAIN.json" --out "$replay/heldout" --split heldout --n 64 --freeze FREEZE.json
```

The 5×3 sensitivity uses `bench.py --split sensitivity --grid 5 --n 32` with the same repository and bank. Stored timing measurements depend on the runtime. The chosen routes, observations and non-CPU outcomes are deterministic under the archived tapes and implementation.
