# Oracle CEM headroom diagnostic

`oracle_cem.py` is a small, auditable diagnostic built on the frozen
`ab_recovery_training_20261115` `World`/`Controller` and its
`results/prodmp_weights.npz`.  It does not modify the benchmark world or the
existing pilot.

## What is compared

Each case runs the public prefix until the first `State.trigger()` checkpoint.
The trigger is selected from public state/belief only (`missed`, ambiguity,
and the controller's public trigger); no role, target, transition coordinate,
or evaluator packet is used to choose it.  The checkpoint and prefix are
stored without `roles_truth` or `target_truth`.

The intervention parameter is

`[segment0_delta_s, segment0_delta_x, segment1_delta_s, segment1_delta_x]`.

Each segment is one second and is generated through the frozen ProDMP
`Controller.candidate`/`Controller.execute` interface.  `delta_s` permits a
two-block continuous revisit, while `delta_x` reaches the frozen single-block
side limit.  Every packet reconditions the segment at the existing 10 Hz
rate.  If public identity/centering is already acceptable after segment one,
the second segment is skipped.  The program then hands off to
`Controller.finish(... allow_recover=True, informative=True)`, exactly the
same informative feedback MPC used by the comparator.

The comparator is named explicitly in every result:
`complete_mpc_informative_true`.  It starts from the same checkpoint and runs
the same informative MPC tail on the same future-noise replica.

The full-MPC result is also an explicit CEM incumbent.  A rejected candidate
may otherwise hand control straight to the MPC tail and appear successful;
such candidates are retained and marked `valid_program=false`, but cannot be
selected as positive demonstrations.

## Two meanings of “oracle”

* **World-oracle CEM (implemented):** each hidden scene is scored with the
  evaluator, and a different theta can be selected for each scene.  This is an
  offline privileged upper bound/headroom diagnostic only.  It cannot be
  called a deployable B policy and must not be used as a truth-labelled actor.
* **Public-history rerank hook (included):** `public_score()` gives a
  single-step ranking of saved candidates from the same public checkpoint
  using only public belief, forecast and information score.  It is **not** a
  complete public-history teacher: a real teacher must share the same first
  action for identical public histories and branch only on later public
  packets.  This hook is not silently substituted for the world-oracle
  result.

Because CEM has continuous two-segment ProDMP freedom while the old baseline
has a finite candidate list, this result is also an **expanded-action/search
joint upper bound**.  A decision-level claim requires giving MPC the same
continuous candidate library and comparable search budget, then comparing
public-history policies.

## Stored evidence

`oracle_cem_cases.jsonl.gz` keeps every sampled candidate and every search
replica (including failures), requested and effective/clipped actions, costs,
strict completion, and wrong outcomes. The public checkpoint/history is stored
once per case. Full public motion/packets and revisions are saved for the final
selected program and paired MPC verification. Each candidate has a rounded
requested-parameter duplicate key; physically identical executed paths may
still appear as separate candidates, so candidate count is not a count of
distinct trajectories. The CEM search uses future-noise replicas
created by `World.resample_tail(scene, checkpoint_frame, replica)`; static
anatomy and the prefix remain paired.

The terminal utility is task-first (`strict_success`, then `ever_wrong`, then
TCP/time costs).  `strict_success` is the frozen `Record.details` definition.
It permanently treats any prior wrong acceptance as a failure.  A later
correction after a wrong submission needs a separate public evidence/retract
ledger, which this diagnostic does not implement and therefore does not claim
to test.  Each case therefore records `prefix_wrong` and
`headroom_eligible`; strict headroom summaries must be stratified or restricted
to cases with no earlier wrong acceptance.

## Smoke command

```bash
python -B NMI_idea/experiments/ab_search_embedding_a_20261117/oracle_cem.py \
  --cases 1 --population 4 --iterations 1 --search-replicas 1 \
  --verify-replicas 2 --out /tmp/oracle_cem_smoke
```

The full run is controlled by `--cases`, `--population`, `--iterations`,
`--search-replicas`, and `--verify-replicas`.  Runtime scales approximately
with cases × four modes × population × iterations × search replicas, plus
verification replicas; the smoke run is about one second on the development
machine, while a 4×16×3×2 run is expected to be on the order of a minute.
