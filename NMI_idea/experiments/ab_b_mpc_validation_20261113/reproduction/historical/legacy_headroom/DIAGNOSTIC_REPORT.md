# Same-action headroom diagnostic

Command used

```text
python work/legacy_headroom.py --n 128
```

The script replays the pinned `cdb1705a02f60382ffbe9e42f2ed420e27a28ddb` moving-Y cohort with the public acoustic generator, the exact archived common-probe clock, and the action legality/stroke/time graph from `FROZEN_MANIFEST.json`. It enumerates every legal first-view/final-view pair in the existing action library. The archived online-B result is checked against the replayed pair for every case and reserve.

The oracle is an after-the-fact evaluator-label selector. It is diagnostic only. It receives no role in the historical B or MPC decisions. Robot FK, collision, sampled travel, and turn are not reconstructed because the optional capability bridge is unavailable. The ceiling is therefore optimistic with respect to the frozen preflight graph.

| reserve | archived B strict | strict ceiling over legal pairs | strict headroom cases | target-visible ceiling | cases with any wrong pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| narrow | 76/128 | 100/128 | 24 | 124/128 | 40 |
| wide | 76/128 | 102/128 | 26 | 124/128 | 40 |

The baseline selected the centered first view in all 128 cases. Of the 24 narrow and 26 wide cases where some enumerated pair succeeds while archived B fails, 23 and 26 respectively have a successful pair with a different first view. This indicates search/action support and decision selection remain plausible bottlenecks. It does not establish that an admissible B can recover the oracle ceiling, because the oracle uses the terminal labels after enumerating all outcomes.

The 103/128 target-visible count for the archived terminal image is lower than the 124/128 action-support ceiling. Among archived failures, 48 cases in each reserve have at least one legal pair with a target-visible terminal image, while 24 narrow and 26 wide have at least one pair that also passes the same confidence/identity gate and continuation gate under the public belief update.

The full per-case and per-pair records are in `HEADROOM_128.json`; the compact values are in `SUMMARY_128.json`.
