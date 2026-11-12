# Independent continuation audit

The confirmation freeze digest is `a463587e33ca83e049eb0c231708181923921d8ef18a84d0b50ac198899cefd5`. The audit verified the frozen source, model and diagnostic input digests against the files on disk. No code or fitted coefficients in the frozen experiment changed during the audit.

## Reproduction

Use the commands in [Independent review](../INDEPENDENT_REVIEW.md).

The all-row check covered 512 independent latent cases, 2,048 public states, 10,240 policy paths and 49,817 logged edges. It recomputed the frozen forecasts and selections, identity endpoints, wrong-return subtype counts, resource use, exact discordance statistics, and paired intervals for 20 contrasts.

The independent sampled replay used case indices 0 through 11 and 63, 127, 191, 255, 319, 383, 447 and 511. It reconstructed 80 public states, 2,235 joint mechanical witnesses, 1,671 packets, and 400 terminal paths with their handoff and final public hashes. It reproduced 259 complete archived MPC or one-command continuation traces. Among the replayed pair policies, 107 executed a second command on the unresolved branch and 26 handed off immediately after a public claim.

The initial pilot replay reconstructed 24 public states, 97 joint witnesses, 1,970 packets, 484 terminal paths, and 288 complete archived comparisons. The feedback diagnostic replay sampled the first four latent cases and first eight tapes. It covered 16 public states, 447 joint witnesses, 15,182 packets, 3,704 terminal paths, and 384 complete archived comparisons. The fitted ridge coefficients satisfied the normal equations on all 21,438 weighted state/action rows from 192 independent training cases. Recalculation of the deterministic cost-price decisions and development forecast errors required no refit.

## Interpretation

The cohort-priced B was active in all four conditions. Its strongest contrast was 10 extra correct daughter completions after two shared MPC commands. The +1.95 percentage-point gain had a paired interval of +0.20 to +3.71 points. Six B-only wrong-confirmation cases gave a 2.53% upper bound on harmful discordance. The local cost selector met sampled motion equivalence in all four conditions and showed no resolved completion gain.

The training-selected fixed daughter continuation `0>4` was mechanically available in 95 of 512 cases after two shared commands, each with one prior public claim. It tried site 3 at center and used the +12° view only if the center capture gave no claim. Its 12 completion wins and one loss gave +2.15 percentage points, with a paired interval of +0.78 to +3.52 points. Two fixed-only wrong-confirmation cases gave a 1.40% harmful discordance upper bound. Its simultaneous motion intervals met the three noninferiority margins. The elapsed interval was entirely faster, from −0.132 to −0.015 seconds, and extended below the lower equivalence margin. This is a secondary fixed-route finding below the five-point efficacy target. Cohort-priced B gained one fewer completion than the fixed policy, with an interval from −1.17 to +0.78 points for the difference. The confirmation does not establish added value from the learned public-history ranking.
