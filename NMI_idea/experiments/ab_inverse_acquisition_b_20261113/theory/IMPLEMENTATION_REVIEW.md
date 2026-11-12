# Theory review of the bounded implementation

Updated before the scientific screen on 2026-11-13. The current design is suitable for the bounded opportunity screen, subject to the independent numerical/mechanical audits. The earlier feasibility and labeling objections listed below have been resolved. This review does not certify execution or a positive result. The original-bank API smoke is a functional check, not scientific evidence. No shared implementation files were edited by this reviewer.

Reviewed source hashes:

- `bench.py`: `f818df577248f3e168459196d9029782c1c79a65d47600e3da9710f2f75ddbce`
- `PROTOCOL.md`: `7c9b91cc730f23ee864d40d478062798e1cb6f85b1b07304edb1b83dbc6e9ebb`

Later code changes require a corresponding audit update rather than assuming this note covers them.

## Aligned with the design

- `world.py` restricts all new angles to the legacy `+/-12` degree envelope. The mechanically generated set is `+/-3`, `+/-6`, and `+/-9` degrees.
- The same public anchor determines a frozen candidate menu for all paired arms and their entire MPC tails.
- `identity_utility` fixes the identity threshold using the public posterior, then evaluates model-particle utilities. It does not grant each model particle a clairvoyant identity rule.
- The decision-information score is the exact finite-observation expected value of information for those fixed particle utilities. It is not an entropy score.
- The code separately identifies the original old graph and a coarse graph with legacy nodes but newly checked old-to-old connections. That separates graph-connectivity gains from new acoustic poses.
- The first forced action is restricted to the next required site. All later actions use the same shared replanning MPC with the real exit state.
- `MENU_CACHE[(task, anchor)]` is valid for this initial design because the full public anchor posterior and start state are determined by those two values. It would be invalid after introducing richer prefixes unless the cache key were expanded.

## Resolved review points

| Earlier point | Current resolution |
| --- | --- |
| Some finite incoming/outgoing edge is weaker than a full-task witness | `route_through` now supplies a sufficient joint-cap complete-task witness under ideal acceptance; admission requires one for a realization of the proposed view |
| A witness in the dense bank might use an angle later removed from the menu | Every proposal is certified using only the shared coarse nodes plus that proposed site/view; the final menu is a superset of this certificate graph, so the witness survives menu selection |
| `J` included reference decisions without checking their reachable motions | Reference views are admitted only when an explicit reached prefix and reference edge fit the joint remaining cap from at least one certified realization |
| The score could be described as completion value despite using `p_c^C-p_c^W` | The protocol explicitly calls it an identity-decision proposal surrogate; complete-scan efficacy comes from the full rollout |
| A one-sided cost screen could be mistaken for strict equivalence | The protocol names the ranking screen as upper-only and separately defines realized no-worse and strict two-sided equivalence |
| Decision-aware proposals might merely gain from more angles | Claim-only and fixed-angle controls now match the number of selected new angles at each site |
| Dense graph repair might be credited to inverse sensing | Original coarse-graph and legacy-node expanded-edge controls are both present |

The final forced-action library additionally checks a full witness in the actual masked menu before a continuation enters the census. The inverse and control menus retain all legacy nodes. Missing witness results from the bounded certificate search are not proofs of global infeasibility; the code documents this distinction.

## Remaining scope limits, not unresolved implementation defects

1. `U_c=p_c^C-p_c^W` is a frozen-rule signed identity utility. It is not a full adaptive suffix value. The final physical action, evidence update, wrong identity, and completion path are assessed separately by full episode execution.
2. Reference availability is existential across the two possible realizations of a proposed acoustic view. The reference union used in its acoustic score need not be available from one particular realized joint configuration. This is another reason to treat `J` as a generator surrogate, while the actual forced action and shared MPC enforce their own realized edges and resources.
3. Smooth interpolation of transported ghost means between different legacy regimes is a declared view-conditioned synthetic model. It does not necessarily describe one rigid world-fixed ghost. Exact legacy parity and view-ID invariance do not independently validate physical ultrasound. The protocol requires a fixed-ghost-regime sensitivity before a positive gain is given a physical interpretation.
4. The opportunity census covers task-preserving first commands followed by the same MPC. Its result is about this finite policy class. A failed screen is not proof of zero population headroom over all policies.
5. The no-worse screen allows an arm to be faster or travel less. A pass is not strict two-sided equivalence unless the separately reported intervals also meet that definition. Compute time is charged in a ledger but does not advance the simulated acoustic clock.
6. Thirty-two independent cases cannot certify the historical 2% harmful-discordance criterion. Repeated tapes reduce noise about a case, not uncertainty about the population of cases.

## A prediction issue that the opportunity result cannot settle

At the anchor checkpoint, every case with the same task and anchor packet has the same public posterior and proposal menu. A per-latent-case split-repeat selector nevertheless ranks using repeated private-case outcomes. It may find a useful action for that hidden case that no online public policy can identify. If opportunity is positive, train a public-packet action-value table on disjoint cases and validate out of case before considering a richer learner. Archive the full-denominator performance and incumbent fallback for unseen packets.

## Attribution required if an effect appears

Use original MPC, legacy-node/expanded-edge MPC, inverse-menu MPC, a matched-cardinality fixed menu or direct-claim-only menu, and the selected continuation on inverse-menu MPC. These contrasts distinguish connectivity, action refinement, decision-information synthesis, and continuation selection. A gain in one contrast must not be attributed to a different mechanism.

## Outcome verdict

The frozen protocol uses 32 new cases per task and 24 suffix tapes per case. Its predeclared daughter screen requires at least a three-point split-repeat completion improvement above inverse-menu MPC, a positive case-paired lower confidence limit, no net wrong-identity increase, realized no-worse cost limits, and improvement over the fixed progress control. Trunk results determine whether a generality claim is supported; a daughter-only positive result does not establish a cross-task contribution.

The screen is now complete and fails that criterion. The cost-screened completion differences are +0.91 points for daughter and +0.39 for trunk, with intervals crossing zero. Neither the continuation comparison against the fixed progress control nor the generator comparison against coarse-diagonal MPC supplies the missing effect. The larger daughter hard-cap gain fails all three simultaneous upper cost margins. Stop the tilt continuation predictor; do not open its reserved training/development/confirmation cohorts. See `OUTCOME_VERDICT.md` for the active-use counts, interpretation, and prior-only aperture result. Any pivot must change and validate the measurement mechanism rather than retune the selector on this screen.
