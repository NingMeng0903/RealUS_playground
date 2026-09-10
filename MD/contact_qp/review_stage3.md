# Stage 3 independent review — interim 2026-09-10

**Decision: stage 3 pending; energy component PASS.** The five energy findings below are now closed; the reviewer independently reran all **11 energy tests, passed in 0.04 s**. Execution coordination, native generic constraints and the production final-send chain are not yet covered by a stage-3 PASS. No implementation files or hardware were modified by this reviewer.

Five concrete findings were reported and reproduced:

1. **Closed — untrusted boolean flags:** `PortBounds(verified="false")` initially yielded `certification_valid=True`; string false values also bypassed interval validity/alignment. Dataclass and ledger method checks now reject such values, including `reserve(stopping=...)` and `mark_no_send(old_exposure_proven_absent=...)`.
2. **Invalid settlement clock:** `settle(..., now_s=NaN)` initially accepted a future negative-work interval and credited it immediately. Current source now rejects a nonfinite clock before changing the ledger.
3. **Nonfinite derived arithmetic:** finite but overflowing wrench/velocity products initially produced NaN work. Python's `min(capacity, balance-NaN)` then changed a 0.5 J balance to the 1 J capacity, released its reservation and retained certification. Current source checks derived power/work/balance before mutation and preserves liability while invalidating certification on overflow. Reservation work and accumulated reservation sums also require finite arithmetic.
4. **Closed — missing asynchronous stop combinations:** the original five branches included both-device stopping but omitted one device stopped while the other retains an old/new command. With x velocities old arm +2, old rail -2, new arm +1, new rail -1 and both stopped 0, the original helper reserved 1 W under environment force -1 N, but rail-first stopping exposes 2 W. Current helper constructs the old/new/stop Cartesian product using per-device stopping components. The cancellation-disappears regression passes, all returned branches are sampled in the random bound test, and a single supplied stopping component is rejected.
5. **Closed — repeated short physical intervals:** the 1e-12 overlap tolerance initially allowed exactly the same interval shorter than 1e-12 to settle repeatedly. For `[1,1+5e-13]` and constant returned power 1e12 W, the duplicate incorrectly credited another 0.50004445 J. Strict overlap checking now rejects it and the new regression passes.

Positive component checks: there is one settled balance; earmarks do not recharge it. Summing each segment's positive upper work conservatively dominates every prefix without borrowing future recovery. Ordinary reservation preserves the fixed stopping reserve. Reject/no-send retains old or mixed exposure unless the caller supplies the required proof. Missing or unaligned measurements leave liability outstanding. Overlapping reservations release their matching earmarks in the same successful settlement that debits the physical work once. Actual overdraw remains negative and loses certification rather than being lower-clipped. Full six-dimensional power uses paired environment-on-tool wrench and velocity at the same declared port; the implementation currently claims no dissipation credit.

The execution adapter still must guarantee all of the following:

- Construct conservative, verified full-horizon branches covering old holds, arm/rail mixtures, intermediate deceleration, independent device stopping, in-flight I/O and late command overtaking; a stopping endpoint or a prediction alone is insufficient. Validate the stopping reserve against that exposure, and continue a necessary mechanical stop even if it must be recorded as unfunded debt.
- Serialize ledger mutations with publication facts. Reserve before the first possible device exposure; bind identity, epoch and expiry to the actual candidate; make no-send release only from authoritative per-device facts. A function named `mark_no_send` cannot itself establish that physical proof.
- Keep old/unknown sends fenced after stop and retain liability until trusted interval closure or replacement/stop evidence. Elapsed reservation horizon alone does not prove an in-flight send cannot still execute; violated bounds lose certification and block ordinary recovery.
- Settle only the full measured physical port with matched timestamps, calibration, reference point, validity and error bounds. Commanded/predicted motion or delayed ultrasound force cannot substitute. Missing intervals remain liabilities and cannot be silently bridged or settled with zeros.
- Treat `PortBounds.verified`/ledger validity as only one part of certification. Real calibration, sensor alignment, finite execution latency, stopping and error bounds must also be verified; otherwise hardware stays MONITOR/UNVERIFIED. A synthetic ledger PASS cannot turn those missing declarations into a hardware certificate.

Reviewed energy SHA-256:

```text
8659aed98c89c9606e05441f00c4f92e769d68b70dc6e505abfa9c6436ec3d21  peirastic/contact_qp/energy.py
12677afa95b53821755c058c5bf86223f268f92cef508ae8e22f967641b27b25  peirastic/tests/test_contact_qp_energy.py
```

Execution fault injection and native/final-send-chain evidence remain pending. No held-out acceptance data or hardware was used.

## Execution coordinator follow-up

Five initial execution blockers were reproduced and corrected by the owner: mutable reviewed command/certificate/progress, commit after ledger certification loss, an older reviewed candidate publishing/committing after a newer one, commit before recorded device-event time, and publication starting after the declared publication-delay bound. Current source uses a frozen publication descriptor with read-only public facts, one outstanding ordinary candidate, per-device/time/sequence/expiry checks, and final reservation/certification checks. A zero full TCP velocity now also requires explicit per-device stopped evidence before reconciliation. Full-horizon absolute velocity-box corners cover historical in-flight/mixed/stop states rather than reserving only the latest nominal branches.

One additional recovery case was reported: `review(p1) → stop → trusted stopped reconciliation → review(p2)` must not remain blocked forever by the retired but un-aborted p1. Stop should logically retire such candidates while preserving their physical facts and liabilities. The owner is addressing this case; execution tests and final hashes remain pending reviewer verification.

## Native preliminary review

Protocol v9 carries up to 96 generic rows and distinct 64-bit candidate/request identities. Source checks cover both `J*qdot_commanded` and `J_arm*qdot_arm + J_rail*rail_execution`, including a final check after rail override/shadow shaping. Late certified replies are discarded instead of being rebound to a new candidate. Functional tests reported by the owner pass, but two transaction issues were reported during independent review:

1. **Reproduced:** receiving a deferred native candidate set `_pending_commit_seq`, and the next update implicitly committed it even without any send or `commit_publication`. In a detached actual native runtime, the second tick's previous-command norm was 0.00532704788, exactly the first unsent candidate. Certified mode must arm native commit only after explicit publication confirmation.
2. **Source finding:** validation failure of the next certificate cleared the prior pending history before processing its commit flag. An actually published and confirmed previous command must remain committed even if the next candidate expires or is malformed. Its exact identity/epoch handling must be independent of the new candidate's admission.

Final Python payload changes also require matching native committed history. The root's selected certified-mode policy is to publish only the original native q_send/qdot and reject a rail reservation target change; it disables the later wall-clock rail rewrite. Final runner tests must verify this policy rather than assuming that a Cartesian-row recheck alone rebases native state.

**Timing requirement remains unmet for the active stress case.** The owner reports 96 active dense rows causing 288/1020 failed-closed candidates, p99 total 9.37 ms, maximum 20.24 ms and 44 calls over 5 ms. Inactive 0/96-row cases pass. These failures must remain in the performance artifact, and functional tests cannot be presented as a general 96-active-row real-time PASS. Hardware remains disabled/unverified while the implementation and detached adapter review continue.
