# Stage 0 independent review — 2026-09-10

**Decision: PASS for the phase-0 contract freeze.** No remaining concrete correctness blocker was found within the reviewed documents. This decision authorizes proceeding to implementation and later stage gates; it does not establish controller efficacy, implementation correctness, or hardware certification.

Scope: `CONTRACT_V1.md` and `acceptance_v1.json`, against baseline `95038b5f33ff28ed38179482f626eacbb5822329`. No implementation or hardware was exercised. The acceptance JSON parsed successfully.

The revision resolves the initial findings:

- Reference advancement explicitly uses the bounded accepted interval and final compatible alpha, with zero-alpha and duplicate-ACK behavior specified. Measured acquisition coverage remains separate.
- Each device send/commit checks expiry and stop epoch; queued sends are fenced, while uncertain in-flight execution retains liability and restricts recovery.
- The transform direction, TCP control subspace, rocking sign, image-x convention, and geometric stiffness-center rows are explicit.
- Image quality and completion gates, force comparators, zero-margin per-scenario and pooled bootstrap rules, failed-run retention, independent scenario RNG streams, and gamma ablation identities are frozen.

The energy requirements preserve old-command exposure, reserve worst-prefix consumption and stopping tails, forbid abort refunds, and settle each physical interval once while releasing its reservation in the same ledger operation. These remain requirements for stage-3 fault injection and audit.

Real geometry, physical power signs, and exposure/stop/error bounds remain unverified. Synthetic assumptions cannot confer hardware certification. Acceptance failures must retain the results and leave the feature NOT ACCEPTED/default disabled.

Reviewed SHA-256:

- `CONTRACT_V1.md`: `497f228396964b905ca47b509b90c6de504da9db1fd109c7f93c3299dff9ee51`
- `acceptance_v1.json`: `cfa260774740d457f827ff238a904428b79ceab5215a33b3ba5aeee23e09fa47`
