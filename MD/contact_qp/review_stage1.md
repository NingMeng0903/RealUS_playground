# Stage 1 independent review — 2026-09-10

**Decision: PASS for stage 1.** All seven concrete findings are closed, and the final exact-entry 100,000-tick equivalence run reports maximum absolute twist error **0.0**. Scope includes the new contact-QP types, geometry, feature extraction/worker, bounded JSON subscriber, their tests, the ProxyOuter/runner measurement-forwarding changes, and the nominal force/tilt transaction adapter. This gate permits stage-2 implementation; it does not establish new-controller efficacy or hardware certification.

The random-walk implementation matches the cached Karamalis thesis §4.3.4 equations 4.10–4.14: depth-scaled intensities, horizontal/vertical/diagonal edge weights, an eight-neighbour Laplacian, and top=1/bottom=0 Dirichlet probabilities. The geometric point-velocity and dual wrench transforms use the original TCP pivot and preserve full port power.

Concrete findings reported to the implementation owner (status updated after recheck):

1. **Closed:** `LatestObservation.accept` checked effective-time ordering before registration/source/window transitions. A delay revision could retain old-version evidence. The revision handles the transition before effective-time ordering, fences retired sources, and adds the counterexample regression.
2. **Closed:** `accepted_alpha` returned immediately for a zero path column, skipping finite final-twist and subspace checks. Both final mappings are now validated before returning zero progress; regressions cover off-subspace and NaN twists in either mapping.
3. **Closed:** the runner initially forwarded its rail-compensated execution prediction as `measured_twist_base`. A separate tracker now uses raw rail encoder differences and arm SDK speed or measured-position differences. Both source timestamps, derivative intervals, freshness/ordering/identity checks, and time skew are explicit. Missing/untrusted components produce an unavailable full measurement. The runner forwards this channel separately, and `port_verified=False` prevents treating asynchronous measurement assembly as a verified physical power port.
4. **Closed:** `ContactObservation.fresh` could return true before receipt. It now requires receipt by the query time, with regression coverage.
5. **Closed:** `process_parts` combined missing source and publisher identities into `":"`. Both nonempty identities are now required before image decoding, with regression coverage.
6. **Closed:** `_lat_soften_hold_s` was armed by a rejected lateral command and survived abort. The reproduced initial-zero timer became 0.995 s after one rejected scan proposal. It is now transactional, and a changed accepted lateral command rebases the timer from its accepted speed. Repeating the counterexample leaves the timer at zero; a focused regression was added.
7. **Closed:** rejecting a contact re-arm restored away its force-point reset after the one-shot measurement event was consumed. In the reproduced sequence the next proposal reverted from the new contact origin to the old episode. The adapter now retains observation-triggered origins/zero resets while deferring command integration. The next proposal correctly retains the new origin after rejection; a focused regression was added.

The nominal adapter uses explicit command-state capture rather than a whole-controller rollback. Changed accepted normal/rocking actions update their integration histories, rejected proposals retain measurement updates, and reset invalidates pending proposals while preserving the measurement sequence high-water mark. Optional active flow/shield/TDPA variants are explicitly excluded. Their diagnostic balances do not provide the new physical energy certificate.

The JSON subscriber bounds each snapshot to four small messages and leaves B-mode decoding/solving outside the control thread. Explicit calibration versioning now joins registration/window versioning. No further concrete blocker was found in these additions.

Final evidence checked:

- `nominal_equivalence.log`: 100,000 ticks, maximum absolute twist error 0.0; 13 tests passed in 168.73 s. The fixture includes `force.yaml`, the ICRA torque profile, `SCAN_FORCE_AXES`, tool frame, 4 N, 0.010 m/s normal cap and seek speed, plus all audit fixes. `nominal_equivalence.command.txt` records the reproduction command.
- `stage1_regression.log`: 83 tests passed in 2.76 s across ingress, geometry, full measurements, proxy forwarding, existing scan/reference behavior, and final-send-chain regression. The implementation owner separately reports 64 nominal-short/measured/proxy/legacy/admittance/tilt tests passed, with the long test deselected.
- `sha256sum -c stage1_nominal_measured.sha256`: all six listed nominal/runner/source/test artifacts matched the files reviewed. That manifest's SHA-256 is `41805adb2cbee3f1d134955a4a7263821d8637960419424657b5c7d22b67c47f`.

Additional reviewed SHA-256 values:

```text
c672e771de6bf53855783af9cda6312d0ef8a3e9dd6f2d2c0156f60a68a7b0ac  peirastic/contact_qp/types.py
76bb55e9bc0cc1356fcbd853a8b9a2ec0ece467ef78672921a8faf835da72893  peirastic/contact_qp/geometry.py
697d3b10c0813e28068bef93747c151a2226545002ca3313193f0bd154dbe46b  peirastic/contact_qp/features.py
1f498e2e8524c200ac8cf1f17d5c7a9d33f0ba1c2593685e33975d1fcb326c9e  peirastic/apps/contact_qp_features.py
73b1a2fa5c2d0799ea3d36341d903dde9c44e84a63a602a851b0823a78ede134  peirastic/realman8dof/force/contact_observer.py
0ef6b4c47b68a2e316bad15cd8bd57ed0bc21a459764e7d1711a941b06d69303  peirastic/realman8dof/session.py
4c31fdba0167a6156fe219a710e809362d09e4aebd138e1f28fc07c3da75e0ee  peirastic/tests/test_contact_qp_proxy.py
826ed701510b00486b6afea12bb59b321b4e2933c1176ea4bae1f882ac1c6bed  MD/contact_qp/nominal_equivalence.log
bc1ed09b8d2790af5bf164ccddd025a62f3318312b653ff306de50d67088173f  MD/contact_qp/stage1_regression.log
```

No held-out acceptance data or hardware motion was used in this review. Geometry, physical port signs, latency/exposure bounds, and hardware passivity remain unverified as frozen in stage 0.
