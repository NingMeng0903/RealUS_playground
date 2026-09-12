# Session 005 failure audit

Read-only audit of `/media/camp/yameng/icra 2027/uncalibrated/005`, 2026-09-11. No production files or hardware were changed. `summarize.py` reproduces `summary.json` and `summary.txt` using Python's standard library; HDF5 spot checks used `/media/camp/EXT_DRIVE/envs/genesis/bin/python`.

## What actually failed

Session metadata has 12 attempts: 5 completed, 7 failed. S_PtD remains pending with zero attempts. Ten attempts have QP logs and recorder logs. Two prepare-only force-stale failures have neither, so their underlying transport cause cannot be established from this session.

All times below are host monotonic seconds. They map to session timestamps with `(anchor_time_ns-anchor_monotonic_ns)/1e9` from `session.json`. Line numbers refer to each attempt's `contact_qp.jsonl`.

| Attempt | Reported error | First specific recorded cause | Timing / evidence |
|---|---|---|---|
| L_DtP/001 | Stale or missing force feedback | Not established beyond 100 ms supervisor force freshness check | Only prepare_begin 11393.489079; no recorder/QP log |
| L_DtP/002 | TCP target timeout | Required image feedback expired | Line 12325, 11432.828832, frame124116 effective time11432.524326; event-record age304.505 ms. Quality .9966/.9957/.9971, all valid, registration matches. Stop at11432.852358; supervisor marks path_done11432.867335 after abort. Reference only3.721861 s. |
| C_DtP/001 | TCP target timeout | Required image feedback expired | Line29875,11517.891872, frame126668 effective11517.589201; event-record age302.670 ms. Quality .9895/.9933/.9952, all valid, registration matches. Stop11517.913497; path_done11517.931810. Reference only13.710213 s. |
| L_PtD/001 | Task ended before scan started | Force source expired during final review | Line805, review11640.438141 minus source11640.422997 =15.143414 ms, exceeding15 ms by143.414 us. Ref0; tank .800 J. |
| C_PtD/001 | Stale or missing force feedback | Force source expired during final review | Line11130, review11745.702924 minus source11745.686977 =15.947303 ms, exceeding15 ms by947.303 us. Ref1.390155 s; last tank .793732 J. Later stop11745.856091. |
| C_PtD/002 | Stale or missing force feedback | Not established beyond 100 ms supervisor force freshness check | Only prepare_begin11758.033419; no recorder/QP log |
| C_PtD/003 | Task ended before scan started | Force source expired between review and dispatch | Line8530, dispatch11782.904474 minus source11782.889196 =15.278076 ms, exceeding15 ms by278.076 us. Final review had passed at14.940 ms. Ref0; tank .791312 J. |

The image `stale_or_future` label means **stale** here: received times precede the rejection, effective times precede it by over300 ms. The registration equality in the error message is incidental diagnostic context, not a registration failure. Event record time is slightly later than the actual check, so the ages above are event-record ages, not claimed exact check ages. The rejection itself establishes that the check exceeded300 ms.

These five recorded failures have no energy depletion precursor and no 6 N force-supervisor stop. Every recorded transport publication reports success before the abort. No session QP event identifies IK infeasibility, rail FA24 failure, Modbus NotConnected, or external_stop_before_send as a first cause. Such messages in a separate controller terminal cannot be assigned to these attempts without timestamps. The only invalid_rail_sample events are eight startup diagnostic alignment events in successful L_DtP/003.

## Failed-versus-successful retries

Statistics span each full QP log, including contact seek. Review ages use explicit review timestamps minus the exact proposal force-source timestamp; dispatch ages use explicit dispatch timestamps. Tank minima come from control samples, not a physical energy certificate.

| Attempt | QP samples | Peak filtered Fz N | Minimum tank J | Review age p99 ms | Review age max ms | Dispatch age max ms | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| L_DtP/002 |1494|4.295|.763|10.857|13.702|13.892|image age stop|
| L_DtP/003 |7482|4.426|.383|10.247|12.603|12.806|completed|
| C_DtP/001 |3642|4.370|.664|10.273|12.867|13.059|image age stop|
| C_DtP/002 |7582|4.397|.456|10.306|11.780|11.981|completed|
| S_DtP/001 |7825|4.647|.378|10.392|12.018|12.210|completed|
| L_PtD/001 |100|.257|.800|11.734|15.143|11.881|review stop|
| L_PtD/002 |7561|4.323|.666|10.703|14.426|14.773|completed|
| C_PtD/001 |1350|4.219|.790|10.821|15.947|14.046|review stop|
| C_PtD/003 |1036|4.108|.791|10.688|14.940|15.278|dispatch stop|
| C_PtD/004 |7624|4.320|.729|10.676|13.638|13.849|completed|

The two final-review failures have no transport result for their rejected proposal; dispatch maxima on those rows describe earlier successfully sent proposals. C_PtD/003 dispatch maximum includes its rejected no-send proposal. All five completed attempts stop with phase_complete. Their final reference times are32.434368,32.858732,34.344186,32.434334,32.820231 s respectively. Successful S_DtP has no stop/timeout explaining a later black image; image quality analysis must address that separately.

All ten recorders saved raw.h5 and report CPU control overlap=none. All ten QP logs close with dropped_records=0 and writer_error=null. Recorder sequence gaps occur in both failures and successes, so a sequence-gap counter alone does not establish a stop cause.

HDF5 force spot check around C_PtD/001: last recorded force source11745.679785, last QP source11745.686977, final review rejection11745.702924, stop11745.856091. The recorder ending before stop is consistent with the supervisor100 ms force check firing during the stop path. It does not establish that force loss caused the earlier15 ms admission failure; the exact no-send admission failure is recorded first. Similarly L_PtD/001 and C_PtD/003 raw force streams end before final stop. Their raw recordings contain no controller message dataset, despite recorder progress counters mentioning controller samples.

All raw recordings have a force-source gap of roughly79–222 ms at contact mode entry, matching `source_filter_epoch_reset` with a new active epoch. The same mode-entry gap exists on successful runs and is seeded from a fresh current measurement without interpolation. Do not mistake it for an in-scan stale-source event.

## Code locations and practical changes

1. `/media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py:350` interprets any `done_seq >= expected_seq` after tracking as path_done. Line376 then waits2 s for the final TCP target using `_await_pose` at188–203. This masks the two confidence-triggered early aborts as TCP target timeout. Before declaring path_done, verify a successful task terminal reason and expose the originating controller/QP stop reason. Do not increase endpoint timeout or pose tolerance to hide this path.
2. `scan_robot.py:49–53` checks LiveSamples timestamps against100 ms. `status():117–145` checks stopped/error statuses, but the early done/force-stale outcomes show the useful underlying reason is not consistently retained by this surface. Persist the original terminal reason in controller state and include it in scan failure.json, with proposal/source/check timestamps. The prepare-only failures need this diagnostic evidence before attributing them to rail or reset state.
3. `peirastic/realman8dof/modes/contact_active.py:251–264` rejects required stale confidence after the contact gate starts. `peirastic/contact_qp/qp.py:49` defaults max_image_age_s=.30. Recorded effective delay is.15196365 s, consuming half the age budget before delivery/hold; C_DtP failure held the latest received observation111 ms before rejection. Reduce feature scheduling/delivery stalls, log capture/effective/receive/check ages separately, and validate fresh image throughput before contact. Retain stale-image rejection; do not reinterpret matching registration as sufficient freshness.
4. `contact_active.py:330` is the15 ms final force-source check; `:349–360` repeats it immediately before the first device send. `loop.py:7219–7232` and`:7241–7251` currently abort all execution on a single expired proposal. These are narrow no-send jitter cases, not evidence for force safety timeout relaxation or tank retuning.
5. `loop.py:7102–7104` emits external_stop_before_send only when stop_check() becomes true. It is a control-stop/handoff path; keep it separate from freshness admission and preserve immediate stop behavior. No such event occurs in these session QP logs.

## Bounded fresh recomputation design (read-only; not implemented)

The current transaction interfaces support dropping an unsent candidate while retaining committed history, but replacing `break` with bare `continue` would be incomplete.

- Only a classified wrench_source_expired review rejection, or a classified stale-source dispatch rejection with every other dispatch invariant valid, is eligible. Current publication_started collapses multiple reasons into missing_review_or_stale_dispatch: split reason reporting before enabling retries. Other invariant failures must remain fatal.
- At that pre-send point, call rail.abort_reservation(), publication_owner.publication_abort(..., definitely_not_sent=True), and inner.abort_publication(). No arm send or rail reservation commit may have occurred. Require an active previous successfully committed logical pair and unlatched budget still within its original expiry **after** abort settlement. Do not renew its lease.
- `contact_active.py:407–420` aborts nominal force/tilt command proposals without rewinding observations or advancing reference. `command_budget.py:196–206` rejects only the unsent new reservation and leaves old epoch accounting in place; `_settle():99–135` expires and latches an old epoch at its original deadline. Preserve all of these semantics.
- `loop.py:1315–1320` discards inner pending publication. Native `wbc_rt/client.py:242–244` sets _abort_next and clears pending commit; `:673–682` sends IN_ABORT_PREV on the next native request instead of committing the rejected candidate. Rail `hw/rail_servo.py:1253–1255` only clears the new reservation.
- Keep the last published q_cmd/qdot, outer _previous, reference time, and nominal committed state. Start the next proposal from a strictly newer source timestamp than the rejected proposal. If no newer valid source arrives within the bounded opportunity, stop; never send a held proposal or substitute a new timestamp onto old computations.
- A skipped-send branch must still pace the loop (`ticks`, `next_tick`, `_wait_until` atloop.py:7431–7433) and record skip/age/retry state, but must not call wd.beat() (normally729? after commit at7303), reference advance, arrival confirmation, or transport-success telemetry. The existing watchdog must continue to bound time since last actual successful send.
- Bound retries to a very small explicit count/time inside the remaining old epoch; recheck external stop, watchdog, source freshness, image freshness and budget on each new iteration. Stop on any failed guard, absent prior active epoch, partial/unknown publication, rail failure, or old-epoch expiry. Do not call fault reset, mode reset, hold replay, or freshness-threshold changes.
- Meaningful tests need first-command expiry (must stop), old-epoch expiry during retry, fresh-source recompute with unchanged reference on rejected candidate, native abort-before-next-prepare, no duplicate transport, dispatch non-age invariant failure, and external stop during pending retry. This is a control-path behavior change and requires those transaction tests before execution.

Unresolved: the cost from source acquisition through outer solve, original IK, rail reservation and final Python review is not separated by these contact logs. The p99 ages are similar between success and failure and failures are single-tail excursions; instrument timing before blaming IK itself. No autonomous recovery action is justified by this read-only audit.
