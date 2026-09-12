# Session 006: force-feedback failures and successful retries

Read-only audit of `/media/camp/yameng/icra 2027/uncalibrated/006`. No production code, source recording, hardware or IPC state was changed. `analyze.py` reproduces `summary.json` and `summary.txt` with the existing genesis Python/h5py environment.

## Main finding

**L_DtP/001 reached the new no-send retry path, then the source-cadence check rejected the next measurement.** The supervisor's later “Stale or missing force feedback” is consistent with processed force publication stopping after this controller failure. The recording process did not crash. Image feedback and tank exhaustion were not the initiating causes.

L_DtP/002 also encountered a source-age admission miss, successfully discarded the unsent candidate, recomputed using a newer source, and completed. Thus the retry path is loaded and functional; the remaining blocker in attempt001 is the separate15 ms source-interval limit.

L_PtD/001 and002 failed during preparation and have no QP log, raw recording or recorder log. The only established cause is the supervisor force-freshness check. This session cannot identify sensor transport, force publication, mode transition, MOVEJ or rail as the underlying cause of those two failures.

## Loaded configuration and session outcome

Nine attempts: six completed, three failed. All six requested paths eventually completed, although session status remains `interrupted`. Seven attempts contain controller QP logs and complete raw recordings. Every study_start loads:

- source period5 ms, variable_step_bilinear_v1, max_interval15 ms, max_age15 ms;
- feature.dropout_policy=pause_visual;
- task_power_source=nominal_command, tank initial.10 J/capacity.15 J/reserve.05 J;
- logical_final_model, two_port_command_model and50 ms command interval.

The loaded configuration and per-file source hash metadata are retained in summary.json. Runtime retry events and nominal task-energy fields additionally establish that these behaviors ran; not every production module is included in the recorded source hash list, so this is not a claim of complete deployed-binary provenance.

## L_DtP/001 timeline

Times are host monotonic seconds. Line numbers refer to its contact_qp.jsonl.

| Time | Evidence | Interpretation |
|---:|---|---|
|13952.694211|session prepare_begin|Preparation began|
|13963.626796|seek_begin|Contact seek began|
|13968.173181|tracking_observed|Supervisor observed scanning|
|13978.270278738|last processed-force source, raw force row2562|Last force sample later used by proposals2567/2568; force3.962605 N|
|13978.281540965|line21202, dual-device publication2567|Last successful arm+rail transport; command expiry13978.330998379|
|13978.282855046|proposal2568 created|It reuses the same source as a held observation; source age at prepare12.194271 ms|
|13978.285985684|next recorded raw TCP source, received13978.286706182|Observed source interval15.706946 ms after old source; raw sensor wrench is finite|
|13978.290740121|line21207, wrench_source_expired at final review|Old proposal source now20.461383 ms old; candidate2568 correctly not sent|
|13978.290906180|line21210, publication_fresh_retry|Unsent proposal aborted; command2567 retained with unchanged expiry, count1|
|13978.291332336|line21212 nested logical_epoch_work end|The next retry-source check advanced the retained ledger; no subsequent control_sample or publication occurred|
|13978.364024635|last raw TCP source, received13978.369778922|Raw TCP and raw sensor wrench continued after processed force stopped|
|13978.370278738|computed last force source+100 ms|Earliest the supervisor's100 ms force-age threshold could expire on this last sample|
|13978.426821|raw HDF5 ended_wall_time converted to monotonic|Recorder finished normally; complete=true,error empty, Saved log|
|13978.470792849|line21213 exception recorded|`ValueError: source interval exceeds declared maximum`|
|13978.470914160|line21217 stop|Same source-interval exception, reference8.928043704 s|

The explicit terminal reason is the source interval check. The retry-source gate runs around13978.2913, processed force ends before the supervisor100 ms threshold, and raw TCP/raw wrench continue. Together these strongly support **controller cadence rejection first, processed force outage second, supervisor stale feedback later**. There is no external_stop_before_send event or recorder exception supporting the alternative “recorder failure first caused an external stop.”

The exception log timestamp follows coordinated fault-stop work; it is not a timestamp of the first thrown exception. Do not compare its13978.4708 time directly with the100 ms freshness threshold and infer the supervisor failed first. The exact failed `prepare_source()` input is not logged. The15.706946 ms gap is the adjacent recorded TCP source gap, consistent with the declared limit and the explicit error; it should not be promoted to an independently logged exact controller-ingress delta. Intermediate recorder sequence omissions are possible.

At the rejection, tank balance was about.087992 J, available.037982 J above reserve; peak filtered force for the whole attempt was4.380 N. The eventual logical_epoch_expired_actual_tail_unknown latch is recorded after fault-stop cleanup because the old lease elapsed by then. It is a downstream settlement result, not evidence that an empty tank initiated the stop.

## L_DtP/002: the same admission retry succeeds

At14027.145558793, candidate5584 fails the dispatch force-age check by only7.592 microseconds: source14027.130551201, allowed-until14027.145551201. It has a passed final review but no transport publication. The retry event retains command5583 until14027.181295327.

Candidate5585 uses a newer source14027.143938662, with measured source interval13.387461 ms and prepare age2.020803 ms. Its dual-device publication succeeds at14027.151553088. Reference time stays23.433438717 s across the rejected proposal and advances only after candidate5585 commits. The new transport occurs5.994 ms after the rejected dispatch, within the original retained lease. The attempt later completes at reference26.887203679 s.

This distinguishes the two attempts: **successful next-source interval13.387 ms versus rejected interval exceeding15 ms**, while force-source age at actual new dispatch remains independently checked.

## Attempt comparison

| Attempt | Result | QP samples | Peak filtered Fz N | Maximum review-source age ms | Retries |
|---|---|---:|---:|---:|---:|
|L_DtP/001|force stale surfaced; cadence exception underneath|2568|4.380|20.461|1, then cadence rejection|
|L_DtP/002|completed|6315|4.574|14.752|1, recovered|
|C_DtP/001|completed|6461|4.460|12.858|0|
|S_DtP/001|completed|6766|4.566|12.935|0|
|L_PtD/001|prepare-only force stale|—|—|—|—|
|L_PtD/002|prepare-only force stale|—|—|—|—|
|L_PtD/003|completed|6427|4.325|12.276|0|
|C_PtD/001|completed|6508|4.342|14.070|0|
|S_PtD/001|completed|6801|4.349|12.980|0|

No required-image rejection, image-unavailable transition, force-supervisor6 N stop, QP preparation failure, native/IK failure or rail publication failure is recorded in these seven QP logs. Every committed publication reports arm sent and rail sent. Six completed logs end phase_complete. All seven close with zero dropped QP records and no writer error. All seven recorder logs report Saved, with CPU control overlap=none; all raw files are complete with an empty error attribute.

Recorder TCP/force sequence gaps occur in successful attempts too. For example, successful L_DtP/002 has a recorded force interval17.645 ms and TCP interval20.707 ms within the supervisor scan window, while controller source intervals remain admissible. Recorder streams are subsampled/paired SHM outputs, so their gap counters cannot substitute for the exact direct-controller ingress timestamps.

Prepare-only failures: L_PtD/001 prepare_begin14134.810131, L_PtD/00214141.209482; L_PtD/003 starts14147.608033 and completes. No standoff_ready or seek_begin was reached on the failures. No before/after source ages or sensor trace exists for those failed preparation intervals.

## Narrow read-only repair recommendation

Keep force-source max_age15 ms at prepare/final-review/dispatch. Do not renew the retained command lease, replay a command, reset source clocks, interpolate fake source samples, or relabel timestamps.

The currently conflated checks are:

1. `peirastic/contact_qp/runtime_source.py:45–46`: variable mode requires `period <= max_interval <= max_age`, coupling sample spacing to command freshness. At`:69`, a fresh new source with elapsed spacing above15 ms is rejected independently of how young that source is now.
2. `rm75_control/.../joint_admittance_8dof/loop.py:6435–6440` passes that same max_interval to the force observer. `.../admittance_common/observer.py:262–264` independently rejects the interval, so changing only SourceClock would merely move the failure downstream.
3. `observer.py:307–309` already feeds the actual elapsed interval into VariableLowpass1; `controller.py:2160–2167` already feeds source_dt_s into VariableHighpass2 and elapsed-time-scaled state updates. These are existing variable-step paths, not a need to fabricate regular samples.

A narrow candidate is to separate **freshness of the current source** from **spacing between consecutive accepted sources**, permitting the latter only within the already declared50 ms command-lease envelope when an unchanged, unexpired previous committed lease and watchdog permit continuation. The interval allowance should derive from that existing lease setting, not an arbitrary multiple of15 ms. Keep stopped/new-epoch seeding limited to actual mode boundaries; a running retry must carry forward existing LP/HP states and use the real interval once.

Both LP and HP models use prewarped fixed analog poles with trapezoidal updates (`variable_step_filter.py`). The low-pass denominator2+omega*dt and second-order denominator1+h*d+h²*w² are positive for positive finite intervals; the implementation checks finite input/state. There is no mathematical singularity at15 ms. This is a numerical-model observation, not a claim that a50 ms unobserved force trajectory is physically certified. Fresh-source age, force magnitude, old-lease expiry and watchdog remain separate controls.

Required regression before implementation: recorded15.706946 ms new-source gap with fresh arrival and live old lease continues without LP/HP reset; both filters consume the same true delta once; a repeated held timestamp does not learn again; old-lease expiry or nonfinite/reversed/new-epoch timestamps still stop; no-source arrival never extends the lease; nominal command integration/reference advance remain tied to actual successful publication. Also add logging of attempted source timestamp, prior source timestamp, interval and check timestamp before any coordinated-stop cleanup to eliminate the present causal-timestamp ambiguity.
