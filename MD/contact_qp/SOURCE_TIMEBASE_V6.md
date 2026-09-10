# Active source timebase v6

Active's nominal period remains 5 ms; fresh measurements advance fixed continuous filter poles using their actual source interval. `source.timebase: variable_step_bilinear_v1` requires separate `max_interval_s` and `max_age_s` (active_probe50: both 15 ms). Old modes retain the original filter path, force law and thresholds. No IK/native change.

The 45 Hz first-order LP and 2.5 Hz second-order HP are prewarped once at 5 ms and integrated by the trapezoidal rule. At fixed 5 ms their transfer function matches the original IIR. Shared observer entry preserves filtered output and previous raw input, and exit reconstructs DFII state; a repeated source is not filtered twice. Only first entry after the existing stopped/prepare boundary may explicitly initialize a new measurement epoch after a long old-history gap, seeded with the current compensated measurement. This logs `source_filter_epoch_reset`; it does not interpolate the pause or claim continuity. Subsequent long gaps, old samples, reversed timestamps and changed source epochs fail normally.

New capability `contact_qp.source_timebase_bilinear_v1` is checked before teaching/prepare and again by HFPC. An already-running older daemon is rejected; the operator must restart the updated Python controller. This work did not connect to or restart hardware.

Verification:

- rm75 environment: `pytest -q peirastic/tests/test_contact_qp_variable_timebase.py peirastic/tests/test_contact_qp_two_clock.py peirastic/tests/test_contact_qp_active.py peirastic/tests/test_contact_qp_runtime_config.py` — 40 passed, 1.63 s.
- genesis environment: `pytest -q /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_contact_qp.py /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_session.py` — 33 passed, 9.41 s.
- `PYTHONPATH=$PWD:$PWD/rm75_control /media/camp/EXT_DRIVE/envs/rm75/bin/python MD/contact_qp/replay_source_timebase_v6.py` regenerates `source_timebase_v6_replay.json`: 13,738 actual shadow records, 13,633 fresh and 105 held, no rejected samples. Actual fresh intervals 2.1516–10.9102 ms; maximum record age 10.0962 ms. LP error versus independent formula 0, HP error versus independent continuous matrix trapezoid oracle 5.33e-15.

These are software timing/filter and transaction results, not active closed-loop force or acoustic performance evidence. Actual active behavior remains to be measured by the operator.
