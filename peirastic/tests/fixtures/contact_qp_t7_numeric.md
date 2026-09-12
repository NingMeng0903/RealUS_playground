# T7 numerical regression fixture

`contact_qp_t7_numeric.npz` contains H/g/C/l/u for eight rejected QPs from
2026-09-12 T7 scans and the final 20 successful samples preceding case 00,
followed by its rejected sample. Keys are `case_00_H` ... `case_07_u` and
`history_0_H` ... `history_20_u`.

The eight source attempts are yuhan_L RH_Per_S_DtP/001, RH_Per_S_PtD/001,
/004, /005, /006; zhongyao2 LH_Per_L_PtD/001, RH_Per_S_DtP/001, /002.
The preparation rejection did not log raw wrench/complete energy snapshot:
reconstruction uses the preceding successful wrench and available budget.
Successful-history absolute angles use zero, away from the inactive angle
limits. This is a numerical regression, not closed-loop or physical replay.

With ProxSuite 0.7.3 the 20-sample numerical history reproduces the logged
19303-iteration rejection; bounded retries solve every fixture while retaining
the full problem, strict status/residual/gap gates and energy row.
Original QP SHA256: c687333ea3db2635e9756f8d3b64802fa074a9a3e283bcd5767cdcce612c2f6d.
