# Raster versus Lissajous contact evidence (2026-09-08)

This is an offline comparison of the three saved scanner runs.  No controller,
SHM, or hardware process was started while collecting this evidence.

## Saved force traces

| run | pattern | approach samples/time | first contact | tracking contact |
|---|---|---:|---:|---:|
| `065825_899879` | raster | 1769 / 17.69 s | `t=17.68 s`, `Fz=+1.02 N`, `vz=+7.0 mm/s` | lost at `t=18.43 s`; reacquired at `t=33.93 s` |
| `065950_647403` | Lissajous | 378 / 3.78 s | `t=3.77 s`, `Fz=+1.15 N`, `vz=+7.0 mm/s` | continuous through the trace |
| `070109_001112` | Lissajous | 378 / 3.78 s | `t=3.77 s`, `Fz=+0.87 N`, `vz=+7.0 mm/s` | continuous through the trace |

The raster trace contains `vz_cmd=+12.0 mm/s` for the long no-contact interval.
Therefore the controller did issue the automatic tool-Z seek; the saved trace
does not support a conclusion that the raster branch disabled force seeking.
The raster run did eventually reach contact, then lost it during the first
part of tracking.  That points to a physical start-point/registration or
runtime pose-following discrepancy, rather than a raster endpoint position
limit.  The exact measured q path was not recorded in these runs, so the
runtime IK branch cannot be identified from these files alone.

## Offline TCP/IK check

Using the gripper2 TCP offset from the force-identification record, FK of each
saved detection q agrees with its captured TCP pose within 0.04 mm and 0.006
degrees.  The production fixed-rail pose resolver accepts both the raster and
Lissajous start and standoff targets.

| pattern | start xyz (mm) | standoff xyz (mm) | offline result | smallest controller-margin distance |
|---|---|---|---|---:|
| raster | `[268.50, 235.22, 176.04]` | `[269.12, 236.14, 216.03]` | IK solved | 20.63 deg (start J4)
| Lissajous | `[307.95, 233.76, 169.30]` | `[311.20, 233.29, 209.16]` | IK solved | 25.58 deg (start arm margin)

Both standoff offsets are approximately 40 mm along the outward surface normal
and both offline solutions remain inside the configured position margins.  The
offline outer-loop check also produces `+12 mm/s` in tool Z for both starts;
there is no simulated world-Z sign or force/path cancellation difference.

## Force plot locations

The generated plots are next to each run's CSV:

* `peirastic/logs/phantom/20260908_065825_899879/force_trace.png`
* `peirastic/logs/phantom/20260908_065950_647403/force_trace.png`
* `peirastic/logs/phantom/20260908_070109_001112/force_trace.png`

Each corresponding `force_trace_summary.json` reports 100 Hz complete
compensated tool-wrench samples and records both `approach` and `tracking`.
