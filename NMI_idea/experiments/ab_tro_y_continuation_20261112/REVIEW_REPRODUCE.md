# Independent replay of the moving-Y evidence

Read the [review request](../../B_REVIEW_REQUEST.md) and the [frozen protocol](PROTOCOL.md). The [study report](REPORT.md), [Astra review](ASTRA_REVIEW.md), and [archived independent statistics](ASTRA_INDEPENDENT_STATS.json) describe the completed run.

From the repository root with Python 3.10 or later, NumPy and SciPy, run

    OPENBLAS_NUM_THREADS=1 python NMI_idea/experiments/ab_tro_y_continuation_20261112/audit_replay.py > review_audit.json

This replays all 768 terminal association confidences from the public masks, checks raw-file SHA-256 digests, independent target labels, joint and clock links, path costs, pairing, and the success and risk comparisons. It reads the originally archived [shared scans](holdout/COMMON.jsonl.xz), [rollouts](holdout/RUNS.jsonl.xz), and [completion hashes](holdout/COMPLETE.json). The two JSONL archives are losslessly compressed. The audit decompresses them in memory and verifies the original raw digests before parsing. To inspect the uncompressed files or run the original verification or plotting tools, unpack with xz -dk first. The portable audit writes its result to stdout. Load the [3D HTML](holdout/TRO_CONTINUATION_3D.html) locally in a browser.

The portable audit uses a hashed snapshot of the [transported synthetic branch](review_inputs/MANIFEST.json) and the shared recorded rule route. It checks the published frozen source files against the original [source manifest](FROZEN_MANIFEST.json). The two upstream atlas/attachment rebake arrays are replaced for this audit by their already-transported branch output. This replay checks the observed-data analysis and logged mechanics. It does not regenerate candidate paths, actuate the robot, or validate physical ultrasound.

The unchanged [study runner](run_study.py), [planner](planning.py) and [sampled executor](mechanics.py) record how the data were produced. Running that original generator requires its original filesystem layout, upstream atlas/attachments and calibrated robot environment. The archived native outputs are fixed-pose [mechanical preflight](native_preflight/bounded/WITNESSES.json) traces. Both native directories contain corresponding numeric traces and state snapshots. No patient captures or raw body-model files are required to run the portable audit.

Historical experiment directories include reports and selected audit results. Their legacy local-path instructions refer to the original experimental workspace.
