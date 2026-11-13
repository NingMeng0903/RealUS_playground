# Independent SOL replay

Run from the RealUS_playground repository root with NumPy and SciPy available.

```bash
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cohorts=(reference minus-plus plus-minus beta2 pose25 transient static-world)
ind=NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/verification/DYNAMIC/independent
runs=/tmp/leg-dynamic
ATLAS_SOL_REPLAY_JSON=/tmp/sol-outcomes.json python "$ind/outcome_replay.py" "$runs" "${cohorts[@]}"
ATLAS_SOL_POSTERIOR_JSON=/tmp/sol-posteriors.json python "$ind/posterior_replay.py" "$runs" "${cohorts[@]}"
ATLAS_SOL_FORECAST_JSON=/tmp/sol-forecasts.json python "$ind/forecast_replay.py" "$runs" "${cohorts[@]}"
python "$ind/causal_prefix.py" --among-us "$among" --repair "$repair"
python ../dynamic-sol-review/causal_prefix.py --among-us "$among" --repair "$repair"
```

The three offline scripts read the saved run JSON and NPZ. They independently reconstruct the plane intersections and FOV-filtered packet order, physical and accepted centring, identity, five-bin coverage, false completion, travel, turn, deadlines, and dropout recovery. They compute the joint filter using direct tensor contractions and an adaptive quadrature for the continuous stationary OU joint cell masses. The same-packet forecast replay consumes the static-map arm's recorded packets and assesses effective offset at 0, 0.1, 0.3, 0.5 and 1 s. The positive-horizon packet log score uses the future saved pose and packet only for evaluating the frozen forecast. The fourth script requires the same RM75 and geometry assets as the scan entry point.

For the causal integration test, held-out seed 113 initialized a reference geometry OU case. All three arms were rerun after adding 4 mm to the motion tape, clearing all visibility bits, and adding 10 mm to the measurement noise tape from exposure 14 onward. Frames and MPC decisions before exposure 14, and realized FK, joint and lateral states through the pose of exposure 14, matched bitwise. In a separate rerun, the evaluator was passed swapped labels and physical lateral positions shifted by 18 mm, retaining the same anonymous packets. The reported truth outcomes changed and the controller's association, selections, decisions, revisions and complete trajectories matched bitwise. Passing an explicit zero-motion tape to the original static controller matched the old static entry on all shared fields. The frozen static atlas replay was byte-identical, and the previous 256-case section benchmark matched all discrete outputs and floating-point traces within 3.02e-14.

Across all seven cohorts, the independent replay verified 516 paths, 103,716 ticks and 21,156 exposures. The maximum association probability difference was 8.57e-12. Effective displacement and residual means matched within 2.19e-8 mm, and central finite-grid interval endpoints agreed exactly. Production and independent common-packet forecasts agreed within 3.38e-10 on RMSE, interval coverage and interval width. In the primary comparison, the 0.1 s RMSE was 1.435 mm for the static map, 1.390 mm for IID and 0.921 mm for OU. Static, IID and OU correctly completed 78, 73 and 80 of 96 scans. The uncertainty interval for both OU completion differences included zero. The independent outcome replay uses its own 50,000-draw cluster bootstrap as a sensitivity check. The frozen experiment uses its declared 10,000-draw seed and supplies the reported intervals.
