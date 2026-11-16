# Causal chart-position embedding

`embedding.py` provides a fixed-size public feature vector for the first-stage
search/reacquisition study. It encodes the current chart location, a few
samples of the candidate `(s, x)` path, the pending query interval, the last
trusted station, the knee angle phase, and the top posterior modes. Each mode
keeps its posterior mass, predicted target offset and field-of-view probability;
the modes are not replaced by one posterior mean. Anonymous packet-to-target
responsibilities are padded to three entries with a mask.

The chart `(s, x)` pair is the recovery benchmark's registered material-chart
proxy. It is not claimed to be a complete harmonic 3-D coordinate. The module
accepts only public predictions and posterior quantities. It does not accept a
`Scene`, truth roles, hidden hypotheses, or evaluator labels.

The embedding is intended for an ablation: current features only, current plus
path, and current plus path plus posterior modes. A controller should be able
to run with the same action and observation interfaces when this feature is
removed.

