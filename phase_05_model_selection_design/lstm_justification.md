## LSTM Justification

A Long Short-Term Memory (LSTM) network is selected as the primary model, for reasons directly tied to findings from earlier phases:

* **Vanishing-gradient resistance:** Standard RNNs struggle to retain information across long sequences. The gated cell-state mechanism in LSTMs (Hochreiter & Schmidhuber, 1997) allows the network to retain or discard information across the full 24-hour input window, which is necessary to capture the diurnal usage pattern identified in Phase 3.
* **Native handling of multivariate sequences:** The dataset's 9 input features per timestep (Phase 4) map directly onto an LSTM's input dimension without manual flattening, unlike the Random Forest baseline.
* **Precedent in the literature:** LSTM-based load forecasting is a well-established approach for exactly this class of problem (smart-meter, sub-hourly/hourly residential demand), which supports the dataset selection rationale's note on academic benchmarking (Phase 2).
* **Matches the scale of the data:** Phase 2's dataset selection rationale explicitly chose this dataset over the smaller alternative *because* it had enough volume to support a deep learning model — an LSTM is the natural model class that justifies that earlier decision.

The Random Forest baseline (Phase 5 → `baseline_models.md`) exists precisely to test this justification empirically rather than accept it on assumption; if the LSTM does not outperform it in Phase 8, that would be reported honestly as a finding, not hidden.
