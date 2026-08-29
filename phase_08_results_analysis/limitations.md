## Limitations

* **Single household, single region:** consistent with the Phase 1 scope, results reflect one household's consumption behavior and cannot be assumed to generalize to other households, climates, or occupancy patterns without retraining.
* **Hourly, not minute-level, forecasting:** the resampling choice in Phase 4 trades away minute-level granularity for a more tractable and less noisy learning problem. Applications that need sub-hourly control signals (e.g. real-time appliance scheduling) would need a different preprocessing design.
* **No exogenous variables:** weather, occupancy, electricity tariffs, and calendar effects (holidays, weekends) are not modeled, even though the rejected Phase 2 dataset alternative contained a compatible weather-linked series that could supply this in future work.
* **No explicit anomaly handling:** sudden appliance failures, meter faults, or one-off unusual usage days are treated the same as normal variation; the model has no mechanism to flag or discount them.
* **Static train/test boundary:** the model is trained once and evaluated once on a fixed future window. A deployed system would need periodic retraining as consumption patterns drift over time (e.g. seasonal effects, appliance upgrades), which is not evaluated here.
* **Small architecture search:** only one LSTM configuration (Phase 6) was evaluated against two baselines (Phase 5); a broader hyperparameter search (different lookback windows, hidden sizes, or attention-based alternatives) was out of scope given the project's hardware constraints (Phase 1).
