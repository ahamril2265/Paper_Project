## Evaluation Metrics

Consistent with the objectives stated in Phase 1, three standard regression metrics are computed on the **held-out test split only**, after inverse-transforming predictions back to the original kW scale (undoing the Phase 4 Min-Max scaling) so results are physically interpretable:

* **MAE (Mean Absolute Error):** average magnitude of prediction error in kW — the most directly interpretable metric for a non-technical energy-management audience.
* **RMSE (Root Mean Square Error):** penalizes large errors more heavily than MAE; important here because occasional large appliance-driven spikes (Phase 3 → data characteristics) are exactly the events a demand-management system most needs to anticipate.
* **R² (Coefficient of Determination):** proportion of variance in global active power explained by the model, giving a normalized (dataset-independent) sense of fit quality.

All three metrics are computed identically for the naive persistence baseline, the Random Forest baseline, and the LSTM (Phase 5), so Phase 8's comparison table is on equal footing.
