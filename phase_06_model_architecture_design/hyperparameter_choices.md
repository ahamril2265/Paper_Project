## Hyperparameter Choices

| Hyperparameter | Value | Rationale |
|---|---|---|
| Sequence length (lookback) | 24 hours | Matches the diurnal cycle identified in Phase 3 — one full day of context per prediction. |
| Hidden units (LSTM-1 / LSTM-2) | 64 / 32 | Funnel shape keeps parameter count modest for a ~12k-sequence training set (Phase 4 splitting strategy), reducing overfitting risk. |
| Dropout | 0.2 | Light regularization; higher values were avoided given the already-small hidden sizes. |
| Batch size | 64 | Balances gradient stability against the limited training set size and CPU-only training (Phase 1 constraint). |
| Learning rate | 1e-3 | Standard starting point for Adam on normalized [0,1] regression targets. |
| Max epochs | 60 | Upper bound; actual training stops earlier via early stopping (Phase 7). |
| Loss function | Mean Squared Error | Directly aligned with the RMSE evaluation metric (Phase 7), and differentiable everywhere for stable gradients. |
| Optimizer | Adam | Adaptive learning rate handles the differing scales/variances across the 9 input features better than plain SGD. |
| Random seed | 42 | Fixed for reproducibility across preprocessing, splitting, and model initialization. |
