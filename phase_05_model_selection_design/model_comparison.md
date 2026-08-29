## Model Comparison Framework

| Model | Handles Non-Linearity | Handles Temporal Order Explicitly | Handles Multivariate Input | Training Cost | Interpretability |
|---|---|---|---|---|---|
| Naive Persistence | No | Implicitly (t-1 only) | No | None | Very high |
| Random Forest | Yes | No (order-agnostic over flattened window) | Yes | Low | Medium |
| LSTM | Yes | Yes (recurrent state) | Yes | Medium | Low |

Phase 3 established that residential power consumption is **non-linear** and **strongly time-dependent** (diurnal cycles, appliance-driven spikes). This rules out the naive baseline as a real candidate and motivates comparing a model that ignores explicit sequence order (Random Forest) against one that is built around it (LSTM). Final model selection is decided empirically in Phase 8 based on MAE/RMSE/R² on the held-out test split, not on architecture alone — but LSTM is the primary hypothesis under test, consistent with the project abstract.
