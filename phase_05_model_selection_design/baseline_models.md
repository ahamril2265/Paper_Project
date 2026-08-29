## Baseline Models

Two baselines are established before introducing the LSTM, so that any claimed improvement in Phase 8 is measured against something concrete rather than asserted in the abstract alone.

### 1. Naive Persistence Baseline
Predicts hour *t+1*'s global active power as simply equal to hour *t*'s value. This is the standard baseline for short-horizon load forecasting: if a model cannot beat "tomorrow looks like today," it has not learned anything useful.

### 2. Random Forest Regressor
A non-sequential, tree-based ensemble (`scikit-learn`) trained on the same 24-hour flattened window (216 features: 24 timesteps × 9 features) with the same target. This tests whether the recurrent structure of an LSTM is actually necessary, or whether a model with no explicit sense of time order can already capture most of the signal from the windowed features.

Both baselines are trained and evaluated inside the same pipeline as the LSTM (same splits, same scaler) so that Phase 8's comparison is apples-to-apples.
