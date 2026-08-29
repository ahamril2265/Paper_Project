## Validation Plan

* The **validation split** (Phase 4 → `data_splitting_strategy.md`, 15% of the resampled series, immediately following the training split in time) is used exclusively during training — for early stopping and for confirming the model is not memorizing the training window order.
* The **test split** (final 15%, chronologically last) is touched exactly once, after training is finalized, to produce the numbers reported in Phase 8. It is never used to tune hyperparameters, preventing the common mistake of quietly overfitting to a "held-out" set through repeated peeking.
* No k-fold cross-validation is used. Standard k-fold assumes exchangeable samples; shuffling a time series across folds would leak future information into training folds, which directly contradicts the no-leakage rule established in Phase 4. A single chronological three-way split is the correct validation strategy for this problem class.
