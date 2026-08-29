## Training Strategy

1. **Fit the scaler** on the training split only (Phase 4).
2. **Build windows** (24-hour lookback → next-hour target) independently within each split, so no window straddles a split boundary (Phase 4 → `data_splitting_strategy.md`).
3. **Train the two baselines first** (naive persistence needs no fitting; Random Forest is fit once on the flattened training windows) to establish the floor a deep model must beat.
4. **Train the LSTM** using the configuration in Phase 6, monitoring validation loss every epoch.
5. **Apply early stopping** (patience = 8 epochs) and restore the best checkpoint by validation loss — this is the single overfitting-prevention mechanism used, chosen over aggressive dropout or weight decay because the architecture (Phase 6) is already small relative to the training set size.
6. **Freeze the model** and run it once on the test split to produce the numbers Phase 8 reports.

This is a linear, one-directional pipeline — no step is allowed to reach back and adjust an earlier one after test-split numbers are seen, which is what keeps the final results honest.
