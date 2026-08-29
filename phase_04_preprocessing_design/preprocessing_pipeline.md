## Preprocessing Pipeline

The raw dataset (1,048,575 minute-level records) requires a structured preprocessing pipeline before it can be consumed by a sequence model. The pipeline is designed as a strict, leakage-free sequence of steps applied in the following order:

### 1. Timestamp Construction & Sorting
`Date` and `Time` are merged into a single `Datetime` index and the frame is sorted chronologically (already validated in Phase 2.5).

### 2. Missing Value Handling
Missing values (~0.39% per numeric column, confirmed random/sensor-related rather than contiguous) are resolved with **linear time-interpolation**, followed by a forward/backward fill for any boundary gaps interpolation cannot reach. This preserves the continuity of the time series without introducing artificial discontinuities that a mean/median fill would create.

### 3. Temporal Resampling
The native 1-minute resolution is aggregated to an **hourly mean**. This is a deliberate design choice, not a limitation:

* It directly addresses the **Hardware Limitations** constraint documented in Phase 1 (training on the full 1M-row minute-level series is unnecessary for the stated forecasting horizon and would multiply training cost for no accuracy benefit).
* Hourly resampling is the standard preprocessing choice used across published work on this exact dataset, since minute-to-minute noise from individual appliance switching is not the signal of interest for demand-management use cases.
* It reframes the task as **next-hour global active power forecasting**, which remains a short-term horizon consistent with the Phase 1 problem definition.

### 4. Feature Scaling
All input features (and the target) are scaled independently with **Min-Max normalization to [0, 1]**. The scaler is **fit only on the training split** and applied to validation/test splits, preventing information leakage from future data into the training process.

### 5. Chronological Train/Validation/Test Split
The resampled series is split **without shuffling**, preserving temporal order:

* **Train:** first 70%
* **Validation:** next 15%
* **Test:** final 15%

This mirrors real-world deployment, where a model is trained on the past and evaluated on data it has never seen, in the order it will actually occur.

### 6. Supervised Sequence Windowing
The scaled series is transformed into fixed-length input/output pairs using a **sliding window of 24 hours** (one full day of history) to predict the **global active power at hour t+1**. Each sample is therefore a `(24, 9)` input tensor (24 timesteps × 9 features — 7 raw electrical measurements plus 2 cyclical time features, see [`feature_engineering.md`](feature_engineering.md)) mapped to a single scalar target.

> [!NOTE]
> All steps are implemented in [`src/preprocess.py`](../src/preprocess.py) and are deterministic given a fixed random seed, so results in Phase 8 are reproducible.
