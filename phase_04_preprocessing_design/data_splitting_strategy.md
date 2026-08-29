## Data Splitting Strategy

Because this is a time-series forecasting problem, the split must respect chronology — a random/shuffled split would leak future information into training (the model could "see" patterns from dates after the ones it is being tested on) and would silently inflate reported accuracy.

### Split Ratios

| Split | Proportion | Approx. Hourly Sequences | Date Range Role |
|---|---|---|---|
| Train | 70% | ~12,200 | Earliest data — used to fit scaler and model weights |
| Validation | 15% | ~2,600 | Middle segment — used for early stopping and hyperparameter checks |
| Test | 15% | ~2,600 | Final segment — held out, touched only once for final Phase 8 results |

(The ~17,470 raw hourly rows, after resampling the 1,048,575-minute series, yield slightly fewer usable sequences per split once the 24-hour lookback window is applied — see actual shapes logged by `src/preprocess.py`.)

### Rules Enforced

1. **No shuffling** at any point before windowing.
2. **Windows never cross a split boundary** — a sequence's 24-hour input and its target must all fall on the same side of a split, so no test-set information ever appears inside a training window.
3. **Scaler fit on train only**, then reused (not refit) on validation and test, as stated in Phase 4 → Feature Scaling.
