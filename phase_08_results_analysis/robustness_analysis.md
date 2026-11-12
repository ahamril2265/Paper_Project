## Robustness Analysis (multi-seed, gradient boosting, significance, cost)

Produced by `src/experiments.py` → `results/robustness.json`. Each learned model was trained with 5 seeds (0–4) on the identical Phase 4 split; the test split has 2,598 hours. Naive persistence is deterministic.

### Test-set metrics (mean ± std over 5 seeds)

| Model | MAE (kW) | RMSE (kW) | R² | Train time (s) | Inference (s, full test set) |
|---|---|---|---|---|---|
| Naive Persistence | 0.456 | 0.702 | 0.429 | — | — |
| Random Forest | 0.391 ± 0.001 | 0.557 ± 0.001 | 0.641 ± 0.001 | 7.1 ± 0.3 | 0.038 |
| **Gradient Boosting** (HistGB) | **0.359 ± 0.000** | **0.524 ± 0.000** | **0.682 ± 0.000** | **0.8 ± 0.1** | **0.005** |
| LSTM | 0.372 ± 0.009 | 0.548 ± 0.007 | 0.653 ± 0.009 | 65.8 ± 17.0 | 0.013 |

CPU only. Gradient boosting shows zero seed variance because it trains without subsampling at this data size.

### Diebold-Mariano tests (squared-error loss, h = 1, two-sided)

A negative statistic means the LSTM has lower loss.

| Comparison | DM statistic per seed | Seeds where LSTM is significantly better (p < 0.05) |
|---|---|---|
| LSTM vs Naive | −11.5 to −12.3 | 5 / 5 |
| LSTM vs Random Forest | −0.42, −2.86, −0.54, −2.42, −3.11 | 3 / 5 |
| LSTM vs Gradient Boosting | +5.67, +3.07, +5.32, +3.91, +3.23 | 0 / 5 (GB significantly better on all 5) |

### What this changes

* **The single-run headline is not robust.** The original run (MAE 0.368) sits near the best of the LSTM's seed range; the 5-seed mean is 0.372. The LSTM beats Random Forest on average, but the gap is statistically significant on only 3 of 5 seeds.
* **A tuned gradient-boosting model beats the LSTM on every metric, on every seed, with statistical significance.** It also trains about 86× faster and runs inference about 2× faster.
* This supports the Phase 8 observation that, with about 12k hourly training windows and a 24-hour lookback, a non-sequential learner on windowed features captures most of the learnable structure. Explicit recurrence does not add accuracy here.
* The finding is honest and publishable. It weakens the "LSTM proposed method" framing, though. The paper should either be reframed as a comparative study or the LSTM needs further work, such as tuning or a different architecture, before it can be claimed as the best model.
