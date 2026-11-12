## Anticipated Q&A

**Q: Why LSTM instead of a Transformer/attention-based model?**
A: The dataset, after the hourly resampling used here (Phase 4), is moderate in size (~12k training sequences). Transformer-style models typically need substantially more data to outperform recurrent architectures at this scale, and would add complexity without a clear accuracy benefit here. This is noted as a direction for future work if more households/years of data are incorporated.

**Q: Why resample to hourly instead of using the full minute-level resolution?**
A: Documented explicitly in Phase 4 — it is a deliberate choice tied to the Phase 1 hardware-constraint assumption, and because minute-level appliance-switching noise is not the signal of interest for the stated demand-management use case. The forecasting horizon (next hour) is still short-term.

**Q: How do you know the model isn't just learning the daily pattern trivially (e.g. copying yesterday same-hour)?**
A: This is exactly why the naive persistence baseline exists (Phase 5) — it's evaluated on the identical test split and metrics, so any claimed LSTM improvement is measured against that possibility directly rather than assumed.

**Q: Does this generalize to other households or climates?**
A: No — this is explicitly out of scope (Phase 1 → `scope_and_constraints.md`). The dataset is a single household; generalization across households/regions is future work, not a claim made here.

**Q: What happens to accuracy during unusual events (holidays, appliance failure, etc.)?**
A: Not explicitly modeled — noted as a limitation in Phase 8 (`limitations.md`). The model is a statistical pattern-learner, not an anomaly-aware system.

**Q: Your title talks about deep learning, but gradient boosting won. Isn't that a negative result?**
A: It's the central finding, not a failure. The study tests whether deep sequence models are justified for single-household, next-hour forecasting when compared against strong baselines on identical inputs. Under that test, they aren't. Gradient boosting is more accurate on every seed (Diebold-Mariano p < 0.01) and about 86× cheaper to train. That is useful to practitioners, and it is often missing from LSTM papers that compare only against naive or ARIMA baselines.

**Q: Maybe the LSTM just wasn't tuned enough?**
A: That's possible, and it's stated as a limitation. Neither the LSTM nor gradient boosting received an extensive search; both used one reasonable configuration. The LSTM also has the largest seed variance (MAE std 0.009 vs ≤ 0.001), so its results are less stable regardless of tuning. A fair future study would tune all model families with equal budget.

**Q: When *would* you expect the LSTM to win?**
A: With much more data (multi-household training), longer lookback windows where flattening the window becomes impractical for trees, or multi-step horizons. None of these was tested here.

**Q: Why does no model handle peaks well?**
A: All learned models undershoot peak hours by about 0.45 kW. Training on MSE rewards predictions near the mean, and spikes driven by appliances are short and intermittent. Peak-aware losses such as quantile/pinball loss, or a separate peak classifier, are the natural next step.
