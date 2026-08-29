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

**Q: Your LSTM only beats Random Forest by ~5.6% MAE — is the extra complexity actually justified?**
A: This is answered directly rather than glossed over (`phase_08_results_analysis/results_summary.md`, `performance_analysis.md`): the LSTM's advantage is concentrated in steady-state hours, and both models underestimate peaks by a similar margin. For a production system where simplicity and training cost matter, Random Forest is a legitimate, nearly-as-accurate choice — the paper reports this honestly rather than overstating the deep-learning result.

**Q: Why Random Forest and not XGBoost/gradient boosting as the second baseline?**
A: Random Forest was chosen for its strong out-of-the-box compatibility with `scikit-learn` here without extra tuning, and because the comparison's purpose (Phase 5) is to test "does explicit temporal recurrence matter at all," not to find the best possible non-sequential model. Gradient boosting is a reasonable additional baseline for future work.
