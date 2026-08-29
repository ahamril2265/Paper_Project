## Presentation Slide Outline (ICCMI 2026, 10–12 min slot)

1. **Title slide** — paper title, authors, affiliation, presenting author.
2. **Motivation** — why short-term residential load forecasting matters (grid reliability, demand-side management); 1 chart: rising smart-meter adoption / diurnal demand curve.
3. **Problem statement** — forecasting horizon, target variable (global active power), why this is hard (non-linear, appliance-driven, Phase 1/3).
4. **Dataset** — UCI Individual Household Electric Power Consumption dataset; scale (1.04M minute-level rows), 9 attributes; one slide showing `Global_active_power` over a sample day (from the Phase 2.5 EDA notebook).
5. **Methodology** — one pipeline diagram: raw data → cleaning/resampling → scaling → windowing → model. Keep this to the diagram in `phase_04_preprocessing_design/preprocessing_pipeline.md`, no code.
6. **Model architecture** — the LSTM diagram from `phase_06_model_architecture_design/architecture_design.md`, plus a one-line justification (Phase 5).
7. **Baselines** — naive persistence + Random Forest, framed as "what has to be beaten," not an afterthought.
8. **Results** — MAE/RMSE/R² comparison table across all three models (Phase 8 `results_summary.md`); one actual-vs-predicted plot for the test period.
9. **Discussion** — where the model does well vs. where it struggles (peaks vs. steady-state), tied to `phase_08_results_analysis/limitations.md`.
10. **Conclusion & future work** — key finding, and 2–3 concrete extensions (see `paper_draft.md` → Conclusion).
11. **Thank you / Q&A**.

> [!TIP]
> Keep slides 5–6 diagram-only — the audience has 10 minutes; don't put the pipeline code or hyperparameter table on screen, point to the paper for that level of detail if asked.
