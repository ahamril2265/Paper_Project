# Paper_Project

**Predictive Analysis of Power Consumption in Residential Buildings Using Deep Learning**

Ahamed Rilwan Mohaaideen H, Guhankarthik A G, Marudhachalam R — Kumaraguru College of Technology.
Abstract submitted to ICCMI 2026 (see `Abstracts/`).

**Status: All 10 phases complete** — problem definition through full paper draft (see `Phases.txt`).

A comparative study of short-term (next-hour) residential load forecasting from smart-meter data. It compares naive persistence, Random Forest, gradient boosting and a stacked LSTM on the UCI *Individual Household Electric Power Consumption* dataset (1,048,575 minute-level records).

## Final Results (test set, mean ± std over 5 seeds)

| Model | MAE (kW) | RMSE (kW) | R² | Train time (s) |
|---|---|---|---|---|
| Naive Persistence | 0.456 | 0.702 | 0.429 | — |
| Random Forest | 0.391 ± 0.001 | 0.557 ± 0.001 | 0.641 ± 0.001 | 7.1 |
| **Gradient Boosting** | **0.359 ± 0.000** | **0.524 ± 0.000** | **0.682 ± 0.000** | **0.8** |
| LSTM | 0.372 ± 0.009 | 0.548 ± 0.007 | 0.653 ± 0.009 | 65.8 |

Gradient boosting is significantly better than the LSTM on all 5 seeds (Diebold-Mariano test, p < 0.01) and trains about 86× faster. All learned models underestimate peak demand by about 0.45 kW. Details are in `phase_08_results_analysis/robustness_analysis.md`.

## Project Structure

The work is organized as a 10-phase research pipeline, one folder per phase:

| Phase | Folder | Contents |
|---|---|---|
| 1 | `phase_01_problem_definition/` | Problem statement, objectives, scope & constraints |
| 2 | `phase_02_dataset_selection/` | Dataset comparison, selection rationale, metadata |
| 2.5 | `phase_02_5_dataset_validation/` | EDA notebook + findings |
| 3 | `phase_03_data_understanding/` | Feature descriptions, target variable justification |
| 4 | `phase_04_preprocessing_design/` | Cleaning, resampling, feature engineering, splitting, windowing |
| 5 | `phase_05_model_selection_design/` | Baseline models, LSTM justification, comparison framework |
| 6 | `phase_06_model_architecture_design/` | LSTM architecture, hyperparameters, training config |
| 7 | `phase_07_training_evaluation_design/` | Evaluation metrics, validation plan, training strategy |
| 8 | `phase_08_results_analysis/` | Final metrics, performance analysis, limitations |
| 9 | `phase_09_paper_presentation/` | Abstract, full paper draft, slide outline, Q&A prep |

## Code

* `src/preprocess.py` — full Phase 4 pipeline (load → clean → resample → engineer → split → scale → window)
* `src/models.py` — the Phase 6 `LSTMForecaster` (PyTorch)
* `src/train.py` — trains the naive persistence baseline, Random Forest, and LSTM; writes all results to `results/`
* `src/make_plots.py` — generates the loss curve, actual-vs-predicted, and baseline-comparison plots from `results/`
* `src/experiments.py` — 5-seed runs of Random Forest, gradient boosting, and LSTM with timing and Diebold-Mariano tests; writes `results/robustness.json`

## Reproducing the Results

```bash
cd src
../.venv/Scripts/python.exe train.py       # trains all 3 models, writes results/
../.venv/Scripts/python.exe make_plots.py  # generates plots from results/
../.venv/Scripts/python.exe experiments.py # multi-seed comparison + significance tests
```

`results/metrics.json` holds the MAE/RMSE/R² for all three models; see `phase_08_results_analysis/results_summary.md` for the write-up.

## Data

* `datasets/household_power_consumption.csv` — primary dataset (selected, Phase 2)
* `datasets/electricity_consumption_based_weather_dataset.csv` — alternative dataset considered and rejected (Phase 2, insufficient granularity); flagged in `paper_draft.md` as a candidate weather feature source for future work.
