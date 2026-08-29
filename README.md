# Paper_Project

**Predictive Analysis of Power Consumption in Residential Buildings Using Deep Learning**

Ahamed Rilwan Mohaaideen H, Guhankarthik A G, Marudhachalam R — Kumaraguru College of Technology.
Abstract submitted to ICCMI 2026 (see `Abstracts/`).

**Status: All 10 phases complete** — problem definition through full paper draft (see `Phases.txt`).

Short-term (next-hour) forecasting of residential electricity demand from smart-meter data, using an LSTM trained against two baselines (naive persistence, Random Forest), on the UCI *Individual Household Electric Power Consumption* dataset (1,048,575 minute-level records).

## Final Results (test set)

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Naive Persistence | 0.456 | 0.702 | 0.429 |
| Random Forest | 0.390 | 0.556 | 0.641 |
| **LSTM** | **0.368** | **0.545** | **0.656** |

The LSTM outperforms both baselines on every metric. Full breakdown (including a steady-state vs. peak-hour regime analysis) in `phase_08_results_analysis/`.

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

## Reproducing the Results

```bash
cd src
../.venv/Scripts/python.exe train.py       # trains all 3 models, writes results/
../.venv/Scripts/python.exe make_plots.py  # generates plots from results/
```

`results/metrics.json` holds the MAE/RMSE/R² for all three models; see `phase_08_results_analysis/results_summary.md` for the write-up.

## Data

* `datasets/household_power_consumption.csv` — primary dataset (selected, Phase 2)
* `datasets/electricity_consumption_based_weather_dataset.csv` — alternative dataset considered and rejected (Phase 2, insufficient granularity); flagged in `paper_draft.md` as a candidate weather feature source for future work.
