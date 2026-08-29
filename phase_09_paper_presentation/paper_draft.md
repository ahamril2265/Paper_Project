## Predictive Analysis of Power Consumption in Residential Buildings Using Deep Learning

*Draft full paper — expands the submitted conference abstract into complete sections. Formatting for the final camera-ready copy should follow `Backup/Paper/IJRDET_PAPER_TEMPLATE.doc`-style formatting if this venue requires it, or the ICCMI 2026 template otherwise.*

### 1. Introduction
See `phase_01_problem_definition/problem_definition.md` — residential electricity demand forecasting, motivation (grid reliability, demand-side management), and the core challenge of non-linear, time-dependent household load.

### 2. Related Work
*(Not previously drafted in this repository — to be expanded with a short literature review before final submission.)* At minimum, cite: Hochreiter & Schmidhuber (1997) on LSTM; Alfares & Nazeeruddin (2002) on load forecasting methods survey — both already referenced in the submitted abstract.

### 3. Dataset
UCI *Individual Household Electric Power Consumption* dataset — 1,048,575 minute-level records, 9 attributes, December 2006–2008, single household (`phase_02_dataset_selection/`, `phase_02_5_dataset_validation/eda_findings.md`). Selected over a coarser daily-aggregated alternative due to insufficient granularity for short-term forecasting (`dataset_selection_rationale.md`).

### 4. Methodology
#### 4.1 Preprocessing
Timestamp construction, time-interpolation of missing values (~0.39% per column), hourly resampling, cyclical hour-of-day encoding, chronological 70/15/15 split, train-fit Min-Max scaling, 24-hour sliding-window sequence construction (`phase_04_preprocessing_design/`).

#### 4.2 Baselines
Naive persistence and Random Forest, trained and evaluated under identical conditions to the LSTM (`phase_05_model_selection_design/baseline_models.md`).

#### 4.3 Model Architecture
Two-layer stacked LSTM (64 → 32 units) with dropout (0.2) and a small dense head, trained with Adam / MSE loss and early stopping on validation loss (`phase_06_model_architecture_design/`).

#### 4.4 Evaluation Protocol
MAE, RMSE, R² computed once on the held-out chronological test split, on the original kW scale (`phase_07_training_evaluation_design/`).

### 5. Results

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Naive Persistence | 0.456 | 0.702 | 0.429 |
| Random Forest | 0.390 | 0.556 | 0.641 |
| **LSTM (proposed)** | **0.368** | **0.545** | **0.656** |

The LSTM achieves the lowest error and highest R² of the three models, improving MAE by 19.3% over naive persistence and 5.6% over Random Forest. The improvement over Random Forest is modest rather than dramatic — at this data scale, a well-tuned non-sequential model already captures most of the learnable structure in the windowed features. Splitting the test set by consumption regime shows the LSTM's advantage is concentrated in steady-state hours (12% lower MAE than Random Forest) rather than peak hours (2.6% lower), and all models systematically underestimate peak consumption (mean actual 2.46 kW vs. LSTM mean prediction 2.03 kW during peak hours). Full breakdown in `phase_08_results_analysis/performance_analysis.md`.

See `phase_08_results_analysis/results_summary.md` and `performance_analysis.md` for the full breakdown, and `results/` for the plots and raw metrics referenced here.

### 6. Limitations
See `phase_08_results_analysis/limitations.md` — single-household scope, no explicit anomaly/holiday modeling, hourly (not minute-level) forecasting granularity.

### 7. Conclusion & Future Work

This study developed an LSTM-based short-term forecasting model for residential electricity consumption using large-scale smart-meter data, and evaluated it against a naive persistence baseline and a Random Forest baseline under an identical, leakage-free chronological evaluation protocol. The LSTM achieved the best test-set performance (MAE 0.368 kW, RMSE 0.545 kW, R² 0.656), confirming that deep sequence models can capture the non-linear, temporally-dependent structure of household load identified during exploratory analysis. The improvement over a well-tuned non-sequential baseline was real but modest, and all models — including the proposed LSTM — underestimate sharp, appliance-driven consumption peaks, indicating that the recurrent architecture alone does not fully resolve the hardest part of the forecasting problem.

Future extensions: multi-household generalization, gradient-boosting baseline, attention-based sequence models at larger data scale, and incorporating exogenous weather data (the rejected Phase 2 dataset already contains a compatible weather-linked series that could be joined in a follow-up study).

### References
1. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 9(8), pp. 1735–1780, 1997.
2. H. K. Alfares and M. Nazeeruddin, "Electric load forecasting: Literature survey and classification of methods", *International Journal of Systems Science*, 33(1), pp. 23–34, 2002.
3. Individual Household Electric Power Consumption Dataset, UCI Machine Learning Repository, 2012.
