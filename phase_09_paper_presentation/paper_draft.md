## Predictive Analysis of Power Consumption in Residential Buildings: A Comparative Study of Deep Sequence Models and Tree Ensembles

*Draft full paper, reframed as a comparative study. The abstract submitted to ICCMI 2026 used the title "…Using Deep Learning" and presented the LSTM as the proposed model. Check with the organizers whether a title or abstract revision is allowed at camera-ready. The revised abstract is in `abstract.md`. Format the camera-ready copy with the official ICCMI 2026 author template, which isn't in this repository yet; download it from the conference site.*

### 1. Introduction
Accurate short-term forecasting of residential electricity consumption supports demand-side management and reliable grid operation (`phase_01_problem_definition/problem_definition.md`). Smart meters now record household load at minute resolution. As a result, deep sequence models such as LSTM have become a common default for this task, on the premise that household load is non-linear and time-dependent in ways that simpler learners cannot capture.

That premise is rarely tested directly. Many studies compare an LSTM against weak baselines only. This leaves open whether a well-configured non-sequential learner, given the same windowed history, would do just as well at a fraction of the cost.

This paper asks that question for single-household, next-hour forecasting. It compares four models of increasing complexity under one leakage-free chronological protocol:
* naive persistence;
* Random Forest;
* gradient boosting;
* a stacked LSTM.

The evaluation goes beyond single-run point metrics:
* every learned model is repeated over 5 seeds;
* differences are tested with the Diebold-Mariano test;
* training and inference cost are reported;
* errors are broken down by consumption regime (steady-state vs. peak).

**Contributions:**
1. **A controlled comparison.** All four models see the same 24-hour multivariate window, on the same split, scored by the same metrics.
2. **Seed-robust, significance-tested results.** These show that gradient boosting outperforms the LSTM on every seed (p < 0.01) while training about 86× faster.
3. **A regime-level error analysis.** Every model, deep or not, systematically underestimates peak demand. This locates the open problem in peak events rather than in model family.

### 2. Related Work

Residential and building-level load forecasting has progressed through three broad generations of methods. **Classical statistical models** — ARIMA, exponential smoothing, and regression-based approaches — were the earliest tools for this problem and remain common baselines; Alfares and Nazeeruddin's survey [2] catalogs this generation and documents its central weakness: these models assume linear, stationary structure that residential load, driven by discrete human and appliance behavior, does not reliably exhibit.

**Classical machine learning** methods (SVR, Random Forest, gradient boosting) were adopted next, treating forecasting as a windowed supervised-regression problem rather than a time-series problem per se. These models handle non-linearity better than ARIMA-class methods but, like the Random Forest baseline used in this paper, depend entirely on manually windowed/lagged features to represent temporal structure — they have no inherent mechanism for carrying state across time steps.

**Deep sequence models**, particularly LSTM [1] and its variants, were introduced to address this gap directly, by learning temporal dependencies end-to-end rather than through hand-engineered lag features. Marino et al. [3] were among the first to apply LSTM architectures — a standard stacked LSTM and a sequence-to-sequence (S2S) LSTM variant — to single-household smart-meter consumption data at one-minute and one-hour resolution (a benchmark consistent with, though not confirmed identical to, the dataset used in this work), finding that the standard LSTM performed well at hourly resolution but degraded at minute resolution, while the S2S variant remained robust at both — a resolution-sensitivity finding directly relevant to this paper's own choice to resample to hourly granularity (Phase 4). Kong et al. [4] extended LSTM-based residential load forecasting to the individual-household level more broadly (rather than aggregated/regional load, which dominated earlier smart-grid forecasting literature) and showed that per-household LSTM models captured household-specific consumption idiosyncrasies that aggregate models smoothed away. Kim and Cho [5] combined convolutional and LSTM layers (CNN-LSTM) for residential energy prediction, using the convolutional stage to extract short-range local features before the LSTM models longer-range temporal dependence, and reported improvements over LSTM-only and purely statistical baselines. Bouktif et al. [6] combined LSTM with explicit feature selection and a genetic-algorithm hyperparameter search for electric load forecasting, highlighting that LSTM performance is sensitive to lookback window and architecture choices — a sensitivity this paper's own limitations section (Phase 8) also surfaces, since only a single architecture configuration was evaluated here.

**Positioning of this work.** Much of the literature above reports a deep model's accuracy either in isolation or against a single weak baseline, most often a naive or ARIMA-class comparator. That makes it hard to tell how much of the reported gain comes from temporal modeling and how much from simply fitting a flexible model to windowed features. This paper isolates the contribution of recurrence. It gives two strong non-sequential learners, Random Forest and gradient boosting, exactly the same windowed input as the LSTM. It then tests the differences for statistical significance across seeds rather than reporting a single run. It also reports *where* errors arise, through a steady-state/peak regime split, which is largely absent from prior single-number reporting.

### 3. Dataset
UCI *Individual Household Electric Power Consumption* dataset — 1,048,575 minute-level records, 9 attributes, December 2006–2008, single household (`phase_02_dataset_selection/`, `phase_02_5_dataset_validation/eda_findings.md`). Selected over a coarser daily-aggregated alternative due to insufficient granularity for short-term forecasting (`dataset_selection_rationale.md`).

### 4. Methodology
#### 4.1 Preprocessing
Timestamp construction, time-interpolation of missing values (~0.39% per column), hourly resampling, cyclical hour-of-day encoding, chronological 70/15/15 split, train-fit Min-Max scaling, 24-hour sliding-window sequence construction (`phase_04_preprocessing_design/`).

#### 4.2 Compared Models
All learned models receive the identical input: a 24-hour window of 9 scaled features (7 electrical measurements plus a sine/cosine hour-of-day encoding). Tree models receive this window flattened to 216 values.
* **Naive persistence:** the forecast equals the previous hour's consumption. This is a floor that any useful model must beat.
* **Random Forest:** 200 trees, maximum depth 12.
* **Gradient boosting:** scikit-learn `HistGradientBoostingRegressor`, learning rate 0.05, maximum depth 6, up to 500 iterations. It uses early stopping on the chronological validation split.
* **LSTM:** two stacked layers (64 → 32 units), dropout 0.2 and a small dense head. It is trained with Adam on MSE loss, with early stopping (patience 8) on the validation loss (`phase_06_model_architecture_design/`).

#### 4.3 Evaluation Protocol
MAE, RMSE and R² are computed on the held-out chronological test split (2,598 hours) in the original kW units. The validation split is used only for early stopping. Each learned model is trained with 5 seeds (0–4), and results are reported as mean ± standard deviation.

Pairwise differences against the LSTM are tested with a two-sided Diebold-Mariano test (squared-error loss, horizon 1), separately for each seed. Wall-clock training and inference times are measured on CPU. Errors are also broken down by consumption regime, splitting the test set at the 75th percentile of actual load (1.674 kW) into steady-state hours (n = 1,948) and peak hours (n = 650).

### 5. Results

#### 5.1 Overall accuracy and cost

| Model | MAE (kW) | RMSE (kW) | R² | Train time (s) | Inference (ms, test set) |
|---|---|---|---|---|---|
| Naive Persistence | 0.456 | 0.702 | 0.429 | — | — |
| Random Forest | 0.391 ± 0.001 | 0.557 ± 0.001 | 0.641 ± 0.001 | 7.1 | 38 |
| **Gradient Boosting** | **0.359 ± 0.000** | **0.524 ± 0.000** | **0.682 ± 0.000** | **0.8** | **5** |
| LSTM | 0.372 ± 0.009 | 0.548 ± 0.007 | 0.653 ± 0.009 | 65.8 | 13 |

*Mean ± std over 5 seeds; CPU timing.*

**Gradient boosting is the most accurate model on all three metrics.** It is also the cheapest to train: about 86× faster than the LSTM. The LSTM ranks second. It improves on Random Forest by about 5% MAE and on naive persistence by about 18%.

The LSTM also has the largest seed-to-seed variance (MAE std 0.009, against ≤ 0.001 for the tree models). Its single-run result therefore depends noticeably on initialization.

#### 5.2 Statistical significance

| Comparison | DM statistic (5 seeds) | Seeds with p < 0.05 |
|---|---|---|
| LSTM vs Naive | −11.5 to −12.3 | 5/5 (LSTM better) |
| LSTM vs Random Forest | −0.42 to −3.11 | 3/5 (LSTM better) |
| LSTM vs Gradient Boosting | +3.07 to +5.67 | 5/5 (GB better) |

*A negative statistic favors the LSTM.*

The ranking Gradient Boosting > LSTM holds on every seed and is statistically significant. The LSTM's advantage over Random Forest is real on average, but it doesn't survive significance testing on 2 of the 5 seeds.

#### 5.3 Errors by consumption regime

| Model | Steady-state MAE (kW) | Peak MAE (kW) | Mean peak forecast (kW) |
|---|---|---|---|
| Naive Persistence | 0.328 | 0.838 | 2.06 |
| Random Forest | 0.312 | 0.624 | 2.00 |
| Gradient Boosting | **0.274** | 0.613 | 2.01 |
| LSTM | 0.288 | **0.608** | 2.03 |

*Seed-0 run for each learned model. The mean actual load in peak hours is 2.46 kW.*

Gradient boosting's overall lead comes from steady-state hours. At peaks, the three learned models are within 0.016 kW of each other, and all of them underestimate peak demand by about 0.45 kW. The hardest part of the problem, sharp appliance-driven spikes, is not solved by either model family. Models trained on MSE regress toward the mean and are never rewarded for anticipating rare high-load events.

Plots and raw metrics are in `results/`. Per-seed details are in `phase_08_results_analysis/robustness_analysis.md`.

### 6. Discussion
The results do not support the common assumption that a deep sequence model is the natural choice for single-household, next-hour load forecasting. There are three likely reasons:
1. **The data is small.** About 12k hourly training windows is a small dataset for a recurrent network. Boosted trees are known to be strong on small, tabular-shaped problems.
2. **The relevant history is short.** With a 24-hour lookback, the window covers most of the useful information, and flattening it loses little. Tree ensembles can therefore exploit the same lag structure the LSTM learns recurrently.
3. **Much of the signal is autocorrelation.** Naive persistence alone reaches R² = 0.43, so a large share of the predictable signal is short-range autocorrelation that every model captures.

In practice, gradient boosting is the better default for this setting. It is more accurate, deterministic across seeds, about 86× cheaper to train and easier to deploy on edge or meter hardware. The LSTM's advantage would need to come from conditions not tested here: longer histories, multi-household training data at much larger scale, or multi-step horizons.

### 7. Limitations
See `phase_08_results_analysis/limitations.md`. In summary:
* The study covers one household and a single next-hour horizon at hourly resolution.
* No exogenous variables are used (weather, holidays, occupancy).
* Neither the LSTM nor gradient boosting received an extensive hyperparameter search. Both used one reasonable configuration, and a broader search could narrow or widen the gap.
* The regime analysis uses a single seed.

### 8. Conclusion & Future Work
This study compared naive persistence, Random Forest, gradient boosting and a stacked LSTM for next-hour residential load forecasting on smart-meter data. All four models used one leakage-free chronological protocol, with multi-seed evaluation, Diebold-Mariano significance testing and cost measurement.

Gradient boosting was the most accurate model (MAE 0.359 kW, RMSE 0.524 kW, R² 0.682), significantly better than the LSTM on every seed, and about 86× faster to train. The LSTM came second, ahead of Random Forest on average but not consistently significantly. All learned models systematically underestimated peak demand by about 0.45 kW. This shows that the main remaining error lies in peak events, not in the choice between deep and tree-based models.

For practitioners forecasting a single household at this horizon, a gradient-boosted model on windowed features is the stronger and cheaper baseline. Any deep model should be justified against it rather than against naive or ARIMA-class comparators.

**Future work:**
* **Peak-aware training:** quantile/pinball loss or an auxiliary peak-classification task.
* **Larger-scale comparison:** multi-household datasets, where deep models may have the data volume they need.
* **Longer horizons and lookbacks:** multi-step horizons and longer windows.
* **Exogenous features:** weather data. The rejected Phase 2 dataset contains a compatible weather-linked series.
* **Fair tuning:** hyperparameter search applied equally to all model families.

### References
1. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 9(8), pp. 1735–1780, 1997.
2. H. K. Alfares and M. Nazeeruddin, "Electric load forecasting: Literature survey and classification of methods", *International Journal of Systems Science*, 33(1), pp. 23–34, 2002.
3. D. L. Marino, K. Amarasinghe, and M. Manic, "Building Energy Load Forecasting using Deep Neural Networks", *IECON 2016 — 42nd Annual Conference of the IEEE Industrial Electronics Society*, pp. 7046–7051, 2016. (arXiv:1610.09460)
4. W. Kong, Z. Y. Dong, Y. Jia, D. J. Hill, Y. Xu, and Y. Zhang, "Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network", *IEEE Transactions on Smart Grid*, 10(1), pp. 841–851, 2019.
5. T.-Y. Kim and S.-B. Cho, "Predicting Residential Energy Consumption Using CNN-LSTM Neural Networks", *Energy*, 182, pp. 72–81, 2019. (DOI: 10.1016/j.energy.2019.05.230)
6. S. Bouktif, A. Fiaz, A. Ouni, and M. A. Serhani, "Optimal Deep Learning LSTM Model for Electric Load Forecasting Using Feature Selection and Genetic Algorithm: Comparison with Machine Learning Approaches", *Energies*, 11(7), 1636, 2018. (DOI: 10.3390/en11071636)
7. G. Hebrail and A. Berard, "Individual Household Electric Power Consumption" [Dataset], UCI Machine Learning Repository, 2006. (DOI: 10.24432/C58K54)

*Note: [3]'s page range and arXiv ID, [5]'s and [6]'s DOIs, and [7]'s correct author/year/DOI were confirmed via web search on 2026-11-12. [4] was independently confirmed by two bibliographic sources (DBLP, CityU Scholars) but no DOI surfaced — check IEEE Xplore directly before camera-ready. The dataset match claimed for [3] (single-household, one-minute/one-hour resolution) is consistent with but not confirmed identical to the dataset used in this work; the in-text wording above reflects that uncertainty.*
