## Revised Abstract (comparative-study framing — for the full paper)

**Predictive Analysis of Power Consumption in Residential Buildings: A Comparative Study of Deep Sequence Models and Tree Ensembles**

Accurate short-term forecasting of residential electricity consumption supports demand-side management and reliable grid operation. Deep sequence models such as Long Short-Term Memory (LSTM) networks are widely used for this task, but they are often evaluated only against weak baselines. This paper presents a controlled comparison of naive persistence, Random Forest, gradient boosting and a stacked LSTM for next-hour load forecasting. It uses over one million minute-level smart-meter observations from a single household, resampled to hourly resolution. All models receive an identical 24-hour multivariate window and are evaluated on a leakage-free chronological split. Each learned model is trained with five random seeds. Differences are assessed with Diebold-Mariano tests, and training cost and errors by consumption regime are reported. Gradient boosting achieved the best accuracy (MAE 0.359 kW, RMSE 0.524 kW, R² 0.682). It was significantly better than the LSTM on every seed (p < 0.01) and trained about 86 times faster. The LSTM (MAE 0.372 kW, R² 0.653) outperformed Random Forest on average, but not consistently significantly. All learned models underestimated peak demand by about 0.45 kW. This indicates that peak events, rather than model family, are the main remaining source of error. The results suggest that gradient-boosted models on windowed features are a strong, low-cost default for household-level forecasting, and that deep sequence models should be justified against them.

**Keywords:** Residential Load Forecasting, Smart Meter Data, LSTM, Gradient Boosting, Comparative Study, Diebold-Mariano Test.

---

## Abstract (as submitted)

*(Final text as submitted to ICCMI 2026 — see `Abstracts/Predictive Analysis of Power Consumption in Residential Buildings.docx` and `Abstracts/Abstract_submission.pdf` for the formatted conference version.)*

**Predictive Analysis of Power Consumption in Residential Buildings Using Deep Learning**

Ahamed Rilwan Mohaaideen H¹, Guhankarthik A G², Marudhachalam R³

Accurate short-term forecasting of residential electricity consumption is essential for efficient energy management, demand response planning, and reliable operation of modern power systems. The increasing deployment of smart meters has enabled the collection of high-resolution electrical consumption data, creating opportunities for advanced data-driven prediction techniques. This paper addresses the problem of short-term residential load forecasting using supervised deep learning models applied to smart meter data. The study employs a large-scale multivariate time-series dataset comprising over one million minute-level observations, including global active power, global reactive power, voltage, current intensity, and appliance-level sub-metering measurements. A systematic preprocessing pipeline is designed to handle missing values, normalize features, and transform the time-series data into supervised learning sequences. A Long Short-Term Memory (LSTM) neural network is then developed to capture temporal dependencies and non-linear relationships inherent in residential electricity consumption patterns. The proposed model is trained to predict global active power consumption over short-term horizons and evaluated using Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and the coefficient of determination (R²). The results demonstrate that the LSTM-based approach achieves accurate short-term power consumption predictions and effectively models residential load dynamics. The findings confirm the suitability of deep learning techniques for smart meter data analysis and highlight their potential application in intelligent energy management systems and demand-side planning for residential buildings.

**Keywords:** Residential Load Forecasting, Smart Meter Data, LSTM, Time-Series Prediction, Energy Management.
