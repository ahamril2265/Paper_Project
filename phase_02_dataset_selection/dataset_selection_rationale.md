## Dataset Selection Rationale

**Dataset B** was selected as the primary dataset for this study due to its large scale, high temporal resolution, and comprehensive feature set. The selection is based on the following key factors:

* **Data Volume & Depth:** The presence of over one million minute-level observations provides sufficient historical depth for training deep learning models, such as **Long Short-Term Memory (LSTM)** networks, which require extensive sequential data to effectively learn temporal dependencies.
* **Multivariate Complexity:** Dataset B includes multiple electrical parameters describing both aggregate and appliance-level energy usage. This allows the model to capture interactions between different consumption sources and electrical characteristics, ultimately improving forecasting accuracy.
* **Academic Validity:** The dataset has been widely used as a benchmark in residential load forecasting research, supporting its suitability for a conference-level paper presentation.

### Exclusion Criteria

**Dataset A** was excluded from primary experimentation for the following reasons:

* **Insufficient Granularity:** Its daily aggregation is insufficient for the requirements of short-term forecasting.
* **Scale Constraints:** The limited size of the dataset does not justify the use of complex deep learning architectures.

