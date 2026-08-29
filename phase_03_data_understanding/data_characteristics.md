## Data Characteristics

The dataset represents a **large-scale multivariate time series** consisting of over one million observations. Temporal analysis confirms a predominantly consistent **one-minute sampling interval**, enabling fine-grained modeling of short-term consumption patterns.

### Key Observed Characteristics

During dataset validation, the following key characteristics were identified:

* **Strong Temporal Dependency:** Clear diurnal (daily) usage patterns are visible throughout the series.
* **Non-linear Behavior:** Consumption is driven by intermittent appliance usage, creating complex patterns.
* **Variable Power vs. Stable Voltage:** High variability is present in active power, contrasted with relatively stable voltage levels.
* **Intermittent Sparsity:** Sub-metering features show sparse activity, reflecting the event-based nature of household appliance operation.

> [!IMPORTANT]
> These characteristics indicate that traditional linear regression models may be insufficient to capture the complex underlying dynamics of the data.

