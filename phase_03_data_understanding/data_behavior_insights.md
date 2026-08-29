## Data Behavior Insights

Exploratory visualization reveals **consistent daily consumption cycles**, with elevated power usage during morning and evening hours. These patterns align with typical residential activity schedules.

### Key Observations

* **Appliance Spikes:** Sub-metering data shows short-duration spikes corresponding to high-power device operation, which contribute to sudden increases in total power demand.
* **Feature Correlation:** Analysis highlights a **strong relationship** between global active power and global intensity, validating their joint use as predictive features.
* **Contextual Data:** Voltage and reactive power provide secondary information that may improve model robustness against fluctuations.

### Conclusion

The observed behavior confirms the presence of both **short-term fluctuations** and **longer temporal dependencies**, reinforcing the suitability of sequence-based deep learning models for this problem.

