## Dataset Comparison

| Criterion | Dataset A | Dataset B |
| --- | --- | --- |
| **Number of records** | 1,434 | 1,048,576 |
| **Temporal resolution** | Daily | Minute-level |
| **Data type** | Aggregated | High-resolution |
| **Power metrics** | Single consumption value | Multiple electrical parameters |
| **Appliance-level data** | Not available | Available |
| **Suitability for LSTM** | Low | High |
| **Use in literature** | Limited | Extensive |

### Summary Analysis

**Dataset A** provides a simplified representation of electricity consumption and is more suitable for basic regression or statistical analysis. However, its limited size and coarse temporal granularity restrict its ability to capture short-term load variations.

**Dataset B**, in contrast, offers detailed and continuous measurements, enabling the modeling of complex temporal and non-linear consumption patterns, making it ideal for deep learning applications.

