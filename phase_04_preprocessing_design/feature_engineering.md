## Feature Engineering

Beyond the seven raw electrical measurements, two lightweight temporal features are engineered to help the model represent daily periodicity, which Phase 3 identified as a dominant pattern in the data:

* **hour_sin, hour_cos:** The hour-of-day (0–23) is encoded cyclically as `sin(2π·hour/24)` and `cos(2π·hour/24)`, rather than as a raw integer or one-hot vector. A raw integer implies hour 23 and hour 0 are far apart, which is physically wrong for a diurnal cycle; the sin/cos pair preserves the correct circular distance and adds only two columns.

No additional lag features (e.g. manually shifted `t-1`, `t-2` columns) are engineered by hand, because the LSTM's recurrent state is explicitly responsible for learning temporal dependencies from the windowed sequence (Phase 4 → Section 6). Hand-crafted lag features would duplicate information already available to the model through the 24-hour input window and would mainly benefit a non-sequential baseline model instead.

### Final Feature Set (9 columns per timestep)

| Feature | Type | Source |
|---|---|---|
| Global_active_power | Raw (target + autoregressive input) | Smart meter |
| Global_reactive_power | Raw | Smart meter |
| Voltage | Raw | Smart meter |
| Global_intensity | Raw | Smart meter |
| Sub_metering_1 | Raw | Smart meter |
| Sub_metering_2 | Raw | Smart meter |
| Sub_metering_3 | Raw | Smart meter |
| hour_sin | Engineered | Timestamp |
| hour_cos | Engineered | Timestamp |
