## Feature Description

The dataset consists of high-resolution smart meter measurements recorded from a residential household at minute-level intervals. Each feature represents a distinct electrical or appliance-level characteristic contributing to overall electricity consumption.

### Data Attributes

* **Datetime:** Unified timestamp created by merging date and time attributes, used to maintain chronological ordering.
* **Global_active_power:** Total active power consumed by the household. This variable reflects actual energy usage and serves as the **prediction target**.
* **Global_reactive_power:** Reactive power associated with inductive and capacitive loads, providing insight into power inefficiencies.
* **Voltage:** Electrical supply voltage, observed to remain relatively stable with minor fluctuations.
* **Global_intensity:** Current intensity drawn by the household, strongly correlated with active power consumption.
* **Sub_metering_1:** Energy consumption associated with kitchen appliances.
* **Sub_metering_2:** Energy consumption associated with laundry-related appliances.
* **Sub_metering_3:** Energy consumption associated with climate control and miscellaneous appliances.

> [!NOTE]
> Exploratory analysis confirms that all features contain numerically valid values within physically realistic ranges.

