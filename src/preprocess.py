"""
Preprocessing pipeline for the residential power-consumption forecasting task.
Implements Phase 4 exactly as documented in phase_04_preprocessing_design/.

Pipeline: load -> merge timestamp -> interpolate missing values -> resample
to hourly mean -> engineer cyclical hour features -> chronological split ->
fit-on-train scaling -> sliding-window sequence construction.
"""

import numpy as np
import pandas as pd

SEED = 42
LOOKBACK_HOURS = 24
TARGET_COL = "Global_active_power"
RAW_FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
ENGINEERED_COLS = ["hour_sin", "hour_cos"]
FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_COLS


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=",",
        na_values=["?", "NA", ""],
        low_memory=False,
    )
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.drop(columns=["Date", "Time"]).sort_values("Datetime").reset_index(drop=True)
    return df


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("Datetime")
    for col in RAW_FEATURE_COLS:
        df[col] = df[col].interpolate(method="time").ffill().bfill()
    hourly = df[RAW_FEATURE_COLS].resample("1h").mean()
    hourly = hourly.interpolate(method="time").ffill().bfill()
    return hourly


def engineer_features(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hour = hourly.index.hour.values.astype(float)
    hourly["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    hourly["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return hourly[FEATURE_COLS]


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


class MinMaxScalerFitOnTrain:
    """Column-wise min-max scaler, fit on train split only (Phase 4)."""

    def fit(self, df: pd.DataFrame):
        self.min_ = df.min()
        self.max_ = df.max()
        self.range_ = (self.max_ - self.min_).replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.min_) / self.range_

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        col_min = self.min_[TARGET_COL]
        col_range = self.range_[TARGET_COL]
        return values * col_range + col_min


def make_windows(df: pd.DataFrame, lookback: int = LOOKBACK_HOURS):
    """Turn a scaled (T, F) frame into (N, lookback, F) X and (N,) y arrays."""
    values = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    target_idx = FEATURE_COLS.index(TARGET_COL)
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i])
        y.append(values[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset(csv_path: str, lookback: int = LOOKBACK_HOURS):
    """Full pipeline entrypoint. Returns dict of arrays + fitted scaler."""
    raw = load_raw(csv_path)
    hourly = clean_and_resample(raw)
    featured = engineer_features(hourly)

    train_df, val_df, test_df = chronological_split(featured)

    scaler = MinMaxScalerFitOnTrain().fit(train_df)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = make_windows(train_scaled, lookback)
    X_val, y_val = make_windows(val_scaled, lookback)
    X_test, y_test = make_windows(test_scaled, lookback)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "hourly_index": featured.index,
        "n_features": len(FEATURE_COLS),
    }


if __name__ == "__main__":
    data = build_dataset("../datasets/household_power_consumption.csv")
    print("Train:", data["X_train"].shape, data["y_train"].shape)
    print("Val:  ", data["X_val"].shape, data["y_val"].shape)
    print("Test: ", data["X_test"].shape, data["y_test"].shape)
