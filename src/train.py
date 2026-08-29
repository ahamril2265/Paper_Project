"""
End-to-end training + evaluation script.
Implements Phase 5 (baselines), Phase 6 (LSTM), and Phase 7 (training/validation
strategy), and writes everything Phase 8 needs into ../results/.

Run from inside src/:  ..\.venv\Scripts\python.exe train.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import LSTMForecaster
from preprocess import SEED, build_dataset

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def naive_persistence(X, y, target_idx: int):
    # Prediction = last timestep's own value of the target feature.
    preds = X[:, -1, target_idx]
    return preds


def train_random_forest(X_train, y_train, X_test):
    n, t, f = X_train.shape
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train.reshape(n, t * f), y_train)
    preds = rf.predict(X_test.reshape(X_test.shape[0], t * f))
    return preds, rf


def train_lstm(data, epochs=60, patience=8, batch_size=64, lr=1e-3):
    device = torch.device("cpu")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    model = LSTMForecaster(n_features=data["n_features"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    n = X_train_t.shape[0]
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx].to(device), y_train_t[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:3d}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    return model, history


def main():
    t0 = time.time()
    print("Building dataset (load -> clean -> resample -> window)...")
    data = build_dataset("../datasets/household_power_consumption.csv")
    print(f"Dataset ready in {time.time()-t0:.1f}s")
    for split in ["train", "val", "test"]:
        print(f"  X_{split}: {data[f'X_{split}'].shape}  y_{split}: {data[f'y_{split}'].shape}")

    scaler = data["scaler"]
    target_idx = 0  # Global_active_power is FEATURE_COLS[0] in preprocess.py

    results = {}

    # --- Baseline 1: naive persistence ---
    naive_pred_scaled = naive_persistence(data["X_test"], data["y_test"], target_idx)
    naive_pred = scaler.inverse_transform_target(naive_pred_scaled)
    y_test_kw = scaler.inverse_transform_target(data["y_test"])
    results["naive_persistence"] = metrics(y_test_kw, naive_pred)
    print("Naive persistence:", results["naive_persistence"])

    # --- Baseline 2: Random Forest ---
    rf_pred_scaled, rf_model = train_random_forest(data["X_train"], data["y_train"], data["X_test"])
    rf_pred = scaler.inverse_transform_target(rf_pred_scaled)
    results["random_forest"] = metrics(y_test_kw, rf_pred)
    print("Random Forest:", results["random_forest"])

    # --- LSTM ---
    print("Training LSTM...")
    model, history = train_lstm(data)
    model.eval()
    with torch.no_grad():
        lstm_pred_scaled = model(torch.from_numpy(data["X_test"])).numpy()
    lstm_pred = scaler.inverse_transform_target(lstm_pred_scaled)
    results["lstm"] = metrics(y_test_kw, lstm_pred)
    print("LSTM:", results["lstm"])

    # --- Save everything Phase 8 needs ---
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    np.savez(
        RESULTS_DIR / "test_predictions.npz",
        y_true=y_test_kw,
        naive_pred=naive_pred,
        rf_pred=rf_pred,
        lstm_pred=lstm_pred,
    )

    torch.save(model.state_dict(), RESULTS_DIR / "lstm_model.pt")

    print(f"\nAll done in {time.time()-t0:.1f}s. Results written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
