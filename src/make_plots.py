"""
Generates the plots Phase 8 references, from artifacts written by train.py.
Run after train.py:  ..\.venv\Scripts\python.exe make_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("../results")


def plot_loss_curve():
    with open(RESULTS_DIR / "training_history.json") as f:
        history = json.load(f)
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled)")
    plt.title("LSTM Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close()


def plot_predictions():
    data = np.load(RESULTS_DIR / "test_predictions.npz")
    y_true, lstm_pred = data["y_true"], data["lstm_pred"]

    n = min(200, len(y_true))  # first ~200 test hours for readability
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:n], label="Actual", linewidth=1.5)
    plt.plot(lstm_pred[:n], label="LSTM Predicted", linewidth=1.2, alpha=0.85)
    plt.xlabel("Test-set hour index")
    plt.ylabel("Global Active Power (kW)")
    plt.title("Actual vs. Predicted Global Active Power (first 200 test hours)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_baseline_comparison():
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    models = ["naive_persistence", "random_forest", "lstm"]
    labels = ["Naive Persistence", "Random Forest", "LSTM"]
    mae = [metrics[m]["MAE"] for m in models]
    rmse = [metrics[m]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, mae, width, label="MAE")
    plt.bar(x + width / 2, rmse, width, label="RMSE")
    plt.xticks(x, labels)
    plt.ylabel("kW")
    plt.title("Baseline vs. LSTM — Test Set Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_loss_curve()
    plot_predictions()
    plot_baseline_comparison()
    print("Plots written to", RESULTS_DIR.resolve())
