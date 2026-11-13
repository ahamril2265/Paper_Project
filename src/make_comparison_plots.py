"""Figures for the four-model comparison, from results/robustness.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("../results")
r = json.loads((R / "robustness.json").read_text())
s = r["summary"]

names = ["Naive\npersistence", "Random\nforest", "Gradient\nboosting", "LSTM"]
keys = ["naive_persistence", "random_forest", "gradient_boosting", "lstm"]
mae = [s[k]["mean"]["MAE"] if k == "naive_persistence" else s[k]["MAE"]["mean"] for k in keys]
err = [0 if k == "naive_persistence" else s[k]["MAE"]["std"] for k in keys]
colors = ["#9e9e9e", "#6a8caf", "#2e7d32", "#c26a2e"]

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axes[0]
ax.bar(names, mae, yerr=err, capsize=3, color=colors)
for i, v in enumerate(mae):
    ax.text(i, v + err[i] + 0.008, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("Test MAE (kW)")
ax.set_ylim(0, 0.52)
ax.set_title("(a) Accuracy (mean ± std, 5 seeds)", fontsize=10)

ax = axes[1]
t = [s[k]["train_s"]["mean"] for k in keys[1:]]
ax.bar(names[1:], t, color=colors[1:])
ax.set_yscale("log")
for i, v in enumerate(t):
    ax.text(i, v * 1.15, f"{v:.1f} s", ha="center", fontsize=8)
ax.set_ylabel("Training time (s, log scale)")
ax.set_ylim(0.3, 300)
ax.set_title("(b) Training cost (CPU)", fontsize=10)
fig.tight_layout()
fig.savefig(R / "fig_model_comparison.png", dpi=300)

steady = [0.328, 0.312, 0.274, 0.288]
peak = [0.838, 0.624, 0.613, 0.608]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.bar(x - 0.2, steady, 0.4, label="Steady-state hours", color="#6a8caf")
ax.bar(x + 0.2, peak, 0.4, label="Peak hours", color="#c26a2e")
ax.set_xticks(x, names)
ax.set_ylabel("Test MAE (kW)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(R / "fig_regime_errors.png", dpi=300)

p = np.load(R / "test_predictions.npz")
fig, ax = plt.subplots(figsize=(7.0, 2.3))
ax.plot(p["y_true"][:200], lw=1.1, color="#1f3b5c", label="Actual")
ax.plot(p["lstm_pred"][:200], lw=1.1, color="#c26a2e", label="LSTM forecast")
ax.set_xlabel("Test-set hour")
ax.set_ylabel("Global active power (kW)")
ax.set_xlim(0, 199)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(R / "fig_actual_vs_predicted.png", dpi=300)
print("ok")
