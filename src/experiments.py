"""
Robustness experiments: repeats every model over several seeds, adds a
gradient-boosting baseline, records train/inference time, and runs
Diebold-Mariano tests of LSTM against each baseline.

Run from inside src/:  ../.venv/Scripts/python.exe experiments.py
"""

import json
import time

import numpy as np
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from preprocess import build_dataset
from train import RESULTS_DIR, metrics, naive_persistence, train_lstm

SEEDS = [0, 1, 2, 3, 4]


def diebold_mariano(y, pred_a, pred_b):
    """Two-sided DM test (h=1, squared-error loss). Negative stat => model A better."""
    d = (y - pred_a) ** 2 - (y - pred_b) ** 2
    stat = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def flat(X):
    return X.reshape(X.shape[0], -1)


def run_rf(data, seed):
    t0 = time.perf_counter()
    m = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
    m.fit(flat(data["X_train"]), data["y_train"])
    t1 = time.perf_counter()
    pred = m.predict(flat(data["X_test"]))
    return pred, t1 - t0, time.perf_counter() - t1


def run_gb(data, seed):
    t0 = time.perf_counter()
    m = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=6, early_stopping=True,
        random_state=seed,
    )
    # Early-stop on the chronological validation split, not a random holdout.
    m.fit(flat(data["X_train"]), data["y_train"],
          X_val=flat(data["X_val"]), y_val=data["y_val"])
    t1 = time.perf_counter()
    pred = m.predict(flat(data["X_test"]))
    return pred, t1 - t0, time.perf_counter() - t1


def run_lstm(data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    model, _ = train_lstm(data)
    t1 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(data["X_test"])).numpy()
    return pred, t1 - t0, time.perf_counter() - t1


def main():
    data = build_dataset("../datasets/household_power_consumption.csv")
    scaler = data["scaler"]
    y = scaler.inverse_transform_target(data["y_test"])

    runners = {"random_forest": run_rf, "gradient_boosting": run_gb, "lstm": run_lstm}
    per_seed = {name: [] for name in runners}
    preds = {name: [] for name in runners}

    naive = scaler.inverse_transform_target(naive_persistence(data["X_test"], None, 0))

    for seed in SEEDS:
        for name, fn in runners.items():
            print(f"seed {seed}  {name}...")
            p_scaled, t_train, t_inf = fn(data, seed)
            p = scaler.inverse_transform_target(p_scaled)
            preds[name].append(p)
            per_seed[name].append({**metrics(y, p), "seed": seed,
                                   "train_s": t_train, "infer_s": t_inf})
            print("   ", per_seed[name][-1])

    summary = {"naive_persistence": {"mean": metrics(y, naive)}}
    for name, runs in per_seed.items():
        summary[name] = {
            k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs], ddof=1))}
            for k in ["MAE", "RMSE", "R2", "train_s", "infer_s"]
        }

    dm = {}
    for base in ["naive_persistence", "random_forest", "gradient_boosting"]:
        per = []
        for i, seed in enumerate(SEEDS):
            b = naive if base == "naive_persistence" else preds[base][i]
            per.append(diebold_mariano(y, preds["lstm"][i], b))
        dm[f"lstm_vs_{base}"] = {
            "per_seed": [{"seed": s, "dm_stat": st, "p_value": p} for s, (st, p) in zip(SEEDS, per)],
            "seeds_significant_p<0.05_lstm_better": sum(1 for st, p in per if p < 0.05 and st < 0),
        }

    out = {"seeds": SEEDS, "n_test": int(len(y)), "summary": summary,
           "per_seed": per_seed, "diebold_mariano": dm}
    with open(RESULTS_DIR / "robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"summary": summary, "dm": {k: v["seeds_significant_p<0.05_lstm_better"] for k, v in dm.items()}}, indent=2))


if __name__ == "__main__":
    main()
