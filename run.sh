#!/usr/bin/env bash
# Run script for Paper_Project — Predictive Analysis of Power Consumption
# in Residential Buildings Using Deep Learning.
#
# Trains the naive persistence baseline, Random Forest baseline, and the
# LSTM model, then generates the result plots. All outputs are written to
# the results/ folder.

set -e
cd "$(dirname "$0")"

PY=".venv/Scripts/python.exe"

if [ ! -f "$PY" ]; then
    echo "[ERROR] Virtual environment not found at $PY"
    echo "Create it first with:  python -m venv .venv"
    echo "Then install deps:     .venv/Scripts/python.exe -m pip install numpy pandas scikit-learn matplotlib seaborn torch"
    exit 1
fi

echo "============================================================"
echo " Paper_Project — training pipeline"
echo "============================================================"

cd src

echo
echo "[1/2] Training models (naive persistence, Random Forest, LSTM)..."
"../$PY" train.py

echo
echo "[2/2] Generating plots..."
"../$PY" make_plots.py

cd ..

echo
echo "============================================================"
echo " Done. Results written to results/"
echo "   - metrics.json               MAE / RMSE / R2 for all 3 models"
echo "   - training_history.json      per-epoch LSTM train/val loss"
echo "   - test_predictions.npz       raw predictions on the test split"
echo "   - lstm_model.pt              trained LSTM weights"
echo "   - loss_curve.png"
echo "   - actual_vs_predicted.png"
echo "   - baseline_comparison.png"
echo "============================================================"
