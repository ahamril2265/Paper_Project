@echo off
REM Run script for Paper_Project — Predictive Analysis of Power Consumption
REM in Residential Buildings Using Deep Learning.
REM
REM Trains the naive persistence baseline, Random Forest baseline, and the
REM LSTM model, then generates the result plots. All outputs are written to
REM the results\ folder.

setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at %PY%
    echo Create it first with:  python -m venv .venv
    echo Then install deps:     .venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib seaborn torch
    exit /b 1
)

echo ============================================================
echo  Paper_Project — training pipeline
echo ============================================================

cd src

echo.
echo [1/2] Training models (naive persistence, Random Forest, LSTM)...
..\%PY% train.py
if errorlevel 1 (
    echo [ERROR] train.py failed.
    exit /b 1
)

echo.
echo [2/2] Generating plots...
..\%PY% make_plots.py
if errorlevel 1 (
    echo [ERROR] make_plots.py failed.
    exit /b 1
)

cd ..

echo.
echo ============================================================
echo  Done. Results written to results\
echo    - metrics.json               MAE / RMSE / R2 for all 3 models
echo    - training_history.json      per-epoch LSTM train/val loss
echo    - test_predictions.npz       raw predictions on the test split
echo    - lstm_model.pt              trained LSTM weights
echo    - loss_curve.png
echo    - actual_vs_predicted.png
echo    - baseline_comparison.png
echo ============================================================

endlocal
