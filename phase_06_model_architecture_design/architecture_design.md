## LSTM Architecture

```
Input:  (batch, 24, 9)                # 24-hour window, 9 features
  │
  ▼
LSTM layer 1:  64 units, return_sequences=True
  │
  ▼
Dropout: 0.2
  │
  ▼
LSTM layer 2:  32 units, return_sequences=False
  │
  ▼
Dropout: 0.2
  │
  ▼
Dense: 16 units, ReLU
  │
  ▼
Dense: 1 unit, linear                 # predicted Global_active_power (t+1)
```

### Design Rationale

* **Two stacked LSTM layers** (64 → 32 units): the first layer learns short, local temporal patterns across the 24-hour window; the second compresses that into a lower-dimensional temporal summary. Going deeper than two layers was avoided given the moderate dataset size (~12k training sequences after resampling, Phase 4) — additional depth would risk overfitting rather than improving accuracy.
* **Dropout (0.2) after each LSTM layer:** standard regularization against overfitting, given the model has no external validation signal beyond the held-out split.
* **A small dense head (16 → 1)** rather than mapping the LSTM output straight to a scalar: gives the model a little extra non-linear capacity to combine the temporal summary before the final regression, without meaningfully increasing parameter count.
* **Linear output activation:** this is a regression task (predicting a continuous power value in normalized [0,1] space), not classification, so no output non-linearity is applied.

Implementation: [`src/models.py`](../src/models.py) (`LSTMForecaster` class, PyTorch).
