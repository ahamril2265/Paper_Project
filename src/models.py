"""
Model definitions. Implements Phase 6 (architecture_design.md) exactly.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Two-layer stacked LSTM -> dense head -> scalar regression output.

    Input:  (batch, lookback, n_features)
    Output: (batch,) predicted next-hour Global_active_power (scaled [0,1])
    """

    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, (h_n, _) = self.lstm2(out)
        out = self.dropout2(h_n[-1])  # last layer's final hidden state
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)
