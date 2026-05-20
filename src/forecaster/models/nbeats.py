"""Lightweight N-BEATS forecaster (PyTorch).

Implements the generic-block N-BEATS architecture; small enough to train
on CPU and benefit from GPU when available.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


def _device():
    import torch  # local import
    want = os.environ.get("FORECASTER_USE_CUDA", "0") == "1"
    return torch.device("cuda" if (want and torch.cuda.is_available()) else "cpu")


@register("nbeats")
class NBeatsForecaster(BaseForecaster):
    def __init__(
        self,
        layer_size: int = 64,
        num_blocks: int = 2,
        epochs: int = 20,
        lags: int = 48,
        batch_size: int = 64,
        lr: float = 1e-3,
        forecast_horizon: int = 1,
        **hp: Any,
    ):
        super().__init__(
            layer_size=layer_size,
            num_blocks=num_blocks,
            epochs=epochs,
            lags=lags,
            batch_size=batch_size,
            lr=lr,
            **hp,
        )
        self.layer_size = int(layer_size)
        self.num_blocks = int(num_blocks)
        self.epochs = int(epochs)
        self.lags = int(lags)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.fh = int(forecast_horizon)  # we use 1-step then recurse

    def fit(self, series: pd.Series) -> None:
        import torch
        from torch import nn

        if len(series) < self.lags + 2:
            raise ValueError(f"nbeats: need at least {self.lags + 2} points, got {len(series)}")

        values = series.astype(float).to_numpy()
        self._mean = float(values.mean())
        self._std = float(values.std() or 1.0)
        norm = (values - self._mean) / self._std

        X = np.stack([norm[i: i + self.lags] for i in range(len(norm) - self.lags)])
        y = norm[self.lags:]
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        device = _device()

        class Block(nn.Module):
            def __init__(self, lookback: int, layer_size: int, horizon: int):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(lookback, layer_size), nn.ReLU(),
                    nn.Linear(layer_size, layer_size), nn.ReLU(),
                    nn.Linear(layer_size, layer_size), nn.ReLU(),
                    nn.Linear(layer_size, layer_size), nn.ReLU(),
                )
                self.theta_b = nn.Linear(layer_size, lookback)
                self.theta_f = nn.Linear(layer_size, horizon)

            def forward(self, x):
                h = self.fc(x)
                return self.theta_b(h), self.theta_f(h)

        class NBeats(nn.Module):
            def __init__(self, lookback: int, layer_size: int, num_blocks: int, horizon: int):
                super().__init__()
                self.blocks = nn.ModuleList([Block(lookback, layer_size, horizon) for _ in range(num_blocks)])

            def forward(self, x):
                residual = x
                forecast = 0.0
                for b in self.blocks:
                    backcast, f = b(residual)
                    residual = residual - backcast
                    forecast = forecast + f
                return forecast

        net = NBeats(self.lags, self.layer_size, self.num_blocks, self.fh).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        ds = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = net(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            preds = net(X_t.to(device)).cpu().numpy().ravel()
        self._residuals = (y - preds).astype(float) * self._std
        self._state = {k: v.cpu() for k, v in net.state_dict().items()}
        self._ctor = (self.lags, self.layer_size, self.num_blocks, self.fh)
        self._history = series.astype(float).copy()
        self._step = self._history.index[1] - self._history.index[0]
        self._fitted = True
        self._net = None  # type: ignore[assignment]

    def _build_net(self):
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self, lookback: int, layer_size: int, horizon: int):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(lookback, layer_size), nn.ReLU(),
                    nn.Linear(layer_size, layer_size), nn.ReLU(),
                    nn.Linear(layer_size, layer_size), nn.ReLU(),
                    nn.Linear(layer_size, layer_size), nn.ReLU(),
                )
                self.theta_b = nn.Linear(layer_size, lookback)
                self.theta_f = nn.Linear(layer_size, horizon)

            def forward(self, x):
                h = self.fc(x)
                return self.theta_b(h), self.theta_f(h)

        class NBeats(nn.Module):
            def __init__(self, lookback: int, layer_size: int, num_blocks: int, horizon: int):
                super().__init__()
                self.blocks = nn.ModuleList([Block(lookback, layer_size, horizon) for _ in range(num_blocks)])

            def forward(self, x):
                residual = x
                forecast = 0.0
                for b in self.blocks:
                    backcast, f = b(residual)
                    residual = residual - backcast
                    forecast = forecast + f
                return forecast

        net = NBeats(*self._ctor)
        net.load_state_dict(self._state)
        net.eval()
        return net.to(_device()), torch

    def predict(self, steps: int) -> np.ndarray:
        net, torch = self._build_net()
        history = self._history.to_numpy().astype(float).copy()
        out: list[float] = []
        for _ in range(steps):
            window = history[-self.lags:]
            norm = (window - self._mean) / self._std
            x = torch.tensor(norm, dtype=torch.float32).reshape(1, self.lags).to(_device())
            with torch.no_grad():
                y_hat = float(net(x).cpu().numpy().ravel()[0]) * self._std + self._mean
            out.append(y_hat)
            history = np.append(history, y_hat)
        return np.asarray(out, dtype=float)

    def lookback_required(self) -> int:
        return self.lags * 4
