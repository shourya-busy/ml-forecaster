import numpy as np
import pandas as pd

from forecaster.features.lag_features import build_lag_frame, recursive_forecast


def _series(n: int = 100) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(np.sin(np.arange(n) / 5.0), index=idx)


def test_build_lag_frame_shape():
    s = _series(60)
    X, y = build_lag_frame(s, lags=10)
    assert len(X) == len(y) == 50
    assert "lag_10" in X.columns
    assert "hour" in X.columns and "dow" in X.columns


def test_recursive_forecast_returns_correct_length():
    s = _series(60)
    step = s.index[1] - s.index[0]
    out = recursive_forecast(s, steps=12, lags=10, predict_one=lambda f: float(f[0]), step=step)
    assert out.shape == (12,)
