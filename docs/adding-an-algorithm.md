# Adding a new forecasting algorithm

Three steps.

## 1. Implement the protocol

```python
# src/forecaster/models/my_algo.py
from .base import BaseForecaster
from .registry import register
import numpy as np
import pandas as pd


@register("my_algo")
class MyAlgoForecaster(BaseForecaster):
    def __init__(self, my_hyperparam: int = 7, **hp):
        super().__init__(my_hyperparam=my_hyperparam, **hp)
        self.my_hyperparam = my_hyperparam

    def fit(self, series: pd.Series) -> None:
        # train; store anything you need to predict + intervals
        self._last_mean = float(series.iloc[-self.my_hyperparam:].mean())
        self._residuals = series.diff().dropna().to_numpy()
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self._last_mean)

    def lookback_required(self) -> int:
        return self.my_hyperparam
```

`predict_interval` is inherited from `BaseForecaster` and uses
residual-based normal-approximation bands — override it if your model
has a native interval estimator.

## 2. Register the import

Add to `src/forecaster/models/__init__.py`:

```python
from . import my_algo  # noqa: F401
```

## 3. Turn it on in config

`config/default.yaml`:

```yaml
algorithms:
  enabled: [..., my_algo]
  defaults:
    my_algo: {my_hyperparam: 10}
```

That's it. The next training run will train `my_algo` alongside the
others, score it, and rank it.
