import math

import numpy as np

from forecaster.evaluation.metrics import all_metrics, mae, mape, r2, rmse, smape


def test_mae_basic():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([1, 2, 3], [2, 3, 4]) == 1.0


def test_rmse_basic():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([1, 2, 3], [2, 3, 4]) == 1.0


def test_mape_skips_zeros():
    # All zeros → nan
    v = mape([0, 0], [1, 1])
    assert math.isnan(v)
    # Otherwise correct
    assert mape([100, 100], [110, 90]) == 10.0


def test_smape_basic():
    v = smape([100, 100], [100, 100])
    assert v == 0.0


def test_r2_perfect():
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_all_metrics_returns_5_keys():
    keys = all_metrics(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])).keys()
    assert set(keys) == {"mae", "rmse", "mape", "smape", "r2"}


def test_nan_safe():
    out = all_metrics(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, np.nan]))
    assert out["mae"] == 0.0  # only first row is clean
