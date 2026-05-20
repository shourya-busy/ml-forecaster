import os
from pathlib import Path

from forecaster.config.loader import load_settings


def test_default_config_loads(tmp_path: Path):
    # The shipped config under /Users/shouryagautam/ml/ml-forecaster/config
    config_dir = Path(__file__).resolve().parents[2] / "config"
    settings = load_settings(config_dir)
    assert "short" in settings.horizons
    assert "medium" in settings.horizons
    assert "long" in settings.horizons
    assert len(settings.algorithms.enabled) == 10
    assert set(settings.ranking.metrics) == {"mae", "rmse", "mape", "smape", "r2"}


def test_env_overrides():
    config_dir = Path(__file__).resolve().parents[2] / "config"
    os.environ["FORECASTER__TRAINING__LOOKBACK_DAYS"] = "7"
    try:
        s = load_settings(config_dir)
        assert s.training.lookback_days == 7
    finally:
        os.environ.pop("FORECASTER__TRAINING__LOOKBACK_DAYS", None)
