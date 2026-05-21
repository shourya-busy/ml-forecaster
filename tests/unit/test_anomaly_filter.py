"""Tests for the Isolation-Forest outlier-cleaning preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.config.schema import AnomalyFilter
from forecaster.training.preprocess import remove_anomalies


def _clean_series(n: int = 500) -> pd.Series:
    idx = pd.date_range("2026-05-21", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    values = 50 + np.sin(np.arange(n) / 20.0) * 5 + rng.normal(0, 0.5, n)
    return pd.Series(values, index=idx)


def _series_with_spikes(n: int = 500, n_spikes: int = 20) -> pd.Series:
    s = _clean_series(n).copy()
    rng = np.random.default_rng(1)
    spike_idx = rng.choice(n, size=n_spikes, replace=False)
    s.iloc[spike_idx] += 50.0   # huge spikes
    return s


def test_disabled_filter_is_identity():
    s = _series_with_spikes()
    cleaned, n = remove_anomalies(s, AnomalyFilter(enabled=False))
    assert n == 0
    assert len(cleaned) == len(s)


def test_filter_drops_obvious_spikes():
    """With window=1 (point-level), IF should cleanly nuke the spikes.

    Larger windows include the spike's neighbours in every overlapping
    feature vector, so distinguishing each spike from its halo gets hard.
    For a baseline 'does the filter actually filter?' test, window=1 is
    the canonical setup.
    """
    pytest.importorskip("sklearn")
    s = _series_with_spikes(n_spikes=20)
    cleaned, n = remove_anomalies(
        s,
        AnomalyFilter(enabled=True, contamination=0.05, window=1),
    )
    assert n >= 15, f"expected to drop ≥15 outliers, dropped {n}"
    spike_threshold = 70.0
    spikes_remaining = int((cleaned > spike_threshold).sum())
    spikes_original = int((s > spike_threshold).sum())
    assert spikes_remaining < spikes_original / 4, (
        f"cleaned still has {spikes_remaining} of {spikes_original} spikes"
    )


def test_short_series_returns_unchanged():
    s = _clean_series(n=8)
    cleaned, n = remove_anomalies(
        s, AnomalyFilter(enabled=True, contamination=0.1, window=4),
    )
    assert n == 0
    assert len(cleaned) == len(s)


def test_schema_clamps_contamination():
    """contamination must stay in [0.001, 0.5]; pydantic enforces it."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AnomalyFilter(contamination=0.0)
    with pytest.raises(ValidationError):
        AnomalyFilter(contamination=0.9)
    # In-range is fine
    ok = AnomalyFilter(contamination=0.05)
    assert ok.contamination == 0.05
