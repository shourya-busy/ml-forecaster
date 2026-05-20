"""The to_local Jinja filter must convert UTC strings to the configured tz."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from forecaster.config import loader as cfg_loader


def test_to_local_renders_in_kolkata():
    cfg_loader._settings = None
    os.environ.pop("FORECASTER__DISPLAY_TIMEZONE", None)
    settings = cfg_loader.load_settings()
    assert settings.display_timezone == "Asia/Kolkata"
    cfg_loader._settings = settings

    from forecaster.ui.routes import to_local

    # 18:00:00 UTC → 23:30:00 IST (+5:30)
    iso = "2026-05-20T18:00:00+00:00"
    out = to_local(iso)
    assert "23:30:00" in out
    assert "IST" in out or "+0530" in out or "Asia/Kolkata" in out


def test_to_local_handles_naive_datetime_as_utc():
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.display_timezone = "Asia/Kolkata"
    cfg_loader._settings = settings

    from forecaster.ui.routes import to_local

    naive = datetime(2026, 5, 20, 18, 0, 0)
    out = to_local(naive)
    assert "23:30:00" in out


def test_to_local_dash_on_none():
    from forecaster.ui.routes import to_local
    assert to_local(None) == "—"
    assert to_local("") == "—"


def test_to_local_uses_runtime_timezone():
    """Filter must reflect the current Settings, not the import-time tz."""
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.display_timezone = "UTC"
    cfg_loader._settings = settings

    from forecaster.ui.routes import to_local
    out = to_local("2026-05-20T18:00:00+00:00")
    assert "18:00:00" in out
