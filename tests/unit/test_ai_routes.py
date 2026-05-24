"""Unit tests for /ai/explain + /ai/status routes."""

from __future__ import annotations

from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    db = tmp_path / "ai.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "art"))

    from fastapi.testclient import TestClient

    from forecaster.api import deps
    deps._repo.cache_clear()
    from forecaster.api.main import create_app
    from forecaster.config.loader import get_settings
    from forecaster.registry.repo import RegistryRepo
    repo = RegistryRepo(get_settings().database_url)
    repo.create_schema()
    return TestClient(create_app())


def _stub_ollama(monkeypatch, *, generate_text="ok", reachable=True, generate_raises=None):
    """Replace OllamaClient in both modules that import it."""
    from forecaster.ai import explainer as exp_mod
    from forecaster.api.routes import ai as ai_route

    class _Stub:
        def __init__(self, cfg):
            self.cfg = cfg
        def is_reachable(self):
            return (True, None) if reachable else (False, "stub down")
        def generate(self, prompt, *, system=None, model=None, options=None):
            if generate_raises:
                raise generate_raises
            return generate_text

    monkeypatch.setattr(exp_mod, "OllamaClient", _Stub)
    monkeypatch.setattr(ai_route, "OllamaClient", _Stub)


def test_status_reports_enabled_and_reachable(client, monkeypatch):
    _stub_ollama(monkeypatch, reachable=True)
    r = client.get("/ai/status")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["reachable"] is True
    assert body["model"]
    assert body["base_url"]


def test_status_unreachable(client, monkeypatch):
    _stub_ollama(monkeypatch, reachable=False)
    body = client.get("/ai/status").json()
    assert body["enabled"] is True
    assert body["reachable"] is False
    assert "stub down" in (body.get("reason") or "")


def test_explain_happy_path(client, monkeypatch):
    _stub_ollama(monkeypatch, generate_text="The CPU is trending upward.")
    body = {
        "kind": "forecast",
        "title": "srv-1 cpu medium",
        "instance": "srv-1", "metric": "cpu", "horizon": "medium",
        "best_algo": "ets",
        "actuals": [{"ts": "2026-05-20T00:00:00Z", "value": 10.0},
                    {"ts": "2026-05-20T00:05:00Z", "value": 11.0}],
        "forecast": [{"ts": "2026-05-20T00:10:00Z", "point": 12.0,
                      "lower": 11.0, "upper": 13.0}],
    }
    r = client.post("/ai/explain", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["explanation"] == "The CPU is trending upward."
    assert payload["model"]


def test_explain_422_when_no_data(client, monkeypatch):
    _stub_ollama(monkeypatch)
    r = client.post("/ai/explain", json={"kind": "forecast", "title": "x"})
    assert r.status_code == 422
    assert "actuals or forecast" in r.json()["detail"].lower()


def test_explain_502_on_ollama_error(client, monkeypatch):
    from forecaster.ai.ollama_client import OllamaError
    _stub_ollama(monkeypatch, generate_raises=OllamaError("connection refused"))
    body = {
        "kind": "forecast", "title": "x",
        "actuals": [{"ts": "2026-05-20T00:00:00Z", "value": 1.0}],
    }
    r = client.post("/ai/explain", json=body)
    assert r.status_code == 502
    assert "connection refused" in r.json()["detail"]


def test_explain_503_when_disabled(client, monkeypatch):
    # Toggle config to disabled via env-style override on the loader
    from forecaster.config import loader as cfg_loader
    s = cfg_loader.get_settings()
    s.ai.enabled = False
    try:
        body = {
            "kind": "forecast", "title": "x",
            "actuals": [{"ts": "2026-05-20T00:00:00Z", "value": 1.0}],
        }
        r = client.post("/ai/explain", json=body)
        assert r.status_code == 503
    finally:
        s.ai.enabled = True
