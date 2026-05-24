"""Tests for the Manage → AI page and its save/reset/test endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest


def _setup(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    db = tmp_path / "ai.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER_OVERRIDE_TTL", "0")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "art"))

    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None

    from forecaster.registry.repo import RegistryRepo
    repo = RegistryRepo(cfg_loader.load_settings().database_url)
    repo.create_schema()
    return repo


def _client():
    from fastapi.testclient import TestClient

    from forecaster.api import deps
    deps._repo.cache_clear()
    from forecaster.api.main import create_app
    return TestClient(create_app(), follow_redirects=False)


def test_manage_ai_renders(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    r = _client().get("/ui/manage/ai")
    assert r.status_code == 200
    body = r.text
    assert "AI explainer" in body
    assert "base_url" in body
    assert 'name="model"' in body
    assert 'name="timeout_seconds"' in body
    assert 'name="max_points_in_prompt"' in body
    assert "Test connection" in body
    assert "Reset to defaults" in body


def test_sidebar_links_to_ai(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    body = _client().get("/ui/").text
    assert "/ui/manage/ai" in body
    # The active class wiring should mark only the AI link active when on
    # the AI page itself — quick sanity check using its label.
    on_page = _client().get("/ui/manage/ai").text
    assert ">AI<" in on_page


def test_save_persists_overrides(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post(
        "/ui/manage/ai/save",
        data={
            "enabled": "1",
            "base_url": "http://my-ollama:9999",
            "model": "qwen2.5",
            "timeout_seconds": "90",
            "max_points_in_prompt": "120",
        },
    )
    assert r.status_code == 303
    assert "saved=1" in r.headers["location"]

    overrides = repo.get_all_settings_overrides()
    assert overrides["ai.enabled"] is True
    assert overrides["ai.ollama.base_url"] == "http://my-ollama:9999"
    assert overrides["ai.ollama.model"] == "qwen2.5"
    assert overrides["ai.ollama.timeout_seconds"] == 90
    assert overrides["ai.ollama.max_points_in_prompt"] == 120

    # Page reload should reflect the new values
    body = client.get("/ui/manage/ai?saved=1").text
    assert "my-ollama:9999" in body
    assert "qwen2.5" in body


def test_save_with_enabled_unchecked_persists_false(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/manage/ai/save", data={
        "base_url": "http://x", "model": "m",
        "timeout_seconds": "30", "max_points_in_prompt": "60",
    })
    assert r.status_code == 303
    assert repo.get_all_settings_overrides()["ai.enabled"] is False


def test_save_empty_string_clears_override(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    # Set an override first
    repo.set_settings_override("ai.ollama.base_url", "http://override")
    assert "ai.ollama.base_url" in repo.get_all_settings_overrides()
    # Now save with empty base_url → that override should be removed.
    client = _client()
    r = client.post("/ui/manage/ai/save", data={
        "enabled": "1", "base_url": "",
        "model": "llama3.2", "timeout_seconds": "60", "max_points_in_prompt": "60",
    })
    assert r.status_code == 303
    assert "ai.ollama.base_url" not in repo.get_all_settings_overrides()


def test_reset_clears_all_ai_overrides(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    for k, v in [
        ("ai.enabled", False),
        ("ai.ollama.base_url", "http://x"),
        ("ai.ollama.model", "m"),
        ("ai.ollama.timeout_seconds", 10),
        ("ai.ollama.max_points_in_prompt", 25),
        # Unrelated override must NOT be touched
        ("training.lookback_days", 7),
    ]:
        repo.set_settings_override(k, v)

    r = _client().post("/ui/manage/ai/reset")
    assert r.status_code == 303
    remaining = repo.get_all_settings_overrides()
    for k in [
        "ai.enabled", "ai.ollama.base_url", "ai.ollama.model",
        "ai.ollama.timeout_seconds", "ai.ollama.max_points_in_prompt",
    ]:
        assert k not in remaining
    # Unrelated override survived
    assert remaining.get("training.lookback_days") == 7


def test_test_endpoint_reports_unreachable(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # Stub the OllamaClient that ui.routes pulls in
    from forecaster.ai import ollama_client as oc_mod
    class _Stub:
        def __init__(self, cfg): self.cfg = cfg
        def is_reachable(self): return (False, "connection refused")
        def list_models(self): return []
    monkeypatch.setattr(oc_mod, "OllamaClient", _Stub)

    r = _client().post("/ui/manage/ai/test", data={"base_url": "http://x"})
    assert r.status_code == 200
    assert "unreachable" in r.text
    assert "connection refused" in r.text


def test_test_endpoint_reports_models(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from forecaster.ai import ollama_client as oc_mod
    class _Stub:
        def __init__(self, cfg): self.cfg = cfg
        def is_reachable(self): return (True, None)
        def list_models(self): return ["llama3.2", "qwen2.5", "mistral"]
    monkeypatch.setattr(oc_mod, "OllamaClient", _Stub)

    r = _client().post("/ui/manage/ai/test", data={"base_url": "http://x"})
    assert r.status_code == 200
    assert "reachable" in r.text
    assert "llama3.2" in r.text


def test_ai_models_route_uses_supplied_base_url(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # Stub on the ai routes module too so /ai/models picks it up
    from forecaster.api.routes import ai as ai_route
    seen = {}
    class _Stub:
        def __init__(self, cfg):
            seen["base_url"] = cfg.base_url
        def list_models(self): return ["a", "b"]
    monkeypatch.setattr(ai_route, "OllamaClient", _Stub)

    r = _client().get("/ai/models?base_url=http://other:11434")
    assert r.status_code == 200
    assert r.json()["models"] == ["a", "b"]
    assert seen["base_url"] == "http://other:11434"
