"""YAML config loader.

Loads four YAML files from FORECASTER_CONFIG_DIR (default: ./config) and
merges them into one Settings object. Env vars override individual keys
using a __ delimiter, e.g. FORECASTER__TRAINING__LOOKBACK_DAYS=14.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

from .schema import Settings

_DEFAULT_DIR = Path(os.environ.get("FORECASTER_CONFIG_DIR", "config"))
_ENV_PREFIX = "FORECASTER__"

_settings: Settings | None = None
_lock = threading.Lock()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply env-var overrides like FORECASTER__TRAINING__LOOKBACK_DAYS=14."""
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith(_ENV_PREFIX):
            continue
        path = raw_key[len(_ENV_PREFIX):].lower().split("__")
        cur = data
        for part in path[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):  # collision; bail
                break
        else:
            cur[path[-1]] = _coerce(raw_val)
    return data


def load_settings(config_dir: Path | None = None) -> Settings:
    """Load and validate Settings from the config directory."""
    cd = Path(config_dir) if config_dir else _DEFAULT_DIR
    merged: dict[str, Any] = {}
    for fname in ("default.yaml", "data_sources.yaml", "targets.yaml", "exposition.yaml"):
        merged = _deep_merge(merged, _read_yaml(cd / fname))

    # Infra runtime vars come from env (DATABASE_URL etc.)
    infra = {
        "database_url": os.environ.get("DATABASE_URL", merged.get("database_url", "postgresql+psycopg://forecaster:forecaster@postgres:5432/forecaster")),
        "celery_broker_url": os.environ.get("CELERY_BROKER_URL", merged.get("celery_broker_url", "redis://redis:6379/0")),
        "celery_result_backend": os.environ.get("CELERY_RESULT_BACKEND", merged.get("celery_result_backend", "redis://redis:6379/1")),
        "log_level": os.environ.get("LOG_LEVEL", merged.get("log_level", "INFO")),
        "use_cuda": os.environ.get("FORECASTER_USE_CUDA", "0") == "1",
    }
    merged = _deep_merge(merged, infra)
    merged = _apply_env_overrides(merged)
    return Settings.model_validate(merged)


def get_settings() -> Settings:
    """Module-level cached settings. Call reload_settings() to invalidate."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = load_settings()
    return _settings


def reload_settings() -> Settings:
    """Force-reload and replace the cached settings (used on SIGHUP)."""
    global _settings
    with _lock:
        _settings = load_settings()
    return _settings
