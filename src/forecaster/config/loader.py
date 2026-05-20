"""Config loader.

Three layers, applied in order:

1. YAML files under FORECASTER_CONFIG_DIR (default.yaml, data_sources.yaml,
   targets.yaml, exposition.yaml).
2. Env vars with FORECASTER__SECTION__KEY style (double-underscore).
3. **Live DB overrides** from the `settings_overrides` table — these are
   what the UI Manage pages write to. They take precedence over YAML and
   env, deep-merged in via dotted keys (e.g. `training.lookback_days`).

Layers 1+2 are immutable at runtime; layer 3 is re-read with a short TTL
so UI edits propagate to the next scheduler tick / API request without
needing a reload.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from .schema import Settings

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path(os.environ.get("FORECASTER_CONFIG_DIR", "config"))
_ENV_PREFIX = "FORECASTER__"
_OVERRIDE_TTL_SECONDS = float(os.environ.get("FORECASTER_OVERRIDE_TTL", "15"))

# Module-level cache. Tests inject by setting `_settings = some_settings`,
# or invalidate by setting `_settings = None`.
_settings: Settings | None = None
_settings_loaded_at: float = 0.0
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


def _set_by_dotted_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Set data[a][b][c] = value given dotted='a.b.c'. Creates dicts as needed."""
    parts = dotted.split(".")
    cur = data
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _load_static_dict(config_dir: Path | None = None) -> dict[str, Any]:
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
    return _apply_env_overrides(merged)


def _read_db_overrides(database_url: str) -> dict[str, Any]:
    """Pull overrides from the `settings_overrides` table.

    Fail-open: returns {} on any error (DB unreachable, table missing
    pre-migration, etc.) so the API/scheduler can boot.
    """
    try:
        from ..registry.repo import RegistryRepo  # lazy: avoid import cycle
        repo = RegistryRepo(database_url)
        return repo.get_all_settings_overrides()
    except Exception as exc:  # noqa: BLE001
        log.debug("settings overrides unavailable: %s", exc)
        return {}


def load_settings(config_dir: Path | None = None, *, with_db_overrides: bool = True) -> Settings:
    """Load Settings from YAML + env, then merge DB overrides if requested.

    Tests typically pass `with_db_overrides=False` (or no DB exists yet).
    """
    base = _load_static_dict(config_dir)
    settings = Settings.model_validate(base)
    if not with_db_overrides:
        return settings
    overrides = _read_db_overrides(settings.database_url)
    if not overrides:
        return settings
    for key, value in overrides.items():
        _set_by_dotted_path(base, key, value)
    try:
        return Settings.model_validate(base)
    except Exception as exc:  # noqa: BLE001
        log.warning("settings overrides failed validation, discarding: %s", exc)
        return settings


def get_settings() -> Settings:
    """Cached effective settings. Refreshes from DB every TTL seconds.

    Tests can short-circuit by assigning to the module-level _settings
    directly (e.g. cfg_loader._settings = my_settings).
    """
    global _settings, _settings_loaded_at
    now = time.time()
    if _settings is not None and (now - _settings_loaded_at) < _OVERRIDE_TTL_SECONDS:
        return _settings
    with _lock:
        if _settings is not None and (time.time() - _settings_loaded_at) < _OVERRIDE_TTL_SECONDS:
            return _settings
        _settings = load_settings()
        _settings_loaded_at = time.time()
    return _settings


def reload_settings() -> Settings:
    """Force-reload everything (YAML + env + DB). Used on SIGHUP / Reload button."""
    global _settings, _settings_loaded_at
    with _lock:
        _settings = load_settings()
        _settings_loaded_at = time.time()
    return _settings


def invalidate_settings_cache() -> None:
    """Bump the cache so the next get_settings() re-reads DB overrides.

    Call this from any UI endpoint that writes to settings_overrides so
    the change is visible on the next request without waiting for the TTL.
    """
    global _settings_loaded_at
    _settings_loaded_at = 0.0
