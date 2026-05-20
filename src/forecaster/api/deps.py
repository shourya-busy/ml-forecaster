"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache

from ..config.loader import get_settings
from ..config.schema import Settings
from ..registry.repo import RegistryRepo


def settings_dep() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _repo() -> RegistryRepo:
    return RegistryRepo(get_settings().database_url)


def repo_dep() -> RegistryRepo:
    return _repo()
