"""Artifact storage abstraction.

Default impl writes pickled artifacts to a local volume. S3/MinIO could
be added later behind the same interface without touching call sites.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path

from ..models.base import Forecaster


class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    def save(self, forecaster: Forecaster, *, instance: str, metric: str, horizon: str, algo: str, run_id: int) -> tuple[str, int]:
        """Persist forecaster; return (path, size_bytes)."""

    @abc.abstractmethod
    def load(self, path: str) -> Forecaster:
        """Load a forecaster by path."""

    @abc.abstractmethod
    def delete(self, path: str) -> None:
        """Delete an artifact; safe to call on missing paths."""


class VolumeArtifactStore(ArtifactStore):
    def __init__(self, base_dir: str | os.PathLike[str]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, *, instance: str, metric: str, horizon: str, algo: str, run_id: int) -> Path:
        safe_instance = instance.replace("/", "_").replace(":", "_")
        rel = Path(metric) / horizon / safe_instance / algo / f"run-{run_id}.pkl"
        return self.base_dir / rel

    def save(self, forecaster: Forecaster, *, instance: str, metric: str, horizon: str, algo: str, run_id: int) -> tuple[str, int]:
        p = self._path_for(instance=instance, metric=metric, horizon=horizon, algo=algo, run_id=run_id)
        forecaster.save(p)
        return str(p), p.stat().st_size

    def load(self, path: str) -> Forecaster:
        return Forecaster.load(Path(path))  # type: ignore[abstract]

    def delete(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            p.unlink()
