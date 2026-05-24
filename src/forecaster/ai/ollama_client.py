"""Thin synchronous wrapper around the Ollama HTTP API.

Only the endpoints we actually use: /api/tags (health/model list) and
/api/generate (single-shot completion). Streaming is intentionally not
exposed yet — the UI shows a spinner and waits for the full response.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config.schema import OllamaConfig

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised on any non-recoverable error talking to the Ollama server."""


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self._cfg = cfg
        # Strip trailing slash so URL joins are predictable
        self._base = cfg.base_url.rstrip("/")

    # ----- public -----

    def is_reachable(self) -> tuple[bool, str | None]:
        """Cheap liveness probe — used by /ai/status to gate the UI button."""
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(f"{self._base}/api/tags")
                r.raise_for_status()
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{self._base}/api/tags")
                r.raise_for_status()
                payload = r.json()
            return sorted(m.get("name", "") for m in payload.get("models", []))
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"could not list models: {exc}") from exc

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Single-shot completion. Returns the model's full response text."""
        body: dict[str, Any] = {
            "model": model or self._cfg.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            body["system"] = system
        if options:
            body["options"] = options
        try:
            with httpx.Client(timeout=self._cfg.timeout_seconds) as c:
                r = c.post(f"{self._base}/api/generate", json=body)
                r.raise_for_status()
                payload = r.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else ""
            raise OllamaError(
                f"ollama returned {exc.response.status_code}: {detail}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"ollama request failed: {exc}") from exc

        text = (payload.get("response") or "").strip()
        if not text:
            raise OllamaError("ollama returned an empty response")
        return text
