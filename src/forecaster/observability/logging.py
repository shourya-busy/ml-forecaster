"""Structured-JSON logging for stdout (suitable for Loki ingestion)."""

from __future__ import annotations

import logging
import os

from pythonjsonlogger import jsonlogger


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. by uvicorn)
    handler = logging.StreamHandler()
    fmt = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(fmt)
    root.addHandler(handler)
    root.setLevel(level)
