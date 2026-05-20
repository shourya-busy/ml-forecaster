"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src/` importable without installing
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Point config loader at the test config dir
os.environ.setdefault("FORECASTER_CONFIG_DIR", str(ROOT / "config"))
