"""OpenManus shim for legacy `app.*` imports."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

__path__ = [
    str(_Path(__file__).resolve().parents[1] / "multi_agent_framework" / "openmanus" / "app")
]

from multi_agent_framework.openmanus import app as _app

_sys.modules["app"] = _app
