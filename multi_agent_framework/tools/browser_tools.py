"""Browser tool wrappers (placeholder)."""

from __future__ import annotations

from typing import Any, Dict

from multi_agent_framework.tools.registry import ToolExecutorBridge


_bridge = ToolExecutorBridge()


def browser_use(action: str, **kwargs: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {"action": action, **kwargs}
    return _bridge.run("browser_use", params)
