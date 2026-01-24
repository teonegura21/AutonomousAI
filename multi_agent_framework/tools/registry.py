"""Tool registry for the multi_agent_framework package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ai_autonom.tools.tool_executor import ToolExecutor


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def execute(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {name}"}
        return tool.handler(params)


class ToolExecutorBridge:
    def __init__(self) -> None:
        self._executor = ToolExecutor()

    def run(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        success, result = self._executor.execute(tool_id, params)
        return {"success": success, "result": result}
