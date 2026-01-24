"""Shell execution wrappers."""

from __future__ import annotations

from typing import Any, Dict

from multi_agent_framework.tools.registry import ToolExecutorBridge


_bridge = ToolExecutorBridge()


def run_bash(command: str) -> Dict[str, Any]:
    return _bridge.run("bash_exec", {"command": command})


def run_generic(command: str) -> Dict[str, Any]:
    return _bridge.run("generic_linux_command", {"command": command})
