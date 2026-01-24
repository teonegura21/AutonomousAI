"""Filesystem tool wrappers."""

from __future__ import annotations

from typing import Any, Dict

from multi_agent_framework.tools.registry import ToolExecutorBridge


_bridge = ToolExecutorBridge()


def read_file(path: str) -> Dict[str, Any]:
    return _bridge.run("filesystem_read", {"path": path})


def write_file(path: str, content: str) -> Dict[str, Any]:
    return _bridge.run("filesystem_write", {"path": path, "content": content})


def list_directory(directory: str = ".") -> Dict[str, Any]:
    return _bridge.run("filesystem_list", {"directory": directory})


def search_files(directory: str, pattern: str) -> Dict[str, Any]:
    return _bridge.run("filesystem_search", {"directory": directory, "pattern": pattern})
