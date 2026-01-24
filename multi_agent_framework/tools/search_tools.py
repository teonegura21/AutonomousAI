"""Search and web retrieval wrappers."""

from __future__ import annotations

from typing import Any, Dict

from multi_agent_framework.tools.registry import ToolExecutorBridge


_bridge = ToolExecutorBridge()


def web_search(query: str) -> Dict[str, Any]:
    return _bridge.run("web_search", {"query": query})


def web_fetch(url: str) -> Dict[str, Any]:
    return _bridge.run("web_fetch", {"url": url})


def web_scrape(url: str) -> Dict[str, Any]:
    return _bridge.run("web_fetch", {"url": url})
