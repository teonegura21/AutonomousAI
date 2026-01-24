"""Security tool wrappers (CAI-based)."""

from __future__ import annotations

from typing import Any, Dict

from multi_agent_framework.tools.registry import ToolExecutorBridge


_bridge = ToolExecutorBridge()


def nmap_scan(target: str, args: str = "") -> Dict[str, Any]:
    return _bridge.run("nmap_scan", {"target": target, "args": args})


def curl_request(url: str, args: str = "") -> Dict[str, Any]:
    return _bridge.run("curl_request", {"url": url, "args": args})


def web_request_framework(url: str, method: str = "GET", data: str = "") -> Dict[str, Any]:
    return _bridge.run("web_request_framework", {"url": url, "method": method, "data": data})


def js_surface_mapper(url: str) -> Dict[str, Any]:
    return _bridge.run("js_surface_mapper", {"url": url})
