from __future__ import annotations

from typing import Any, Dict, Optional

from app.tool.base import BaseTool, ToolResult

from multi_agent_framework.tools.registry import ToolExecutorBridge


_BRIDGE = ToolExecutorBridge()


class ExecutorTool(BaseTool):
    tool_id: str

    def _execute(self, params: Dict[str, Any]) -> ToolResult:
        result = _BRIDGE.run(self.tool_id, params)
        if result.get("success"):
            return ToolResult(output=str(result.get("result")))
        return ToolResult(error=str(result.get("result")))

    async def execute(self, **kwargs) -> Any:
        return self._execute(kwargs)


class FilesystemRead(ExecutorTool):
    name: str = "filesystem_read"
    description: str = "Read a file from disk"
    tool_id: str = "filesystem_read"
    parameters: dict = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


class FilesystemWrite(ExecutorTool):
    name: str = "filesystem_write"
    description: str = "Write content to a file"
    tool_id: str = "filesystem_write"
    parameters: dict = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }


class FilesystemSearch(ExecutorTool):
    name: str = "filesystem_search"
    description: str = "Search for files matching a pattern"
    tool_id: str = "filesystem_search"
    parameters: dict = {
        "type": "object",
        "properties": {
            "directory": {"type": "string"},
            "pattern": {"type": "string"},
        },
    }


class FilesystemList(ExecutorTool):
    name: str = "filesystem_list"
    description: str = "List files in a directory"
    tool_id: str = "filesystem_list"
    parameters: dict = {
        "type": "object",
        "properties": {"directory": {"type": "string"}},
    }


class BashExec(ExecutorTool):
    name: str = "bash_exec"
    description: str = "Execute a bash command"
    tool_id: str = "bash_exec"
    parameters: dict = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }


class PythonExec(ExecutorTool):
    name: str = "python_exec"
    description: str = "Execute Python code"
    tool_id: str = "python_exec"
    parameters: dict = {
        "type": "object",
        "properties": {"code": {"type": "string"}, "filename": {"type": "string"}},
        "required": ["code"],
    }


class WebSearch(ExecutorTool):
    name: str = "web_search"
    description: str = "Search the web"
    tool_id: str = "web_search"
    parameters: dict = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


class WebFetch(ExecutorTool):
    name: str = "web_fetch"
    description: str = "Fetch a URL"
    tool_id: str = "web_fetch"
    parameters: dict = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }


class GenericLinuxCommand(ExecutorTool):
    name: str = "generic_linux_command"
    description: str = "Run a generic Linux command"
    tool_id: str = "generic_linux_command"
    parameters: dict = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }


class NmapScan(ExecutorTool):
    name: str = "nmap_scan"
    description: str = "Run an nmap scan"
    tool_id: str = "nmap_scan"
    parameters: dict = {
        "type": "object",
        "properties": {"target": {"type": "string"}, "args": {"type": "string"}},
        "required": ["target"],
    }


class CurlRequest(ExecutorTool):
    name: str = "curl_request"
    description: str = "Issue a curl request"
    tool_id: str = "curl_request"
    parameters: dict = {
        "type": "object",
        "properties": {"url": {"type": "string"}, "args": {"type": "string"}},
        "required": ["url"],
    }


def build_default_tools() -> list[BaseTool]:
    tools: list[BaseTool] = [
        FilesystemRead(),
        FilesystemWrite(),
        FilesystemSearch(),
        FilesystemList(),
        BashExec(),
        PythonExec(),
        WebSearch(),
        WebFetch(),
        GenericLinuxCommand(),
        NmapScan(),
        CurlRequest(),
    ]

    try:
        from app.tool.browser_use_tool import BrowserUseTool

        tools.append(BrowserUseTool())
    except Exception:
        pass

    return tools
