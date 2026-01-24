"""Docker sandbox wrapper."""

from __future__ import annotations

from ai_autonom.sandbox.container_router import ContainerToolRouter, get_router


class DockerSandbox:
    def __init__(self) -> None:
        self._router: ContainerToolRouter = get_router()

    def is_available(self) -> bool:
        return self._router.is_available()

    def execute_tool(self, tool_id: str, params):
        return self._router.execute_tool(tool_id, params)
