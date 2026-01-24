"""E2B sandbox placeholder."""

from __future__ import annotations


class E2BSandbox:
    def __init__(self) -> None:
        self.enabled = False

    def is_available(self) -> bool:
        return False

    def execute(self, command: str):
        raise RuntimeError("E2B sandbox not configured.")
