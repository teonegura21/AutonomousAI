"""Short-term memory wrapper."""

from __future__ import annotations

from ai_autonom.memory.shared_context import SharedContext


class ShortTermMemory:
    def __init__(self, base_dir: str = "./.runtime/data") -> None:
        self._context = SharedContext(base_dir=base_dir)

    def add(self, key: str, value: str) -> None:
        self._context.store(key, value)

    def get(self, key: str) -> str:
        return self._context.get(key) or ""

    def clear(self) -> None:
        self._context.clear()
