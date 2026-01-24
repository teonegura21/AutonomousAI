#!/usr/bin/env python3
"""
Simple TUI fallback.
Provides a minimal text-based interface used by cli/orchestrator.py.
"""

import sys
from typing import Optional


class SimpleTUI:
    """Minimal stateful TUI used by the CLI loop."""

    def __init__(self):
        self.running = False
        self._goal: str = ""
        self._status: str = "Idle"
        self._last_result: str = ""

    def update_goal(self, goal: str) -> None:
        self._goal = goal or ""

    def set_orchestrator_status(self, status: str) -> None:
        self._status = status or ""

    def set_command_result(self, result: str) -> None:
        self._last_result = result or ""

    def print_status(self) -> None:
        header = "=" * 60
        print(header)
        print("AI Autonom - Simple TUI")
        print(f"Goal   : {self._goal or '(none)'}")
        print(f"Status : {self._status or '(unknown)'}")
        if self._last_result:
            print(f"Last   : {self._last_result}")
        print("Commands: goal <text> | status | agents | tools | quit")
        print(header)
        sys.stdout.flush()


_simple_tui: Optional[SimpleTUI] = None


def get_simple_tui() -> SimpleTUI:
    """Get singleton SimpleTUI instance."""
    global _simple_tui
    if _simple_tui is None:
        _simple_tui = SimpleTUI()
    return _simple_tui
