#!/usr/bin/env python3
"""
Simple command prompt for the SimpleTUI loop.
"""


def get_simple_tui_command() -> str:
    """Read a command from stdin."""
    try:
        return input("TUI> ").strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"
