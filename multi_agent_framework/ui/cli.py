"""CLI entry for the multi_agent_framework UI."""

from __future__ import annotations

import argparse

from multi_agent_framework.ui.web.backend import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent framework UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
