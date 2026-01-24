"""Simple routing helper for inference providers."""

from __future__ import annotations

from typing import Dict

from multi_agent_framework.inference.ollama import create_client as create_ollama
from multi_agent_framework.inference.api import create_openai_client


def build_router(config: Dict):
    provider = config.get("provider", "ollama")
    if provider == "openai":
        return create_openai_client(config.get("api_key", ""), config.get("api_base"))
    return create_ollama(config.get("api_base", "http://localhost:11434"))
