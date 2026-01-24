"""Ollama inference wrapper."""

from __future__ import annotations

from ai_autonom.core.llm_provider import LLMProviderFactory, LLMConfig, ProviderType


def create_client(api_base: str = "http://localhost:11434"):
    config = LLMConfig(provider=ProviderType.OLLAMA, api_base=api_base)
    return LLMProviderFactory.create(config)
