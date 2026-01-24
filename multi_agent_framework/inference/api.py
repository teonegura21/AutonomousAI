"""API provider wrappers."""

from __future__ import annotations

from ai_autonom.core.llm_provider import LLMProviderFactory, LLMConfig, ProviderType


def create_openai_client(api_key: str, api_base: str = None):
    config = LLMConfig(provider=ProviderType.OPENAI, api_key=api_key, api_base=api_base)
    return LLMProviderFactory.create(config)
