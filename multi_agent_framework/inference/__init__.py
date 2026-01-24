"""Inference package exports."""

from multi_agent_framework.inference.ollama import create_client as create_ollama_client
from multi_agent_framework.inference.api import create_openai_client
from multi_agent_framework.inference.router import build_router

__all__ = ["create_ollama_client", "create_openai_client", "build_router"]
