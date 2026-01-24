"""Agents package exports."""

from multi_agent_framework.agents.registry import register_framework_agents, resolve_agent_for_role

__all__ = ["register_framework_agents", "resolve_agent_for_role"]
