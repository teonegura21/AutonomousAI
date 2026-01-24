"""Nemotron orchestrator agent spec placeholder."""

from __future__ import annotations

from typing import List

from multi_agent_framework.agents.base import FrameworkAgentSpec


def build_orchestrator_spec() -> FrameworkAgentSpec:
    tools: List[str] = []
    return FrameworkAgentSpec(
        agent_id="nemotron_orchestrator",
        name="Nemotron Orchestrator",
        description="Planning-only orchestrator role (Nemotron).",
        capabilities=["planning", "task_decomposition", "agent_assignment"],
        tools=tools,
        system_prompt=None,
    )
