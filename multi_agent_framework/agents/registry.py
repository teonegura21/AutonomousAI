"""Register OpenManus/CAI agent specs with the runtime registry."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ai_autonom.core.agent_registry import AgentRegistry

from multi_agent_framework.agents.coder import build_coder_spec
from multi_agent_framework.agents.researcher import build_researcher_spec
from multi_agent_framework.agents.security import (
    build_red_team_spec,
    build_reporting_spec,
    build_web_pentester_spec,
)


ROLE_MAP = {
    "coder": "openmanus_coder",
    "researcher": "openmanus_researcher",
    "security": "cai_red_team",
    "web_pentester": "cai_web_pentester",
    "reporter": "cai_reporter",
}


def register_framework_agents(
    registry: AgentRegistry,
    *,
    workspace_dir: Optional[Path] = None,
    coder_model: str,
    linguistic_model: str,
    provider: str = "ollama",
    vram_gb: float = 4.0,
) -> Dict[str, str]:
    workspace_dir = workspace_dir or Path.cwd()

    coder_spec = build_coder_spec(workspace_dir)
    researcher_spec = build_researcher_spec(workspace_dir)
    red_team_spec = build_red_team_spec(workspace_dir)
    web_pentester_spec = build_web_pentester_spec(workspace_dir)
    reporter_spec = build_reporting_spec(workspace_dir)

    specs = [
        (coder_spec, coder_model, vram_gb, 70.0, 88.0),
        (researcher_spec, coder_model, vram_gb, 60.0, 85.0),
        (red_team_spec, coder_model, vram_gb, 55.0, 90.0),
        (web_pentester_spec, coder_model, vram_gb, 55.0, 90.0),
        (reporter_spec, linguistic_model, vram_gb, 50.0, 86.0),
    ]

    for spec, model_name, vram, speed, quality in specs:
        registry.register_agent(
            spec.to_definition(
                model_name=model_name,
                model_size_gb=vram,
                vram_required=vram,
                speed_tokens_per_sec=speed,
                quality_score=quality,
                provider=provider,
            )
        )

    return ROLE_MAP.copy()


def resolve_agent_for_role(
    registry: AgentRegistry,
    role: str,
    role_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    role_map = role_map or ROLE_MAP
    agent_id = role_map.get(role)
    if not agent_id:
        return None
    for agent in registry.get_all_agents():
        if agent.id == agent_id:
            return agent_id
    return None
