"""Framework agent definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ai_autonom.core.agent_registry import AgentDefinition


@dataclass
class FrameworkAgentSpec:
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    tools: List[str]
    system_prompt: Optional[str] = None

    def to_definition(
        self,
        *,
        model_name: str,
        model_size_gb: float,
        vram_required: float,
        speed_tokens_per_sec: float,
        quality_score: float,
        provider: str = "ollama",
    ) -> AgentDefinition:
        return AgentDefinition(
            id=self.agent_id,
            name=self.name,
            model_name=model_name,
            model_size_gb=model_size_gb,
            capabilities=self.capabilities,
            tools=self.tools,
            vram_required=vram_required,
            speed_tokens_per_sec=speed_tokens_per_sec,
            quality_score=quality_score,
            description=self.description,
            provider=provider,
            system_prompt=self.system_prompt,
        )


def load_prompt_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
