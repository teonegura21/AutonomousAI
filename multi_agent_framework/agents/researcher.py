"""OpenManus-inspired research agent spec."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import List

from multi_agent_framework.agents.base import FrameworkAgentSpec, load_prompt_text


def _load_openmanus_prompt(module_path: str, fallback_path: Path) -> str:
    try:
        module = import_module(module_path)
        prompt = getattr(module, "SYSTEM_PROMPT", "")
        return prompt
    except Exception:
        return load_prompt_text(fallback_path)


def build_researcher_spec(workspace_dir: Path) -> FrameworkAgentSpec:
    raw_prompt = _load_openmanus_prompt(
        "app.prompt.manus",
        workspace_dir / "multi_agent_framework" / "openmanus" / "app" / "prompt" / "manus.py",
    )
    prompt = raw_prompt.format(directory=str(workspace_dir)).strip()

    tools: List[str] = [
        "web_search",
        "web_fetch",
        "web_scrape",
        "filesystem_read",
        "filesystem_write",
    ]

    return FrameworkAgentSpec(
        agent_id="openmanus_researcher",
        name="OpenManus Research Agent",
        description="OpenManus Manus prompt tuned for research and synthesis.",
        capabilities=["research", "information_gathering", "summarization"],
        tools=tools,
        system_prompt=prompt,
    )
