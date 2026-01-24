"""OpenManus-inspired coding agent spec."""

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


def build_coder_spec(workspace_dir: Path) -> FrameworkAgentSpec:
    prompt = _load_openmanus_prompt(
        "app.prompt.swe",
        workspace_dir / "multi_agent_framework" / "openmanus" / "app" / "prompt" / "swe.py",
    )

    tools: List[str] = [
        "filesystem_read",
        "filesystem_write",
        "filesystem_search",
        "filesystem_list",
        "bash_exec",
        "python_exec",
        "pytest_run",
    ]

    return FrameworkAgentSpec(
        agent_id="openmanus_coder",
        name="OpenManus SWE Coder",
        description="OpenManus SWE prompt + toolchain for coding tasks.",
        capabilities=[
            "code_generation",
            "debugging",
            "refactoring",
            "testing",
            "technical_tasks",
        ],
        tools=tools,
        system_prompt=prompt,
    )
