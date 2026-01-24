"""CAI-inspired security agent specs."""

from __future__ import annotations

from pathlib import Path
from typing import List

from multi_agent_framework.agents.base import FrameworkAgentSpec, load_prompt_text


def _prompt_path(workspace_dir: Path, filename: str) -> Path:
    return workspace_dir / "multi_agent_framework" / "cai" / "prompts" / filename


def build_red_team_spec(workspace_dir: Path) -> FrameworkAgentSpec:
    prompt = load_prompt_text(_prompt_path(workspace_dir, "system_red_team_agent.md"))
    tools: List[str] = [
        "generic_linux_command",
        "nmap_scan",
        "netcat",
        "filesystem_read",
        "filesystem_write",
        "python_exec",
        "capture_remote_traffic",
        "record_finding",
        "record_credential",
    ]
    return FrameworkAgentSpec(
        agent_id="cai_red_team",
        name="CAI Red Team Agent",
        description="CAI red team prompt + security tooling.",
        capabilities=["red_teaming", "penetration_testing", "exploitation"],
        tools=tools,
        system_prompt=prompt,
    )


def build_web_pentester_spec(workspace_dir: Path) -> FrameworkAgentSpec:
    prompt = load_prompt_text(_prompt_path(workspace_dir, "system_web_pentester.md"))
    tools: List[str] = [
        "generic_linux_command",
        "curl_request",
        "web_spider",
        "js_surface_mapper",
        "web_request_framework",
        "shodan_search",
        "record_finding",
    ]
    return FrameworkAgentSpec(
        agent_id="cai_web_pentester",
        name="CAI Web Pentester",
        description="CAI web pentester prompt + web tooling.",
        capabilities=["web_hacking", "api_testing", "reconnaissance"],
        tools=tools,
        system_prompt=prompt,
    )


def build_reporting_spec(workspace_dir: Path) -> FrameworkAgentSpec:
    prompt = load_prompt_text(_prompt_path(workspace_dir, "system_reporting_agent.md"))
    tools: List[str] = [
        "filesystem_read",
        "filesystem_write",
        "record_finding",
    ]
    return FrameworkAgentSpec(
        agent_id="cai_reporter",
        name="CAI Reporting Agent",
        description="CAI security reporting prompt.",
        capabilities=["reporting", "summarization", "documentation"],
        tools=tools,
        system_prompt=prompt,
    )
