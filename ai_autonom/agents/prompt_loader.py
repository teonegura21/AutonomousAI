"""
Prompt Template Loader for ai_autonom Agents

Provides utilities to load and render system prompts from markdown files.
Prompts are stored in the prompts/ directory as .md files.

Usage:
    from ai_autonom.agents.prompt_loader import load_prompt, get_available_prompts
    
    # Load a specific prompt
    red_team_prompt = load_prompt("red_team_agent")
    
    # List all available prompts
    prompts = get_available_prompts()
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from functools import lru_cache


# Directory containing prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .md extension)
    
    Returns:
        The prompt text content
    
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    
    Example:
        >>> prompt = load_prompt("red_team_agent")
        >>> print(prompt[:50])
        You are a highly specialized red team agent...
    """
    # Try with and without .md extension
    if not prompt_name.endswith('.md'):
        prompt_file = PROMPTS_DIR / f"{prompt_name}.md"
    else:
        prompt_file = PROMPTS_DIR / prompt_name
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    
    return prompt_file.read_text(encoding='utf-8')


def load_prompt_safe(prompt_name: str, default: str = "") -> str:
    """
    Load a prompt template, returning default if not found.
    
    Args:
        prompt_name: Name of the prompt file (without .md extension)
        default: Default text to return if file not found
    
    Returns:
        The prompt text content or default
    """
    try:
        return load_prompt(prompt_name)
    except FileNotFoundError:
        return default


def get_available_prompts() -> List[str]:
    """
    List all available prompt templates.
    
    Returns:
        List of prompt names (without .md extension)
    """
    if not PROMPTS_DIR.exists():
        return []
    
    prompts = []
    for file in PROMPTS_DIR.glob("*.md"):
        prompts.append(file.stem)
    
    return sorted(prompts)


def get_prompt_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about all available prompts.
    
    Returns:
        Dict mapping prompt name to info dict with 'path', 'size', 'first_line'
    """
    info = {}
    
    for prompt_name in get_available_prompts():
        prompt_file = PROMPTS_DIR / f"{prompt_name}.md"
        content = prompt_file.read_text(encoding='utf-8')
        
        # Get first non-empty line as description
        first_line = ""
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                first_line = stripped[:100]
                break
        
        info[prompt_name] = {
            "path": str(prompt_file),
            "size": len(content),
            "first_line": first_line
        }
    
    return info


def create_agent_prompt(
    base_prompt: str,
    additional_context: Optional[str] = None,
    target_info: Optional[Dict[str, str]] = None
) -> str:
    """
    Create a complete agent prompt by combining base prompt with context.
    
    Args:
        base_prompt: The base system prompt
        additional_context: Optional additional instructions
        target_info: Optional dict with target-specific info (ip, host, etc.)
    
    Returns:
        Complete prompt string
    """
    parts = [base_prompt]
    
    if additional_context:
        parts.append(f"\n\n## Additional Context\n{additional_context}")
    
    if target_info:
        target_section = "\n\n## Target Information\n"
        for key, value in target_info.items():
            target_section += f"- **{key}**: {value}\n"
        parts.append(target_section)
    
    return "".join(parts)


# Pre-defined prompt loaders for common agents
class AgentPrompts:
    """
    Convenient access to common agent prompts.
    
    Usage:
        prompts = AgentPrompts()
        red_team = prompts.red_team()
        web_pentester = prompts.web_pentester()
    """
    
    @staticmethod
    def red_team() -> str:
        """Red team agent prompt for offensive operations."""
        return load_prompt_safe("red_team_agent", _DEFAULT_RED_TEAM)
    
    @staticmethod
    def blue_team() -> str:
        """Blue team agent prompt for defensive operations."""
        return load_prompt_safe("blue_team_agent", _DEFAULT_BLUE_TEAM)
    
    @staticmethod
    def reporting() -> str:
        """Reporting agent prompt for generating security reports."""
        return load_prompt_safe("reporting_agent", _DEFAULT_REPORTING)
    
    @staticmethod
    def triage() -> str:
        """Triage agent prompt for vulnerability verification."""
        return load_prompt_safe("triage_agent", _DEFAULT_TRIAGE)
    
    @staticmethod
    def dfir() -> str:
        """DFIR agent prompt for forensics and incident response."""
        return load_prompt_safe("dfir_agent", _DEFAULT_DFIR)
    
    @staticmethod
    def web_pentester() -> str:
        """Web pentester agent prompt for web application testing."""
        return load_prompt_safe("web_pentester", _DEFAULT_WEB_PENTESTER)


# Default fallback prompts (minimal versions if files not found)
_DEFAULT_RED_TEAM = """You are a red team security agent focused on system penetration.
Objectives: Network scanning, service exploitation, privilege escalation, lateral movement.
Use non-interactive commands only. Execute one command at a time. Document findings."""

_DEFAULT_BLUE_TEAM = """You are a blue team security agent focused on defense.
Objectives: Threat detection, log analysis, incident response, security hardening."""

_DEFAULT_REPORTING = """You are a security reporting agent.
Create professional security assessment reports with executive summaries,
findings organized by severity, and remediation recommendations."""

_DEFAULT_TRIAGE = """You are a vulnerability triage agent.
Verify vulnerabilities, assess exploitability, eliminate false positives."""

_DEFAULT_DFIR = """You are a digital forensics and incident response agent.
Analyze digital evidence, investigate incidents, identify malicious activity."""

_DEFAULT_WEB_PENTESTER = """You are a web application penetration testing agent.
Test web applications for security vulnerabilities using methodical approaches."""


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROMPT LOADER TEST")
    print("="*60 + "\n")
    
    print("Available prompts:")
    for prompt_name in get_available_prompts():
        print(f"  - {prompt_name}")
    
    print("\nPrompt info:")
    for name, info in get_prompt_info().items():
        print(f"  {name}: {info['size']} bytes")
        if info['first_line']:
            print(f"    → {info['first_line'][:60]}...")
    
    print("\nTesting AgentPrompts class:")
    prompts = AgentPrompts()
    print(f"  Red team prompt length: {len(prompts.red_team())} chars")
    print(f"  Web pentester prompt length: {len(prompts.web_pentester())} chars")
    
    print("\n" + "="*60)
    print("Prompt loader tests completed!")
    print("="*60)
