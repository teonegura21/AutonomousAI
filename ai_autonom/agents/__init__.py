"""
AI Autonom Agents Module
Includes CAI-adapted security agents, custom agents, and prompt loader
"""

from typing import Dict, List, Any

# Import prompt loader (ported from CAI)
try:
    from .prompt_loader import (
        load_prompt,
        load_prompt_safe,
        get_available_prompts,
        get_prompt_info,
        create_agent_prompt,
        AgentPrompts,
    )
    PROMPT_LOADER_AVAILABLE = True
except ImportError:
    PROMPT_LOADER_AVAILABLE = False
    AgentPrompts = None

# Import all agent definitions
try:
    from .reverse_engineering_agent import REVERSE_ENGINEERING_AGENT
    from .dfir_agent import DFIR_AGENT
    from .cai_security_agents import (
        MEMORY_ANALYSIS_AGENT,
        BUG_BOUNTY_AGENT,
        THOUGHT_AGENT,
        REPORTER_AGENT
    )
    
    # Collect all agents
    ALL_CAI_AGENTS = {
        "reverse_engineering_agent": REVERSE_ENGINEERING_AGENT,
        "dfir_agent": DFIR_AGENT,
        "memory_analysis_agent": MEMORY_ANALYSIS_AGENT,
        "bug_bounty_agent": BUG_BOUNTY_AGENT,
        "thought_agent": THOUGHT_AGENT,
        "reporter_agent": REPORTER_AGENT
    }
    
except ImportError as e:
    print(f"Warning: Could not import some CAI agents: {e}")
    ALL_CAI_AGENTS = {}


def get_all_cai_agents() -> Dict[str, Dict[str, Any]]:
    """
    Returns all CAI-adapted agents
    
    Returns:
        Dictionary of agent_id -> agent_definition
    """
    return ALL_CAI_AGENTS


def get_agent_by_id(agent_id: str) -> Dict[str, Any]:
    """
    Get specific agent by ID
    
    Args:
        agent_id: The unique identifier for the agent
        
    Returns:
        Agent definition dict or None
    """
    return ALL_CAI_AGENTS.get(agent_id)


def list_available_agents() -> List[str]:
    """
    List all available agent IDs
    
    Returns:
        List of agent IDs
    """
    return list(ALL_CAI_AGENTS.keys())


def get_agents_by_capability(capability: str) -> List[Dict[str, Any]]:
    """
    Get all agents that have a specific capability
    
    Args:
        capability: The capability to search for
        
    Returns:
        List of agent definitions
    """
    return [
        agent for agent in ALL_CAI_AGENTS.values()
        if capability in agent.get("capabilities", [])
    ]


__all__ = [
    "ALL_CAI_AGENTS",
    "get_all_cai_agents",
    "get_agent_by_id",
    "list_available_agents",
    "get_agents_by_capability"
]
