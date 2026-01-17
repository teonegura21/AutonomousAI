"""
CAI Handoffs Adapter
Manages dynamic agent-to-agent delegation logic.
"""

from typing import Dict, Any, Optional
from ..core.agent_registry import AgentRegistry

class HandoffManager:
    """
    Manages dynamic handoffs between agents.
    Allows agents to request help from specialists.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        
    def evaluate_handoff_request(self, current_agent: str, target_agent: str, context: str) -> Dict[str, Any]:
        """
        Evaluate if a handoff should be permitted.
        Returns: {
            "approved": bool,
            "target_agent_id": str,
            "filtered_context": str,
            "reason": str
        }
        """
        # Verify target agent exists
        agents = self.registry.get_all_agents()
        target = next((a for a in agents if a.id == target_agent or a.name.lower() == target_agent.lower()), None)
        
        if not target:
            return {
                "approved": False,
                "reason": f"Target agent '{target_agent}' not found in registry."
            }
            
        # Basic approval logic (can be enhanced with LLM evaluation later)
        # For now, allow handoffs to valid agents
        
        # Filter context (Context Slicing)
        # We don't want to dump the entire conversation history on the new agent.
        # We only pass the specific request context.
        filtered_context = f"""
[HANDOFF FROM {current_agent}]
Request: {context}

You have been activated to assist with this specific request.
Execute the task and return your findings to {current_agent}.
"""
        
        return {
            "approved": True,
            "target_agent_id": target.id,
            "target_agent_name": target.name,
            "filtered_context": filtered_context,
            "reason": "Valid handoff target"
        }