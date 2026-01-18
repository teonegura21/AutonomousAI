"""
CAI Handoffs Adapter
Manages dynamic agent-to-agent delegation with context slicing.

Ported from CAI with adaptations for ai_autonom framework.
"""

from typing import Dict, Any, Optional, Set, List
from ..core.agent_registry import AgentRegistry


# Context keys that are always passed in handoffs
ESSENTIAL_KEYS = {'target_ip', 'credentials', 'objective', 'current_stage', 'flags_found'}

# Agent-specific context requirements
AGENT_CONTEXT_MAP = {
    'web_agent': {'ports', 'web_technologies', 'directories', 'urls'},
    'web_pentester': {'ports', 'web_technologies', 'directories', 'urls', 'cookies'},
    'exploit_agent': {'vulnerabilities', 'service_versions', 'cve_ids'},
    'priv_esc_agent': {'user_context', 'suid_binaries', 'kernel_version', 'processes'},
    'lateral_movement': {'hosts', 'credentials', 'ssh_keys', 'network_map'},
    'reporting_agent': {'findings', 'vulnerabilities', 'evidence', 'timeline'},
    'dfir_agent': {'artifacts', 'logs', 'timeline', 'iocs'},
    'triage_agent': {'vulnerabilities', 'scan_results', 'false_positives'},
}


def slice_context_for_handoff(
    full_context: Dict[str, Any], 
    target_agent: str,
    additional_keys: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Filter context to only include relevant information for target agent.
    
    This prevents overwhelming the next agent with irrelevant data and
    reduces token usage while preserving essential information.
    
    Args:
        full_context: Complete context dictionary
        target_agent: Target agent identifier
        additional_keys: Optional extra keys to include
    
    Returns:
        Filtered context dictionary
    
    Example:
        >>> context = {'target_ip': '10.0.0.1', 'kernel_version': '5.4', 
        ...            'large_scan_output': '...thousands of lines...'}
        >>> sliced = slice_context_for_handoff(context, 'exploit_agent')
        >>> 'large_scan_output' in sliced
        False
    """
    # Get agent-specific keys or empty set
    agent_keys = AGENT_CONTEXT_MAP.get(target_agent.lower(), set())
    
    # Combine all relevant keys
    relevant_keys = ESSENTIAL_KEYS | agent_keys
    if additional_keys:
        relevant_keys |= additional_keys
    
    # Filter context
    return {k: v for k, v in full_context.items() if k in relevant_keys}


def summarize_context(context: Dict[str, Any], max_value_length: int = 500) -> Dict[str, Any]:
    """
    Summarize context by truncating long values.
    
    Args:
        context: Context dictionary
        max_value_length: Maximum length for string values
    
    Returns:
        Summarized context
    """
    summarized = {}
    for key, value in context.items():
        if isinstance(value, str) and len(value) > max_value_length:
            summarized[key] = value[:max_value_length] + f"... [truncated, {len(value)} chars total]"
        elif isinstance(value, list) and len(value) > 10:
            summarized[key] = value[:10] + [f"... and {len(value) - 10} more items"]
        elif isinstance(value, dict) and len(value) > 10:
            summarized[key] = dict(list(value.items())[:10])
            summarized[key]["_truncated"] = f"{len(value) - 10} more keys"
        else:
            summarized[key] = value
    return summarized


class HandoffManager:
    """
    Manages dynamic handoffs between agents with context slicing.
    Allows agents to request help from specialists while efficiently
    managing context transfer.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.handoff_history: List[Dict[str, Any]] = []
        
    def evaluate_handoff_request(
        self, 
        current_agent: str, 
        target_agent: str, 
        context: str,
        full_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if a handoff should be permitted.
        
        Args:
            current_agent: ID of requesting agent
            target_agent: ID of target agent
            context: Text description of handoff request
            full_context: Optional full context dict for slicing
        
        Returns:
            Dict with 'approved', 'target_agent_id', 'filtered_context', 'reason'
        """
        # Verify target agent exists
        agents = self.registry.get_all_agents()
        target = next((a for a in agents if a.id == target_agent or a.name.lower() == target_agent.lower()), None)
        
        if not target:
            return {
                "approved": False,
                "reason": f"Target agent '{target_agent}' not found in registry."
            }
        
        # Slice context if full_context provided
        if full_context:
            sliced = slice_context_for_handoff(full_context, target_agent)
            sliced = summarize_context(sliced)
            context_section = "\n".join([f"- {k}: {v}" for k, v in sliced.items()])
        else:
            context_section = context
            
        # Create filtered handoff context
        filtered_context = f"""
[HANDOFF FROM {current_agent}]
Target: {target.name}

Request: {context}

Relevant Context:
{context_section}

You have been activated to assist with this specific request.
Execute the task and return your findings to {current_agent}.
"""
        
        # Record handoff
        self.handoff_history.append({
            "from": current_agent,
            "to": target.id,
            "context_length": len(filtered_context),
            "timestamp": __import__('time').time()
        })
        
        return {
            "approved": True,
            "target_agent_id": target.id,
            "target_agent_name": target.name,
            "filtered_context": filtered_context,
            "reason": "Valid handoff target"
        }
    
    def get_handoff_history(self) -> List[Dict[str, Any]]:
        """Get history of all handoffs."""
        return self.handoff_history
    
    def create_specialist_request(
        self,
        specialist_type: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Create a formatted request for a specialist agent.
        
        Args:
            specialist_type: Type of specialist needed
            task_description: What needs to be done
            context: Relevant context data
        
        Returns:
            Formatted request string
        """
        sliced = slice_context_for_handoff(context, specialist_type)
        sliced = summarize_context(sliced)
        
        context_items = "\n".join([f"  {k}: {v}" for k, v in sliced.items()])
        
        return f"""
SPECIALIST REQUEST: {specialist_type}
TASK: {task_description}

CONTEXT:
{context_items}

Please complete this task and report findings.
"""