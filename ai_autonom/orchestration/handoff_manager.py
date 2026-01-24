"""
Handoff Manager - "Context Slicing" Logic
Handles the intelligent transfer of control between agents, ensuring that
context is filtered ("sliced") relevant to the new agent's role.
"""

from typing import Dict, Any, List, Optional
from ai_autonom.memory.knowledge_base import KnowledgeBase
from ai_autonom.core.agent_registry import AgentRegistry

class HandoffManager:
    def __init__(self, agent_registry: AgentRegistry, kb: KnowledgeBase):
        self.registry = agent_registry
        self.kb = kb

    def evaluate_handoff_request(
        self,
        current_agent_id: str,
        target_agent_id: str,
        context_message: str,
        dependency_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate and prepare a handoff request.
        1. Validate target agent exists.
        2. "Slice" the Knowledge Base to find relevant facts for the target.
        3. Construct the prompt/Task for the new agent.
        """
        
        # 1. Validation
        target_agent = None
        for agent in self.registry.get_all_agents():
            if agent.id == target_agent_id or agent.name.lower() == target_agent_id.lower() or target_agent_id.lower() in agent.id:
                target_agent = agent
                break
        
        if not target_agent:
            # Try to infer by capability
            best_agent = self.registry.find_agent_for_task(context_message)
            if best_agent and best_agent.id != current_agent_id:
                target_agent = best_agent
        
        if not target_agent:
             return {
                "approved": False,
                "reason": f"Target agent '{target_agent_id}' not found and no suitable alternative inferred."
            }

        if target_agent.id == current_agent_id:
             return {
                 "approved": False,
                 "reason": "You are already the best agent for this task. Continue execution."
             }

        # 2. Context Slicing
        # Filter KB based on target agent's specialization
        sliced_knowledge = self._slice_context(target_agent, context_message)

        # Build richer context block from dependency outputs/decisions/artifacts
        dep_lines: List[str] = []
        if dependency_context:
            prev_outputs = dependency_context.get("previous_outputs", {})
            if prev_outputs:
                dep_lines.append("[PREVIOUS OUTPUTS]")
                for dep_id, data in prev_outputs.items():
                    dep_lines.append(f"- {dep_id}: {data.get('output','')[:300]}")
            decisions = dependency_context.get("decisions", [])
            if decisions:
                dep_lines.append("\n[DECISIONS]")
                for d in decisions:
                    dep_lines.append(f"- {d}")
            artifacts = dependency_context.get("code_artifacts", [])
            if artifacts:
                dep_lines.append("\n[ARTIFACTS]")
                for art in artifacts[:5]:
                    dep_lines.append(f"- {art[:120]}")
        dep_context_str = "\n".join(dep_lines).strip()

        # 3. Decision
        return {
            "approved": True,
            "target_agent_id": target_agent.id,
            "target_agent_name": target_agent.name,
            "filtered_context": f"""
[TASK HANDOFF RECEIVED]
From: {current_agent_id}
Context: {context_message}

[RELEVANT KNOWLEDGE]
{sliced_knowledge}

[DEPENDENCY CONTEXT]
{dep_context_str or 'None provided'}
""",
            "tools_allowed": target_agent.tools
        }

    def _slice_context(self, target_agent, context_message: str) -> str:
        """
        Filter the global Knowledge Base to only return facts relevant 
        to the target agent's capabilities.
        """
        kb_dump = self.kb.get_summary() # Get fully formatted summary first
        # Ideally, we parse the objects, but for MVP we do string/tag matching
        
        # This is where "Pattern: CTF_SWARM" and "Context Slicing" happens.
        
        slicing_rules = {
            "web_pentester_agent": ["http", "https", "url", "web", "apache", "nginx", "port 80", "port 443", "api", "endpoint"],
            "red_team_agent": ["ssh", "ftp", "rdp", "smb", "credential", "password", "hash", "nmap", "port"],
            "reporting_agent": ["vulnerability", "finding", "critical", "high", "medium", "low"]
        }
        
        keywords = slicing_rules.get(target_agent.id, [])
        if not keywords: 
            return kb_dump # Fallback: give them everything
            
        # Parse the structured data directly for better filtering
        # (Accessing private lists of KB for filtering - pythonic)
        
        sliced_kb = ""
        
        # 1. Assets
        relevant_assets = []
        for asset in self.kb.assets:
            # Check if asset has relevant open ports or tags
            asset_str = str(asset).lower()
            if any(k in asset_str for k in keywords):
                relevant_assets.append(asset)
                
        if relevant_assets:
            sliced_kb += "\n[RELEVANT ASSETS]\n"
            for a in relevant_assets:
                sliced_kb += f"- {a['ip']} ({a.get('hostname','')}) Tags: {a.get('tags')}\n"

        # 2. Credentials (Always relevant for Red Team / Web)
        if "password" in keywords or "credential" in keywords:
            if self.kb.credentials:
                sliced_kb += "\n[KNOWN CREDENTIALS]\n"
                for c in self.kb.credentials:
                     sliced_kb += f"- {c['username']} @ {c['service']}\n"

        # 3. Findings
        relevant_findings = []
        for f in self.kb.findings:
            f_str = f"{f.title} {f.details} {f.type}".lower()
            if any(k in f_str for k in keywords):
                 relevant_findings.append(f)
                 
        if relevant_findings:
            sliced_kb += "\n[PRIOR FINDINGS]\n"
            for f in relevant_findings:
                sliced_kb += f"- [{f.severity}] {f.title}: {f.details[:100]}...\n"

        # If slice is empty, default to full dump
        if not sliced_kb.strip():
            return kb_dump
            
        return sliced_kb
