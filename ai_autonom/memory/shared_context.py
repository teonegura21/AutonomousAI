#!/usr/bin/env python3
"""
Shared Context Manager

Unified context sharing between agents/threads with vector memory integration.
Provides semantic search across all agent contexts and automatic context slicing.

Features:
- Store/retrieve context by process/agent ID
- Semantic search across all contexts (via VectorMemoryStore)
- Automatic context merging for synthesis tasks
- Push notifications on context updates
- Context expiration and cleanup

Usage:
    from ai_autonom.memory.shared_context import SharedContextManager, get_shared_context
    
    ctx = get_shared_context()
    
    # Store agent context
    ctx.store("agent_1", {"target": "10.0.0.1", "findings": [...]})
    
    # Search across all contexts
    results = ctx.search("SQL injection vulnerabilities")
    
    # Get merged context for synthesis
    merged = ctx.merge(["agent_1", "agent_2", "agent_3"])
"""

import json
import time
import threading
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import vector store if available
try:
    from ai_autonom.memory.vector_store import VectorMemoryStore
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    VectorMemoryStore = None

# Import IPC broker
try:
    from ai_autonom.orchestration.ipc_broker import get_broker, IPCMessage
    IPC_AVAILABLE = True
except ImportError:
    IPC_AVAILABLE = False
    get_broker = None


@dataclass
class AgentContext:
    """Context data for a single agent."""
    agent_id: str
    context: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "parent_id": self.parent_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        return cls(**data)


# Keys that are always included in context transfers
ESSENTIAL_KEYS = {
    'target_ip', 'target_host', 'objective', 'current_stage',
    'credentials', 'session_tokens', 'flags_found', 'critical_findings'
}

# Agent-specific context preferences
AGENT_CONTEXT_PREFERENCES = {
    'recon_agent': {'ports', 'services', 'technologies', 'domains', 'ips'},
    'web_agent': {'urls', 'parameters', 'cookies', 'forms', 'endpoints'},
    'exploit_agent': {'vulnerabilities', 'cves', 'exploits', 'payloads'},
    'priv_esc_agent': {'user_context', 'suid', 'kernel', 'processes', 'capabilities'},
    'lateral_agent': {'hosts', 'credentials', 'ssh_keys', 'network_map'},
    'reporting_agent': {'findings', 'evidence', 'timeline', 'severity'},
}


class SharedContextManager:
    """
    Manages shared context across all agents with vector memory integration.
    
    Provides:
    - Context storage per agent/process
    - Semantic search across all contexts
    - Context slicing for efficient handoffs
    - Automatic merge for synthesis tasks
    - Push notifications on updates
    
    Example:
        manager = SharedContextManager()
        
        # Store context
        manager.store("scanner_agent", {
            "target_ip": "10.0.0.1",
            "open_ports": [22, 80, 443],
            "services": {"22": "SSH", "80": "nginx"}
        })
        
        # Search semantically
        results = manager.search("web services running")
        
        # Get sliced context for specific agent
        sliced = manager.get_for_agent("exploit_agent", "scanner_agent")
    """
    
    def __init__(
        self,
        enable_vector: bool = True,
        enable_ipc: bool = True,
        vector_persist_dir: str = ".runtime/data/chromadb"
    ):
        """
        Initialize shared context manager.
        
        Args:
            enable_vector: Enable vector memory for semantic search
            enable_ipc: Enable IPC for push notifications
            vector_persist_dir: ChromaDB persistence directory
        """
        # In-memory context store
        self._contexts: Dict[str, AgentContext] = {}
        self._lock = threading.RLock()
        
        # Vector store for semantic search
        self._vector_store: Optional[VectorMemoryStore] = None
        if enable_vector and VECTOR_AVAILABLE:
            try:
                self._vector_store = VectorMemoryStore(persist_dir=vector_persist_dir)
                print("[SharedContext] Vector memory enabled")
            except Exception as e:
                print(f"[SharedContext] Vector memory failed: {e}")
        
        # IPC broker for notifications
        self._broker = None
        if enable_ipc and IPC_AVAILABLE:
            try:
                self._broker = get_broker()
                self._broker.subscribe("_context_updates", self._handle_context_update)
                print("[SharedContext] IPC notifications enabled")
            except Exception as e:
                print(f"[SharedContext] IPC failed: {e}")
        
        # Update callbacks
        self._callbacks: List[Callable[[str, Dict], None]] = []
    
    # =========================================================================
    # CORE STORAGE METHODS
    # =========================================================================
    
    def store(
        self,
        agent_id: str,
        context: Dict[str, Any],
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_in_vector: bool = True
    ) -> int:
        """
        Store or update context for an agent.
        
        Args:
            agent_id: Unique agent identifier
            context: Context dictionary to store
            parent_id: Optional parent agent (for hierarchy)
            metadata: Additional metadata
            index_in_vector: Whether to index in vector store
        
        Returns:
            New version number
        """
        with self._lock:
            if agent_id in self._contexts:
                existing = self._contexts[agent_id]
                existing.context.update(context)
                existing.updated_at = datetime.now().isoformat()
                existing.version += 1
                if metadata:
                    existing.metadata.update(metadata)
                version = existing.version
            else:
                self._contexts[agent_id] = AgentContext(
                    agent_id=agent_id,
                    context=context,
                    parent_id=parent_id,
                    metadata=metadata or {}
                )
                version = 1
        
        # Index in vector store for semantic search
        if index_in_vector and self._vector_store:
            self._index_context(agent_id, context)
        
        # Notify via IPC
        if self._broker:
            self._broker.set_shared(
                f"ctx_{agent_id}",
                self._contexts[agent_id].to_dict(),
                notify=True
            )
        
        # Notify local callbacks
        for callback in self._callbacks:
            try:
                callback(agent_id, context)
            except Exception:
                pass
        
        return version
    
    def get(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a specific agent."""
        with self._lock:
            if agent_id in self._contexts:
                return self._contexts[agent_id].context.copy()
        
        # Try IPC broker if not in local cache
        if self._broker:
            data = self._broker.get_shared(f"ctx_{agent_id}")
            if data:
                ctx = AgentContext.from_dict(data)
                with self._lock:
                    self._contexts[agent_id] = ctx
                return ctx.context.copy()
        
        return None
    
    def get_full(self, agent_id: str) -> Optional[AgentContext]:
        """Get full context object including metadata."""
        with self._lock:
            if agent_id in self._contexts:
                return self._contexts[agent_id]
        return None
    
    def delete(self, agent_id: str) -> bool:
        """Delete context for an agent."""
        with self._lock:
            if agent_id in self._contexts:
                del self._contexts[agent_id]
                
                if self._broker:
                    self._broker.delete_shared(f"ctx_{agent_id}")
                
                return True
        return False
    
    def list_agents(self) -> List[str]:
        """List all agents with stored context."""
        with self._lock:
            return list(self._contexts.keys())
    
    # =========================================================================
    # CONTEXT SLICING
    # =========================================================================
    
    def get_for_agent(
        self,
        target_agent: str,
        source_agent: str,
        additional_keys: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Get sliced context optimized for target agent.
        
        Only includes keys relevant to the target agent's capabilities.
        
        Args:
            target_agent: Agent that will receive the context
            source_agent: Agent whose context to slice
            additional_keys: Extra keys to include
        
        Returns:
            Sliced context dictionary
        """
        source_context = self.get(source_agent)
        if not source_context:
            return {}
        
        # Get relevant keys for target
        agent_keys = AGENT_CONTEXT_PREFERENCES.get(target_agent, set())
        relevant_keys = ESSENTIAL_KEYS | agent_keys
        if additional_keys:
            relevant_keys |= additional_keys
        
        # Filter context
        return {k: v for k, v in source_context.items() if k in relevant_keys}
    
    def get_combined_for_agent(
        self,
        target_agent: str,
        source_agents: List[str]
    ) -> Dict[str, Any]:
        """
        Get combined sliced context from multiple sources.
        
        Args:
            target_agent: Agent that will receive the context
            source_agents: List of agents to gather context from
        
        Returns:
            Combined sliced context
        """
        combined = {}
        
        for source in source_agents:
            sliced = self.get_for_agent(target_agent, source)
            combined.update(sliced)
        
        return combined
    
    # =========================================================================
    # VECTOR SEARCH
    # =========================================================================
    
    def _index_context(self, agent_id: str, context: Dict[str, Any]) -> None:
        """Index context in vector store for semantic search."""
        if not self._vector_store:
            return
        
        # Convert context to searchable text
        text_parts = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, default=str)[:500]
            else:
                value_str = str(value)[:500]
            text_parts.append(f"{key}: {value_str}")
        
        content = "\n".join(text_parts)
        
        # Store in vector database
        try:
            self._vector_store.store_conversation(
                task_id=f"context_{agent_id}",
                agent_id=agent_id,
                content=content,
                role="context"
            )
        except Exception as e:
            print(f"[SharedContext] Vector indexing failed: {e}")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        agent_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across all agent contexts.
        
        Args:
            query: Natural language query
            n_results: Maximum results to return
            agent_filter: Only search specific agents
        
        Returns:
            List of matching context entries with scores
        """
        if not self._vector_store:
            # Fallback to keyword search
            return self._keyword_search(query, n_results, agent_filter)
        
        try:
            results = self._vector_store.semantic_search(
                query=query,
                collection="conversations",
                n_results=n_results
            )
            
            # Filter by agent if specified
            if agent_filter:
                results = [
                    r for r in results 
                    if any(agent in r.get('metadata', {}).get('agent_id', '') 
                           for agent in agent_filter)
                ]
            
            return results
        except Exception as e:
            print(f"[SharedContext] Search failed: {e}")
            return []
    
    def _keyword_search(
        self,
        query: str,
        n_results: int,
        agent_filter: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based search fallback."""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            agents = agent_filter or list(self._contexts.keys())
            
            for agent_id in agents:
                if agent_id not in self._contexts:
                    continue
                
                ctx = self._contexts[agent_id]
                ctx_str = json.dumps(ctx.context, default=str).lower()
                
                # Simple keyword matching
                if query_lower in ctx_str:
                    results.append({
                        "agent_id": agent_id,
                        "context": ctx.context,
                        "score": ctx_str.count(query_lower)
                    })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    # =========================================================================
    # CONTEXT MERGING
    # =========================================================================
    
    def merge(
        self,
        agent_ids: List[str],
        strategy: str = "latest",
        conflict_keys: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Merge contexts from multiple agents.
        
        Args:
            agent_ids: Agents to merge from
            strategy: 'latest' (newest wins), 'first' (first wins), 'combine' (lists)
            conflict_keys: Per-key strategy override {key: strategy}
        
        Returns:
            Merged context dictionary
        """
        merged = {}
        conflict_keys = conflict_keys or {}
        
        # Get all contexts with timestamps
        contexts_with_time = []
        for agent_id in agent_ids:
            ctx = self.get_full(agent_id)
            if ctx:
                contexts_with_time.append((ctx.updated_at, agent_id, ctx.context))
        
        # Sort by timestamp (newest last for 'latest' strategy)
        contexts_with_time.sort(key=lambda x: x[0])
        
        for _, agent_id, context in contexts_with_time:
            for key, value in context.items():
                key_strategy = conflict_keys.get(key, strategy)
                
                if key not in merged:
                    merged[key] = value
                elif key_strategy == 'latest':
                    merged[key] = value
                elif key_strategy == 'first':
                    pass  # Keep existing
                elif key_strategy == 'combine':
                    # Combine lists/sets
                    existing = merged[key]
                    if isinstance(existing, list) and isinstance(value, list):
                        merged[key] = list(set(existing + value))
                    elif isinstance(existing, dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = [existing, value] if existing != value else existing
        
        return merged
    
    def synthesize_findings(
        self,
        agent_ids: List[str],
        include_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Create synthesized summary of all findings.
        
        Args:
            agent_ids: Agents to gather findings from
            include_evidence: Include supporting evidence
        
        Returns:
            Synthesized findings dictionary
        """
        synthesis = {
            "timestamp": datetime.now().isoformat(),
            "sources": agent_ids,
            "vulnerabilities": [],
            "credentials": [],
            "hosts": [],
            "services": [],
            "timeline": [],
            "critical_findings": []
        }
        
        for agent_id in agent_ids:
            ctx = self.get(agent_id)
            if not ctx:
                continue
            
            # Gather vulnerabilities
            if 'vulnerabilities' in ctx:
                synthesis["vulnerabilities"].extend(ctx['vulnerabilities'])
            if 'cves' in ctx:
                synthesis["vulnerabilities"].extend(ctx['cves'])
            
            # Gather credentials
            if 'credentials' in ctx:
                synthesis["credentials"].extend(ctx['credentials'])
            
            # Gather hosts/IPs
            if 'hosts' in ctx:
                synthesis["hosts"].extend(ctx['hosts'])
            if 'ips' in ctx:
                synthesis["hosts"].extend(ctx['ips'])
            
            # Gather services
            if 'services' in ctx:
                if isinstance(ctx['services'], dict):
                    synthesis["services"].append(ctx['services'])
                else:
                    synthesis["services"].extend(ctx['services'])
            
            # Gather timeline events
            if 'timeline' in ctx:
                synthesis["timeline"].extend(ctx['timeline'])
            
            # Gather critical findings
            if 'critical_findings' in ctx:
                synthesis["critical_findings"].extend(ctx['critical_findings'])
            if 'flags_found' in ctx:
                synthesis["critical_findings"].extend(ctx['flags_found'])
        
        # Deduplicate
        for key in ['vulnerabilities', 'credentials', 'hosts']:
            synthesis[key] = list(set(
                json.dumps(x, default=str) if isinstance(x, dict) else str(x)
                for x in synthesis[key]
            ))
        
        return synthesis
    
    # =========================================================================
    # NOTIFICATION HANDLING
    # =========================================================================
    
    def _handle_context_update(self, message: 'IPCMessage') -> None:
        """Handle context update notification from IPC."""
        key = message.payload.get("key", "")
        action = message.payload.get("action")
        
        if not key.startswith("ctx_"):
            return
        
        agent_id = key[4:]  # Remove "ctx_" prefix
        
        if action == "set":
            # Refresh from IPC
            data = self._broker.get_shared(key)
            if data:
                ctx = AgentContext.from_dict(data)
                with self._lock:
                    self._contexts[agent_id] = ctx
        elif action == "delete":
            with self._lock:
                self._contexts.pop(agent_id, None)
    
    def on_update(self, callback: Callable[[str, Dict], None]) -> None:
        """Register callback for context updates."""
        self._callbacks.append(callback)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        with self._lock:
            agent_count = len(self._contexts)
            total_keys = sum(len(c.context) for c in self._contexts.values())
        
        return {
            "agents_with_context": agent_count,
            "total_context_keys": total_keys,
            "vector_enabled": self._vector_store is not None,
            "ipc_enabled": self._broker is not None
        }
    
    def export_all(self) -> Dict[str, Any]:
        """Export all contexts for backup/transfer."""
        with self._lock:
            return {
                agent_id: ctx.to_dict()
                for agent_id, ctx in self._contexts.items()
            }
    
    def import_all(self, data: Dict[str, Any]) -> int:
        """Import contexts from backup. Returns count imported."""
        count = 0
        for agent_id, ctx_data in data.items():
            try:
                ctx = AgentContext.from_dict(ctx_data)
                with self._lock:
                    self._contexts[agent_id] = ctx
                count += 1
            except Exception:
                continue
        return count
    
    def clear(self) -> int:
        """Clear all contexts. Returns count cleared."""
        with self._lock:
            count = len(self._contexts)
            self._contexts.clear()
        return count


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_context: Optional[SharedContextManager] = None
_context_lock = threading.Lock()


def get_shared_context() -> SharedContextManager:
    """Get the global shared context manager (singleton)."""
    global _global_context
    
    with _context_lock:
        if _global_context is None:
            _global_context = SharedContextManager()
        return _global_context


def reset_shared_context() -> None:
    """Reset the global shared context (for testing)."""
    global _global_context
    
    with _context_lock:
        if _global_context:
            _global_context.clear()
        _global_context = None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SHARED CONTEXT MANAGER TEST")
    print("="*60 + "\n")
    
    # Initialize without vector/IPC for testing
    ctx = SharedContextManager(enable_vector=False, enable_ipc=False)
    
    # Test storing context
    print("Storing agent contexts...")
    ctx.store("scanner_agent", {
        "target_ip": "10.0.0.1",
        "ports": [22, 80, 443],
        "services": {"22": "SSH", "80": "nginx", "443": "nginx"}
    })
    
    ctx.store("exploit_agent", {
        "target_ip": "10.0.0.1",
        "vulnerabilities": ["CVE-2021-44228", "CVE-2022-22965"],
        "exploits_found": 2
    })
    
    ctx.store("priv_esc_agent", {
        "target_ip": "10.0.0.1",
        "user_context": "www-data",
        "suid": ["/usr/bin/sudo", "/usr/bin/pkexec"],
        "kernel": "5.4.0-42-generic"
    })
    
    print(f"  Stored {len(ctx.list_agents())} agent contexts")
    
    # Test slicing
    print("\nTesting context slicing...")
    sliced = ctx.get_for_agent("exploit_agent", "scanner_agent")
    print(f"  Scanner -> Exploit: {list(sliced.keys())}")
    
    # Test merging
    print("\nTesting context merging...")
    merged = ctx.merge(["scanner_agent", "exploit_agent", "priv_esc_agent"])
    print(f"  Merged keys: {list(merged.keys())}")
    
    # Test synthesis
    print("\nTesting findings synthesis...")
    synthesis = ctx.synthesize_findings(["scanner_agent", "exploit_agent"])
    print(f"  Vulnerabilities: {synthesis['vulnerabilities']}")
    
    # Test search
    print("\nTesting keyword search...")
    results = ctx.search("nginx")
    print(f"  Found {len(results)} results for 'nginx'")
    
    # Stats
    print(f"\nStats: {ctx.get_stats()}")
    
    print("\n" + "="*60)
    print("Shared context tests completed!")
    print("="*60)
