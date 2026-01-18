"""
Memory module - Task Memory, Vector Store, and Shared Context
"""

from .task_memory import TaskMemory, TaskContext
from .vector_store import VectorMemoryStore

# Import Shared Context Manager
try:
    from .shared_context import (
        SharedContextManager,
        get_shared_context,
        reset_shared_context,
        AgentContext,
        ESSENTIAL_KEYS,
        AGENT_CONTEXT_PREFERENCES,
    )
    SHARED_CONTEXT_AVAILABLE = True
except ImportError:
    SHARED_CONTEXT_AVAILABLE = False
    SharedContextManager = None

__all__ = [
    'TaskMemory',
    'TaskContext',
    'VectorMemoryStore',
    # Shared Context
    'SharedContextManager',
    'get_shared_context',
    'AgentContext',
    'SHARED_CONTEXT_AVAILABLE',
]

