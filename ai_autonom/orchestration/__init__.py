"""
Orchestration module - Core orchestration components
"""

# Import components that don't require ollama
from .intent_analyzer import IntentAnalyzer, IntentAnalysisResult
from .agent_messaging import AgentMessageBus, AgentMessage, MessageType
from .error_recovery import ErrorRecovery, RecoveryAction, RecoveryDecision
from .human_checkpoint import (
    HumanCheckpointManager,
    CheckpointDecision,
    CheckpointType,
    CheckpointContext,
    CheckpointResult,
    RiskLevel
)
from .testing_workflow import TestingWorkflow, TestResult
from .langgraph_workflow import MultiAgentWorkflow, WorkflowState

# NemotronOrchestrator requires ollama, import lazily
def get_orchestrator():
    """Get NemotronOrchestrator (requires ollama)"""
    from .nemotron_orchestrator import NemotronOrchestrator
    return NemotronOrchestrator

__all__ = [
    'IntentAnalyzer',
    'IntentAnalysisResult',
    'AgentMessageBus',
    'AgentMessage',
    'MessageType',
    'ErrorRecovery',
    'RecoveryAction',
    'RecoveryDecision',
    'HumanCheckpointManager',
    'CheckpointDecision',
    'CheckpointType',
    'CheckpointContext',
    'CheckpointResult',
    'RiskLevel',
    'TestingWorkflow',
    'TestResult',
    'MultiAgentWorkflow',
    'WorkflowState',
    'get_orchestrator'
]

