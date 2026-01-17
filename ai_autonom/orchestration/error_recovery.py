#!/usr/bin/env python3
"""
Error Recovery
Handle agent failures gracefully - retry, fallback, escalate
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class RecoveryAction(Enum):
    """Possible recovery actions"""
    RETRY = "retry"
    RETRY_WITH_DIFFERENT_MODEL = "retry_different_model"
    SPAWN_DEBUGGER = "spawn_debugger"
    SIMPLIFY_TASK = "simplify_task"
    ESCALATE_TO_HUMAN = "escalate_human"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class RecoveryDecision:
    """A decision on how to handle an error"""
    action: RecoveryAction
    params: Dict[str, Any]
    reason: str


class ErrorRecovery:
    """
    Handle agent failures gracefully.
    Decides what to do when tasks fail - retry, fallback, escalate.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        escalation_callback: Optional[Callable] = None
    ):
        self.max_retries = max_retries
        self.escalation_callback = escalation_callback
        self.retry_counts: Dict[str, int] = {}  # task_id -> retry count
        self.error_history: List[Dict[str, Any]] = []
    
    def handle_failure(
        self,
        task: Dict[str, Any],
        error: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryDecision:
        """
        Decide recovery action based on error type
        
        Args:
            task: The failed task
            error: Error message/description
            context: Additional context (code, output, etc.)
        
        Returns:
            RecoveryDecision with action and params
        """
        task_id = task.get("id", "unknown")
        error_lower = error.lower()
        context = context or {}
        
        # Track retries
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1
        retries = self.retry_counts[task_id]
        
        # Log error
        self._log_error(task_id, error, retries)
        
        # Max retries exceeded
        if retries >= self.max_retries:
            return RecoveryDecision(
                action=RecoveryAction.ESCALATE_TO_HUMAN,
                params={
                    "task": task,
                    "error": error,
                    "retries": retries,
                    "history": self.get_error_history(task_id)
                },
                reason=f"Max retries ({self.max_retries}) exceeded for task {task_id}"
            )
        
        # Timeout errors
        if self._is_timeout_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={
                    "timeout_multiplier": 2.0,
                    "new_timeout": context.get("timeout", 30) * 2
                },
                reason="Timeout - retrying with longer timeout"
            )
        
        # Syntax/Compilation errors
        if self._is_syntax_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.SPAWN_DEBUGGER,
                params={
                    "error_log": error,
                    "code": context.get("code", ""),
                    "error_type": "syntax"
                },
                reason="Syntax error - spawning debugger agent to fix"
            )
        
        # Memory errors
        if self._is_memory_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_DIFFERENT_MODEL,
                params={
                    "prefer_smaller": True,
                    "max_vram": context.get("current_vram", 4.0) * 0.5
                },
                reason="Out of memory - switching to smaller model"
            )
        
        # Import/Module errors
        if self._is_import_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={
                    "install_deps": True,
                    "missing_module": self._extract_module_name(error)
                },
                reason="Missing module - installing dependencies and retrying"
            )
        
        # Rate limiting
        if self._is_rate_limit_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={
                    "delay_seconds": 60,
                    "backoff": True
                },
                reason="Rate limited - waiting and retrying"
            )
        
        # Network errors
        if self._is_network_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={
                    "delay_seconds": 5,
                    "check_connectivity": True
                },
                reason="Network error - retrying after delay"
            )
        
        # Model errors (Ollama specific)
        if self._is_model_error(error_lower):
            return RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_DIFFERENT_MODEL,
                params={
                    "fallback_model": self._get_fallback_model(task)
                },
                reason="Model error - trying different model"
            )
        
        # Generic retry for first failure
        if retries == 1:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={"attempt": retries + 1},
                reason=f"First failure - retrying (attempt {retries + 1})"
            )
        
        # Second failure - try simplifying
        if retries == 2:
            return RecoveryDecision(
                action=RecoveryAction.SIMPLIFY_TASK,
                params={
                    "original_task": task,
                    "simplification_hints": self._get_simplification_hints(task, error)
                },
                reason="Multiple failures - simplifying task"
            )
        
        # Unknown error - escalate
        return RecoveryDecision(
            action=RecoveryAction.ESCALATE_TO_HUMAN,
            params={"task": task, "error": error},
            reason="Unknown error pattern - escalating to human"
        )
    
    def _is_timeout_error(self, error: str) -> bool:
        patterns = ["timeout", "timed out", "deadline exceeded", "took too long"]
        return any(p in error for p in patterns)
    
    def _is_syntax_error(self, error: str) -> bool:
        patterns = ["syntax", "syntaxerror", "parse error", "unexpected token", "invalid syntax"]
        return any(p in error for p in patterns)
    
    def _is_memory_error(self, error: str) -> bool:
        patterns = ["memory", "oom", "out of memory", "cuda", "gpu memory", "memoryerror"]
        return any(p in error for p in patterns)
    
    def _is_import_error(self, error: str) -> bool:
        patterns = ["import", "modulenotfounderror", "no module named", "importerror"]
        return any(p in error for p in patterns)
    
    def _is_rate_limit_error(self, error: str) -> bool:
        patterns = ["rate limit", "429", "too many requests", "throttl"]
        return any(p in error for p in patterns)
    
    def _is_network_error(self, error: str) -> bool:
        patterns = ["connection", "network", "socket", "unreachable", "refused", "dns"]
        return any(p in error for p in patterns)
    
    def _is_model_error(self, error: str) -> bool:
        patterns = ["model not found", "model error", "ollama", "failed to load"]
        return any(p in error for p in patterns)
    
    def _extract_module_name(self, error: str) -> Optional[str]:
        """Extract module name from import error"""
        import re
        match = re.search(r"no module named ['\"]?(\w+)['\"]?", error.lower())
        if match:
            return match.group(1)
        return None
    
    def _get_fallback_model(self, task: Dict) -> str:
        """Get fallback model for a task"""
        # Default fallbacks
        fallbacks = ["qwen3:1.7b", "phi3:mini", "llama3.2:1b"]
        current = task.get("assigned_agent", "")
        
        for fb in fallbacks:
            if fb not in current:
                return fb
        return fallbacks[0]
    
    def _get_simplification_hints(self, task: Dict, error: str) -> List[str]:
        """Get hints for simplifying a task"""
        hints = []
        
        description = task.get("description", "").lower()
        
        if "and" in description:
            hints.append("Split into multiple smaller tasks")
        
        if "all" in description or "every" in description:
            hints.append("Focus on core functionality first")
        
        if "error handling" in description:
            hints.append("Implement basic version without comprehensive error handling")
        
        if not hints:
            hints.append("Reduce scope of requirements")
            hints.append("Remove optional features")
        
        return hints
    
    def _log_error(self, task_id: str, error: str, retry_count: int) -> None:
        """Log error for history"""
        self.error_history.append({
            "task_id": task_id,
            "error": error[:500],
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat()
        })
    
    def reset_retries(self, task_id: str) -> None:
        """Reset retry count for a task (on success)"""
        self.retry_counts[task_id] = 0
    
    def get_retry_count(self, task_id: str) -> int:
        """Get current retry count for a task"""
        return self.retry_counts.get(task_id, 0)
    
    def get_error_history(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get error history, optionally filtered by task"""
        if task_id:
            return [e for e in self.error_history if e["task_id"] == task_id]
        return self.error_history
    
    def clear_history(self) -> None:
        """Clear error history"""
        self.error_history.clear()
        self.retry_counts.clear()


class RecoveryExecutor:
    """
    Execute recovery actions
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
    
    def execute_recovery(
        self,
        decision: RecoveryDecision,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a recovery decision
        
        Returns:
            Result of recovery action
        """
        action = decision.action
        params = decision.params
        
        print(f"[RECOVERY] Executing: {action.value}")
        print(f"[RECOVERY] Reason: {decision.reason}")
        
        if action == RecoveryAction.RETRY:
            return self._execute_retry(task, params)
        
        elif action == RecoveryAction.RETRY_WITH_DIFFERENT_MODEL:
            return self._execute_model_switch(task, params)
        
        elif action == RecoveryAction.SPAWN_DEBUGGER:
            return self._execute_debugger(task, params)
        
        elif action == RecoveryAction.SIMPLIFY_TASK:
            return self._execute_simplification(task, params)
        
        elif action == RecoveryAction.ESCALATE_TO_HUMAN:
            return self._execute_escalation(task, params)
        
        elif action == RecoveryAction.SKIP:
            return {"action": "skip", "task_id": task.get("id")}
        
        elif action == RecoveryAction.ABORT:
            return {"action": "abort", "task_id": task.get("id")}
        
        return {"action": "unknown", "error": "Unknown recovery action"}
    
    def _execute_retry(self, task: Dict, params: Dict) -> Dict:
        """Execute simple retry"""
        import time
        
        delay = params.get("delay_seconds", 0)
        if delay:
            print(f"[RECOVERY] Waiting {delay}s before retry...")
            time.sleep(delay)
        
        return {
            "action": "retry",
            "task": task,
            "timeout": params.get("new_timeout"),
            "install_deps": params.get("install_deps", False)
        }
    
    def _execute_model_switch(self, task: Dict, params: Dict) -> Dict:
        """Switch to different model"""
        new_task = task.copy()
        
        if params.get("fallback_model"):
            # This would need to update the agent assignment
            new_task["model_override"] = params["fallback_model"]
        
        if params.get("prefer_smaller"):
            new_task["constraints"] = {"max_vram": params.get("max_vram", 2.0)}
        
        return {
            "action": "retry_with_model",
            "task": new_task,
            "new_model": params.get("fallback_model")
        }
    
    def _execute_debugger(self, task: Dict, params: Dict) -> Dict:
        """Spawn debugger to fix code"""
        return {
            "action": "spawn_debugger",
            "task": task,
            "error_log": params.get("error_log"),
            "code": params.get("code"),
            "debugger_prompt": f"Fix this error: {params.get('error_log', '')[:200]}"
        }
    
    def _execute_simplification(self, task: Dict, params: Dict) -> Dict:
        """Simplify task and retry"""
        simplified = task.copy()
        hints = params.get("simplification_hints", [])
        
        # Add simplification instruction to description
        simplified["description"] = (
            f"SIMPLIFIED VERSION: {task.get('description', '')}\n\n"
            f"Focus on basic functionality only. Hints: {'; '.join(hints)}"
        )
        
        return {
            "action": "retry_simplified",
            "task": simplified,
            "hints": hints
        }
    
    def _execute_escalation(self, task: Dict, params: Dict) -> Dict:
        """Escalate to human"""
        print("\n" + "="*60)
        print("HUMAN ESCALATION REQUIRED")
        print("="*60)
        print(f"\nTask: {task.get('id', 'unknown')}")
        print(f"Description: {task.get('description', '')[:200]}")
        print(f"\nError: {params.get('error', 'Unknown error')}")
        print(f"Retries: {params.get('retries', 0)}")
        print("="*60 + "\n")
        
        return {
            "action": "escalated",
            "task": task,
            "requires_human": True
        }


if __name__ == "__main__":
    # Test error recovery
    recovery = ErrorRecovery(max_retries=3)
    
    print("\n" + "="*60)
    print("ERROR RECOVERY TEST")
    print("="*60 + "\n")
    
    # Test different error types
    test_cases = [
        {"task": {"id": "task_1"}, "error": "Execution timed out after 30s"},
        {"task": {"id": "task_2"}, "error": "SyntaxError: invalid syntax at line 5"},
        {"task": {"id": "task_3"}, "error": "CUDA out of memory"},
        {"task": {"id": "task_4"}, "error": "ModuleNotFoundError: No module named 'pandas'"},
        {"task": {"id": "task_5"}, "error": "HTTP 429: Too Many Requests"},
    ]
    
    for case in test_cases:
        print(f"\nError: {case['error'][:50]}...")
        decision = recovery.handle_failure(case["task"], case["error"])
        print(f"  Action: {decision.action.value}")
        print(f"  Reason: {decision.reason}")
        print(f"  Params: {list(decision.params.keys())}")
