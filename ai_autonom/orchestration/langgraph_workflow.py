#!/usr/bin/env python3
"""
LangGraph Workflow Engine with CAI Pattern Integration
Graph-based workflow execution with checkpointing, state management, and agentic patterns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional, TypedDict, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import sqlite3

# Import CAI pattern system
try:
    from patterns.cai_patterns import AgenticPattern, PatternExecutor, PatternLibrary
    from patterns.handoffs import HandoffManager, HandoffContext, HandoffStatus
    CAI_PATTERNS_AVAILABLE = True
except ImportError:
    CAI_PATTERNS_AVAILABLE = False
    print("[WARNING] CAI patterns not available")

# Try to import langgraph (optional dependency)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[WARNING] LangGraph not installed. Using fallback workflow engine.")


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowState:
    """State passed through the workflow"""
    workflow_id: str
    current_task_id: Optional[str] = None
    task_status: Dict[str, str] = field(default_factory=dict)
    task_outputs: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "current_task_id": self.current_task_id,
            "task_status": self.task_status,
            "task_outputs": {k: v[:500] for k, v in self.task_outputs.items()},
            "context": self.context,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        return cls(
            workflow_id=data.get("workflow_id", "unknown"),
            current_task_id=data.get("current_task_id"),
            task_status=data.get("task_status", {}),
            task_outputs=data.get("task_outputs", {}),
            context=data.get("context", {}),
            errors=data.get("errors", []),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at")
        )


class WorkflowCheckpointer:
    """
    Checkpoint management for workflow state
    Allows pausing and resuming workflows
    """
    
    def __init__(self, db_path: str = ".runtime/data/workflow_checkpoints.db"):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self):
        """Initialize database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                workflow_id TEXT PRIMARY KEY,
                state_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save(self, state: WorkflowState) -> None:
        """Save workflow state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        state_json = json.dumps(state.to_dict())
        
        cursor.execute("""
            INSERT OR REPLACE INTO checkpoints 
            (workflow_id, state_json, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM checkpoints WHERE workflow_id = ?), ?), ?)
        """, (state.workflow_id, state_json, state.workflow_id, now, now))
        
        conn.commit()
        conn.close()
    
    def load(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT state_json FROM checkpoints WHERE workflow_id = ?",
            (workflow_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = json.loads(row[0])
            return WorkflowState.from_dict(data)
        return None
    
    def delete(self, workflow_id: str) -> bool:
        """Delete checkpoint"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE workflow_id = ?", (workflow_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved checkpoints"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT workflow_id, created_at, updated_at FROM checkpoints")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"workflow_id": r[0], "created_at": r[1], "updated_at": r[2]}
            for r in rows
        ]


class MultiAgentWorkflow:
    """
    Workflow engine for multi-agent task execution with CAI pattern support.
    This is Layer 2 of the architecture (Execution Controller).
    
    Supports:
    - DAG-based task execution
    - CAI agentic patterns (swarm, chain, recursive, etc.)
    - Agent handoffs with callbacks
    - Parallel execution of independent tasks
    - Checkpointing and resume
    - Error handling and recovery
    """
    
    def __init__(
        self,
        orchestrator=None,
        checkpointer: Optional[WorkflowCheckpointer] = None,
        message_bus=None
    ):
        self.orchestrator = orchestrator
        self.checkpointer = checkpointer or WorkflowCheckpointer()
        self.tasks: List[Dict[str, Any]] = []
        self.task_map: Dict[str, Dict[str, Any]] = {}
        self.execution_order: List[str] = []
        
        # CAI pattern integration
        self.pattern_executor = None
        self.handoff_manager = None
        if CAI_PATTERNS_AVAILABLE:
            self.pattern_executor = PatternExecutor(message_bus=message_bus)
            self.handoff_manager = HandoffManager(message_bus=message_bus)
            print("[WORKFLOW] CAI pattern support enabled")
        
        self.current_pattern: Optional[AgenticPattern] = None
        self.message_bus = message_bus
    
    def create_workflow(
        self,
        tasks: List[Dict[str, Any]],
        workflow_id: Optional[str] = None,
        pattern_name: Optional[str] = None
    ) -> str:
        """
        Create workflow from task list with optional CAI pattern
        
        Args:
            tasks: List of task dicts with 'id', 'dependencies', etc.
            workflow_id: Optional workflow ID
            pattern_name: Optional CAI pattern name (e.g., 'security_pipeline', 'ctf_swarm')
        
        Returns:
            Workflow ID
        """
        self.tasks = tasks
        self.task_map = {t["id"]: t for t in tasks}
        
        # Generate workflow ID
        wf_id = workflow_id or f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load pattern if specified
        if pattern_name and CAI_PATTERNS_AVAILABLE:
            self.current_pattern = PatternLibrary.get_pattern(pattern_name)
            if self.current_pattern:
                print(f"[WORKFLOW] Using pattern: {pattern_name} ({self.current_pattern.pattern_type.value})")
            else:
                print(f"[WORKFLOW] Warning: Pattern '{pattern_name}' not found")
        
        # Compute execution order (topological sort)
        self.execution_order = self._topological_sort(tasks)
        
        print(f"[WORKFLOW] Created workflow {wf_id}")
        print(f"[WORKFLOW] Tasks: {len(tasks)}")
        print(f"[WORKFLOW] Execution order: {self.execution_order}")
        
        return wf_id
    
    def _topological_sort(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Topological sort of tasks based on dependencies"""
        # Build dependency graph
        in_degree = {t["id"]: 0 for t in tasks}
        dependents = {t["id"]: [] for t in tasks}
        
        for task in tasks:
            for dep in task.get("dependencies", []):
                if dep in in_degree:
                    in_degree[task["id"]] += 1
                    dependents[dep].append(task["id"])
        
        # Find tasks with no dependencies
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(order) != len(tasks):
            print("[WORKFLOW] Warning: Cycle detected in task dependencies")
            # Return what we can
            remaining = [t["id"] for t in tasks if t["id"] not in order]
            order.extend(remaining)
        
        return order
    
    def run(
        self,
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        use_pattern: bool = True
    ) -> WorkflowState:
        """
        Execute the workflow with optional pattern-based execution
        
        Args:
            workflow_id: Workflow ID
            initial_context: Initial context dict
            use_pattern: Whether to use CAI pattern if available
        
        Returns:
            Final workflow state
        """
        # If pattern is available and requested, use pattern execution
        if use_pattern and self.current_pattern and CAI_PATTERNS_AVAILABLE:
            return self._run_with_pattern(workflow_id, initial_context)
        
        # Otherwise, standard DAG execution
        return self._run_dag(workflow_id, initial_context)
    
    def _run_with_pattern(
        self,
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Execute workflow using CAI agentic pattern"""
        print(f"\n[WORKFLOW] Starting pattern-based execution: {self.current_pattern.name}")
        
        # Initialize state
        state = WorkflowState(
            workflow_id=workflow_id,
            context=initial_context or {},
            started_at=datetime.now().isoformat()
        )
        
        # Build agent executor function
        def agent_executor(agent_id: str, context: Dict) -> Dict:
            """Execute agent through orchestrator"""
            # Find task for this agent
            task = next((t for t in self.tasks if t.get("assigned_agent") == agent_id), None)
            if not task:
                print(f"[WORKFLOW] No task found for agent {agent_id}")
                return {"success": False, "error": "Agent task not found"}
            
            task_id = task["id"]
            state.current_task_id = task_id
            state.task_status[task_id] = TaskStatus.RUNNING.value
            
            # Execute via orchestrator
            try:
                if self.orchestrator:
                    result = self.orchestrator.execute_single_task(task, context)
                else:
                    result = self._mock_execute(task, context)
                
                # Update state
                if result.get("success"):
                    state.task_status[task_id] = TaskStatus.COMPLETED.value
                    state.task_outputs[task_id] = result.get("output", "")
                    state.context[task_id] = result
                    
                    # Execute handoff if pattern requires it
                    if self.handoff_manager:
                        handoff_result = self.handoff_manager.auto_handoff(
                            agent_id,
                            result,
                            self.current_pattern
                        )
                        if handoff_result and handoff_result.status == HandoffStatus.SUCCESS:
                            result.update(handoff_result.context.input_data)
                else:
                    state.task_status[task_id] = TaskStatus.FAILED.value
                    state.errors.append({
                        "task_id": task_id,
                        "error": result.get("error", "Unknown"),
                        "timestamp": datetime.now().isoformat()
                    })
                
                self.checkpointer.save(state)
                return result
                
            except Exception as e:
                state.task_status[task_id] = TaskStatus.FAILED.value
                state.errors.append({
                    "task_id": task_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                self.checkpointer.save(state)
                return {"success": False, "error": str(e)}
        
        # Execute pattern
        try:
            pattern_context = self.pattern_executor.execute_pattern(
                self.current_pattern,
                state.context,
                agent_executor
            )
            
            # Update final state
            state.context.update(pattern_context)
            state.completed_at = datetime.now().isoformat()
            self.checkpointer.save(state)
            
            print(f"[WORKFLOW] Pattern execution completed")
            return state
            
        except Exception as e:
            print(f"[WORKFLOW] Pattern execution failed: {e}")
            state.errors.append({
                "task_id": "pattern_execution",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.completed_at = datetime.now().isoformat()
            self.checkpointer.save(state)
            return state
    
    def _run_dag(
        self,
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Execute workflow using standard DAG approach"""
        # Initialize state
        state = WorkflowState(
            workflow_id=workflow_id,
            context=initial_context or {},
            started_at=datetime.now().isoformat()
        )
        
        # Initialize task statuses
        for task_id in self.execution_order:
            state.task_status[task_id] = TaskStatus.PENDING.value
        
        print(f"\n[WORKFLOW] Starting DAG execution of {workflow_id}")
        
        # Execute tasks in order
        for task_id in self.execution_order:
            task = self.task_map.get(task_id)
            if not task:
                continue
            
            # Check if dependencies are satisfied
            deps = task.get("dependencies", [])
            deps_satisfied = all(
                state.task_status.get(dep) == TaskStatus.COMPLETED.value
                for dep in deps
            )
            
            if not deps_satisfied:
                print(f"[WORKFLOW] Skipping {task_id} - dependencies not met")
                state.task_status[task_id] = TaskStatus.SKIPPED.value
                continue
            
            # Execute task
            state.current_task_id = task_id
            state.task_status[task_id] = TaskStatus.RUNNING.value
            
            # Checkpoint before execution
            self.checkpointer.save(state)
            
            try:
                # Build context from dependencies
                task_context = self._build_task_context(task, state)
                
                # Execute via orchestrator
                if self.orchestrator:
                    result = self.orchestrator.execute_single_task(task, task_context)
                else:
                    result = self._mock_execute(task, task_context)
                
                # Update state
                if result.get("success"):
                    state.task_status[task_id] = TaskStatus.COMPLETED.value
                    state.task_outputs[task_id] = result.get("output", "")
                    state.context[task_id] = result
                    print(f"[WORKFLOW] Completed: {task_id}")
                else:
                    state.task_status[task_id] = TaskStatus.FAILED.value
                    state.errors.append({
                        "task_id": task_id,
                        "error": result.get("error", "Unknown error"),
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"[WORKFLOW] Failed: {task_id}")
                
            except Exception as e:
                state.task_status[task_id] = TaskStatus.FAILED.value
                state.errors.append({
                    "task_id": task_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                print(f"[WORKFLOW] Error in {task_id}: {e}")
            
            # Checkpoint after execution
            self.checkpointer.save(state)
        
        state.completed_at = datetime.now().isoformat()
        state.current_task_id = None
        
        # Final checkpoint
        self.checkpointer.save(state)
        
        print(f"[WORKFLOW] Completed workflow {workflow_id}")
        return state
    
    def _build_task_context(
        self,
        task: Dict[str, Any],
        state: WorkflowState
    ) -> Dict[str, Any]:
        """Build context for task execution from dependencies"""
        context = {}
        
        for dep in task.get("dependencies", []):
            if dep in state.task_outputs:
                context[dep] = {
                    "output": state.task_outputs[dep],
                    "status": state.task_status.get(dep)
                }
        
        # Add global context
        context["global"] = state.context.get("global", {})
        
        return context
    
    def _mock_execute(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock execution for testing without orchestrator"""
        print(f"[MOCK] Executing: {task.get('id')}")
        return {
            "success": True,
            "output": f"Mock output for {task.get('id')}",
            "execution_time": 1.0
        }
    
    def resume(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Resume workflow from checkpoint
        
        Args:
            workflow_id: Workflow to resume
        
        Returns:
            Final state or None if not found
        """
        state = self.checkpointer.load(workflow_id)
        if not state:
            print(f"[WORKFLOW] No checkpoint found for {workflow_id}")
            return None
        
        print(f"[WORKFLOW] Resuming {workflow_id}")
        
        # Find incomplete tasks
        incomplete = [
            tid for tid, status in state.task_status.items()
            if status in (TaskStatus.PENDING.value, TaskStatus.RUNNING.value)
        ]
        
        if not incomplete:
            print("[WORKFLOW] All tasks already completed")
            return state
        
        # Continue execution from where we left off
        for task_id in self.execution_order:
            if task_id not in incomplete:
                continue
            
            task = self.task_map.get(task_id)
            if not task:
                continue
            
            # Check dependencies
            deps = task.get("dependencies", [])
            deps_satisfied = all(
                state.task_status.get(dep) == TaskStatus.COMPLETED.value
                for dep in deps
            )
            
            if not deps_satisfied:
                continue
            
            # Execute task (similar to run())
            state.current_task_id = task_id
            state.task_status[task_id] = TaskStatus.RUNNING.value
            
            try:
                task_context = self._build_task_context(task, state)
                
                if self.orchestrator:
                    result = self.orchestrator.execute_single_task(task, task_context)
                else:
                    result = self._mock_execute(task, task_context)
                
                if result.get("success"):
                    state.task_status[task_id] = TaskStatus.COMPLETED.value
                    state.task_outputs[task_id] = result.get("output", "")
                    state.context[task_id] = result
                else:
                    state.task_status[task_id] = TaskStatus.FAILED.value
                    state.errors.append({
                        "task_id": task_id,
                        "error": result.get("error", "Unknown"),
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                state.task_status[task_id] = TaskStatus.FAILED.value
                state.errors.append({
                    "task_id": task_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            self.checkpointer.save(state)
        
        state.completed_at = datetime.now().isoformat()
        self.checkpointer.save(state)
        
        return state
    
    def get_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        state = self.checkpointer.load(workflow_id)
        if not state:
            return {"error": "Workflow not found"}
        
        completed = sum(1 for s in state.task_status.values() if s == TaskStatus.COMPLETED.value)
        failed = sum(1 for s in state.task_status.values() if s == TaskStatus.FAILED.value)
        pending = sum(1 for s in state.task_status.values() if s == TaskStatus.PENDING.value)
        
        return {
            "workflow_id": workflow_id,
            "status": "completed" if not pending else "in_progress" if completed > 0 else "pending",
            "tasks_total": len(state.task_status),
            "tasks_completed": completed,
            "tasks_failed": failed,
            "tasks_pending": pending,
            "errors": len(state.errors),
            "started_at": state.started_at,
            "completed_at": state.completed_at
        }


if __name__ == "__main__":
    # Test workflow engine
    workflow = MultiAgentWorkflow()
    
    print("\n" + "="*60)
    print("LANGGRAPH WORKFLOW ENGINE TEST")
    print("="*60 + "\n")
    
    # Create test tasks with dependencies
    tasks = [
        {
            "id": "task_1",
            "description": "Setup project structure",
            "dependencies": []
        },
        {
            "id": "task_2",
            "description": "Write core functionality",
            "dependencies": ["task_1"]
        },
        {
            "id": "task_3",
            "description": "Write utility functions",
            "dependencies": ["task_1"]
        },
        {
            "id": "task_4",
            "description": "Integration and testing",
            "dependencies": ["task_2", "task_3"]
        },
        {
            "id": "task_5_synthesis",
            "description": "Combine all outputs",
            "dependencies": ["task_4"],
            "type": "synthesis"
        }
    ]
    
    # Create workflow
    wf_id = workflow.create_workflow(tasks)
    
    # Run workflow
    state = workflow.run(wf_id)
    
    print(f"\n{'='*60}")
    print("WORKFLOW RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Workflow ID: {state.workflow_id}")
    print(f"Started: {state.started_at}")
    print(f"Completed: {state.completed_at}")
    print(f"\nTask Status:")
    for tid, status in state.task_status.items():
        print(f"  {tid}: {status}")
    
    if state.errors:
        print(f"\nErrors: {len(state.errors)}")
        for err in state.errors:
            print(f"  - {err['task_id']}: {err['error']}")
    
    # Get status
    status = workflow.get_status(wf_id)
    print(f"\nStatus: {status}")
