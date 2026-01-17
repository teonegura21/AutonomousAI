#!/usr/bin/env python3
"""
Telemetry
Track what agents are doing in real-time - metrics, logging, dashboard data
"""

import json
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ExecutionMetrics:
    """Metrics for a single task execution"""
    task_id: str
    agent_id: str
    started_at: str
    completed_at: Optional[str] = None
    duration_sec: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tools_used: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionMonitor:
    """
    Track what agents are doing in real-time.
    Provides dashboard data and logging.
    """
    
    def __init__(
        self,
        log_file: str = ".runtime/logs/execution.jsonl",
        enable_logging: bool = True
    ):
        self.log_file = log_file
        self.enable_logging = enable_logging
        
        self.active_tasks: Dict[str, ExecutionMetrics] = {}
        self.completed_tasks: List[ExecutionMetrics] = []
        self.total_tokens = 0
        self.lock = threading.Lock()
        
        # Ensure log directory exists
        if enable_logging:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_task_start(
        self,
        task_id: str,
        agent_id: str,
        task_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Task started"""
        metrics = ExecutionMetrics(
            task_id=task_id,
            agent_id=agent_id,
            started_at=datetime.now().isoformat()
        )
        
        with self.lock:
            self.active_tasks[task_id] = metrics
        
        self._log_event("task_start", {
            "task_id": task_id,
            "agent_id": agent_id,
            **(task_info or {})
        })
        
        print(f"[MONITOR] Started: {task_id} on {agent_id}")
    
    def log_tool_execution(
        self,
        task_id: str,
        tool: str,
        duration: float,
        success: bool
    ) -> None:
        """Tool was used"""
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].tools_used.append(tool)
        
        self._log_event("tool_exec", {
            "task_id": task_id,
            "tool": tool,
            "duration_sec": duration,
            "success": success
        })
    
    def log_task_complete(
        self,
        task_id: str,
        success: bool,
        tokens: int = 0,
        error: Optional[str] = None
    ) -> Optional[ExecutionMetrics]:
        """Task finished"""
        with self.lock:
            if task_id not in self.active_tasks:
                return None
            
            metrics = self.active_tasks.pop(task_id)
            metrics.completed_at = datetime.now().isoformat()
            metrics.success = success
            metrics.error = error
            metrics.tokens_output = tokens
            
            # Calculate duration
            start = datetime.fromisoformat(metrics.started_at)
            end = datetime.fromisoformat(metrics.completed_at)
            metrics.duration_sec = (end - start).total_seconds()
            
            self.completed_tasks.append(metrics)
            self.total_tokens += tokens
        
        self._log_event("task_complete", {
            "task_id": task_id,
            "success": success,
            "duration_sec": metrics.duration_sec,
            "tokens": tokens,
            "error": error
        })
        
        status = "OK" if success else "FAIL"
        print(f"[MONITOR] Completed: {task_id} [{status}] ({metrics.duration_sec:.2f}s)")
        
        return metrics
    
    def log_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str
    ) -> None:
        """Log agent-to-agent message"""
        self._log_event("agent_message", {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type
        })
    
    def log_error(
        self,
        task_id: str,
        error: str,
        error_type: str = "general"
    ) -> None:
        """Log an error"""
        self._log_event("error", {
            "task_id": task_id,
            "error": error[:500],
            "error_type": error_type
        })
        
        print(f"[MONITOR] Error in {task_id}: {error[:100]}")
    
    def log_checkpoint(
        self,
        task_id: str,
        decision: str,
        feedback: Optional[str] = None
    ) -> None:
        """Log human checkpoint decision"""
        self._log_event("checkpoint", {
            "task_id": task_id,
            "decision": decision,
            "feedback": feedback
        })
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Current system state for dashboard"""
        with self.lock:
            completed = len(self.completed_tasks)
            successful = sum(1 for t in self.completed_tasks if t.success)
            
            # Calculate average duration
            durations = [t.duration_sec for t in self.completed_tasks if t.duration_sec > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Tools usage
            all_tools = []
            for t in self.completed_tasks:
                all_tools.extend(t.tools_used)
            tool_counts = {}
            for tool in all_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            return {
                "active_tasks": list(self.active_tasks.keys()),
                "active_agents": list(set(t.agent_id for t in self.active_tasks.values())),
                "tasks_completed": completed,
                "tasks_successful": successful,
                "tasks_failed": completed - successful,
                "tasks_pending": len(self.active_tasks),
                "total_tokens": self.total_tokens,
                "estimated_cost": f"${self.total_tokens * 0.000001:.4f}",
                "avg_duration_sec": round(avg_duration, 2),
                "success_rate": round(successful / completed * 100, 1) if completed > 0 else 0,
                "tool_usage": tool_counts,
                "recent_errors": [
                    {"task_id": t.task_id, "error": t.error[:100] if t.error else None}
                    for t in self.completed_tasks[-5:] if t.error
                ]
            }
    
    def get_task_history(
        self,
        limit: int = 50,
        agent_id: Optional[str] = None,
        success_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get task execution history"""
        with self.lock:
            history = self.completed_tasks.copy()
        
        if agent_id:
            history = [t for t in history if t.agent_id == agent_id]
        
        if success_only is not None:
            history = [t for t in history if t.success == success_only]
        
        return [t.to_dict() for t in history[-limit:]]
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get currently active tasks"""
        with self.lock:
            return [t.to_dict() for t in self.active_tasks.values()]
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        with self.lock:
            agent_tasks = [t for t in self.completed_tasks if t.agent_id == agent_id]
        
        if not agent_tasks:
            return {"agent_id": agent_id, "tasks": 0}
        
        successful = sum(1 for t in agent_tasks if t.success)
        total_time = sum(t.duration_sec for t in agent_tasks)
        total_tokens = sum(t.tokens_output for t in agent_tasks)
        
        return {
            "agent_id": agent_id,
            "tasks_total": len(agent_tasks),
            "tasks_successful": successful,
            "tasks_failed": len(agent_tasks) - successful,
            "success_rate": round(successful / len(agent_tasks) * 100, 1),
            "total_time_sec": round(total_time, 2),
            "avg_time_sec": round(total_time / len(agent_tasks), 2),
            "total_tokens": total_tokens
        }
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write event to log file"""
        if not self.enable_logging:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "event": event_type,
                    "timestamp": datetime.now().isoformat(),
                    **data
                }) + "\n")
        except Exception as e:
            print(f"[MONITOR] Log error: {e}")
    
    def clear_history(self) -> None:
        """Clear completed tasks history"""
        with self.lock:
            self.completed_tasks.clear()
            self.total_tokens = 0
    
    def export_report(self, filepath: str) -> bool:
        """Export execution report to JSON file"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": self.get_dashboard_data(),
                "tasks": [t.to_dict() for t in self.completed_tasks]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            print(f"[MONITOR] Report exported to {filepath}")
            return True
        except Exception as e:
            print(f"[MONITOR] Export failed: {e}")
            return False


class WorkflowMonitor:
    """
    Monitor multiple workflows
    """
    
    def __init__(self):
        self.monitors: Dict[str, ExecutionMonitor] = {}
        self.default_monitor = ExecutionMonitor()
    
    def get_monitor(self, workflow_id: str) -> ExecutionMonitor:
        """Get or create monitor for workflow"""
        if workflow_id not in self.monitors:
            self.monitors[workflow_id] = ExecutionMonitor(
                log_file=f".runtime/logs/workflow_{workflow_id}.jsonl"
            )
        return self.monitors[workflow_id]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all workflows"""
        return {
            wf_id: monitor.get_dashboard_data()
            for wf_id, monitor in self.monitors.items()
        }


if __name__ == "__main__":
    # Test telemetry
    monitor = ExecutionMonitor()
    
    print("\n" + "="*60)
    print("TELEMETRY TEST")
    print("="*60 + "\n")
    
    # Simulate task execution
    monitor.log_task_start("task_1", "coder_qwen", {"description": "Write factorial"})
    
    # Simulate tool usage
    import time
    time.sleep(0.1)
    monitor.log_tool_execution("task_1", "filesystem_write", 0.05, True)
    monitor.log_tool_execution("task_1", "python_exec", 0.1, True)
    
    # Complete task
    monitor.log_task_complete("task_1", True, tokens=150)
    
    # Another task (failed)
    monitor.log_task_start("task_2", "linguistic_dictalm")
    time.sleep(0.1)
    monitor.log_task_complete("task_2", False, error="Syntax error")
    
    # Get dashboard data
    print("\nDashboard Data:")
    dashboard = monitor.get_dashboard_data()
    for key, value in dashboard.items():
        print(f"  {key}: {value}")
    
    # Get agent stats
    print("\nAgent Stats (coder_qwen):")
    stats = monitor.get_agent_stats("coder_qwen")
    for key, value in stats.items():
        print(f"  {key}: {value}")
