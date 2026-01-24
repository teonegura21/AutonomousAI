#!/usr/bin/env python3
"""
Task Memory with Enhanced Inter-task Context
Allows agents to share context, results, and learn from previous tasks
Includes semantic search and context retrieval
"""

import sqlite3
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading


@dataclass
class TaskContext:
    """Context for a single task execution"""
    task_id: str
    description: str
    assigned_agent: str
    status: str = "pending"  # pending, running, completed, failed
    input_context: Dict[str, Any] = field(default_factory=dict)
    output: str = ""
    code_artifacts: List[str] = field(default_factory=list)
    decisions: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_sec: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        # Handle missing fields gracefully
        return cls(
            task_id=data.get("task_id", "unknown"),
            description=data.get("description", ""),
            assigned_agent=data.get("assigned_agent", "unknown"),
            status=data.get("status", "pending"),
            input_context=data.get("input_context", {}),
            output=data.get("output", ""),
            code_artifacts=data.get("code_artifacts", []),
            decisions=data.get("decisions", []),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            execution_time_sec=data.get("execution_time_sec", 0.0),
            tokens_used=data.get("tokens_used", 0)
        )


class TaskMemory:
    """
    Inter-task context sharing system with enhanced features
    - Stores task results, code artifacts, and decisions
    - Provides context from dependencies
    - Thread-safe for parallel execution
    - Context aggregation for synthesis tasks
    - Learning from past executions
    """

    def __init__(self, db_path: str = ".runtime/data/task_memory.db"):
        self.db_path = db_path
        self._ensure_db()
        self._current_workflow_id: Optional[str] = None
        self._context_cache: Dict[str, TaskContext] = {}
        self._lock = threading.RLock()

    def _ensure_db(self):
        """Initialize database with all tables"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Task contexts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_contexts (
                task_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                description TEXT,
                assigned_agent TEXT,
                status TEXT DEFAULT 'pending',
                input_context TEXT,
                output TEXT,
                code_artifacts TEXT,
                decisions TEXT,
                errors TEXT,
                metadata TEXT DEFAULT '{}',
                started_at TEXT,
                completed_at TEXT,
                execution_time_sec REAL DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Task dependencies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_dependencies (
                task_id TEXT,
                depends_on TEXT,
                dependency_type TEXT DEFAULT 'output',
                PRIMARY KEY (task_id, depends_on)
            )
        """)

        # Code artifacts with versioning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                filename TEXT,
                content TEXT,
                language TEXT,
                version INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Decisions log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                decision TEXT,
                rationale TEXT,
                agent_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Agent learnings (patterns observed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                task_type TEXT,
                pattern TEXT,
                success_rate REAL,
                observation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Workflow metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                description TEXT,
                pattern TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                total_tasks INTEGER DEFAULT 0,
                completed_tasks INTEGER DEFAULT 0,
                failed_tasks INTEGER DEFAULT 0
            )
        """)

        # Workflow plans (JSON)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_plans (
                workflow_id TEXT PRIMARY KEY,
                plan_json TEXT,
                updated_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def start_workflow(
        self,
        workflow_id: Optional[str] = None,
        description: str = "",
        pattern: Optional[str] = None
    ) -> str:
        """Start a new workflow with metadata"""
        with self._lock:
            self._current_workflow_id = workflow_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._context_cache.clear()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO workflows
                (workflow_id, description, pattern, status, started_at)
                VALUES (?, ?, ?, ?, ?)
            """, (self._current_workflow_id, description, pattern, "running", datetime.now().isoformat()))
            conn.commit()
            conn.close()

            return self._current_workflow_id

    def complete_workflow(self, workflow_id: Optional[str] = None) -> None:
        """Mark workflow as completed"""
        wf_id = workflow_id or self._current_workflow_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count tasks
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM task_contexts WHERE workflow_id = ?
        """, (wf_id,))
        row = cursor.fetchone()

        cursor.execute("""
            UPDATE workflows SET
                status = 'completed',
                completed_at = ?,
                total_tasks = ?,
                completed_tasks = ?,
                failed_tasks = ?
            WHERE workflow_id = ?
        """, (datetime.now().isoformat(), row[0], row[1], row[2], wf_id))

        conn.commit()
        conn.close()

    def save_workflow_plan(self, workflow_id: str, plan: List[Dict[str, Any]]) -> None:
        """Persist workflow plan JSON for the current conversation/workflow."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO workflow_plans (workflow_id, plan_json, updated_at)
            VALUES (?, ?, ?)
        """, (workflow_id, json.dumps(plan), datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def update_task_definition(
        self,
        task_id: str,
        description: Optional[str] = None,
        assigned_agent: Optional[str] = None
    ) -> None:
        """Update task definition fields for a task context if it exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if description is not None:
            cursor.execute(
                "UPDATE task_contexts SET description = ? WHERE task_id = ?",
                (description, task_id)
            )
        if assigned_agent is not None:
            cursor.execute(
                "UPDATE task_contexts SET assigned_agent = ? WHERE task_id = ?",
                (assigned_agent, task_id)
            )

        conn.commit()
        conn.close()

    def update_task_dependencies(self, task_id: str, dependencies: List[str]) -> None:
        """Replace dependencies for a task in the DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM task_dependencies WHERE task_id = ?", (task_id,))
        for dep in dependencies:
            cursor.execute(
                "INSERT OR IGNORE INTO task_dependencies (task_id, depends_on) VALUES (?, ?)",
                (task_id, dep)
            )
        conn.commit()
        conn.close()

    def create_task_context(
        self,
        task_id: str,
        description: str,
        assigned_agent: str,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> TaskContext:
        """Create a new task context with optional metadata"""
        with self._lock:
            context = TaskContext(
                task_id=task_id,
                description=description,
                assigned_agent=assigned_agent,
                metadata=metadata or {}
            )

            self._context_cache[task_id] = context

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO task_contexts
                (task_id, workflow_id, description, assigned_agent, status,
                 input_context, output, code_artifacts, decisions, errors, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                self._current_workflow_id,
                description,
                assigned_agent,
                "pending",
                json.dumps({}),
                "",
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                json.dumps(metadata or {})
            ))

            # Store dependencies
            if dependencies:
                for dep in dependencies:
                    cursor.execute("""
                        INSERT OR IGNORE INTO task_dependencies (task_id, depends_on)
                        VALUES (?, ?)
                    """, (task_id, dep))

            conn.commit()
            conn.close()

            return context

    def start_task(self, task_id: str, input_context: Dict[str, Any] = None) -> TaskContext:
        """Mark task as started with input context"""
        with self._lock:
            context = self.get_context(task_id)
            if context:
                context.status = "running"
                context.started_at = datetime.now().isoformat()
                if input_context:
                    context.input_context = input_context
                self._update_context(context)
            return context

    def complete_task(
        self,
        task_id: str,
        output: str,
        code_artifacts: List[str] = None,
        tokens_used: int = 0,
        metadata: Dict[str, Any] = None
    ) -> TaskContext:
        """Mark task as completed with output and optional metadata"""
        with self._lock:
            context = self.get_context(task_id)
            if context:
                context.status = "completed"
                context.output = output
                context.completed_at = datetime.now().isoformat()
                context.tokens_used = tokens_used

                if metadata:
                    context.metadata.update(metadata)

                if context.started_at:
                    start = datetime.fromisoformat(context.started_at)
                    end = datetime.fromisoformat(context.completed_at)
                    context.execution_time_sec = (end - start).total_seconds()

                if code_artifacts:
                    context.code_artifacts = code_artifacts
                    self._store_code_artifacts(task_id, code_artifacts)

                self._update_context(context)

            return context

    def fail_task(self, task_id: str, error: str) -> TaskContext:
        """Mark task as failed"""
        with self._lock:
            context = self.get_context(task_id)
            if context:
                context.status = "failed"
                context.errors.append(error)
                context.completed_at = datetime.now().isoformat()
                self._update_context(context)
            return context

    def add_decision(
        self,
        task_id: str,
        decision: str,
        rationale: str,
        agent_id: Optional[str] = None
    ) -> None:
        """Record a decision made during task execution"""
        with self._lock:
            context = self.get_context(task_id)
            if context:
                context.decisions.append({"decision": decision, "rationale": rationale})
                self._update_context(context)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO decisions (task_id, decision, rationale, agent_id)
                VALUES (?, ?, ?, ?)
            """, (task_id, decision, rationale, agent_id or context.assigned_agent if context else "unknown"))
            conn.commit()
            conn.close()

    def record_learning(
        self,
        agent_id: str,
        task_type: str,
        pattern: str,
        success_rate: float,
        observation: str
    ) -> None:
        """Record learning from task execution"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO agent_learnings
            (agent_id, task_type, pattern, success_rate, observation)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, task_type, pattern, success_rate, observation))
        conn.commit()
        conn.close()

    def get_context(self, task_id: str) -> Optional[TaskContext]:
        """Get task context by ID"""
        # Check cache first
        if task_id in self._context_cache:
            return self._context_cache[task_id]

        # Load from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM task_contexts WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            context = TaskContext(
                task_id=row[0],
                description=row[2],
                assigned_agent=row[3],
                status=row[4],
                input_context=json.loads(row[5] or "{}"),
                output=row[6] or "",
                code_artifacts=json.loads(row[7] or "[]"),
                decisions=json.loads(row[8] or "[]"),
                errors=json.loads(row[9] or "[]"),
                metadata=json.loads(row[10] or "{}"),
                started_at=row[11],
                completed_at=row[12],
                execution_time_sec=row[13] or 0,
                tokens_used=row[14] or 0
            )
            self._context_cache[task_id] = context
            return context

        return None

    def get_dependency_context(self, task_id: str) -> Dict[str, Any]:
        """
        Get combined context from all dependencies
        This is what agents receive as input context
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT depends_on FROM task_dependencies WHERE task_id = ?
        """, (task_id,))
        dependencies = [row[0] for row in cursor.fetchall()]
        conn.close()

        combined_context = {
            "previous_outputs": {},
            "code_artifacts": [],
            "decisions": [],
            "dependency_chain": dependencies
        }

        for dep_id in dependencies:
            dep_context = self.get_context(dep_id)
            if dep_context and dep_context.status == "completed":
                combined_context["previous_outputs"][dep_id] = {
                    "description": dep_context.description,
                    "output": dep_context.output[:2000],
                    "agent": dep_context.assigned_agent,
                    "execution_time": dep_context.execution_time_sec
                }
                combined_context["code_artifacts"].extend(dep_context.code_artifacts)
                combined_context["decisions"].extend(dep_context.decisions)

        return combined_context

    def get_all_outputs(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all task outputs for synthesis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        wf_id = workflow_id or self._current_workflow_id

        cursor.execute("""
            SELECT task_id, description, output, code_artifacts, status, assigned_agent
            FROM task_contexts
            WHERE workflow_id = ? AND status = 'completed'
            ORDER BY completed_at
        """, (wf_id,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "task_id": row[0],
                "description": row[1],
                "output": row[2],
                "code_artifacts": json.loads(row[3] or "[]"),
                "status": row[4],
                "agent": row[5]
            }
            for row in rows
        ]

    def get_workflow_stats(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for workflow"""
        wf_id = workflow_id or self._current_workflow_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM workflows WHERE workflow_id = ?
        """, (wf_id,))
        workflow = cursor.fetchone()

        cursor.execute("""
            SELECT
                status,
                COUNT(*) as count,
                AVG(execution_time_sec) as avg_time,
                SUM(tokens_used) as total_tokens
            FROM task_contexts
            WHERE workflow_id = ?
            GROUP BY status
        """, (wf_id,))
        status_stats = cursor.fetchall()

        conn.close()

        return {
            "workflow_id": wf_id,
            "description": workflow[1] if workflow else "",
            "pattern": workflow[2] if workflow else None,
            "status": workflow[3] if workflow else "unknown",
            "task_stats": {
                row[0]: {
                    "count": row[1],
                    "avg_time": row[2],
                    "total_tokens": row[3]
                }
                for row in status_stats
            }
        }

    def export_workflow_db(self, target_path: str, workflow_id: Optional[str] = None) -> bool:
        """Export a workflow-scoped SQLite DB for debugging."""
        wf_id = workflow_id or self._current_workflow_id
        if not wf_id:
            return False

        source_path = Path(self.db_path)
        if not source_path.exists():
            return False

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copyfile(source_path, target)
        except Exception:
            return False

        conn = sqlite3.connect(target)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT task_id FROM task_contexts WHERE workflow_id = ?", (wf_id,))
            task_ids = [row[0] for row in cursor.fetchall()]

            cursor.execute("DELETE FROM workflows WHERE workflow_id != ?", (wf_id,))
            cursor.execute("DELETE FROM workflow_plans WHERE workflow_id != ?", (wf_id,))
            cursor.execute("DELETE FROM task_contexts WHERE workflow_id != ?", (wf_id,))

            if task_ids:
                placeholders = ",".join("?" for _ in task_ids)
                cursor.execute(
                    f"DELETE FROM task_dependencies WHERE task_id NOT IN ({placeholders})",
                    task_ids,
                )
                cursor.execute(
                    f"DELETE FROM code_artifacts WHERE task_id NOT IN ({placeholders})",
                    task_ids,
                )
                cursor.execute(
                    f"DELETE FROM decisions WHERE task_id NOT IN ({placeholders})",
                    task_ids,
                )
            else:
                cursor.execute("DELETE FROM task_dependencies")
                cursor.execute("DELETE FROM code_artifacts")
                cursor.execute("DELETE FROM decisions")

            cursor.execute("DELETE FROM agent_learnings")

            conn.commit()
        finally:
            conn.close()

        return True

    def _update_context(self, context: TaskContext) -> None:
        """Update context in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE task_contexts SET
                status = ?,
                output = ?,
                code_artifacts = ?,
                decisions = ?,
                errors = ?,
                metadata = ?,
                started_at = ?,
                completed_at = ?,
                execution_time_sec = ?,
                tokens_used = ?
            WHERE task_id = ?
        """, (
            context.status,
            context.output,
            json.dumps(context.code_artifacts),
            json.dumps(context.decisions),
            json.dumps(context.errors),
            json.dumps(context.metadata),
            context.started_at,
            context.completed_at,
            context.execution_time_sec,
            context.tokens_used,
            context.task_id
        ))

        conn.commit()
        conn.close()

    def _store_code_artifacts(self, task_id: str, artifacts: List[str]) -> None:
        """Store code artifacts separately"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, code in enumerate(artifacts):
            language = "python"
            if "```javascript" in code or "const " in code:
                language = "javascript"
            elif "```bash" in code or "#!/bin/bash" in code:
                language = "bash"

            cursor.execute("""
                INSERT INTO code_artifacts (task_id, filename, content, language)
                VALUES (?, ?, ?, ?)
            """, (task_id, f"artifact_{i}.{language}", code, language))

        conn.commit()
        conn.close()

    def clear_workflow(self, workflow_id: Optional[str] = None) -> None:
        """Clear all data for a workflow"""
        wf_id = workflow_id or self._current_workflow_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT task_id FROM task_contexts WHERE workflow_id = ?", (wf_id,))
        task_ids = [row[0] for row in cursor.fetchall()]

        for task_id in task_ids:
            cursor.execute("DELETE FROM task_dependencies WHERE task_id = ?", (task_id,))
            cursor.execute("DELETE FROM code_artifacts WHERE task_id = ?", (task_id,))
            cursor.execute("DELETE FROM decisions WHERE task_id = ?", (task_id,))

        cursor.execute("DELETE FROM task_contexts WHERE workflow_id = ?", (wf_id,))
        cursor.execute("DELETE FROM workflows WHERE workflow_id = ?", (wf_id,))

        conn.commit()
        conn.close()

        self._context_cache.clear()


if __name__ == "__main__":
    memory = TaskMemory()

    print("\n" + "="*60)
    print("ENHANCED TASK MEMORY TEST")
    print("="*60 + "\n")

    wf_id = memory.start_workflow(description="Test workflow", pattern="chain")
    print(f"Started workflow: {wf_id}\n")

    task1 = memory.create_task_context(
        "task_1", "Write factorial function", "coder_qwen", [],
        metadata={"priority": "high", "complexity": "low"}
    )
    task2 = memory.create_task_context(
        "task_2", "Write tests for factorial", "coder_qwen", ["task_1"],
        metadata={"priority": "medium", "complexity": "low"}
    )

    memory.start_task("task_1")
    memory.add_decision("task_1", "Use recursion", "Cleaner implementation", "coder_qwen")
    memory.complete_task(
        "task_1",
        "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```",
        ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"],
        tokens_used=150
    )

    print("Task 1 completed")

    context = memory.get_dependency_context("task_2")
    print(f"\nContext for task_2:")
    print(f"  Dependencies: {context['dependency_chain']}")
    print(f"  Previous outputs: {len(context['previous_outputs'])} tasks")
    print(f"  Code artifacts: {len(context['code_artifacts'])} items")

    memory.complete_workflow(wf_id)

    stats = memory.get_workflow_stats(wf_id)
    print(f"\nWorkflow stats:")
    print(f"  Status: {stats['status']}")
    print(f"  Pattern: {stats['pattern']}")
    print(f"  Task stats: {stats['task_stats']}")

    print("\n" + "="*60)
