"""Working memory wrapper for task state."""

from __future__ import annotations

from ai_autonom.memory.task_memory import TaskMemory


class WorkingMemory:
    def __init__(self, db_path: str = "./.runtime/data/task_memory.db") -> None:
        self._memory = TaskMemory(db_path=db_path)

    def start_workflow(self) -> str:
        return self._memory.start_workflow()

    def save_plan(self, workflow_id: str, plan) -> None:
        self._memory.save_workflow_plan(workflow_id, plan)

    def get_plan(self, workflow_id: str):
        return self._memory.get_workflow_plan(workflow_id)

    def clear(self) -> None:
        self._memory.clear_all()
