"""State schema for the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class AgentState(TypedDict):
    task: str
    plan: List[Dict[str, Any]]
    current_step: int
    step_results: Annotated[List[Dict[str, Any]], operator.add]
    memory: Dict[str, Any]
    final_output: str
    errors: Annotated[List[str], operator.add]


def build_initial_state(task: str) -> AgentState:
    return {
        "task": task,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "memory": {},
        "final_output": "",
        "errors": [],
    }
