"""LangGraph workflow definition."""

from __future__ import annotations

from typing import Any, Dict

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator

from multi_agent_framework.workflow.nodes import (
    WorkflowNodes,
    check_completion,
    route_to_agent,
)
from multi_agent_framework.workflow.state import AgentState, build_initial_state


def build_workflow(orchestrator: NemotronOrchestrator):
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is not installed.")

    nodes = WorkflowNodes(orchestrator)
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", nodes.orchestrator_node)
    workflow.add_node("coder", nodes.coder_node)
    workflow.add_node("security", nodes.security_node)
    workflow.add_node("researcher", nodes.researcher_node)
    workflow.add_node("synthesizer", nodes.synthesizer_node)

    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_to_agent,
        {
            "coder": "coder",
            "security": "security",
            "researcher": "researcher",
            "done": "synthesizer",
        },
    )

    for agent in ("coder", "security", "researcher"):
        workflow.add_conditional_edges(
            agent,
            check_completion,
            {
                "continue": "orchestrator",
                "done": "synthesizer",
            },
        )

    workflow.add_edge("synthesizer", END)

    checkpoint_path = ".runtime/data/langgraph_checkpoints.sqlite"
    saver = SqliteSaver.from_conn_string(checkpoint_path)
    return workflow.compile(checkpointer=saver)


def run_workflow(task: str, orchestrator: NemotronOrchestrator) -> Dict[str, Any]:
    if not LANGGRAPH_AVAILABLE:
        return orchestrator.run(task)

    if hasattr(orchestrator, "start_session"):
        orchestrator.start_session(task)
        orchestrator.current_workflow_id = orchestrator.task_memory.start_workflow()

    app = build_workflow(orchestrator)
    state = build_initial_state(task)
    config = {"configurable": {"thread_id": orchestrator.current_workflow_id or "default"}}
    result = app.invoke(state, config)

    try:
        tasks = result.get("plan", []) if isinstance(result, dict) else []
        step_results = result.get("step_results", []) if isinstance(result, dict) else []
        outputs = []
        for item in step_results:
            outputs.append(
                {
                    "success": item.get("success", False),
                    "output": item.get("output", ""),
                    "error": item.get("error"),
                    "agent": item.get("agent", "unknown"),
                }
            )
        if hasattr(orchestrator, "_persist_session_artifacts"):
            orchestrator._persist_session_artifacts(task, tasks, outputs)
    except Exception:
        pass

    return result
