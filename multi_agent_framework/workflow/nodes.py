"""Workflow nodes for the LangGraph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List
import asyncio
from datetime import datetime
from pathlib import Path

from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator

from multi_agent_framework.workflow.state import AgentState
from multi_agent_framework.agents.registry import (
    register_framework_agents,
    resolve_agent_for_role,
)


def _agent_label(assigned_agent: str) -> str:
    if not assigned_agent:
        return "coder"
    name = assigned_agent.lower()
    if "security" in name or "pentest" in name or "red" in name:
        return "security"
    if "research" in name or "analyst" in name:
        return "researcher"
    return "coder"


def route_to_agent(state: AgentState) -> str:
    plan = state.get("plan", [])
    index = state.get("current_step", 0)
    if not plan or index >= len(plan):
        return "done"
    task = plan[index]
    return _agent_label(task.get("assigned_agent", ""))


def check_completion(state: AgentState) -> str:
    plan = state.get("plan", [])
    index = state.get("current_step", 0)
    if plan and index < len(plan):
        return "continue"
    return "done"


class WorkflowNodes:
    def __init__(self, orchestrator: NemotronOrchestrator) -> None:
        self.orchestrator = orchestrator
        self._role_map = register_framework_agents(
            orchestrator.registry,
            workspace_dir=Path.cwd(),
            coder_model=self.orchestrator.config.get("agents.coder.model", "qwen2.5-coder:7b"),
            linguistic_model=self.orchestrator.config.get(
                "agents.linguistic.model",
                "dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0",
            ),
            provider=self.orchestrator.config.get("providers.default", "ollama"),
            vram_gb=2.0,
        )

    def orchestrator_node(self, state: AgentState) -> Dict[str, Any]:
        if state.get("plan"):
            return {}

        intent_result = self.orchestrator.analyze_intent(state["task"])
        enhancement = intent_result.get("enhancement", "")
        analysis = intent_result.get("analysis")
        pattern_name = analysis.suggested_pattern if analysis else None

        try:
            analysis_payload = {
                "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else {},
                "enhancement": enhancement,
                "created_at": datetime.now().isoformat(),
            }
            if hasattr(self.orchestrator, "_write_memory_json"):
                self.orchestrator._write_memory_json("intent_analysis.json", analysis_payload)
            if getattr(self.orchestrator, "ui", None):
                self.orchestrator.ui.set_intent_analysis(analysis_payload)
        except Exception:
            pass

        tasks = self.orchestrator.decompose_and_assign(
            state["task"],
            enhancement,
            pattern_name=pattern_name,
        )
        self.orchestrator._ensure_sequential_dependencies(tasks)

        try:
            if hasattr(self.orchestrator, "_build_task_briefs") and hasattr(self.orchestrator, "_write_memory_json"):
                briefs_payload = {
                    "created_at": datetime.now().isoformat(),
                    "tasks": self.orchestrator._build_task_briefs(tasks),
                }
                self.orchestrator._write_memory_json("task_briefs.json", briefs_payload)
                if getattr(self.orchestrator, "ui", None):
                    self.orchestrator.ui.set_task_briefs(briefs_payload)
        except Exception:
            pass

        memory = dict(state.get("memory", {}))
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            from app.tool.planning import PlanningTool

            plan_tool = PlanningTool()
            steps = [task.get("description", "") for task in tasks]
            try:
                asyncio.run(
                    plan_tool.execute(
                        command="create",
                        plan_id=plan_id,
                        title=state["task"],
                        steps=steps,
                    )
                )
            except RuntimeError:
                pass
            memory["openmanus_plan_id"] = plan_id
        except Exception:
            memory.setdefault("openmanus_plan_id", plan_id)

        return {
            "plan": tasks,
            "current_step": 0,
            "memory": memory,
        }

    def _execute_current_task(self, state: AgentState, role: str) -> Dict[str, Any]:
        plan = state.get("plan", [])
        index = state.get("current_step", 0)
        if not plan or index >= len(plan):
            return {}

        task = plan[index]
        override = resolve_agent_for_role(
            self.orchestrator.registry, role, self._role_map
        )
        if override and task.get("assigned_agent") != override:
            task["assigned_agent"] = override
            plan[index] = task
        context = state.get("memory", {})
        result = self.orchestrator.execute_single_task(task, context=context)

        step_result = {
            "task_id": task.get("id"),
            "agent": task.get("assigned_agent"),
            "success": result.get("success", False),
            "output": result.get("output", ""),
            "error": result.get("error"),
        }

        memory = dict(state.get("memory", {}))
        task_id = task.get("id")
        if task_id:
            memory[task_id] = result

        updates: Dict[str, Any] = {
            "step_results": [step_result],
            "memory": memory,
            "current_step": index + 1,
        }

        if not result.get("success"):
            updates["errors"] = [result.get("error", "Task failed")]

        return updates

    def coder_node(self, state: AgentState) -> Dict[str, Any]:
        return self._execute_current_task(state, "coder")

    def security_node(self, state: AgentState) -> Dict[str, Any]:
        return self._execute_current_task(state, "security")

    def researcher_node(self, state: AgentState) -> Dict[str, Any]:
        return self._execute_current_task(state, "researcher")

    def synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        outputs: List[str] = []
        for item in state.get("step_results", []):
            task_id = item.get("task_id", "unknown")
            output = item.get("output", "")
            if output:
                outputs.append(f"[{task_id}] {output}")

        final_output = "\n\n".join(outputs)
        return {"final_output": final_output}
