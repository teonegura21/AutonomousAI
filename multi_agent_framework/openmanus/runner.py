from __future__ import annotations

import asyncio
from typing import Optional

from app.agent.toolcall import ToolCallAgent
from app.schema import AgentState
from app.tool import Terminate, ToolCollection

from multi_agent_framework.openmanus.app.tool.autonom_tools import build_default_tools


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def _build_tool_collection() -> ToolCollection:
    tools = build_default_tools()
    tools.append(Terminate())
    return ToolCollection(*tools)


async def _run_agent(agent: ToolCallAgent, request: str, cancel_event=None) -> str:
    if agent.state != AgentState.IDLE:
        raise RuntimeError(f"Cannot run agent from state: {agent.state}")

    if request:
        agent.update_memory("user", request)

    results = []
    agent.state = AgentState.RUNNING
    try:
        while agent.current_step < agent.max_steps and agent.state != AgentState.FINISHED:
            if cancel_event and cancel_event.is_set():
                agent.state = AgentState.ERROR
                results.append("Cancelled")
                break

            agent.current_step += 1
            step_result = await agent.step()
            results.append(f"Step {agent.current_step}: {step_result}")

            if agent.is_stuck():
                agent.handle_stuck_state()

        if agent.current_step >= agent.max_steps:
            agent.current_step = 0
            agent.state = AgentState.IDLE
            results.append(f"Terminated: Reached max steps ({agent.max_steps})")
    finally:
        agent.state = AgentState.IDLE
        try:
            from app.sandbox.client import SANDBOX_CLIENT

            await SANDBOX_CLIENT.cleanup()
        except Exception:
            pass

    return "\n".join(results) if results else "No steps executed"


def run_toolcall_task(
    *,
    goal: str,
    system_prompt: Optional[str],
    next_step_prompt: Optional[str] = None,
    max_steps: int = 12,
    cancel_event=None,
) -> str:
    tools = _build_tool_collection()

    agent = ToolCallAgent(
        system_prompt=system_prompt,
        next_step_prompt=next_step_prompt,
        available_tools=tools,
        max_steps=max_steps,
    )

    return _run_async(_run_agent(agent, goal, cancel_event=cancel_event))
