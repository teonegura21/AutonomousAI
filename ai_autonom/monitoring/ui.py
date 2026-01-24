"""
Terminal UI
Readable, structured output with optional Rich formatting.
"""

from collections import deque
from datetime import datetime
import threading
import sys
from typing import List, Dict, Any, Optional, Callable

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.box import SIMPLE
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Panel = None
    Table = None
    SIMPLE = None


STATUS_LABELS = {
    "pending": "PENDING",
    "running": "RUNNING",
    "completed": "COMPLETED",
    "failed": "FAILED"
}

LEVEL_STYLES = {
    "ERROR": "red",
    "WARNING": "yellow",
    "SUCCESS": "green",
    "INFO": "cyan",
    "AGENT": "magenta",
    "TOOL": "blue"
}


class Dashboard:
    """
    Structured terminal dashboard.
    Uses Rich when available, otherwise plain text.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._goal = ""
        self._plan: List[Dict[str, Any]] = []
        self._task_status: Dict[str, str] = {}
        self._active_agent: Optional[str] = None
        self._active_tool: Optional[str] = None
        self._active_model: Optional[str] = None
        self._session_id: Optional[str] = None
        self._session_dir: Optional[str] = None
        self._intent_analysis: Optional[Dict[str, Any]] = None
        self._task_briefs: Optional[Dict[str, Any]] = None
        self._final_report: Optional[Dict[str, Any]] = None
        self._trace = deque(maxlen=300)
        self._logs = deque(maxlen=200)
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._console = Console() if RICH_AVAILABLE else None

    def _print(self, text: str) -> None:
        with self._lock:
            print(text)
            sys.stdout.flush()

    def _log_line(self, level: str, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_tag = level.upper() if level else "INFO"
        return f"[{timestamp}] [{level_tag}] {message}"

    def _emit(self, event: Dict[str, Any]) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                continue

    def subscribe(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def _render_plan_table(self) -> None:
        if not self._plan:
            return
        if not RICH_AVAILABLE:
            self._print("PLAN")
            for task in self._plan:
                task_id = task.get("id", "unknown")
                agent = task.get("assigned_agent", "unassigned")
                status = self._task_status.get(task_id, "pending").upper()
                self._print(f"  - {task_id} | {agent} | {status}")
            return

        table = Table(title="Plan", box=SIMPLE, show_header=True, header_style="bold")
        table.add_column("Task", style="white", no_wrap=True)
        table.add_column("Agent", style="green")
        table.add_column("Status", style="cyan")
        for task in self._plan:
            task_id = task.get("id", "unknown")
            agent = task.get("assigned_agent", "unassigned")
            status = self._task_status.get(task_id, "pending").upper()
            table.add_row(task_id, agent, status)
        self._console.print(table)

    # --- API Methods ---

    def start(self, goal: str) -> None:
        self._goal = goal or ""
        header = f"MISSION START: {self._goal or '(no goal)'}"
        self._emit({"type": "session_start", "goal": self._goal, "ts": datetime.now().isoformat()})
        if RICH_AVAILABLE:
            self._console.print(Panel(header, title="AI Autonom", box=SIMPLE))
        else:
            self._print("=" * 70)
            self._print(header)
            self._print("=" * 70)

    def stop(self) -> None:
        self._emit({"type": "session_end", "ts": datetime.now().isoformat()})
        if RICH_AVAILABLE:
            self._console.print(Panel("MISSION END", box=SIMPLE))
        else:
            self._print("=" * 70)
            self._print("MISSION END")
            self._print("=" * 70)

    def set_plan(self, tasks: List[Dict]) -> None:
        self._plan = tasks or []
        for task in self._plan:
            task_id = task.get("id")
            if task_id and task_id not in self._task_status:
                self._task_status[task_id] = "pending"
        self._emit({"type": "plan", "plan": self._plan, "ts": datetime.now().isoformat()})
        self._render_plan_table()

    def update_task_status(self, task_id: str, status: str) -> None:
        normalized = status.lower() if status else "pending"
        self._task_status[task_id] = normalized
        status_label = STATUS_LABELS.get(normalized, normalized.upper())
        line = f"TASK {task_id}: {status_label}"
        self._emit({
            "type": "task_status",
            "task_id": task_id,
            "status": normalized,
            "ts": datetime.now().isoformat()
        })
        if RICH_AVAILABLE:
            self._console.print(line)
        else:
            self._print(line)

    def log(self, message: str, level: str = "INFO") -> None:
        line = self._log_line(level, message)
        self._logs.append(line)
        self._emit({
            "type": "log",
            "level": level.upper(),
            "message": message,
            "line": line,
            "ts": datetime.now().isoformat()
        })
        if RICH_AVAILABLE:
            style = LEVEL_STYLES.get(level.upper(), "white")
            self._console.print(line, style=style)
        else:
            self._print(line)

    def set_active_agent(self, agent_name: str) -> None:
        self._active_agent = agent_name
        self._emit({
            "type": "active_agent",
            "agent": agent_name,
            "ts": datetime.now().isoformat()
        })

    def set_active_tool(self, tool_name: str) -> None:
        self._active_tool = tool_name
        self._emit({
            "type": "active_tool",
            "tool": tool_name,
            "ts": datetime.now().isoformat()
        })

    def set_active_model(self, model_name: str) -> None:
        previous = self._active_model
        self._active_model = model_name
        self._emit({
            "type": "active_model",
            "model": model_name,
            "ts": datetime.now().isoformat()
        })
        if model_name and model_name != previous:
            self.trace_event(
                "model_switch",
                {"from": previous, "to": model_name}
            )

    def set_session(self, session_id: Optional[str], session_dir: Optional[str]) -> None:
        self._session_id = session_id
        self._session_dir = session_dir
        self._emit({
            "type": "session",
            "session_id": session_id,
            "session_dir": session_dir,
            "ts": datetime.now().isoformat()
        })

    def trace_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "payload": payload,
            "ts": datetime.now().isoformat()
        }
        self._trace.append(event)
        self._emit({
            "type": "trace",
            "event": event,
            "ts": event["ts"]
        })

    def set_intent_analysis(self, payload: Dict[str, Any]) -> None:
        self._intent_analysis = payload
        self._emit({
            "type": "intent_analysis",
            "payload": payload,
            "ts": datetime.now().isoformat()
        })

    def set_task_briefs(self, payload: Dict[str, Any]) -> None:
        self._task_briefs = payload
        self._emit({
            "type": "task_briefs",
            "payload": payload,
            "ts": datetime.now().isoformat()
        })

    def set_final_report(self, payload: Dict[str, Any]) -> None:
        self._final_report = payload
        self._emit({
            "type": "final_report",
            "payload": payload,
            "ts": datetime.now().isoformat()
        })

    def snapshot(self) -> Dict[str, Any]:
        return {
            "goal": self._goal,
            "plan": self._plan,
            "task_status": self._task_status,
            "active_agent": self._active_agent,
            "active_tool": self._active_tool,
            "active_model": self._active_model,
            "session": {
                "id": self._session_id,
                "dir": self._session_dir
            },
            "intent_analysis": self._intent_analysis,
            "task_briefs": self._task_briefs,
            "final_report": self._final_report,
            "trace": list(self._trace),
            "logs": list(self._logs)
        }

    def reset(self) -> None:
        with self._lock:
            self._goal = ""
            self._plan = []
            self._task_status = {}
            self._active_agent = None
            self._active_tool = None
            self._active_model = None
            self._session_id = None
            self._session_dir = None
            self._intent_analysis = None
            self._task_briefs = None
            self._final_report = None
            self._trace.clear()
            self._logs.clear()
        self._emit({"type": "reset", "ts": datetime.now().isoformat()})


_ui = None


def get_ui() -> Dashboard:
    global _ui
    if _ui is None:
        _ui = Dashboard()
    return _ui
