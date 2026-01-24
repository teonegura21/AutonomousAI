"""
FastAPI backend for the AI Autonom web UI.
Streams UI events over SSE and triggers orchestration runs.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ai_autonom.core.config import Config
from ai_autonom.core.llm_provider import LLMMessage, get_provider
from ai_autonom.core.model_discovery import ModelDiscovery
from ai_autonom.core.model_selector import DynamicModelSelector
from ai_autonom.core.session_manager import SessionManager
from ai_autonom.monitoring.ui import get_ui
from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
from ai_autonom.tools.tool_executor import ToolExecutor
from multi_agent_framework.workflow.graph import run_workflow


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Content-Type": "text/event-stream",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class EventHub:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._queues.discard(queue)

    async def _publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            queues = list(self._queues)
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop events when the client is too slow.
                continue

    def publish(self, event: Dict[str, Any]) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._publish(event), self._loop)


@dataclass
class RunState:
    running: bool = False
    cancelling: bool = False
    use_langgraph: bool = True
    goal: str = ""
    session_id: str = ""
    session_dir: str = ""
    last_result: Optional[Dict[str, Any]] = None
    thread: Optional[threading.Thread] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


hub = EventHub()
run_state = RunState()

CONVERSATIONS_DIR = Path("outputs/conversations")
_tool_executor: Optional[ToolExecutor] = None
_ollama_provider = None
TESTS_PATH = Path(".runtime/data/regression_tests.json")


def _get_tool_executor() -> ToolExecutor:
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor(workspace_dir="outputs", use_containers=True)
    return _tool_executor


def _get_ollama_provider():
    global _ollama_provider
    if _ollama_provider is None:
        cfg = Config().load("config/settings.yaml")
        api_base = cfg.get("providers.ollama.api_base")
        default_model = cfg.get("providers.ollama.default_model", "qwen2.5-coder:7b")
        _ollama_provider = get_provider(
            provider_type="ollama",
            model=default_model,
            api_base=api_base,
        )
    return _ollama_provider


def _list_sessions() -> List[Dict[str, Any]]:
    sessions = []
    if not CONVERSATIONS_DIR.exists():
        return sessions
    for entry in CONVERSATIONS_DIR.iterdir():
        if not entry.is_dir():
            continue
        info = {
            "id": entry.name,
            "path": str(entry),
            "goal": "",
            "created_at": "",
            "status": "unknown",
        }
        info_path = entry / "session_info.json"
        if info_path.exists():
            try:
                data = json.loads(info_path.read_text(encoding="utf-8"))
                info.update(
                    {
                        "goal": data.get("goal", ""),
                        "created_at": data.get("created_at", ""),
                        "status": data.get("status", "unknown"),
                    }
                )
            except Exception:
                pass
        sessions.append(info)
    return sorted(sessions, key=lambda item: item["id"], reverse=True)


def _resolve_session_dir(session_id: str) -> Optional[Path]:
    if not session_id:
        return None
    candidate = CONVERSATIONS_DIR / session_id
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _list_artifacts(session_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not session_dir.exists():
        return entries
    allowed_roots = ["src", "bin", "docs", "memory"]
    for root in allowed_roots:
        root_path = session_dir / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_dir():
                continue
            rel_path = path.relative_to(session_dir).as_posix()
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            entries.append(
                {
                    "path": rel_path,
                    "size": size,
                }
            )
    return sorted(entries, key=lambda item: item["path"])


def _safe_read_artifact(session_dir: Path, rel_path: str) -> Dict[str, Any]:
    result = {"ok": False, "path": rel_path, "content": "", "error": ""}
    if not rel_path:
        result["error"] = "Path is required."
        return result
    target = (session_dir / rel_path).resolve()
    try:
        session_root = session_dir.resolve()
    except Exception:
        result["error"] = "Invalid session directory."
        return result
    if session_root not in target.parents and target != session_root:
        result["error"] = "Path outside session workspace."
        return result
    if not target.exists() or not target.is_file():
        result["error"] = "File not found."
        return result
    try:
        size = target.stat().st_size
    except Exception:
        size = 0
    if size > 200_000:
        result["error"] = f"File too large to preview ({size} bytes)."
        return result
    try:
        result["content"] = target.read_text(encoding="utf-8", errors="replace")
        result["ok"] = True
        return result
    except Exception:
        result["error"] = "Failed to read file."
        return result


def _load_tests() -> List[Dict[str, Any]]:
    if not TESTS_PATH.exists():
        return []
    try:
        data = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return data.get("tests", [])
    except Exception:
        return []


def _save_tests(tests: List[Dict[str, Any]]) -> None:
    TESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tests": tests, "updated_at": datetime.now().isoformat()}
    TESTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


app = FastAPI(title="AI Autonom Web", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_sse(event_type: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True)
    return f"event: {event_type}\ndata: {data}\n\n"


def _attach_ui_bridge() -> None:
    ui = get_ui()
    if getattr(ui, "_sse_bridge", False):
        return

    def _listener(event: Dict[str, Any]) -> None:
        hub.publish(event)

    ui.subscribe(_listener)
    setattr(ui, "_sse_bridge", True)


def _run_orchestrator(
    goal: str,
    config_path: str,
    use_langgraph: bool,
    session_dir: Optional[str],
) -> None:
    orchestrator = NemotronOrchestrator(
        orchestrator_model=None,
        config_path=config_path,
        enable_checkpoints=False,
        enable_testing=False,
        enable_dashboard=True,
        cancel_event=run_state.cancel_event,
    )
    if session_dir:
        try:
            orchestrator.attach_session(session_dir)
        except Exception:
            pass
    hub.publish({"type": "run_start", "goal": goal})
    if use_langgraph:
        state = run_workflow(goal, orchestrator)
        if isinstance(state, dict) and "success" in state and "task" not in state:
            result = state
        else:
            errors = state.get("errors", []) if isinstance(state, dict) else []
            result = {
                "success": len(errors) == 0,
                "state": state,
            }
    else:
        result = orchestrator.run(goal)
    with run_state.lock:
        run_state.running = False
        run_state.cancelling = False
        run_state.last_result = result
        run_state.cancel_event.clear()
    if result.get("error") == "Cancelled":
        hub.publish({"type": "run_cancelled"})
    hub.publish({"type": "run_result", "result": result})


@app.on_event("startup")
async def _startup() -> None:
    hub.set_loop(asyncio.get_running_loop())
    _attach_ui_bridge()


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
async def state() -> Dict[str, Any]:
    snapshot = get_ui().snapshot()
    with run_state.lock:
        return {
            "running": run_state.running,
            "cancelling": run_state.cancelling,
            "use_langgraph": run_state.use_langgraph,
            "goal": run_state.goal,
            "session_id": run_state.session_id,
            "session_dir": run_state.session_dir,
            "last_result": run_state.last_result,
            "ui": snapshot,
        }


@app.get("/api/sessions")
async def sessions() -> Dict[str, Any]:
    return {"sessions": _list_sessions()}


@app.get("/api/session/current")
async def session_current() -> Dict[str, Any]:
    with run_state.lock:
        return {
            "session_id": run_state.session_id,
            "session_dir": run_state.session_dir,
        }


@app.post("/api/session/new")
async def session_new(request: Request) -> JSONResponse:
    payload = await request.json()
    label = str(payload.get("label", "")).strip() or "New session"
    session_dir = SessionManager().create_session(label)
    with run_state.lock:
        run_state.session_dir = session_dir
        run_state.session_id = Path(session_dir).name
    try:
        get_ui().set_session(run_state.session_id, run_state.session_dir)
    except Exception:
        pass
    return JSONResponse({"ok": True, "session_id": run_state.session_id, "session_dir": session_dir})


@app.post("/api/session/restore")
async def session_restore(request: Request) -> JSONResponse:
    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return JSONResponse({"ok": False, "error": "session_id is required."}, status_code=400)

    if session_id.lower() == "latest":
        sessions = _list_sessions()
        if not sessions:
            return JSONResponse({"ok": False, "error": "No sessions available."}, status_code=404)
        session_id = sessions[0]["id"]

    resolved = _resolve_session_dir(session_id)
    if not resolved:
        return JSONResponse({"ok": False, "error": "Session not found."}, status_code=404)

    with run_state.lock:
        run_state.session_dir = str(resolved)
        run_state.session_id = resolved.name

    try:
        get_ui().set_session(run_state.session_id, run_state.session_dir)
    except Exception:
        pass

    return JSONResponse({"ok": True, "session_id": run_state.session_id, "session_dir": run_state.session_dir})


@app.get("/api/artifacts/tree")
async def artifacts_tree(session_id: Optional[str] = None) -> Dict[str, Any]:
    session_dir = None
    if session_id:
        resolved = _resolve_session_dir(session_id)
        if resolved:
            session_dir = resolved
    if not session_dir:
        with run_state.lock:
            session_dir = Path(run_state.session_dir) if run_state.session_dir else None
    if not session_dir:
        return {"ok": False, "error": "No active session."}
    return {
        "ok": True,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "entries": _list_artifacts(session_dir),
    }


@app.get("/api/artifacts/file")
async def artifacts_file(path: str, session_id: Optional[str] = None) -> JSONResponse:
    session_dir = None
    if session_id:
        resolved = _resolve_session_dir(session_id)
        if resolved:
            session_dir = resolved
    if not session_dir:
        with run_state.lock:
            session_dir = Path(run_state.session_dir) if run_state.session_dir else None
    if not session_dir:
        return JSONResponse({"ok": False, "error": "No active session."}, status_code=400)

    result = _safe_read_artifact(session_dir, path)
    status = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status)


@app.post("/api/tools/check")
async def tools_check(request: Request) -> JSONResponse:
    payload = await request.json()
    tool = str(payload.get("tool", "")).strip()
    container = str(payload.get("container", "sandbox")).strip().lower()
    if not tool:
        return JSONResponse({"ok": False, "error": "tool is required."}, status_code=400)
    executor = _get_tool_executor()
    success, output = executor.execute_in_container(container, f"command -v {tool}")
    return JSONResponse(
        {
            "ok": True,
            "tool": tool,
            "container": container,
            "found": bool(success and output.strip()),
            "output": output,
        }
    )


@app.post("/api/tools/install")
async def tools_install(request: Request) -> JSONResponse:
    payload = await request.json()
    package = str(payload.get("package", "")).strip()
    install_type = str(payload.get("install_type", "pip")).strip().lower()
    container = str(payload.get("container", "sandbox")).strip().lower()
    if not package:
        return JSONResponse({"ok": False, "error": "package is required."}, status_code=400)

    executor = _get_tool_executor()
    if install_type == "apt":
        command = f"apt-get update && apt-get install -y {package}"
        success, output = executor.execute_in_container(container, command, timeout=300, user="root")
    else:
        command = f"python -m pip install --user {package}"
        success, output = executor.execute_in_container(container, command, timeout=300)

    return JSONResponse(
        {
            "ok": bool(success),
            "package": package,
            "container": container,
            "install_type": install_type,
            "output": output,
        }
    )


@app.get("/api/models")
async def models() -> Dict[str, Any]:
    cfg = Config().load("config/settings.yaml")
    discovery = ModelDiscovery()
    selector = DynamicModelSelector()
    available = discovery.scan_ollama_models()
    rankings = selector.get_model_rankings("balanced")
    snapshot = get_ui().snapshot()
    return {
        "active_model": snapshot.get("active_model"),
        "available": available,
        "rankings": rankings,
        "config": {
            "coding_models": cfg.get("ollama_models.coding_models", []),
            "linguistic_models": cfg.get("ollama_models.linguistic_models", []),
            "reasoning_models": cfg.get("ollama_models.reasoning_models", []),
            "vram_limit_gb": cfg.get("execution.vram_limit_gb", None),
        },
    }


@app.post("/api/models/switch")
async def models_switch(request: Request) -> JSONResponse:
    payload = await request.json()
    model = str(payload.get("model", "")).strip()
    unload_current = bool(payload.get("unload_current", True))
    current_model = str(payload.get("current_model", "")).strip()
    persist = bool(payload.get("persist", False))
    warm = bool(payload.get("warm", True))
    force = bool(payload.get("force", False))

    if not model:
        return JSONResponse({"ok": False, "error": "model is required."}, status_code=400)

    with run_state.lock:
        if run_state.running and not force:
            return JSONResponse(
                {"ok": False, "error": "Cannot switch models during an active run."},
                status_code=409,
            )

    available = {
        m.get("name")
        for m in ModelDiscovery().scan_ollama_models()
        if isinstance(m, dict) and m.get("name")
    }
    if available and model not in available:
        return JSONResponse(
            {"ok": False, "error": f"Model not available: {model}"},
            status_code=404,
        )

    provider = _get_ollama_provider()
    if unload_current and current_model and current_model != model:
        if hasattr(provider, "unload_model"):
            provider.unload_model(current_model)

    if warm:
        try:
            provider.chat(
                [LLMMessage(role="user", content="ping")],
                model=model,
                temperature=0.0,
                max_tokens=1,
            )
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "error": f"Failed to warm model: {exc}"},
                status_code=500,
            )

    try:
        get_ui().set_active_model(model)
    except Exception:
        pass

    if persist:
        cfg = Config().load("config/settings.yaml")
        cfg.set("providers.ollama.default_model", model)
        cfg.save("config/settings.yaml")

    return JSONResponse({"ok": True, "active_model": model})


@app.get("/api/tests")
async def tests_list() -> Dict[str, Any]:
    return {"tests": _load_tests()}


@app.post("/api/tests")
async def tests_add(request: Request) -> JSONResponse:
    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt is required."}, status_code=400)
    tests = _load_tests()
    test_id = payload.get("id") or f"test_{len(tests)+1}"
    tests.append({"id": test_id, "name": name or test_id, "prompt": prompt})
    _save_tests(tests)
    return JSONResponse({"ok": True, "tests": tests})


@app.delete("/api/tests/{test_id}")
async def tests_delete(test_id: str) -> JSONResponse:
    tests = _load_tests()
    tests = [t for t in tests if t.get("id") != test_id]
    _save_tests(tests)
    return JSONResponse({"ok": True, "tests": tests})


@app.post("/api/run")
async def run(request: Request) -> JSONResponse:
    payload = await request.json()
    goal = str(payload.get("goal", "")).strip()
    config_path = str(payload.get("config_path", "config/settings.yaml"))
    use_langgraph = bool(payload.get("use_langgraph", True))
    session_id = str(payload.get("session_id", "")).strip()
    new_session = bool(payload.get("new_session", False))

    if not goal:
        return JSONResponse({"ok": False, "error": "Goal is required."}, status_code=400)

    session_dir: Optional[str] = None
    if new_session:
        session_dir = SessionManager().create_session(goal)
    elif session_id:
        resolved = _resolve_session_dir(session_id)
        if not resolved:
            return JSONResponse({"ok": False, "error": "Session not found."}, status_code=404)
        session_dir = str(resolved)

    with run_state.lock:
        if run_state.running:
            return JSONResponse(
                {"ok": False, "error": "Run already in progress."}, status_code=409
            )
        if not session_dir:
            if run_state.session_dir:
                session_dir = run_state.session_dir
            else:
                session_dir = SessionManager().create_session(goal)
        run_state.session_dir = session_dir
        run_state.session_id = Path(session_dir).name if session_dir else ""
        run_state.running = True
        run_state.cancelling = False
        run_state.use_langgraph = use_langgraph
        run_state.goal = goal
        run_state.last_result = None
        run_state.cancel_event.clear()

    thread = threading.Thread(
        target=_run_orchestrator,
        args=(goal, config_path, use_langgraph, session_dir),
        daemon=True,
    )
    run_state.thread = thread
    thread.start()

    return JSONResponse({"ok": True})


@app.post("/api/cancel")
async def cancel() -> JSONResponse:
    with run_state.lock:
        if not run_state.running:
            return JSONResponse(
                {"ok": False, "error": "No active run."}, status_code=409
            )
        run_state.cancelling = True
        run_state.cancel_event.set()

    hub.publish({"type": "run_cancelling"})
    return JSONResponse({"ok": True})


@app.post("/api/reset")
async def reset() -> JSONResponse:
    with run_state.lock:
        if run_state.running:
            return JSONResponse(
                {"ok": False, "error": "Cannot reset during an active run."}, status_code=409
            )
        run_state.goal = ""
        run_state.last_result = None
        run_state.cancelling = False
        run_state.use_langgraph = True
        run_state.cancel_event.clear()

    get_ui().reset()
    if run_state.session_dir:
        try:
            get_ui().set_session(run_state.session_id, run_state.session_dir)
        except Exception:
            pass
    return JSONResponse({"ok": True})


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    queue = await hub.subscribe()

    async def event_stream():
        try:
            snapshot = get_ui().snapshot()
            yield _format_sse("snapshot", snapshot)

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                event_type = str(event.get("type", "message"))
                yield _format_sse(event_type, event)
        finally:
            await hub.unsubscribe(queue)

    return StreamingResponse(event_stream(), headers=SSE_HEADERS)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("multi_agent_framework.ui.web.backend:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
