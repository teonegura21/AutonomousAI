# Architecture Gap Assessment

This is a short delta list between the current codebase and the requirements in
`multi_agent_framework_architecture.md`. It is intentionally concise and meant
to guide the refactor.

## Major Gaps

- `multi_agent_framework/` package structure is mostly missing or empty.
- Web UI (FastAPI + React + SSE) is not yet the default runtime.
- LangGraph workflow (`workflow/graph.py`, `workflow/state.py`, `workflow/nodes.py`) is missing.
- Split memory layers (`short_term`, `working`, `long_term`) are not implemented under the new package.
- New inference router and model routing config (`config/models.yaml`) are missing.
- Sandbox boundary (`sandbox/docker_sandbox.py`, `sandbox/e2b_sandbox.py`) is not wired to the new flow.
- Required code imports from CAI / OpenManus / OpenCode are not yet integrated.

## Existing Capabilities That Map Well

- Dynamic Ollama model discovery and role assignment exist in `ai_autonom/`.
- Vector memory (ChromaDB) and shared context store exist and are now path-configurable.
- Orchestrator already emits UI events that can be streamed to a web UI.

## Next Alignment Steps

- Create the full `multi_agent_framework/` module tree and wire imports.
- Bring in the specified files from `third_party` and wrap them with the new agent base.
- Replace the CLI default with the web UI and add SSE streaming.
- Move orchestration state into a LangGraph workflow layer.
