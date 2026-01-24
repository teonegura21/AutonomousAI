#!/usr/bin/env python3
"""
Enhanced Nemotron Orchestrator
Integrates all MVP features: model discovery, memory, tools, error recovery, checkpoints
Supports multiple LLM providers: Ollama (local), OpenAI, Azure OpenAI, and compatible APIs
"""

from typing import Dict, List, Optional, Any
import threading
import json
import time
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Core imports
from ai_autonom.core.config import Config, get_config
from ai_autonom.core.agent_registry import AgentRegistry, AgentDefinition
from ai_autonom.core.model_discovery import ModelDiscovery
from ai_autonom.core.model_selector import DynamicModelSelector
from ai_autonom.core.model_watcher import ModelWatcher
from ai_autonom.core.llm_provider import (
    LLMProviderFactory, LLMConfig, LLMMessage, ProviderType,
    BaseLLMProvider, OllamaProvider, OpenAIProvider, MultiProvider
)

# Memory imports
from ai_autonom.memory.task_memory import TaskMemory
from ai_autonom.memory.vector_store import VectorMemoryStore

# Tool imports
from ai_autonom.tools.tool_executor import ToolExecutor
from ai_autonom.tools.code_executor import CodeExecutor

# Orchestration imports
from ai_autonom.orchestration.intent_analyzer import IntentAnalyzer
from multi_agent_framework.openmanus.runner import run_toolcall_task
from ai_autonom.orchestration.agent_messaging import AgentMessageBus, MessageType
from ai_autonom.orchestration.error_recovery import ErrorRecovery, RecoveryExecutor, RecoveryAction
from ai_autonom.orchestration.human_checkpoint import HumanCheckpointManager
from ai_autonom.orchestration.testing_workflow import TestingWorkflow
from ai_autonom.orchestration.langgraph_workflow import MultiAgentWorkflow, WorkflowState
# from ai_autonom.patterns.handoffs import HandoffManager (REPLACED WITH NEW MODULE)
from ai_autonom.orchestration.handoff_manager import HandoffManager
from ai_autonom.memory.knowledge_base import KnowledgeBase
from ai_autonom.core.session_manager import SessionManager
from ai_autonom.monitoring.debug_logger import log_debug

# Capability checking for honest failure detection
try:
    from ai_autonom.core.capability_checker import (
        CapabilityChecker,
        get_enhanced_system_prompt,
        HONEST_FAILURE_PROMPT
    )
    CAPABILITY_CHECKER_AVAILABLE = True
except ImportError:
    CAPABILITY_CHECKER_AVAILABLE = False
    CapabilityChecker = None

# IPC Broker for DB-based agent communication
try:
    from ai_autonom.orchestration.ipc_broker import get_broker, IPCBroker
    IPC_AVAILABLE = True
except ImportError:
    IPC_AVAILABLE = False
    get_broker = None
    IPCBroker = None

# Monitoring imports
from ai_autonom.monitoring.telemetry import ExecutionMonitor
from ai_autonom.monitoring.ui import get_ui


class NemotronOrchestrator:
    """
    Enhanced Nemotron Orchestrator with full MVP features:
    - Multi-provider LLM support (Ollama, OpenAI, Azure, etc.)
    - Dynamic model selection
    - Intent analysis
    - Task memory (inter-task context)
    - Vector memory (semantic search)
    - Tool execution
    - Error recovery
    - Human checkpoints
    - Testing workflow
    - Telemetry
    """
    
    def __init__(
        self,
        orchestrator_model: str = None,
        orchestrator_provider: str = None,
        config_path: str = "config/settings.yaml",
        enable_checkpoints: bool = True,
        enable_testing: bool = True,
        enable_dashboard: bool = True,
        cancel_event: Optional[threading.Event] = None
    ):
        # Load configuration
        self.config = Config().load(config_path)
        self.enable_dashboard = enable_dashboard
        self._refresh_dynamic_models()
        
        if self.enable_dashboard:
            self.ui = get_ui()
        else:
            self.ui = None

        self.cancel_event = cancel_event or threading.Event()
        
        # Orchestrator LLM settings
        self.orchestrator_model = orchestrator_model or self.config.get(
            'orchestrator.model', 'huihui_ai/orchestrator-abliterated'
        )
        self.orchestrator_provider_type = orchestrator_provider or self.config.get(
            'orchestrator.provider', 'ollama'
        )
        
        # Initialize LLM providers
        self._init_providers()
        
        # Initialize components
        self._init_components(enable_checkpoints, enable_testing)
        
        provider_name = self.orchestrator_provider_type.upper()
        print(f"\n[ORCHESTRATOR] Initialized")
        print(f"  Model: {self.orchestrator_model}")
        print(f"  Provider: {provider_name}")

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def _cancelled(self) -> bool:
        return bool(self.cancel_event and self.cancel_event.is_set())

    def _cancel_result(self) -> Dict[str, Any]:
        if self.ui:
            self.ui.log("Cancellation requested. Stopping workflow.", "WARNING")
        return {"success": False, "error": "Cancelled"}
    
    def _init_providers(self):
        """Initialize LLM providers based on configuration"""
        self.providers: Dict[str, BaseLLMProvider] = {}
        
        # Initialize Ollama provider
        ollama_config = self.config.get_section('providers').get('ollama', {})
        if ollama_config.get('enabled', True):
            try:
                config = LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model=ollama_config.get('default_model', 'qwen2.5-coder:7b'),
                    api_base=ollama_config.get('api_base')
                )
                self.providers['ollama'] = LLMProviderFactory.create(config)
                if self.providers['ollama'].is_available():
                    print("[PROVIDER] Ollama: Available")
                else:
                    print("[PROVIDER] Ollama: Not running")
            except Exception as e:
                print(f"[PROVIDER] Ollama: Failed - {e}")
        
        # Initialize OpenAI provider
        openai_config = self.config.get_section('providers').get('openai', {})
        if openai_config.get('enabled', False):
            api_key = openai_config.get('api_key', '').replace('${OPENAI_API_KEY}', os.getenv('OPENAI_API_KEY', ''))
            if api_key:
                try:
                    config = LLMConfig(
                        provider=ProviderType.OPENAI,
                        model=openai_config.get('default_model', 'gpt-4o-mini'),
                        api_key=api_key,
                        organization=openai_config.get('organization')
                    )
                    self.providers['openai'] = LLMProviderFactory.create(config)
                    print("[PROVIDER] OpenAI: Available")
                except Exception as e:
                    print(f"[PROVIDER] OpenAI: Failed - {e}")
        
        # Initialize Azure OpenAI provider
        azure_config = self.config.get_section('providers').get('azure_openai', {})
        if azure_config.get('enabled', False):
            api_key = azure_config.get('api_key', '').replace('${AZURE_OPENAI_API_KEY}', os.getenv('AZURE_OPENAI_API_KEY', ''))
            endpoint = azure_config.get('endpoint', '').replace('${AZURE_OPENAI_ENDPOINT}', os.getenv('AZURE_OPENAI_ENDPOINT', ''))
            if api_key and endpoint:
                try:
                    config = LLMConfig(
                        provider=ProviderType.AZURE_OPENAI,
                        model=azure_config.get('deployment', 'gpt-4'),
                        api_key=api_key,
                        api_base=endpoint,
                        extra_params={'api_version': azure_config.get('api_version', '2024-02-01')}
                    )
                    self.providers['azure_openai'] = LLMProviderFactory.create(config)
                    print("[PROVIDER] Azure OpenAI: Available")
                except Exception as e:
                    print(f"[PROVIDER] Azure OpenAI: Failed - {e}")
        
        # Initialize OpenAI-compatible provider (for local servers)
        compat_config = self.config.get_section('providers').get('openai_compatible', {})
        if compat_config.get('enabled', False):
            try:
                config = LLMConfig(
                    provider=ProviderType.OPENAI_COMPATIBLE,
                    model=compat_config.get('default_model', 'local-model'),
                    api_key=compat_config.get('api_key', 'not-needed'),
                    api_base=compat_config.get('api_base', 'http://localhost:8080/v1')
                )
                self.providers['openai_compatible'] = LLMProviderFactory.create(config)
                print("[PROVIDER] OpenAI-Compatible: Available")
            except Exception as e:
                print(f"[PROVIDER] OpenAI-Compatible: Failed - {e}")
        
        # Set orchestrator provider
        self.orchestrator_provider = self.providers.get(
            self.orchestrator_provider_type,
            self.providers.get('ollama')
        )
        
        if not self.orchestrator_provider:
            raise RuntimeError("No LLM provider available!")
    
    def get_provider_for_agent(self, agent: AgentDefinition) -> BaseLLMProvider:
        """Get the appropriate provider for an agent"""
        provider_type = getattr(agent, 'provider', 'ollama')
        return self.providers.get(provider_type, self.orchestrator_provider)
    
    def _init_components(self, enable_checkpoints: bool, enable_testing: bool):
        """Initialize all components"""
        # Agent Registry
        self.registry = AgentRegistry()
        
        # Model Management
        self.model_discovery = ModelDiscovery()
        self.model_selector = DynamicModelSelector()
        auto_benchmark = self.config.get("ollama_models.auto_benchmark", False)
        self.model_watcher = ModelWatcher(
            self.model_discovery,
            interval=60,
            auto_benchmark=auto_benchmark,
            on_new_model=self._on_new_model
        )
        self.model_watcher.start()

        # IPC Broker (DB-based inter-agent comms)
        self.ipc: Optional[IPCBroker] = get_broker() if IPC_AVAILABLE else None
        self.ipc_enabled = self.ipc is not None
        
        # Memory Systems
        self.task_memory = TaskMemory()
        self.vector_store = VectorMemoryStore()
        self.knowledge_base = KnowledgeBase.get_instance()
        
        # Tools
        require_kali = self.config.get('execution.require_kali_for_security', True)
        self.tool_executor = ToolExecutor(
            workspace_dir="outputs",
            require_kali_for_security=require_kali
        )
        self.code_executor = CodeExecutor(workspace_dir="outputs")
        
        # Orchestration Components
        self.intent_analyzer = IntentAnalyzer()
        self.message_bus = AgentMessageBus()
        max_retries = self.config.get("execution.max_retries", 3)
        self.error_recovery = ErrorRecovery(max_retries=max_retries)
        self.checkpoint = HumanCheckpointManager(auto_approve_low_risk=not enable_checkpoints)
        self.testing_workflow = TestingWorkflow()
        self.workflow_engine = MultiAgentWorkflow(orchestrator=self)
        self.handoff_manager = HandoffManager(self.registry, self.knowledge_base)
        self.session_manager = SessionManager()
        
        # Monitoring
        self.monitor = ExecutionMonitor()
        
        # State
        self.current_workflow_id = None
        self.current_plan: List[Dict[str, Any]] = []
        self.workflow_paused = False
        self.enable_testing = enable_testing
        self.enable_checkpoints = enable_checkpoints
        self.session_workspace: Optional[str] = None
        
        # Auto-pull Ollama models if configured
        self._ensure_ollama_models()

        # Setup registry after all components are initialized
        self._setup_registry()

    def start_session(self, user_goal: str) -> str:
        """Create a session workspace and configure tools to use it."""
        session_path = self.session_manager.create_session(user_goal)
        self._configure_session_workspace(session_path)
        if self.ui:
            try:
                self.ui.set_session(
                    self.session_manager.current_session_id,
                    self.session_manager.current_session_dir,
                )
            except Exception:
                pass
        return session_path

    def attach_session(self, session_dir: str) -> str:
        """Attach to an existing session workspace without creating a new folder."""
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        for folder in ("src", "bin", "docs", "memory"):
            (session_path / folder).mkdir(parents=True, exist_ok=True)

        self.session_manager.current_session_dir = str(session_path)
        self.session_manager.current_session_id = session_path.name
        self._configure_session_workspace(str(session_path))
        if self.ui:
            try:
                self.ui.set_session(
                    self.session_manager.current_session_id,
                    self.session_manager.current_session_dir,
                )
            except Exception:
                pass
        return str(session_path)

    def _configure_session_workspace(self, session_path: str) -> None:
        workspace = Path(session_path)
        workspace.mkdir(parents=True, exist_ok=True)
        self.session_workspace = workspace.as_posix()

        os.environ["AI_AUTONOM_WORKSPACE"] = str(workspace)
        os.environ["AI_AUTONOM_WORKSPACE_NAME"] = workspace.name

        self.tool_executor.set_workspace(str(workspace))
        if hasattr(self.code_executor, "set_workspace"):
            self.code_executor.set_workspace(str(workspace))
        else:
            self.code_executor.workspace_dir = str(workspace)

    def _persist_session_artifacts(
        self,
        user_goal: str,
        tasks: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> None:
        session_dir = self.session_manager.current_session_dir
        if not session_dir:
            return

        session_path = Path(session_dir)
        docs_dir = session_path / "docs"
        memory_dir = session_path / "memory"
        docs_dir.mkdir(parents=True, exist_ok=True)
        memory_dir.mkdir(parents=True, exist_ok=True)

        def trim(text: str, limit: int = 2500) -> str:
            if not text:
                return ""
            return text[:limit] + ("..." if len(text) > limit else "")

        success_count = sum(1 for r in results if r.get("success"))
        snapshot = {
            "goal": user_goal,
            "workflow_id": self.current_workflow_id,
            "created_at": datetime.now().isoformat(),
            "tasks_total": len(tasks),
            "tasks_successful": success_count,
            "tasks_failed": len(tasks) - success_count,
            "outputs": [],
        }

        try:
            snapshot["outputs"] = self.task_memory.get_all_outputs(self.current_workflow_id)
        except Exception:
            pass

        try:
            self.task_memory.export_workflow_db(
                str(memory_dir / "task_memory_session.db"),
                self.current_workflow_id,
            )
        except Exception:
            pass

        try:
            self._write_memory_json("task_outputs.json", snapshot, versioned=True)
        except Exception:
            pass

        try:
            kb_path = Path(self.knowledge_base.db_path)
            if kb_path.exists():
                shutil.copyfile(kb_path, memory_dir / "knowledge_base.json")
        except Exception:
            pass

        try:
            vector_stats = self.vector_store.get_stats()
            with open(memory_dir / "vector_stats.json", "w", encoding="utf-8") as f:
                json.dump(vector_stats, f, indent=2, ensure_ascii=True)
        except Exception:
            pass

        try:
            vector_snapshot = {}
            for task in tasks:
                task_id = task.get("id")
                if not task_id:
                    continue
                context = self.vector_store.get_task_context(task_id)
                trimmed_context = {}
                for key, items in context.items():
                    trimmed_context[key] = [
                        {
                            "content": trim(item.get("content", ""), 2500),
                            "metadata": item.get("metadata", {}),
                        }
                        for item in items
                    ]
                if any(trimmed_context.values()):
                    vector_snapshot[task_id] = trimmed_context

            if vector_snapshot:
                with open(
                    memory_dir / "vector_context.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(vector_snapshot, f, indent=2, ensure_ascii=True)
        except Exception:
            pass

        summary_path = memory_dir / "summary.md"
        if not summary_path.exists():
            kb_summary = self.knowledge_base.get_summary().strip()
            summary_lines = [
                "# Session Memory Summary",
                "",
                f"Goal: {user_goal}",
                f"Session: {self.session_manager.current_session_id}",
                f"Workflow: {self.current_workflow_id}",
                f"Status: {success_count}/{len(tasks)} tasks succeeded",
                "",
                "Notes:",
                "- Source code should live in src/.",
                "- Compiled binaries should live in bin/.",
                "- Final report should live in docs/.",
                "- Session notes and snapshots live in memory/.",
                "- Session DB snapshot: memory/task_memory_session.db",
                "- Vector snapshot: memory/vector_stats.json",
                "",
            ]
            if kb_summary:
                summary_lines.extend(["Knowledge Base Summary:", kb_summary, ""])
            summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        report_path = docs_dir / "final_report.md"
        report_lines = [
            "# AI Autonom Session Report",
            "",
            f"Goal: {user_goal}",
            f"Session: {self.session_manager.current_session_id}",
            f"Workflow: {self.current_workflow_id}",
            f"Status: {success_count}/{len(tasks)} tasks succeeded",
            "",
            "## Plan",
        ]
        for task in tasks:
            task_id = task.get("id", "task")
            description = trim(task.get("description", ""), 200)
            report_lines.append(f"- {task_id}: {description}")

        report_lines.extend(["", "## Results"])
        for result in results:
            agent = result.get("agent", "unknown")
            status = "SUCCESS" if result.get("success") else "FAILED"
            output = trim(result.get("output", ""), 1000)
            report_lines.append(f"- {agent}: {status}")
            if output:
                report_lines.append(f"  Output: {output}")

        report_lines.extend(
            [
                "",
                "## Artifacts",
                f"- Source code: {session_path / 'src'}",
                f"- Binaries: {session_path / 'bin'}",
                f"- Docs: {session_path / 'docs'}",
                f"- Memory: {session_path / 'memory'}",
                "",
                "## Instructions",
                "- Review memory/intent_analysis.json for the structured plan summary.",
                "- Review memory/task_briefs.json for recommended tools per task.",
                "- If a task failed, inspect full_orchestration_log.txt for the tool error.",
            ]
        )

        report_content = "\n".join(report_lines)
        report_path.write_text(report_content, encoding="utf-8")
        if self.current_workflow_id:
            versioned_path = docs_dir / f"final_report_{self.current_workflow_id}.md"
            versioned_path.write_text(report_content, encoding="utf-8")
        if self.ui:
            try:
                self.ui.set_final_report(
                    {
                        "path": str(report_path),
                        "content": trim(report_content, 2500),
                    }
                )
            except Exception:
                pass

    def _write_memory_json(
        self,
        filename: str,
        payload: Dict[str, Any],
        versioned: bool = False,
    ) -> None:
        session_dir = self.session_manager.current_session_dir
        if not session_dir:
            return
        memory_dir = Path(session_dir) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(memory_dir / filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
        except Exception:
            pass
        if versioned and self.current_workflow_id:
            try:
                base = Path(filename)
                versioned_name = f"{base.stem}_{self.current_workflow_id}{base.suffix}"
                with open(memory_dir / versioned_name, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=True)
            except Exception:
                pass

    def _build_task_briefs(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        briefs: List[Dict[str, Any]] = []
        agents = self.registry.get_all_agents()
        agent_map = {agent.id: agent for agent in agents}
        max_iterations = int(self.config.get("execution.max_tool_iterations", 3))

        for task in tasks:
            task_id = task.get("id")
            assigned_id = task.get("assigned_agent")
            agent = agent_map.get(assigned_id)
            model = None
            if agent:
                model = self._select_model_for_task(task, agent)

            tools = task.get("tools", []) or []
            recommended_steps = [
                "Write source files under src/ using filesystem_write.",
                "Execute code using python_exec or bash_exec.",
            ]
            if "pytest_run" in tools:
                recommended_steps.append("Run tests with pytest_run.")
            if task.get("type") == "documentation":
                recommended_steps.append("Save the report in docs/.")

            briefs.append(
                {
                    "task_id": task_id,
                    "description": task.get("description", ""),
                    "assigned_agent": assigned_id,
                    "recommended_model": model,
                    "recommended_tools": tools,
                    "recommended_steps": recommended_steps,
                    "max_tool_iterations": max_iterations,
                }
            )

        return briefs
    def _on_new_model(self, capabilities: Dict[str, Any]) -> None:
        """Refresh dynamic models and registry when new models appear."""
        self._refresh_dynamic_models()
        try:
            from ai_autonom.core.agent_registry import setup_initial_registry
            setup_initial_registry()
        except Exception:
            pass
        self._setup_registry()

    def _refresh_dynamic_models(self) -> None:
        """Sync Ollama models and pick best defaults for core roles."""
        try:
            discovery = ModelDiscovery()
            auto_register = self.config.get('ollama_models.auto_register', True)
            auto_benchmark = self.config.get('ollama_models.auto_benchmark', False)
            sync = discovery.sync_ollama_models(auto_register=auto_register, auto_benchmark=auto_benchmark)
            available = {m for m in set(sync.get("available", [])) if not discovery.is_embedding_model(m)}
            if not available:
                self._available_models = set()
                print("[ORCHESTRATOR] No Ollama models available; keeping configured defaults")
                return
            self._available_models = set(available)

            selector = DynamicModelSelector()
            max_vram = self.config.get("execution.vram_limit_gb", 20)

            def pick_model(task_type: str, config_key: str, fallback: str) -> str:
                preferred = self.config.get(config_key)
                if preferred and preferred in available:
                    return preferred
                best = selector.select_best_model(task_type, {"max_vram": max_vram})
                if isinstance(best, dict) and best.get("model_name") in available:
                    return best["model_name"]
                if available:
                    return sorted(available)[0]
                return fallback

            coder_model = pick_model("coding", "agents.coder.model", "qwen2.5-coder:7b")
            doc_model = pick_model("documentation", "agents.linguistic.model", coder_model)
            orchestrator_model = pick_model("reasoning", "orchestrator.model", coder_model)

            self.config.set("agents.coder.model", coder_model)
            self.config.set("agents.linguistic.model", doc_model)
            self.config.set("orchestrator.model", orchestrator_model)
            self.config.set("providers.ollama.default_model", coder_model)
        except Exception as e:
            self._available_models = set()
            print(f"[ORCHESTRATOR] Dynamic model refresh failed: {e}")

    def _ensure_ollama_models(self) -> None:
        """Auto-pull and register configured Ollama models."""
        ollama_cfg = self.config.get('ollama_models', {})
        if not ollama_cfg or not ollama_cfg.get('auto_pull', False):
            return

        model_lists: List[str] = []
        for key in ("coding_models", "linguistic_models", "reasoning_models"):
            model_lists.extend(ollama_cfg.get(key, []) or [])

        if not model_lists:
            return

        auto_register = ollama_cfg.get('auto_register', True)
        auto_benchmark = ollama_cfg.get('auto_benchmark', True)

        self.model_discovery.ensure_models(
            model_lists,
            auto_register=auto_register,
            auto_benchmark=auto_benchmark,
            pull_missing=True
        )

    def _select_model_for_capability(self, capability: str, fallback: str) -> str:
        """Select best available model for a capability, with fallback."""
        try:
            task_type = self._capability_to_task_type(capability)
            preferred = self._select_preferred_model(task_type)
            if preferred:
                return preferred
            max_vram = self.config.get('execution.vram_limit_gb', 20)
            best = self.model_selector.get_model_for_capability(capability, max_vram=max_vram)
            return best or fallback
        except Exception:
            return fallback

    def _capability_to_task_type(self, capability: Optional[str]) -> str:
        cap = (capability or "").lower()
        coding_caps = {"code_generation", "debugging", "testing", "refactoring", "python", "technical_tasks"}
        doc_caps = {"documentation", "summarization", "text_generation", "formatting", "explanation"}
        reason_caps = {"task_decomposition", "planning", "analysis", "reasoning", "math"}
        fast_caps = {"simple_tasks"}

        if cap in coding_caps:
            return "coding"
        if cap in doc_caps:
            return "documentation"
        if cap in reason_caps:
            return "reasoning"
        if cap in fast_caps:
            return "fast"
        return "balanced"

    def _select_preferred_model(self, task_type: str) -> Optional[str]:
        """Pick a preferred model for a task type from config if available."""
        model_lists = {
            "coding": self.config.get("ollama_models.coding_models", []) or [],
            "documentation": self.config.get("ollama_models.linguistic_models", []) or [],
            "reasoning": self.config.get("ollama_models.reasoning_models", []) or [],
        }
        preferred_list = model_lists.get(task_type, [])
        if not preferred_list:
            return None

        available = getattr(self, "_available_models", None)
        if not available:
            try:
                available = {
                    m.get("name")
                    for m in self.model_discovery.scan_ollama_models()
                    if m.get("name")
                }
            except Exception:
                available = set()

        for model_name in preferred_list:
            if not available or model_name in available:
                return model_name
        return None

    def _infer_capability_from_description(self, description: str) -> Optional[str]:
        if not description:
            return None
        desc = description.lower()
        if any(token in desc for token in ("analy", "reason", "logic", "math", "prove", "derive", "calculate")):
            return "analysis"
        if any(token in desc for token in ("document", "summar", "report", "explain", "format")):
            return "documentation"
        if any(token in desc for token in ("implement", "code", "build", "fix", "debug", "refactor", "test")):
            return "code_generation"
        return None

    def _select_model_for_task(self, task: Dict[str, Any], agent: AgentDefinition) -> str:
        override = task.get("model_override")
        if override:
            return override
        capability = task.get("required_capability") or self._infer_capability_from_description(
            task.get("description", "")
        )
        if not capability:
            return agent.model_name
        return self._select_model_for_capability(capability, agent.model_name)

    def _is_security_agent_id(self, agent_id: Optional[str]) -> bool:
        if not agent_id:
            return False
        agent_id = agent_id.lower()
        markers = [
            "red_team",
            "pentester",
            "security",
            "kali",
            "cai",
            "dfir",
            "bug_bounty",
            "recon",
        ]
        return any(marker in agent_id for marker in markers)

    def _sanitize_task_tools(self, tasks: List[Dict[str, Any]]) -> None:
        """Remove cai_* tools from non-security tasks and set sane defaults."""
        replacements = {
            "cai_generic_linux_command": "bash_exec",
            "cai_execute_code": "python_exec",
            "cai_filesystem_read": "filesystem_read",
            "cai_filesystem_write": "filesystem_write",
            "cai_filesystem_search": "filesystem_search",
        }
        for task in tasks:
            agent_id = task.get("assigned_agent")
            if self._is_security_agent_id(agent_id):
                continue

            tools = list(task.get("tools", []) or [])
            cleaned: List[str] = []
            for tool_id in tools:
                if tool_id.startswith("cai_"):
                    replacement = replacements.get(tool_id)
                    if replacement:
                        cleaned.append(replacement)
                    continue
                cleaned.append(tool_id)

            cleaned = list(dict.fromkeys(cleaned))
            if not cleaned:
                capability = (task.get("required_capability") or "").lower()
                if capability in {"documentation", "summarization", "formatting", "explanation"}:
                    cleaned = ["filesystem_read", "filesystem_write"]
                elif capability in {"research", "information_gathering"}:
                    cleaned = ["web_search", "web_fetch", "filesystem_read", "filesystem_write"]
                else:
                    cleaned = ["filesystem_read", "filesystem_write", "bash_exec", "python_exec"]

            task["tools"] = cleaned

    def _estimate_model_vram(self, model_name: str) -> float:
        try:
            return self.model_discovery._estimate_vram(model_name)
        except Exception:
            return 4.0

    def _get_ipc_dependency_context(self, dependencies: List[str]) -> Dict[str, Any]:
        """Fetch dependency outputs from IPC shared context (DB)."""
        if not self.ipc_enabled or not dependencies:
            return {}

        context: Dict[str, Any] = {}
        for dep_id in dependencies:
            output = self.ipc.get_shared(f"task:{dep_id}:output")
            if output:
                context[dep_id] = {"output": output}
        return context

    def _publish_task_output(self, task_id: str, output: str, agent_name: str) -> None:
        """Publish task outputs to IPC shared context (DB)."""
        if not self.ipc_enabled:
            return

        summary = output[:2000] if output else ""
        self.ipc.set_shared(f"task:{task_id}:output", output)
        self.ipc.set_shared(f"task:{task_id}:summary", summary)
        self.ipc.publish("task_completed", {
            "task_id": task_id,
            "agent": agent_name,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })

        if self.ui:
            self.ui.log(f"IPC: task_completed: {task_id} ({agent_name})", "INFO")

    def _ensure_sequential_dependencies(self, tasks: List[Dict[str, Any]]) -> None:
        """Ensure each task depends on the previous one if no dependencies are set."""
        prev_task_id = None
        for task in tasks:
            deps = task.get("dependencies") or []
            if prev_task_id and not deps:
                task["dependencies"] = [prev_task_id]
            prev_task_id = task.get("id") or prev_task_id

    def _update_plan_storage(self, tasks: List[Dict[str, Any]]) -> None:
        """Persist plan to DB + IPC for this conversation."""
        if self.current_workflow_id:
            self.task_memory.save_workflow_plan(self.current_workflow_id, tasks)
        if self.ipc_enabled and self.current_workflow_id:
            self.ipc.set_shared(f"workflow:{self.current_workflow_id}:plan", tasks)
            self.ipc.publish("workflow_plan_updated", {
                "workflow_id": self.current_workflow_id,
                "task_count": len(tasks),
                "timestamp": datetime.now().isoformat()
            })
            if self.ui:
                self.ui.log("IPC: workflow_plan_updated", "INFO")

    def _handle_control_command(self, command: str, tasks: List[Dict[str, Any]]) -> str:
        """Handle workflow control commands and update plan/DB."""
        parts = command.strip().split()
        if not parts:
            return "Empty command"

        cmd = parts[0].lower()

        if cmd == "help":
            return "Commands: resume | pause | stop | list | set | assign | deps | move | insert | remove | swap"

        if cmd == "pause":
            self.workflow_paused = True
            return "Workflow paused"

        if cmd == "resume":
            self.workflow_paused = False
            return "Workflow resumed"

        if cmd == "stop":
            self.workflow_paused = True
            return "Workflow stopped"

        if cmd == "list":
            return ", ".join([f"{t.get('id')}:{t.get('assigned_agent')}" for t in tasks]) or "No tasks"

        if cmd == "set" and len(parts) >= 4 and parts[2].lower() == "desc":
            task_id = parts[1]
            desc = " ".join(parts[3:]).strip()
            for t in tasks:
                if t.get("id") == task_id:
                    t["description"] = desc
                    self.task_memory.update_task_definition(task_id, description=desc)
                    self._update_plan_storage(tasks)
                    return f"Updated description for {task_id}"
            return f"Task not found: {task_id}"

        if cmd == "assign" and len(parts) >= 3:
            task_id = parts[1]
            agent_id = parts[2]
            for t in tasks:
                if t.get("id") == task_id:
                    t["assigned_agent"] = agent_id
                    self.task_memory.update_task_definition(task_id, assigned_agent=agent_id)
                    self._update_plan_storage(tasks)
                    return f"Assigned {task_id} to {agent_id}"
            return f"Task not found: {task_id}"

        if cmd == "deps" and len(parts) >= 3:
            task_id = parts[1]
            deps = parts[2].split(",") if parts[2] else []
            for t in tasks:
                if t.get("id") == task_id:
                    t["dependencies"] = deps
                    self.task_memory.update_task_dependencies(task_id, deps)
                    self._update_plan_storage(tasks)
                    return f"Updated dependencies for {task_id}"
            return f"Task not found: {task_id}"

        if cmd == "move" and len(parts) >= 4:
            task_id = parts[1]
            position = parts[2].lower()
            ref_id = parts[3]
            task = next((t for t in tasks if t.get("id") == task_id), None)
            ref = next((t for t in tasks if t.get("id") == ref_id), None)
            if not task or not ref:
                return "Task or reference not found"
            tasks.remove(task)
            ref_index = tasks.index(ref)
            insert_index = ref_index if position == "before" else ref_index + 1
            tasks.insert(insert_index, task)
            self._ensure_sequential_dependencies(tasks)
            self._update_plan_storage(tasks)
            return f"Moved {task_id} {position} {ref_id}"

        if cmd == "insert" and len(parts) >= 5:
            after_id = parts[1]
            new_id = parts[2]
            agent_id = parts[3]
            desc = " ".join(parts[4:]).strip()
            new_task = {
                "id": new_id,
                "description": desc,
                "assigned_agent": agent_id,
                "tools": [],
                "dependencies": []
            }
            if after_id == "start":
                tasks.insert(0, new_task)
            else:
                ref = next((t for t in tasks if t.get("id") == after_id), None)
                if not ref:
                    return f"Reference task not found: {after_id}"
                idx = tasks.index(ref)
                tasks.insert(idx + 1, new_task)
            self._ensure_sequential_dependencies(tasks)
            self._update_plan_storage(tasks)
            return f"Inserted {new_id} after {after_id}"

        if cmd == "remove" and len(parts) >= 2:
            task_id = parts[1]
            task = next((t for t in tasks if t.get("id") == task_id), None)
            if not task:
                return f"Task not found: {task_id}"
            tasks.remove(task)
            self._ensure_sequential_dependencies(tasks)
            self._update_plan_storage(tasks)
            return f"Removed {task_id}"

        if cmd == "swap" and len(parts) >= 3:
            t1 = next((t for t in tasks if t.get("id") == parts[1]), None)
            t2 = next((t for t in tasks if t.get("id") == parts[2]), None)
            if not t1 or not t2:
                return "Task not found"
            i1, i2 = tasks.index(t1), tasks.index(t2)
            tasks[i1], tasks[i2] = tasks[i2], tasks[i1]
            self._ensure_sequential_dependencies(tasks)
            self._update_plan_storage(tasks)
            return f"Swapped {parts[1]} and {parts[2]}"

        return "Unknown command"
    
    def _setup_registry(self):
        """Register default execution agents"""
        try:
            from ai_autonom.core.agent_registry import setup_initial_registry
            setup_initial_registry()
        except Exception:
            pass
        config = get_config()
        coder_model = config.get('agents.coder.model', 'qwen2.5-coder:7b')
        linguistic_model = config.get('agents.linguistic.model', 'dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0')

        # CAI Retester Agent
        self.registry.register_agent(AgentDefinition(
            id="retester_agent",
            name="Retester Agent",
            model_name=coder_model,
            model_size_gb=0.0,
            capabilities=["vulnerability_verification", "triage", "false_positive_elimination"],
            tools=["cai_generic_linux_command", "python_exec", "cai_google_search"],
            vram_required=0.0,
            speed_tokens_per_sec=50.0,
            quality_score=95.0,
            description="Specialist in validating vulnerabilities and eliminating false positives",
            provider="ollama",
            system_prompt="""# Cybersecurity Triage Agent

You are an autonomous cybersecurity agent specializing in vulnerability 
verification and false positive elimination. Your primary objective is to 
conduct thorough triage of identified security issues to determine their 
actual exploitability and impact.

## Core Objectives

1. **Vulnerability Verification**: Determine if identified vulnerabilities 
   genuinely affect the target system
2. **Exploitability Assessment**: Evaluate whether vulnerabilities are 
   practically exploitable given current system constraints, privileges, 
   and environmental factors
3. **False Positive Elimination**: Filter out non-exploitable findings to 
   focus on genuine security risks

## Triage Methodology

When vulnerabilities are present in your context, process each one 
systematically using this structured approach:

### Phase 1: Initial Assessment
- Review the vulnerability detection data and scan results
- Analyze the specific system context and configuration
- Identify the vulnerability type, severity, and potential impact
- Document current privilege level and system access constraints

### Phase 2: Intelligence Gathering
If vulnerability information is insufficient, conduct comprehensive research:
- Search public vulnerability databases and resources
- Query NIST database via curl commands for CVE details
- Use searchsploit for exploit availability and techniques
- Leverage Google searches for additional context and proof-of-concepts
- Access any available structured databases through curl or Linux commands
- Gather vendor advisories and security bulletins

### Phase 3: Exploitation Validation
- Develop and execute targeted proof-of-concept exploits
- Test vulnerability under current system conditions
- Verify that exploitation succeeds with available privileges
- Document all exploitation attempts and results
- Assess real-world exploitability factors

### Phase 4: Impact Analysis
- Determine the actual security impact if exploitation succeeds
- Evaluate potential for privilege escalation or lateral movement
- Assess data exposure or system compromise risks
- Consider business impact and operational consequences

## Success Criteria

A vulnerability is confirmed as genuine and exploitable when:
- Proof-of-concept exploitation demonstrates actual system compromise
- The exploit works within current privilege and access constraints
- The vulnerability poses measurable security risk to the organization
- Exploitation can be reliably reproduced

## Output Standards

For each vulnerability assessed:
- **Status**: Confirmed Vulnerable / Not Exploitable / False Positive
- **Evidence**: Detailed exploitation steps and proof-of-concept
- **Impact**: Realistic assessment of potential damage
- **Constraints**: Factors limiting or enabling exploitation
- **Recommendations**: Specific remediation guidance
- **Artifacts**: Complete file directory path to exploit code, all 
  supporting files, scripts, and documentation required to reproduce 
  the vulnerability exploitation
## Guidelines

- Prioritize thorough verification over speed
- Always attempt practical exploitation before confirming vulnerabilities
- Consider environmental factors that may affect exploitability

Remember: The goal is to provide definitive answers about vulnerability 
exploitability, eliminating uncertainty and enabling informed security 
decision-making."""
        ))

        # Technical/Coding agent - auto-select best coding model if available
        coder_fallback = self.config.get('agents.coder.model', 'qwen2.5-coder:7b')
        coder_model = self._select_model_for_capability("code_generation", coder_fallback)
        coder_vram = self._estimate_model_vram(coder_model)
        self.registry.register_agent(AgentDefinition(
            id="coder_qwen",
            name="Qwen3 Technical Coder",
            model_name=coder_model,
            model_size_gb=coder_vram,
            capabilities=["code_generation", "debugging", "refactoring", "python", "testing", "technical_tasks"],
            tools=["filesystem_read", "filesystem_write", "python_exec", "bash_exec", "pytest_run"],
            vram_required=coder_vram,
            speed_tokens_per_sec=70.0,
            quality_score=85.0,
            description="Auto-selected coder model for technical tasks",
            provider="ollama"
        ))
        
        # Linguistic/Simple tasks agent - auto-select best documentation model if available
        linguistic_fallback = self.config.get('agents.linguistic.model', 'dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0')
        linguistic_model = self._select_model_for_capability("documentation", linguistic_fallback)
        linguistic_vram = self._estimate_model_vram(linguistic_model)
        self.registry.register_agent(AgentDefinition(
            id="linguistic_dictalm",
            name="DictaLM Linguistic Agent",
            model_name=linguistic_model,
            model_size_gb=linguistic_vram,
            capabilities=["text_generation", "documentation", "summarization", "formatting", "explanation", "simple_tasks"],
            tools=["filesystem_read", "filesystem_write"],
            vram_required=linguistic_vram,
            speed_tokens_per_sec=50.0,
            quality_score=80.0,
            description="Auto-selected linguistic model for documentation",
            provider="ollama"
        ))
        
        # Register OpenAI agent if provider is available
        if 'openai' in self.providers:
            self.registry.register_agent(AgentDefinition(
                id="gpt4_coder",
                name="GPT-4o Mini Coder",
                model_name="gpt-4o-mini",
                model_size_gb=0.0,  # Cloud-based
                capabilities=["code_generation", "debugging", "refactoring", "python", "testing", "technical_tasks", "complex_reasoning"],
                tools=["filesystem_read", "filesystem_write", "python_exec", "bash_exec", "pytest_run"],
                vram_required=0.0,
                speed_tokens_per_sec=100.0,
                quality_score=95.0,
                description="OpenAI GPT-4o-mini - high quality cloud-based coder",
                provider="openai"
            ))
            self.message_bus.register_agent("gpt4_coder")
        
        # Register agents with message bus
        self.message_bus.register_agent("orchestrator")
        self.message_bus.register_agent("coder_qwen")
        self.message_bus.register_agent("linguistic_dictalm")
    
    def analyze_intent(self, user_goal: str) -> Dict[str, Any]:
        """Analyze user intent before planning"""
        print(f"\n{'─'*70}")
        print("INTENT ANALYSIS")
        print(f"{'─'*70}")
        
        result = self.intent_analyzer.analyze(user_goal)
        
        print(f"Task Type: {result.task_type}")
        print(f"Complexity: {result.complexity}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Single File: {result.requirements.get('single_file', False)}")
        
        if result.needs_clarification and not self.checkpoint.auto_approve:
            clarification = self.intent_analyzer.clarify_with_user(result.ambiguities)
            if clarification:
                # Re-analyze with clarification
                enhanced_goal = f"{user_goal}\n\nClarification: {clarification}"
                result = self.intent_analyzer.analyze(enhanced_goal)
        
        return {
            "analysis": result,
            "enhancement": self.intent_analyzer.get_prompt_enhancement(result)
        }
    
    def decompose_and_assign(
        self,
        user_goal: str,
        intent_enhancement: str = "",
        pattern_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Nemotron decomposes task AND assigns agents dynamically
        Returns: List of subtasks with assigned agents and tools
        """
        print(f"\n{'='*70}")
        print("TASK DECOMPOSITION & ASSIGNMENT")
        print(f"{'='*70}\n")
        print(f"Goal: {user_goal}\n")
        
        # Get available agents from registry
        agents = self.registry.get_all_agents()
        agent_info = "\n".join([
            f"- {a.name} (ID: {a.id}): {', '.join(a.capabilities)} (VRAM: {a.vram_required}GB)"
            for a in agents
        ])
        
        # Get available tools
        tool_info = self.tool_executor.get_tool_descriptions()
        
        pattern_hint = ""
        if pattern_name:
            try:
                from ai_autonom.patterns.cai_patterns import PatternLibrary
                pattern = PatternLibrary.get_pattern(pattern_name)
                if pattern:
                    seq = pattern.agents or pattern.sequence or []
                    if seq:
                        pattern_hint = f"Suggested pattern: {pattern_name} with agent sequence: {', '.join(seq)}. Follow this sequence unless there's a strong reason to deviate."
                    else:
                        pattern_hint = f"Suggested pattern: {pattern_name}."
                else:
                    pattern_hint = f"Suggested pattern: {pattern_name}."
            except Exception:
                pattern_hint = f"Suggested pattern: {pattern_name}."

        prompt = f"""You are Nemotron, a meta-orchestrator. You ONLY plan and assign - you NEVER execute.

User Goal: "{user_goal}"

{f'Additional Context: {intent_enhancement}' if intent_enhancement else ''}
{f'Pattern Guidance: {pattern_hint}' if pattern_hint else ''}

Available Execution Agents:
{agent_info}

Available Tools:
{tool_info}

Your job:
1. Break the goal into 3-6 executable subtasks
2. For EACH subtask, assign:
   - The best agent for the task
   - The specific tools needed
3. Add dependencies between tasks if needed
4. If user requested a single file output, add a FINAL synthesis task

IMPORTANT: MERGE sequential "Write Code" and "Execute Code" steps into ONE single task.
- BAD: Task 1: Write file. Task 2: Execute file.
- GOOD: Task 1: Write AND Execute file.
- REASON: Persistent state between tasks is not guaranteed.

Return ONLY this JSON format:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "detailed task description",
      "required_capability": "code_generation",
      "assigned_agent": "coder_qwen",
      "tools": ["filesystem_write", "python_exec"],
      "dependencies": [],
      "reasoning": "why this agent?"
    }},
    ...
  ]
}}

Rules:
- Use "coder_qwen" for: code generation, debugging, testing, technical tasks
- Use "linguistic_dictalm" for: documentation, text, explanations
- Use "research_agent" for: research, discovery, external context gathering
- Use "synthesizer_agent" for: final synthesis across multiple task outputs
- Add dependencies: tasks that need previous task output
- For multi-file output that needs combining: add final synthesis task
- Use exact agent IDs and tool IDs"""
        
        start = time.time()
        
        try:
            # Use provider abstraction
            messages = [
                LLMMessage(role="system", content="You are a meta-orchestrator. You plan and assign but never execute. Always respond with valid JSON."),
                LLMMessage(role="user", content=prompt)
            ]
            response = self.orchestrator_provider.chat(messages, self.orchestrator_model, format="json")
            elapsed = time.time() - start
            
            content = response.content
            print(f"Planning time: {elapsed:.2f}s\n")
            
            data = json.loads(content)
            tasks = data.get('tasks', [])
            
            # DEBUG LOG
            log_debug("NEMOTRON", data, "PLAN")
            
            # Add testing tasks if enabled
            if self.enable_testing:
                tasks = self.testing_workflow.add_testing_tasks(tasks)
            
            # Add synthesis task if multiple coding tasks and single file requested
            tasks = self._add_synthesis_if_needed(tasks, user_goal)

            # Remove cai_* tools from non-security tasks
            self._sanitize_task_tools(tasks)

            # Apply pattern ordering/deps if available
            if pattern_name:
                tasks = self._apply_pattern_to_tasks(tasks, pattern_name)
            
            print(f"Generated {len(tasks)} tasks\n")
            
            # Display plan
            for i, task in enumerate(tasks, 1):
                print(f"  {i}. [{task.get('id')}]")
                print(f"     Agent: {task.get('assigned_agent')}")
                print(f"     Tools: {', '.join(task.get('tools', []))}")
                if task.get('dependencies'):
                    print(f"     Deps: {task.get('dependencies')}")
                print()
            
            if not tasks and pattern_name:
                print(f"[FALLBACK] No tasks generated, applying pattern: {pattern_name}")
                tasks = self._build_tasks_from_pattern(pattern_name, user_goal)

            return tasks
            
        except Exception as e:
            print(f"Decomposition failed: {e}")
            if pattern_name:
                print(f"[FALLBACK] Using pattern {pattern_name} due to decomposition failure")
                return self._build_tasks_from_pattern(pattern_name, user_goal)
            return []

    def _build_tasks_from_pattern(self, pattern_name: str, user_goal: str) -> List[Dict]:
        """Create a basic task list from a predefined pattern."""
        try:
            from ai_autonom.patterns.cai_patterns import PatternLibrary, PatternType
            pattern = PatternLibrary.get_pattern(pattern_name)
            if not pattern:
                return []

            tasks: List[Dict] = []
            agents = pattern.agents or pattern.sequence or []
            prev_task_id = None

            for idx, agent_id in enumerate(agents, 1):
                agent = next((a for a in self.registry.get_all_agents() if a.id == agent_id), None)
                tools = agent.tools if agent else []
                task_id = f"task_{idx}"
                task = {
                    "id": task_id,
                    "description": f"{pattern.description}\nUser goal: {user_goal}\nFocus: {agent_id}",
                    "assigned_agent": agent_id,
                    "tools": tools,
                    "dependencies": [] if not prev_task_id or pattern.pattern_type == PatternType.PARALLEL else [prev_task_id]
                }
                tasks.append(task)
                prev_task_id = task_id

            return tasks
        except Exception:
            return []

    def _apply_pattern_to_tasks(self, tasks: List[Dict], pattern_name: str) -> List[Dict]:
        """Reorder tasks to match pattern agent sequence and align dependencies."""
        try:
            from ai_autonom.patterns.cai_patterns import PatternLibrary, PatternType
            pattern = PatternLibrary.get_pattern(pattern_name)
            if not pattern or not tasks:
                return tasks

            seq = pattern.agents or pattern.sequence or []
            if not seq:
                return tasks

            ordered: List[Dict] = []
            remaining = tasks[:]

            for agent_id in seq:
                for t in list(remaining):
                    if t.get("assigned_agent") == agent_id:
                        ordered.append(t)
                        remaining.remove(t)

            ordered.extend(remaining)

            # Enforce dependencies for chain-like patterns
            if pattern.pattern_type in (PatternType.SEQUENTIAL, PatternType.CHAIN, PatternType.HIERARCHICAL):
                prev = None
                for t in ordered:
                    if prev and not t.get("dependencies"):
                        t["dependencies"] = [prev]
                    prev = t.get("id")
            elif pattern.pattern_type == PatternType.PARALLEL:
                for t in ordered:
                    t["dependencies"] = []

            return ordered
        except Exception:
            return tasks
    
    def _add_synthesis_if_needed(self, tasks: List[Dict], user_goal: str) -> List[Dict]:
        """Add final synthesis task if needed"""
        goal_lower = user_goal.lower()
        single_file = any(p in goal_lower for p in ["one file", "single file", "combined", "all in one"])
        
        coding_tasks = [t for t in tasks if t.get("type") not in ("synthesis", "testing", "documentation")]
        
        if len(coding_tasks) > 1 and single_file:
            synthesis_task = {
                "id": f"task_{len(tasks)+1}_synthesis",
                "description": f"""FINAL SYNTHESIS TASK:
Combine ALL outputs from previous coding tasks into ONE complete file.

Previous tasks: {[t['id'] for t in coding_tasks]}

Requirements:
1. Read all code from previous task outputs
2. Merge into single, complete, runnable file
3. Remove duplicate imports
4. Ensure no conflicts
5. Add proper main() if needed
6. Output ONLY the final combined code""",
                "assigned_agent": "coder_qwen",
                "tools": ["filesystem_read", "filesystem_write", "python_exec"],
                "dependencies": [t["id"] for t in tasks if t.get("type") != "testing"],
                "type": "synthesis"
            }
            tasks.append(synthesis_task)
            print("[SYNTHESIS] Added final synthesis task")
        
        return tasks
    
    def _evaluate_tool_request(self, agent_name: str, task_desc: str, request: str, current_tools: List[str]) -> Dict[str, Any]:
        """
        Orchestrator Logic: Evaluate if an agent should be granted a requested tool.
        """
        if self.ui:
            self.ui.log(f"Orchestrator evaluating tool request: {request}", "AGENT")
            
        # Get all available tools
        all_tools = self.tool_executor.get_available_tools()
        # Simplify tool list for prompt
        registry_summary = [f"{t['id']}: {t['description']}" for t in all_tools]
        
        prompt = f"""ACTION: Tool Access Request Evaluation

Context:
Agent '{agent_name}' is working on task: "{task_desc}"
Current Tools: {current_tools}

The Agent has requested: "{request}"

Your Job:
1. Identify if the requested tool exists in the registry.
2. Determine if granting this tool is safe and necessary for the task.
3. If DENIED, formulate a specific "wrapper solution" or alternative strategy using the CURRENT tools.

Available Tool Registry:
{json.dumps(registry_summary, indent=2)}

Return JSON ONLY:
{{
    "tool_id": "exact_tool_id_from_registry" or null,
    "approved": boolean,
    "reason": "explanation for decision",
    "usage_instructions": "if approved, brief usage example",
    "alternative_strategy": "if denied, detailed instructions on how to proceed without the tool"
}}"""

        try:
            messages = [
                LLMMessage(role="system", content="You are the Orchestrator Security Controller. You manage tool access."),
                LLMMessage(role="user", content=prompt)
            ]
            response = self.orchestrator_provider.chat(messages, self.orchestrator_model, format="json")
            return json.loads(response.content)
        except Exception as e:
            print(f"[ORCHESTRATOR] Tool evaluation failed: {e}")
            return {"approved": False, "reason": f"Evaluation error: {e}", "alternative_strategy": "Continue with current tools."}

    def execute_single_task(self, task: Dict, context: Dict = None) -> Dict:
        """Execute a single task with the assigned agent - ACTUALLY EXECUTE TOOLS!"""
        if self._cancelled():
            return {"success": False, "error": "Cancelled"}

        task_id = task.get('id', 'unknown')
        context = context or {}
        
        # Start monitoring
        self.monitor.log_task_start(task_id, task.get('assigned_agent', 'unknown'))
        
        # Create task context
        self.task_memory.create_task_context(
            task_id,
            task.get('description', ''),
            task.get('assigned_agent', ''),
            task.get('dependencies', [])
        )
        # Get dependency context
        dep_context = self.task_memory.get_dependency_context(task_id)
        ipc_dep_context = self._get_ipc_dependency_context(task.get('dependencies', []))

        # Record input context in DB for downstream agents
        self.task_memory.start_task(task_id, input_context={
            "dependencies": dep_context,
            "ipc_dependencies": ipc_dep_context,
            "explicit_context": context or {}
        })
        
        # Look up agent
        agents = self.registry.get_all_agents()
        agent_id = task.get('assigned_agent')
        agent = next((a for a in agents if a.id == agent_id), None)
        if not agent and agent_id:
            alias_map = {
                "web_pentester": "web_pentester_agent",
                "report_agent": "reporting_agent",
                "retester": "retester_agent",
                "red_team": "red_team_agent"
            }
            mapped = alias_map.get(agent_id)
            if mapped:
                agent = next((a for a in agents if a.id == mapped), None)
                if agent:
                    task["assigned_agent"] = mapped
        
        if not agent:
            error = f"Agent not found: {task.get('assigned_agent')}"
            self.task_memory.fail_task(task_id, error)
            self.monitor.log_task_complete(task_id, False, error=error)
            return {"success": False, "error": error}

        model_to_use = self._select_model_for_task(task, agent)
        if model_to_use != agent.model_name and self.ui:
            self.ui.log(
                f"Model override for {task_id}: {agent.model_name} -> {model_to_use}",
                "INFO",
            )
            
        if self.ui:
            self.ui.set_active_agent(agent.name)
            self.ui.update_task_status(task_id, "running")
            self.ui.log(f"Starting task: {task_id}", "INFO")
            self.ui.set_active_model(model_to_use)
            self.ui.trace_event(
                "task_start",
                {
                    "task_id": task_id,
                    "agent": agent.name,
                    "model": model_to_use,
                },
            )
        
        print(f"\n{'─'*70}")
        print(f"EXECUTING: {task_id}")
        print(f"Agent: {agent.name} ({model_to_use})")
        print(f"Tools: {', '.join(task.get('tools', []))}")
        print(f"{'─'*70}\n")
        
        # Build task description with context
        context_str = ""
        
        # 1. Dependency Context (Previous output)
        if dep_context.get("previous_outputs"):
            context_str += "\n\nContext from previous tasks:\n"
            for dep_id, dep_info in dep_context["previous_outputs"].items():
                context_str += f"\n--- {dep_id} ---\n{dep_info.get('output', '')[:1000]}\n"

        # 1b. IPC Dependency Context (DB shared outputs)
        if ipc_dep_context:
            context_str += "\n\nContext from DB (IPC shared outputs):\n"
            for dep_id, dep_info in ipc_dep_context.items():
                context_str += f"\n--- {dep_id} ---\n{dep_info.get('output', '')[:1000]}\n"

        # 1c. Explicit context passed from prior step
        if context:
            context_str += "\n\nContext from prior step:\n"
            for k, v in context.items():
                context_str += f"- {k}: {str(v)[:1000]}\n"
        
        # 2. Knowledge Base Context (The Blackboard)
        kb_summary = self.knowledge_base.get_summary()
        context_str += f"\n\n{kb_summary}"

        # 2b. Vector Memory Context (Long-term)
        try:
            memory_hint = self.vector_store.query_natural(task.get('description', ''))
            if memory_hint and "No relevant information" not in memory_hint:
                context_str += f"\n\n[VECTOR MEMORY]\n{memory_hint[:1500]}"
        except Exception:
            pass
        
        # 3. Session Context
        container_workspace = self.session_workspace or "outputs"

        context_str += f"""

[SESSION WORKSPACE]
You are operating in a lightweight sandboxed workspace.
- Workspace Root: {container_workspace}
- Use relative paths like "src/main.py" or "docs/final_report.md".
- Source code -> src/ | Binaries -> bin/ | Report -> docs/ | Notes -> memory/
- If a tool runs inside a container, the host workspace is under /outputs/.
- If you need a system binary (g++, nmap, etc.) and it is missing, follow the IMPOSSIBLE_TASK protocol.

IMPORTANT WORKFLOW:
1. **Plan**: Decide which tools you need (python_exec, bash_exec, filesystem_*).
2. **Create**: Write files under src/ (or bin/docs/memory as appropriate) using 'filesystem_write'.
3. **Execute**: Run them using 'bash_exec' or 'python_exec'.
"""
        
        task_description = task.get('description', '') + context_str
        
        # Initialize full log if not exists
        full_log_path = Path(self.session_manager.current_session_dir) / "full_orchestration_log.txt"
        if not full_log_path.exists():
            with open(full_log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== ORCHESTRATION LOG ===\nStarted: {datetime.now().isoformat()}\nGoal: {task.get('description', '')}\n\n")
            print(f"\n[LOGGING] Full orchestration log available at: {full_log_path}\n")

        try:
            task_brief = {
                "task_id": task_id,
                "assigned_agent": agent.id,
                "model": model_to_use,
                "tools": task.get("tools", []),
                "max_tool_iterations": max_iterations,
            }
            with open(full_log_path, "a", encoding="utf-8") as f:
                f.write(f"[TASK BRIEF]: {json.dumps(task_brief, ensure_ascii=True)}\n")
        except Exception:
            pass

        if agent.id in {"openmanus_coder", "openmanus_researcher"}:
            if self._cancelled():
                return self._cancel_result()
            if self.ui:
                self.ui.log(f"Running OpenManus ToolCall agent: {agent.name}", "INFO")
            try:
                next_step_prompt = None
                if agent.id == "openmanus_coder":
                    try:
                        from app.prompt.swe import NEXT_STEP_PROMPT
                        next_step_prompt = NEXT_STEP_PROMPT
                    except Exception:
                        pass
                elif agent.id == "openmanus_researcher":
                    try:
                        from app.prompt.manus import NEXT_STEP_PROMPT
                        next_step_prompt = NEXT_STEP_PROMPT
                    except Exception:
                        pass

                output = run_toolcall_task(
                    goal=task_description,
                    system_prompt=agent.system_prompt or f"You are {agent.name}.",
                    next_step_prompt=next_step_prompt,
                    max_steps=max(1, int(self.config.get("execution.max_tool_iterations", 3))),
                    cancel_event=self.cancel_event,
                )
                return {"success": True, "output": output}
            except Exception as e:
                return {"success": False, "error": f"OpenManus execution failed: {e}"}

        start = time.time()
        try:
            # === KEY CHANGE: TOOL-CALLING LOOP ===
            # Instead of just getting one response, we loop:
            # 1. Agent thinks -> wants to use tool
            # 2. Execute tool -> get result  
            # 3. Give result to agent -> agent continues
            # 4. Repeat until agent says "done"
            
            # Determine system prompt
            if hasattr(agent, 'system_prompt') and agent.system_prompt:
                base_prompt = agent.system_prompt
            else:
                base_prompt = f"You are {agent.name}."
            
            # Enhanced system message with capability checking and honest failure protocol
            capability_task_description = task.get('description', '')
            
            # Check capabilities BEFORE execution
            capability_warning = ""
            if CAPABILITY_CHECKER_AVAILABLE:
                checker = CapabilityChecker()
                cap_check = checker.validate(capability_task_description)
                if not cap_check.can_complete:
                    capability_warning = cap_check.to_prompt_message()
                    print(f"[CAPABILITY] Task may be impossible: {cap_check.explanation}")

            max_iterations = int(task.get("max_iterations") or self.config.get("execution.max_tool_iterations", 3))
            max_iterations = max(1, max_iterations)
            
            # Construct full system message with enhanced rules
            system_message = f"""{base_prompt}

  Available tools: {', '.join(task.get('tools', []))}
  Only use tools listed above. Do NOT use cai_* tools unless explicitly listed.
  You have at most {max_iterations} tool iterations for this task.
  Prefer a single response with multiple TOOL blocks if you need more than one tool.

You MUST use tools to complete this task. Don't just describe what to do - ACTUALLY DO IT!

*** GLOBAL MANDATE: AUTONOMY & SELF-CORRECTION ***
You are an autonomous agent operating in a real environment.
We expect you to encounter errors. This is normal.

WHEN A TOOL FAILS:
1. DO NOT STOP.
2. READ the error message carefully.
3. SELF-CORRECT: Modify your command parameters based on the error.
   - "Is a directory" -> You forgot the filename.
   - "Host unreachable" -> Try a different flag or check network.
   - "Permission denied" -> Verify you are in the session workspace.
4. RETRY immediately.

You have permission to "fail forward". Iterate until you succeed or exhaust all options.
Only ask for human help if you are completely blocked after 3 distinct attempts.

*** CRITICAL: HONEST FAILURE PROTOCOL ***

Before attempting ANY task, you MUST verify capability:

1. VERIFY TOOLS EXIST:
   - Run "which <tool>" for every tool you plan to use
   - If tool doesn't exist -> CANNOT proceed, must report IMPOSSIBLE_TASK

2. IMPOSSIBLE TASKS - These CANNOT be done without specific tools:
   - Creating .exe files: Requires mingw-w64 or PyInstaller (Python ALONE cannot do this!)
   - Compiling C/C++: Requires gcc/g++/clang (NOT Python)
   - Network scanning: Requires nmap (cannot be emulated with Python)
   - Password cracking: Requires john/hashcat
   - Memory forensics: Requires volatility

3. WHEN TASK IS IMPOSSIBLE:
   DO NOT:
   - Pretend to do it or simulate the output
   - Use Python to "emulate" missing tools
   - Say "I'll try anyway"
   - Hallucinate fake results
   
   DO:
   - Stop immediately
   - Output: IMPOSSIBLE_TASK: [clear explanation of why]  
   - Suggest: What tool is needed and how to install it
   - Then: ASK_USER: "Should I install the required tool or try an alternative?"

{capability_warning}

*** TOOL USAGE ***

To use a tool, you MUST provide your reasoning first.
REQUIRED FORMAT:

THOUGHT:
[Explain your reasoning: why you are choosing this tool, what you expect to happen]

TOOL: tool_name
target: value
args: value

Alternate JSON format (also accepted):

THOUGHT:
[Explain reasoning]

ACTION: tool_name
ACTION INPUT: {{"param": "value"}}

Chaining tool outputs:
- Use $LAST_OUTPUT or $LAST_OUTPUT_1K in tool params to inject the previous tool's output.

Or for Linux commands:
    
THOUGHT:
I need to check if the file exists before writing to it.

  TOOL: bash_exec
command: ls -la .

*** PRE-FLIGHT EXAMPLE ***

User asks: "Create a Windows .exe from my Python script"

CORRECT BEHAVIOR:
    TOOL: bash_exec
    command: which x86_64-w64-mingw32-gcc && which pyinstaller
    
    [Output: nothing found]
    
    IMPOSSIBLE_TASK: Cannot create Windows .exe files.
    - Python alone cannot produce .exe files
    - Required: mingw-w64 (apt install mingw-w64) OR PyInstaller (pip install pyinstaller)
    ASK_USER: "Neither mingw-w64 nor PyInstaller is installed. Should I install one of them?"

WRONG BEHAVIOR:
    "I'll write a Python script that creates an exe..." <- WRONG, this is impossible
    "Creating exe file..." <- WRONG, hallucinating

*** DYNAMIC REQUESTS ***

REQUEST_TOOL: <tool_name> - Request additional tool access
REQUEST_HANDOFF: <agent_name> - Hand off to specialist agent
ASK_USER: <question> - Ask user for decision or clarification

*** EXECUTION RULES ***

1. ALWAYS run "which <tool>" before using a tool
2. If tool doesn't exist -> IMPOSSIBLE_TASK, do not proceed
3. If a command fails, READ THE ERROR and fix parameters
4. Focus on the ACTUAL target provided by user
5. NEVER say COMPLETE unless task was actually accomplished
6. Cite memory when used: mention [VECTOR MEMORY] or [KB] in your THOUGHT
7. You have FULL FREEDOM to execute commands within the available sandbox tools"""

            conversation_history = [
                LLMMessage(
                    role="system",
                    content=system_message
                ),
                LLMMessage(role="user", content=task_description)
            ]
            
            provider = self.get_provider_for_agent(agent)
            full_response = ""
            iteration = 0
            completed = False
            
            while iteration < max_iterations:
                iteration += 1
                
                print(f"\n[Iteration {iteration}]")
                
                # Get agent response
                response = provider.chat(conversation_history, model_to_use)
                agent_msg = response.content

                full_response += f"\n\n=== Iteration {iteration} ===\n{agent_msg}"

                # Enforce ReAct schema: require THOUGHT and TOOL/ACTION blocks
                has_thought = "THOUGHT:" in agent_msg
                has_tool = "TOOL:" in agent_msg or "ACTION:" in agent_msg
                if "COMPLETE:" not in agent_msg and (not has_thought or not has_tool):
                    format_hint = """FORMAT ERROR: Use the ReAct schema every turn.
THOUGHT:
[your reasoning]

TOOL: tool_name
param: value

Or:
ACTION: tool_name
ACTION INPUT: {"param": "value"}

If done, reply with COMPLETE: <summary>."""
                    conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                    conversation_history.append(LLMMessage(role="user", content=format_hint))
                    print("[FORMAT] Missing THOUGHT/TOOL - prompting agent to follow ReAct\n")
                    continue
                
                # === EXTRACT THINKING ===
                # "Thinking" is generally everything before the first tool call
                thinking_content = agent_msg
                if "THOUGHT:" in agent_msg:
                    # Extract content between THOUGHT: and TOOL:
                    parts = agent_msg.split("THOUGHT:", 1)[1]
                    if "TOOL:" in parts:
                        thinking_content = parts.split("TOOL:")[0].strip()
                    elif "ACTION:" in parts:
                        thinking_content = parts.split("ACTION:")[0].strip()
                    elif "COMPLETE:" in parts:
                        thinking_content = parts.split("COMPLETE:")[0].strip()
                    else:
                        thinking_content = parts.strip()
                elif "TOOL:" in agent_msg:
                    thinking_content = agent_msg.split("TOOL:")[0].strip()
                elif "ACTION:" in agent_msg:
                    thinking_content = agent_msg.split("ACTION:")[0].strip()
                elif "COMPLETE:" in agent_msg:
                    thinking_content = agent_msg.split("COMPLETE:")[0].strip()
                
                # Check if it was empty but perhaps malformed (Fallback for blind spots)
                thinking_content = thinking_content.strip()
                if not thinking_content and len(agent_msg) > 10 and ("TOOL:" in agent_msg or "ACTION:" in agent_msg):
                    # If we found nothing but there IS a TOOL call, maybe standard text before it is the thought
                    marker = "TOOL:" if "TOOL:" in agent_msg else "ACTION:"
                    thinking_content = agent_msg.split(marker)[0].strip()

                if thinking_content:
                    print(f"\n[AGENT THOUGHT] {agent.name}:\n{thinking_content.strip()}\n")
                    
                    # Append to full log
                    with open(full_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n--- Iteration {iteration} ---\n")
                        f.write(f"[{agent.name} THINKING]:\n{thinking_content.strip()}\n")

                    # Capture decisions/assumptions for memory
                    for raw_line in thinking_content.splitlines():
                        line = raw_line.strip()
                        lower = line.lower()
                        if lower.startswith("decision:") or lower.startswith("assumption:") or lower.startswith("constraint:"):
                            decision_text = line.split(":", 1)[1].strip()
                            if decision_text:
                                try:
                                    self.task_memory.add_decision(
                                        task_id,
                                        decision_text,
                                        "from agent thought",
                                        agent.id
                                    )
                                except Exception:
                                    pass
                                try:
                                    self.vector_store.store_decision(
                                        task_id=task_id,
                                        decision=decision_text,
                                        rationale="from agent thought"
                                    )
                                except Exception:
                                    pass
                else:
                    print(f"\n[AGENT THOUGHT] {agent.name}: (No thought provided)\n")

                # DEBUG LOG
                log_debug(f"AGENT_{agent.name}", agent_msg, "THOUGHT")
                
                if self.ui:
                    self.ui.log(f"{agent.name} thinking...", "AGENT")
                
                # Check if agent wants to use a tool or request one
                if "REQUEST_TOOL:" in agent_msg:
                    import re
                    req_match = re.search(r'REQUEST_TOOL:\s*(.+)', agent_msg)
                    reason_match = re.search(r'reason:\s*(.+)', agent_msg, re.IGNORECASE)
                    
                    requested_tool = req_match.group(1).strip() if req_match else "unknown"
                    
                    if self.ui:
                        self.ui.log(f"Agent requesting tool: {requested_tool}", "TOOL")
                    
                    print(f"\n[ORCHESTRATOR] Agent requesting tool: {requested_tool}")
                    
                    # Evaluate request
                    decision = self._evaluate_tool_request(
                        agent.name, 
                        task.get('description', ''), 
                        requested_tool, 
                        task.get('tools', [])
                    )
                    
                    if decision.get("approved") and decision.get("tool_id"):
                        tool_id = decision["tool_id"]
                        # Grant access
                        if tool_id not in task['tools']:
                            task['tools'].append(tool_id)
                            
                            # Get tool details for the agent
                            tool_def = next((t for t in self.tool_executor.get_available_tools() if t['id'] == tool_id), None)
                            params = tool_def.get('parameters', {}) if tool_def else {}
                            
                            feedback = f"""✅ REQUEST APPROVED.
                            
Tool '{tool_id}' has been added to your kit.
Usage Instructions: {decision.get('usage_instructions', '')}
Parameters: {json.dumps(params)}

You may now use this tool in your next turn."""
                        else:
                            feedback = f"Info: You already have access to '{tool_id}'."
                            
                    else:
                        # Deny access
                        feedback = f"""🚫 REQUEST DENIED.
                        
Reason: {decision.get('reason')}

ALTERNATIVE STRATEGY (Wrapper Solution):
{decision.get('alternative_strategy')}

Please proceed using your currently available tools."""
                    
                    conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                    conversation_history.append(LLMMessage(role="user", content=feedback))
                    print(f"[ORCHESTRATOR] Decision: {'Approved' if decision.get('approved') else 'Denied'}")
                    
                    with open(full_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[TOOL REQUEST]: {requested_tool} -> {decision.get('approved')}\n")

                elif "REQUEST_HANDOFF:" in agent_msg:
                    import re
                    handoff_match = re.search(r'REQUEST_HANDOFF:\s*(.+)', agent_msg)
                    context_match = re.search(r'context:\s*(.+)', agent_msg, re.IGNORECASE)
                    
                    target_agent = handoff_match.group(1).strip() if handoff_match else "unknown"
                    handoff_context = context_match.group(1).strip() if context_match else ""
                    
                    print(f"\n[ORCHESTRATOR] Agent requesting handoff to: {target_agent}")
                    
                    # Evaluate handoff
                    handoff_dependency_context = {
                        "previous_outputs": dep_context.get("previous_outputs", {}),
                        "code_artifacts": dep_context.get("code_artifacts", []),
                        "decisions": dep_context.get("decisions", []),
                        "explicit_context": context or {}
                    }
                    handoff_decision = self.handoff_manager.evaluate_handoff_request(
                        agent.name,
                        target_agent,
                        handoff_context,
                        dependency_context=handoff_dependency_context
                    )
                    
                    if handoff_decision.get("approved"):
                        target_id = handoff_decision.get("target_agent_id")
                        print(f"[HANDOFF] Transferring control to {target_id}...")
                        
                        # Create a sub-task for the new agent
                        subtask = {
                            "id": f"{task_id}_sub_{target_id}",
                            "description": handoff_decision.get("filtered_context"),
                            "assigned_agent": target_id,
                            "tools": self.tool_executor.get_tools_for_agent(["reconnaissance", "web_hacking", "reporting"]) # Give them their standard kit
                        }
                        
                        # Execute subtask RECURSIVELY
                        subtask_result = self.execute_single_task(subtask)
                        
                        # Feed result back to original agent
                        feedback = f"""✅ HANDOFF COMPLETE.
                        
Agent '{target_agent}' has completed their task.
Result:
{subtask_result.get('output', 'No output received.')}

You may now continue your task using these findings."""
                        
                    else:
                        feedback = f"""🚫 HANDOFF DENIED.
                        
Reason: {handoff_decision.get('reason')}
Please continue the task yourself."""
                        
                    conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                    conversation_history.append(LLMMessage(role="user", content=feedback))
                    
                    with open(full_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[HANDOFF REQUEST]: {target_agent} -> {handoff_decision.get('approved')}\n")

                elif "ASK_USER:" in agent_msg:
                    import re
                    question_match = re.search(r'ASK_USER:\s*(.+)', agent_msg, re.DOTALL)
                    question = question_match.group(1).strip() if question_match else "User attention required."
                    
                    if self.ui:
                        self.ui.log(f"Agent asking: {question[:50]}...", "AGENT")
                    
                    # Pause and get input
                    # We need a way to get input from the TUI/REPL context.
                    # For now, we'll use standard input if TUI isn't blocking, 
                    # OR if we are in REPL, we need a callback.
                    
                    print(f"\n\n[AGENT ASK] {agent.name}: {question}")
                    print(">> Type your answer and press Enter:")
                    
                    # If TUI is active, this might break layout unless we handle it.
                    # Ideally, the TUI should have an input box.
                    # Fallback to simple input for MVP correctness.
                    user_response = input("> ")
                    
                    feedback = f"USER RESPONSE: {user_response}"
                    conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                    conversation_history.append(LLMMessage(role="user", content=feedback))
                    
                    if self.ui:
                        self.ui.log("User responded.", "INFO")
                    
                    with open(full_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[USER ASK]: {question}\n[RESPONSE]: {user_response}\n")


                elif "TOOL:" in agent_msg or "ACTION:" in agent_msg:
                    # Parse tool call - supports MULTIPLE SEQUENTIAL TOOLS
                    tools_getting_executed = []
                    if "ACTION:" in agent_msg and "ACTION INPUT:" in agent_msg:
                        import re

                        action_match = re.search(r"ACTION:\s*(.+)", agent_msg)
                        raw_input = agent_msg.split("ACTION INPUT:", 1)[1]
                        # Trim trailing content after a new tool/action block
                        for marker in ["\nTOOL:", "\nACTION:"]:
                            if marker in raw_input:
                                raw_input = raw_input.split(marker, 1)[0]
                        raw_input = raw_input.strip()
                        if raw_input.startswith("```"):
                            raw_input = raw_input.split("\n", 1)[1] if "\n" in raw_input else ""
                            if "```" in raw_input:
                                raw_input = raw_input.rsplit("```", 1)[0]
                            raw_input = raw_input.strip()

                        try:
                            action_name = action_match.group(1).strip() if action_match else ""
                            params = json.loads(raw_input) if raw_input else {}
                            if action_name:
                                tools_getting_executed.append({"name": action_name, "params": params})
                        except Exception as e:
                            error_feedback = f"""❌ ACTION FAILED!

OBSERVATION:
Invalid ACTION INPUT JSON: {e}

INSTRUCTIONS:
- Provide valid JSON after ACTION INPUT.
- Example: ACTION INPUT: {{"query": "example"}}
"""
                            conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                            conversation_history.append(LLMMessage(role="user", content=error_feedback))
                            print("[FORMAT] Invalid ACTION INPUT JSON - prompting retry\n")
                            continue

                    elif "TOOL:" in agent_msg:
                        import re
                        
                        # Split message into lines to robustly parse multiple tools
                        lines = agent_msg.split('\n')
                        tools_getting_executed = []
                        
                        current_tool = None
                        current_params = {}
                        last_param_key = None
                        
                        for line in lines:
                            stripped_line = line.strip()
                            # Do NOT strip empty lines if capturing multiline content
                            
                            if stripped_line.startswith("TOOL:"):
                                # If we were parsing a tool, save it
                                if current_tool:
                                    tools_getting_executed.append({"name": current_tool, "params": current_params})
                                
                                # Start new tool
                                parts = stripped_line.split(":", 1)
                                if len(parts) > 1:
                                    current_tool = parts[1].strip()
                                    current_params = {}
                                    last_param_key = None
                            
                            elif current_tool:
                                # Check if this line is a new parameter key
                                # Heuristic: keys are usually single words or snake_case, no spaces, followed by colon
                                is_new_param = False
                                
                                if ":" in stripped_line:
                                    potential_key, potential_val = stripped_line.split(":", 1)
                                    potential_key = potential_key.strip()
                                    
                                    # Valid param key validation
                                    # 1. No spaces in key (e.g. "for i in range" -> has spaces, NOT a key)
                                    # 2. Not a python keyword (basic heuristic)
                                    # 3. Not a path or url (http://...)
                                    if (len(potential_key.split()) == 1 and 
                                        potential_key not in ["def", "class", "if", "for", "while", "http", "https"] and
                                        not potential_key.startswith("/")):
                                        
                                        val = potential_val.strip()
                                        # Remove YAML block scalars if they are the only content
                                        if val in ["|", ">"]:
                                            val = ""
                                        current_params[potential_key] = val
                                        last_param_key = potential_key
                                        is_new_param = True
                                
                                if not is_new_param:
                                    if last_param_key:
                                        # It's continuation of the previous parameter value (e.g. valid python code with colons)
                                        # Use original 'line' to preserve indentation
                                        current_params[last_param_key] += "\n" + line
                                    else:
                                        # Content before any param? Ignore or log.
                                        pass
                        
                        # Append the last tool
                        if current_tool:
                            tools_getting_executed.append({"name": current_tool, "params": current_params})
                        
                        if not tools_getting_executed:
                             print("[ERROR] Malformed tool call")
                    else:
                        error_feedback = """❌ ACTION FAILED!

OBSERVATION:
ACTION block missing ACTION INPUT JSON.

INSTRUCTIONS:
- Use ACTION INPUT with valid JSON.
- Example: ACTION INPUT: {"query": "example"}
"""
                        conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                        conversation_history.append(LLMMessage(role="user", content=error_feedback))
                        print("[FORMAT] ACTION missing ACTION INPUT - prompting retry\n")
                        continue

                    if not tools_getting_executed:
                        error_feedback = """❌ ACTION FAILED!

OBSERVATION:
No tool parsed from the response.

INSTRUCTIONS:
- Provide TOOL or ACTION with valid parameters.
"""
                        conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                        conversation_history.append(LLMMessage(role="user", content=error_feedback))
                        print("[FORMAT] Empty tool list - prompting retry\n")
                        continue

                    # Execute sequential tools
                    execution_log = ""
                    overall_success = True
                    last_tool_output = ""

                    def _apply_output_placeholders(value, last_output):
                        if not last_output:
                            return value
                        if isinstance(value, str):
                            return value.replace("$LAST_OUTPUT", last_output[:4000]).replace("$LAST_OUTPUT_1K", last_output[:1000])
                        if isinstance(value, list):
                            return [_apply_output_placeholders(v, last_output) for v in value]
                        if isinstance(value, dict):
                            return {k: _apply_output_placeholders(v, last_output) for k, v in value.items()}
                        return value

                    for i, tool_call in enumerate(tools_getting_executed):
                        tool_name = tool_call["name"]
                        params = _apply_output_placeholders(tool_call["params"], last_tool_output)
                        
                        print(f"[TOOL {i+1}/{len(tools_getting_executed)}] Executing: {tool_name}")
                        print(f"[PARAMS] {params}\n")
                        
                        if self.ui:
                            self.ui.set_active_tool(f"{tool_name}")
                            self.ui.log(f"Executing {tool_name} ({i+1})", "TOOL")
                            self.ui.trace_event(
                                "tool_start",
                                {
                                    "task_id": task_id,
                                    "tool": tool_name,
                                    "index": i + 1,
                                    "total": len(tools_getting_executed),
                                },
                            )
                        
                        # === EXECUTE ===
                        tool_started = time.time()
                        success, tool_output = self.tool_executor.execute(
                            tool_name,
                            params,
                            agent_id=agent.id,
                            task_id=task_id
                        )
                        last_tool_output = tool_output
                        tool_elapsed = time.time() - tool_started
                        
                        execution_log += f"\n=== TOOL {i+1}: {tool_name} ===\nStatus: {'SUCCESS' if success else 'FAILED'}\nOutput:\n{tool_output}\n"
                        
                        # Log individual tool result
                        with open(full_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"[TOOL EXECUTION]: {tool_name}\nParams: {params}\nResult: {tool_output}\n")
                        
                        # Log to Debug
                        log_debug(f"TOOL_{tool_name}", {"params": params, "output": tool_output, "success": success}, "EXECUTION")

                        if self.ui:
                            self.ui.trace_event(
                                "tool_end",
                                {
                                    "task_id": task_id,
                                    "tool": tool_name,
                                    "index": i + 1,
                                    "total": len(tools_getting_executed),
                                    "status": "success" if success else "failed",
                                    "duration_sec": round(tool_elapsed, 2),
                                },
                            )

                        # Persist code artifacts to vector memory
                        if success:
                            try:
                                code_content = None
                                path_hint = None
                                if tool_name in ["filesystem_write", "filesystem_append", "cai_filesystem_write"]:
                                    code_content = params.get("content") or params.get("args")
                                    path_hint = params.get("path") or params.get("file") or params.get("target")
                                elif tool_name in ["python_exec", "cai_execute_code", "execute_code"]:
                                    code_content = params.get("code") or params.get("content")
                                    path_hint = params.get("filename") or params.get("target")

                                if code_content:
                                    self.vector_store.store_code(
                                        task_id=task_id,
                                        code=code_content,
                                        metadata={"path": path_hint or "", "tool": tool_name}
                                    )
                            except Exception:
                                pass

                        if not success:
                            overall_success = False
                            # STP CHAIN ON FAILURE
                            execution_log += "\n[CHAIN ABORTED DUE TO FAILURE]\n"
                            break
                        
                    # === CONSOLIDATED FEEDBACK ===
                    # We must provide exactly ONE robust feedback message to the agent.
                    
                    if not overall_success:
                         # Smart Error Analysis
                         hint = "Check your parameters."
                         if "No such file" in execution_log:
                             hint = f"path does not exist. Did you create the file under the session workspace ({container_workspace})?"
                         if "Is a directory" in execution_log:
                             hint = "you targeted a directory but the tool expects a file path."
                         if "Permission denied" in execution_log:
                             hint = "you do not have permission. Try a different path."
                         if "command not found" in execution_log.lower():
                             hint = "Tool dependency missing. Use `execute_command` with 'apt-get install' or verify the binary name."
                             
                         error_feedback = f"""❌ ACTION FAILED!

OBSERVATION:
{execution_log}

DEBUGGING CONTEXT:
1. The tool execution failed.
2. The specific error output is provided above.
3. HINT: {hint}

INSTRUCTIONS FOR RECOVERY (Attempt {iteration}/{max_iterations}):
1. Analyze the error message in the output above.
2. Adjust your parameters (checks paths, filenames, content).
3. Try the tool again with corrected values.

DO NOT say COMPLETE until you have fixed this error and successfully executed the task."""
                         
                         conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                         conversation_history.append(LLMMessage( role="user", content=error_feedback ))
                         print(f"[WARNING] Tool chain failed - forcing retry\n")
                         
                    else:
                         success_feedback = f"""✅ ACTION SUCCESSFUL!

OBSERVATION:
{execution_log}

CONTEXTUAL AWARENESS:
1. Tools executed successfully.
2. If you wrote a code file, the next logical step is usually to EXECUTE it using run_script or execute_command.
3. If you ran a script, check the output above. Does it satisfy the user's request?

DECISION:
- If the task is fully complete, respond with "COMPLETE: [summary]".
- If more steps are needed (like running the script you just wrote), continue immediately with the next tool."""
                         
                         conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                         conversation_history.append(LLMMessage( role="user", content=success_feedback ))

                    if self.ui:
                        self.ui.set_active_tool(None)
                        status = "Success" if overall_success else "Failed"
                        level = "SUCCESS" if overall_success else "ERROR"
                        self.ui.log(f"Tool Chain: {status}", level)
                    
                    print(f"[RESULT CHAIN] {execution_log[:200]}...\n")
                
                elif "COMPLETE:" in agent_msg:
                    # Agent wants to finish - check if they actually did anything useful
                    
                    # Check conversation history for successful tool executions
                    # We look for "ACTION SUCCESSFUL" in user messages which indicates a successful previous turn
                    history_success = any("ACTION SUCCESSFUL" in msg.content for msg in conversation_history)
                    
                    # Or check current response
                    current_success = "SUCCESS" in full_response or "passed" in full_response
                    
                    if not history_success and not current_success:
                        # Agent trying to quit without doing any work!
                        rejection = f"""🚫 REJECTED: You cannot say COMPLETE yet!

You have not successfully executed ANY tools yet. You must:
1. Actually USE the tools available to you
2. Successfully execute at least one tool
3. Gather real data about the target
4. THEN provide a summary

Do NOT give up. Use the tools and try again!"""
                        
                        conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                        conversation_history.append(LLMMessage(
                            role="user",
                            content=rejection
                        ))
                        
                        print("[WARNING] Agent tried to quit early - forcing retry\n")
                    else:
                        # Agent actually did work - allow completion
                        print("[COMPLETE] Agent finished task\n")
                        completion_msg = agent_msg.split("COMPLETE:")[1].strip() if "COMPLETE:" in agent_msg else "Done."
                        with open(full_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"[COMPLETED]: {completion_msg}\n")
                        completed = True
                        break
                
                else:
                    # Agent didn't use proper format - prompt them
                    conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                    conversation_history.append(LLMMessage(
                        role="user",
                        content="Please use tools! Format:\nTOOL: tool_name\nparameter: value\n\nOr:\nACTION: tool_name\nACTION INPUT: {\"param\": \"value\"}\n\nOr say COMPLETE: [summary] if done."
                    ))
            
            elapsed = time.time() - start
            
            # Save results
            output_file = f".runtime/outputs/{task_id}_output.txt"
            Path("outputs").mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Task: {task_id}\n")
                f.write(f"Agent: {agent.name}\n")
                f.write(f"Time: {elapsed:.2f}s\n")
                f.write(f"Iterations: {iteration}\n")
                f.write("="*70 + "\n\n")
                f.write(full_response)

            if not completed:
                error = f"Max tool iterations ({max_iterations}) exceeded without completion."
                self.task_memory.fail_task(task_id, error)
                self.monitor.log_task_complete(task_id, False, error=error)
                if self.ui:
                    self.ui.update_task_status(task_id, "failed")
                    self.ui.log(f"Task {task_id} failed: {error}", "ERROR")
                    self.ui.trace_event(
                        "task_failed",
                        {"task_id": task_id, "reason": error},
                    )
                return {
                    "success": False,
                    "error": error,
                    "output": full_response,
                    "output_file": output_file,
                    "execution_time": elapsed,
                    "agent": agent.name
                }
            
            # Update memory
            self.task_memory.complete_task(task_id, full_response, [])
            self.monitor.log_task_complete(task_id, True, tokens=len(full_response.split()))

            # Store output in vector memory for long-term recall
            try:
                self.vector_store.store_conversation(
                    task_id=task_id,
                    agent_id=agent.id,
                    content=full_response,
                    role="assistant"
                )
            except Exception:
                pass

            # Publish output to IPC DB for downstream agents
            self._publish_task_output(task_id, full_response, agent.name)
            
            # === VRAM OPTIMIZATION: Unload Agent ===
            # After task completion, unload the agent's model to free VRAM for the next agent
            if self.config.get('execution.vram_optimized', True):
                if hasattr(provider, 'unload_model'):
                    if self.ui:
                        self.ui.log(f"Unloading Agent Model ({model_to_use})...", "INFO")
                    provider.unload_model(model_to_use)
            
            if self.ui:
                self.ui.update_task_status(task_id, "completed")
                self.ui.log(f"Task {task_id} complete", "SUCCESS")
                self.ui.trace_event(
                    "task_complete",
                    {
                        "task_id": task_id,
                        "agent": agent.name,
                        "model": model_to_use,
                        "duration_sec": round(elapsed, 2),
                    },
                )
            
            print(f"Execution time: {elapsed:.2f}s")
            print(f"Saved to: {output_file}")
            
            return {
                "success": True,
                "output": full_response,
                "output_file": output_file,
                "execution_time": elapsed,
                "agent": agent.name
            }
            
        except Exception as e:
            error = str(e)
            self.task_memory.fail_task(task_id, error)
            self.monitor.log_task_complete(task_id, False, error=error)
            
            print(f"Error: {error}")
            if self.ui:
                self.ui.trace_event(
                    "task_failed",
                    {"task_id": task_id, "reason": error},
                )
            
            return {"success": False, "error": error}
    
    def run(self, user_goal: str) -> Dict[str, Any]:
        """Complete orchestration workflow"""
        if self.ui:
            self.ui.start(user_goal)
            self.ui.log("Orchestrator started", "INFO")
        
        print(f"\n{'#'*70}")
        print("NEMOTRON ORCHESTRATOR")
        print(f"{'#'*70}\n")
        print(f"Goal: {user_goal}\n")
        
        total_start = time.time()
        
        # Initialize Session (Create Folders)
        if not self.session_manager.current_session_dir:
            session_path = self.start_session(user_goal)
        else:
            session_path = self.session_manager.current_session_dir
            self._configure_session_workspace(session_path)
            if self.ui:
                try:
                    self.ui.set_session(
                        self.session_manager.current_session_id,
                        self.session_manager.current_session_dir,
                    )
                except Exception:
                    pass
        print(f"[SESSION] Active Workspace: {session_path}")
        
        # Start workflow
        self.current_workflow_id = self.task_memory.start_workflow()

        if self.ui:
            self.ui.log("Analyzing Intent...", "INFO")

        if self._cancelled():
            return self._cancel_result()
            
        # Phase 1: Intent Analysis
        intent_result = self.analyze_intent(user_goal)
        enhancement = intent_result.get("enhancement", "")
        analysis = intent_result.get("analysis")
        pattern_name = analysis.suggested_pattern if analysis else None
        if self.ui:
            try:
                self.ui.trace_event(
                    "intent_analysis",
                    {
                        "task_type": getattr(analysis.task_type, "value", None) if analysis else None,
                        "complexity": getattr(analysis.complexity, "value", None) if analysis else None,
                        "confidence": getattr(analysis, "confidence", None) if analysis else None,
                    },
                )
            except Exception:
                pass

        try:
            analysis_payload = {
                "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else {},
                "enhancement": enhancement,
                "created_at": datetime.now().isoformat(),
            }
            self._write_memory_json("intent_analysis.json", analysis_payload, versioned=True)
            if self.ui:
                self.ui.set_intent_analysis(analysis_payload)
        except Exception:
            pass
        
        if self.ui:
            self.ui.log("Decomposing Task...", "INFO")

        if self._cancelled():
            return self._cancel_result()
            
        # Phase 2: Decomposition
        tasks = self.decompose_and_assign(user_goal, enhancement, pattern_name=pattern_name)

        # Ensure outputs flow to the next task by default
        self._ensure_sequential_dependencies(tasks)
        if self.ui:
            try:
                self.ui.trace_event(
                    "plan_ready",
                    {"task_count": len(tasks), "pattern": pattern_name},
                )
            except Exception:
                pass

        try:
            briefs_payload = {
                "created_at": datetime.now().isoformat(),
                "tasks": self._build_task_briefs(tasks),
            }
            self._write_memory_json("task_briefs.json", briefs_payload, versioned=True)
            if self.ui:
                self.ui.set_task_briefs(briefs_payload)
        except Exception:
            pass

        # Store current plan and persist to DB/IPC
        self.current_plan = tasks
        self._update_plan_storage(tasks)
        
        # === SAVE NEMOTRON JSON PLAN ===
        try:
            plan_path = Path(self.session_manager.current_session_dir) / "nemotron_plan.json"
            with open(plan_path, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2)
            print(f"[ORCHESTRATOR] Saved Nemotron plan to: {plan_path}")
        except Exception as e:
            print(f"[ORCHESTRATOR] Failed to save plan JSON: {e}")
        
        # === VRAM OPTIMIZATION: Unload Orchestrator ===
        # Unload Nemotron to free VRAM for the agents
        if self.orchestrator_provider_type == 'ollama' and self.config.get('execution.vram_optimized', True):
            if hasattr(self.orchestrator_provider, 'unload_model'):
                if self.ui:
                    self.ui.log(f"Unloading Orchestrator ({self.orchestrator_model})...", "INFO")
                self.orchestrator_provider.unload_model(self.orchestrator_model)
        
        if self.ui:
            self.ui.set_plan(tasks)
            self.ui.log("Plan created, executing...", "INFO")

        if not tasks:
            return {"success": False, "error": "No tasks generated"}

        if self._cancelled():
            return self._cancel_result()
        
        # Phase 3: Create workflow
        wf_id = self.workflow_engine.create_workflow(tasks, self.current_workflow_id, pattern_name=pattern_name)
        
        if self.ui:
            self.ui.log("Executing Plan...", "INFO")
        
        # Phase 4: Execute tasks
        results = []
        prev_task_id = None
        prev_output = None
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*70}")
            print(f"TASK {i}/{len(tasks)}")
            print(f"{'='*70}")

            if self._cancelled():
                return self._cancel_result()

            context = {}
            if prev_task_id and prev_output:
                context = {
                    "previous_task_id": prev_task_id,
                    "previous_task_output": prev_output
                }

            attempt_context = context.copy()
            while True:
                if self._cancelled():
                    return self._cancel_result()
                result = self.execute_single_task(task, context=attempt_context)
                results.append(result)

                if result.get("success"):
                    prev_task_id = task.get("id")
                    prev_output = result.get("output")
                    break

                # Handle failure with recovery strategy
                if self.ui:
                    self.ui.update_task_status(task.get("id"), "failed")
                    self.ui.log(f"Task failed: {result.get('error')}", "ERROR")

                decision = self.error_recovery.handle_failure(task, result.get("error", ""))
                attempt_context = {
                    **context,
                    "previous_error": result.get("error", ""),
                    "recovery_reason": decision.reason
                }

                if decision.action == RecoveryAction.RETRY:
                    if self.ui:
                        self.ui.log(f"Retrying task {task.get('id')}: {decision.reason}", "WARNING")
                    continue
                if decision.action == RecoveryAction.SIMPLIFY_TASK:
                    hint = decision.params.get("simplification_hints")
                    if hint:
                        task["description"] = f"{task.get('description','')}\n\nSimplify: {hint}"
                    if self.ui:
                        self.ui.log(f"Simplifying task {task.get('id')} and retrying", "WARNING")
                    continue
                if decision.action == RecoveryAction.RETRY_WITH_DIFFERENT_MODEL:
                    # Fallback to a simpler agent if available
                    if task.get("assigned_agent") != "simple_qwen":
                        task["assigned_agent"] = "simple_qwen"
                    if self.ui:
                        self.ui.log(f"Retrying with fallback agent for {task.get('id')}", "WARNING")
                    continue

                if decision.action in (RecoveryAction.ESCALATE_TO_HUMAN, RecoveryAction.ABORT, RecoveryAction.SKIP):
                    if self.ui:
                        self.ui.log(f"Stopping task {task.get('id')}: {decision.reason}", "ERROR")
                    break

                break

            # Human checkpoint for critical tasks
            if self.checkpoint.should_checkpoint(task) and result.get("success"):
                checkpoint_result = self.checkpoint.request_approval(
                    task,
                    result.get("output", ""),
                    "task_complete"
                )
                if not checkpoint_result.get("approved"):
                    if checkpoint_result.get("abort"):
                        break
        
        total_elapsed = time.time() - total_start
        
        # Summary
        successful = sum(1 for r in results if r.get('success'))
        
        print(f"\n{'='*70}")
        print("EXECUTION SUMMARY")
        print(f"{'='*70}\n")
        print(f"Total tasks: {len(tasks)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(tasks) - successful}")
        print(f"Total time: {total_elapsed:.2f}s")
        
        # Agent usage
        agent_usage = Counter(r.get('agent', 'unknown') for r in results if r.get('success'))
        if agent_usage:
            print("\nAgent Usage:")
            for agent, count in agent_usage.items():
                print(f"  {agent}: {count} tasks")
        
        print(f"\n{'#'*70}")
        print("ORCHESTRATION COMPLETE")
        print(f"{'#'*70}\n")

        try:
            self._persist_session_artifacts(user_goal, tasks, results)
        except Exception:
            pass
        
        # === VRAM OPTIMIZATION: Reload Orchestrator ===
        # Reload Nemotron so it's ready for the next prompt
        if self.orchestrator_provider_type == 'ollama' and self.config.get('execution.vram_optimized', True):
            if hasattr(self.orchestrator_provider, 'chat'):
                if self.ui:
                    self.ui.log(f"Reloading Orchestrator ({self.orchestrator_model})...", "INFO")
                # Send dummy request to preload
                try:
                    self.orchestrator_provider.chat(
                        [LLMMessage(role="user", content="ping")], 
                        self.orchestrator_model,
                        max_tokens=1
                    )
                except:
                    pass

        if self.ui:
            self.ui.stop()
            try:
                self.ui.trace_event(
                    "run_complete",
                    {
                        "success": successful == len(tasks),
                        "tasks_total": len(tasks),
                        "tasks_successful": successful,
                        "duration_sec": round(total_elapsed, 2),
                    },
                )
            except Exception:
                pass
        
        return {
            "success": successful == len(tasks),
            "tasks_total": len(tasks),
            "tasks_successful": successful,
            "total_time": total_elapsed,
            "workflow_id": self.current_workflow_id,
            "results": results
        }
    
    def scan_models(self) -> List[str]:
        """Scan for new Ollama models"""
        new_models = self.model_watcher.scan_now()
        return [m['name'] for m in new_models]
    
    def list_models(self) -> List[Dict]:
        """List all registered models with capabilities"""
        return self.model_selector.get_model_rankings("balanced")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "orchestrator_model": self.orchestrator_model,
            "registered_agents": len(self.registry.get_all_agents()),
            "registered_tools": len(self.tool_executor.get_available_tools()),
            "model_watcher": self.model_watcher.get_status(),
            "monitor": self.monitor.get_dashboard_data()
        }


def main():
    """Test the orchestrator"""
    orchestrator = NemotronOrchestrator(
        enable_checkpoints=False,
        enable_testing=False
    )
    
    test_goal = "Create a Python function that validates email addresses using regex and write unit tests for it"
    orchestrator.run(test_goal)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED NEMOTRON ORCHESTRATOR")
    print("="*70)
    print("\nFeatures:")
    print("  - Dynamic model selection")
    print("  - Intent analysis")
    print("  - Task memory (inter-task context)")
    print("  - Vector memory (semantic search)")
    print("  - Error recovery")
    print("  - Human checkpoints")
    print("  - Testing workflow")
    print("  - Telemetry")
    print("="*70 + "\n")
    
    main()
