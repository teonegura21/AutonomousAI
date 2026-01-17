#!/usr/bin/env python3
"""
Enhanced Nemotron Orchestrator
Integrates all MVP features: model discovery, memory, tools, error recovery, checkpoints
Supports multiple LLM providers: Ollama (local), OpenAI, Azure OpenAI, and compatible APIs
"""

from typing import Dict, List, Optional, Any
import json
import time
import sys
import os
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
from ai_autonom.orchestration.agent_messaging import AgentMessageBus, MessageType
from ai_autonom.orchestration.error_recovery import ErrorRecovery, RecoveryExecutor
from ai_autonom.orchestration.human_checkpoint import HumanCheckpointManager
from ai_autonom.orchestration.testing_workflow import TestingWorkflow
from ai_autonom.orchestration.langgraph_workflow import MultiAgentWorkflow, WorkflowState
from ai_autonom.patterns.handoffs import HandoffManager
from ai_autonom.memory.knowledge_base import KnowledgeBase

# Monitoring imports
from ai_autonom.monitoring.telemetry import ExecutionMonitor
from ai_autonom.monitoring.live_dashboard import get_dashboard

try:
    from ai_autonom.monitoring.live_executor import get_live_monitor
    LIVE_MONITOR_AVAILABLE = True
except ImportError:
    LIVE_MONITOR_AVAILABLE = False


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
        enable_dashboard: bool = True
    ):
        # Load configuration
        self.config = Config().load(config_path)
        self.enable_dashboard = enable_dashboard
        
        if self.enable_dashboard:
            self.dashboard = get_dashboard()
            import threading
            self.dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
        else:
            self.dashboard = None
        
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
    
    def _init_providers(self):
        """Initialize LLM providers based on configuration"""
        self.providers: Dict[str, BaseLLMProvider] = {}
        
        # Initialize Ollama provider
        ollama_config = self.config.get_section('providers').get('ollama', {})
        if ollama_config.get('enabled', True):
            try:
                config = LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model=ollama_config.get('default_model', 'qwen3:1.7b'),
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
        self.model_watcher = ModelWatcher(self.model_discovery, interval=60)
        
        # Memory Systems
        self.task_memory = TaskMemory()
        self.vector_store = VectorMemoryStore()
        self.knowledge_base = KnowledgeBase.get_instance()
        
        # Tools
        self.tool_executor = ToolExecutor(workspace_dir="outputs")
        self.code_executor = CodeExecutor(workspace_dir="outputs")
        
        # Orchestration Components
        self.intent_analyzer = IntentAnalyzer()
        self.message_bus = AgentMessageBus()
        self.error_recovery = ErrorRecovery(max_retries=3)
        self.checkpoint = HumanCheckpointManager(auto_approve_low_risk=not enable_checkpoints)
        self.testing_workflow = TestingWorkflow()
        self.workflow_engine = MultiAgentWorkflow(orchestrator=self)
        self.handoff_manager = HandoffManager(self.registry)
        
        # Monitoring
        self.monitor = ExecutionMonitor()
        
        # State
        self.current_workflow_id = None
        self.enable_testing = enable_testing
        self.enable_checkpoints = enable_checkpoints
        
        # Setup registry after all components are initialized
        self._setup_registry()
    
    def _setup_registry(self):
        """Register default execution agents"""
        # CAI Retester Agent
        self.registry.register_agent(AgentDefinition(
            id="retester_agent",
            name="Retester Agent",
            model_name="alias1", # Will fallback to config default if not available
            model_size_gb=0.0,
            capabilities=["vulnerability_verification", "triage", "false_positive_elimination"],
            tools=["cai_generic_linux_command", "python_exec", "cai_google_search"],
            vram_required=0.0,
            speed_tokens_per_sec=50.0,
            quality_score=95.0,
            description="Specialist in validating vulnerabilities and eliminating false positives",
            provider="openai", # Assume alias1/openai compatible
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

        # Technical/Coding agent - Qwen3 1.7B (Ollama)
        self.registry.register_agent(AgentDefinition(
            id="coder_qwen",
            name="Qwen3 Technical Coder",
            model_name="qwen3:1.7b",
            model_size_gb=1.4,
            capabilities=["code_generation", "debugging", "refactoring", "python", "testing", "technical_tasks"],
            tools=["filesystem_read", "filesystem_write", "python_exec", "bash_exec", "pytest_run"],
            vram_required=1.4,
            speed_tokens_per_sec=70.0,
            quality_score=85.0,
            description="1.7B general purpose coder - fast and technical",
            provider="ollama"
        ))
        
        # Linguistic/Simple tasks agent - DictaLM 1.7B (Ollama)
        self.registry.register_agent(AgentDefinition(
            id="linguistic_dictalm",
            name="DictaLM Linguistic Agent",
            model_name="dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0",
            model_size_gb=1.8,
            capabilities=["text_generation", "documentation", "summarization", "formatting", "explanation", "simple_tasks"],
            tools=["filesystem_read", "filesystem_write"],
            vram_required=1.8,
            speed_tokens_per_sec=50.0,
            quality_score=80.0,
            description="1.7B thinking model for text and documentation",
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
    
    def decompose_and_assign(self, user_goal: str, intent_enhancement: str = "") -> List[Dict]:
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
        
        prompt = f"""You are Nemotron, a meta-orchestrator. You ONLY plan and assign - you NEVER execute.

User Goal: "{user_goal}"

{f'Additional Context: {intent_enhancement}' if intent_enhancement else ''}

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
            
            # Add testing tasks if enabled
            if self.enable_testing:
                tasks = self.testing_workflow.add_testing_tasks(tasks)
            
            # Add synthesis task if multiple coding tasks and single file requested
            tasks = self._add_synthesis_if_needed(tasks, user_goal)
            
            print(f"Generated {len(tasks)} tasks\n")
            
            # Display plan
            for i, task in enumerate(tasks, 1):
                print(f"  {i}. [{task.get('id')}]")
                print(f"     Agent: {task.get('assigned_agent')}")
                print(f"     Tools: {', '.join(task.get('tools', []))}")
                if task.get('dependencies'):
                    print(f"     Deps: {task.get('dependencies')}")
                print()
            
            return tasks
            
        except Exception as e:
            print(f"Decomposition failed: {e}")
            return []
    
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
        if self.dashboard:
            self.dashboard.set_orchestrator_status("Evaluating Tool Request")
            self.dashboard.log(f"Agent requested: {request}")
            
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
        self.task_memory.start_task(task_id)
        
        # Get dependency context
        dep_context = self.task_memory.get_dependency_context(task_id)
        
        # Look up agent
        agents = self.registry.get_all_agents()
        agent = next((a for a in agents if a.id == task.get('assigned_agent')), None)
        
        if not agent:
            error = f"Agent not found: {task.get('assigned_agent')}"
            self.task_memory.fail_task(task_id, error)
            self.monitor.log_task_complete(task_id, False, error=error)
            return {"success": False, "error": error}
            
        if self.dashboard:
            self.dashboard.set_active_agent(agent.name, "Initializing")
            self.dashboard.log(f"Starting task: {task_id}")
        
        print(f"\n{'─'*70}")
        print(f"EXECUTING: {task_id}")
        print(f"Agent: {agent.name} ({agent.model_name})")
        print(f"Tools: {', '.join(task.get('tools', []))}")
        print(f"{'─'*70}\n")
        
        # === LIVE MONITOR: Show task start ===
        if LIVE_MONITOR_AVAILABLE:
            monitor = get_live_monitor()
            monitor.show_task_start(task_id, agent.name, task.get('description', ''))
        
        # Build task description with context
        context_str = ""
        
        # 1. Dependency Context (Previous output)
        if dep_context.get("previous_outputs"):
            context_str += "\n\nContext from previous tasks:\n"
            for dep_id, dep_info in dep_context["previous_outputs"].items():
                context_str += f"\n--- {dep_id} ---\n{dep_info.get('output', '')[:1000]}\n"
        
        # 2. Knowledge Base Context (The Blackboard)
        kb_summary = self.knowledge_base.get_summary()
        context_str += f"\n\n{kb_summary}"
        
        task_description = task.get('description', '') + context_str
        
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
            
            # Construct full system message with tools and rules
            system_message = f"""{base_prompt}

Available tools: {', '.join(task.get('tools', []))}

You MUST use tools to complete this task. Don't just describe what to do - ACTUALLY DO IT!

DYNAMIC TOOL ACCESS:
If you determine that the assigned tools are insufficient, you may request access to any other tool in the registry.
To do this, output:
REQUEST_TOOL: <tool_name_or_description>
reason: <why you need it>

DYNAMIC AGENT HANDOFF:
If you need help from a specialist agent (e.g., 'Retester Agent' for verifying a bug, 'Web App Pentester' for complex web attacks), request a handoff.
To do this, output:
REQUEST_HANDOFF: <agent_name>
context: <what you want them to do>

The Orchestrator will evaluate your request. If approved, the tool will be added to your kit. If denied, you will be given an alternative strategy.

IMPORTANT RULES:
1. Start with reconnaissance tools (nmap, curl) to gather information
2. Only analyze files that EXIST (check with 'ls' first)
3. If a tool fails, READ THE ERROR and fix your parameters
4. DO NOT give up after one failure - retry with corrected parameters
5. Use 'cai_generic_linux_command' for any bash command
6. Focus on the ACTUAL target provided by the user

To use a tool, respond with:
                    TOOL: tool_name
                    target: value
                    args: value

Or for direct commands:
                    TOOL: cai_generic_linux_command
                    command: your command here

Examples:
                    TOOL: cai_nmap_scan
                    target: 192.168.1.0/24
                    args: -sV -sC

                    TOOL: cai_generic_linux_command
                    command: nmap -sV 192.168.1.1

After analyzing tool output:
                    - If tool FAILED: Fix parameters and try again
                    - If you need more data: Use another tool
                    - If task is FULLY COMPLETE: Say COMPLETE: [summary]

You have FULL FREEDOM to execute ANY command in the Kali container.

REMEMBER: DO NOT say COMPLETE until you've successfully gathered meaningful information!"""

            conversation_history = [
                LLMMessage(
                    role="system",
                    content=system_message
                ),
                LLMMessage(role="user", content=task_description)
            ]
            
            provider = self.get_provider_for_agent(agent)
            full_response = ""
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # === LIVE MONITOR: Show iteration ===
                if LIVE_MONITOR_AVAILABLE:
                    monitor = get_live_monitor()
                    monitor.show_iteration_header(iteration, max_iterations)
                else:
                    print(f"\n[Iteration {iteration}]")
                
                # Get agent response
                response = provider.chat(conversation_history, agent.model_name)
                agent_msg = response.content
                full_response += f"\n\n=== Iteration {iteration} ===\n{agent_msg}"
                
                if self.dashboard:
                    self.dashboard.set_active_agent(agent.name, "Thinking")
                    self.dashboard.update_agent_thought(agent_msg)
                
                # === LIVE MONITOR: Show agent thinking ===
                if LIVE_MONITOR_AVAILABLE:
                    monitor = get_live_monitor()
                    monitor.show_agent_response(agent.name, agent_msg, truncate=300)
                else:
                    print(f"Agent: {agent_msg[:500]}...\n")
                
                # Check if agent wants to use a tool or request one
                if "REQUEST_TOOL:" in agent_msg:
                    import re
                    req_match = re.search(r'REQUEST_TOOL:\s*(.+)', agent_msg)
                    reason_match = re.search(r'reason:\s*(.+)', agent_msg, re.IGNORECASE)
                    
                    requested_tool = req_match.group(1).strip() if req_match else "unknown"
                    
                    if LIVE_MONITOR_AVAILABLE:
                        monitor = get_live_monitor()
                        monitor.show_warning(f"Agent requesting new tool: {requested_tool}")
                    else:
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

                elif "REQUEST_HANDOFF:" in agent_msg:
                    import re
                    handoff_match = re.search(r'REQUEST_HANDOFF:\s*(.+)', agent_msg)
                    context_match = re.search(r'context:\s*(.+)', agent_msg, re.IGNORECASE)
                    
                    target_agent = handoff_match.group(1).strip() if handoff_match else "unknown"
                    handoff_context = context_match.group(1).strip() if context_match else ""
                    
                    print(f"\n[ORCHESTRATOR] Agent requesting handoff to: {target_agent}")
                    
                    # Evaluate handoff
                    handoff_decision = self.handoff_manager.evaluate_handoff_request(
                        agent.name,
                        target_agent,
                        handoff_context
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

                elif "TOOL:" in agent_msg or "COMPLETE:" in agent_msg:
                    # Parse tool call - simple line-based format
                    if "TOOL:" in agent_msg:
                        import re
                        
                        # Extract FIRST tool name only (agents sometimes try to call multiple)
                        tool_matches = re.findall(r'TOOL:\s*([\w_]+)', agent_msg)
                        if not tool_matches:
                            print("[ERROR] Malformed tool call\n")
                            break
                        
                        tool_name = tool_matches[0]  # Take only the FIRST tool
                        
                        if len(tool_matches) > 1:
                            print(f"[WARNING] Agent tried to call multiple tools at once: {tool_matches}")
                            print(f"[WARNING] Executing only first tool: {tool_name}\n")
                        
                        # Parse parameters from lines like "key: value"
                        # Only parse params for the FIRST tool
                        params = {}
                        in_first_tool = False
                        for line in agent_msg.split('\n'):
                            # Start parsing when we hit the first TOOL:
                            if line.strip().startswith('TOOL:') and tool_name in line:
                                in_first_tool = True
                                continue
                            # Stop if we hit another TOOL:
                            if line.strip().startswith('TOOL:') and tool_name not in line:
                                break
                            # Parse parameter lines
                            if in_first_tool and ':' in line:
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    value = parts[1].strip()
                                    if key and value and not key.startswith('TOOL'):
                                        params[key] = value
                        
                        print(f"[TOOL] Executing: {tool_name}")
                        print(f"[PARAMS] {params}\n")
                        
                        if self.dashboard:
                            self.dashboard.set_active_tool(f"{tool_name} {params.get('target', '')}")
                            self.dashboard.log(f"Agent executing {tool_name}")
                        
                        # === LIVE MONITOR: Show tool selection ===
                        if LIVE_MONITOR_AVAILABLE:
                            monitor = get_live_monitor()
                            monitor.show_tool_selection(agent.name, tool_name, params)
                        
                        # === ACTUALLY EXECUTE THE TOOL ===
                        success, tool_output = self.tool_executor.execute(
                            tool_name,
                            params,
                            agent_id=agent.id,
                            task_id=task_id
                        )
                        
                        if self.dashboard:
                            self.dashboard.set_active_tool(None)
                            status = "Success" if success else "Failed"
                            self.dashboard.log(f"Tool {tool_name}: {status}")
                        
                        print(f"[RESULT] {tool_output[:500]}...\n")
                        
                        # === RETRY LOGIC: Don't accept failure easily ===
                        if not success:
                            # Tool failed - force agent to retry with better parameters
                            error_feedback = f"""❌ TOOL EXECUTION FAILED!

Error: {tool_output}

You MUST fix this and try again. Common fixes:
- Check file paths exist (use 'ls' first to verify)
- Fix command syntax (check 'man command' for correct usage)
- Use correct parameter format
- Verify network targets are reachable

DO NOT say COMPLETE until the tool succeeds!
Analyze the error, fix the parameters, and call the tool again."""
                            
                            conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                            conversation_history.append(LLMMessage(
                                role="user",
                                content=error_feedback
                            ))
                            
                            if LIVE_MONITOR_AVAILABLE:
                                monitor = get_live_monitor()
                                monitor.show_warning(f"Tool '{tool_name}' failed - forcing retry")
                            else:
                                print(f"[WARNING] Tool failed - forcing agent to retry\n")
                        else:
                            # Tool succeeded - give output and continue
                            success_feedback = f"""✅ Tool '{tool_name}' executed successfully!

Output:
{tool_output}

Analyze the output. If you need more information:
- Use another tool to gather additional data
- Run the same tool with different parameters
- When task is fully complete, say: COMPLETE: [your summary]

Do not say COMPLETE if you still need to gather more information!"""
                            
                            conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                            conversation_history.append(LLMMessage(
                                role="user",
                                content=success_feedback
                            ))
                    
                    elif "COMPLETE:" in agent_msg:
                        # Agent wants to finish - check if they actually did anything useful
                        
                        # Count successful tool executions
                        successful_tools = full_response.count("✅ Tool") + full_response.count("SUCCESS")
                        
                        if successful_tools < 1:
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
                            
                            if LIVE_MONITOR_AVAILABLE:
                                monitor = get_live_monitor()
                                monitor.show_warning("Agent tried to complete task without gathering data - rejected")
                            else:
                                print("[WARNING] Agent tried to quit early - forcing retry\n")
                        else:
                            # Agent actually did work - allow completion
                            print("[COMPLETE] Agent finished task\n")
                            break
                
                else:
                    # Agent didn't use proper format - prompt them
                    conversation_history.append(LLMMessage(role="assistant", content=agent_msg))
                    conversation_history.append(LLMMessage(
                        role="user",
                        content="Please use tools! Format:\nTOOL: tool_name\nparameter: value\n\nOr say COMPLETE: [summary] if done."
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
            
            # Update memory
            self.task_memory.complete_task(task_id, full_response, [])
            self.monitor.log_task_complete(task_id, True, tokens=len(full_response.split()))
            
            # === LIVE MONITOR: Show task complete ===
            if LIVE_MONITOR_AVAILABLE:
                monitor = get_live_monitor()
                monitor.show_task_complete(task_id, True, elapsed)
            
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
            
            return {"success": False, "error": error}
    
    def run(self, user_goal: str) -> Dict[str, Any]:
        """Complete orchestration workflow"""
        if self.dashboard:
            self.dashboard.update_goal(user_goal)
            if not self.dashboard.running:
                self.dashboard_thread.start()
        
        print(f"\n{'#'*70}")
        print("NEMOTRON ORCHESTRATOR")
        print(f"{'#'*70}\n")
        print(f"Goal: {user_goal}\n")
        
        total_start = time.time()
        
        # Start workflow
        self.current_workflow_id = self.task_memory.start_workflow()
        
        if self.dashboard:
            self.dashboard.set_orchestrator_status("Analyzing Intent")
            
        # Phase 1: Intent Analysis
        intent_result = self.analyze_intent(user_goal)
        enhancement = intent_result.get("enhancement", "")
        
        if self.dashboard:
            self.dashboard.set_orchestrator_status("Decomposing Task")
            
        # Phase 2: Decomposition
        tasks = self.decompose_and_assign(user_goal, enhancement)
        
        if self.dashboard:
            self.dashboard.update_plan(tasks)
        
        if not tasks:
            return {"success": False, "error": "No tasks generated"}
        
        # Phase 3: Create workflow
        wf_id = self.workflow_engine.create_workflow(tasks, self.current_workflow_id)
        
        if self.dashboard:
            self.dashboard.set_orchestrator_status("Executing Plan")
        
        # Phase 4: Execute tasks
        results = []
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*70}")
            print(f"TASK {i}/{len(tasks)}")
            print(f"{'='*70}")
            
            result = self.execute_single_task(task)
            results.append(result)
            
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
            
            # Handle failure
            if not result.get("success"):
                decision = self.error_recovery.handle_failure(task, result.get("error", ""))
                if decision.action.value == "abort":
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
