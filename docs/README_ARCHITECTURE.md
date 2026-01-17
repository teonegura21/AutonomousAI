# AI Autonom - Modular Architecture

## Project Overview

AI Autonom is a production-grade multi-agent orchestration system featuring CAI (Cybersecurity AI) framework integration, dynamic model management, and comprehensive security tooling.

---

## Directory Structure

```
ai-autonom/
├── ai_autonom/              # Main package - Core library code
│   ├── core/               # Core systems (model management, config, providers)
│   ├── agents/             # Agent definitions (security agents, specialized agents)
│   ├── patterns/           # CAI agentic patterns & handoffs
│   ├── orchestration/      # Workflow orchestration & execution
│   ├── memory/             # Memory systems (task memory, vector store)
│   ├── tools/              # Tool execution & registry
│   ├── sandbox/            # Docker sandbox for isolated execution
│   └── monitoring/         # Telemetry & logging
├── cli/                    # Command-line interfaces
├── tests/                  # Test suites (unit, integration, fixtures)
├── examples/               # Example workflows & demonstrations
├── docs/                   # Documentation
├── docker/                 # Docker configurations
├── config/                 # Configuration templates
├── scripts/                # Utility scripts
└── .runtime/               # Runtime data (logs, databases, outputs)
```

---

## Module Documentation

### 1. `ai_autonom/core/` - Core Systems

**Purpose:** Foundation layer providing model management, configuration, and LLM provider abstraction.

**Key Files:**
- `config.py` - Configuration management and YAML loading
- `llm_provider.py` - Multi-provider LLM abstraction (Ollama, OpenAI, Azure)
- `model_discovery.py` - Auto-discovery of Ollama models with capability assessment
- `model_selector.py` - Dynamic model selection based on task requirements
- `model_watcher.py` - Background thread for continuous model monitoring
- `agent_registry.py` - Central registry for all agents

**Dependencies:**
- External: `pyyaml`, `requests`, `ollama` (optional)
- Internal: None (foundation layer)

**Key Functionality:**
- **Multi-Provider Support**: Unified interface for Ollama (local), OpenAI, Azure OpenAI, and compatible APIs
- **Dynamic Model Discovery**: Automatically detect and benchmark new models
- **Capability Assessment**: Test models for coding, reasoning, documentation, and speed
- **Model Selection**: Choose optimal model based on task type and requirements
- **Configuration Management**: YAML-based config with environment variable substitution

---

### 2. `ai_autonom/agents/` - Agent Definitions

**Purpose:** Define specialized agents with their capabilities, tools, prompts, and handoff rules.

**Key Files:**
- `web_pentester_agent.py` - Web application security testing agent
- `retester_agent.py` - Vulnerability validation and false positive elimination
- `report_agent.py` - Professional security report generation

**Dependencies:**
- Internal: `core/agent_registry.py`, `core/llm_provider.py`

**Key Functionality:**
- **Agent Definitions**: Structured agent definitions with CAI pattern compliance
- **System Prompts**: Comprehensive prompts for each agent's specialty
- **Dynamic Instructions**: Context-aware instruction generation
- **Tool Assignment**: Each agent has specific tools it can use
- **Handoff Rules**: Define which agents can delegate to which other agents

**Agent Structure:**
```python
AgentDefinition(
    id="agent_id",
    name="Agent Name",
    model_name="model",
    provider="ollama",
    capabilities=["capability1", "capability2"],
    tools=["tool1", "tool2"],
    handoffs=["other_agent_id"]
)
```

---

### 3. `ai_autonom/patterns/` - CAI Agentic Patterns

**Purpose:** Implement CAI framework patterns: AP = (A, H, D, C, E) - Agents, Handoffs, Decision, Communication, Execution.

**Key Files:**
- `cai_patterns.py` - Pattern definitions and executor
- `handoffs.py` - Agent-to-agent handoff system with callbacks and filters

**Dependencies:**
- Internal: `agents/`, `orchestration/agent_messaging.py`

**Key Functionality:**
- **Pattern Types**: SWARM, HIERARCHICAL, CHAIN, AUCTION, RECURSIVE, PARALLEL, CTF_SECURITY
- **Pattern Library**: Pre-defined patterns (security_pipeline, code_review_recursive, ctf_swarm)
- **Pattern Executor**: Execute patterns with state management and handoff orchestration
- **Handoff Manager**: Manages agent-to-agent delegation with:
  - Input filters (transform context between agents)
  - Callbacks (execute logic during handoffs)
  - Database logging
  - Message bus integration
  - Auto-handoff based on pattern rules

**Pattern Definition:**
```python
AgenticPattern(
    name="pattern_name",
    pattern_type=PatternType.CHAIN,
    agents=["agent1", "agent2"],
    handoff_rules={"agent1": "agent2"},
    entry_agent="agent1",
    termination_condition=lambda state: state.get("done"),
    max_iterations=10
)
```

---

### 4. `ai_autonom/orchestration/` - Workflow Orchestration

**Purpose:** High-level orchestration, workflow execution, intent analysis, error recovery, and human checkpoints.

**Key Files:**
- `nemotron_orchestrator.py` - Main orchestrator coordinating all components
- `langgraph_workflow.py` - LangGraph integration for DAG-based workflows
- `intent_analyzer.py` - Analyze user goals and suggest patterns/agents
- `error_recovery.py` - Multi-strategy error recovery (retry, exponential backoff, agent swap)
- `human_checkpoint.py` - HITL (Human-in-the-Loop) approval workflow
- `testing_workflow.py` - Automated test generation and quality gates
- `agent_messaging.py` - Pub/sub message bus for inter-agent communication

**Dependencies:**
- Internal: All modules (orchestration layer)
- External: `langgraph`, `langchain` (optional)

**Key Functionality:**
- **Goal Decomposition**: Break user goals into task DAGs
- **Intent Analysis**: Classify tasks (coding, security, documentation) and suggest patterns
- **Workflow Execution**: Execute task DAGs with LangGraph or fallback engine
- **Error Recovery**: Automatic retry with strategies:
  - RETRY: Simple retry with delay
  - EXPONENTIAL_BACKOFF: Increasing delays
  - AGENT_SWAP: Try different model
  - PROMPT_REFINEMENT: Add error context to prompt
  - HUMAN_INTERVENTION: Escalate to human
- **Human Checkpoints**: Pause execution for approval at:
  - CRITICAL: High-risk security actions
  - SYNTHESIS: Before combining outputs
  - ERROR: After retry exhaustion
- **Testing Workflow**: Auto-generate tests, run them, enforce quality gates

**Orchestrator Flow:**
```
User Goal → Intent Analysis → Pattern Selection → Task DAG Creation →
→ Agent Assignment → Execution → Error Recovery → Human Checkpoints →
→ Output Synthesis
```

---

### 5. `ai_autonom/memory/` - Memory Systems

**Purpose:** Manage task context, inter-task data flow, and semantic search capabilities.

**Key Files:**
- `task_memory.py` - In-memory and persistent task context management
- `vector_store.py` - ChromaDB integration for semantic search

**Dependencies:**
- Internal: None (can be used standalone)
- External: `chromadb`, `sentence-transformers` (optional)

**Key Functionality:**
- **Task Memory**:
  - Track task execution (status, inputs, outputs, decisions, errors)
  - Workflow management with metadata
  - Agent learning tracking (patterns, success rates)
  - Dependency context aggregation
  - Thread-safe operations with RLock
  - SQLite persistence
- **Vector Memory**:
  - Semantic search with embeddings
  - Store task outputs, tool executions, decisions, code artifacts
  - Query by natural language
  - Collections: task_outputs, tool_executions, agent_decisions, code_artifacts

**Task Memory Flow:**
```
start_workflow() → create_task_context() → start_task() →
→ add_decision() → record_learning() → complete_task() →
→ get_dependency_context() (for next task) → complete_workflow()
```

---

### 6. `ai_autonom/tools/` - Tool Execution

**Purpose:** Execute tools safely with sandboxing, routing, and comprehensive tool registry.

**Key Files:**
- `tool_executor.py` - Main tool execution engine with container routing
- `builtin_tools.py` - Built-in tools (filesystem, code execution, web, security)
- `tool_registry.py` - Tool registration and discovery
- `code_executor.py` - Safe code execution with timeout and output capture

**Dependencies:**
- Internal: `sandbox/` for Docker execution
- External: `docker` (optional), various tool libraries

**Key Functionality:**
- **Tool Categories**:
  - Filesystem: read, write, search, list, delete
  - Code Execution: python_exec, bash_exec, pytest_run
  - Web: web_fetch, web_search, selenium, playwright
  - Data: json_parse, json_format
  - Security: bandit_scan, safety_check, hash_file, ssl_check
  - Node.js: npm_run, node_exec, typescript_exec, eslint, jest
- **Sandboxing**: Route tools to Docker containers based on requirements
- **Container Routing**: 
  - Python tools → sandbox container
  - Security tools → security container
  - Web tools → web container (with browsers)
  - Node.js tools → nodejs container
- **Safety Features**:
  - Timeout enforcement
  - Output size limits
  - Error handling and logging
  - Isolated execution

---

### 7. `ai_autonom/sandbox/` - Docker Sandbox

**Purpose:** Provide isolated execution environments for tools and code.

**Key Files:**
- `docker_executor.py` - Docker container management and execution
- `container_router.py` - Route tools to appropriate containers

**Dependencies:**
- Internal: None (can be used standalone)
- External: `docker`

**Key Functionality:**
- **Container Management**:
  - Build and start containers on demand
  - Execute commands with timeout
  - Stream stdout/stderr in real-time
  - Cleanup and resource management
- **Container Types**:
  - `sandbox`: Python execution with common packages
  - `security`: Security analysis tools (nmap, nikto, etc.)
  - `web`: Browser automation (Selenium, Playwright)
  - `nodejs`: Node.js and TypeScript execution
- **Security Features**:
  - Network isolation (optional)
  - Resource limits (CPU, RAM)
  - Read-only mounts
  - No privileged access

---

### 8. `ai_autonom/monitoring/` - Telemetry & Logging

**Purpose:** Track execution metrics, task statistics, and system telemetry.

**Key Files:**
- `telemetry.py` - Execution monitoring and statistics

**Dependencies:**
- Internal: None (monitoring layer)

**Key Functionality:**
- **Execution Tracking**:
  - Task start/completion times
  - Success/failure rates
  - Token usage tracking
  - Agent utilization
- **Metrics Collection**:
  - Total tasks executed
  - Average task duration
  - Model performance by task type
  - Error rates and patterns
- **Export**:
  - JSON reports
  - Statistics queries
  - Performance analytics

---

## Data Flow

### Typical Execution Flow

```
1. CLI Entry (cli/orchestrator.py)
   ↓
2. Orchestrator Init (orchestration/nemotron_orchestrator.py)
   ↓
3. Intent Analysis (orchestration/intent_analyzer.py)
   ↓
4. Pattern Selection (patterns/cai_patterns.py)
   ↓
5. Agent Selection (core/model_selector.py)
   ↓
6. Workflow Creation (orchestration/langgraph_workflow.py)
   ↓
7. Task Execution Loop:
   - Get agent from registry (core/agent_registry.py)
   - Execute task with LLM (core/llm_provider.py)
   - Execute tools if needed (tools/tool_executor.py → sandbox/)
   - Record in task memory (memory/task_memory.py)
   - Check for handoffs (patterns/handoffs.py)
   - Error recovery if needed (orchestration/error_recovery.py)
   - Human checkpoint if critical (orchestration/human_checkpoint.py)
   ↓
8. Output Synthesis
   ↓
9. Store results (memory/vector_store.py)
   ↓
10. Monitor metrics (monitoring/telemetry.py)
```

---

## Inter-Module Dependencies

```
core/
  ↓ (provides foundation)
agents/ patterns/
  ↓ (use core)
orchestration/
  ↓ (coordinates all)
tools/ memory/ sandbox/ monitoring/
  ↓ (execution layer)
cli/
  ↓ (user interface)
```

**Dependency Rules:**
1. `core/` has no internal dependencies (foundation)
2. `agents/` and `patterns/` depend only on `core/`
3. `orchestration/` can depend on all modules (coordination layer)
4. `tools/`, `memory/`, `sandbox/`, `monitoring/` are independent utility modules
5. `cli/` depends on `orchestration/` (entry point)

---

## Configuration

**Primary Config:** `config/settings.yaml`

**Key Sections:**
- `llm`: Provider configuration (Ollama, OpenAI, Azure)
- `agents`: Agent-to-model mappings
- `memory`: Vector store and task memory settings
- `tools`: Sandbox and Docker configuration
- `monitoring`: Telemetry settings

**Environment Variables:**
- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_API_KEY`: Azure OpenAI key
- `AZURE_OPENAI_ENDPOINT`: Azure endpoint URL

---

## Runtime Data

**Location:** `.runtime/` (gitignored)

**Contents:**
- `data/`: SQLite databases (agent_registry.db, task_memory.db, checkpoints.db, etc.)
- `logs/`: Execution logs (execution.jsonl, orchestrator.log)
- `outputs/`: Generated outputs and reports
- `chromadb/`: ChromaDB vector store persistence

---

## Testing

**Test Organization:**
- `tests/unit/`: Unit tests for individual modules
- `tests/integration/`: Integration tests across modules
- `tests/fixtures/`: Test data and fixtures

**Key Test Files:**
- `test_integration.py`: Complete end-to-end integration tests
- `test_checkpoint.py`: Human checkpoint system tests
- `test_testing_workflow.py`: Testing workflow automation tests

**Run Tests:**
```bash
python tests/test_integration.py
python tests/test_checkpoint.py
python tests/test_testing_workflow.py
```

---

## Examples

**Location:** `examples/`

**Available Examples:**
- `security_assessment.py`: Full CAI security pipeline workflow
- `code_generation.py`: Code generation with automated testing
- `dynamic_models.py`: Model discovery and selection demonstration

**Run Examples:**
```bash
python examples/security_assessment.py
python examples/code_generation.py
python examples/dynamic_models.py
```

---

## Development Guidelines

### Adding a New Agent

1. Create agent file in `ai_autonom/agents/`
2. Define AgentDefinition with capabilities, tools, handoffs
3. Implement `get_system_prompt()` and `get_instructions(context)`
4. Register in `ai_autonom/core/agent_registry.py`

### Adding a New Tool

1. Add tool function to `ai_autonom/tools/builtin_tools.py`
2. Register in BUILTIN_TOOLS dictionary with metadata
3. Update `ai_autonom/sandbox/container_router.py` if needs specific container
4. Test tool execution via `tool_executor.py`

### Adding a New Pattern

1. Create pattern function in `ai_autonom/patterns/cai_patterns.py`
2. Define agents, handoff_rules, entry_agent, termination_condition
3. Register in PatternLibrary
4. Test pattern execution via `langgraph_workflow.py`

### Adding a New Provider

1. Create provider class in `ai_autonom/core/llm_provider.py` extending BaseLLMProvider
2. Implement `chat()`, `chat_stream()`, `list_models()`, `is_available()`
3. Add to ProviderType enum
4. Register in LLMProviderFactory
5. Update `config/settings.yaml` with provider section

---

## Performance Considerations

1. **Model Selection**: Use lightweight models (qwen3:1.7b) for fast tasks
2. **Memory Management**: Clear task memory between workflows
3. **Docker**: Pre-build containers, use volume mounts for persistence
4. **Vector Search**: Use smaller embedding models for faster search
5. **Parallel Execution**: Enable parallel execution in patterns when possible

---

## Security Considerations

1. **Sandboxing**: All tool execution happens in Docker containers
2. **Guardrails**: Input/output validation for security agents
3. **Scope Validation**: Security tools validate targets against allowed scope
4. **Human Checkpoints**: Critical actions require human approval
5. **API Keys**: Never commit API keys, use environment variables
6. **Tool Isolation**: Network isolation for untrusted code execution

---

## Troubleshooting

### Common Issues

1. **"Ollama not installed"**: Install Ollama or configure alternative provider
2. **"ChromaDB not installed"**: `pip install chromadb` or disable vector memory
3. **"Docker not available"**: Install Docker or disable sandboxing
4. **"Model not found"**: Run `python cli/orchestrator.py --scan-models --register`
5. **Import errors**: Ensure all dependencies installed: `pip install -r requirements.txt`

---

## Architecture Principles

1. **Modularity**: Each module has clear, single responsibility
2. **Loose Coupling**: Modules communicate through well-defined interfaces
3. **High Cohesion**: Related functionality grouped together
4. **Separation of Concerns**: Core, agents, orchestration, execution layers separated
5. **Dependency Inversion**: Depend on abstractions (interfaces), not concrete implementations
6. **Open/Closed**: Open for extension, closed for modification (plugin-style agents/tools)

---

## Future Enhancements

1. **Web UI**: Add web dashboard for monitoring and control
2. **API Server**: REST API for programmatic access
3. **More Patterns**: Implement additional CAI patterns (auction, tournament)
4. **Cloud Execution**: Support for distributed execution across cloud instances
5. **Advanced Memory**: Long-term memory with forgetting curves
6. **Feedback Loop**: Learn from human corrections and improve over time

---

## License

[Add license information]

---

## Contributing

[Add contribution guidelines]

---

**Last Updated:** 2026-01-17  
**Version:** 1.0.0  
**Maintainer:** [Your name/team]
