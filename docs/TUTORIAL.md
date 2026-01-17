# AI Autonom - Multi-Agent Orchestration System
## Complete Setup and Usage Tutorial

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick Start](#quick-start)
6. [CLI Commands](#cli-commands)
7. [Running Examples](#running-examples)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## System Overview

AI Autonom is a production-grade multi-agent orchestration system featuring:

- **Multi-Provider LLM Support**: Ollama (local), OpenAI, Azure OpenAI, and compatible APIs
- **CAI Framework Integration**: Agentic patterns, handoffs, guardrails, and HITL checkpoints
- **Dynamic Model Management**: Auto-discovery, capability assessment, and intelligent selection
- **Security Agents**: Web pentesting, vulnerability validation, and report generation
- **Testing Automation**: Auto-generated tests with quality gates
- **Docker Sandbox**: Isolated tool execution for security
- **Vector Memory**: ChromaDB for semantic search and context management

**Architecture:**
```
Nemotron-8B (Orchestrator)
    ├── Qwen3:1.7b (Coding tasks)
    ├── DictaLM-1.7B (Documentation/Hebrew)
    └── Security Agents (web_pentester, retester, report_agent)
```

---

## Prerequisites

### Required
- **Python 3.9+** (tested with Python 3.13)
- **Git** (for cloning repository)

### Optional (for full functionality)
- **Ollama** (for local LLM execution)
  - Download: https://ollama.ai
  - Models: `nemotron`, `qwen3:1.7b`, `dicta-il/dictalm2.0-instruct:1.7b`
- **Docker** (for sandboxed tool execution)
  - Download: https://www.docker.com/get-started
- **ChromaDB** (for vector memory)
  - Installed via pip: `pip install chromadb`

### API Keys (if using cloud providers)
- OpenAI API key
- Azure OpenAI credentials
- Or any OpenAI-compatible API endpoint

---

## Installation

### Step 1: Clone the Repository

```bash
git
cd AI_Autonom
``` clone <repository-url>

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pyyaml
requests
sqlite3 (built-in)
```

**Optional dependencies for full features:**
```bash
# For LangGraph workflow engine
pip install langgraph langchain

# For vector memory
pip install chromadb sentence-transformers

# For Docker integration
pip install docker

# For rich CLI formatting
pip install rich

# For Ollama integration
pip install ollama
```

### Step 4: Install Ollama Models (Optional)

If using Ollama for local execution:

```bash
# Install Ollama from https://ollama.ai

# Pull orchestrator model
ollama pull huihui_ai/orchestrator-abliterated

# Pull coding model
ollama pull qwen3:1.7b

# Pull documentation model
ollama pull dicta-il/dictalm2.0-instruct:1.7b

# Verify installation
ollama list
```

---

## Configuration

### Step 1: Review Settings

Edit `config/settings.yaml`:

```yaml
llm:
  # Provider priority order
  default_provider: "ollama"
  
  # Ollama configuration (local)
  ollama:
    base_url: "http://localhost:11434"
    default_model: "huihui_ai/orchestrator-abliterated"
  
  # OpenAI configuration (cloud)
  openai:
    api_key: "${OPENAI_API_KEY}"  # Set via environment variable
    model: "gpt-4"
    base_url: "https://api.openai.com/v1"
  
  # Azure OpenAI configuration
  azure_openai:
    api_key: "${AZURE_OPENAI_API_KEY}"
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    deployment: "gpt-4"
    api_version: "2024-02-15-preview"

agents:
  orchestrator:
    model: "huihui_ai/orchestrator-abliterated"
    provider: "ollama"
  
  coder_qwen:
    model: "qwen3:1.7b"
    provider: "ollama"
  
  linguistic_dictalm:
    model: "dicta-il/dictalm2.0-instruct:1.7b"
    provider: "ollama"

memory:
  vector_store:
    type: "chromadb"
    path: "data/chromadb"
    embedding_model: "all-MiniLM-L6-v2"
  
  task_memory:
    database: "data/task_memory.db"

tools:
  sandbox:
    enabled: true
    timeout: 30
    docker_image: "ai-autonom-sandbox"
```

### Step 2: Set Environment Variables (if using cloud APIs)

**Windows:**
```bash
set OPENAI_API_KEY=sk-your-key-here
set AZURE_OPENAI_API_KEY=your-azure-key
set AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=sk-your-key-here
export AZURE_OPENAI_API_KEY=your-azure-key
export AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
```

---

## Quick Start

### 1. Verify Installation

```bash
python run_orchestrator.py --status
```

**Expected output:**
```
System Status:

Orchestrator Model: huihui_ai/orchestrator-abliterated
Registered Agents: 5
Registered Tools: 25

Model Watcher:
  Running: True
  Last Scan: Never
  Discovered: 0

Execution Stats:
  Completed: 0
  Successful: 0
  Failed: 0
```

### 2. Scan for Models

```bash
python run_orchestrator.py --scan-models --register
```

This will:
- Detect all installed Ollama models
- Benchmark their capabilities (coding, reasoning, speed)
- Register them in the agent registry

### 3. List Available Models

```bash
python run_orchestrator.py --list-models
```

### 4. Run Your First Goal

```bash
python run_orchestrator.py "Write a Python function to calculate factorial"
```

Or in interactive mode:
```bash
python run_orchestrator.py --interactive
```

---

## CLI Commands

### Model Management

```bash
# Scan for new models
python run_orchestrator.py --scan-models

# Scan and auto-register
python run_orchestrator.py --scan-models --register

# List registered models with scores
python run_orchestrator.py --list-models

# List models ranked for specific task
python run_orchestrator.py --list-models --task-type coding

# Benchmark specific model
python run_orchestrator.py --benchmark qwen3:1.7b

# Benchmark and save
python run_orchestrator.py --benchmark qwen3:1.7b --save
```

### Agent Management

```bash
# List available agents
python run_orchestrator.py --list-agents

# Show system status
python run_orchestrator.py --status
```

### Tool Management

```bash
# List all available tools
python run_orchestrator.py --list-tools
```

### Execution

```bash
# Execute a goal directly
python run_orchestrator.py "Build a REST API with authentication"

# Execute with human checkpoints enabled
python run_orchestrator.py "Scan localhost:8000 for vulnerabilities" --checkpoints

# Execute with automatic testing
python run_orchestrator.py "Write email validator" --testing

# Dry run (show plan without execution)
python run_orchestrator.py "Refactor user authentication" --dry-run
```

### Memory Management

```bash
# Query vector memory
python run_orchestrator.py --query "What security vulnerabilities were found?"

# Clear vector memory
python run_orchestrator.py --clear-memory --memory-type vector --confirm

# Clear task memory
python run_orchestrator.py --clear-memory --memory-type task --confirm

# Clear all memory
python run_orchestrator.py --clear-memory --memory-type all --confirm
```

### Reports and Monitoring

```bash
# Export execution report
python run_orchestrator.py --export-report

# Export to specific file
python run_orchestrator.py --export-report --output reports/my_report.json

# Show live dashboard (requires rich)
python run_orchestrator.py --dashboard
```

### Interactive Mode

```bash
# Start interactive mode
python run_orchestrator.py --interactive

# In interactive mode, use these commands:
# /status    - Show system status
# /models    - List models
# /tools     - List tools
# /quit      - Exit
```

---

## Running Examples

### Example 1: Security Assessment Workflow

```bash
python examples/security_assessment.py
```

**What it demonstrates:**
- CAI security_pipeline pattern
- Agent handoffs (web_pentester → retester → report_agent)
- Human checkpoints for critical actions
- Intent analysis and pattern selection

**Output:**
```
[1/5] Initializing orchestrator...
[2/5] Loading security_pipeline pattern...
  Pattern: security_pipeline
  Agents: web_pentester -> retester -> report_agent
[3/5] Defining assessment goal...
  Goal: Security assessment of localhost:8000
[4/5] Analyzing intent...
  Task Type: security_assessment
  Complexity: moderate
[5/5] Executing workflow...
  Step 1: web_pentester scans target
  Step 2: Handoff to retester
  Step 3: Human checkpoint (CRITICAL)
  Step 4: Handoff to report_agent
```

### Example 2: Code Generation with Testing

```bash
python examples/code_generation.py
```

**What it demonstrates:**
- Code generation
- Automatic test creation
- code_review_recursive pattern
- Quality gates and validation

### Example 3: Dynamic Model Discovery

```bash
python examples/dynamic_models.py
```

**What it demonstrates:**
- Model discovery and benchmarking
- Capability assessment
- Dynamic model selection
- Model rankings

---

## Advanced Usage

### 1. Using Custom Configurations

```bash
python run_orchestrator.py --config config/custom_settings.yaml "Your goal here"
```

### 2. Running Integration Tests

```bash
python tests/test_integration.py
```

**Expected output:**
```
INTEGRATION TEST REPORT
Passed: 12
Failed: 0
Total: 12
```

### 3. Running Specific Component Tests

```bash
# Test checkpoint system
python tests/test_checkpoint.py

# Test testing workflow
python tests/test_testing_workflow.py

# Test error recovery
python tests/test_error_recovery.py
```

### 4. Docker Sandbox Setup

Build Docker containers for isolated tool execution:

```bash
# Build sandbox container
docker build -f docker/Dockerfile.sandbox -t ai-autonom-sandbox .

# Build security container
docker build -f docker/Dockerfile.security -t ai-autonom-security .

# Build web automation container
docker build -f docker/Dockerfile.web -t ai-autonom-web .

# Build Node.js container
docker build -f docker/Dockerfile.node -t ai-autonom-node .

# Or use docker-compose
cd docker
docker-compose up -d
```

### 5. Customizing Agents

Edit agent definitions in `src/agents/`:

```python
# Example: src/agents/custom_agent.py
from core.agent_registry import AgentDefinition

custom_agent = AgentDefinition(
    id="custom_agent",
    name="Custom Agent",
    description="Specialized agent for custom tasks",
    model_name="qwen3:1.7b",
    provider="ollama",
    capabilities=["custom_task", "special_processing"],
    tools=["filesystem_read", "python_exec"],
    handoffs=["report_agent"]
)

def get_system_prompt() -> str:
    return """You are a custom agent specialized in..."""

def get_instructions(context: dict) -> str:
    return f"""Process the following task: {context.get('task')}"""
```

Register in `src/core/agent_registry.py`:

```python
from agents.custom_agent import custom_agent
registry.register_agent(custom_agent)
```

### 6. Adding Custom Tools

Edit `src/tools/builtin_tools.py`:

```python
def my_custom_tool(input_data: str) -> str:
    """
    Custom tool implementation
    
    Args:
        input_data: Input for the tool
    
    Returns:
        Tool execution result
    """
    # Your tool logic here
    return f"Processed: {input_data}"

# Register tool
BUILTIN_TOOLS["my_custom_tool"] = {
    "id": "my_custom_tool",
    "description": "Custom tool for specific task",
    "category": "custom",
    "requires_sandbox": False,
    "function": my_custom_tool
}
```

### 7. Creating Custom Patterns

Add patterns in `src/patterns/cai_patterns.py`:

```python
def my_custom_pattern() -> AgenticPattern:
    """Custom agentic pattern"""
    return AgenticPattern(
        name="my_custom_pattern",
        pattern_type=PatternType.CHAIN,
        agents=["agent1", "agent2", "agent3"],
        handoff_rules={
            "agent1": "agent2",
            "agent2": "agent3"
        },
        entry_agent="agent1",
        termination_condition=lambda state: state.get("complete", False),
        max_iterations=10
    )

# Register pattern
PatternLibrary._patterns["my_custom_pattern"] = my_custom_pattern()
```

---

## Troubleshooting

### Issue 1: "Ollama not installed"

**Solution:**
1. Install Ollama from https://ollama.ai
2. Start Ollama service: `ollama serve`
3. Verify: `ollama list`

Or configure to use OpenAI/Azure instead:
```yaml
llm:
  default_provider: "openai"  # Change from "ollama"
```

### Issue 2: "ChromaDB not installed"

**Solution:**
```bash
pip install chromadb sentence-transformers
```

Or disable vector memory in `config/settings.yaml`:
```yaml
memory:
  vector_store:
    enabled: false
```

### Issue 3: "Docker SDK not installed"

**Solution:**
```bash
pip install docker
```

Or disable Docker sandbox:
```yaml
tools:
  sandbox:
    enabled: false
```

### Issue 4: "LangGraph not installed"

**Solution:**
```bash
pip install langgraph langchain
```

The system will use fallback workflow engine if LangGraph is not available.

### Issue 5: Model Not Found

**Error:** `Model 'qwen3:1.7b' not found`

**Solution:**
```bash
ollama pull qwen3:1.7b
python run_orchestrator.py --scan-models --register
```

### Issue 6: Unicode Encoding Errors (Windows)

**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:**
Set console encoding to UTF-8:
```bash
chcp 65001
```

Or in Python script:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

### Issue 7: Permission Denied (Docker)

**Solution:**

**Windows:** Run Docker Desktop as Administrator

**Linux:** Add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Issue 8: Port Already in Use

**Error:** `Address already in use: 11434` (Ollama)

**Solution:**
```bash
# Stop Ollama
killall ollama

# Restart
ollama serve
```

---

## Performance Tips

### 1. Model Selection
- Use `qwen3:1.7b` for fast coding tasks (45 tokens/sec)
- Use `codellama:7b` for complex algorithms (higher quality)
- Use `tinyllama:1.1b` for rapid prototyping (120 tokens/sec)

### 2. Memory Management
- Clear task memory between workflows:
  ```bash
  python run_orchestrator.py --clear-memory --memory-type task
  ```
- Use vector memory for long-term context retention

### 3. Docker Optimization
- Pre-build Docker images before heavy usage
- Use volume mounts for persistent data
- Limit container resources in `docker-compose.yml`

### 4. Batch Operations
- Use patterns for multi-agent workflows instead of sequential execution
- Enable parallel execution in patterns when possible

---

## System Architecture

```
AI_Autonom/
├── src/
│   ├── core/              # Core systems
│   │   ├── llm_provider.py       # Multi-provider LLM abstraction
│   │   ├── model_discovery.py   # Auto-discover Ollama models
│   │   ├── model_selector.py    # Dynamic model selection
│   │   ├── agent_registry.py    # Agent management
│   │   └── model_watcher.py     # Background model monitoring
│   ├── agents/            # Agent definitions
│   │   ├── web_pentester_agent.py
│   │   ├── retester_agent.py
│   │   └── report_agent.py
│   ├── patterns/          # CAI agentic patterns
│   │   ├── cai_patterns.py      # Pattern definitions
│   │   ├── handoffs.py          # Agent handoff system
│   │   └── guardrails.py        # Security guardrails
│   ├── orchestration/     # Workflow orchestration
│   │   ├── nemotron_orchestrator.py  # Main orchestrator
│   │   ├── langgraph_workflow.py     # LangGraph integration
│   │   ├── intent_analyzer.py        # Goal understanding
│   │   ├── error_recovery.py         # Error handling
│   │   ├── human_checkpoint.py       # HITL system
│   │   └── testing_workflow.py       # Test automation
│   ├── memory/            # Memory systems
│   │   ├── task_memory.py       # Inter-task context
│   │   └── vector_store.py      # ChromaDB integration
│   ├── tools/             # Tool execution
│   │   ├── tool_executor.py     # Main executor
│   │   └── builtin_tools.py     # Built-in tools
│   └── monitoring/        # Telemetry and logging
│       └── telemetry.py
├── tests/                 # Test suites
│   ├── test_integration.py
│   ├── test_checkpoint.py
│   └── test_testing_workflow.py
├── examples/              # Example workflows
│   ├── security_assessment.py
│   ├── code_generation.py
│   └── dynamic_models.py
├── config/                # Configuration
│   └── settings.yaml
├── docker/                # Docker containers
│   ├── Dockerfile.sandbox
│   ├── Dockerfile.security
│   └── docker-compose.yml
└── run_orchestrator.py    # Main CLI entry point
```

---

## Next Steps

1. **Start with examples:**
   ```bash
   python examples/security_assessment.py
   python examples/code_generation.py
   ```

2. **Run integration tests:**
   ```bash
   python tests/test_integration.py
   ```

3. **Try interactive mode:**
   ```bash
   python run_orchestrator.py --interactive
   ```

4. **Customize for your use case:**
   - Add custom agents in `src/agents/`
   - Define custom patterns in `src/patterns/`
   - Create custom tools in `src/tools/`

---

## Support and Documentation

- **Configuration:** `config/settings.yaml`
- **Agent Definitions:** `src/agents/`
- **Pattern Library:** `src/patterns/cai_patterns.py`
- **Tool Registry:** `src/tools/builtin_tools.py`
- **CLI Help:** `python run_orchestrator.py --help`

---

## License

[Add your license information here]

---

**Built with:**
- Python 3.9+
- Ollama / OpenAI / Azure OpenAI
- LangGraph
- ChromaDB
- Docker
