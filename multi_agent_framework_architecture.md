# The Complete Multi-Agent AI Framework Architecture

## A Ground-Up Guide for Building Production-Grade Autonomous AI Systems

**Author:** Claude (Anthropic) for Teo  
**Purpose:** Complete architectural explanation for building a hybrid local/API multi-agent system  
**Target Hardware:** RTX 4060 (8GB VRAM), 24GB RAM DDR5, Hybrid execution

---

# Part 1: Why Your Current System is "Dumb" — The Fundamental Problem

Before we build something better, let's understand exactly why your existing Nemotron-based orchestrator gets stuck after one prompt and doesn't understand nuances.

## 1.1 The Single-Turn Trap

Your current architecture likely works like this:

```
User Prompt → Nemotron Parses → JSON Plan → Agents Execute → Done
```

The problem? There's no **feedback loop**. When an agent runs `nmap` and gets results, those results don't inform the next decision. The system is essentially a **blind pipeline** — it plans once and executes once.

**What intelligent systems do:**

```
User Prompt → Plan → Execute Step 1 → Observe Result → Revise Plan → Execute Step 2 → ...
```

This is called the **ReAct pattern** (Reasoning + Acting), and it's the foundation of every capable agent system. The key insight is that agents must **think after every action**, not just at the beginning.

## 1.2 The Context Starvation Problem

When your agent "doesn't understand fine things," it's because it lacks **contextual continuity**. Consider this scenario:

```
Task: "Scan the network and find vulnerable services"

Step 1: Nmap runs, finds port 22, 80, 443 open
Step 2: Agent is asked to "analyze vulnerabilities"
```

If Step 2 doesn't receive the **full context** from Step 1 (the nmap output, the target IP, the reasoning about why we scanned), the agent is operating blind. It might ask "what network?" or try to scan again.

This is the **handoff problem** — how do you pass not just data, but understanding, from one step to the next?

## 1.3 The State Blindness Problem

Your system likely has no concept of "where we are" in a task. Consider:

```
User: "Build me a web app for tracking expenses"

What the system sees at step 5:
- Current task: "Create the database schema"
- What it doesn't know: 
  - What files have been created so far?
  - What decisions were made about the tech stack?
  - What's the overall architecture?
  - What has succeeded and what has failed?
```

Without **state management**, each step operates in isolation. The agent at step 5 might contradict decisions made at step 2 because it has no memory of them.

---

# Part 2: The Core Concepts — What Makes Agents "Intelligent"

## 2.1 The ReAct Pattern (Reasoning + Acting)

This is the most important concept. Every intelligent agent operates in a loop:

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct Loop                           │
│                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│   │  Think  │───▶│   Act   │───▶│ Observe │───┐       │
│   └─────────┘    └─────────┘    └─────────┘   │       │
│        ▲                                       │       │
│        └───────────────────────────────────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

In practice, this means the LLM generates output in a specific format:

```
Thought: I need to find open ports on the target. Nmap is the appropriate tool.
Action: run_nmap
Action Input: {"target": "192.168.1.1", "flags": "-sV -sC"}
Observation: [nmap output appears here]
Thought: Port 22 is running OpenSSH 7.6. This version has known vulnerabilities...
Action: search_cve
Action Input: {"product": "OpenSSH", "version": "7.6"}
...
```

The **Thought** step is crucial — it forces the model to reason before acting.

## 2.2 Tool Use — The Agent's Hands

An agent without tools is just a chatbot. Tools are functions the LLM can invoke:

```python
# A tool is just a function with a description
def run_nmap(target: str, flags: str = "-sV") -> str:
    """
    Run an nmap scan against a target.
    
    Args:
        target: IP address or hostname to scan
        flags: Nmap flags (default: -sV for version detection)
    
    Returns:
        The nmap scan output as a string
    """
    result = subprocess.run(["nmap", flags, target], capture_output=True)
    return result.stdout.decode()
```

The LLM sees the function signature and docstring, then decides when to call it.

## 2.3 Memory — Short-term, Long-term, and Working

**Short-term memory:** The current conversation context. Limited by the model's context window (e.g., 128K tokens for Qwen, 200K for Claude).

**Working memory:** The current task's state — what steps have been completed, what decisions were made, what files exist.

**Long-term memory:** Persisted information across sessions — user preferences, past solutions, learned patterns. Usually stored in a vector database.

Your system likely only has short-term memory, and even that is probably reset between steps.

## 2.4 Handoffs — Passing the Baton

A handoff is when one agent transfers control to another. This requires:

1. **Context transfer:** The receiving agent needs to know what happened before
2. **Goal clarity:** Why is control being transferred?
3. **State snapshot:** What's the current state of the task?

CAI formalizes this as: `H: A × T → A` (Handoff function maps Agent × Task to new Agent)

```python
# A proper handoff includes everything the next agent needs
handoff = {
    "from_agent": "reconnaissance_agent",
    "to_agent": "exploitation_agent",
    "reason": "Vulnerability found in OpenSSH 7.6, transferring for exploitation",
    "context": {
        "target": "192.168.1.1",
        "open_ports": [22, 80, 443],
        "vulnerability": "CVE-2018-15473",
        "nmap_output": "...",
        "work_completed": ["port_scan", "service_detection", "cve_lookup"],
    },
    "goal": "Exploit CVE-2018-15473 to gain initial access"
}
```

## 2.5 Orchestration — The Conductor

The orchestrator (your Nemotron) has a specific job: **decompose complex tasks and route them**. It does NOT execute tasks itself.

Good orchestration requires:

1. **Task decomposition:** Breaking "build me an app" into atomic steps
2. **Agent selection:** Matching each step to the right specialist
3. **Dependency tracking:** Knowing step 3 depends on step 1 and 2
4. **Progress monitoring:** Detecting when a step fails or stalls
5. **Replanning:** Adjusting the plan when things go wrong

---

# Part 3: The Complete Technology Stack

## 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LAYER 7: UI                                 │
│              (Web Interface, CLI, IDE Integration)                  │
├─────────────────────────────────────────────────────────────────────┤
│                      LAYER 6: ORCHESTRATOR                          │
│            (Nemotron-Orchestrator-8B via Ollama/vLLM)              │
├─────────────────────────────────────────────────────────────────────┤
│                   LAYER 5: WORKFLOW ENGINE                          │
│                  (LangGraph State Machine)                          │
├─────────────────────────────────────────────────────────────────────┤
│                     LAYER 4: AGENT LAYER                            │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│     │ OpenCode │ │   CAI    │ │ Research │ │ General  │           │
│     │  Coder   │ │ Security │ │  Agent   │ │  Agent   │           │
│     └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├─────────────────────────────────────────────────────────────────────┤
│                      LAYER 3: TOOL LAYER                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │
│  │ Files  │ │ Shell  │ │ Search │ │ Browser│ │ Nmap   │          │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘          │
├─────────────────────────────────────────────────────────────────────┤
│                   LAYER 2: EXECUTION SANDBOX                        │
│              (E2B / Daytona / Docker Container)                     │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: MODEL INFERENCE                         │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │ LOCAL: Ollama/vLLM (Nemotron Q4, Qwen3 1.7B)              │   │
│   │ API: OpenAI, Anthropic, DeepSeek (fallback/heavy tasks)   │   │
│   └────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 0: PERSISTENCE                             │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│   │   SQLite     │ │   ChromaDB   │ │    Redis     │              │
│   │ (State/Logs) │ │ (Vector/RAG) │ │ (Task Queue) │              │
│   └──────────────┘ └──────────────┘ └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

Let me explain each layer in detail.

---

## 3.2 Layer 0: Persistence — Where State Lives

### SQLite: Operational State

Stores task state, agent logs, execution history. Why SQLite?
- Zero configuration, file-based
- ACID compliant (no corrupted state)
- Fast enough for single-machine deployments
- Built into Python

```python
# What we store in SQLite
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    parent_id TEXT,              -- For subtasks
    status TEXT,                 -- pending, running, completed, failed
    assigned_agent TEXT,
    input_context TEXT,          -- JSON blob of input
    output_result TEXT,          -- JSON blob of output  
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE agent_messages (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    agent_name TEXT,
    role TEXT,                   -- system, user, assistant, tool
    content TEXT,
    tool_calls TEXT,             -- JSON of any tool invocations
    timestamp TIMESTAMP
);
```

### ChromaDB: Vector Memory

Stores embeddings for semantic search. Use cases:
- "Find similar code I wrote before"
- "What did we discuss about authentication?"
- "Retrieve relevant documentation for this error"

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("agent_memory")

# Store a memory
collection.add(
    documents=["Successfully exploited CVE-2018-15473 using paramiko"],
    metadatas=[{"task_type": "exploitation", "cve": "CVE-2018-15473"}],
    ids=["memory_001"]
)

# Retrieve relevant memories
results = collection.query(
    query_texts=["How do I exploit OpenSSH vulnerabilities?"],
    n_results=5
)
```

### Redis (Optional): Task Queue

For distributed execution or high-throughput scenarios. Overkill for single-machine, but useful if you scale.

---

## 3.3 Layer 1: Model Inference — The Brains

### Local Inference with Ollama

Ollama is the simplest way to run local models. Your setup:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull your models
ollama pull nemotron                    # Orchestrator (will use Q4 by default)
ollama pull qwen3:1.7b                 # Light agent for simple tasks

# Run with GPU (auto-detected on Linux with NVIDIA drivers)
ollama serve
```

**API Usage:**

```python
import requests

def query_local_model(prompt: str, model: str = "nemotron") -> str:
    """Query a local Ollama model."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]
```

### API Fallback (Commercial/Heavy Tasks)

For tasks that exceed local capacity (long context, complex reasoning):

```python
from openai import OpenAI
import anthropic

class ModelRouter:
    """Routes requests to local or API models based on complexity."""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.openai = OpenAI()
        self.anthropic = anthropic.Anthropic()
    
    def route(self, prompt: str, task_complexity: str = "simple") -> str:
        """
        Route to appropriate model:
        - simple: Qwen 1.7B local
        - medium: Nemotron local  
        - complex: API (Claude/GPT-4)
        """
        if task_complexity == "simple":
            return self._query_ollama(prompt, "qwen3:1.7b")
        elif task_complexity == "medium":
            return self._query_ollama(prompt, "nemotron")
        else:
            return self._query_anthropic(prompt)  # Or OpenAI
```

### VRAM Budget (RTX 4060 8GB)

| Model | VRAM (Q4) | Use Case |
|-------|-----------|----------|
| Nemotron-Orchestrator-8B | ~5GB | Orchestration |
| Qwen3-1.7B | ~2GB | Simple agents |
| **Combined** | ~7GB | Both running simultaneously |

You have 1GB headroom. This works, but you can't add a third model without swapping.

---

## 3.4 Layer 2: Execution Sandbox — Where Code Runs Safely

Agent-generated code is dangerous. It could `rm -rf /`, exfiltrate data, or mine cryptocurrency. Sandboxing is non-negotiable.

### Option A: E2B (Recommended for Cloud/Hybrid)

```python
from e2b_code_interpreter import Sandbox

# Create isolated environment
sandbox = Sandbox()

# Execute code safely
execution = sandbox.run_code("""
import nmap
nm = nmap.PortScanner()
nm.scan('192.168.1.1', '22-443')
print(nm.csv())
""")

print(execution.text)  # Safe output
sandbox.kill()         # Clean up
```

**E2B gives you:**
- Firecracker microVMs (stronger isolation than Docker)
- ~150ms startup time
- Filesystem, network, process isolation
- Pre-built images with common tools

### Option B: Daytona (Self-Hosted Alternative)

```python
from daytona import Daytona

daytona = Daytona()
sandbox = daytona.create()

# Execute code
response = sandbox.process.code_run('print("Hello from sandbox!")')
print(response.result)

# Execute shell commands
response = sandbox.process.exec('nmap -sV 192.168.1.1', timeout=60)
print(response.result)
```

### Option C: Docker (Manual but Free)

```python
import docker

client = docker.from_env()

def execute_in_sandbox(code: str, image: str = "python:3.11-slim") -> str:
    """Execute code in an isolated Docker container."""
    container = client.containers.run(
        image,
        command=["python", "-c", code],
        detach=True,
        mem_limit="512m",           # Memory limit
        cpu_period=100000,
        cpu_quota=50000,            # 50% CPU max
        network_disabled=True,       # No network by default
        read_only=True,             # Read-only filesystem
        remove=True                  # Auto-cleanup
    )
    logs = container.logs().decode()
    return logs
```

**For your use case:** Start with Docker (free, you control it), migrate to E2B/Daytona for production.

---

## 3.5 Layer 3: Tool Layer — The Agent's Capabilities

Tools are the functions agents can invoke. Here's a comprehensive toolkit:

### File Operations

```python
from pathlib import Path
from typing import Optional

def read_file(path: str) -> str:
    """Read contents of a file."""
    return Path(path).read_text()

def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} bytes to {path}"

def list_directory(path: str = ".") -> list[str]:
    """List files and directories at path."""
    return [str(p) for p in Path(path).iterdir()]
```

### Shell Execution (Sandboxed)

```python
import subprocess

def run_command(command: str, timeout: int = 30) -> dict:
    """
    Execute a shell command in the sandbox.
    
    Returns:
        dict with 'stdout', 'stderr', 'returncode'
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
```

### Web Search

```python
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} 
                for r in results]
```

### Security Tools (From CAI)

```python
import nmap

def nmap_scan(target: str, arguments: str = "-sV -sC") -> str:
    """
    Run an nmap scan.
    
    Args:
        target: IP/hostname/range to scan
        arguments: Nmap arguments (default: version + script scan)
    """
    nm = nmap.PortScanner()
    nm.scan(target, arguments=arguments)
    return nm.csv()

def search_cve(product: str, version: str = "") -> list[dict]:
    """Search for CVEs affecting a product."""
    # Implementation using NVD API or local CVE database
    pass
```

### Tool Registry

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: dict  # JSON Schema for parameters

class ToolRegistry:
    """Central registry of all available tools."""
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        return self.tools[name]
    
    def list_for_agent(self, agent_type: str) -> list[Tool]:
        """Return tools available for a specific agent type."""
        # Different agents have access to different tools
        tool_permissions = {
            "coder": ["read_file", "write_file", "run_command", "web_search"],
            "security": ["nmap_scan", "search_cve", "run_command", "read_file"],
            "research": ["web_search", "read_file", "write_file"],
        }
        allowed = tool_permissions.get(agent_type, [])
        return [t for name, t in self.tools.items() if name in allowed]
```

---

## 3.6 Layer 4: Agent Layer — The Specialists

This is where CAI, OpenCode, and OpenManus come together.

### Agent Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentContext:
    """Everything an agent needs to do its job."""
    task_description: str
    goal: str
    available_tools: list[Tool]
    conversation_history: list[dict]
    working_memory: dict              # Current task state
    long_term_memory_results: list    # Retrieved from vector DB

@dataclass  
class AgentResult:
    """What an agent returns after execution."""
    success: bool
    output: str
    artifacts: list[str]              # Files created, etc.
    handoff_to: Optional[str]         # Next agent if handoff needed
    handoff_context: Optional[dict]   # Context for handoff

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, model: str, system_prompt: str):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
    
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task. Must be implemented by subclasses."""
        pass
    
    def _create_prompt(self, context: AgentContext) -> str:
        """Build the full prompt for the LLM."""
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" 
            for t in context.available_tools
        ])
        
        return f"""
{self.system_prompt}

## Available Tools
{tool_descriptions}

## Current Task
{context.task_description}

## Goal
{context.goal}

## Relevant Context from Memory
{context.long_term_memory_results}

## Conversation History
{self._format_history(context.conversation_history)}

Now, think step by step and use tools to accomplish the goal.
"""
```

### Coding Agent (Borrowing from OpenCode)

```python
class CodingAgent(BaseAgent):
    """Agent specialized for code generation and modification."""
    
    def __init__(self):
        super().__init__(
            name="coder",
            model="nemotron",  # Or API for complex tasks
            system_prompt="""You are an expert software engineer.
You write clean, well-documented, production-quality code.

When given a coding task:
1. First understand the requirements fully
2. Plan your approach before coding
3. Write code incrementally, testing as you go
4. Use appropriate design patterns
5. Handle errors gracefully
6. Document your code

You have access to file operations and shell commands.
Always verify your code works before marking the task complete."""
        )
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute coding task using ReAct loop."""
        max_iterations = 20
        conversation = context.conversation_history.copy()
        
        for i in range(max_iterations):
            # Generate next action
            prompt = self._create_prompt(context)
            response = query_model(prompt, self.model)
            
            # Parse response for tool calls
            thought, action, action_input = self._parse_response(response)
            
            if action == "FINISH":
                return AgentResult(
                    success=True,
                    output=thought,
                    artifacts=self._collect_artifacts(),
                    handoff_to=None,
                    handoff_context=None
                )
            
            # Execute tool
            tool = context.available_tools.get(action)
            observation = tool.function(**action_input)
            
            # Add to conversation
            conversation.append({
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"
            })
            conversation.append({
                "role": "user", 
                "content": f"Observation: {observation}"
            })
            
            context.conversation_history = conversation
        
        return AgentResult(success=False, output="Max iterations reached")
```

### Security Agent (Borrowing from CAI)

```python
class SecurityAgent(BaseAgent):
    """Agent specialized for cybersecurity tasks."""
    
    def __init__(self, mode: str = "offensive"):
        # CAI-style system prompts
        if mode == "offensive":
            system_prompt = """You are a red team security expert.
Your goal is to identify and demonstrate vulnerabilities.

Methodology:
1. Reconnaissance: Gather information about the target
2. Enumeration: Identify services, versions, potential attack vectors
3. Vulnerability Analysis: Map findings to known CVEs
4. Exploitation: Demonstrate impact (within authorized scope)
5. Documentation: Record findings with evidence

Always operate within authorized scope. Document everything."""
        else:
            system_prompt = """You are a blue team security expert.
Your goal is to identify and remediate vulnerabilities.

Methodology:
1. Asset Discovery: Understand what needs protection
2. Vulnerability Assessment: Identify weaknesses
3. Risk Analysis: Prioritize by impact and likelihood
4. Remediation: Provide actionable fixes
5. Verification: Confirm fixes are effective"""
        
        super().__init__(
            name="security",
            model="nemotron",
            system_prompt=system_prompt
        )
```

### Research Agent (Borrowing from OpenManus)

```python
class ResearchAgent(BaseAgent):
    """Agent specialized for information gathering and analysis."""
    
    def __init__(self):
        super().__init__(
            name="researcher",
            model="nemotron",
            system_prompt="""You are an expert researcher and analyst.

Your methodology:
1. Understand the research question
2. Search multiple sources (web, documentation, code)
3. Evaluate source credibility
4. Synthesize findings
5. Present conclusions with citations

Always cite sources. Distinguish facts from inferences."""
        )
```

---

## 3.7 Layer 5: Workflow Engine — LangGraph

LangGraph is what ties agents together with proper state management.

### Why LangGraph?

1. **State persistence:** Every step is checkpointed
2. **Conditional routing:** "If vulnerability found, go to exploitation agent"
3. **Parallel execution:** Run independent tasks simultaneously
4. **Human-in-the-loop:** Pause for approval when needed
5. **Replay/debug:** Go back to any checkpoint

### Basic LangGraph Setup

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define the state schema
class AgentState(TypedDict):
    # The task we're working on
    task: str
    
    # Current plan (list of steps)
    plan: list[dict]
    
    # Which step we're on
    current_step: int
    
    # Results from completed steps (accumulates)
    step_results: Annotated[list[dict], operator.add]
    
    # Working memory (key-value store)
    memory: dict
    
    # Final output
    final_output: str
    
    # Error tracking
    errors: list[str]

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes (each node is a function)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("coder", coder_node)
workflow.add_node("security", security_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("synthesizer", synthesizer_node)

# Add edges
workflow.set_entry_point("orchestrator")

# Conditional routing from orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    route_to_agent,  # Function that decides which agent
    {
        "coder": "coder",
        "security": "security",
        "researcher": "researcher",
        "done": "synthesizer"
    }
)

# All agents can loop back to orchestrator or finish
for agent in ["coder", "security", "researcher"]:
    workflow.add_conditional_edges(
        agent,
        check_completion,
        {
            "continue": "orchestrator",
            "done": "synthesizer"
        }
    )

workflow.add_edge("synthesizer", END)

# Compile the graph
app = workflow.compile()
```

### Node Implementations

```python
def orchestrator_node(state: AgentState) -> dict:
    """
    The orchestrator analyzes the task and creates/updates the plan.
    This is where Nemotron-Orchestrator-8B shines.
    """
    # Build prompt for orchestrator
    prompt = f"""
You are an orchestrator agent. Your job is to break down tasks and assign them.

## Current Task
{state['task']}

## Plan So Far
{state['plan']}

## Completed Steps
{state['step_results']}

## Available Agents
- coder: For all programming and file manipulation tasks
- security: For vulnerability scanning, CVE research, exploitation
- researcher: For web search, documentation lookup, analysis

Analyze the current state. If more work is needed, output the next step.
If all work is complete, output DONE.

Output format:
```json
{{
    "reasoning": "Your analysis of what needs to happen next",
    "next_step": {{
        "agent": "coder|security|researcher",
        "task": "Specific task description",
        "dependencies": ["list of step IDs this depends on"],
        "tools_needed": ["list of required tools"]
    }},
    "status": "continue|done"
}}
```
"""
    
    response = query_model(prompt, "nemotron-orchestrator-8b")
    parsed = json.loads(extract_json(response))
    
    # Update plan
    new_plan = state['plan'].copy()
    if parsed['status'] == 'continue':
        new_plan.append({
            "step_id": len(new_plan),
            **parsed['next_step']
        })
    
    return {
        "plan": new_plan,
        "current_step": len(new_plan) - 1 if parsed['status'] == 'continue' else -1,
        "memory": {**state['memory'], "last_reasoning": parsed['reasoning']}
    }

def coder_node(state: AgentState) -> dict:
    """Execute the current coding task."""
    current_task = state['plan'][state['current_step']]
    
    # Build context for coding agent
    context = AgentContext(
        task_description=current_task['task'],
        goal=f"Complete step {state['current_step']} of the plan",
        available_tools=get_tools_for_agent("coder"),
        conversation_history=[],
        working_memory=state['memory'],
        long_term_memory_results=query_vector_db(current_task['task'])
    )
    
    # Execute agent
    agent = CodingAgent()
    result = agent.execute(context)
    
    return {
        "step_results": [{
            "step_id": state['current_step'],
            "agent": "coder",
            "success": result.success,
            "output": result.output,
            "artifacts": result.artifacts
        }],
        "memory": {**state['memory'], f"step_{state['current_step']}_result": result.output}
    }

def route_to_agent(state: AgentState) -> str:
    """Decide which agent should handle the next step."""
    if state['current_step'] < 0:
        return "done"
    
    current_task = state['plan'][state['current_step']]
    return current_task['agent']

def check_completion(state: AgentState) -> str:
    """Check if the current agent completed successfully."""
    last_result = state['step_results'][-1] if state['step_results'] else None
    
    if last_result and last_result['success']:
        return "continue"  # Go back to orchestrator for next step
    else:
        return "continue"  # Orchestrator will handle failures
```

### Running the Workflow

```python
# Initialize state
initial_state = {
    "task": "Build a web scraper that extracts job listings from Indeed",
    "plan": [],
    "current_step": -1,
    "step_results": [],
    "memory": {},
    "final_output": "",
    "errors": []
}

# Run with streaming
for event in app.stream(initial_state):
    print(f"Step: {event}")

# Or run to completion
final_state = app.invoke(initial_state)
print(final_state['final_output'])
```

---

## 3.8 Layer 6: Orchestrator — Nemotron-Orchestrator-8B

The orchestrator is the brain that plans and coordinates. Let's understand why Nemotron-Orchestrator-8B is special.

### What Makes It Different

Nemotron-Orchestrator-8B was specifically trained with:

1. **Multi-objective RL:** Optimized for accuracy, cost, AND latency simultaneously
2. **Tool orchestration:** Knows how to pick the right tool/model for each subtask
3. **Preference adaptation:** Can adjust behavior based on user preferences (fast vs accurate)

### Running Locally

```bash
# Option 1: Ollama (easiest)
ollama pull hf.co/nvidia/nemotron-orchestrator-8b:latest

# Option 2: vLLM (better performance)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Nemotron-Orchestrator-8B \
    --quantization awq
```

### Orchestrator Prompt Engineering

The orchestrator needs a specific prompt format to work well:

```python
ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestration agent that coordinates multiple specialized agents and tools.

Your capabilities:
1. Break complex tasks into atomic subtasks
2. Assign each subtask to the most appropriate agent
3. Track dependencies between subtasks
4. Monitor progress and adapt the plan as needed

Available agents:
{agent_descriptions}

Available tools:
{tool_descriptions}

Your output must be valid JSON with this structure:
{
    "analysis": "Your understanding of the task and current state",
    "plan": [
        {
            "step_id": 0,
            "agent": "agent_name",
            "task": "Specific task description",
            "tools": ["tool1", "tool2"],
            "depends_on": [],
            "expected_output": "What this step should produce"
        }
    ],
    "next_action": {
        "execute_step": 0,
        "or": "wait_for_completion" | "request_clarification" | "mark_complete"
    }
}
"""
```

---

## 3.9 Layer 7: UI — Making It Usable

### Option A: Terminal UI (Like OpenCode)

OpenCode uses Bubble Tea (Go library) for its TUI. For Python:

```python
from textual.app import App
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import Container

class AgentUI(App):
    """Terminal UI for the agent framework."""
    
    CSS = """
    #conversation {
        height: 80%;
        border: solid green;
    }
    #input {
        dock: bottom;
        height: 3;
    }
    """
    
    def compose(self):
        yield Header()
        yield RichLog(id="conversation", highlight=True, markup=True)
        yield Input(id="input", placeholder="Enter your task...")
        yield Footer()
    
    async def on_input_submitted(self, event):
        log = self.query_one("#conversation")
        log.write(f"[bold cyan]You:[/] {event.value}")
        
        # Run the agent workflow
        result = await run_agent_async(event.value)
        
        for step in result:
            log.write(f"[bold green]{step['agent']}:[/] {step['output']}")
```

### Option B: Web UI (Production)

For a rich UI, use FastAPI + React:

**Backend (FastAPI):**

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # Receive task from frontend
        data = await websocket.receive_json()
        task = data.get("task")
        
        # Stream agent execution
        async for event in run_agent_streaming(task):
            await websocket.send_json({
                "type": "step",
                "agent": event["agent"],
                "content": event["content"],
                "status": event["status"]
            })
        
        await websocket.send_json({"type": "complete"})
```

**Frontend (React):**

```jsx
import { useState, useEffect, useRef } from 'react';

function AgentChat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const wsRef = useRef(null);
    
    useEffect(() => {
        wsRef.current = new WebSocket('ws://localhost:8000/ws');
        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setMessages(prev => [...prev, data]);
        };
    }, []);
    
    const sendTask = () => {
        wsRef.current.send(JSON.stringify({ task: input }));
        setInput('');
    };
    
    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, i) => (
                    <div key={i} className={`message ${msg.agent}`}>
                        <strong>{msg.agent}:</strong> {msg.content}
                    </div>
                ))}
            </div>
            <input 
                value={input} 
                onChange={e => setInput(e.target.value)}
                onKeyPress={e => e.key === 'Enter' && sendTask()}
            />
        </div>
    );
}
```

---

# Part 4: What to Take from Each Open-Source Project

Now that you understand the layers, here's exactly what to extract from each project:

## 4.1 From CAI (aliasrobotics/cai)

**Repository:** `github.com/aliasrobotics/cai`

### Take These Files:

| Path | What It Is | Use For |
|------|------------|---------|
| `src/cai/agents/` | Agent definitions | Blueprint for security agents |
| `src/cai/agents/patterns/` | Agentic patterns | Hierarchical, chain-of-thought patterns |
| `src/cai/tools/` | Security tool implementations | Nmap, CVE search, exploitation tools |
| `src/cai/sdk/agents/` | Core SDK with handoff mechanism | The handoff implementation |
| `src/cai/prompts/` | Battle-tested prompts | Security-specific system prompts |
| `src/cai/util.py` | Utilities | Helper functions |

### Key Code to Study:

```python
# From src/cai/sdk/agents/model/handoff.py (conceptual)
class Handoff:
    """Represents a transfer of control between agents."""
    
    def __init__(
        self,
        target_agent: str,
        reason: str,
        context_to_transfer: dict,
        tools_to_inherit: list[str]
    ):
        self.target = target_agent
        self.reason = reason
        self.context = context_to_transfer
        self.tools = tools_to_inherit
```

### Take These Agents:

1. **Red Team Agent:** For offensive security
2. **Bug Bounty Hunter:** For web vulnerability discovery
3. **One Tool Agent:** Simple agent pattern for quick tasks

---

## 4.2 From OpenManus (FoundationAgents/OpenManus)

**Repository:** `github.com/FoundationAgents/OpenManus`

### Take These Files:

| Path | What It Is | Use For |
|------|------------|---------|
| `app/agent/planning.py` | PlanningAgent | Task decomposition logic |
| `app/agent/toolcall.py` | ToolCallAgent | Tool invocation mechanism |
| `app/agent/manus.py` | Main orchestrator | High-level coordination |
| `app/tool/` | Tool implementations | Browser, file, shell tools |
| `app/flow/` | Flow definitions | Workflow patterns |

### Key Concepts:

```python
# OpenManus's PlanningAgent approach
class PlanningAgent:
    """Creates and manages execution plans."""
    
    async def create_plan(self, task: str) -> list[Step]:
        """Break task into steps."""
        pass
    
    async def track_progress(self, step_id: int, result: Any) -> None:
        """Update step status and handle dependencies."""
        pass
```

---

## 4.3 From OpenCode (opencode-ai/opencode)

**Repository:** `github.com/opencode-ai/opencode` (Go) or `github.com/sst/opencode`

### Take These Concepts:

1. **Build/Plan dual-agent pattern:** Separate planning from execution
2. **LSP integration:** Use Language Server Protocol for code intelligence
3. **Session management:** How to persist coding sessions
4. **Custom commands:** Template system for common operations

### Configuration Pattern:

```json
{
    "agents": {
        "coder": {
            "model": "local.nemotron",
            "systemPrompt": "path/to/prompt.md",
            "tools": ["read_file", "write_file", "run_command"]
        },
        "planner": {
            "model": "local.qwen",
            "systemPrompt": "path/to/planner.md",
            "tools": ["read_file", "list_directory"]
        }
    }
}
```

---

## 4.4 From LangGraph

**Package:** `pip install langgraph`

### Take These Patterns:

1. **StateGraph:** The core state machine implementation
2. **Checkpointing:** How to persist state between runs
3. **Conditional edges:** Dynamic routing based on state
4. **Human-in-the-loop:** Pausing for approval

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Persistent checkpointing
checkpointer = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=checkpointer)

# Now you can resume from any checkpoint
config = {"configurable": {"thread_id": "my_task_123"}}
result = app.invoke(state, config)
```

---

# Part 5: Putting It All Together — The Integration

## 5.1 Project Structure

```
multi_agent_framework/
├── agents/
│   ├── __init__.py
│   ├── base.py              # BaseAgent class
│   ├── coder.py             # CodingAgent (OpenCode-inspired)
│   ├── security.py          # SecurityAgent (CAI-inspired)
│   ├── researcher.py        # ResearchAgent (OpenManus-inspired)
│   └── orchestrator.py      # OrchestratorAgent (Nemotron)
│
├── tools/
│   ├── __init__.py
│   ├── registry.py          # Tool registry
│   ├── file_tools.py        # File operations
│   ├── shell_tools.py       # Command execution
│   ├── search_tools.py      # Web search
│   ├── security_tools.py    # Nmap, CVE search (from CAI)
│   └── browser_tools.py     # Web automation
│
├── workflow/
│   ├── __init__.py
│   ├── graph.py             # LangGraph workflow definition
│   ├── state.py             # State schema
│   └── nodes.py             # Node implementations
│
├── memory/
│   ├── __init__.py
│   ├── short_term.py        # Conversation history
│   ├── working.py           # Task state
│   └── long_term.py         # Vector DB (ChromaDB)
│
├── sandbox/
│   ├── __init__.py
│   ├── docker_sandbox.py    # Docker isolation
│   └── e2b_sandbox.py       # E2B integration
│
├── inference/
│   ├── __init__.py
│   ├── ollama.py            # Local inference
│   ├── api.py               # API providers
│   └── router.py            # Model routing
│
├── ui/
│   ├── cli.py               # Terminal interface
│   ├── web/
│   │   ├── backend.py       # FastAPI
│   │   └── frontend/        # React app
│   └── api.py               # REST API
│
├── config/
│   ├── agents.yaml          # Agent configurations
│   ├── tools.yaml           # Tool configurations
│   └── models.yaml          # Model routing config
│
├── main.py                  # Entry point
└── requirements.txt
```

## 5.2 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                  │
│                     "Build a vulnerability scanner"                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         1. UI LAYER                                      │
│  • Receives input                                                        │
│  • Creates task ID                                                       │
│  • Initializes state in SQLite                                          │
│  • Sends to workflow                                                     │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    2. LANGGRAPH WORKFLOW                                 │
│  • Loads/creates state                                                   │
│  • Routes to orchestrator node                                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              3. ORCHESTRATOR (Nemotron-Orchestrator-8B)                  │
│                                                                          │
│  Input: Task + Current State                                             │
│                                                                          │
│  Process:                                                                │
│  • Queries long-term memory (ChromaDB) for similar past tasks           │
│  • Analyzes task requirements                                            │
│  • Decomposes into steps:                                               │
│    Step 0: Research existing vulnerability scanners (researcher)         │
│    Step 1: Design scanner architecture (coder)                          │
│    Step 2: Implement port scanning module (coder)                       │
│    Step 3: Implement CVE lookup module (security)                       │
│    Step 4: Integrate and test (coder)                                   │
│                                                                          │
│  Output: JSON plan + next step to execute                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  4. CONDITIONAL ROUTING                                  │
│  • Reads next step from plan                                            │
│  • Routes to appropriate agent node                                     │
│  • Step 0 → researcher node                                             │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    5. AGENT EXECUTION                                    │
│                                                                          │
│  Agent: ResearchAgent                                                    │
│  Task: "Research existing vulnerability scanners"                        │
│                                                                          │
│  ReAct Loop:                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Thought: I should search for popular open-source scanners          │ │
│  │ Action: web_search                                                  │ │
│  │ Action Input: {"query": "best open source vulnerability scanners"} │ │
│  │ Observation: [search results...]                                    │ │
│  │                                                                      │ │
│  │ Thought: Nmap, OpenVAS, and Nuclei are popular. Let me get details │ │
│  │ Action: web_search                                                  │ │
│  │ Action Input: {"query": "nmap vs nuclei vulnerability scanning"}   │ │
│  │ Observation: [comparison results...]                                │ │
│  │                                                                      │ │
│  │ Thought: I have enough info. Creating summary.                     │ │
│  │ Action: FINISH                                                      │ │
│  │ Output: {detailed research summary}                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Result: AgentResult(success=True, output=summary, artifacts=[])        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    6. STATE UPDATE                                       │
│  • Step result saved to state.step_results                              │
│  • Working memory updated with research findings                        │
│  • Research saved to long-term memory (ChromaDB)                        │
│  • State checkpointed (can resume from here if crash)                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 7. BACK TO ORCHESTRATOR                                  │
│  • Sees Step 0 complete                                                 │
│  • Advances to Step 1 (design architecture)                             │
│  • Routes to coder node                                                 │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    8. CODER AGENT                                        │
│                                                                          │
│  Agent: CodingAgent                                                      │
│  Task: "Design scanner architecture"                                     │
│  Context: Research from Step 0 is in working memory                     │
│                                                                          │
│  ReAct Loop:                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Thought: Based on research, I'll use a modular design with:        │ │
│  │   - Core scanner engine                                             │ │
│  │   - Plugin system for different scan types                          │ │
│  │   - CVE database integration                                        │ │
│  │                                                                      │ │
│  │ Action: write_file                                                  │ │
│  │ Action Input: {                                                     │ │
│  │   "path": "scanner/architecture.md",                                │ │
│  │   "content": "# Scanner Architecture\n..."                          │ │
│  │ }                                                                   │ │
│  │ Observation: Written 2048 bytes to scanner/architecture.md         │ │
│  │                                                                      │ │
│  │ Thought: Now creating the module structure                          │ │
│  │ Action: write_file                                                  │ │
│  │ Action Input: {"path": "scanner/__init__.py", "content": "..."}    │ │
│  │ ...                                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  │ [Loop continues through all steps]
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    9. SYNTHESIZER NODE                                   │
│  • All steps complete                                                   │
│  • Collects all artifacts (files created)                               │
│  • Generates final summary                                              │
│  • Saves to long-term memory for future reference                       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       10. OUTPUT TO USER                                 │
│                                                                          │
│  "I've created a vulnerability scanner with the following structure:    │
│   - scanner/core.py: Main scanning engine                               │
│   - scanner/plugins/port_scan.py: Port scanning module                  │
│   - scanner/plugins/cve_check.py: CVE lookup integration                │
│   - scanner/cli.py: Command-line interface                              │
│                                                                          │
│   To use: python -m scanner --target 192.168.1.1                        │
│                                                                          │
│   Files are in /workspace/scanner/"                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5.3 The Handoff Mechanism in Detail

This is your core concern, so let's make it crystal clear.

### What a Handoff Contains

```python
@dataclass
class Handoff:
    """Complete handoff specification."""
    
    # Who is giving and receiving
    from_agent: str
    to_agent: str
    
    # Why the handoff is happening
    reason: str
    
    # The specific task for the receiving agent
    task: str
    
    # Everything the receiving agent needs to know
    context: HandoffContext

@dataclass
class HandoffContext:
    """All context needed by the receiving agent."""
    
    # Summary of what's been accomplished
    work_summary: str
    
    # Specific outputs from previous work
    artifacts: list[str]  # File paths, URLs, etc.
    
    # Key decisions that were made
    decisions: list[str]
    
    # Information that might be needed
    relevant_data: dict
    
    # The full conversation history (optional, can be large)
    conversation_history: list[dict]
    
    # Pointers to long-term memory entries
    memory_references: list[str]
```

### How Handoffs Flow Through LangGraph

```python
def coder_node(state: AgentState) -> dict:
    """Coder agent that might hand off to security agent."""
    
    # ... do coding work ...
    
    # Agent detects it needs security expertise
    if needs_security_review(result):
        # Create handoff
        handoff = Handoff(
            from_agent="coder",
            to_agent="security",
            reason="Code complete, needs security review before deployment",
            task="Review the scanner code for security vulnerabilities",
            context=HandoffContext(
                work_summary="Implemented port scanner with CVE lookup",
                artifacts=["scanner/core.py", "scanner/plugins/"],
                decisions=[
                    "Used nmap library for port scanning",
                    "Integrated with NVD API for CVE data"
                ],
                relevant_data={
                    "scan_techniques": ["TCP SYN", "UDP"],
                    "cve_sources": ["NVD", "ExploitDB"]
                },
                conversation_history=state['conversation_history'][-10:],
                memory_references=[]
            )
        )
        
        return {
            "pending_handoff": handoff.to_dict(),
            "step_results": [...],
        }
    
    # Normal completion
    return {"step_results": [...]}

def handle_handoff(state: AgentState) -> str:
    """Routing function that checks for handoffs."""
    
    if state.get('pending_handoff'):
        handoff = state['pending_handoff']
        
        # Update state with handoff context
        state['handoff_context'] = handoff['context']
        
        # Route to receiving agent
        return handoff['to_agent']
    
    # No handoff, continue normal flow
    return "orchestrator"
```

### Visualizing Handoffs

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HANDOFF SEQUENCE                               │
│                                                                       │
│  CoderAgent                          SecurityAgent                    │
│  ┌─────────┐                         ┌─────────┐                     │
│  │         │                         │         │                     │
│  │  Code   │ ─────HANDOFF───────────▶│ Review  │                     │
│  │         │                         │         │                     │
│  └─────────┘                         └─────────┘                     │
│                                                                       │
│  Handoff contains:                                                    │
│  ├── reason: "Needs security review"                                 │
│  ├── task: "Review for vulnerabilities"                              │
│  └── context:                                                         │
│      ├── artifacts: [file1.py, file2.py]                             │
│      ├── decisions: ["Used X library", "Chose Y approach"]           │
│      └── relevant_data: {config, dependencies}                       │
│                                                                       │
│  SecurityAgent receives FULL context, can:                           │
│  • Read the files that were created                                   │
│  • Understand WHY certain decisions were made                        │
│  • Continue the conversation naturally                                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

# Part 6: Implementation Roadmap

## Phase 1: Foundation (Week 1)

1. **Set up project structure**
2. **Implement inference layer**
   - Ollama integration for local models
   - API integration for fallback
   - Model router
3. **Implement basic tools**
   - File operations
   - Shell execution
   - Web search
4. **Create base agent class**

## Phase 2: Core Agents (Week 2)

1. **Implement CodingAgent** (from OpenCode patterns)
2. **Implement SecurityAgent** (from CAI)
3. **Implement ResearchAgent** (from OpenManus)
4. **Implement OrchestratorAgent** (Nemotron)

## Phase 3: Workflow Engine (Week 3)

1. **Set up LangGraph**
2. **Implement state schema**
3. **Create workflow nodes**
4. **Implement handoff mechanism**
5. **Add checkpointing**

## Phase 4: Memory & Persistence (Week 4)

1. **Set up SQLite for state**
2. **Set up ChromaDB for vector memory**
3. **Implement memory retrieval in agents**

## Phase 5: Sandbox & Security (Week 5)

1. **Docker sandbox implementation**
2. **Tool permission system**
3. **Input validation**

## Phase 6: UI (Week 6)

1. **CLI interface**
2. **Web UI (FastAPI + React)**
3. **Streaming output**

---

# Part 7: Key Files to Create

I'll provide the core implementation files. Would you like me to continue with actual code implementations for:

1. **The main workflow graph** (`workflow/graph.py`)
2. **The orchestrator agent** (`agents/orchestrator.py`)
3. **The handoff system** (`workflow/handoff.py`)
4. **The tool registry** (`tools/registry.py`)
5. **The inference router** (`inference/router.py`)

Let me know which pieces you want me to implement first, or if you want me to create the complete starter codebase.

---

# Summary: The Key Insights

1. **Your current system is dumb because** it's single-turn (no ReAct loop) and has no state continuity between steps.

2. **Intelligence comes from** the ReAct loop (think → act → observe → think), proper handoffs (full context transfer), and state management (knowing where you are).

3. **The stack is:**
   - Persistence: SQLite + ChromaDB
   - Inference: Ollama (local) + API (fallback)
   - Sandbox: Docker/E2B
   - Tools: File, shell, search, security
   - Agents: Specialized workers
   - Workflow: LangGraph orchestration
   - Orchestrator: Nemotron for planning
   - UI: CLI/Web

4. **From each project, take:**
   - **CAI:** Security agents, tools, handoff patterns
   - **OpenManus:** Planning agent, flow definitions
   - **OpenCode:** Coding patterns, session management
   - **LangGraph:** State machine, checkpointing

5. **The handoff is the key:** Pass not just data, but understanding — work summary, decisions made, relevant context.

Would you like me to proceed with creating the actual implementation files?
