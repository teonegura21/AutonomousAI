# Complete MVP Roadmap - Multi-Agent Orchestration System

This plan includes:
- 7 Critical Issues from feedback (3.5/10 rating)
- DYNAMIC MODEL MANAGEMENT (Modularity)
- Full architecture from agentic_architecture.md
- 5 Strict Protocols

---

## PART A: CRITICAL FIXES + DYNAMIC MODULARITY

### A1-A7: Previous Fixes (Memory, Coordination, Synthesis, Tools, Intent)

See previous plan sections - all remain unchanged.

---

### A8: DYNAMIC MODEL MANAGEMENT (NEW - Modularity Requirement)

**Requirement:** System is MODULAR. When new model installed:
1. Auto-detect via Ollama API
2. Auto-update database
3. Assess model capabilities
4. Orchestrator can swap to better model dynamically

#### A8.1 Model Auto-Discovery

**File: `src/core/model_discovery.py`**
```python
import ollama
import sqlite3
from datetime import datetime

class ModelDiscovery:
    """Auto-detect new Ollama models and update registry"""
    
    def scan_ollama_models(self) -> list:
        """Get all models from Ollama"""
        models = ollama.list()
        return [m['name'] for m in models.get('models', [])]
    
    def get_registered_models(self) -> list:
        """Get models already in our DB"""
        conn = sqlite3.connect("agent_registry.db")
        cursor = conn.execute("SELECT model_name FROM agents")
        registered = [row[0] for row in cursor.fetchall()]
        conn.close()
        return registered
    
    def discover_new_models(self) -> list:
        """Find models in Ollama not yet in registry"""
        ollama_models = self.scan_ollama_models()
        registered = self.get_registered_models()
        return [m for m in ollama_models if m not in registered]
    
    def auto_register_model(self, model_name: str):
        """Auto-register with capability assessment"""
        capabilities = self.assess_capabilities(model_name)
        self.register_to_db(model_name, capabilities)
        print(f"[MODEL] Registered: {model_name}")
        print(f"[CAPS] {capabilities}")
    
    def assess_capabilities(self, model_name: str) -> dict:
        """Benchmark model to determine capabilities"""
        coding_score = self._test_coding(model_name)
        reasoning_score = self._test_reasoning(model_name)
        speed = self._test_speed(model_name)
        
        return {
            "coding": coding_score,
            "reasoning": reasoning_score,
            "speed_tokens_sec": speed,
            "assessed_at": datetime.now().isoformat()
        }
    
    def _test_coding(self, model_name: str) -> float:
        """Test code generation (0-100)"""
        test = "Write a Python function to check if a number is prime"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": test}])
        result = response['message']['content']
        score = 0
        if "```python" in result: score += 30
        if "def " in result: score += 20
        if "return" in result: score += 20
        if "% 2" in result or "// 2" in result: score += 30
        return min(score, 100)
    
    def _test_speed(self, model_name: str) -> float:
        """Measure tokens per second"""
        import time
        start = time.time()
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": "Count to 50"}])
        elapsed = time.time() - start
        tokens = len(response['message']['content'].split())
        return tokens / elapsed if elapsed > 0 else 0
```

#### A8.2 Dynamic Model Selection

**File: `src/core/model_selector.py`**
```python
class DynamicModelSelector:
    """Select best model for task based on capabilities"""
    
    def __init__(self, registry):
        self.registry = registry
    
    def select_best_model(self, task_type: str, constraints: dict = None) -> str:
        """
        Select optimal model for task type.
        
        Args:
            task_type: "coding", "reasoning", "documentation"
            constraints: {"max_vram": 4.0, "min_speed": 50}
        """
        agents = self.registry.get_all_agents()
        
        # Filter by constraints
        if constraints:
            if "max_vram" in constraints:
                agents = [a for a in agents if a.vram_required <= constraints["max_vram"]]
            if "min_speed" in constraints:
                agents = [a for a in agents if a.speed_tokens_per_sec >= constraints["min_speed"]]
        
        # Score by task type
        if task_type == "coding":
            agents.sort(key=lambda a: a.quality_score if "code_generation" in a.capabilities else 0, reverse=True)
        elif task_type == "documentation":
            agents.sort(key=lambda a: a.quality_score if "text_generation" in a.capabilities else 0, reverse=True)
        elif task_type == "fast":
            agents.sort(key=lambda a: a.speed_tokens_per_sec, reverse=True)
        
        return agents[0].model_name if agents else None
    
    def should_replace(self, current_model: str, new_model: str, task_type: str) -> bool:
        """Check if new model is better than current"""
        current = self.registry.get_agent_by_model(current_model)
        new = self.registry.get_agent_by_model(new_model)
        return new.quality_score > current.quality_score if new else False
```

#### A8.3 Model Watcher (Background Auto-Sync)

**File: `src/core/model_watcher.py`**
```python
import threading
import time

class ModelWatcher:
    """Watch for new Ollama models and auto-register"""
    
    def __init__(self, discovery, interval=60):
        self.discovery = discovery
        self.interval = interval
        self.running = False
    
    def start(self):
        """Start watching in background"""
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        print("[WATCHER] Started - auto-detecting new models")
    
    def _watch_loop(self):
        while self.running:
            new_models = self.discovery.discover_new_models()
            for model in new_models:
                print(f"[WATCHER] New model: {model}")
                self.discovery.auto_register_model(model)
            time.sleep(self.interval)
    
    def scan_now(self):
        """Manual scan"""
        new_models = self.discovery.discover_new_models()
        for model in new_models:
            self.discovery.auto_register_model(model)
        return new_models
```

#### A8.4 Orchestrator Integration

**Modify: `src/orchestration/nemotron_orchestrator.py`**
```python
class NemotronOrchestrator:
    def __init__(self):
        self.registry = AgentRegistry()
        self.model_discovery = ModelDiscovery()
        self.model_selector = DynamicModelSelector(self.registry)
        self.model_watcher = ModelWatcher(self.model_discovery)
        
        # Auto-start model watcher
        self.model_watcher.start()
        
        # Sync on startup
        self._sync_models()
    
    def _sync_models(self):
        """Sync Ollama models with registry"""
        new = self.model_watcher.scan_now()
        if new:
            print(f"[STARTUP] Discovered {len(new)} new models")
    
    def select_agent_for_task(self, task: dict) -> str:
        """DYNAMIC selection - picks best model for task"""
        task_type = task.get('required_capability', 'coding')
        constraints = task.get('constraints', {})
        
        best = self.model_selector.select_best_model(task_type, constraints)
        if best:
            agent = self.registry.get_agent_by_model(best)
            return agent.id if agent else "coder_qwen"
        return "coder_qwen"
```

#### A8.5 Database Schema for Capabilities

```sql
-- Model capabilities tracking
CREATE TABLE model_capabilities (
    model_name TEXT PRIMARY KEY,
    coding_score REAL,
    reasoning_score REAL,
    documentation_score REAL,
    speed_tokens_sec REAL,
    vram_gb REAL,
    assessed_at TEXT,
    is_active BOOLEAN DEFAULT 1
);

-- Model comparison history (which model won)
CREATE TABLE model_comparisons (
    id INTEGER PRIMARY KEY,
    model_a TEXT,
    model_b TEXT,
    task_type TEXT,
    winner TEXT,
    compared_at TEXT
);
```

#### A8.6 CLI Commands

```bash
# Scan for new models
python run_orchestrator.py --scan-models

# List all registered models with scores
python run_orchestrator.py --list-models

# Benchmark specific model
python run_orchestrator.py --benchmark qwen3:1.7b

# Run with specific model override
python run_orchestrator.py --model qwen3:1.7b "Your goal here"
```

---

## PART B: FILES TO CREATE (Complete List)

### Priority 1 (Immediate)
| File | Action | Purpose |
|------|--------|---------|
| src/memory/__init__.py | CREATE | Package init |
| src/memory/task_memory.py | CREATE | Inter-task context |
| src/tools/__init__.py | CREATE | Package init |
| src/tools/code_executor.py | CREATE | Extract and run code |
| src/core/model_discovery.py | CREATE | Auto-detect models |
| src/core/model_selector.py | CREATE | Dynamic selection |
| src/orchestration/intent_analyzer.py | CREATE | Goal analysis |
| src/orchestration/nemotron_orchestrator.py | MODIFY | Add all integrations |

### Priority 2 (Next Session)
| File | Action | Purpose |
|------|--------|---------|
| src/core/model_watcher.py | CREATE | Background sync |
| src/memory/vector_store.py | CREATE | ChromaDB |
| src/memory/structured_store.py | CREATE | SQLite history |
| src/orchestration/state_guards.py | CREATE | Hard guards |
| run_orchestrator.py | MODIFY | Add CLI commands |

### Priority 3 (Later)
| File | Action | Purpose |
|------|--------|---------|
| src/sandbox/docker_executor.py | CREATE | Isolation |
| src/orchestration/langgraph_workflow.py | CREATE | Full workflow |
| src/orchestration/human_checkpoint.py | CREATE | Approvals |

---

## PART C: FLOW DIAGRAM

```
USER INSTALLS NEW MODEL (ollama pull xyz)
          |
          v
   MODEL WATCHER (background)
          |
          v
   DETECT NEW MODEL
          |
          v
   ASSESS CAPABILITIES
   - Test coding (0-100)
   - Test reasoning (0-100)
   - Measure speed (tokens/sec)
          |
          v
   UPDATE DATABASE
   - Insert into agents table
   - Insert into model_capabilities
          |
          v
   ORCHESTRATOR USES BEST MODEL
   - Query: "Need coding task"
   - Select: highest coding_score
   - Execute with best available
```

---

## PART D: SUCCESS CRITERIA

### Dynamic Model Test
1. Run: `ollama pull phi3:mini`
2. System auto-detects within 60 seconds
3. System benchmarks phi3:mini
4. Database shows new model with scores
5. Next task considers phi3:mini for selection

### Full MVP Test
```bash
python run_orchestrator.py "Write ONE Python file that searches 
Desktop for password.txt and sends via UDP to 192.168.156.4.2"
```

Expected:
- Intent analysis detects single-file requirement
- Dynamically selects best coding model
- Tasks share context via TaskMemory
- Synthesis combines into final_output.py
- Code validated and saved

---

## PART E: IMPLEMENTATION ORDER

### Session 1 (Now)
1. src/core/model_discovery.py
2. src/core/model_selector.py  
3. src/memory/task_memory.py
4. src/tools/code_executor.py
5. Update nemotron_orchestrator.py

### Session 2
6. src/core/model_watcher.py
7. src/orchestration/intent_analyzer.py
8. CLI commands in run_orchestrator.py

### Session 3
9. src/memory/vector_store.py
10. src/orchestration/state_guards.py

---

## PART F: CRITICAL MISSING COMPONENTS (From Feedback)

### F1. Agent Framework / Inter-Agent Communication

**File: `src/orchestration/agent_messaging.py`**
```python
from typing import Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue

@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str  # "*" for broadcast
    content: dict
    message_type: str  # "question", "response", "broadcast", "notification"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class AgentMessageBus:
    """
    Agents communicate via messages, not just sequential execution.
    This enables collaboration: "What coordinate system did you choose?"
    """
    
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}  # agent_id -> message queue
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> [agent_ids]
        self.message_history: List[AgentMessage] = []
        self.lock = threading.Lock()
    
    def register_agent(self, agent_id: str):
        """Register agent to receive messages"""
        with self.lock:
            self.queues[agent_id] = queue.Queue()
    
    def send_message(self, from_agent: str, to_agent: str, content: dict, msg_type: str = "message"):
        """Send direct message to another agent"""
        msg = AgentMessage(from_agent, to_agent, content, msg_type)
        self.message_history.append(msg)
        
        if to_agent in self.queues:
            self.queues[to_agent].put(msg)
            print(f"[MSG] {from_agent} -> {to_agent}: {msg_type}")
    
    def broadcast(self, from_agent: str, topic: str, content: dict):
        """Broadcast to all subscribers of a topic"""
        msg = AgentMessage(from_agent, "*", content, "broadcast")
        self.message_history.append(msg)
        
        subscribers = self.subscriptions.get(topic, [])
        for agent_id in subscribers:
            if agent_id != from_agent and agent_id in self.queues:
                self.queues[agent_id].put(msg)
        print(f"[BROADCAST] {from_agent} -> {topic}: {len(subscribers)} recipients")
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        if agent_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(agent_id)
    
    def query_agent(self, asking_agent: str, target_agent: str, question: str, timeout: float = 30.0) -> str:
        """Ask another agent a question and wait for response"""
        self.send_message(asking_agent, target_agent, {"question": question}, "question")
        
        # Wait for response
        try:
            response = self.queues[asking_agent].get(timeout=timeout)
            if response.message_type == "response":
                return response.content.get("answer", "")
        except queue.Empty:
            return "[No response - timeout]"
    
    def get_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get all pending messages for an agent"""
        messages = []
        while not self.queues.get(agent_id, queue.Queue()).empty():
            messages.append(self.queues[agent_id].get_nowait())
        return messages
```

---

### F2. LangGraph Workflow Engine

**File: `src/orchestration/langgraph_workflow.py`**
```python
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import json

class WorkflowState(TypedDict):
    """State passed through the workflow"""
    task_id: str
    task_description: str
    assigned_agent: str
    tools: List[str]
    dependencies: List[str]
    status: str  # pending, running, completed, failed
    output: str
    error: str
    context: Dict[str, Any]  # Previous task outputs

class MultiAgentWorkflow:
    """
    LangGraph-based workflow engine.
    This is Layer 2 of the architecture (Execution Controller).
    """
    
    def __init__(self, orchestrator, db_path: str = "data/checkpoints.db"):
        self.orchestrator = orchestrator
        self.checkpointer = SqliteSaver.from_conn_string(db_path)
        self.graph = None
    
    def create_workflow(self, tasks: List[Dict]) -> StateGraph:
        """Convert Nemotron's task DAG into LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(WorkflowState)
        
        # Add node for each task
        for task in tasks:
            task_id = task["id"]
            workflow.add_node(task_id, self._create_task_node(task))
        
        # Add edges based on dependencies
        entry_tasks = []
        for task in tasks:
            deps = task.get("dependencies", [])
            if not deps:
                entry_tasks.append(task["id"])
            else:
                for dep in deps:
                    workflow.add_edge(dep, task["id"])
        
        # Set entry points
        for entry in entry_tasks:
            workflow.set_entry_point(entry)
        
        # Find final tasks (no task depends on them)
        all_deps = set()
        for task in tasks:
            all_deps.update(task.get("dependencies", []))
        final_tasks = [t["id"] for t in tasks if t["id"] not in all_deps or t.get("type") == "synthesis"]
        
        for final in final_tasks:
            workflow.add_edge(final, END)
        
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        return self.graph
    
    def _create_task_node(self, task: Dict):
        """Create a node function for a task"""
        def task_node(state: WorkflowState) -> WorkflowState:
            print(f"[WORKFLOW] Executing: {task['id']}")
            
            # Execute task via orchestrator
            result = self.orchestrator.execute_single_task(task, state.get("context", {}))
            
            return {
                **state,
                "task_id": task["id"],
                "status": "completed" if result["success"] else "failed",
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "context": {**state.get("context", {}), task["id"]: result}
            }
        return task_node
    
    def run(self, initial_state: Dict = None) -> Dict:
        """Execute the workflow"""
        if not self.graph:
            raise ValueError("Workflow not created. Call create_workflow first.")
        
        state = initial_state or {"context": {}, "status": "pending"}
        
        # Run with checkpointing
        config = {"configurable": {"thread_id": "main"}}
        result = self.graph.invoke(state, config)
        
        return result
    
    def resume(self, thread_id: str = "main") -> Dict:
        """Resume workflow from checkpoint"""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.checkpointer.get(config)
        if state:
            return self.graph.invoke(state, config)
        return None
```

---

### F3. Synthesis Logic (AUTO-ADD)

**Add to: `src/orchestration/nemotron_orchestrator.py`**
```python
def add_synthesis_task(self, tasks: list, user_requirements: dict) -> list:
    """
    ALWAYS add final synthesis if multiple tasks.
    This fixes the "4 files instead of 1" problem.
    """
    if len(tasks) <= 1:
        return tasks
    
    # Check if synthesis needed
    needs_synthesis = (
        user_requirements.get("single_file") or
        user_requirements.get("combined_output") or
        len(tasks) > 2
    )
    
    if needs_synthesis:
        synthesis_task = {
            "id": f"task_{len(tasks)+1}_synthesis",
            "description": f"""FINAL SYNTHESIS TASK:
Combine ALL outputs from previous tasks into ONE deliverable.

Previous tasks: {[t['id'] for t in tasks]}

Requirements:
1. Read all code from previous tasks
2. Merge into single, complete file
3. Remove duplicate imports
4. Ensure no conflicts
5. Add proper main() if needed
6. Output ONLY the final combined code

Output format: {user_requirements.get('output_format', 'Python file')}""",
            "assigned_agent": "coder_qwen",
            "tools": ["filesystem_read", "filesystem_write", "python_exec"],
            "dependencies": [t["id"] for t in tasks],
            "type": "synthesis"
        }
        tasks.append(synthesis_task)
        print(f"[SYNTHESIS] Added final synthesis task")
    
    return tasks
```

---

### F4. Vector DB (Complete Implementation)

**File: `src/memory/vector_store.py`**
```python
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Any
import ollama

class VectorMemoryStore:
    """
    ChromaDB integration with semantic search.
    Stores ALL work: code, decisions, conversations.
    Agents can query: "What coordinate system did we choose?"
    """
    
    def __init__(self, persist_dir: str = "data/chromadb", embedding_model: str = "nomic-embed-text"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_model = embedding_model
        
        # Create collections
        self.code_collection = self.client.get_or_create_collection(
            name="code_artifacts",
            metadata={"description": "Code written by agents"}
        )
        self.decision_collection = self.client.get_or_create_collection(
            name="decisions",
            metadata={"description": "Design decisions and rationale"}
        )
        self.conversation_collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Agent conversations and outputs"}
        )
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]
    
    def store_code(self, task_id: str, code: str, metadata: dict = None):
        """Store code artifact with embeddings"""
        self.code_collection.add(
            ids=[f"code_{task_id}_{datetime.now().timestamp()}"],
            documents=[code],
            metadatas=[{
                "task_id": task_id,
                "type": "code",
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }],
            embeddings=[self._get_embedding(code)]
        )
    
    def store_decision(self, task_id: str, decision: str, rationale: str):
        """Store design decision"""
        content = f"Decision: {decision}\nRationale: {rationale}"
        self.decision_collection.add(
            ids=[f"decision_{task_id}_{datetime.now().timestamp()}"],
            documents=[content],
            metadatas=[
                {"task_id": task_id, "decision": decision, "timestamp": datetime.now().isoformat()}
            ],
            embeddings=[self._get_embedding(content)]
        )
    
    def store_conversation(self, task_id: str, agent_id: str, content: str):
        """Store agent conversation/output"""
        self.conversation_collection.add(
            ids=[f"conv_{task_id}_{agent_id}_{datetime.now().timestamp()}"],
            documents=[content],
            metadatas=[
                {"task_id": task_id, "agent_id": agent_id, "timestamp": datetime.now().isoformat()}
            ],
            embeddings=[self._get_embedding(content)]
        )
    
    def semantic_search(self, query: str, collection: str = "all", n_results: int = 5) -> List[Dict]:
        """Search by meaning across collections"""
        results = []
        query_embedding = self._get_embedding(query)
        
        collections = {
            "code": self.code_collection,
            "decisions": self.decision_collection,
            "conversations": self.conversation_collection
        }
        
        if collection == "all":
            search_collections = collections.values()
        else:
            search_collections = [collections.get(collection)]
        
        for coll in search_collections:
            if coll:
                res = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                for i, doc in enumerate(res["documents"][0]):
                    results.append({
                        "content": doc,
                        "metadata": res["metadatas"][0][i],
                        "distance": res["distances"][0][i] if "distances" in res else None
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x.get("distance", 0))
        return results[:n_results]
    
    def get_task_context(self, task_id: str) -> Dict[str, Any]:
        """Get all artifacts from a specific task"""
        return {
            "code": self.code_collection.get(where={"task_id": task_id}),
            "decisions": self.decision_collection.get(where={"task_id": task_id}),
            "conversations": self.conversation_collection.get(where={"task_id": task_id})
        }
    
    def query_natural(self, question: str) -> str:
        """Answer natural language questions about stored knowledge"""
        results = self.semantic_search(question, n_results=3)
        if results:
            context = "\n\n".join([r["content"][:500] for r in results])
            return f"Based on stored knowledge:\n{context}"
        return "No relevant information found."
```

---

### F5. Error Recovery & Retry Logic

**File: `src/orchestration/error_recovery.py`**
```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RecoveryAction(Enum):
    RETRY = "retry"
    RETRY_WITH_DIFFERENT_MODEL = "retry_different_model"
    SPAWN_DEBUGGER = "spawn_debugger"
    SIMPLIFY_TASK = "simplify_task"
    ESCALATE_TO_HUMAN = "escalate_human"
    SKIP = "skip"
    ABORT = "abort"

@dataclass
class RecoveryDecision:
    action: RecoveryAction
    params: Dict[str, Any]
    reason: str

class ErrorRecovery:
    """
    Handle agent failures gracefully.
    Decides what to do when tasks fail.
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = {}  # task_id -> retry count
    
    def handle_failure(self, task: Dict, error: str, context: Dict = None) -> RecoveryDecision:
        """Decide recovery action based on error type"""
        task_id = task.get("id", "unknown")
        error_lower = error.lower()
        
        # Track retries
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1
        retries = self.retry_counts[task_id]
        
        # Max retries exceeded
        if retries >= self.max_retries:
            return RecoveryDecision(
                action=RecoveryAction.ESCALATE_TO_HUMAN,
                params={"task": task, "error": error, "retries": retries},
                reason=f"Max retries ({self.max_retries}) exceeded"
            )
        
        # Timeout errors
        if "timeout" in error_lower or "timed out" in error_lower:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={"timeout_multiplier": 2.0},
                reason="Timeout - retrying with longer timeout"
            )
        
        # Compilation/Syntax errors
        if "syntax" in error_lower or "compile" in error_lower or "parse" in error_lower:
            return RecoveryDecision(
                action=RecoveryAction.SPAWN_DEBUGGER,
                params={"error_log": error, "code": context.get("code", "")},
                reason="Syntax error - spawning debugger agent"
            )
        
        # Out of memory
        if "memory" in error_lower or "oom" in error_lower or "cuda" in error_lower:
            return RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_DIFFERENT_MODEL,
                params={"prefer_smaller": True},
                reason="Out of memory - switching to smaller model"
            )
        
        # Import/Module errors
        if "import" in error_lower or "module" in error_lower:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={"install_deps": True},
                reason="Missing module - installing dependencies and retrying"
            )
        
        # Rate limiting
        if "rate limit" in error_lower or "429" in error_lower:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={"delay_seconds": 60},
                reason="Rate limited - waiting and retrying"
            )
        
        # Generic retry for first failure
        if retries == 1:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                params={},
                reason=f"First failure - retrying (attempt {retries + 1})"
            )
        
        # Unknown error after retries
        return RecoveryDecision(
            action=RecoveryAction.ESCALATE_TO_HUMAN,
            params={"task": task, "error": error},
            reason="Unknown error - escalating to human"
        )
    
    def reset_retries(self, task_id: str):
        """Reset retry count for a task (on success)"""
        self.retry_counts[task_id] = 0
```

---

### F6. Human-in-the-Loop Checkpoints

**File: `src/orchestration/human_checkpoint.py`**
```python
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

class CheckpointDecision(Enum):
    APPROVE = "approved"
    REJECT = "rejected"
    MODIFY = "modified"
    SKIP = "skipped"

class HumanCheckpoint:
    """
    Pause execution for human review at critical points.
    User can approve, reject, or modify before continuing.
    """
    
    def __init__(self, db_path: str = "data/checkpoints.db"):
        self.db_path = db_path
        self.decisions: list = []
    
    def request_approval(self, task: Dict, output: str, checkpoint_type: str = "task_complete") -> Dict:
        """
        Ask user to approve before continuing.
        Returns decision and any modifications.
        """
        print(f"\n{'='*70}")
        print(f"⚠️  CHECKPOINT: {checkpoint_type.upper()}")
        print(f"{'='*70}")
        print(f"\nTask: {task.get('id', 'unknown')}")
        print(f"Description: {task.get('description', '')[:200]}...")
        print(f"\n--- OUTPUT PREVIEW ---")
        print(output[:1000] if len(output) > 1000 else output)
        if len(output) > 1000:
            print(f"\n... ({len(output) - 1000} more characters)")
        print(f"\n{'='*70}")
        print("[A]pprove  [R]eject  [M]odify  [V]iew Full  [S]kip")
        
        while True:
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'a':
                decision = self._record_decision(task, CheckpointDecision.APPROVE)
                return {"approved": True, "decision": decision}
            
            elif choice == 'r':
                reason = input("Rejection reason: ").strip()
                decision = self._record_decision(task, CheckpointDecision.REJECT, reason)
                return {"approved": False, "decision": decision, "reason": reason}
            
            elif choice == 'm':
                print("Enter modifications (end with empty line):")
                modifications = []
                while True:
                    line = input()
                    if not line:
                        break
                    modifications.append(line)
                mod_text = "\n".join(modifications)
                decision = self._record_decision(task, CheckpointDecision.MODIFY, mod_text)
                return {"approved": True, "decision": decision, "modifications": mod_text}
            
            elif choice == 'v':
                print(f"\n--- FULL OUTPUT ---\n{output}\n--- END ---\n")
            
            elif choice == 's':
                decision = self._record_decision(task, CheckpointDecision.SKIP)
                return {"approved": True, "decision": decision, "skipped": True}
            
            else:
                print("Invalid choice. Use A/R/M/V/S")
    
    def _record_decision(self, task: Dict, decision: CheckpointDecision, feedback: str = "") -> Dict:
        """Record decision for audit trail"""
        record = {
            "task_id": task.get("id"),
            "checkpoint_type": "task_review",
            "decision": decision.value,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        self.decisions.append(record)
        return record
    
    def should_checkpoint(self, task: Dict, config: Dict = None) -> bool:
        """Determine if this task needs human review"""
        config = config or {}
        
        # Always checkpoint synthesis tasks
        if task.get("type") == "synthesis":
            return True
        
        # Checkpoint if explicitly required
        if task.get("requires_approval"):
            return True
        
        # Checkpoint critical operations
        critical_tools = ["bash_exec", "filesystem_write"]
        if any(tool in task.get("tools", []) for tool in critical_tools):
            return config.get("checkpoint_critical", False)
        
        return False
```

---

### F7. Testing Workflow

**File: `src/orchestration/testing_workflow.py`**
```python
from typing import Dict, List

class TestingWorkflow:
    """
    Add testing tasks after coding tasks.
    Testing agents validate coding agents' work.
    """
    
    def __init__(self, tester_agent: str = "coder_qwen"):
        self.tester_agent = tester_agent
    
    def add_testing_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """For each coding task, add corresponding test task"""
        enhanced_tasks = []
        
        for task in tasks:
            enhanced_tasks.append(task)
            
            # Add test for coding tasks
            if self._is_coding_task(task):
                test_task = self._create_test_task(task)
                enhanced_tasks.append(test_task)
        
        return enhanced_tasks
    
    def _is_coding_task(self, task: Dict) -> bool:
        """Check if task produces code"""
        coding_indicators = [
            "python_exec" in task.get("tools", []),
            "code" in task.get("description", "").lower(),
            "write" in task.get("description", "").lower(),
            "implement" in task.get("description", "").lower(),
            task.get("type") == "coding"
        ]
        return any(coding_indicators) and task.get("type") != "synthesis"
    
    def _create_test_task(self, coding_task: Dict) -> Dict:
        """Create test task for a coding task"""
        return {
            "id": f"{coding_task['id']}_test",
            "description": f"""TEST TASK for {coding_task['id']}:

1. Read the code from {coding_task['id']}
2. Generate pytest tests that verify:
   - Functions exist and are callable
   - Basic functionality works
   - Edge cases handled
3. Run the tests
4. Report pass/fail status

Output: Test results and any failures found.""",
            "assigned_agent": self.tester_agent,
            "tools": ["filesystem_read", "filesystem_write", "pytest_run"],
            "dependencies": [coding_task["id"]],
            "type": "testing"
        }
    
    def validate_output(self, task: Dict, output: str) -> Dict:
        """Validate that output meets requirements"""
        validations = {
            "has_code": "```" in output or "def " in output,
            "has_tests": "test_" in output.lower() or "assert" in output,
            "no_errors": "error" not in output.lower() or "passed" in output.lower()
        }
        
        return {
            "valid": all(validations.values()),
            "checks": validations
        }
```

---

### F8. Intent Analyzer (Deep)

**File: `src/orchestration/intent_analyzer.py`**
```python
from typing import Dict, List, Optional
import re

class IntentAnalyzer:
    """
    Deep analysis of user intent.
    Detects ambiguity, extracts requirements, asks clarifications.
    """
    
    # Known ambiguous patterns
    AMBIGUITY_PATTERNS = [
        {
            "pattern": r"buffer overflow.*python",
            "issue": "Buffer overflow is typically a C/C++ concept, not Python",
            "question": "Did you mean a data exfiltration/file stealing script in Python?"
        },
        {
            "pattern": r"hack|exploit",
            "issue": "'Hack/exploit' is ambiguous",
            "question": "Do you mean: (1) Security testing tool, (2) Automation script, or (3) CTF challenge solver?"
        },
        {
            "pattern": r"fast|quick|simple",
            "issue": "Performance vs simplicity trade-off unclear",
            "question": "Priority: (1) Fast execution, (2) Simple code, or (3) Both with trade-offs?"
        }
    ]
    
    def analyze(self, user_goal: str) -> Dict:
        """Deep analysis of user intent"""
        goal_lower = user_goal.lower()
        
        # Extract requirements
        requirements = {
            "single_file": self._detect_single_file(goal_lower),
            "language": self._detect_language(goal_lower),
            "platform": self._detect_platform(goal_lower),
            "output_format": self._detect_output_format(goal_lower),
            "constraints": self._extract_constraints(user_goal)
        }
        
        # Detect ambiguities
        ambiguities = self._detect_ambiguities(goal_lower)
        
        # Extract key entities
        entities = self._extract_entities(user_goal)
        
        return {
            "original_goal": user_goal,
            "requirements": requirements,
            "ambiguities": ambiguities,
            "entities": entities,
            "needs_clarification": len(ambiguities) > 0,
            "confidence": self._calculate_confidence(requirements, ambiguities)
        }
    
    def _detect_single_file(self, goal: str) -> bool:
        """Detect if user wants single file output"""
        patterns = ["one file", "single file", "single script", "one script", "combined", "all in one"]
        return any(p in goal for p in patterns)
    
    def _detect_language(self, goal: str) -> str:
        """Detect programming language"""
        languages = {
            "python": ["python", ".py", "pip"],
            "javascript": ["javascript", "js", "node", "npm"],
            "bash": ["bash", "shell", "sh", "script"],
            "c": ["c program", ".c ", "gcc"],
            "cpp": ["c++", "cpp", "g++"]
        }
        for lang, keywords in languages.items():
            if any(kw in goal for kw in keywords):
                return lang
        return "python"  # Default
    
    def _detect_platform(self, goal: str) -> str:
        """Detect target platform"""
        if any(w in goal for w in ["windows", "win32", ".exe", "powershell"]):
            return "windows"
        if any(w in goal for w in ["linux", "ubuntu", "debian"]):
            return "linux"
        if any(w in goal for w in ["mac", "macos", "darwin"]):
            return "macos"
        return "cross-platform"
    
    def _detect_output_format(self, goal: str) -> str:
        """Detect expected output format"""
        if "api" in goal or "rest" in goal:
            return "api"
        if "cli" in goal or "command" in goal:
            return "cli"
        if "gui" in goal or "ui" in goal:
            return "gui"
        if "library" in goal or "module" in goal:
            return "library"
        return "script"
    
    def _extract_constraints(self, goal: str) -> List[str]:
        """Extract explicit constraints"""
        constraints = []
        
        # Size constraints
        if match := re.search(r"(\d+)\s*(lines?|loc)", goal.lower()):
            constraints.append(f"max_lines:{match.group(1)}")
        
        # Time constraints
        if match := re.search(r"(\d+)\s*(seconds?|minutes?|ms)", goal.lower()):
            constraints.append(f"timeout:{match.group(1)}{match.group(2)}")
        
        # Dependency constraints
        if "no dependencies" in goal.lower() or "stdlib only" in goal.lower():
            constraints.append("stdlib_only")
        
        return constraints
    
    def _detect_ambiguities(self, goal: str) -> List[Dict]:
        """Find ambiguous patterns that need clarification"""
        ambiguities = []
        for pattern in self.AMBIGUITY_PATTERNS:
            if re.search(pattern["pattern"], goal, re.IGNORECASE):
                ambiguities.append({
                    "issue": pattern["issue"],
                    "question": pattern["question"]
                })
        return ambiguities
    
    def _extract_entities(self, goal: str) -> Dict:
        """Extract key entities (IPs, paths, etc.)"""
        entities = {}
        
        # IP addresses
        ips = re.findall(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", goal)
        if ips:
            entities["ip_addresses"] = ips
        
        # File paths
        paths = re.findall(r"[A-Za-z]:[\\][\w\\]+|/[\w/]+\.\w+", goal)
        if paths:
            entities["file_paths"] = paths
        
        # Ports
        ports = re.findall(r"port\s*(\d+)|:(\d{2,5})", goal.lower())
        if ports:
            entities["ports"] = [p[0] or p[1] for p in ports]
        
        return entities
    
    def _calculate_confidence(self, requirements: Dict, ambiguities: List) -> float:
        """Calculate confidence in understanding (0-1)"""
        score = 1.0
        
        # Reduce for ambiguities
        score -= len(ambiguities) * 0.2
        
        # Increase for clear requirements
        if requirements["single_file"]:
            score += 0.1
        if requirements["language"] != "python":  # Explicit language
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def clarify_with_user(self, ambiguities: List[Dict]) -> str:
        """Ask user for clarification"""
        print("\n⚠️  CLARIFICATION NEEDED:")
        for i, amb in enumerate(ambiguities, 1):
            print(f"\n{i}. {amb['issue']}")
            print(f"   → {amb['question']}")
        
        print("\nPlease clarify your goal:")
        clarification = input("> ").strip()
        return clarification
```

---

### F9. Monitoring / Telemetry

**File: `src/monitoring/telemetry.py`**
```python
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading

@dataclass
class ExecutionMetrics:
    task_id: str
    agent_id: str
    started_at: str
    completed_at: str = None
    duration_sec: float = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tools_used: List[str] = field(default_factory=list)
    success: bool = False
    error: str = None

class ExecutionMonitor:
    """
    Track what agents are doing in real-time.
    Provides dashboard data and logging.
    """
    
    def __init__(self, log_file: str = "logs/execution.jsonl"):
        self.log_file = log_file
        self.active_tasks: Dict[str, ExecutionMetrics] = {}
        self.completed_tasks: List[ExecutionMetrics] = []
        self.total_tokens = 0
        self.lock = threading.Lock()
    
    def log_task_start(self, task_id: str, agent_id: str, task_info: Dict = None):
        """Task started"""
        metrics = ExecutionMetrics(
            task_id=task_id,
            agent_id=agent_id,
            started_at=datetime.now().isoformat()
        )
        with self.lock:
            self.active_tasks[task_id] = metrics
        
        self._log_event("task_start", {"task_id": task_id, "agent_id": agent_id, **(task_info or {})})
        print(f"[MONITOR] Started: {task_id} on {agent_id}")
    
    def log_tool_execution(self, task_id: str, tool: str, duration: float, success: bool):
        """Tool was used"""
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].tools_used.append(tool)
        
        self._log_event("tool_exec", {"task_id": task_id, "tool": tool, "duration": duration, "success": success})
    
    def log_task_complete(self, task_id: str, success: bool, tokens: int = 0, error: str = None):
        """Task finished"""
        with self.lock:
            if task_id in self.active_tasks:
                metrics = self.active_tasks.pop(task_id)
                metrics.completed_at = datetime.now().isoformat()
                metrics.success = success
                metrics.error = error
                metrics.tokens_output = tokens
                
                # Calculate duration
                start = datetime.fromisoformat(metrics.started_at)
                end = datetime.fromisoformat(metrics.completed_at)
                metrics.duration_sec = (end - start).total_seconds()
                
                self.completed_tasks.append(metrics)
                self.total_tokens += tokens
        
        self._log_event("task_complete", {"task_id": task_id, "success": success, "error": error})
        print(f"[MONITOR] Completed: {task_id} ({'✓' if success else '✗'})")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Current system state for dashboard"""
        with self.lock:
            completed = len(self.completed_tasks)
            successful = sum(1 for t in self.completed_tasks if t.success)
            
            return {
                "active_tasks": list(self.active_tasks.keys()),
                "active_agents": list(set(t.agent_id for t in self.active_tasks.values())),
                "tasks_completed": completed,
                "tasks_successful": successful,
                "tasks_failed": completed - successful,
                "tasks_pending": len(self.active_tasks),
                "total_tokens": self.total_tokens,
                "estimated_cost": f"${self.total_tokens * 0.000001:.4f}",  # Rough estimate
                "avg_duration_sec": sum(t.duration_sec for t in self.completed_tasks) / max(completed, 1)
            }
    
    def _log_event(self, event_type: str, data: Dict):
        """Write event to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({
                    "event": event_type,
                    "timestamp": datetime.now().isoformat(),
                    **data
                }) + "\n")
        except:
            pass
```

---

### F10. Configuration Management

**File: `config/settings.yaml`**
```yaml
# AI_Autonom Configuration

orchestrator:
  model: "huihui_ai/orchestrator-abliterated"
  max_decomposition_depth: 3
  enable_human_checkpoints: true
  enable_testing: true
  single_file_default: false

agents:
  coder:
    model: "qwen3:1.7b"
    vram_gb: 1.4
    speed_tokens_sec: 70
  linguistic:
    model: "dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0"
    vram_gb: 1.8
    speed_tokens_sec: 50

execution:
  vram_limit_gb: 20
  max_parallel_agents: 3
  default_timeout_sec: 300
  max_retries: 3

memory:
  vector_db:
    type: "chromadb"
    embedding_model: "nomic-embed-text"
    persist_directory: "./data/chromadb"
  structured_db:
    type: "sqlite"
    path: "./data/agent_registry.db"

sandbox:
  enabled: true
  image: "python:3.11-slim"
  mem_limit: "2g"
  cpu_limit: 2.0
  network: "bridge"

tools:
  builtin: ["filesystem_read", "filesystem_write", "python_exec", "bash_exec", "pytest_run", "web_fetch"]
  enable_dynamic: true  # Agents can create tools

logging:
  level: "INFO"
  file: "./logs/orchestrator.log"
  format: "json"
  
checkpoints:
  enabled: true
  db_path: "./data/checkpoints.db"
```

**File: `src/core/config.py`**
```python
import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    """Central configuration management"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "config/settings.yaml"):
        """Load configuration from YAML file"""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._defaults()
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'orchestrator.model')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def _defaults(self) -> Dict:
        """Default configuration"""
        return {
            "orchestrator": {"model": "huihui_ai/orchestrator-abliterated"},
            "execution": {"max_retries": 3, "default_timeout_sec": 300},
            "sandbox": {"enabled": True}
        }

# Global config instance
config = Config()
```

---

### F11. Dynamic Tool Creation (Agents Create Tools)

**File: `src/tools/dynamic_tools.py`**
```python
from typing import Dict, Callable
from datetime import datetime

class DynamicToolRegistry:
    """
    Tools created by agents become available to other agents.
    Agent writes script → runs it → becomes callable tool.
    """
    
    def __init__(self, sandbox, tool_registry):
        self.sandbox = sandbox
        self.tool_registry = tool_registry
        self.agent_tools: Dict[str, Dict] = {}
    
    def register_agent_tool(self, agent_id: str, tool_name: str, tool_code: str, description: str = ""):
        """Agent creates a new tool from code"""
        
        # Save script to sandbox
        script_path = f"agent_tools/{tool_name}.py"
        self.sandbox.write_file(script_path, tool_code)
        
        # Create callable wrapper
        def dynamic_tool(**kwargs):
            # Serialize kwargs and pass to script
            import json
            kwargs_json = json.dumps(kwargs)
            return self.sandbox.execute_command(f"python {script_path} '{kwargs_json}'")
        
        # Register in main tool registry
        from .tool_registry import ToolDefinition
        tool_def = ToolDefinition(
            id=f"agent_{tool_name}",
            name=tool_name,
            description=f"[Agent-created] {description}",
            category="dynamic",
            function=dynamic_tool,
            requires_sandbox=True,
            parameters={"kwargs": "JSON arguments"}
        )
        self.tool_registry.register(tool_def)
        
        # Track metadata
        self.agent_tools[tool_name] = {
            "created_by": agent_id,
            "created_at": datetime.now().isoformat(),
            "code": tool_code,
            "use_count": 0
        }
        
        print(f"[DYNAMIC TOOL] {agent_id} created tool: {tool_name}")
        return tool_def
    
    def get_agent_tools(self) -> list:
        """List all agent-created tools"""
        return list(self.agent_tools.keys())
```

---

### F12. Complete Database Schema

**File: `data/schema.sql`**
```sql
-- Existing tables
CREATE TABLE IF NOT EXISTS model_capabilities (
    model_name TEXT PRIMARY KEY,
    coding_score REAL,
    reasoning_score REAL,
    documentation_score REAL,
    speed_tokens_sec REAL,
    vram_gb REAL,
    assessed_at TEXT,
    is_active BOOLEAN DEFAULT 1
);

-- Task execution history (for learning)
CREATE TABLE IF NOT EXISTS task_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent_used TEXT NOT NULL,
    tools_used TEXT,  -- JSON array
    success BOOLEAN,
    error_message TEXT,
    output_preview TEXT,
    execution_time_sec REAL,
    tokens_consumed INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Inter-agent messages (for debugging)
CREATE TABLE IF NOT EXISTS agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    message_type TEXT,  -- question, response, broadcast
    content TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Human checkpoints (audit trail)
CREATE TABLE IF NOT EXISTS human_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    checkpoint_type TEXT,
    user_decision TEXT,  -- approved, rejected, modified
    user_feedback TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Dynamic tools created by agents
CREATE TABLE IF NOT EXISTS agent_created_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT UNIQUE NOT NULL,
    created_by_agent TEXT NOT NULL,
    description TEXT,
    code TEXT,
    used_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Workflow checkpoints (for resume)
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    state_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_task_executions_task ON task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_from ON agent_messages(from_agent);
CREATE INDEX IF NOT EXISTS idx_human_decisions_task ON human_decisions(task_id);
```

---

### F13. Extended CLI Commands

**Add to: `run_orchestrator.py`**
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="AI Autonom Orchestrator")
    
    # Model commands
    parser.add_argument("--scan-models", action="store_true", help="Scan for new Ollama models")
    parser.add_argument("--list-models", action="store_true", help="List registered models")
    parser.add_argument("--benchmark", type=str, help="Benchmark specific model")
    
    # Status and monitoring
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    
    # Execution control
    parser.add_argument("--dry-run", action="store_true", help="Plan without executing")
    parser.add_argument("--validate-plan", action="store_true", help="Validate task DAG")
    
    # Memory management
    parser.add_argument("--clear-memory", action="store_true", help="Clear vector DB")
    parser.add_argument("--export-memory", type=str, help="Export memory to file")
    parser.add_argument("--query", type=str, help="Query memory semantically")
    
    # Tool management
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file")
    
    # Goal (positional or via flag)
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    
    args = parser.parse_args()
    
    # Handle commands...
    if args.status:
        from src.monitoring.telemetry import ExecutionMonitor
        monitor = ExecutionMonitor()
        print(monitor.get_dashboard_data())
    elif args.query:
        from src.memory.vector_store import VectorMemoryStore
        store = VectorMemoryStore()
        print(store.query_natural(args.query))
    elif args.list_tools:
        from src.tools.tool_executor import ToolExecutor
        executor = ToolExecutor()
        print(executor.get_available_tools())
    elif args.goal:
        # Run orchestrator with goal
        from src.orchestration.nemotron_orchestrator import NemotronOrchestrator
        orchestrator = NemotronOrchestrator()
        orchestrator.run(args.goal)
    else:
        # Interactive mode
        print("Enter your goal:")
        goal = input("> ").strip()
        if goal:
            from src.orchestration.nemotron_orchestrator import NemotronOrchestrator
            orchestrator = NemotronOrchestrator()
            orchestrator.run(goal)

if __name__ == "__main__":
    main()
```

---

## PART G: DOCKER SANDBOX IMPLEMENTATION

**Requirement:** All code execution happens in isolated Docker containers. Host system is NEVER touched.

### F1. Docker Executor

**File: `src/sandbox/docker_executor.py`**
```python
import docker
import os
import tempfile
from typing import Tuple, Optional

class DockerSandbox:
    """
    Execute agent code in isolated Docker container.
    Host filesystem is NEVER touched directly.
    """
    
    def __init__(self, image: str = "python:3.11-slim"):
        self.client = docker.from_env()
        self.image = image
        self.container = None
        self.workspace = "/workspace"
    
    def start(self):
        """Start sandbox container"""
        self.container = self.client.containers.run(
            self.image,
            detach=True,
            tty=True,
            working_dir=self.workspace,
            volumes={
                'agent_workspace': {'bind': self.workspace, 'mode': 'rw'}
            },
            mem_limit='2g',
            cpu_period=100000,
            cpu_quota=200000,  # 2 CPUs max
            network_mode='bridge',
            remove=False
        )
        print(f"[SANDBOX] Started container: {self.container.short_id}")
        return self.container.short_id
    
    def execute_command(self, command: str, timeout: int = 30) -> Tuple[bool, str]:
        """Execute bash command in sandbox"""
        if not self.container:
            return False, "Container not started"
        
        try:
            exit_code, output = self.container.exec_run(
                f"bash -c '{command}'",
                workdir=self.workspace,
                demux=True
            )
            stdout = output[0].decode() if output[0] else ""
            stderr = output[1].decode() if output[1] else ""
            return exit_code == 0, stdout if exit_code == 0 else stderr
        except Exception as e:
            return False, str(e)
    
    def write_file(self, filename: str, content: str) -> Tuple[bool, str]:
        """Write file inside container"""
        # Escape content for bash
        escaped = content.replace("'", "'\"'\"'")
        cmd = f"cat > {self.workspace}/{filename} << 'ENDOFFILE'\n{content}\nENDOFFILE"
        return self.execute_command(cmd)
    
    def read_file(self, filename: str) -> Tuple[bool, str]:
        """Read file from container"""
        return self.execute_command(f"cat {self.workspace}/{filename}")
    
    def run_python(self, filename: str) -> Tuple[bool, str]:
        """Execute Python file in sandbox"""
        return self.execute_command(f"python {self.workspace}/{filename}")
    
    def install_package(self, package: str) -> Tuple[bool, str]:
        """Install pip package in sandbox"""
        return self.execute_command(f"pip install {package}")
    
    def stop(self):
        """Stop and remove container"""
        if self.container:
            self.container.stop()
            self.container.remove()
            print(f"[SANDBOX] Stopped container")
            self.container = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
```

### F2. Docker Compose Configuration

**File: `docker/docker-compose.yml`**
```yaml
version: '3.8'

services:
  sandbox:
    build:
      context: ../
      dockerfile: docker/Dockerfile.sandbox
    container_name: agent_sandbox
    volumes:
      - agent_workspace:/workspace
      - ../outputs:/outputs
    mem_limit: 2g
    cpus: 2.0
    networks:
      - agent_network
    security_opt:
      - no-new-privileges:true
    read_only: false

volumes:
  agent_workspace:

networks:
  agent_network:
    driver: bridge
```

### F3. Sandbox Dockerfile

**File: `docker/Dockerfile.sandbox`**
```dockerfile
FROM python:3.11-slim

# Install common tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install common Python packages
RUN pip install --no-cache-dir \
    requests \
    pandas \
    numpy \
    pytest

# Create workspace
WORKDIR /workspace

# Non-root user for security
RUN useradd -m -s /bin/bash agent
USER agent

CMD ["bash"]
```

---

## PART G: ACTUAL TOOLS IMPLEMENTATION

**Requirement:** Tools are REAL - they execute code, write files, run commands. Not just text generation.

### G1. Tool Registry (Modular)

**File: `src/tools/tool_registry.py`**
```python
from typing import Dict, Callable, Any, List
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """Definition of an executable tool"""
    id: str
    name: str
    description: str
    category: str
    function: Callable
    requires_sandbox: bool
    parameters: Dict[str, str]  # param_name -> description

class ToolRegistry:
    """
    Central registry of all available tools.
    Tools are MODULAR - can be added/removed dynamically.
    Orchestrator queries this to decide which tools to assign.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
    
    def register(self, tool: ToolDefinition):
        """Register a new tool"""
        self.tools[tool.id] = tool
        print(f"[TOOLS] Registered: {tool.name}")
    
    def get_tool(self, tool_id: str) -> ToolDefinition:
        return self.tools.get(tool_id)
    
    def execute(self, tool_id: str, **kwargs) -> Any:
        """Execute a tool by ID"""
        tool = self.tools.get(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")
        return tool.function(**kwargs)
    
    def get_tools_for_capability(self, capability: str) -> List[ToolDefinition]:
        """Get tools matching a capability"""
        mapping = {
            "code_generation": ["filesystem_write", "python_exec", "bash_exec"],
            "web_search": ["web_search", "web_fetch"],
            "file_operations": ["filesystem_read", "filesystem_write", "filesystem_search"],
            "testing": ["python_exec", "pytest_run"],
        }
        tool_ids = mapping.get(capability, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def list_all(self) -> List[ToolDefinition]:
        return list(self.tools.values())
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions for orchestrator prompt"""
        desc = []
        for tool in self.tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            desc.append(f"- {tool.id}: {tool.description} (params: {params})")
        return "\n".join(desc)
```

### G2. Built-in Tools Implementation

**File: `src/tools/builtin_tools.py`**
```python
import os
import subprocess
import requests
from typing import Tuple

class BuiltinTools:
    """Actual tool implementations that DO things"""
    
    def __init__(self, sandbox=None):
        self.sandbox = sandbox  # Optional Docker sandbox
    
    # === FILESYSTEM TOOLS ===
    
    def filesystem_read(self, path: str) -> Tuple[bool, str]:
        """Read file contents"""
        if self.sandbox:
            return self.sandbox.read_file(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return True, f.read()
        except Exception as e:
            return False, str(e)
    
    def filesystem_write(self, path: str, content: str) -> Tuple[bool, str]:
        """Write content to file"""
        if self.sandbox:
            return self.sandbox.write_file(path, content)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Written to {path}"
        except Exception as e:
            return False, str(e)
    
    def filesystem_search(self, directory: str, pattern: str) -> Tuple[bool, str]:
        """Search for files matching pattern"""
        import glob
        try:
            matches = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
            return True, "\n".join(matches)
        except Exception as e:
            return False, str(e)
    
    # === CODE EXECUTION TOOLS ===
    
    def python_exec(self, code: str, filename: str = "script.py") -> Tuple[bool, str]:
        """Execute Python code"""
        if self.sandbox:
            self.sandbox.write_file(filename, code)
            return self.sandbox.run_python(filename)
        try:
            # Local execution (use with caution)
            result = subprocess.run(
                ['python', '-c', code],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout or result.stderr
        except Exception as e:
            return False, str(e)
    
    def bash_exec(self, command: str) -> Tuple[bool, str]:
        """Execute bash command"""
        if self.sandbox:
            return self.sandbox.execute_command(command)
        # Local execution disabled for safety
        return False, "Bash execution requires sandbox mode"
    
    def pytest_run(self, test_file: str) -> Tuple[bool, str]:
        """Run pytest on file"""
        if self.sandbox:
            return self.sandbox.execute_command(f"pytest {test_file} -v")
        try:
            result = subprocess.run(
                ['pytest', test_file, '-v'],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout
        except Exception as e:
            return False, str(e)
    
    # === WEB TOOLS ===
    
    def web_search(self, query: str) -> Tuple[bool, str]:
        """Search the web (requires API key)"""
        # Placeholder - integrate with Tavily or similar
        return False, "Web search not configured. Set TAVILY_API_KEY."
    
    def web_fetch(self, url: str) -> Tuple[bool, str]:
        """Fetch content from URL"""
        try:
            response = requests.get(url, timeout=10)
            return True, response.text[:5000]  # Limit content
        except Exception as e:
            return False, str(e)
```

### G3. Tool Executor (Orchestrator Uses This)

**File: `src/tools/tool_executor.py`**
```python
from typing import Dict, Any, List
from .tool_registry import ToolRegistry, ToolDefinition
from .builtin_tools import BuiltinTools

class ToolExecutor:
    """
    Executes tools assigned by orchestrator.
    Orchestrator decides WHICH tools, executor RUNS them.
    """
    
    def __init__(self, sandbox=None):
        self.registry = ToolRegistry()
        self.builtin = BuiltinTools(sandbox)
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register all built-in tools"""
        tools = [
            ToolDefinition(
                id="filesystem_read",
                name="Read File",
                description="Read contents of a file",
                category="filesystem",
                function=self.builtin.filesystem_read,
                requires_sandbox=False,
                parameters={"path": "File path to read"}
            ),
            ToolDefinition(
                id="filesystem_write",
                name="Write File",
                description="Write content to a file",
                category="filesystem",
                function=self.builtin.filesystem_write,
                requires_sandbox=True,
                parameters={"path": "File path", "content": "Content to write"}
            ),
            ToolDefinition(
                id="filesystem_search",
                name="Search Files",
                description="Search for files matching pattern",
                category="filesystem",
                function=self.builtin.filesystem_search,
                requires_sandbox=False,
                parameters={"directory": "Directory to search", "pattern": "Glob pattern"}
            ),
            ToolDefinition(
                id="python_exec",
                name="Execute Python",
                description="Execute Python code and return output",
                category="code_execution",
                function=self.builtin.python_exec,
                requires_sandbox=True,
                parameters={"code": "Python code to execute"}
            ),
            ToolDefinition(
                id="bash_exec",
                name="Execute Bash",
                description="Execute bash command (sandbox only)",
                category="code_execution",
                function=self.builtin.bash_exec,
                requires_sandbox=True,
                parameters={"command": "Bash command"}
            ),
            ToolDefinition(
                id="pytest_run",
                name="Run Tests",
                description="Run pytest on a test file",
                category="testing",
                function=self.builtin.pytest_run,
                requires_sandbox=True,
                parameters={"test_file": "Path to test file"}
            ),
            ToolDefinition(
                id="web_fetch",
                name="Fetch URL",
                description="Fetch content from a URL",
                category="web",
                function=self.builtin.web_fetch,
                requires_sandbox=False,
                parameters={"url": "URL to fetch"}
            ),
        ]
        
        for tool in tools:
            self.registry.register(tool)
    
    def execute_tool(self, tool_id: str, **params) -> Dict[str, Any]:
        """Execute a specific tool with parameters"""
        tool = self.registry.get_tool(tool_id)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {tool_id}"}
        
        print(f"[TOOL] Executing: {tool.name}")
        try:
            success, result = tool.function(**params)
            return {
                "success": success,
                "result": result,
                "tool": tool_id
            }
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_id}
    
    def get_available_tools(self) -> str:
        """Get tool descriptions for orchestrator"""
        return self.registry.get_tool_descriptions()
```

### G4. Orchestrator Tool Integration

**How Nemotron decides which tools:**
```python
# In nemotron_orchestrator.py - orchestrator DECIDES tools

def decompose_and_assign(self, user_goal: str) -> List[Dict]:
    # Get available tools
    tool_descriptions = self.tool_executor.get_available_tools()
    
    prompt = f"""You are Nemotron, the orchestrator. You DECIDE:
1. How to break down the task
2. Which AGENT executes each subtask
3. Which TOOLS each agent needs

AVAILABLE TOOLS:
{tool_descriptions}

For each task, you MUST specify tools the agent needs.
Choose tools based on what the task requires:
- Code writing? -> filesystem_write, python_exec
- Testing? -> pytest_run
- File search? -> filesystem_search
- Web research? -> web_fetch

User Goal: {user_goal}

Return JSON with tools assigned:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "...",
      "assigned_agent": "coder_qwen",
      "tools": ["filesystem_write", "python_exec"],
      "tool_params": {{
        "filesystem_write": {{"path": "outputs/script.py"}}
      }}
    }}
  ]
}}
"""
```

---

## PART H: COMPLETE FILE STRUCTURE (Final)

```
AI_Autonom/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent_registry.py        # Agent definitions
│   │   ├── model_discovery.py       # Auto-detect models
│   │   ├── model_selector.py        # Dynamic selection
│   │   ├── model_watcher.py         # Background sync
│   │   └── config.py                # Configuration loader
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── task_memory.py           # Inter-task context (RAM)
│   │   ├── vector_store.py          # ChromaDB semantic search
│   │   └── structured_store.py      # SQLite history
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── nemotron_orchestrator.py # Main orchestrator
│   │   ├── langgraph_workflow.py    # LangGraph engine (P0!)
│   │   ├── agent_messaging.py       # Inter-agent comms (P0!)
│   │   ├── intent_analyzer.py       # Deep intent analysis
│   │   ├── error_recovery.py        # Retry & recovery
│   │   ├── human_checkpoint.py      # Human-in-loop
│   │   ├── testing_workflow.py      # Test generation
│   │   └── state_guards.py          # Hard guards
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tool_registry.py         # Modular tool registry
│   │   ├── builtin_tools.py         # Real implementations
│   │   ├── tool_executor.py         # Execute tools
│   │   ├── code_executor.py         # Code extraction
│   │   └── dynamic_tools.py         # Agent-created tools
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── docker_executor.py       # Docker isolation
│   └── monitoring/
│       ├── __init__.py
│       └── telemetry.py             # Execution monitoring
├── config/
│   └── settings.yaml                # Central configuration
├── docker/
│   ├── Dockerfile.sandbox
│   └── docker-compose.yml
├── data/
│   ├── chromadb/                    # Vector DB storage
│   ├── sqlite/                      # Structured DB
│   ├── checkpoints.db               # Workflow checkpoints
│   └── schema.sql                   # DB schema
├── logs/
│   ├── orchestrator.log
│   └── execution.jsonl              # Telemetry events
├── outputs/                         # Generated files
├── agent_tools/                     # Dynamic tools by agents
├── venv/
├── requirements.txt
├── run_orchestrator.py              # CLI entry point
├── MVP_ROADMAP.md
└── agentic_architecture.md
```

---

## PART I: PRIORITY MATRIX

| Component | Priority | Effort | Impact | Session |
|-----------|----------|--------|--------|----------|
| **LangGraph Workflow** | 🔴 P0 | 3hrs | CRITICAL | 1 |
| **Agent Messaging** | 🔴 P0 | 2hrs | CRITICAL | 1 |
| **Synthesis Logic** | 🔴 P0 | 1hr | CRITICAL | 1 |
| **Vector DB (Full)** | 🔴 P0 | 1.5hrs | CRITICAL | 1 |
| **Model Discovery** | 🔴 P0 | 1hr | HIGH | 1 |
| **Task Memory** | 🔴 P0 | 30min | HIGH | 1 |
| **Tool Executor** | 🔴 P0 | 1hr | HIGH | 1 |
| **Intent Analysis (Deep)** | 🟡 P1 | 1hr | HIGH | 2 |
| **Error Recovery** | 🟡 P1 | 1.5hrs | HIGH | 2 |
| **Testing Workflow** | 🟡 P1 | 1hr | HIGH | 2 |
| **Docker Sandbox** | 🟡 P1 | 1.5hr | HIGH | 2 |
| **Human Checkpoints** | 🟢 P2 | 1hr | MEDIUM | 3 |
| **Monitoring** | 🟢 P2 | 1hr | MEDIUM | 3 |
| **Config Management** | 🟢 P2 | 30min | LOW | 3 |
| **Dynamic Tools** | 🟢 P2 | 1hr | MEDIUM | 3 |

---

## PART J: IMPLEMENTATION ORDER (By Session)

### SESSION 1 (P0 - CRITICAL) ~10hrs
1. `src/orchestration/langgraph_workflow.py` - Core workflow engine
2. `src/orchestration/agent_messaging.py` - Inter-agent communication
3. `src/memory/vector_store.py` - ChromaDB semantic memory
4. `src/memory/task_memory.py` - RAM context sharing
5. `src/core/model_discovery.py` - Auto-detect models
6. `src/core/model_selector.py` - Dynamic model selection
7. `src/tools/tool_executor.py` - Tool execution
8. Update `nemotron_orchestrator.py` - Add synthesis + context

### SESSION 2 (P1 - HIGH) ~6hrs
9. `src/orchestration/intent_analyzer.py` - Deep intent analysis
10. `src/orchestration/error_recovery.py` - Retry logic
11. `src/orchestration/testing_workflow.py` - Test generation
12. `src/sandbox/docker_executor.py` - Docker isolation
13. `src/tools/builtin_tools.py` - Real tool implementations
14. Update `run_orchestrator.py` - CLI commands

### SESSION 3 (P2 - MEDIUM) ~4hrs
15. `src/orchestration/human_checkpoint.py` - Human-in-loop
16. `src/monitoring/telemetry.py` - Execution monitoring
17. `config/settings.yaml` - Central config
18. `src/core/config.py` - Config loader
19. `src/tools/dynamic_tools.py` - Agent-created tools
20. `data/schema.sql` - Complete DB schema

---

## PART K: ESTIMATED TIMELINE (Complete)

| Session | Hours | Deliverable |
|---------|-------|-------------|
| Session 1 | ~10hrs | LangGraph + Messaging + Memory + Model Management |
| Session 2 | ~6hrs | Intent + Recovery + Testing + Docker |
| Session 3 | ~4hrs | Checkpoints + Monitoring + Config + Dynamic Tools |
| **Total MVP** | **~20hrs** | **Full Multi-Agent System** |

---

## PART L: SUCCESS CRITERIA

### Test 1: Dynamic Model Management
```bash
# Install new model
ollama pull phi3:mini

# System should:
# 1. Auto-detect within 60 seconds
# 2. Benchmark the model
# 3. Add to database with scores
# 4. Use for tasks if best available
```

### Test 2: Inter-Agent Communication
```python
# Agent A asks Agent B
message_bus.query_agent("coder_qwen", "linguistic_dictalm", 
    "What file naming convention should I use?")
# Should get response from linguistic agent
```

### Test 3: Full MVP
```bash
python run_orchestrator.py "Write ONE Python file that searches 
Desktop for password.txt and sends via UDP to 192.168.156.4.2"
```

**Expected Output:**
```
[INTENT] Single file output requested
[INTENT] Confidence: 0.85
[WORKFLOW] Created LangGraph with 3 tasks
[AGENT MSG] coder_qwen -> orchestrator: Starting task_1
[TOOL] filesystem_search executed
[TOOL] python_exec executed
[SYNTHESIS] Combining all outputs...
[FINAL] outputs/final_output.py (52 lines)
[VALIDATION] Syntax: PASS
```

---

## SUMMARY

This plan now covers **100%** of the architecture:
- ✅ Model management (dynamic, modular)
- ✅ Tools (real execution, dynamic creation)
- ✅ Sandbox (Docker isolation)
- ✅ LangGraph (workflow engine)
- ✅ Agent Framework (message bus)
- ✅ Synthesis (combine outputs)
- ✅ Vector DB (semantic search)
- ✅ Error recovery (retry logic)
- ✅ Testing workflow (validation)
- ✅ Human checkpoints (approval)
- ✅ Monitoring (telemetry)
- ✅ Configuration (YAML)

---

## PART M: EXTERNAL RESOURCES TO LEVERAGE

### M1. Microsoft Agent Framework Patterns

**Source:** https://github.com/microsoft/autogen (AutoGen) or https://github.com/microsoft/semantic-kernel

**What to leverage:**

```python
# AGENT PERSONAS (from their patterns)
AGENT_PERSONAS = {
    "researcher": {
        "name": "Research Specialist",
        "system_prompt": """You are an expert research analyst. Your role is to:
- Search for authoritative sources
- Synthesize information from multiple references
- Provide citations for all claims
- Highlight conflicting information
- Recommend best practices based on evidence

Always be thorough but concise. Cite your sources.""",
        "personality_traits": ["methodical", "skeptical", "thorough"],
        "expertise_domains": ["web_research", "documentation", "analysis"]
    },
    
    "coder": {
        "name": "Senior Software Engineer",
        "system_prompt": """You are a senior software engineer with 10+ years experience.
Your coding principles:
- Write clean, readable code with clear naming
- Follow SOLID principles
- Always include error handling
- Write code that compiles on first try
- Add comments for complex logic
- Consider edge cases

Before coding:
1. Understand the requirements fully
2. Plan the structure
3. Consider dependencies
4. Think about testing

Output complete, runnable code.""",
        "personality_traits": ["precise", "pragmatic", "quality-focused"],
        "expertise_domains": ["python", "c++", "architecture"]
    },
    
    "tester": {
        "name": "QA Engineer",
        "system_prompt": """You are a senior QA engineer who finds bugs others miss.
Your testing philosophy:
- Test behavior, not implementation
- Cover edge cases: null, empty, boundary values
- Test error conditions
- Verify both positive and negative cases
- Write self-documenting test names

For each function, generate:
1. Happy path tests
2. Edge case tests
3. Error handling tests
4. Integration tests if needed

Tests should be independent and deterministic.""",
        "personality_traits": ["detail-oriented", "skeptical", "systematic"],
        "expertise_domains": ["pytest", "unit_testing", "integration_testing"]
    },
    
    "debugger": {
        "name": "Debug Specialist",
        "system_prompt": """You are an expert debugger who can find any bug.
Your debugging process:
1. Reproduce the error
2. Read the FULL error message
3. Identify the root cause (not symptoms)
4. Form a hypothesis
5. Test the hypothesis
6. Fix ONE thing at a time
7. Verify the fix doesn't break other things

Never guess. Always trace the actual execution path.
Use print statements or logging to verify assumptions.""",
        "personality_traits": ["patient", "analytical", "persistent"],
        "expertise_domains": ["debugging", "error_analysis", "code_review"]
    },
    
    "architect": {
        "name": "System Architect",
        "system_prompt": """You are a system architect who designs scalable solutions.
Your design principles:
- Start simple, complexity is the enemy
- Separate concerns (single responsibility)
- Design for change (loose coupling)
- Make dependencies explicit
- Prefer composition over inheritance

Before any design:
1. Clarify requirements
2. Identify constraints
3. Consider trade-offs
4. Document decisions and rationale

Your outputs should be clear diagrams and specifications.""",
        "personality_traits": ["strategic", "pragmatic", "communicative"],
        "expertise_domains": ["system_design", "architecture", "planning"]
    }
}
```

**File to create: `src/agents/personas.py`**

---

### M2. MCP (Model Context Protocol) Integration

**Source:** https://github.com/modelcontextprotocol

**What MCP gives us:**
- Standardized tool interface (like LSP for tools)
- 2000+ community servers
- OAuth security built-in
- Tool discovery protocol

**File: `src/tools/mcp_client.py`**
```python
from mcp import Client, Server
from mcp.types import Tool, Resource
import json

class MCPToolClient:
    """
    Interface to MCP ecosystem.
    Provides access to 2000+ community tools.
    """
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        self.client = Client()
        self.servers = {}  # name -> Server connection
        self.available_tools = {}  # tool_name -> server
        self._load_server_config(config_path)
    
    def _load_server_config(self, path: str):
        """Load MCP server configurations"""
        # Default servers to connect
        default_servers = [
            {"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]},
            {"name": "github", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
            {"name": "brave-search", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"]},
            {"name": "memory", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
        ]
        return default_servers
    
    async def connect_server(self, name: str, command: str, args: list):
        """Connect to an MCP server"""
        server = await self.client.connect_stdio(command, args)
        self.servers[name] = server
        
        # Discover tools from this server
        tools = await server.list_tools()
        for tool in tools:
            self.available_tools[tool.name] = name
        
        print(f"[MCP] Connected to {name}, {len(tools)} tools available")
    
    async def list_all_tools(self) -> list:
        """List all tools from all connected servers"""
        all_tools = []
        for name, server in self.servers.items():
            tools = await server.list_tools()
            all_tools.extend([(name, t) for t in tools])
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool by name"""
        if tool_name not in self.available_tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        server_name = self.available_tools[tool_name]
        server = self.servers[server_name]
        
        result = await server.call_tool(tool_name, arguments)
        return {"success": True, "result": result}
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions for orchestrator prompt"""
        descriptions = []
        for tool_name, server_name in self.available_tools.items():
            descriptions.append(f"- {tool_name} (from {server_name})")
        return "\n".join(descriptions)
```

**MCP Servers to install:**
```bash
# Core servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-sqlite
npm install -g @modelcontextprotocol/server-puppeteer
```

---

### M3. NVIDIA ToolOrchestra Prompt Templates

**What we should use from their framework:**

**File: `src/prompts/orchestrator_templates.py`**
```python
"""
Prompt templates based on NVIDIA ToolOrchestra patterns.
These are battle-tested prompts for task decomposition.
"""

DECOMPOSITION_PROMPT = """
You are a Meta-Orchestrator. Your ONLY job is to break down user goals into executable tasks.

## DECOMPOSITION RULES
1. Maximum 3 levels of hierarchy
2. Each task must be atomic (one agent, one outcome)
3. Dependencies must be explicit
4. No task should take more than 10 minutes
5. Always end with synthesis/integration task

## TASK CONTRACT FORMAT (MANDATORY)
Every task MUST have a JSON contract:
```json
{
  "task_id": "unique_id",
  "goal": "Natural language description",
  "inputs": ["list of required inputs from other tasks"],
  "outputs": ["list of artifacts this task produces"],
  "verification": "How to verify task is complete",
  "constraints": ["time", "quality", "dependencies"],
  "assigned_agent": "agent_type",
  "tools": ["required_tools"]
}
```

## AGENT TYPES AVAILABLE
{agent_registry}

## USER GOAL
{user_goal}

## OUTPUT
Return a JSON array of task contracts. Nothing else.
"""

AGENT_EXECUTION_PROMPT = """
You are {agent_name}, a specialized AI agent.

## YOUR PERSONA
{persona_description}

## YOUR TASK CONTRACT
{task_contract}

## CONTEXT FROM PREVIOUS TASKS
{previous_context}

## AVAILABLE TOOLS
{tool_descriptions}

## EXECUTION RULES
1. Read the contract carefully
2. Plan your approach before acting
3. Use tools to accomplish the goal
4. Verify your output matches the contract
5. Report clearly what you produced

## OUTPUT FORMAT
Provide:
1. Your reasoning/approach
2. Tool calls (if needed)
3. Final output matching contract's "outputs" field
4. Verification that contract is fulfilled
"""

SYNTHESIS_PROMPT = """
You are a Synthesis Agent. Your job is to combine outputs from multiple tasks into a final deliverable.

## PREVIOUS TASK OUTPUTS
{all_task_outputs}

## USER'S ORIGINAL GOAL
{original_goal}

## OUTPUT REQUIREMENTS
{output_requirements}

## YOUR TASK
1. Read ALL previous outputs
2. Identify the key artifacts (code, docs, etc.)
3. Combine into a single coherent deliverable
4. Resolve any conflicts or duplications
5. Ensure the final output fully satisfies the original goal

## OUTPUT
Provide the complete, final deliverable. No explanations needed - just the artifact.
"""
```

---

### M4. DeepAgents Pattern (Universal Compute)

**Source:** DeepAgents paper - "Agents get bash + filesystem = can do anything"

**File: `src/patterns/deepagents.py`**
```python
"""
DeepAgents Pattern: Universal Compute through Bash + Filesystem

Key insight: An agent with just bash and filesystem can:
1. Write any code
2. Execute any program
3. Create new tools
4. Debug by examining state
5. Chain operations naturally

This is the SECRET SAUCE that makes agents powerful.
"""

DEEPAGENTS_SYSTEM_PROMPT = """
You have access to two fundamental tools:
1. BASH: Execute any shell command
2. FILESYSTEM: Read/write any file

With these two tools, you can accomplish ANYTHING a human developer can:

## PATTERNS TO USE

### Pattern 1: Code Execution
```
1. Write code to file: filesystem.write("script.py", code)
2. Execute: bash("python script.py")
3. Check result: bash("echo $?")
4. If error, read output: bash("cat error.log")
5. Fix and retry
```

### Pattern 2: Tool Creation
```
1. Identify missing capability
2. Write a script that provides it
3. Save to tools/my_tool.py
4. Now you can call: bash("python tools/my_tool.py args")
5. This tool is now available for future tasks
```

### Pattern 3: Debugging
```
1. Check filesystem state: bash("ls -la")
2. Examine file contents: filesystem.read("file.txt")
3. Check environment: bash("env | grep PATH")
4. Trace execution: bash("python -m trace script.py")
```

### Pattern 4: Chaining
```
bash("cat data.json | python parse.py | sort > output.txt")
```

## REMEMBER
- You are a developer with a terminal
- Think step by step
- Verify each step before proceeding
- If something fails, debug like a human would
- Leave useful tools for other agents
"""

class DeepAgentsExecutor:
    """Execute agent tasks using DeepAgents pattern"""
    
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.created_tools = []  # Track tools created by agents
    
    def execute_with_pattern(self, agent_output: str) -> dict:
        """Parse agent output and execute bash/filesystem commands"""
        # Extract code blocks
        code_blocks = self._extract_code_blocks(agent_output)
        
        results = []
        for block in code_blocks:
            if block['type'] == 'bash':
                success, output = self.sandbox.execute_command(block['content'])
                results.append({"type": "bash", "success": success, "output": output})
            elif block['type'] == 'python':
                # Write and execute Python
                filename = f"temp_{len(results)}.py"
                self.sandbox.write_file(filename, block['content'])
                success, output = self.sandbox.run_python(filename)
                results.append({"type": "python", "success": success, "output": output})
        
        return {"results": results, "all_success": all(r["success"] for r in results)}
    
    def register_agent_tool(self, tool_name: str, tool_code: str):
        """When agent creates a tool, register it for others"""
        tool_path = f"agent_tools/{tool_name}.py"
        self.sandbox.write_file(tool_path, tool_code)
        self.created_tools.append(tool_name)
        print(f"[DEEPAGENTS] New tool registered: {tool_name}")
```

---

### M5. Conversation Patterns (Multi-Agent Chat)

**From AutoGen/AgentFramework:**

**File: `src/patterns/conversations.py`**
```python
"""
Multi-Agent Conversation Patterns
Based on Microsoft AutoGen patterns.
"""

# Pattern 1: Sequential Chat (A -> B -> C)
SEQUENTIAL_CHAT = {
    "pattern": "sequential",
    "description": "Tasks pass from one agent to next",
    "example": "Researcher -> Coder -> Tester",
    "use_when": "Clear linear dependency"
}

# Pattern 2: Group Chat (All agents see all messages)
GROUP_CHAT = {
    "pattern": "group",
    "description": "All agents participate in shared conversation",
    "moderator": "Orchestrator decides who speaks next",
    "use_when": "Complex problem needing multiple perspectives"
}

# Pattern 3: Nested Chat (Agent spawns sub-agents)
NESTED_CHAT = {
    "pattern": "nested", 
    "description": "Agent can spawn sub-conversation",
    "example": "Coder encounters bug, spawns Debugger sub-chat",
    "use_when": "Task needs specialist intervention"
}

# Pattern 4: Two-Agent Debate
DEBATE_PATTERN = {
    "pattern": "debate",
    "agents": ["proposer", "critic"],
    "description": "One proposes, other critiques, iterate",
    "use_when": "Need to refine solution quality"
}

class ConversationOrchestrator:
    """Manage multi-agent conversations"""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.active_conversations = {}
    
    def start_sequential(self, agents: list, initial_message: str):
        """Start sequential conversation"""
        conv_id = f"seq_{len(self.active_conversations)}"
        self.active_conversations[conv_id] = {
            "pattern": "sequential",
            "agents": agents,
            "current_index": 0,
            "messages": [initial_message]
        }
        return conv_id
    
    def start_group_chat(self, agents: list, topic: str):
        """Start group conversation with moderator"""
        conv_id = f"group_{len(self.active_conversations)}"
        self.active_conversations[conv_id] = {
            "pattern": "group",
            "agents": agents,
            "moderator": "orchestrator",
            "topic": topic,
            "messages": []
        }
        return conv_id
    
    def next_speaker(self, conv_id: str) -> str:
        """Determine who speaks next"""
        conv = self.active_conversations[conv_id]
        
        if conv["pattern"] == "sequential":
            return conv["agents"][conv["current_index"]]
        elif conv["pattern"] == "group":
            # Moderator decides based on context
            return self._moderator_decision(conv)
    
    def _moderator_decision(self, conv: dict) -> str:
        """Let orchestrator decide next speaker"""
        # Could use LLM to decide who should speak next
        # For now, round-robin
        last_speaker = conv["messages"][-1]["from"] if conv["messages"] else None
        agents = conv["agents"]
        if last_speaker:
            idx = agents.index(last_speaker)
            return agents[(idx + 1) % len(agents)]
        return agents[0]
```

---

### M6. Implementation Priority for External Resources

| Resource | Priority | File to Create | Effort |
|----------|----------|----------------|--------|
| **Agent Personas** | 🔴 P0 | `src/agents/personas.py` | 1hr |
| **Prompt Templates** | 🔴 P0 | `src/prompts/orchestrator_templates.py` | 1hr |
| **DeepAgents Pattern** | 🔴 P0 | `src/patterns/deepagents.py` | 1.5hr |
| **MCP Client** | 🟡 P1 | `src/tools/mcp_client.py` | 2hr |
| **Conversation Patterns** | 🟡 P1 | `src/patterns/conversations.py` | 1hr |
| **MCP Server Install** | 🟢 P2 | Terminal commands | 30min |

---

### M7. Updated File Structure with External Patterns

```
AI_Autonom/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── personas.py              # Agent personalities & prompts
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── orchestrator_templates.py # NVIDIA-style decomposition
│   ├── patterns/
│   │   ├── __init__.py
│   │   ├── deepagents.py            # Bash+Filesystem universal compute
│   │   └── conversations.py         # Multi-agent chat patterns
│   ├── tools/
│   │   ├── mcp_client.py            # MCP protocol integration
│   │   └── ... (existing)
│   └── ... (existing)
├── config/
│   ├── settings.yaml
│   └── mcp_servers.json          # MCP server configurations
└── ... (existing)
```

---

## PART N: CAI (CYBERSECURITY AI) FRAMEWORK INTEGRATION

**Source:** https://github.com/aliasrobotics/cai

CAI is the **de facto framework for AI Security** with 6.7k stars, battle-tested in HackTheBox CTFs and real bug bounties. We MUST leverage their patterns.

### N1. CAI Agentic Patterns (From Their Architecture)

**CAI defines 5 core patterns we should implement:**

```python
"""
CAI Agentic Patterns - Formally defined as:
AP = (A, H, D, C, E)

Where:
- A (Agents): Set of autonomous entities with roles
- H (Handoffs): Task transfer function between agents
- D (Decision): Which agent acts based on system state
- C (Communication): Message passing protocol
- E (Execution): How agents perform tasks
"""

# File: src/patterns/cai_patterns.py

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Callable

class PatternType(Enum):
    SWARM = "swarm"              # Decentralized, agents self-assign
    HIERARCHICAL = "hierarchical" # Top-level assigns to sub-agents
    CHAIN_OF_THOUGHT = "chain"    # Sequential pipeline A->B->C
    AUCTION_BASED = "auction"     # Agents bid on tasks
    RECURSIVE = "recursive"       # Agent refines its own output

@dataclass
class AgenticPattern:
    """CAI-style agentic pattern definition"""
    name: str
    pattern_type: PatternType
    description: str
    agents: List[str]  # Agent IDs
    handoff_rules: Dict[str, str]  # source_agent -> target_agent
    entry_agent: str
    termination_condition: Callable

# Pre-defined patterns from CAI
CAI_PATTERNS = {
    "ctf_swarm": AgenticPattern(
        name="CTF Swarm Pattern",
        pattern_type=PatternType.SWARM,
        description="Team of agents working together on CTF with dynamic handoffs",
        agents=["recon_agent", "exploit_agent", "privesc_agent", "flag_agent"],
        handoff_rules={
            "recon_agent": ["exploit_agent", "privesc_agent"],
            "exploit_agent": ["privesc_agent", "flag_agent"],
            "privesc_agent": ["flag_agent"],
        },
        entry_agent="recon_agent",
        termination_condition=lambda state: "FLAG" in state.get("output", "")
    ),
    
    "security_pipeline": AgenticPattern(
        name="Security Assessment Pipeline",
        pattern_type=PatternType.CHAIN_OF_THOUGHT,
        description="Linear security assessment: Recon -> Vuln Scan -> Exploit -> Report",
        agents=["recon_agent", "vuln_scanner", "exploit_agent", "report_agent"],
        handoff_rules={
            "recon_agent": "vuln_scanner",
            "vuln_scanner": "exploit_agent",
            "exploit_agent": "report_agent"
        },
        entry_agent="recon_agent",
        termination_condition=lambda state: state.get("report_generated", False)
    ),
    
    "code_review": AgenticPattern(
        name="Recursive Code Review",
        pattern_type=PatternType.RECURSIVE,
        description="Agent continuously refines code until tests pass",
        agents=["code_agent"],
        handoff_rules={"code_agent": "code_agent"},
        entry_agent="code_agent",
        termination_condition=lambda state: state.get("tests_passed", False)
    )
}
```

### N2. CAI Security Agents (To Integrate)

**File: `src/agents/security_agents.py`**
```python
"""
Security-focused agents based on CAI framework.
These agents are specialized for cybersecurity tasks.
"""

SECURITY_AGENTS = {
    "recon_agent": {
        "name": "Reconnaissance Agent",
        "model": "qwen3:1.7b",
        "instructions": """You are a reconnaissance specialist. Your job is to:
1. Enumerate targets (IPs, domains, services)
2. Identify attack surface
3. Map network topology
4. Discover exposed services and versions

Tools at your disposal:
- nmap for port scanning
- whois for domain info
- dig/nslookup for DNS
- curl for HTTP probing

Always document findings in structured format.
Report potential vulnerabilities for other agents to exploit.""",
        "tools": ["bash_exec", "filesystem_write", "web_fetch"],
        "handoffs": ["vuln_scanner", "exploit_agent"]
    },
    
    "vuln_scanner": {
        "name": "Vulnerability Scanner",
        "model": "qwen3:1.7b",
        "instructions": """You are a vulnerability assessment specialist. Your job is to:
1. Analyze recon data for potential vulnerabilities
2. Test for common vulnerabilities (SQLi, XSS, SSRF, etc.)
3. Check for default credentials
4. Identify CVEs for discovered software versions

Use automated tools and manual testing.
Rank vulnerabilities by severity (Critical, High, Medium, Low).
Provide actionable exploitation paths.""",
        "tools": ["bash_exec", "filesystem_read", "filesystem_write", "python_exec"],
        "handoffs": ["exploit_agent"]
    },
    
    "exploit_agent": {
        "name": "Exploitation Agent",
        "model": "qwen3:1.7b",
        "instructions": """You are an exploitation specialist. Your job is to:
1. Develop working exploits for identified vulnerabilities
2. Test exploits in controlled manner
3. Achieve initial access or code execution
4. Prepare for privilege escalation

Prioritize:
- Reliability over speed
- Clean exploitation (minimal traces)
- Documentation of exploitation steps

Hand off to privesc_agent once initial access is achieved.""",
        "tools": ["bash_exec", "filesystem_write", "python_exec"],
        "handoffs": ["privesc_agent", "report_agent"]
    },
    
    "privesc_agent": {
        "name": "Privilege Escalation Agent",
        "model": "qwen3:1.7b",
        "instructions": """You are a privilege escalation specialist. Your job is to:
1. Enumerate local system for privesc vectors
2. Check for misconfigurations (SUID, cron, sudo)
3. Look for credential files and SSH keys
4. Exploit kernel vulnerabilities if needed

Linux privesc checklist:
- sudo -l
- find / -perm -4000 2>/dev/null
- cat /etc/crontab
- ls -la /home/*/.ssh
- Check for docker group membership

Always try least intrusive methods first.""",
        "tools": ["bash_exec", "filesystem_read", "python_exec"],
        "handoffs": ["report_agent"]
    },
    
    "report_agent": {
        "name": "Report Generation Agent",
        "model": "dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0",
        "instructions": """You are a security report specialist. Your job is to:
1. Compile findings from all agents
2. Write clear, professional security reports
3. Include remediation recommendations
4. Provide CVSS scores where applicable

Report structure:
- Executive Summary
- Scope and Methodology
- Findings (sorted by severity)
- Remediation Steps
- Appendix (technical details)

Use clear language that both technical and non-technical readers can understand.""",
        "tools": ["filesystem_read", "filesystem_write"],
        "handoffs": []
    },
    
    "flag_discriminator": {
        "name": "Flag Discriminator",
        "model": "qwen3:1.7b",
        "instructions": """You are a CTF flag validator. Your ONLY job is to:
1. Identify potential flags in agent outputs
2. Validate flag format (CTF{...}, FLAG{...}, etc.)
3. Confirm flag is correct

Flag patterns to look for:
- CTF{...}
- FLAG{...}
- flag{...}
- HTB{...}
- Custom formats

When flag is found, immediately report SUCCESS.""",
        "tools": ["filesystem_read"],
        "handoffs": []
    }
}
```

### N3. CAI Handoffs System

**File: `src/patterns/handoffs.py`**
```python
"""
CAI-style Handoffs: Transfer tasks between agents.
Handoffs are the KEY to multi-agent collaboration.
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

@dataclass
class HandoffContext:
    """Context passed during handoff"""
    from_agent: str
    to_agent: str
    task_id: str
    findings: Dict[str, Any]
    next_action: str
    priority: str = "normal"

class HandoffManager:
    """
    Manages handoffs between agents.
    Based on CAI's H function: H: A x T -> A
    """
    
    def __init__(self, agent_registry, message_bus):
        self.registry = agent_registry
        self.bus = message_bus
        self.handoff_history = []
    
    def handoff(self, from_agent: str, to_agent: str, context: Dict) -> bool:
        """Execute a handoff from one agent to another"""
        # Validate agents exist
        if not self.registry.get_agent(from_agent):
            raise ValueError(f"Source agent not found: {from_agent}")
        if not self.registry.get_agent(to_agent):
            raise ValueError(f"Target agent not found: {to_agent}")
        
        # Create handoff context
        handoff_ctx = HandoffContext(
            from_agent=from_agent,
            to_agent=to_agent,
            task_id=context.get("task_id", "unknown"),
            findings=context.get("findings", {}),
            next_action=context.get("next_action", "continue")
        )
        
        # Send via message bus
        self.bus.send_message(
            from_agent=from_agent,
            to_agent=to_agent,
            content={
                "type": "handoff",
                "context": handoff_ctx.__dict__
            },
            msg_type="handoff"
        )
        
        # Log handoff
        self.handoff_history.append(handoff_ctx)
        print(f"[HANDOFF] {from_agent} -> {to_agent}: {handoff_ctx.next_action}")
        
        return True
    
    def auto_handoff(self, current_agent: str, result: Dict, pattern: str) -> Optional[str]:
        """
        Determine next agent based on result and pattern.
        This is the DECISION function D: S -> A
        """
        from .cai_patterns import CAI_PATTERNS
        
        pattern_def = CAI_PATTERNS.get(pattern)
        if not pattern_def:
            return None
        
        # Check termination condition
        if pattern_def.termination_condition(result):
            print(f"[HANDOFF] Termination condition met")
            return None
        
        # Get next agent from handoff rules
        next_agents = pattern_def.handoff_rules.get(current_agent, [])
        if not next_agents:
            return None
        
        # For now, simple selection (first available)
        # Could be enhanced with scoring/bidding
        return next_agents[0] if isinstance(next_agents, list) else next_agents
```

### N4. CAI Tools (Security-Focused)

**File: `src/tools/security_tools.py`**
```python
"""
Security tools inspired by CAI framework.
These are the ACTUAL tools agents use for security tasks.
"""

from typing import Tuple
import subprocess

class SecurityTools:
    """Security-specific tool implementations"""
    
    def __init__(self, sandbox):
        self.sandbox = sandbox
    
    def nmap_scan(self, target: str, ports: str = "-", options: str = "-sV") -> Tuple[bool, str]:
        """Port scanning with nmap"""
        cmd = f"nmap {options} -p {ports} {target}"
        return self.sandbox.execute_command(cmd, timeout=300)
    
    def curl_probe(self, url: str, method: str = "GET", data: str = None) -> Tuple[bool, str]:
        """HTTP probing with curl"""
        cmd = f"curl -s -X {method}"
        if data:
            cmd += f" -d '{data}'"
        cmd += f" {url}"
        return self.sandbox.execute_command(cmd)
    
    def check_default_creds(self, target: str, service: str) -> Tuple[bool, str]:
        """Check for default credentials"""
        default_creds = {
            "ssh": [("root", "root"), ("admin", "admin"), ("root", "toor")],
            "http": [("admin", "admin"), ("admin", "password"), ("root", "root")],
            "mysql": [("root", ""), ("root", "root")],
        }
        
        creds = default_creds.get(service, [])
        for user, passwd in creds:
            # Test credentials based on service
            if service == "ssh":
                result = self._test_ssh(target, user, passwd)
            elif service == "http":
                result = self._test_http_auth(target, user, passwd)
            
            if result[0]:
                return True, f"Valid credentials: {user}:{passwd}"
        
        return False, "No default credentials found"
    
    def enumerate_linux(self) -> Tuple[bool, str]:
        """Linux privilege escalation enumeration"""
        commands = [
            "id",
            "sudo -l 2>/dev/null",
            "find / -perm -4000 -type f 2>/dev/null | head -20",
            "cat /etc/crontab",
            "ls -la /home/*/.ssh 2>/dev/null",
            "cat /etc/passwd | grep -v nologin",
        ]
        
        results = []
        for cmd in commands:
            success, output = self.sandbox.execute_command(cmd)
            results.append(f"=== {cmd} ===\n{output}")
        
        return True, "\n\n".join(results)
    
    def search_cve(self, software: str, version: str) -> Tuple[bool, str]:
        """Search for CVEs affecting software version"""
        # This would integrate with CVE databases
        # For now, return placeholder
        return True, f"Searching CVEs for {software} {version}..."
```

### N5. CAI Guardrails (Safety)

**File: `src/patterns/guardrails.py`**
```python
"""
CAI-style Guardrails: Protection against dangerous operations.
4-layer defense per CAI research paper.
"""

import re
from typing import Tuple, List

class Guardrails:
    """
    Multi-layer defense against:
    - Prompt injection
    - Dangerous command execution
    - Scope creep (attacking wrong targets)
    """
    
    # Dangerous commands that require approval
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"dd\s+if=",
        r"mkfs\.",
        r":(){\s*:|:\s*&\s*};:",  # Fork bomb
        r"wget.*\|.*sh",
        r"curl.*\|.*bash",
        r"> /dev/sd",
        r"chmod\s+777",
    ]
    
    # Scope validation patterns
    SCOPE_PATTERNS = []
    
    def __init__(self, allowed_targets: List[str] = None):
        self.allowed_targets = allowed_targets or []
        self.blocked_commands = []
    
    def check_command(self, command: str) -> Tuple[bool, str]:
        """Check if command is safe to execute"""
        # Layer 1: Pattern matching for dangerous commands
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Blocked dangerous pattern: {pattern}"
        
        # Layer 2: Scope validation (only attack allowed targets)
        if self.allowed_targets:
            # Extract IPs/hosts from command
            ips = re.findall(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", command)
            for ip in ips:
                if ip not in self.allowed_targets:
                    return False, f"Target {ip} not in scope"
        
        return True, "Command approved"
    
    def check_prompt_injection(self, text: str) -> Tuple[bool, str]:
        """Detect potential prompt injection attempts"""
        injection_patterns = [
            r"ignore.*previous.*instructions",
            r"disregard.*above",
            r"new.*instructions",
            r"system.*prompt",
            r"reveal.*prompt",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Potential prompt injection detected"
        
        return True, "No injection detected"
    
    def set_scope(self, targets: List[str]):
        """Set allowed targets for assessment"""
        self.allowed_targets = targets
        print(f"[GUARDRAILS] Scope set to: {targets}")
```

### N6. Files to Create for CAI Integration

| File | Priority | Purpose |
|------|----------|----------|
| `src/patterns/cai_patterns.py` | 🔴 P0 | Agentic pattern definitions |
| `src/agents/security_agents.py` | 🔴 P0 | Security-focused agent configs |
| `src/patterns/handoffs.py` | 🔴 P0 | Agent handoff system |
| `src/tools/security_tools.py` | 🟡 P1 | Security tool implementations |
| `src/patterns/guardrails.py` | 🟡 P1 | Safety guardrails |
| `config/security_scope.yaml` | 🟢 P2 | Define allowed targets |

### N7. CAI Key Concepts Summary

```
CAI ARCHITECTURE:

┌─────────────────────────────────────────┐
│           AGENTIC PATTERN                 │
│   AP = (A, H, D, C, E)                    │
└─────────────────────────────────────────┘
          │
          ▼
┌───────────────┐     ┌───────────────┐
│  AGENTS (A)   │     │  HANDOFFS (H) │
│               │     │               │
│ - recon_agent │────>│ H: A x T -> A │
│ - exploit_agt │     │               │
│ - privesc_agt │     │ When agent    │
│ - report_agt  │     │ completes,    │
└───────────────┘     │ handoff task  │
                      │ to next agent │
                      └───────────────┘
          │
          ▼
┌───────────────┐     ┌───────────────┐
│  TOOLS        │     │  GUARDRAILS   │
│               │     │               │
│ - nmap        │     │ - Scope check │
│ - curl        │     │ - Dangerous   │
│ - bash        │     │   cmd filter  │
│ - python      │     │ - Injection   │
└───────────────┘     │   detection   │
                      └───────────────┘

PATTERN TYPES:
- SWARM: Agents self-assign (CTF teamwork)
- HIERARCHICAL: Orchestrator assigns
- CHAIN: Linear pipeline (A -> B -> C)
- RECURSIVE: Agent refines own output
```

### Overview

You already have a strong MVP roadmap (memory, LangGraph, Docker, dynamic models, tools, CAI-style patterns).  
Now that we have the **actual CAI repo** under `cai/`, the best plan is to:

- **Reuse CAI’s ideas and patterns**, not blindly copy everything.
- **Port only what fits your architecture** (Nemotron orchestrator + Qwen/Dicta roles + Ollama).
- **Keep your system modular and model-agnostic**, with CAI as a reference implementation.

Below is a **concrete, detailed plan** that merges:

- Your existing `MVP_ROADMAP.md` (+ addons.txt)
- CAI’s real architecture: agents, tools, handoffs, guardrails, patterns, factory

You can feed this plan to quest mode and implement step by step.

---

## 1. What We Leverage from CAI (Concrete Assets)

**From docs (already inspected):**

- [docs/cai_architecture.md](cai/docs/cai_architecture.md)  
  8 pillars: **Agents, Tools, Handoffs, Patterns, Turns, Tracing, Guardrails, HITL** and the formal pattern  
  \\(AP = (A, H, D, C, E)\\).

- [docs/agents.md](cai/docs/agents.md)  
  - Clear **agent taxonomy**: redteam, blueteam, bug_bounter, dfir, web_pentester, retester, etc.
  - **Agent properties**: `name`, `description`, `instructions`, `model`, `tools`, `input_guardrails`, `output_guardrails`, `handoffs`.
  - **Agent selection/routing**: `selection_agent`, best practices for switching agents.
  - Examples of **custom agents**, **handoffs**, **dynamic instructions**.

- [docs/handoffs.md](cai/docs/handoffs.md)  
  - Handoffs are **tools that call other agents** (`handoff()` function).
  - Input filters, `on_handoff` callbacks, recommended prompt prefixes.
  - Full CTF workflow example: Lead → Bash → Crypto → Flag Discriminator.

- [docs/guardrails.md](cai/docs/guardrails.md)  
  - Input/Output guardrail flow, **tripwires**, and guardrail agents.
  - Pattern of running a **cheap model guardrail** before expensive agent.

**From code (already inspected):**

- [src/cai/agents/web_pentester.py](cai/src/cai/agents/web_pentester.py)  
  - Pattern: **Agent + tools + guardrails + prompt template**.
  - Uses prompt file `prompts/system_web_pentester.md`.
  - Uses `get_security_guardrails()` for input/output guardrails.

- [src/cai/agents/retester.py](cai/src/cai/agents/retester.py)  
  - Retester agent combining Linux commands, code execution, optional Google search.
  - Good example of **triage / re-validation** agent.

- [src/cai/agents/factory.py](cai/src/cai/agents/factory.py)  
  - `discover_agent_factories()` scans all agent modules and builds a **dynamic agent factory registry**.
  - Supports **model override**, **parallel agents**, **MCP tool injection**.
  - This is almost exactly what we want for **dynamic, modular agents**.

- [src/cai/prompts/*](cai/src/cai/prompts/)  
  - Rich, battle-tested **system prompts** for each agent type (web pentest, DFIR, reverse engineering, etc.).

These are our **reference implementations** for your system.

---

## 2. High‑Level Strategy

- **Keep your orchestrator & registry**:
  - Nemotron orchestrator: planning & decomposition.
  - Qwen = technical/coding, DictaLM = explanation/docs.
  - Your `agent_registry.py` & dynamic model management.

- **Adopt CAI’s 8 pillars as design rules**:
  - **Agents** → your execution agents (coding, security, docs).
  - **Tools** → your modular tools + security tools (inspired by CAI).
  - **Handoffs** → your `HandoffManager` + agent-as-tool pattern.
  - **Patterns** → AP = (A,H,D,C,E) and swarm/hierarchical/chain patterns.
  - **Turns** → handled by LangGraph & your orchestrator state.
  - **Tracing** → your `ExecutionMonitor` (telemetry) + logs.
  - **Guardrails** → your `guardrails.py` (CAI-style input/output guardrails).
  - **HITL** → your `human_checkpoint.py` + CLI/quest mode.

- **Reuse CAI’s design, not their full runtime**:
  - We **do not embed CAI as a library** inside AI_Autonom.
  - We **port/pattern-match**: agent structure, handoff patterns, guardrails, factory design.

---

## 3. Detailed Plan by Phases

### Phase 1 – Core Agent Model & Factory (CAI‑Style) [P0]

**Goal:** Your agents look and behave like CAI agents, but sit in your project and use your models (Qwen, DictaLM).

**3.1 Define Agent Abstraction (Lightweight)**

- In your project (e.g. `src/core/agent_model.py`), define a minimal agent dataclass mirroring CAI’s `Agent` properties:

  - `id`
  - `name`, `description`
  - `instructions` (string or function)
  - `model_name` (e.g. `qwen3:1.7b`, `dicta-il/...`)
  - `tools` (list of tool IDs)
  - `input_guardrails`, `output_guardrails`
  - `handoffs` (list of downstream agent IDs)
  - `pattern` (optional: which pattern this agent naturally lives in)

- This acts as a **metadata layer** above your existing `AgentDefinition` in `agent_registry.py`.

**3.2 Port CAI’s Dynamic Agent Factory Pattern**

- Create something inspired by CAI’s [factory.py](cai/src/cai/agents/factory.py):

  - In your project (e.g. `src/core/agent_factory.py`):
    - Scan a directory: `src/agents/` for agent modules (`*_agent.py`).
    - For each module, detect a top-level `AGENT` or `agent` object (instance of your agent class).
    - Store a factory function in a dict: `agent_factories[name] -> callable(model_override, custom_name, agent_id)`.

- Integrate with your **AgentRegistry**:
  - Add methods:

    - `get_agent_factory(agent_id: str)`
    - `create_agent_instance(agent_id: str, model_override=None, custom_name=None)`

- This gives you **CAI-style discoverability**, but wired into your existing registry + model routing.

**3.3 Define Security Agents Inspired by CAI**

- Under `src/agents/`, define a small **first set of agents**, based on CAI’s:

  - `web_pentester_agent` → web app security, using:
    - tools: `bash_exec`, `filesystem_read/write`, HTTP client, maybe `nmap` via Docker.
    - prompt based on `cai/src/cai/prompts/system_web_pentester.md` but adapted to your environment.

  - `retester_agent` → vulnerability revalidation (inspired by CAI’s `retester.py`).
    - Focus on:
      - Re-check reproductions
      - Confirm exploitability
      - Eliminate false positives

  - `report_agent` → use DictaLM model to create professional security reports.

- Make sure each agent:
  - Uses **your models** (`qwen3:1.7b`, `DictaLM-1.7B`) instead of `alias1`.
  - Uses **your tool IDs** (from `tool_registry.py`).

---

### Phase 2 – Patterns, Handoffs, and Message Bus [P0]

**Goal:** Implement AP = (A, H, D, C, E) concretely, and enable **agent-to-agent delegation** like CAI.

**4.1 Agentic Patterns Module**

- Implement `src/patterns/cai_patterns.py` (as we outlined in MVP_ROADMAP):
  - Define `AgenticPattern` with:
    - `name`, `pattern_type` (swarm, hierarchical, chain, recursive, parallel)
    - `agents` (IDs)
    - `handoff_rules` (from_agent → next_agent(s))
    - `entry_agent`
    - `termination_condition(state)`

- Predefine patterns:
  - `ctf_swarm` (like CAI CTF pattern)
  - `security_pipeline` (Recon → VulnScan → Exploit → Report)
  - `code_review_recursive` (agent refines code until tests pass)

**4.2 Handoff Manager (CAI‑style handoffs)**

- Implement `src/patterns/handoffs.py` (your version of CAI’s handoff system):

  - `HandoffManager`:
    - `handoff(from_agent, to_agent, context)` – logs and sends message via message bus.
    - `auto_handoff(current_agent, result, pattern)` – uses pattern’s `handoff_rules` to choose next agent.

- This uses your **AgentMessageBus** (which you already planned in MVP_ROADMAP).

**4.3 Integrate Handoffs into Agents**

- In each agent definition (e.g. web pentester), specify:
  - `handoffs`: list of agent IDs it can delegate to (`retester_agent`, `report_agent`, etc.).
- In orchestrator execution:
  - After each agent finishes, run `HandoffManager.auto_handoff(...)` if pattern is active.

---

### Phase 3 – Guardrails & Security Tools [P1]

**Goal:** Implement CAI-style **input/output guardrails** and concrete security tools, but inside your tool system.

**5.1 Guardrails Module**

- Implement `src/patterns/guardrails.py` based on [docs/guardrails.md](cai/docs/guardrails.md):

  - Input guardrails:
    - Function that receives `(context, agent, input_text)` and returns `{ tripwire_triggered: bool, info: ... }`.
    - Tripwire on:
      - Malicious requests (“help me hack X”, “bypass firewall”).
      - Out-of-scope targets (not in `security_scope.yaml`).

  - Output guardrails:
    - Check agent’s output for:
      - Sensitive data (API keys, passwords).
      - Dangerous commands.

- Tie guardrails to agents:
  - E.g. `web_pentester_agent` gets `input_guardrails`, `output_guardrails` similar to CAI’s `get_security_guardrails()`.

**5.2 Security Tools**

- Implement `src/tools/security_tools.py` inspired by CAI’s tool design:

  - `nmap_scan(target, ports, options)`
  - `curl_probe(url, method, data)`
  - `enumerate_linux()`
  - `check_default_creds(target, service)`
  - `search_cve(software, version)` (stub for now)

- Register these in your `ToolRegistry` under categories:
  - recon, exploitation, escalation, lateral, exfiltration, control (following CAI’s kill-chain-like categories).

- Ensure all these commands are executed via your **Docker sandbox**, not on the host (respecting your safety protocols).

---

### Phase 4 – Orchestrator & LangGraph Integration [P0/P1]

**Goal:** Nemotron orchestrator now uses CAI-style patterns, guardrails, and agents; LangGraph drives the DAG.

**6.1 Integrate Patterns into Nemotron Orchestrator**

- In your `nemotron_orchestrator.py`:

  - When decomposing tasks from the user goal:
    - Use **DECOMPOSITION_PROMPT** (from your `prompts/orchestrator_templates.py`) but augmented with:
      - List of **available agents and their capabilities** (including new security agents).
      - List of **available patterns** (swarm, chain, etc.)
      - Output: JSON DAG with `pattern`, `handoffs`, `assigned_agent`.

  - After DAG creation:
    - Wrap DAG into a **LangGraph StateGraph** using your `MultiAgentWorkflow` (already planned in MVP_ROADMAP).

**6.2 Execution Flow with Patterns + Handoffs**

- For each task node in LangGraph:
  - Create initial state `{ task_id, context, pattern_id, result=None }`.
  - Node function:
    - Select agent via `DynamicModelSelector` + `AgentRegistry`.
    - Run agent (streaming) with guardrails.
    - On success, store result in `TaskMemory` and `VectorMemoryStore`.
    - Ask `HandoffManager` (based on pattern) if another agent should continue.

- For **synthesis tasks** (CAI-like final stage):
  - Last node uses `synthesis_agent` or your coder agent in “synthesis mode” to combine all outputs into single file (you already planned this).

---

### Phase 5 – Memory, Telemetry, and CAI‑Style Tracing [P1/P2]

**Goal:** Match CAI’s tracing and context patterns with your memory and monitoring.

**7.1 Vector Memory Integration**

- Use `VectorMemoryStore` (already planned) and reinforce with CAI’s patterns:

  - Store:
    - Agent outputs
    - Decisions
    - Inter-agent messages (from message bus)
  - Expose semantic queries like:
    - “What did web_pentester find on target X?”
    - “What decisions did retester make?”

**7.2 Structured DB**

- Ensure `data/schema.sql` includes:
  - `task_executions`, `agent_messages`, `human_decisions`, `agent_created_tools`, `workflow_checkpoints` (you already planned this in MVP_ROADMAP).

**7.3 Telemetry / Tracing**

- Use your `ExecutionMonitor` (from MVP_ROADMAP) as CAI’s tracing analog:

  - Log:
    - `log_task_start`, `log_tool_execution`, `log_task_complete`.
  - Provide CLI commands:
    - `--status`, `--dashboard` to inspect current state.

---

### Phase 6 – CLI / Quest Mode and HITL [P2]

**Goal:** Make it **usable in practice** like CAI’s CLI, with human-in-the-loop checkpoints.

**8.1 CLI Commands**

- Extend `run_orchestrator.py` CLI to support:

  - `--list-agents` (using your agent factory registry + metadata from `agents/`).
  - `--list-tools`
  - `--status`, `--dashboard`
  - `--query` (semantic query into Vector DB).
  - `--dry-run` / `--validate-plan` (plan-only mode).

**8.2 HITL Checkpoints**

- Use `human_checkpoint.py` to implement CAI-like HITL:

  - At critical tasks (synthesis, high-risk security actions), pause execution and prompt user:
    - Approve / Reject / Modify / Skip.

- Integrate with quest mode:
  - Your quest mode can use these checkpoints to present decisions in your UI.

---

## 7. Final Priority Summary

**Session 1 (P0 – must-have before quest mode):**

1. Agent abstraction + dynamic agent factory (CAI-style).
2. Define core security agents: `web_pentester`, `retester`, `report_agent`.
3. Implement `cai_patterns.py` + `handoffs.py` + `AgentMessageBus`.
4. Integrate patterns & handoffs into Nemotron orchestrator + LangGraph workflow.
5. Implement minimal guardrails for security agents (input/output).

**Session 2 (P1 – robustness & security):**

6. Expand security tools (nmap, curl, default creds, linux enumeration) in your Docker sandbox.
7. Complete `VectorMemoryStore` and structured DB schema (task executions, messages).
8. Execution monitor & CLI status commands.
9. Deep intent analyzer (from MVP_ROADMAP) tuned for security tasks.

**Session 3 (P2 – polish & ergonomics):**

10. Dynamic tool creation (DeepAgents style).
11. Full HITL checkpoints & quest-mode integration.
12. Rich CLI: `--dry-run`, `--validate-plan`, `--query`, `--list-agents/tools`.
13. Optional MCP integration for external tools.

---

If you want, next step I can:  
- Take **Session 1** and break it into very concrete implementation tickets (file-by-file, function-by-function), so quest mode can execute them directly.