# Memory Module

## Purpose
The memory module provides persistent context storage for multi-agent workflows. It includes task memory for inter-task context sharing and vector memory for semantic search across historical task executions.

## Key Files

### `task_memory.py`
**Purpose:** Persistent task context storage enabling agents to share information across workflow steps.

**Key Classes:**
- `TaskMemory`: Task context management
  - `store_task_context(task_id, context)`: Store task context
  - `load_task_context(task_id)`: Load task context
  - `append_agent_output(task_id, agent_name, output)`: Add agent output
  - `get_agent_outputs(task_id)`: Get all agent outputs for task
  - `get_workflow_metadata(task_id)`: Get workflow metadata
  - `store_workflow_metadata(task_id, metadata)`: Store workflow info
  - `get_related_tasks(task_id, similarity_threshold)`: Find similar tasks
  - `search_tasks(query)`: Search tasks by content
  - `get_task_history(limit)`: Get recent task history
  - `cleanup_old_tasks(days)`: Remove old task data

**Task Context Structure:**
```python
{
    "task_id": "unique_identifier",
    "goal": "user_goal",
    "pattern": "CHAIN",
    "agents": ["agent1", "agent2"],
    "status": "completed",
    "agent_outputs": [
        {"agent": "agent1", "output": "...", "timestamp": "..."},
        {"agent": "agent2", "output": "...", "timestamp": "..."}
    ],
    "metadata": {
        "started_at": "2024-01-17T10:00:00",
        "completed_at": "2024-01-17T10:05:00",
        "error_count": 0,
        "checkpoint_count": 2
    }
}
```

**Dependencies:**
- External: `sqlite3`, `json`, `datetime`, `logging`
- Internal: None

### `vector_memory.py`
**Purpose:** Semantic search over task history using vector embeddings with ChromaDB.

**Key Classes:**
- `VectorMemory`: Vector-based semantic search
  - `store_task_embedding(task_id, task_description, metadata)`: Store task embedding
  - `search_similar_tasks(query, top_k)`: Find similar tasks by semantic similarity
  - `get_task_context(task_id)`: Retrieve task context
  - `delete_task(task_id)`: Remove task from vector store
  - `get_collection_stats()`: Get collection statistics
  - `cleanup_old_embeddings(days)`: Remove old embeddings

**Vector Storage Structure:**
```python
# Each document in ChromaDB
{
    "id": "task_12345",
    "embedding": [0.123, 0.456, ...],  # 768-dim vector
    "document": "Perform security assessment of web application",
    "metadata": {
        "task_id": "task_12345",
        "pattern": "CTF_SECURITY",
        "agents": ["web_pentester", "retester"],
        "timestamp": "2024-01-17T10:00:00",
        "status": "completed"
    }
}
```

**Dependencies:**
- External: `chromadb`, `logging`, `datetime`
- Internal: `ai_autonom.core.llm_provider` (for embeddings)

## Internal Architecture

```
┌──────────────────────────────────────────┐
│         TaskMemory                        │
│  (Structured context storage)             │
│  - SQLite persistence                     │
│  - Task context                           │
│  - Agent outputs                          │
│  - Workflow metadata                      │
└─────────────┬────────────────────────────┘
              │
              │ provides data to
              ▼
┌──────────────────────────────────────────┐
│         VectorMemory                      │
│  (Semantic search)                        │
│  - ChromaDB storage                       │
│  - Embedding generation                   │
│  - Similarity search                      │
│  - Metadata filtering                     │
└──────────────────────────────────────────┘
              │
              │ uses
              ▼
┌──────────────────────────────────────────┐
│     LLM Provider (Embeddings)             │
│  - Generate embeddings                    │
│  - Semantic similarity                    │
└──────────────────────────────────────────┘
```

## Usage Examples

### Task Memory
```python
from ai_autonom.memory.task_memory import TaskMemory

# Initialize
memory = TaskMemory()

# Store task context
context = {
    "task_id": "task_001",
    "goal": "Perform security assessment",
    "pattern": "CTF_SECURITY",
    "agents": ["web_pentester", "retester", "report_agent"],
    "status": "in_progress"
}
memory.store_task_context("task_001", context)

# Append agent outputs
memory.append_agent_output(
    task_id="task_001",
    agent_name="web_pentester",
    output={"findings": ["XSS", "SQLi"], "severity": "HIGH"}
)

# Load task context
loaded = memory.load_task_context("task_001")
print(loaded["agent_outputs"])

# Store workflow metadata
metadata = {
    "started_at": "2024-01-17T10:00:00",
    "pattern_iterations": 3,
    "checkpoints_triggered": 2
}
memory.store_workflow_metadata("task_001", metadata)

# Search tasks
results = memory.search_tasks("security assessment")
for task_id, context in results:
    print(f"Task: {task_id}, Goal: {context['goal']}")

# Get related tasks
related = memory.get_related_tasks("task_001", similarity_threshold=0.8)
```

### Vector Memory
```python
from ai_autonom.memory.vector_memory import VectorMemory

# Initialize
vector_memory = VectorMemory()

# Store task embedding
vector_memory.store_task_embedding(
    task_id="task_001",
    task_description="Perform security assessment of login page",
    metadata={
        "pattern": "CTF_SECURITY",
        "agents": ["web_pentester", "retester"],
        "status": "completed"
    }
)

# Search similar tasks
similar = vector_memory.search_similar_tasks(
    query="Find vulnerabilities in authentication",
    top_k=5
)
for result in similar:
    print(f"Task: {result['task_id']}")
    print(f"Similarity: {result['distance']}")
    print(f"Description: {result['document']}")

# Get collection stats
stats = vector_memory.get_collection_stats()
print(f"Total tasks: {stats['count']}")

# Cleanup old embeddings
vector_memory.cleanup_old_embeddings(days=30)
```

### Combined Usage
```python
from ai_autonom.memory.task_memory import TaskMemory
from ai_autonom.memory.vector_memory import VectorMemory

# Initialize both
task_memory = TaskMemory()
vector_memory = VectorMemory()

# Store task in both systems
task_id = "task_002"
context = {
    "task_id": task_id,
    "goal": "Generate authentication module",
    "pattern": "RECURSIVE",
    "agents": ["code_generator", "code_reviewer"]
}

# Structured storage
task_memory.store_task_context(task_id, context)

# Vector storage for semantic search
vector_memory.store_task_embedding(
    task_id=task_id,
    task_description=context["goal"],
    metadata={"pattern": context["pattern"], "agents": context["agents"]}
)

# Later: Find similar tasks semantically
similar_tasks = vector_memory.search_similar_tasks(
    "create login system",
    top_k=3
)

# Get full context from task memory
for result in similar_tasks:
    full_context = task_memory.load_task_context(result["task_id"])
    print(f"Found related task: {full_context['goal']}")
```

## Dependencies

**External Dependencies:**
- `sqlite3`: Structured task context storage
- `chromadb`: Vector database for semantic search
- `json`: Data serialization
- `datetime`: Timestamp handling
- `logging`: Event logging

**Internal Dependencies:**
- `ai_autonom.core.llm_provider`: Embedding generation for vector search

## Important Functionality

1. **Persistent Context**: Task context survives across workflow executions
2. **Agent Output History**: Complete audit trail of agent contributions
3. **Semantic Search**: Find similar tasks using natural language queries
4. **Workflow Metadata**: Track execution metrics (duration, errors, checkpoints)
5. **Task Relationships**: Discover related tasks automatically
6. **Cleanup Utilities**: Remove old data to manage storage
7. **Fast Retrieval**: Indexed queries for quick context loading
8. **Metadata Filtering**: Filter vector search by pattern, agents, status
9. **Similarity Scoring**: Distance metrics for ranking search results
10. **Incremental Updates**: Append agent outputs without full context reload

## Memory Storage Locations

- **Task Memory Database**: `.runtime/data/task_memory.db` (SQLite)
- **Vector Memory Database**: `.runtime/chromadb/` (ChromaDB persistent storage)
- **Logs**: `.runtime/logs/memory.log`

## Data Retention Policies

Implement cleanup strategies based on:
- **Age**: Remove tasks older than N days
- **Storage Limit**: Keep most recent N tasks
- **Task Status**: Remove failed/cancelled tasks after shorter period
- **Access Patterns**: Remove tasks not accessed in N days

## Performance Considerations

1. **Task Memory**:
   - Indexed by `task_id` for O(1) retrieval
   - Full-text search on task goals and agent outputs
   - Batch operations for multiple agent outputs

2. **Vector Memory**:
   - Embedding caching to avoid regeneration
   - Top-K search with configurable limit
   - Metadata pre-filtering before similarity computation
   - Collection statistics for monitoring

## Use Cases

1. **Context Continuity**: Agents access previous task outputs
2. **Learning from History**: Discover successful patterns from past tasks
3. **Error Recovery**: Reference similar successful tasks when recovering from errors
4. **Workflow Optimization**: Analyze workflow metadata to identify bottlenecks
5. **Agent Collaboration**: Share intermediate results between agents
6. **Checkpoint Resume**: Resume workflows from checkpoints using stored context
7. **Pattern Selection**: Recommend patterns based on similar historical tasks
8. **Knowledge Base**: Build organizational knowledge from completed tasks
