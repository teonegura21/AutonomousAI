# Patterns Module

## Purpose
The patterns module implements formal Cybersecurity AI (CAI) agentic patterns and agent handoff mechanisms. It provides structured coordination patterns for multi-agent workflows including SWARM, HIERARCHICAL, CHAIN, AUCTION, RECURSIVE, PARALLEL, and CTF_SECURITY patterns.

## Key Files

### `cai_patterns.py`
**Purpose:** Implementation of formal CAI agentic patterns with rigorous mathematical definitions.

**Key Classes:**
- `AgenticPattern`: Enum defining available patterns
  - SWARM: Decentralized collaboration
  - HIERARCHICAL: Top-down command structure
  - CHAIN: Sequential agent chaining
  - AUCTION: Bid-based task allocation
  - RECURSIVE: Self-referential problem decomposition
  - PARALLEL: Concurrent execution
  - CTF_SECURITY: Cybersecurity-specific pattern

- `PatternConfig`: Configuration for pattern execution
  - `pattern`: Pattern type
  - `max_iterations`: Maximum iterations
  - `timeout`: Execution timeout
  - `callbacks`: Callback functions
  - `metadata`: Additional metadata

- `PatternExecutor`: Pattern execution engine
  - `execute_pattern(pattern, task, agents)`: Execute specific pattern
  - `execute_swarm(task, agents, config)`: Swarm pattern execution
  - `execute_hierarchical(task, agents, config)`: Hierarchical execution
  - `execute_chain(task, agents, config)`: Chain execution
  - `execute_auction(task, agents, config)`: Auction-based allocation
  - `execute_recursive(task, agents, config)`: Recursive execution
  - `execute_parallel(task, agents, config)`: Parallel execution
  - `execute_ctf_security(task, agents, config)`: CTF security pattern

**Key Functions:**
- `create_security_pipeline(task)`: Creates security assessment pipeline (web_pentester → retester → report_agent)
- `create_code_review_pipeline(code)`: Creates code review pipeline with recursive quality improvements

**Dependencies:**
- External: `asyncio`, `concurrent.futures`, `json`, `time`
- Internal: `ai_autonom.agents.agent_registry`, `ai_autonom.core.llm_provider`, `ai_autonom.core.config`

### `agent_handoff.py`
**Purpose:** Agent-to-agent delegation and context transfer with message bus integration.

**Key Classes:**
- `HandoffType`: Enum for handoff types
  - SEQUENTIAL: Sequential delegation
  - CONDITIONAL: Conditional branching
  - PARALLEL: Parallel execution
  - CALLBACK: Callback-based
  - ESCALATION: Error escalation
  - DELEGATION: Task delegation
  - COLLABORATION: Collaborative work

- `HandoffCondition`: Condition evaluation for handoffs
  - `evaluate(context)`: Evaluate condition against context
  - Supports operators: ==, !=, >, <, >=, <=, contains, not_contains

- `AgentHandoff`: Handoff definition
  - `from_agent`: Source agent
  - `to_agent`: Target agent
  - `handoff_type`: Type of handoff
  - `condition`: Optional condition
  - `priority`: Execution priority
  - `input_filter`: Filter for input data
  - `callback`: Callback function

- `HandoffManager`: Manages agent handoffs and delegation
  - `register_handoff(handoff)`: Register handoff rule
  - `execute_handoff(from_agent, context)`: Execute appropriate handoff
  - `find_next_agent(from_agent, context)`: Find next agent in chain
  - `get_handoff_chain(start_agent)`: Get full handoff chain
  - `validate_handoff_chain(chain)`: Validate chain for cycles

- `MessageBus`: Event-driven message passing
  - `publish(topic, message)`: Publish message to topic
  - `subscribe(topic, callback)`: Subscribe to topic
  - `unsubscribe(topic, callback)`: Unsubscribe from topic

**Dependencies:**
- External: `dataclasses`, `typing`, `json`, `logging`
- Internal: `ai_autonom.agents.agent_registry`

## Internal Architecture

```
┌─────────────────────────────────────────┐
│       PatternExecutor                    │
│  (Pattern execution engine)              │
│  - SWARM, HIERARCHICAL, CHAIN           │
│  - AUCTION, RECURSIVE, PARALLEL         │
│  - CTF_SECURITY                         │
└─────────────────┬───────────────────────┘
                  │
                  │ uses
                  ▼
┌─────────────────────────────────────────┐
│       HandoffManager                     │
│  (Agent delegation & coordination)       │
│  - Handoff registration                  │
│  - Condition evaluation                  │
│  - Chain validation                      │
│  - Context transfer                      │
└─────────────────┬───────────────────────┘
                  │
                  │ publishes to
                  ▼
┌─────────────────────────────────────────┐
│       MessageBus                         │
│  (Event-driven messaging)                │
│  - Topic-based pub/sub                   │
│  - Async event handling                  │
└─────────────────────────────────────────┘
```

## Usage Examples

### CAI Patterns
```python
from ai_autonom.patterns.cai_patterns import PatternExecutor, AgenticPattern, PatternConfig
from ai_autonom.agents.agent_registry import AgentRegistry

# Initialize
executor = PatternExecutor()
registry = AgentRegistry()

# Security pipeline pattern
result = executor.execute_pattern(
    pattern=AgenticPattern.CTF_SECURITY,
    task="Assess https://example.com",
    agents=["web_pentester", "retester", "report_agent"],
    config=PatternConfig(max_iterations=5, timeout=300)
)

# Code review recursive pattern
result = executor.execute_pattern(
    pattern=AgenticPattern.RECURSIVE,
    task="Review and improve code quality",
    agents=["code_reviewer", "test_generator"],
    config=PatternConfig(max_iterations=3)
)

# Parallel execution pattern
result = executor.execute_pattern(
    pattern=AgenticPattern.PARALLEL,
    task="Analyze multiple components",
    agents=["agent1", "agent2", "agent3"],
    config=PatternConfig(timeout=60)
)
```

### Agent Handoffs
```python
from ai_autonom.patterns.agent_handoff import (
    HandoffManager, AgentHandoff, HandoffType, HandoffCondition
)

# Initialize manager
manager = HandoffManager()

# Register sequential handoff
handoff = AgentHandoff(
    from_agent="web_pentester",
    to_agent="retester",
    handoff_type=HandoffType.SEQUENTIAL,
    input_filter=lambda ctx: {"vulnerabilities": ctx.get("findings", [])}
)
manager.register_handoff(handoff)

# Register conditional handoff
condition = HandoffCondition(
    field="severity",
    operator=">=",
    value="HIGH"
)
escalation = AgentHandoff(
    from_agent="retester",
    to_agent="report_agent",
    handoff_type=HandoffType.CONDITIONAL,
    condition=condition
)
manager.register_handoff(escalation)

# Execute handoff
context = {"findings": ["XSS", "SQLi"], "severity": "HIGH"}
result = manager.execute_handoff("web_pentester", context)

# Get handoff chain
chain = manager.get_handoff_chain("web_pentester")
print(chain)  # ["web_pentester", "retester", "report_agent"]
```

### Message Bus
```python
from ai_autonom.patterns.agent_handoff import MessageBus

# Initialize bus
bus = MessageBus()

# Subscribe to events
def on_vulnerability_found(message):
    print(f"Vulnerability: {message['type']}")

bus.subscribe("security.vulnerability", on_vulnerability_found)

# Publish events
bus.publish("security.vulnerability", {
    "type": "XSS",
    "severity": "HIGH",
    "location": "/login"
})

# Unsubscribe
bus.unsubscribe("security.vulnerability", on_vulnerability_found)
```

## Dependencies

**External Dependencies:**
- `asyncio`: Async execution for parallel patterns
- `concurrent.futures`: Thread pool for parallel execution
- `json`: Data serialization
- `logging`: Event logging

**Internal Dependencies:**
- `ai_autonom.agents.agent_registry`: Agent discovery and management
- `ai_autonom.core.llm_provider`: LLM integration for pattern execution
- `ai_autonom.core.config`: Configuration access

## Important Functionality

1. **Formal CAI Patterns**: Mathematically rigorous pattern implementations
2. **Pattern Composition**: Combine multiple patterns in workflows
3. **Conditional Handoffs**: Context-aware agent delegation
4. **Cycle Detection**: Prevents infinite handoff loops
5. **Input Filtering**: Transform context data between agents
6. **Priority-Based Execution**: Handle multiple handoff candidates
7. **Event-Driven Messaging**: Async pub/sub for loose coupling
8. **Recursive Workflows**: Self-improving agent chains
9. **Parallel Execution**: Concurrent agent coordination
10. **Security-Specific Patterns**: CTF and security assessment workflows

## Pattern Selection Guide

| Pattern | Use Case | Example |
|---------|----------|---------|
| SWARM | Decentralized collaboration | Multi-agent brainstorming |
| HIERARCHICAL | Command hierarchy | Manager-worker delegation |
| CHAIN | Sequential processing | Security pipeline (scan → retest → report) |
| AUCTION | Resource optimization | Task allocation by capability |
| RECURSIVE | Iterative improvement | Code review with refinement |
| PARALLEL | Independent tasks | Multi-target security scans |
| CTF_SECURITY | Security assessment | Penetration testing workflow |

## Handoff Types

| Type | Description | Example |
|------|-------------|---------|
| SEQUENTIAL | Always proceed to next agent | web_pentester → retester |
| CONDITIONAL | Branch based on condition | severity >= HIGH → escalate |
| PARALLEL | Fork to multiple agents | findings → [agent1, agent2] |
| CALLBACK | Return to previous agent | error → retry previous |
| ESCALATION | Escalate to higher authority | critical → senior_agent |
| DELEGATION | Delegate subtask | complex_task → specialist |
| COLLABORATION | Joint work | [agent1, agent2] → shared_output |
