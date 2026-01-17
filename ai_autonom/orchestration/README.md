# Orchestration Module

## Purpose
The orchestration module coordinates multi-agent workflows, manages task execution, implements error recovery, provides human-in-the-loop checkpoints, and automates testing workflows. It acts as the central coordination layer integrating all system components.

## Key Files

### `workflow_orchestrator.py`
**Purpose:** Central orchestration engine coordinating agents, patterns, memory, and tools.

**Key Classes:**
- `WorkflowOrchestrator`: Main orchestration class
  - `execute_workflow(task, pattern, agents)`: Execute complete workflow
  - `_select_pattern(task)`: Intelligent pattern selection
  - `_select_agents(task)`: Agent selection based on capabilities
  - `_execute_with_pattern(task, pattern, agents)`: Pattern-based execution
  - `_handle_agent_execution(agent, task, context)`: Individual agent execution
  - `_store_context(task_id, context)`: Persist task context
  - `_load_context(task_id)`: Retrieve task context

**Dependencies:**
- External: `logging`, `json`, `time`
- Internal: 
  - `ai_autonom.patterns.cai_patterns`: Pattern execution
  - `ai_autonom.agents.agent_registry`: Agent management
  - `ai_autonom.memory.task_memory`: Context storage
  - `ai_autonom.core.llm_provider`: LLM integration
  - `ai_autonom.orchestration.error_recovery`: Error handling
  - `ai_autonom.orchestration.human_checkpoint`: HITL checkpoints
  - `ai_autonom.orchestration.testing_workflow`: Test automation

### `intent_analyzer.py`
**Purpose:** Analyze user intent and determine optimal execution strategy.

**Key Classes:**
- `IntentType`: Enum for intent categories
  - SECURITY_ASSESSMENT
  - CODE_GENERATION
  - CODE_REVIEW
  - DEBUGGING
  - TESTING
  - ANALYSIS
  - GENERAL

- `IntentAnalyzer`: Analyzes user goals and selects patterns/agents
  - `analyze(goal)`: Analyze user goal and return intent
  - `_extract_intent(goal)`: Extract intent from natural language
  - `_extract_entities(goal, intent_type)`: Extract relevant entities
  - `_suggest_pattern(intent_type)`: Recommend pattern
  - `_suggest_agents(intent_type, entities)`: Recommend agents

**Dependencies:**
- External: `json`, `re`
- Internal: 
  - `ai_autonom.core.llm_provider`: LLM for intent analysis
  - `ai_autonom.patterns.cai_patterns`: Pattern types
  - `ai_autonom.agents.agent_registry`: Agent registry

### `error_recovery.py`
**Purpose:** Multi-strategy error recovery with retry logic and agent swapping.

**Key Classes:**
- `ErrorType`: Enum for error categories
  - LLM_ERROR: LLM API failures
  - TOOL_ERROR: Tool execution failures
  - TIMEOUT: Timeout errors
  - VALIDATION_ERROR: Output validation failures
  - AGENT_ERROR: Agent execution failures
  - PATTERN_ERROR: Pattern execution failures
  - UNKNOWN: Unknown errors

- `RecoveryStrategy`: Enum for recovery approaches
  - RETRY: Simple retry
  - EXPONENTIAL_BACKOFF: Retry with backoff
  - FALLBACK_MODEL: Switch to different model
  - AGENT_SWAP: Switch to different agent
  - PROMPT_REFINEMENT: Refine prompt and retry
  - SKIP: Skip failed step
  - ESCALATE: Escalate to human

- `ErrorRecovery`: Error handling and recovery
  - `handle_error(error, context)`: Handle error with recovery
  - `_classify_error(error)`: Classify error type
  - `_select_strategy(error_type, context)`: Select recovery strategy
  - `_execute_retry(context)`: Execute retry
  - `_execute_exponential_backoff(context)`: Retry with backoff
  - `_execute_fallback_model(context)`: Switch model
  - `_execute_agent_swap(context)`: Switch agent
  - `_execute_prompt_refinement(context)`: Refine prompt
  - `_log_error(error, strategy, success)`: Log recovery attempt

**Dependencies:**
- External: `time`, `logging`, `sqlite3`, `json`
- Internal:
  - `ai_autonom.core.model_manager`: Model switching
  - `ai_autonom.agents.agent_registry`: Agent switching

### `human_checkpoint.py`
**Purpose:** Human-in-the-loop approval system for critical decisions.

**Key Classes:**
- `CheckpointType`: Enum for checkpoint categories
  - CRITICAL: Critical actions requiring approval
  - SYNTHESIS: Synthesis/aggregation checkpoints
  - ERROR: Error recovery decisions
  - VALIDATION: Output validation
  - CONFIGURATION: Configuration changes

- `CheckpointDecision`: Enum for user decisions
  - APPROVE: Approve and continue
  - REJECT: Reject and stop
  - MODIFY: Modify and retry
  - SKIP: Skip checkpoint

- `RiskLevel`: Enum for risk assessment
  - LOW: Auto-approved
  - MEDIUM: Optional approval
  - HIGH: Required approval
  - CRITICAL: Required with confirmation

- `HumanCheckpointManager`: Manages checkpoint workflow
  - `create_checkpoint(checkpoint_type, context)`: Create checkpoint
  - `request_approval(checkpoint_id)`: Request human approval
  - `auto_approve_low_risk(checkpoint)`: Auto-approve low-risk actions
  - `get_checkpoint_history(limit)`: Get approval history
  - `get_checkpoint_statistics()`: Get checkpoint stats

**Key Functions:**
- `create_critical_action_checkpoint(action, risk_level)`: Create action checkpoint
- `create_synthesis_checkpoint(results, synthesis)`: Create synthesis checkpoint
- `create_error_recovery_checkpoint(error, strategy)`: Create error checkpoint

**Dependencies:**
- External: `sqlite3`, `json`, `datetime`, `logging`
- Internal: None

### `testing_workflow.py`
**Purpose:** Automated test generation, execution, and quality validation.

**Key Classes:**
- `TestType`: Enum for test categories
  - UNIT: Unit tests
  - INTEGRATION: Integration tests
  - FUNCTIONAL: Functional tests
  - SECURITY: Security tests
  - PERFORMANCE: Performance tests

- `TestStatus`: Enum for test execution status
  - PENDING: Not yet executed
  - RUNNING: Currently executing
  - PASSED: Test passed
  - FAILED: Test failed
  - SKIPPED: Test skipped
  - ERROR: Execution error

- `TestGenerator`: Generates tests from code
  - `generate_tests(code, test_type)`: Generate tests
  - `_analyze_code(code)`: Analyze code structure
  - `_generate_test_code(analysis, test_type)`: Create test code

- `TestRunner`: Executes tests
  - `run_tests(test_file)`: Run test file
  - `_execute_pytest(test_file)`: Execute with pytest
  - `_parse_pytest_output(output)`: Parse test results

- `TestingWorkflow`: Complete testing workflow
  - `add_testing_tasks(task)`: Add test tasks for coding task
  - `execute_testing_workflow(code, test_types)`: Full test workflow
  - `_store_test_suite(suite)`: Persist test suite
  - `_store_test_results(results)`: Persist test results

**Key Functions:**
- `validate_code_quality(test_results)`: Validate code meets quality gates

**Dependencies:**
- External: `sqlite3`, `json`, `subprocess`, `re`, `logging`
- Internal:
  - `ai_autonom.core.llm_provider`: LLM for test generation
  - `ai_autonom.agents.agent_registry`: Test generator agent

## Internal Architecture

```
┌─────────────────────────────────────────────────────┐
│            WorkflowOrchestrator                      │
│  (Central coordination)                              │
│  - Task routing                                      │
│  - Pattern selection                                 │
│  - Agent coordination                                │
└─────┬─────────────┬──────────────┬──────────────────┘
      │             │              │
      ▼             ▼              ▼
┌───────────┐ ┌─────────┐ ┌──────────────────┐
│  Intent   │ │ Error   │ │ Human            │
│  Analyzer │ │Recovery │ │ Checkpoint       │
│           │ │         │ │                  │
└───────────┘ └─────────┘ └──────────────────┘
      │             │              │
      │             │              │
      └─────────────┴──────────────┘
                    │
                    ▼
      ┌──────────────────────────┐
      │   Testing Workflow        │
      │  - Test generation        │
      │  - Test execution         │
      │  - Quality validation     │
      └───────────────────────────┘
```

## Usage Examples

### Workflow Orchestration
```python
from ai_autonom.orchestration.workflow_orchestrator import WorkflowOrchestrator
from ai_autonom.patterns.cai_patterns import AgenticPattern

# Initialize orchestrator
orchestrator = WorkflowOrchestrator()

# Execute workflow
result = orchestrator.execute_workflow(
    task="Perform security assessment of https://example.com",
    pattern=AgenticPattern.CTF_SECURITY,
    agents=["web_pentester", "retester", "report_agent"]
)
```

### Intent Analysis
```python
from ai_autonom.orchestration.intent_analyzer import IntentAnalyzer

# Analyze user intent
analyzer = IntentAnalyzer()
intent = analyzer.analyze("Review my authentication code for security issues")

print(intent.intent_type)  # IntentType.CODE_REVIEW
print(intent.suggested_pattern)  # AgenticPattern.RECURSIVE
print(intent.suggested_agents)  # ["code_reviewer"]
```

### Error Recovery
```python
from ai_autonom.orchestration.error_recovery import ErrorRecovery

# Handle error
recovery = ErrorRecovery()
context = {
    "task": "generate_code",
    "agent": "code_generator",
    "attempt": 1
}

try:
    # ... some operation ...
    pass
except Exception as e:
    result = recovery.handle_error(e, context)
    print(result["strategy"])  # RecoveryStrategy.RETRY
    print(result["success"])   # True/False
```

### Human Checkpoints
```python
from ai_autonom.orchestration.human_checkpoint import (
    HumanCheckpointManager, create_critical_action_checkpoint, RiskLevel
)

# Create checkpoint
manager = HumanCheckpointManager()
checkpoint = create_critical_action_checkpoint(
    action="Execute SQL query on production database",
    risk_level=RiskLevel.CRITICAL
)

# Request approval
decision = manager.request_approval(checkpoint.id)
if decision.decision == "APPROVE":
    # Execute action
    pass
```

### Testing Workflow
```python
from ai_autonom.orchestration.testing_workflow import TestingWorkflow, TestType

# Execute testing workflow
workflow = TestingWorkflow()
code = """
def add(a, b):
    return a + b
"""

result = workflow.execute_testing_workflow(
    code=code,
    test_types=[TestType.UNIT, TestType.INTEGRATION]
)

print(result["tests_passed"])  # 5
print(result["tests_failed"])  # 0
print(result["quality_gates_passed"])  # True
```

## Dependencies

**External Dependencies:**
- `sqlite3`: Persistent storage for checkpoints, errors, tests
- `json`: Data serialization
- `subprocess`: Test execution
- `logging`: Event logging
- `re`: Regex for parsing

**Internal Dependencies:**
- `ai_autonom.patterns.cai_patterns`: Pattern execution engine
- `ai_autonom.agents.agent_registry`: Agent management
- `ai_autonom.memory.task_memory`: Task context storage
- `ai_autonom.core.llm_provider`: LLM integration
- `ai_autonom.core.model_manager`: Model management

## Important Functionality

1. **Intelligent Pattern Selection**: Analyzes task and selects optimal pattern
2. **Dynamic Agent Selection**: Capability-based agent matching
3. **Intent Understanding**: Natural language goal analysis
4. **Multi-Strategy Error Recovery**: Retry, backoff, model swap, agent swap, prompt refinement
5. **Human Checkpoints**: Risk-based approval workflow with auto-approval
6. **Test Automation**: Auto-generate and execute tests with quality gates
7. **Context Management**: Persistent task context across agent handoffs
8. **Escalation Handling**: Automatic escalation for unrecoverable errors
9. **Quality Validation**: Code quality gates for test results
10. **Audit Trail**: Complete logging of all orchestration decisions

## Checkpoint Risk Levels

| Risk Level | Auto-Approve | Example Actions |
|------------|--------------|-----------------|
| LOW | Yes | Read-only operations, info gathering |
| MEDIUM | Optional | Code generation, non-critical modifications |
| HIGH | Required | Database writes, system configuration |
| CRITICAL | Required + Confirmation | Production deployments, data deletion |

## Error Recovery Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| RETRY | Transient errors | Network timeout → retry |
| EXPONENTIAL_BACKOFF | Rate limiting | API rate limit → backoff |
| FALLBACK_MODEL | Model-specific error | Model offline → switch model |
| AGENT_SWAP | Agent failure | Agent timeout → switch agent |
| PROMPT_REFINEMENT | Output validation failure | Invalid output → refine prompt |
| SKIP | Non-critical failure | Optional step fails → skip |
| ESCALATE | Unrecoverable error | All retries exhausted → human |

## Quality Gates

Tests must meet these criteria:
- **Minimum Pass Rate**: 80% of tests must pass
- **No Critical Failures**: Zero tests with severity = CRITICAL
- **Coverage**: Minimum 70% code coverage (if measured)
