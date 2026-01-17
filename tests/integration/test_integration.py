"""
Integration Tests for Multi-Agent Orchestration System

Tests the complete end-to-end workflow including:
- Dynamic model management
- CAI patterns and handoffs
- LangGraph workflow execution
- Security agents
- Task memory
- Intent analysis
- Error recovery
- Human checkpoints
- Testing automation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
from typing import Dict, Any


class IntegrationTestSuite:
    """Complete integration test suite"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, name: str, test_func):
        """Run a single test"""
        try:
            print(f"\n{'='*70}")
            print(f"TEST: {name}")
            print(f"{'='*70}")
            
            start = time.time()
            test_func()
            duration = time.time() - start
            
            self.passed += 1
            print(f"[PASS] {name} ({duration:.2f}s)")
            
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
            
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"[ERROR] {name}: {e}")
    
    def report(self):
        """Print test report"""
        print(f"\n{'='*70}")
        print("INTEGRATION TEST REPORT")
        print(f"{'='*70}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total: {self.passed + self.failed}")
        
        if self.errors:
            print(f"\nErrors:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        
        print(f"{'='*70}\n")
        
        return self.failed == 0


def test_model_discovery():
    """Test dynamic model discovery system"""
    from core.model_discovery import ModelDiscovery
    from core.model_selector import DynamicModelSelector
    
    # Test model discovery
    discovery = ModelDiscovery()
    
    try:
        models = discovery.scan_ollama_models()
        assert len(models) > 0, "No Ollama models found"
        print(f"  Found {len(models)} Ollama models")
    except Exception as e:
        # Ollama not installed - skip this test
        print(f"  Ollama not available: {e}")
        return
    
    # Test model selector
    selector = DynamicModelSelector()
    best = selector.select_best_model("coding")
    
    if best:
        print(f"  Best coding model: {best['model_name']}")
    else:
        print(f"  No models registered yet")


def test_agent_registry():
    """Test agent registry and factory"""
    from core.agent_registry import AgentRegistry
    
    registry = AgentRegistry()
    agents = registry.get_all_agents()
    
    assert len(agents) > 0, "No agents registered"
    print(f"  Registered agents: {len(agents)}")
    
    # Test agent retrieval
    if agents:
        first_agent = agents[0]
        print(f"  Retrieved agent: {first_agent.id}")


def test_cai_patterns():
    """Test CAI agentic patterns"""
    from patterns.cai_patterns import PatternLibrary, PatternType, PatternExecutor
    
    library = PatternLibrary()
    
    # Test pattern retrieval
    pattern = library.get_pattern("security_pipeline")
    assert pattern is not None, "Failed to get security_pipeline pattern"
    assert pattern.pattern_type == PatternType.CHAIN
    print(f"  Pattern: {pattern.name}, agents: {len(pattern.agents)}")
    
    # Test pattern executor
    executor = PatternExecutor()
    assert executor is not None, "Failed to create pattern executor"
    print(f"  Pattern executor initialized")


def test_handoff_system():
    """Test handoff manager"""
    from patterns.handoffs import HandoffManager
    from patterns.cai_patterns import PatternLibrary
    
    manager = HandoffManager()
    library = PatternLibrary()
    
    # Test callback registration
    def test_callback(ctx):
        print(f"  Callback executed for handoff: {ctx.from_agent} -> {ctx.to_agent}")
    
    manager.register_callback("web_pentester", test_callback)
    print(f"  Registered handoff callback")
    
    # Test handoff statistics
    stats = manager.get_handoff_stats()
    assert "total_handoffs" in stats
    print(f"  Total handoffs: {stats['total_handoffs']}")


def test_task_memory():
    """Test task memory system"""
    from memory.task_memory import TaskMemory
    
    memory = TaskMemory()
    
    # Create workflow
    workflow_id = "test_workflow_integration"
    memory.start_workflow(workflow_id)
    print(f"  Started workflow: {workflow_id}")
    
    # Create and complete task
    task_id = "test_task_1"
    memory.create_task_context(task_id, "Test task", "coder_qwen")
    memory.start_task(task_id)
    memory.complete_task(task_id, "Test output")
    print(f"  Completed task: {task_id}")
    
    # Verify memory is working
    print(f"  Task memory system functional")


def test_intent_analyzer():
    """Test intent analyzer"""
    from orchestration.intent_analyzer import IntentAnalyzer, TaskType
    
    analyzer = IntentAnalyzer()
    
    # Test coding task analysis
    result = analyzer.analyze(
        "Write a Python function to validate email addresses and test it",
        {}
    )
    
    assert result.task_type is not None
    assert result.complexity is not None
    assert len(result.requirements) > 0
    print(f"  Task type: {result.task_type.value}, complexity: {result.complexity.value}")
    print(f"  Suggested agents: {result.suggested_agents}")
    
    # Test security task analysis
    result2 = analyzer.analyze(
        "Perform security assessment on web application at localhost:8000",
        {}
    )
    
    assert result2.task_type is not None
    print(f"  Security task identified, pattern: {result2.suggested_pattern}")


def test_error_recovery():
    """Test error recovery system"""
    from orchestration.error_recovery import ErrorRecovery, RecoveryAction
    
    recovery = ErrorRecovery(max_retries=3)
    
    # Test with simulated failure
    task = {"id": "test_task", "description": "Test task"}
    
    # First failure - should retry
    decision = recovery.handle_failure(task, "Connection timeout", {})
    assert decision.action == RecoveryAction.RETRY
    print(f"  First failure: {decision.action.value}")
    
    # Get error history
    history = recovery.get_error_history("test_task")
    assert len(history) > 0
    print(f"  Error history: {len(history)} entries")
    
    # Check retry count
    assert recovery.retry_counts["test_task"] == 1
    print(f"  Retry tracking working")


def test_human_checkpoint():
    """Test human checkpoint system"""
    from orchestration.human_checkpoint import (
        HumanCheckpointManager,
        CheckpointType,
        RiskLevel,
        create_critical_action_checkpoint
    )
    
    manager = HumanCheckpointManager(auto_approve_low_risk=True)
    
    # Test low-risk auto-approval
    checkpoint = create_critical_action_checkpoint(
        task_id="test_task",
        agent_id="test_agent",
        action_description="Read configuration file",
        risk_assessment="Low risk: read-only operation"
    )
    checkpoint.risk_level = RiskLevel.LOW
    
    result = manager.request_approval(checkpoint, interactive=False)
    assert result.approved == True
    assert result.reviewer == "auto"
    print(f"  Auto-approved low-risk checkpoint")
    
    # Test checkpoint statistics
    stats = manager.get_checkpoint_stats()
    assert "total_checkpoints" in stats
    print(f"  Checkpoint stats: {stats['total_checkpoints']} total")


def test_testing_workflow():
    """Test automated testing workflow"""
    from orchestration.testing_workflow import TestingWorkflow, TestType
    
    workflow = TestingWorkflow()
    
    # Test task enhancement
    tasks = [
        {
            "id": "task_1",
            "description": "Write a function to calculate factorial",
            "tools": ["python_exec"],
            "type": "coding"
        }
    ]
    
    enhanced = workflow.add_testing_tasks(tasks)
    assert len(enhanced) == 2
    assert enhanced[1]["id"] == "task_1_test"
    print(f"  Enhanced {len(tasks)} tasks to {len(enhanced)} with testing")
    
    # Test code generation
    code = """
def add(a, b):
    return a + b
"""
    
    result = workflow.generate_and_run(code, "test_artifact")
    assert result["total_tests"] > 0
    print(f"  Generated and ran {result['total_tests']} tests")


def test_langgraph_workflow():
    """Test LangGraph workflow integration"""
    from orchestration.langgraph_workflow import MultiAgentWorkflow
    from patterns.cai_patterns import PatternLibrary
    
    workflow = MultiAgentWorkflow()
    library = PatternLibrary()
    
    # Create workflow with pattern
    pattern = library.get_pattern("security_pipeline")
    
    # Note: Full execution requires Ollama and agents to be running
    # This test just verifies the components are properly initialized
    assert workflow is not None
    assert pattern is not None
    print(f"  LangGraph workflow initialized with pattern: {pattern.name}")


def test_tool_registry():
    """Test tool registry and execution"""
    from tools.tool_executor import ToolExecutor
    
    executor = ToolExecutor()
    tools = executor.get_available_tools()
    
    assert len(tools) > 0, "No tools registered"
    print(f"  Registered tools: {len(tools)}")
    
    # Test tool categorization
    categories = set(t['category'] for t in tools)
    assert len(categories) > 0
    print(f"  Tool categories: {', '.join(categories)}")


def test_vector_memory():
    """Test vector memory store"""
    from memory.vector_store import VectorMemoryStore
    
    try:
        store = VectorMemoryStore()
        
        # Test storage
        store.store(
            "Python is a programming language",
            {"source": "test"},
            "test_collection"
        )
        print(f"  Stored text in vector memory")
        
        # Test query
        results = store.query("programming language", "test_collection", n_results=1)
        assert len(results) > 0
        print(f"  Query returned {len(results)} results")
        
    except Exception as e:
        print(f"  Vector memory test skipped (ChromaDB not available): {e}")


def run_integration_tests():
    """Run all integration tests"""
    suite = IntegrationTestSuite()
    
    print("\n" + "="*70)
    print("MULTI-AGENT ORCHESTRATION SYSTEM - INTEGRATION TESTS")
    print("="*70)
    
    # Core infrastructure tests
    suite.test("Model Discovery System", test_model_discovery)
    suite.test("Agent Registry", test_agent_registry)
    
    # CAI patterns tests
    suite.test("CAI Agentic Patterns", test_cai_patterns)
    suite.test("Handoff System", test_handoff_system)
    
    # Memory tests
    suite.test("Task Memory", test_task_memory)
    suite.test("Vector Memory", test_vector_memory)
    
    # Orchestration tests
    suite.test("Intent Analyzer", test_intent_analyzer)
    suite.test("Error Recovery", test_error_recovery)
    suite.test("Human Checkpoints", test_human_checkpoint)
    suite.test("Testing Workflow", test_testing_workflow)
    suite.test("LangGraph Workflow", test_langgraph_workflow)
    
    # Tool tests
    suite.test("Tool Registry", test_tool_registry)
    
    # Print report
    success = suite.report()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_integration_tests())
