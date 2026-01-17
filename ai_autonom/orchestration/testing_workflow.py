"""
Testing Workflow Automation

Automates test generation, execution, validation, and quality gates
for code artifacts produced by agents.
"""

import logging
import subprocess
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests to generate"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    SECURITY = "security"
    PERFORMANCE = "performance"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case"""
    test_id: str
    test_type: TestType
    name: str
    description: str
    code: str
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of test execution"""
    test_id: str
    status: TestStatus
    duration_sec: float
    output: str
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    success: bool = False
    task_id: str = "unknown"
    coverage: Optional[float] = None


@dataclass
class TestSuite:
    """Collection of tests for a code artifact"""
    suite_id: str
    artifact_id: str
    tests: List[TestCase]
    coverage_target: float = 0.8
    quality_gates: Dict[str, Any] = field(default_factory=dict)


class TestGenerator:
    """
    Generates tests for code artifacts using LLM.
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize test generator.
        
        Args:
            llm_provider: LLM provider for test generation (optional)
        """
        self.llm_provider = llm_provider
    
    def generate_unit_tests(
        self,
        code: str,
        language: str = "python",
        coverage_target: float = 0.8
    ) -> List[TestCase]:
        """
        Generate unit tests for code.
        
        Args:
            code: Source code to test
            language: Programming language
            coverage_target: Target code coverage (0-1)
        
        Returns:
            List of generated test cases
        """
        import uuid
        
        # Simple heuristic test generation for demo
        # In production, use LLM to generate comprehensive tests
        tests = []
        
        # Extract functions from code
        functions = self._extract_functions(code, language)
        
        for func_name in functions:
            test_id = f"test_{uuid.uuid4().hex[:8]}"
            
            if language == "python":
                test_code = self._generate_python_unit_test(func_name, code)
            else:
                test_code = f"# Test for {func_name}\n# TODO: Implement test"
            
            tests.append(TestCase(
                test_id=test_id,
                test_type=TestType.UNIT,
                name=f"test_{func_name}",
                description=f"Unit test for {func_name}",
                code=test_code
            ))
        
        return tests
    
    def _extract_functions(self, code: str, language: str) -> List[str]:
        """Extract function names from code"""
        functions = []
        
        if language == "python":
            import re
            pattern = r"def\s+(\w+)\s*\("
            functions = re.findall(pattern, code)
        
        return functions
    
    def _generate_python_unit_test(self, func_name: str, source_code: str) -> str:
        """Generate Python unit test code"""
        return f"""import pytest

def test_{func_name}_basic():
    \"\"\"Test {func_name} with basic inputs\"\"\"
    # Arrange
    # TODO: Set up test data
    
    # Act
    result = {func_name}()
    
    # Assert
    assert result is not None


def test_{func_name}_edge_cases():
    \"\"\"Test {func_name} with edge cases\"\"\"
    # TODO: Test edge cases
    pass
"""
    
    def generate_integration_tests(
        self,
        code: str,
        dependencies: List[str],
        language: str = "python"
    ) -> List[TestCase]:
        """Generate integration tests for code with dependencies"""
        import uuid
        
        tests = []
        
        test_id = f"test_{uuid.uuid4().hex[:8]}"
        test_code = f"""import pytest

def test_integration():
    \"\"\"Integration test for module\"\"\"
    # TODO: Test integration with: {', '.join(dependencies)}
    pass
"""
        
        tests.append(TestCase(
            test_id=test_id,
            test_type=TestType.INTEGRATION,
            name="test_integration",
            description="Integration test",
            code=test_code,
            dependencies=dependencies
        ))
        
        return tests


class TestRunner:
    """
    Executes tests and collects results.
    """
    
    def __init__(self, docker_executor=None):
        """
        Initialize test runner.
        
        Args:
            docker_executor: Docker executor for sandboxed test execution
        """
        self.docker_executor = docker_executor
    
    def run_test(self, test: TestCase, timeout: int = 30) -> TestResult:
        """
        Run a single test.
        
        Args:
            test: Test case to run
            timeout: Timeout in seconds
        
        Returns:
            Test result
        """
        import time
        start_time = time.time()
        
        try:
            # Write test to temporary file
            test_file = Path(f"temp_tests/test_{test.test_id}.py")
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(test.code)
            
            # Run test with pytest
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResult(
                    test_id=test.test_id,
                    status=TestStatus.PASSED,
                    duration_sec=duration,
                    output=result.stdout
                )
            else:
                return TestResult(
                    test_id=test.test_id,
                    status=TestStatus.FAILED,
                    duration_sec=duration,
                    output=result.stdout,
                    error_message=result.stderr
                )
        
        except subprocess.TimeoutExpired:
            return TestResult(
                test_id=test.test_id,
                status=TestStatus.ERROR,
                duration_sec=timeout,
                output="",
                error_message="Test timeout"
            )
        
        except Exception as e:
            return TestResult(
                test_id=test.test_id,
                status=TestStatus.ERROR,
                duration_sec=time.time() - start_time,
                output="",
                error_message=str(e)
            )
    
    def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """
        Run all tests in a suite.
        
        Args:
            suite: Test suite to run
        
        Returns:
            List of test results
        """
        results = []
        
        for test in suite.tests:
            logger.info(f"Running test: {test.name}")
            result = self.run_test(test)
            results.append(result)
        
        return results


class TestingWorkflow:
    """
    Complete testing workflow with auto-generation and validation.
    Implements code_review_recursive pattern integration.
    """
    
    def __init__(
        self,
        generator: Optional[TestGenerator] = None,
        runner: Optional[TestRunner] = None,
        db_path: str = ".runtime/data/testing.db",
        tester_agent: str = "coder_qwen"
    ):
        """
        Initialize testing workflow.
        
        Args:
            generator: Test generator instance
            runner: Test runner instance
            db_path: Path to testing database
            tester_agent: Agent ID for testing tasks
        """
        self.generator = generator or TestGenerator()
        self.runner = runner or TestRunner()
        self.tester_agent = tester_agent
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self.test_results: List[TestResult] = []
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize testing database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_suites (
                suite_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                coverage_target REAL,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                test_id TEXT PRIMARY KEY,
                suite_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                code TEXT NOT NULL,
                FOREIGN KEY (suite_id) REFERENCES test_suites(suite_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_sec REAL,
                output TEXT,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (test_id) REFERENCES test_cases(test_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_and_run(
        self,
        code: str,
        artifact_id: str,
        language: str = "python",
        test_types: List[TestType] = None
    ) -> Dict[str, Any]:
        """
        Generate tests for code and run them.
        
        Args:
            code: Source code to test
            artifact_id: Identifier for code artifact
            language: Programming language
            test_types: Types of tests to generate
        
        Returns:
            Dictionary with test results and statistics
        """
        import uuid
        
        if test_types is None:
            test_types = [TestType.UNIT]
        
        # Generate tests
        all_tests = []
        
        for test_type in test_types:
            if test_type == TestType.UNIT:
                tests = self.generator.generate_unit_tests(code, language)
                all_tests.extend(tests)
            elif test_type == TestType.INTEGRATION:
                tests = self.generator.generate_integration_tests(code, [], language)
                all_tests.extend(tests)
        
        # Create test suite
        suite_id = f"suite_{uuid.uuid4().hex[:8]}"
        suite = TestSuite(
            suite_id=suite_id,
            artifact_id=artifact_id,
            tests=all_tests
        )
        
        # Save suite to database
        self._save_suite(suite)
        
        # Run tests
        results = self.runner.run_suite(suite)
        
        # Save results to database
        self._save_results(results)
        
        # Calculate statistics
        stats = self._calculate_stats(results)
        
        return {
            "suite_id": suite_id,
            "total_tests": len(results),
            "results": results,
            "statistics": stats,
            "passed": stats["passed"] == len(results)
        }
    
    def add_testing_tasks(
        self,
        tasks: List[Dict[str, Any]],
        enable_testing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        For each coding task, add corresponding test task.
        
        Args:
            tasks: Original task list
            enable_testing: Whether to add test tasks
        
        Returns:
            Enhanced task list with test tasks
        """
        if not enable_testing:
            return tasks
        
        enhanced_tasks = []
        
        for task in tasks:
            enhanced_tasks.append(task)
            
            # Add test for coding tasks (but not synthesis or test tasks)
            if self._is_testable_task(task):
                test_task = self._create_test_task(task)
                enhanced_tasks.append(test_task)
        
        return enhanced_tasks
    
    def _is_testable_task(self, task: Dict[str, Any]) -> bool:
        """Check if task should have tests"""
        # Don't test synthesis or existing test tasks
        if task.get("type") in ("synthesis", "testing", "documentation"):
            return False
        
        # Don't test if already has a test dependency
        task_id = task.get("id", "")
        if "_test" in task_id:
            return False
        
        # Check for coding indicators
        coding_indicators = [
            "python_exec" in task.get("tools", []),
            "code" in task.get("description", "").lower(),
            "write" in task.get("description", "").lower(),
            "implement" in task.get("description", "").lower(),
            "create" in task.get("description", "").lower(),
            "function" in task.get("description", "").lower(),
            "class" in task.get("description", "").lower(),
            task.get("required_capability") == "code_generation"
        ]
        
        return any(coding_indicators)
    
    def _create_test_task(self, coding_task: Dict[str, Any]) -> Dict[str, Any]:
        """Create test task for a coding task"""
        coding_task_id = coding_task.get("id", "unknown")
        
        test_description = f"""TEST TASK for {coding_task_id}:

Original task: {coding_task.get('description', 'No description')[:200]}

Your job:
1. Read the code output from {coding_task_id}
2. Generate pytest tests that verify:
   - All functions exist and are callable
   - Basic functionality works correctly
   - Edge cases are handled (empty input, None, negative numbers, etc.)
   - Error handling works as expected
3. Run the tests using pytest
4. Report pass/fail status

Output format:
```python
# test_{coding_task_id}.py
import pytest

def test_basic_functionality():
    # Test basic case
    pass

def test_edge_cases():
    # Test edge cases
    pass

def test_error_handling():
    # Test error scenarios
    pass
```

After writing tests, execute them and report results."""
        
        return {
            "id": f"{coding_task_id}_test",
            "description": test_description,
            "assigned_agent": self.tester_agent,
            "tools": ["filesystem_read", "filesystem_write", "pytest_run", "python_exec"],
            "dependencies": [coding_task_id],
            "type": "testing",
            "required_capability": "testing"
        }
    
    def _save_suite(self, suite: TestSuite) -> None:
        """Save test suite to database"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO test_suites (suite_id, artifact_id, coverage_target, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                suite.suite_id,
                suite.artifact_id,
                suite.coverage_target,
                datetime.now().isoformat()
            ))
            
            for test in suite.tests:
                cursor.execute("""
                    INSERT INTO test_cases (test_id, suite_id, test_type, name, description, code)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    test.test_id,
                    suite.suite_id,
                    test.test_type.value,
                    test.name,
                    test.description,
                    test.code
                ))
            
            conn.commit()
            conn.close()
    
    def _save_results(self, results: List[TestResult]) -> None:
        """Save test results to database"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute("""
                    INSERT INTO test_results (test_id, status, duration_sec, output, error_message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.test_id,
                    result.status.value,
                    result.duration_sec,
                    result.output,
                    result.error_message,
                    result.timestamp
                ))
            
            conn.commit()
            conn.close()
    
    def _calculate_stats(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate test statistics"""
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        avg_duration = sum(r.duration_sec for r in results) / total if total > 0 else 0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_duration_sec": avg_duration
        }
    
    def get_suite_results(self, suite_id: str) -> Dict[str, Any]:
        """Get results for a test suite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tr.test_id, tr.status, tr.duration_sec, tr.output, tr.error_message, tc.name
            FROM test_results tr
            JOIN test_cases tc ON tr.test_id = tc.test_id
            WHERE tc.suite_id = ?
            ORDER BY tr.timestamp DESC
        """, (suite_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "test_id": row[0],
                "status": row[1],
                "duration_sec": row[2],
                "output": row[3],
                "error_message": row[4],
                "name": row[5]
            })
        
        return {
            "suite_id": suite_id,
            "results": results,
            "statistics": self._calculate_stats([
                TestResult(
                    test_id=r["test_id"],
                    status=TestStatus(r["status"]),
                    duration_sec=r["duration_sec"],
                    output=r["output"],
                    error_message=r["error_message"]
                )
                for r in results
            ])
        }
    
    def record_result(self, task_id: str, output: str) -> TestResult:
        """Record test result"""
        import re
        import uuid
        
        # Parse test results from output
        pattern = r'(\d+)\s+passed(?:,\s*(\d+)\s+failed)?'
        match = re.search(pattern, output.lower())
        
        passed = 0
        failed = 0
        
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2)) if match.group(2) else 0
        else:
            if "passed" in output.lower():
                passed = 1
            if "failed" in output.lower() or "error" in output.lower():
                failed = 1
        
        total = passed + failed
        success = failed == 0 and passed > 0
        
        result = TestResult(
            test_id=f"test_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            tests_passed=passed,
            tests_failed=failed,
            tests_total=total,
            output=output[:1000],
            success=success,
            status=TestStatus.PASSED if success else TestStatus.FAILED,
            duration_sec=0.0
        )
        
        self.test_results.append(result)
        return result


# Helper function for code_review_recursive pattern integration

def validate_code_quality(
    code: str,
    language: str = "python",
    quality_gates: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate code quality with automated tests.
    
    Used in code_review_recursive pattern to determine if code meets
    quality standards before completing the iteration.
    
    Args:
        code: Source code to validate
        language: Programming language
        quality_gates: Quality thresholds (test_pass_rate, coverage, etc.)
    
    Returns:
        Validation result with pass/fail status
    """
    if quality_gates is None:
        quality_gates = {
            "test_pass_rate": 1.0,  # 100% tests must pass
            "min_tests": 1  # At least 1 test required
        }
    
    workflow = TestingWorkflow()
    result = workflow.generate_and_run(code, "validation", language)
    
    stats = result["statistics"]
    
    # Check quality gates
    meets_pass_rate = stats["pass_rate"] >= quality_gates["test_pass_rate"]
    meets_min_tests = stats["total"] >= quality_gates["min_tests"]
    
    passed = meets_pass_rate and meets_min_tests
    
    return {
        "passed": passed,
        "test_results": result,
        "quality_gates": {
            "pass_rate": {
                "required": quality_gates["test_pass_rate"],
                "actual": stats["pass_rate"],
                "met": meets_pass_rate
            },
            "min_tests": {
                "required": quality_gates["min_tests"],
                "actual": stats["total"],
                "met": meets_min_tests
            }
        }
    }
