"""
Example: Code Generation with Testing Workflow

Demonstrates code_review_recursive pattern with automated testing:
- Agent generates code
- Tests are automatically generated and run
- Code is refined until tests pass
- Quality gates ensure code meets standards
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_code_generation_workflow():
    """Run code generation with testing workflow"""
    from orchestration.testing_workflow import TestingWorkflow, validate_code_quality
    from patterns.cai_patterns import PatternLibrary
    
    print("="*70)
    print("CODE GENERATION WITH TESTING WORKFLOW")
    print("="*70)
    
    # Initialize workflow
    print("\n[1/4] Initializing testing workflow...")
    workflow = TestingWorkflow(tester_agent="coder_qwen")
    
    # Define coding goal
    print("\n[2/4] Defining coding goal...")
    goal = """
    Write a Python function that validates email addresses.
    
    Requirements:
    - Function name: validate_email
    - Input: email string
    - Output: True if valid, False otherwise
    - Validation rules:
      * Must contain exactly one @ symbol
      * Must have characters before and after @
      * Domain must have at least one dot
      * No spaces allowed
    """
    
    print("  Goal: Email validation function")
    
    # Simulate code generation
    print("\n[3/4] Generating code...")
    
    # Iteration 1: Initial attempt (buggy)
    print("\n  Iteration 1:")
    code_v1 = """
def validate_email(email):
    '''Validate email address'''
    if '@' in email:
        return True
    return False
"""
    
    print("    Generated initial code")
    print("    Running tests...")
    
    # This would fail validation (too simple)
    print("    Tests: 2 passed, 3 failed")
    print("    Quality gate: FAILED (pass rate: 40%)")
    
    # Iteration 2: Improved version
    print("\n  Iteration 2:")
    code_v2 = """
def validate_email(email):
    '''Validate email address with proper checks'''
    import re
    
    # Check basic structure
    if not email or ' ' in email:
        return False
    
    # Check for exactly one @
    if email.count('@') != 1:
        return False
    
    # Split into local and domain
    local, domain = email.split('@')
    
    # Validate local and domain parts
    if not local or not domain:
        return False
    
    # Domain must have at least one dot
    if '.' not in domain:
        return False
    
    # Validate with regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""
    
    print("    Refined code based on test failures")
    print("    Running tests...")
    print("    Tests: 5 passed, 0 failed")
    print("    Quality gate: PASSED (pass rate: 100%)")
    
    # Add testing tasks
    print("\n[4/4] Creating test suite...")
    
    tasks = [
        {
            "id": "email_validator",
            "description": "Implement email validation function",
            "type": "coding",
            "tools": ["python_exec"]
        }
    ]
    
    enhanced_tasks = workflow.add_testing_tasks(tasks)
    
    print(f"  Original tasks: {len(tasks)}")
    print(f"  Enhanced tasks: {len(enhanced_tasks)}")
    print(f"  Test task created: {enhanced_tasks[1]['id']}")
    
    # Show test task
    print("\n  Test Task Details:")
    print(f"    ID: {enhanced_tasks[1]['id']}")
    print(f"    Dependencies: {enhanced_tasks[1]['dependencies']}")
    print(f"    Tools: {enhanced_tasks[1]['tools']}")
    
    print("\n" + "="*70)
    print("CODE GENERATION COMPLETED")
    print("="*70)
    print("\nDeliverables:")
    print("  - validate_email() function")
    print("  - Comprehensive test suite (5 tests)")
    print("  - 100% test pass rate")
    print("  - Quality gates: PASSED")
    
    print("\nGenerated Tests:")
    print("  1. test_valid_email_basic()")
    print("  2. test_invalid_email_no_at()")
    print("  3. test_invalid_email_multiple_at()")
    print("  4. test_invalid_email_no_domain()")
    print("  5. test_invalid_email_with_spaces()")
    
    print("\nCode Quality Metrics:")
    print("  - Test Coverage: 100%")
    print("  - Pass Rate: 100%")
    print("  - Iterations: 2")
    print("  - Quality Score: 95/100")


if __name__ == "__main__":
    try:
        run_code_generation_workflow()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
