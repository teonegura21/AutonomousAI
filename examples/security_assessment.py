"""
Example: Security Assessment Workflow

Demonstrates full CAI-style security assessment using:
- web_pentester agent for scanning
- retester agent for validation
- report_agent for documentation
- Handoffs between agents
- Human checkpoints for critical actions
- Pattern execution (security_pipeline)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_security_assessment():
    """Run complete security assessment workflow"""
    from orchestration.nemotron_orchestrator import NemotronOrchestrator
    from patterns.cai_patterns import PatternLibrary
    from orchestration.human_checkpoint import HumanCheckpointManager
    
    print("="*70)
    print("SECURITY ASSESSMENT WORKFLOW EXAMPLE")
    print("="*70)
    
    # Initialize components
    print("\n[1/5] Initializing orchestrator...")
    orchestrator = NemotronOrchestrator(
        enable_checkpoints=True,
        enable_testing=False
    )
    
    # Set up pattern
    print("[2/5] Loading security_pipeline pattern...")
    library = PatternLibrary()
    pattern = library.get_pattern("security_pipeline")
    
    print(f"  Pattern: {pattern.name}")
    print(f"  Agents: {' -> '.join(pattern.agents)}")
    
    # Define security assessment goal
    print("\n[3/5] Defining assessment goal...")
    goal = """
    Perform security assessment on local web application:
    
    Target: http://localhost:8000
    Scope: 
      - Authentication mechanisms
      - Input validation
      - Session management
      - Common web vulnerabilities (XSS, SQLi, CSRF)
    
    Constraints:
      - Read-only scanning (no exploitation)
      - No data modification
      - Respect rate limits
    
    Deliverables:
      - Vulnerability report
      - Risk assessment
      - Remediation recommendations
    """
    
    print(f"  Goal: Security assessment of localhost:8000")
    
    # Analyze intent
    print("\n[4/5] Analyzing intent...")
    from orchestration.intent_analyzer import IntentAnalyzer
    
    analyzer = IntentAnalyzer()
    analysis = analyzer.analyze(goal, {})
    
    print(f"  Task Type: {analysis.task_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Suggested Pattern: {analysis.suggested_pattern}")
    print(f"  Suggested Agents: {', '.join(analysis.suggested_agents)}")
    
    # Execute workflow (dry run - agents would actually execute)
    print("\n[5/5] Executing workflow (DRY RUN)...")
    print("\n  Step 1: web_pentester scans target")
    print("    - Reconnaissance: Enumerate endpoints, technologies")
    print("    - Vulnerability scanning: Test for common issues")
    print("    - Authentication testing: Check login mechanisms")
    print("    [OK] Findings: 5 vulnerabilities detected")
    
    print("\n  Step 2: Handoff to retester")
    print("    - Validate each finding")
    print("    - Eliminate false positives")
    print("    - Confirm exploitability")
    print("    [OK] Validated: 3 real vulnerabilities")
    
    print("\n  Step 3: Human checkpoint (CRITICAL)")
    print("    - Review validated vulnerabilities")
    print("    - Approve report generation")
    print("    Decision: [A]pprove / [R]eject / [M]odify")
    print("    [OK] Approved")
    
    print("\n  Step 4: Handoff to report_agent")
    print("    - Generate professional report")
    print("    - Include: Executive summary, findings, remediation")
    print("    - Format: Markdown + PDF")
    print("    [OK] Report generated: security_report_20260117.md")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED")
    print("="*70)
    print("\nDeliverables:")
    print("  - security_report_20260117.md")
    print("  - 3 validated vulnerabilities")
    print("  - Remediation recommendations")
    print("\nNext steps:")
    print("  1. Review report with development team")
    print("  2. Prioritize fixes by severity")
    print("  3. Schedule remediation")
    print("  4. Re-test after fixes")


if __name__ == "__main__":
    try:
        run_security_assessment()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
