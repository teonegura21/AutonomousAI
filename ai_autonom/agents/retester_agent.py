#!/usr/bin/env python3
"""
Retester Agent
Validates security findings from other agents
Eliminates false positives and confirms exploitability

Based on CAI retester agent design
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetesterAgent:
    """
    Security Finding Validation Agent
    
    Capabilities:
    - Revalidate reported vulnerabilities
    - Eliminate false positives
    - Confirm exploitability
    - Assess actual impact
    - Provide detailed reproduction steps
    """
    
    id: str = "retester"
    name: str = "Security Retester"
    model_name: str = "qwen3:1.7b"
    provider: str = "ollama"
    
    capabilities: List[str] = None
    tools: List[str] = None
    handoffs: List[str] = None
    
    description: str = "Validates security findings and eliminates false positives"
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [
                "vulnerability_validation",
                "false_positive_detection",
                "exploit_confirmation",
                "impact_assessment",
                "reproduction_documentation"
            ]
        
        if self.tools is None:
            self.tools = [
                "bash_exec",
                "python_exec",
                "web_fetch",
                "filesystem_read",
                "filesystem_write"
            ]
        
        if self.handoffs is None:
            self.handoffs = [
                "report_agent",
                "web_pentester"
            ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """You are a security finding validation specialist.

Your role is to verify and validate security vulnerabilities reported by other agents, ensuring accuracy and eliminating false positives before they reach the final report.

RESPONSIBILITIES:

1. Finding Validation:
   - Review each reported vulnerability carefully
   - Attempt to reproduce the vulnerability independently
   - Verify the exploit works as described
   - Confirm the vulnerability is genuine, not a false positive

2. False Positive Detection:
   - Identify scanner errors and misconfigurations
   - Distinguish between theoretical and practical vulnerabilities
   - Check for security controls that mitigate the issue
   - Verify the vulnerability exists in the actual application context

3. Impact Assessment:
   - Determine the real-world exploitability
   - Assess actual impact vs. theoretical impact
   - Consider defense-in-depth controls
   - Evaluate severity in context of the application

4. Documentation:
   - Create detailed reproduction steps
   - Document evidence of successful exploitation
   - Note any conditions required for exploitation
   - Provide clear pass/fail validation result

VALIDATION METHODOLOGY:

For each finding:
1. Review the original report
2. Set up the same test environment
3. Follow the reproduction steps exactly
4. Attempt exploitation with variations
5. Document success or failure with evidence
6. If failed: analyze why (false positive, patched, requires conditions)
7. If successful: confirm severity and impact

VALIDATION OUTCOMES:
- CONFIRMED: Vulnerability is real and exploitable (keep in report)
- FALSE POSITIVE: Finding is not a real vulnerability (remove from report)
- NEEDS CLARIFICATION: Requires specific conditions or more testing (note in report)
- PARTIALLY CONFIRMED: Vulnerability exists but with lower impact (adjust severity)

TOOLS AVAILABLE:
- bash_exec: Execute validation commands
- python_exec: Run custom validation scripts
- web_fetch: Test web vulnerabilities
- filesystem operations: Read findings, write validation results

OUTPUT FORMAT:
For each finding:
```
FINDING: [Vulnerability Title]
ORIGINAL SEVERITY: [Critical/High/Medium/Low]
VALIDATION STATUS: [Confirmed/False Positive/Needs Clarification/Partially Confirmed]
REPRODUCTION STEPS:
1. [Step 1]
2. [Step 2]
...
EVIDENCE: [Screenshots, command output, logs]
ACTUAL IMPACT: [Description]
REVISED SEVERITY: [If different from original]
NOTES: [Additional observations]
```

When you complete validation, handoff to:
- report_agent: For final report generation with validated findings
- web_pentester: If additional testing is needed

Be rigorous, objective, and thorough. False positives damage credibility."""

    def get_instructions(self, context: Dict[str, Any]) -> str:
        """Get dynamic instructions based on context"""
        findings = context.get("findings", [])
        validation_mode = context.get("validation_mode", "standard")
        
        instructions = f"""SECURITY FINDING VALIDATION TASK

Number of findings to validate: {len(findings)}
Validation mode: {validation_mode}

"""
        
        if validation_mode == "quick":
            instructions += """QUICK VALIDATION:
- Focus on critical and high severity findings
- Automated validation where possible
- Prioritize exploitability confirmation
- Time limit: 10 minutes per finding
"""
        elif validation_mode == "standard":
            instructions += """STANDARD VALIDATION:
- Validate all reported findings
- Manual reproduction for each
- Detailed documentation
- Conservative approach to false positives
"""
        elif validation_mode == "thorough":
            instructions += """THOROUGH VALIDATION:
- Deep dive into each finding
- Multiple exploitation attempts
- Test variations and edge cases
- Document all observations
- Challenge every assumption
"""
        
        if findings:
            instructions += "\nFINDINGS TO VALIDATE:\n"
            for i, finding in enumerate(findings[:5], 1):  # Show first 5
                instructions += f"{i}. {finding.get('title', 'Unknown')} - {finding.get('severity', 'Unknown')}\n"
            
            if len(findings) > 5:
                instructions += f"... and {len(findings) - 5} more\n"
        
        instructions += """
VALIDATION PROCESS:
1. Review each finding from previous assessment
2. Set up test environment if needed
3. Follow reproduction steps exactly
4. Document validation results with evidence
5. Provide final verdict on each finding

DELIVERABLES:
1. Validation report for each finding
2. List of confirmed vulnerabilities
3. List of false positives (with explanation)
4. Updated severity assessments
5. Detailed reproduction steps for confirmed findings

Begin validation now."""
        
        return instructions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "model_name": self.model_name,
            "provider": self.provider,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "handoffs": self.handoffs,
            "description": self.description,
            "system_prompt": self.get_system_prompt()
        }


# Create default instance
retester_agent = RetesterAgent()


if __name__ == "__main__":
    agent = RetesterAgent()
    
    print("="*60)
    print("RETESTER AGENT")
    print("="*60)
    print(f"ID: {agent.id}")
    print(f"Name: {agent.name}")
    print(f"Model: {agent.model_name}")
    print(f"Provider: {agent.provider}")
    print(f"Capabilities: {', '.join(agent.capabilities)}")
    print(f"Tools: {', '.join(agent.tools)}")
    print(f"Handoffs: {', '.join(agent.handoffs)}")
    print(f"\nSystem Prompt Length: {len(agent.get_system_prompt())} chars")
    
    # Test instructions
    context = {
        "findings": [
            {"title": "SQL Injection in login form", "severity": "Critical"},
            {"title": "XSS in comment field", "severity": "High"},
            {"title": "Insecure cookie flags", "severity": "Medium"}
        ],
        "validation_mode": "standard"
    }
    instructions = agent.get_instructions(context)
    print(f"Instructions Length: {len(instructions)} chars")
    print("\n" + "="*60)
