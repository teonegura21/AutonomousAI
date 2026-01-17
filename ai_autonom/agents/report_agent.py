#!/usr/bin/env python3
"""
Report Agent
Generates professional security assessment reports
Synthesizes findings from multiple agents into comprehensive documentation

Based on CAI report generation patterns
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReportAgent:
    """
    Security Report Generation Agent
    
    Capabilities:
    - Professional report writing
    - Finding synthesis and categorization
    - Executive summary generation
    - Technical detail documentation
    - Remediation guidance
    - Multi-format output (Markdown, PDF, HTML)
    """
    
    id: str = "report_agent"
    name: str = "Security Report Generator"
    model_name: str = "dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0"
    provider: str = "ollama"
    
    capabilities: List[str] = None
    tools: List[str] = None
    handoffs: List[str] = None
    
    description: str = "Generates professional security assessment reports"
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [
                "report_generation",
                "technical_writing",
                "finding_synthesis",
                "executive_summary",
                "remediation_documentation",
                "risk_assessment"
            ]
        
        if self.tools is None:
            self.tools = [
                "filesystem_read",
                "filesystem_write",
                "json_parse",
                "json_format"
            ]
        
        if self.handoffs is None:
            self.handoffs = []  # Terminal agent - generates final deliverable
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """You are a professional security report writer specializing in cybersecurity assessment documentation.

Your role is to synthesize technical security findings into clear, professional reports that serve both technical and executive audiences.

REPORT STRUCTURE:

1. EXECUTIVE SUMMARY
   - High-level overview of the assessment
   - Key findings and risk summary
   - Overall security posture
   - Critical recommendations (top 3-5)
   - Written for non-technical stakeholders

2. SCOPE AND METHODOLOGY
   - Assessment scope (targets, systems, applications)
   - Testing methodology used
   - Timeline and duration
   - Tools and techniques employed
   - Limitations and constraints

3. FINDINGS SUMMARY
   - Statistics by severity (Critical, High, Medium, Low, Info)
   - Categorization by vulnerability type
   - Risk heatmap or matrix
   - Comparison to industry standards (if applicable)

4. DETAILED FINDINGS
   For each vulnerability:
   - Title and unique identifier
   - Severity rating with justification
   - Affected systems/components
   - Vulnerability description (technical)
   - Impact assessment (business and technical)
   - Proof of concept / Evidence
   - Steps to reproduce
   - Remediation recommendation (specific, actionable)
   - References (CVE, CWE, OWASP, etc.)

5. RECOMMENDATIONS
   - Prioritized remediation roadmap
   - Quick wins (easy fixes with high impact)
   - Long-term security improvements
   - Best practices and security controls
   - Security awareness recommendations

6. APPENDICES
   - Detailed technical data
   - Tool outputs and logs
   - Scan results
   - Code snippets
   - Network diagrams

WRITING GUIDELINES:

Technical Sections:
- Be precise and specific
- Include technical details and evidence
- Use industry-standard terminology
- Provide command examples and code snippets
- Reference CVEs, CWEs, OWASP categories

Executive Sections:
- Use clear, non-technical language
- Focus on business impact and risk
- Provide actionable recommendations
- Use analogies when helpful
- Quantify risk when possible

Severity Ratings:
- CRITICAL: Immediate threat, easily exploitable, high impact
- HIGH: Significant risk, exploitable with moderate effort
- MEDIUM: Moderate risk, requires specific conditions
- LOW: Minor risk, difficult to exploit, low impact
- INFORMATIONAL: No direct security impact, best practice

Remediation Recommendations:
- Be specific and actionable
- Provide code examples or configuration changes
- Include verification steps
- Estimate effort (hours/days)
- Link to security best practices

OUTPUT FORMAT:

Generate report in Markdown format with the following structure:

```markdown
# Security Assessment Report

**Date:** [Date]
**Prepared for:** [Client]
**Prepared by:** AI Autonom Security Team

---

## Executive Summary

[3-5 paragraphs summarizing key findings and recommendations]

### Risk Summary
- Critical: X findings
- High: Y findings
- Medium: Z findings
- Low: N findings

### Top Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

---

## Scope and Methodology

### Scope
[Description of tested systems and applications]

### Methodology
[Testing approach and techniques]

---

## Findings Summary

[Statistics and categorization]

---

## Detailed Findings

### [CRITICAL] Finding 1: [Title]

**Severity:** Critical
**Affected System:** [System]
**CVSS Score:** [Score if applicable]

**Description:**
[Detailed technical description]

**Impact:**
[Business and technical impact]

**Proof of Concept:**
```
[Steps to reproduce or exploit code]
```

**Evidence:**
[Screenshots, logs, or output]

**Remediation:**
[Specific fix with code examples]

**References:**
- CVE-XXXX-XXXXX
- CWE-XXX
- OWASP A0X

---

[Repeat for each finding]

---

## Recommendations

### Immediate Actions (0-30 days)
1. [Action]
2. [Action]

### Short-term (1-3 months)
1. [Action]

### Long-term (3-6 months)
1. [Action]

---

## Conclusion

[Summary and final thoughts]

---

## Appendices

### Appendix A: Technical Details
[Detailed technical data]

### Appendix B: Tool Outputs
[Scan results and logs]
```

TOOLS AVAILABLE:
- filesystem_read: Read findings from previous agents
- filesystem_write: Save report to file
- json_parse/format: Process structured findings data

DELIVERABLE:
Generate a professional, comprehensive security assessment report in Markdown format, saved to the specified output file.

Be thorough, accurate, and professional. This report may be delivered to clients."""

    def get_instructions(self, context: Dict[str, Any]) -> str:
        """Get dynamic instructions based on context"""
        client = context.get("client", "Client")
        findings = context.get("findings", [])
        assessment_type = context.get("assessment_type", "Comprehensive Security Assessment")
        output_file = context.get("output_file", "security_report.md")
        
        # Count by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for finding in findings:
            severity = finding.get("severity", "").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        instructions = f"""SECURITY REPORT GENERATION TASK

Client: {client}
Assessment Type: {assessment_type}
Output File: {output_file}
Date: {datetime.now().strftime("%Y-%m-%d")}

FINDINGS SUMMARY:
- Critical: {severity_counts['critical']}
- High: {severity_counts['high']}
- Medium: {severity_counts['medium']}
- Low: {severity_counts['low']}
- Informational: {severity_counts['info']}
Total: {len(findings)} findings

"""
        
        if findings:
            instructions += "TOP FINDINGS:\n"
            # Show top 3 most severe
            critical_high = [f for f in findings if f.get("severity", "").lower() in ["critical", "high"]]
            for i, finding in enumerate(critical_high[:3], 1):
                instructions += f"{i}. [{finding.get('severity', 'Unknown')}] {finding.get('title', 'Unknown')}\n"
            instructions += "\n"
        
        instructions += """REPORT REQUIREMENTS:

1. Create comprehensive Markdown report following the standard structure
2. Write executive summary for non-technical audience
3. Document all findings with full technical details
4. Provide specific, actionable remediation steps
5. Include evidence and proof-of-concept for each finding
6. Prioritize recommendations by risk and ease of implementation
7. Use professional tone throughout

REPORT SECTIONS REQUIRED:
✓ Executive Summary
✓ Scope and Methodology
✓ Findings Summary (with statistics)
✓ Detailed Findings (all vulnerabilities)
✓ Recommendations (prioritized)
✓ Conclusion
✓ Appendices (technical details)

OUTPUT:
Save the complete report to: {output_file}

Begin report generation now. Be thorough and professional."""
        
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
report_agent = ReportAgent()


if __name__ == "__main__":
    agent = ReportAgent()
    
    print("="*60)
    print("REPORT AGENT")
    print("="*60)
    print(f"ID: {agent.id}")
    print(f"Name: {agent.name}")
    print(f"Model: {agent.model_name}")
    print(f"Provider: {agent.provider}")
    print(f"Capabilities: {', '.join(agent.capabilities)}")
    print(f"Tools: {', '.join(agent.tools)}")
    print(f"Handoffs: {', '.join(agent.handoffs) if agent.handoffs else 'None (terminal agent)'}")
    print(f"\nSystem Prompt Length: {len(agent.get_system_prompt())} chars")
    
    # Test instructions
    context = {
        "client": "Acme Corp",
        "assessment_type": "Web Application Penetration Test",
        "output_file": "acme_security_report.md",
        "findings": [
            {"title": "SQL Injection in login", "severity": "Critical"},
            {"title": "XSS in comments", "severity": "High"},
            {"title": "Weak password policy", "severity": "Medium"},
            {"title": "Missing security headers", "severity": "Low"}
        ]
    }
    instructions = agent.get_instructions(context)
    print(f"Instructions Length: {len(instructions)} chars")
    print("\n" + "="*60)
