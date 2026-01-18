#!/usr/bin/env python3
"""
Capability Checker - Validates if tasks can be completed with available tools.

PROBLEM SOLVED:
An agent asked to "create a .exe file" when only Python is available MUST:
1. Detect that .exe creation requires a compiler (gcc/mingw) or pyinstaller
2. Check if these tools exist
3. If not -> STOP and report IMPOSSIBLE_TASK instead of hallucinating

This module provides:
- Task requirement detection
- Tool availability checking
- Honest failure messaging
"""

import re
import shutil
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class CapabilityCheck:
    """Result of capability verification."""
    can_complete: bool
    missing_tools: List[str]
    missing_capabilities: List[str]
    alternatives: List[str]
    explanation: str
    
    def to_prompt_message(self) -> str:
        """Generate message for agent prompt."""
        if self.can_complete:
            return ""
        
        msg = f"*** IMPOSSIBLE TASK DETECTED ***\n"
        msg += f"Reason: {self.explanation}\n"
        
        if self.missing_tools:
            msg += f"Missing tools: {', '.join(self.missing_tools)}\n"
        
        if self.alternatives:
            msg += f"Alternatives: {', '.join(self.alternatives)}\n"
        else:
            msg += "No alternatives available. You MUST report this to the user.\n"
        
        return msg


# =============================================================================
# TASK REQUIREMENT MAPPINGS
# =============================================================================

# What capabilities are needed for specific task types
TASK_REQUIREMENTS: Dict[str, Dict[str, any]] = {
    # Compilation tasks
    "create_exe": {
        "patterns": [r"\.exe\b", r"windows\s+executable", r"compile.*windows", r"cross.?compile"],
        "requires_tools": ["x86_64-w64-mingw32-gcc", "x86_64-w64-mingw32-g++", "pyinstaller"],
        "any_of": True,  # Any ONE of these tools is sufficient
        "cannot_use": ["python"],  # Python alone cannot do this
        "explanation": "Creating .exe files requires a Windows cross-compiler (mingw-w64) or PyInstaller.",
        "alternatives": [
            "Install mingw-w64: apt install mingw-w64",
            "Install PyInstaller: pip install pyinstaller",
            "Create a Linux binary instead (gcc/g++)"
        ]
    },
    "create_linux_binary": {
        "patterns": [r"compile\s+c\b", r"compile\s+c\+\+", r"create.*elf", r"linux\s+binary"],
        "requires_tools": ["gcc", "g++", "clang"],
        "any_of": True,
        "explanation": "Compiling C/C++ requires gcc, g++, or clang.",
        "alternatives": ["Install build-essential: apt install build-essential"]
    },
    "network_scan": {
        "patterns": [r"\bnmap\b", r"port\s+scan", r"network\s+scan", r"service\s+detection"],
        "requires_tools": ["nmap"],
        "explanation": "Network scanning requires nmap.",
        "alternatives": ["Use netcat for basic port checks: nc -zv host port"]
    },
    "web_exploit": {
        "patterns": [r"\bsqlmap\b", r"sql\s+injection\s+test"],
        "requires_tools": ["sqlmap"],
        "explanation": "Automated SQL injection testing requires sqlmap.",
        "alternatives": ["Manual SQL injection testing with curl"]
    },
    "metasploit": {
        "patterns": [r"\bmsf\b", r"metasploit", r"msfconsole", r"meterpreter"],
        "requires_tools": ["msfconsole"],
        "explanation": "Metasploit framework required.",
        "alternatives": ["Use manual exploitation techniques", "Use searchsploit for exploit code"]
    },
    "password_crack": {
        "patterns": [r"\bjohn\b", r"hashcat", r"password\s+crack", r"hash\s+crack"],
        "requires_tools": ["john", "hashcat"],
        "any_of": True,
        "explanation": "Password cracking requires john or hashcat.",
        "alternatives": ["Check for default/common passwords manually"]
    },
    "memory_analysis": {
        "patterns": [r"volatility", r"memory\s+dump", r"memory\s+forensics"],
        "requires_tools": ["volatility", "volatility3"],
        "any_of": True,
        "explanation": "Memory forensics requires Volatility framework.",
        "alternatives": []
    },
    "docker_required": {
        "patterns": [r"docker\s+run", r"container\s+exec", r"docker\s+build"],
        "requires_tools": ["docker"],
        "explanation": "Docker operations require docker to be installed and running.",
        "alternatives": []
    }
}

# Common impossible task patterns
IMPOSSIBLE_PATTERNS: List[Dict] = [
    {
        "pattern": r"python.*\.exe\b(?!.*pyinstaller)",
        "reason": "Python cannot directly create .exe files. PyInstaller or a compiler is needed.",
        "suggestion": "ASK_USER: 'Creating a Windows .exe requires PyInstaller or mingw-w64. Should I install one?'"
    },
    {
        "pattern": r"(?:compile|build).*(?:c\+\+|cpp|c\b).*(?:no|without).*compiler",
        "reason": "Cannot compile C/C++ without a compiler installed.",
        "suggestion": "ASK_USER: 'A C/C++ compiler (gcc/g++) is not installed. Should I install build-essential?'"
    },
    {
        "pattern": r"crack.*hash.*(?:no|without).*(?:john|hashcat)",
        "reason": "Cannot crack password hashes without cracking tools.",
        "suggestion": "ASK_USER: 'Password cracking requires john or hashcat. Neither is installed.'"
    },
]


# =============================================================================
# TOOL AVAILABILITY CHECKING
# =============================================================================

def check_tool_exists(tool_name: str) -> bool:
    """
    Check if a tool exists in PATH.
    
    Args:
        tool_name: Name of the tool to check
    
    Returns:
        True if tool exists
    """
    return shutil.which(tool_name) is not None


def check_tools_available(tools: List[str], any_of: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if required tools are available.
    
    Args:
        tools: List of tool names to check
        any_of: If True, only ONE tool needs to exist
    
    Returns:
        Tuple of (all_available, list_of_missing)
    """
    missing = [tool for tool in tools if not check_tool_exists(tool)]
    
    if any_of:
        # At least one must exist
        available = len(missing) < len(tools)
    else:
        # All must exist
        available = len(missing) == 0
    
    return available, missing


def detect_task_type(task_description: str) -> List[str]:
    """
    Detect what type of task this is based on description.
    
    Returns list of matching task type keys.
    """
    task_lower = task_description.lower()
    matches = []
    
    for task_type, config in TASK_REQUIREMENTS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, task_lower, re.IGNORECASE):
                matches.append(task_type)
                break
    
    return matches


def check_impossible_patterns(task_description: str) -> Optional[Dict]:
    """
    Check if task matches known impossible patterns.
    
    Returns matching pattern info or None.
    """
    task_lower = task_description.lower()
    
    for pattern_info in IMPOSSIBLE_PATTERNS:
        if re.search(pattern_info["pattern"], task_lower, re.IGNORECASE):
            return pattern_info
    
    return None


# =============================================================================
# MAIN CAPABILITY CHECKER
# =============================================================================

class CapabilityChecker:
    """
    Validates if tasks can be completed with available tools.
    
    Usage:
        checker = CapabilityChecker()
        result = checker.validate("Create a Windows .exe from my Python script")
        
        if not result.can_complete:
            print(result.to_prompt_message())
            # Output: IMPOSSIBLE TASK DETECTED...
    """
    
    def __init__(self, cached_tools: Optional[Set[str]] = None):
        """
        Initialize checker.
        
        Args:
            cached_tools: Pre-checked set of available tools
        """
        self._tool_cache: Dict[str, bool] = {}
        if cached_tools:
            for tool in cached_tools:
                self._tool_cache[tool] = True
    
    def tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists (with caching)."""
        if tool_name not in self._tool_cache:
            self._tool_cache[tool_name] = check_tool_exists(tool_name)
        return self._tool_cache[tool_name]
    
    def validate(self, task_description: str) -> CapabilityCheck:
        """
        Validate if a task can be completed.
        
        Args:
            task_description: Natural language task description
        
        Returns:
            CapabilityCheck result
        """
        # Check for known impossible patterns first
        impossible = check_impossible_patterns(task_description)
        if impossible:
            return CapabilityCheck(
                can_complete=False,
                missing_tools=[],
                missing_capabilities=[impossible["reason"]],
                alternatives=[impossible["suggestion"]],
                explanation=impossible["reason"]
            )
        
        # Detect task types
        task_types = detect_task_type(task_description)
        
        if not task_types:
            # Unknown task type - assume possible
            return CapabilityCheck(
                can_complete=True,
                missing_tools=[],
                missing_capabilities=[],
                alternatives=[],
                explanation="Task type recognized, tools available."
            )
        
        # Check each detected task type
        all_missing_tools = []
        all_alternatives = []
        all_explanations = []
        
        for task_type in task_types:
            config = TASK_REQUIREMENTS[task_type]
            required_tools = config.get("requires_tools", [])
            any_of = config.get("any_of", False)
            
            available, missing = check_tools_available(required_tools, any_of)
            
            if not available:
                all_missing_tools.extend(missing)
                all_alternatives.extend(config.get("alternatives", []))
                all_explanations.append(config["explanation"])
        
        if all_missing_tools:
            return CapabilityCheck(
                can_complete=False,
                missing_tools=list(set(all_missing_tools)),
                missing_capabilities=[],
                alternatives=list(set(all_alternatives)),
                explanation=" | ".join(all_explanations)
            )
        
        return CapabilityCheck(
            can_complete=True,
            missing_tools=[],
            missing_capabilities=[],
            alternatives=[],
            explanation="All required tools available."
        )
    
    def get_preflight_commands(self, task_description: str) -> List[str]:
        """
        Get commands to run before task to verify capabilities.
        
        Returns list of 'which' commands to run.
        """
        task_types = detect_task_type(task_description)
        tools_to_check = set()
        
        for task_type in task_types:
            config = TASK_REQUIREMENTS.get(task_type, {})
            tools_to_check.update(config.get("requires_tools", []))
        
        return [f"which {tool}" for tool in tools_to_check]
    
    def generate_validation_prompt(self, task_description: str) -> str:
        """
        Generate prompt addition for capability validation.
        
        This should be appended to agent system prompts.
        """
        result = self.validate(task_description)
        
        if not result.can_complete:
            return f"""
*** CAPABILITY CHECK FAILED ***

{result.to_prompt_message()}

YOU MUST:
1. Do NOT attempt this task as-is
2. Output: IMPOSSIBLE_TASK: {result.explanation}
3. Suggest alternatives or ASK_USER how to proceed
"""
        
        # Even if tools seem available, force verification
        preflight = self.get_preflight_commands(task_description)
        
        if preflight:
            return f"""
*** PRE-FLIGHT VERIFICATION REQUIRED ***

Before starting, run these commands to verify tools:
{chr(10).join(preflight)}

If ANY command returns empty/error:
1. STOP immediately
2. Output: IMPOSSIBLE_TASK: Required tool not installed
3. ASK_USER how to proceed
"""
        
        return ""


# =============================================================================
# ENHANCED SYSTEM PROMPT ADDITIONS
# =============================================================================

HONEST_FAILURE_PROMPT = """
*** CRITICAL: HONEST FAILURE PROTOCOL ***

Before attempting ANY task, you MUST verify capability:

1. VERIFY TOOLS EXIST:
   - Run "which <tool>" for every tool you plan to use
   - If tool doesn't exist -> CANNOT proceed

2. DETECT IMPOSSIBLE TASKS:
   These tasks CANNOT be done without specific tools:
   - Creating .exe files: Requires mingw-w64 or PyInstaller (NOT just Python)
   - Compiling C/C++: Requires gcc/g++/clang (NOT just Python)
   - Network scanning: Requires nmap (cannot be emulated)
   - Password cracking: Requires john/hashcat
   - Memory forensics: Requires volatility

3. WHEN TASK IS IMPOSSIBLE:
   DO NOT:
   - Pretend to do it
   - Use Python to "simulate" missing tools
   - Say "I'll try anyway"
   - Hallucinate output
   
   DO:
   - Stop immediately
   - Output: IMPOSSIBLE_TASK: [clear explanation]
   - Suggest: What tool is needed and how to install it
   - Then output: ASK_USER: "Should I install the required tool?"

4. EXAMPLES OF HONEST FAILURE:

   User: "Create a Windows .exe from my script"
   You: 
   TOOL: cai_generic_linux_command
   command: which x86_64-w64-mingw32-gcc
   
   [Output: empty/error]
   
   IMPOSSIBLE_TASK: Cannot create Windows .exe - no cross-compiler installed.
   Required: mingw-w64 (apt install mingw-w64) or PyInstaller (pip install pyinstaller)
   ASK_USER: "Should I install mingw-w64 to enable Windows executable creation?"

   ---

   User: "Scan the network for open ports"
   You:
   TOOL: cai_generic_linux_command
   command: which nmap
   
   [Output: /usr/bin/nmap]
   
   Great, nmap is available. Proceeding...

5. NEVER say COMPLETE if the actual task wasn't done!
"""


def get_enhanced_system_prompt(base_prompt: str, task_description: str) -> str:
    """
    Enhance system prompt with capability checking.
    
    Args:
        base_prompt: Original system prompt
        task_description: Task to be performed
    
    Returns:
        Enhanced prompt with validation rules
    """
    checker = CapabilityChecker()
    validation_prompt = checker.generate_validation_prompt(task_description)
    
    return f"""{base_prompt}

{HONEST_FAILURE_PROMPT}

{validation_prompt}
"""


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAPABILITY CHECKER TEST")
    print("="*60 + "\n")
    
    checker = CapabilityChecker()
    
    test_cases = [
        "Create a Windows .exe from my Python script",
        "Compile my C++ code into an executable",
        "Scan the network with nmap",
        "Crack the MD5 hash with john",
        "Run Metasploit to exploit the target",
        "Analyze the memory dump with volatility",
        "List files in the current directory",  # Should pass
    ]
    
    for task in test_cases:
        print(f"Task: {task}")
        result = checker.validate(task)
        
        if result.can_complete:
            print(f"  ✓ CAN COMPLETE")
        else:
            print(f"  ✗ IMPOSSIBLE: {result.explanation}")
            if result.missing_tools:
                print(f"    Missing: {result.missing_tools}")
            if result.alternatives:
                print(f"    Alternatives: {result.alternatives[:2]}")
        print()
    
    print("="*60)
    print("Test complete!")
    print("="*60)
