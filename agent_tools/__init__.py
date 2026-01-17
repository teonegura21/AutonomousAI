"""
Agent Tools - Dynamic Tool Directory
=====================================
This directory stores tools dynamically created by agents during execution.

When an agent needs a custom tool that doesn't exist, it can:
1. Generate the Python code for the tool
2. Save it here as a .py file
3. Register it with the ToolExecutor
4. Use it for subsequent tasks

These tools persist across sessions and can be reused.

Example structure:
    agent_tools/
    ├── __init__.py
    ├── parse_csv.py          # Custom CSV parser created by coder agent
    ├── api_validator.py      # API validation tool
    └── log_analyzer.py       # Custom log analyzer

Usage by agents:
    from agent_tools import parse_csv
    result = parse_csv.run(data)
"""

import os
import importlib
import sys
from pathlib import Path
from typing import Dict, Any, Callable, Optional

# Add this directory to path for dynamic imports
AGENT_TOOLS_DIR = Path(__file__).parent
if str(AGENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS_DIR))


def list_available_tools() -> list:
    """List all dynamically created tools"""
    tools = []
    for file in AGENT_TOOLS_DIR.glob("*.py"):
        if file.name.startswith("_"):
            continue
        tools.append(file.stem)
    return tools


def load_tool(tool_name: str) -> Optional[Any]:
    """Dynamically load a tool module"""
    try:
        return importlib.import_module(tool_name)
    except ImportError:
        return None


def create_tool(
    tool_name: str,
    code: str,
    description: str = "",
    created_by: str = "agent"
) -> bool:
    """
    Create a new dynamic tool
    
    Args:
        tool_name: Name of the tool (will become filename)
        code: Python code for the tool
        description: Tool description
        created_by: Which agent created this tool
    
    Returns:
        True if created successfully
    """
    # Sanitize tool name
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tool_name)
    filepath = AGENT_TOOLS_DIR / f"{safe_name}.py"
    
    # Add header
    header = f'''"""
Dynamic Tool: {tool_name}
Created by: {created_by}
Description: {description}
"""

'''
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + code)
        return True
    except Exception as e:
        print(f"[AGENT_TOOLS] Failed to create tool: {e}")
        return False


__all__ = ['list_available_tools', 'load_tool', 'create_tool']
