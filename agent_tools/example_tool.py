"""
Dynamic Tool: example_tool
Created by: system
Description: Example dynamic tool to demonstrate the agent_tools structure

This is an example of how agents can create custom tools during execution.
"""

from typing import Any, Dict


def run(data: Any) -> Dict[str, Any]:
    """
    Main entry point for the tool.
    
    All dynamic tools should have a `run()` function as the main entry point.
    
    Args:
        data: Input data to process
        
    Returns:
        Dictionary with results
    """
    return {
        "success": True,
        "message": "Example tool executed successfully",
        "input_type": type(data).__name__,
        "input_preview": str(data)[:100]
    }


def validate(data: Any) -> bool:
    """Optional validation function"""
    return data is not None


# Tool metadata
TOOL_INFO = {
    "name": "example_tool",
    "description": "Example dynamic tool template",
    "version": "1.0.0",
    "created_by": "system",
    "parameters": {
        "data": "Any input data to process"
    },
    "returns": "Dictionary with processing results"
}


if __name__ == "__main__":
    # Test the tool
    result = run("test data")
    print(f"Result: {result}")
