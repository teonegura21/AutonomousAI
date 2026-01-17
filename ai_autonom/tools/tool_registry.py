#!/usr/bin/env python3
"""
Tool Registry
Central registry of all available tools - modular and dynamic
"""

from typing import Dict, Callable, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolDefinition:
    """Definition of an executable tool"""
    id: str
    name: str
    description: str
    category: str  # 'filesystem', 'code_execution', 'web', 'database', 'analysis'
    function: Callable
    requires_sandbox: bool = False
    parameters: Dict[str, str] = field(default_factory=dict)  # param_name -> description
    returns: str = ""
    examples: List[str] = field(default_factory=list)
    created_by: str = "system"  # "system" or agent_id for dynamic tools
    use_count: int = 0
    last_used: Optional[str] = None


class ToolRegistry:
    """
    Central registry of all available tools.
    Tools are MODULAR - can be added/removed dynamically.
    Orchestrator queries this to decide which tools to assign.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}  # category -> [tool_ids]
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool"""
        self.tools[tool.id] = tool
        
        # Track by category
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.id not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.id)
        
        print(f"[TOOLS] Registered: {tool.name} ({tool.id})")
    
    def unregister(self, tool_id: str) -> bool:
        """Unregister a tool"""
        if tool_id in self.tools:
            tool = self.tools.pop(tool_id)
            if tool.category in self._categories:
                self._categories[tool.category] = [
                    t for t in self._categories[tool.category] if t != tool_id
                ]
            print(f"[TOOLS] Unregistered: {tool_id}")
            return True
        return False
    
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool by ID"""
        return self.tools.get(tool_id)
    
    def execute(self, tool_id: str, **kwargs) -> Any:
        """Execute a tool by ID"""
        tool = self.tools.get(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")
        
        # Update usage stats
        tool.use_count += 1
        tool.last_used = datetime.now().isoformat()
        
        return tool.function(**kwargs)
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a category"""
        tool_ids = self._categories.get(category, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def get_tools_for_capability(self, capability: str) -> List[ToolDefinition]:
        """Get tools matching a capability"""
        mapping = {
            "code_generation": ["filesystem_write", "python_exec", "bash_exec"],
            "code_execution": ["python_exec", "bash_exec"],
            "web_search": ["web_search", "web_fetch"],
            "file_operations": ["filesystem_read", "filesystem_write", "filesystem_search"],
            "testing": ["python_exec", "pytest_run"],
            "debugging": ["python_exec", "filesystem_read"],
            "documentation": ["filesystem_read", "filesystem_write"],
            "database": ["sqlite_query", "chromadb"],
        }
        tool_ids = mapping.get(capability, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def list_all(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def list_by_sandbox_requirement(self, requires_sandbox: bool) -> List[ToolDefinition]:
        """Get tools filtered by sandbox requirement"""
        return [t for t in self.tools.values() if t.requires_sandbox == requires_sandbox]
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions for orchestrator prompt"""
        lines = []
        for category in sorted(self._categories.keys()):
            lines.append(f"\n## {category.upper()}")
            for tool_id in self._categories[category]:
                tool = self.tools.get(tool_id)
                if tool:
                    params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
                    sandbox = " [SANDBOX]" if tool.requires_sandbox else ""
                    lines.append(f"- {tool.id}: {tool.description}{sandbox}")
                    if params:
                        lines.append(f"  Parameters: {params}")
        return "\n".join(lines)
    
    def get_tool_list_for_agent(self, agent_capabilities: List[str]) -> List[str]:
        """Get list of tool IDs appropriate for an agent's capabilities"""
        tool_ids = set()
        for cap in agent_capabilities:
            for tool in self.get_tools_for_capability(cap):
                tool_ids.add(tool.id)
        return list(tool_ids)
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self._categories.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "categories": len(self._categories),
            "by_category": {cat: len(tools) for cat, tools in self._categories.items()},
            "sandbox_required": len(self.list_by_sandbox_requirement(True)),
            "most_used": sorted(
                [(t.id, t.use_count) for t in self.tools.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: ToolDefinition) -> None:
    """Register a tool in the global registry"""
    get_global_registry().register(tool)


if __name__ == "__main__":
    # Test tool registry
    registry = ToolRegistry()
    
    print("\n" + "="*60)
    print("TOOL REGISTRY TEST")
    print("="*60 + "\n")
    
    # Register a test tool
    def dummy_tool(**kwargs):
        return f"Executed with: {kwargs}"
    
    registry.register(ToolDefinition(
        id="test_tool",
        name="Test Tool",
        description="A test tool",
        category="testing",
        function=dummy_tool,
        requires_sandbox=False,
        parameters={"input": "The input string"}
    ))
    
    # Execute
    result = registry.execute("test_tool", input="hello")
    print(f"Execution result: {result}")
    
    # Stats
    print(f"\nRegistry stats: {registry.get_stats()}")
    
    # Descriptions
    print("\nTool descriptions:")
    print(registry.get_tool_descriptions())
