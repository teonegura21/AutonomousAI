# Tools Module

## Purpose
The tools module provides a unified interface for executing various tools including system commands, file operations, security tools, and custom scripts. It supports both local and Docker-sandboxed execution.

## Key Files

### `tool_registry.py`
**Purpose:** Central registry for discovering and managing available tools.

**Key Classes:**
- `ToolCategory`: Enum for tool categories
  - SYSTEM: System commands (ls, ps, etc.)
  - FILE: File operations (read, write, etc.)
  - SECURITY: Security tools (nmap, nikto, etc.)
  - NETWORK: Network tools (ping, curl, etc.)
  - CODE: Code analysis tools (linter, formatter, etc.)
  - WEB: Web tools (browser, scraper, etc.)
  - CUSTOM: Custom scripts and tools

- `Tool`: Tool definition
  - `name`: Tool identifier
  - `category`: ToolCategory
  - `description`: Tool description
  - `command_template`: Command template with placeholders
  - `requires_docker`: Whether tool needs Docker sandbox
  - `docker_image`: Docker image to use
  - `input_schema`: JSON schema for inputs
  - `output_parser`: Function to parse tool output

- `ToolRegistry`: Tool lifecycle management
  - `register_tool(tool)`: Register new tool
  - `get_tool(name)`: Get tool by name
  - `find_tools(category)`: Find tools by category
  - `list_tools()`: List all registered tools
  - `tool_exists(name)`: Check if tool exists
  - `get_tools_for_agent(agent_name)`: Get tools assigned to agent

**Dependencies:**
- External: `sqlite3`, `json`, `enum`, `dataclasses`
- Internal: None

### `tool_executor.py`
**Purpose:** Execute tools locally or in Docker sandboxes with timeout and error handling.

**Key Classes:**
- `ExecutionMode`: Enum for execution modes
  - LOCAL: Execute locally
  - DOCKER: Execute in Docker container
  - AUTO: Automatically choose based on tool config

- `ToolExecutor`: Tool execution engine
  - `execute(tool_name, inputs, mode)`: Execute tool with inputs
  - `execute_local(tool, inputs)`: Local execution
  - `execute_docker(tool, inputs)`: Docker execution
  - `_validate_inputs(tool, inputs)`: Validate inputs against schema
  - `_render_command(tool, inputs)`: Render command from template
  - `_parse_output(tool, output)`: Parse tool output
  - `_handle_timeout(tool, timeout)`: Handle execution timeout
  - `_handle_error(tool, error)`: Handle execution error

**Key Functions:**
- `register_system_tools()`: Register common system tools (ls, ps, cat, etc.)
- `register_security_tools()`: Register security tools (nmap, nikto, sqlmap, etc.)
- `register_code_tools()`: Register code analysis tools (pylint, black, mypy, etc.)

**Dependencies:**
- External: `subprocess`, `json`, `jsonschema`, `logging`, `timeout_decorator`
- Internal: 
  - `ai_autonom.tools.tool_registry`: Tool definitions
  - `ai_autonom.sandbox.docker_executor`: Docker execution

## Internal Architecture

```
┌──────────────────────────────────────────┐
│         ToolRegistry                      │
│  (Tool discovery & management)            │
│  - Tool registration                      │
│  - Category-based search                  │
│  - Agent tool mapping                     │
└─────────────┬────────────────────────────┘
              │
              │ provides tools to
              ▼
┌──────────────────────────────────────────┐
│         ToolExecutor                      │
│  (Execution engine)                       │
│  - Input validation                       │
│  - Command rendering                      │
│  - Output parsing                         │
│  - Error handling                         │
└─────────────┬────────────────────────────┘
              │
              │ executes via
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌─────────┐     ┌──────────────┐
│ Local   │     │   Docker     │
│Execution│     │ Sandbox      │
└─────────┘     └──────────────┘
```

## Usage Examples

### Tool Registry
```python
from ai_autonom.tools.tool_registry import ToolRegistry, Tool, ToolCategory

# Initialize registry
registry = ToolRegistry()

# Register custom tool
tool = Tool(
    name="my_scanner",
    category=ToolCategory.SECURITY,
    description="Custom security scanner",
    command_template="python scan.py --target {target} --mode {mode}",
    requires_docker=True,
    docker_image="security-tools",
    input_schema={
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "mode": {"type": "string", "enum": ["fast", "full"]}
        },
        "required": ["target"]
    }
)
registry.register_tool(tool)

# Find tools by category
security_tools = registry.find_tools(ToolCategory.SECURITY)
for tool in security_tools:
    print(f"{tool.name}: {tool.description}")

# Get tools for agent
pentester_tools = registry.get_tools_for_agent("web_pentester")
```

### Tool Execution
```python
from ai_autonom.tools.tool_executor import ToolExecutor, ExecutionMode

# Initialize executor
executor = ToolExecutor()

# Execute tool locally
result = executor.execute(
    tool_name="ls",
    inputs={"path": "/tmp", "options": "-la"},
    mode=ExecutionMode.LOCAL
)
print(result["stdout"])

# Execute tool in Docker
result = executor.execute(
    tool_name="nmap",
    inputs={"target": "192.168.1.1", "scan_type": "syn"},
    mode=ExecutionMode.DOCKER
)
print(result["parsed_output"])

# Auto-select execution mode
result = executor.execute(
    tool_name="my_scanner",
    inputs={"target": "example.com", "mode": "fast"},
    mode=ExecutionMode.AUTO
)
```

### Registering Built-in Tools
```python
from ai_autonom.tools.tool_executor import (
    register_system_tools,
    register_security_tools,
    register_code_tools
)

# Register all built-in tools
register_system_tools()
register_security_tools()
register_code_tools()

# Now tools are available
executor = ToolExecutor()
result = executor.execute("nmap", {"target": "192.168.1.1"})
```

### Custom Tool with Output Parser
```python
from ai_autonom.tools.tool_registry import Tool, ToolRegistry, ToolCategory
import json

def parse_scan_output(output):
    """Parse custom scanner output"""
    lines = output.split("\n")
    findings = []
    for line in lines:
        if "VULNERABILITY:" in line:
            findings.append(line.split(":")[1].strip())
    return {"vulnerabilities": findings, "count": len(findings)}

tool = Tool(
    name="custom_scanner",
    category=ToolCategory.SECURITY,
    description="Custom vulnerability scanner",
    command_template="./scanner {target}",
    requires_docker=True,
    docker_image="scanner:latest",
    input_schema={
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"]
    },
    output_parser=parse_scan_output
)

registry = ToolRegistry()
registry.register_tool(tool)
```

## Dependencies

**External Dependencies:**
- `subprocess`: Local command execution
- `json`: Data serialization
- `jsonschema`: Input validation
- `logging`: Event logging
- `timeout_decorator`: Execution timeout handling
- `sqlite3`: Tool registry persistence

**Internal Dependencies:**
- `ai_autonom.sandbox.docker_executor`: Docker sandbox execution

## Important Functionality

1. **Tool Discovery**: Category-based tool search and discovery
2. **Input Validation**: JSON schema-based input validation
3. **Command Templating**: Jinja2-style command templates with variable substitution
4. **Output Parsing**: Structured output from tool execution
5. **Execution Modes**: Local, Docker, or auto-selection
6. **Timeout Handling**: Configurable timeouts with automatic termination
7. **Error Recovery**: Graceful error handling with detailed error messages
8. **Agent Tool Mapping**: Associate tools with specific agents
9. **Custom Tools**: Easy registration of custom tools
10. **Persistent Registry**: SQLite-based tool registry

## Built-in Tools

### System Tools
- `ls`: List directory contents
- `cat`: Read file contents
- `grep`: Search file contents
- `ps`: List processes
- `kill`: Terminate processes

### Security Tools
- `nmap`: Network port scanner
- `nikto`: Web server scanner
- `sqlmap`: SQL injection testing
- `burp`: Web application security testing
- `metasploit`: Penetration testing framework

### Code Tools
- `pylint`: Python linter
- `black`: Python code formatter
- `mypy`: Python type checker
- `pytest`: Python test runner
- `coverage`: Code coverage analysis

## Tool Registration Pattern

```python
from ai_autonom.tools.tool_registry import Tool, ToolRegistry, ToolCategory

def register_my_tools():
    """Register custom tools"""
    registry = ToolRegistry()
    
    tools = [
        Tool(
            name="tool1",
            category=ToolCategory.CUSTOM,
            description="...",
            command_template="...",
            requires_docker=False,
            input_schema={...}
        ),
        Tool(
            name="tool2",
            category=ToolCategory.CUSTOM,
            description="...",
            command_template="...",
            requires_docker=True,
            docker_image="custom:latest",
            input_schema={...}
        )
    ]
    
    for tool in tools:
        registry.register_tool(tool)
```

## Command Template Syntax

Templates use Python string formatting:
```python
# Simple substitution
"nmap {target}"

# Multiple parameters
"nmap -p {ports} -sV {target}"

# Optional parameters with defaults
"nmap {target} {options}"  # options can be empty

# Complex templates
"python scanner.py --target {target} --output {output_file} {flags}"
```

## Input Schema Examples

```python
# Simple string input
{
    "type": "object",
    "properties": {
        "target": {"type": "string"}
    },
    "required": ["target"]
}

# Multiple typed inputs
{
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "timeout": {"type": "number"},
        "verbose": {"type": "boolean"}
    },
    "required": ["target"]
}

# Enum validation
{
    "type": "object",
    "properties": {
        "scan_type": {"type": "string", "enum": ["syn", "ack", "udp"]},
        "target": {"type": "string"}
    },
    "required": ["target", "scan_type"]
}
```

## Error Handling

```python
from ai_autonom.tools.tool_executor import ToolExecutor

executor = ToolExecutor()

try:
    result = executor.execute("nmap", {"target": "invalid"})
except ValueError as e:
    # Input validation error
    print(f"Invalid input: {e}")
except TimeoutError as e:
    # Execution timeout
    print(f"Tool timed out: {e}")
except RuntimeError as e:
    # Execution error
    print(f"Tool execution failed: {e}")
```
