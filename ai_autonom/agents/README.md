# Agents Module

## Purpose
The agents module defines specialized AI agents for different tasks including security assessment, code generation, analysis, and general-purpose operations. Each agent has specific capabilities, tools, and prompts tailored to their domain.

## Key Files

### `agent_registry.py`
**Purpose:** Central registry for managing and discovering agents across the system.

**Key Classes:**
- `AgentType`: Enum defining agent categories
  - GENERAL: General-purpose agents
  - SECURITY: Security-focused agents
  - CODE: Code-related agents
  - ANALYSIS: Analysis agents

- `AgentCapability`: Enum for agent capabilities
  - WEB_PENTEST: Web penetration testing
  - NETWORK_SCAN: Network scanning
  - CODE_REVIEW: Code review
  - VULNERABILITY_ANALYSIS: Vulnerability assessment
  - REPORT_GENERATION: Report creation
  - CODE_GENERATION: Code generation
  - CODE_REFACTORING: Code refactoring
  - TESTING: Test creation
  - DEBUGGING: Debugging
  - DOCUMENTATION: Documentation generation

- `Agent`: Agent definition dataclass
  - `name`: Agent identifier
  - `type`: AgentType
  - `capabilities`: List of capabilities
  - `description`: Agent description
  - `tools`: Available tools
  - `model`: LLM model to use
  - `prompt_template`: System prompt

- `AgentRegistry`: Agent lifecycle management
  - `register_agent(agent)`: Register new agent
  - `get_agent(name)`: Retrieve agent by name
  - `find_agents(capability, agent_type)`: Find agents by criteria
  - `list_agents()`: List all registered agents
  - `agent_exists(name)`: Check agent existence

**Dependencies:**
- External: `sqlite3`, `json`, `enum`, `dataclasses`
- Internal: None

### `security_agents.py`
**Purpose:** Specialized agents for security assessment following CAI security patterns.

**Key Agent Definitions:**
- `web_pentester`: Web application penetration testing
  - Capabilities: WEB_PENTEST, VULNERABILITY_ANALYSIS
  - Tools: nmap, nikto, sqlmap, burp
  - Prompt: Security expert focused on web vulnerabilities (OWASP Top 10)

- `retester`: Vulnerability validation and verification
  - Capabilities: WEB_PENTEST, VULNERABILITY_ANALYSIS
  - Tools: nmap, nikto, custom_scripts
  - Prompt: Retesting specialist verifying findings

- `report_agent`: Security report generation
  - Capabilities: REPORT_GENERATION, VULNERABILITY_ANALYSIS
  - Tools: report_generator
  - Prompt: Professional security report writer

**Key Functions:**
- `register_security_agents()`: Registers all security agents in registry

**Dependencies:**
- External: None
- Internal: `ai_autonom.agents.agent_registry`, `ai_autonom.core.config`

### `code_agents.py`
**Purpose:** Code generation, review, and refactoring agents.

**Key Agent Definitions:**
- `code_generator`: General code generation
  - Capabilities: CODE_GENERATION, DOCUMENTATION
  - Tools: python_interpreter, file_writer
  - Prompt: Expert programmer with best practices focus

- `code_reviewer`: Code quality and security review
  - Capabilities: CODE_REVIEW, VULNERABILITY_ANALYSIS
  - Tools: linter, static_analyzer
  - Prompt: Senior code reviewer for quality, security, performance

- `test_generator`: Automated test creation
  - Capabilities: TESTING, CODE_GENERATION
  - Tools: pytest, unittest
  - Prompt: Test engineer creating comprehensive test suites

**Key Functions:**
- `register_code_agents()`: Registers all code agents in registry

**Dependencies:**
- External: None
- Internal: `ai_autonom.agents.agent_registry`, `ai_autonom.core.config`

## Internal Architecture

```
┌─────────────────────────────────────────┐
│        AgentRegistry                     │
│  (Central agent management)              │
│  - SQLite persistence                    │
│  - Agent discovery                       │
│  - Capability matching                   │
└─────────────────┬───────────────────────┘
                  │
                  │ stores
                  ▼
┌─────────────────────────────────────────┐
│           Agent                          │
│  - name, type, capabilities              │
│  - tools, model, prompt                  │
└─────────────────────────────────────────┘
                  ▲
                  │
                  │ defines
    ┌─────────────┴─────────────┐
    │                           │
┌───▼────────────┐  ┌──────────▼────────┐
│ Security       │  │   Code            │
│ Agents         │  │   Agents          │
│ - web_pentester│  │ - code_generator  │
│ - retester     │  │ - code_reviewer   │
│ - report_agent │  │ - test_generator  │
└────────────────┘  └───────────────────┘
```

## Usage Examples

### Agent Registry
```python
from ai_autonom.agents.agent_registry import AgentRegistry, AgentCapability, AgentType

# Initialize registry
registry = AgentRegistry()

# Register agent
from ai_autonom.agents.agent_registry import Agent
agent = Agent(
    name="my_agent",
    type=AgentType.GENERAL,
    capabilities=[AgentCapability.CODE_REVIEW],
    description="Custom code reviewer",
    tools=["linter"],
    model="llama3:8b",
    prompt_template="You are a code reviewer..."
)
registry.register_agent(agent)

# Find agents by capability
security_agents = registry.find_agents(
    capability=AgentCapability.WEB_PENTEST
)

# Get specific agent
agent = registry.get_agent("web_pentester")

# List all agents
all_agents = registry.list_agents()
```

### Security Agents
```python
from ai_autonom.agents.security_agents import register_security_agents
from ai_autonom.agents.agent_registry import AgentRegistry

# Register all security agents
registry = AgentRegistry()
register_security_agents()

# Get web pentester
pentester = registry.get_agent("web_pentester")
print(pentester.capabilities)  # [WEB_PENTEST, VULNERABILITY_ANALYSIS]
print(pentester.tools)         # ["nmap", "nikto", "sqlmap", "burp"]
```

### Code Agents
```python
from ai_autonom.agents.code_agents import register_code_agents
from ai_autonom.agents.agent_registry import AgentRegistry

# Register all code agents
registry = AgentRegistry()
register_code_agents()

# Get code generator
generator = registry.get_agent("code_generator")
print(generator.capabilities)  # [CODE_GENERATION, DOCUMENTATION]

# Get test generator
test_gen = registry.get_agent("test_generator")
print(test_gen.tools)  # ["pytest", "unittest"]
```

## Dependencies

**External Dependencies:**
- `sqlite3`: Agent metadata persistence
- `json`: Serialization for agent data
- `enum`: Type definitions
- `dataclasses`: Agent data structure

**Internal Dependencies:**
- `ai_autonom.core.config`: Configuration access for model settings

## Important Functionality

1. **Agent Discovery**: Find agents by capability or type
2. **Persistent Storage**: SQLite-based agent registry with ACID guarantees
3. **Capability Matching**: Intelligent agent selection based on task requirements
4. **Flexible Registration**: Dynamic agent registration at runtime
5. **Tool Assignment**: Each agent has specific tools for their domain
6. **Prompt Templates**: Pre-configured system prompts optimized for each agent role
7. **Type Safety**: Enum-based type definitions for capabilities and agent types

## Agent Design Principles

1. **Single Responsibility**: Each agent focuses on one domain
2. **Clear Capabilities**: Explicit capability declarations for matching
3. **Tool Constraints**: Agents only have access to relevant tools
4. **Prompt Engineering**: Specialized prompts for optimal performance
5. **Model Flexibility**: Support for different models per agent

## Extending with Custom Agents

```python
from ai_autonom.agents.agent_registry import Agent, AgentRegistry, AgentType, AgentCapability

# Define custom agent
custom_agent = Agent(
    name="data_scientist",
    type=AgentType.ANALYSIS,
    capabilities=[AgentCapability.ANALYSIS],
    description="Data analysis and visualization expert",
    tools=["pandas", "matplotlib", "scikit-learn"],
    model="llama3:8b",
    prompt_template="You are an expert data scientist..."
)

# Register
registry = AgentRegistry()
registry.register_agent(custom_agent)
```
