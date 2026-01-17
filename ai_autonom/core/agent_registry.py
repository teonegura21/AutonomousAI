#!/usr/bin/env python3
"""
Agent Registry - Dynamic Model & Tool Selection
Nemotron orchestrator queries this to assign the right agent for each task
Supports multiple LLM providers (Ollama, OpenAI, Azure, etc.)
"""

import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AgentDefinition:
    """Definition of an available agent"""
    id: str
    name: str
    model_name: str
    model_size_gb: float
    capabilities: List[str]
    tools: List[str]
    vram_required: float
    speed_tokens_per_sec: float
    quality_score: float  # 0-100
    description: str
    provider: str = "ollama"  # ollama, openai, azure_openai, openai_compatible
    api_base: Optional[str] = None  # Custom API endpoint
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None  # Specialized system prompt

@dataclass
class ToolDefinition:
    """Definition of an available tool"""
    id: str
    name: str
    category: str  # 'filesystem', 'web', 'code_execution', etc.
    description: str
    requires_sandbox: bool

class AgentRegistry:
    """
    Central registry of available agents and tools
    Nemotron queries this to make intelligent assignments
    """
    
    def __init__(self, db_path: str = ".runtime/data/agent_registry.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with agents and tools"""
        # Ensure directory exists for database file
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create agents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_size_gb REAL,
                capabilities TEXT,  -- JSON array
                tools TEXT,         -- JSON array
                vram_required REAL,
                speed_tokens_per_sec REAL,
                quality_score REAL,
                description TEXT,
                provider TEXT DEFAULT 'ollama',
                api_base TEXT,
                temperature REAL DEFAULT 0.7,
                max_tokens INTEGER DEFAULT 4096,
                system_prompt TEXT
            )
        """)
        
        # Migration: Add new columns if they don't exist (for existing databases)
        try:
            cursor.execute("SELECT provider FROM agents LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE agents ADD COLUMN provider TEXT DEFAULT 'ollama'")
            cursor.execute("ALTER TABLE agents ADD COLUMN api_base TEXT")
            cursor.execute("ALTER TABLE agents ADD COLUMN temperature REAL DEFAULT 0.7")
            cursor.execute("ALTER TABLE agents ADD COLUMN max_tokens INTEGER DEFAULT 4096")
            print("[AGENT_REGISTRY] Migrated database schema with provider columns")
            
        try:
            cursor.execute("SELECT system_prompt FROM agents LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE agents ADD COLUMN system_prompt TEXT")
            print("[AGENT_REGISTRY] Migrated database schema with system_prompt column")
        
        # Create tools table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                requires_sandbox BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_agent(self, agent: AgentDefinition):
        """Register a new agent in the registry"""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agents 
            (id, name, model_name, model_size_gb, capabilities, tools, 
             vram_required, speed_tokens_per_sec, quality_score, description,
             provider, api_base, temperature, max_tokens, system_prompt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent.id,
            agent.name,
            agent.model_name,
            agent.model_size_gb,
            json.dumps(agent.capabilities),
            json.dumps(agent.tools),
            agent.vram_required,
            agent.speed_tokens_per_sec,
            agent.quality_score,
            agent.description,
            agent.provider,
            agent.api_base,
            agent.temperature,
            agent.max_tokens,
            agent.system_prompt
        ))
        
        conn.commit()
        conn.close()
    
    def register_tool(self, tool: ToolDefinition):
        """Register a new tool in the registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO tools 
            (id, name, category, description, requires_sandbox)
            VALUES (?, ?, ?, ?, ?)
        """, (
            tool.id,
            tool.name,
            tool.category,
            tool.description,
            tool.requires_sandbox
        ))
        
        conn.commit()
        conn.close()
    
    def find_agent_for_task(self, task_description: str, 
                           required_capabilities: Optional[List[str]] = None,
                           max_vram: float = 8.0) -> Optional[AgentDefinition]:
        """
        Find the best agent for a given task
        This is queried by Nemotron orchestrator
        """
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all agents within VRAM budget
        cursor.execute("""
            SELECT * FROM agents 
            WHERE vram_required <= ?
            ORDER BY quality_score DESC
        """, (max_vram,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        # Parse and return best match
        for row in rows:
            agent = AgentDefinition(
                id=row[0],
                name=row[1],
                model_name=row[2],
                model_size_gb=row[3],
                capabilities=json.loads(row[4]),
                tools=json.loads(row[5]),
                vram_required=row[6],
                speed_tokens_per_sec=row[7],
                quality_score=row[8],
                description=row[9],
                provider=row[10] if len(row) > 10 else "ollama",
                api_base=row[11] if len(row) > 11 else None,
                temperature=row[12] if len(row) > 12 else 0.7,
                max_tokens=row[13] if len(row) > 13 else 4096,
                system_prompt=row[14] if len(row) > 14 else None
            )
            
            # Check if agent has required capabilities
            if required_capabilities:
                if all(cap in agent.capabilities for cap in required_capabilities):
                    return agent
            else:
                return agent  # Return highest quality
        
        return None
    
    def get_all_agents(self) -> List[AgentDefinition]:
        """Get all registered agents"""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agents")
        rows = cursor.fetchall()
        conn.close()
        
        agents = []
        for row in rows:
            agents.append(AgentDefinition(
                id=row[0],
                name=row[1],
                model_name=row[2],
                model_size_gb=row[3],
                capabilities=json.loads(row[4]),
                tools=json.loads(row[5]),
                vram_required=row[6],
                speed_tokens_per_sec=row[7],
                quality_score=row[8],
                description=row[9],
                provider=row[10] if len(row) > 10 else "ollama",
                api_base=row[11] if len(row) > 11 else None,
                temperature=row[12] if len(row) > 12 else 0.7,
                max_tokens=row[13] if len(row) > 13 else 4096,
                system_prompt=row[14] if len(row) > 14 else None
            ))
        
        return agents
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a specific category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM tools WHERE category = ?", (category,))
        rows = cursor.fetchall()
        conn.close()
        
        tools = []
        for row in rows:
            tools.append(ToolDefinition(
                id=row[0],
                name=row[1],
                category=row[2],
                description=row[3],
                requires_sandbox=bool(row[4])
            ))
        
        return tools


def setup_initial_registry():
    """Setup registry with current agents and tools"""
    registry = AgentRegistry()
    
    # ===== REGISTER AGENTS =====
    
    # Orchestrator - Nemotron 8B
    registry.register_agent(AgentDefinition(
        id="orchestrator_main",
        name="Main Orchestrator",
        model_name="huihui_ai/orchestrator-abliterated",
        model_size_gb=5.0,
        capabilities=["task_decomposition", "agent_assignment", "planning", "meta_orchestration"],
        tools=["json_output", "agent_registry_query"],
        vram_required=5.0,
        speed_tokens_per_sec=30.0,
        quality_score=90.0,
        description="8.2B parameter orchestrator - ONLY plans and assigns, never executes"
    ))
    
    # Technical/Coding Agent - Qwen3 1.7B
    registry.register_agent(AgentDefinition(
        id="coder_qwen",
        name="Qwen3 Technical Coder",
        model_name="qwen3:1.7b",
        model_size_gb=1.4,
        capabilities=["code_generation", "debugging", "testing", "python", "refactoring", "technical_tasks"],
        tools=["filesystem", "python_interpreter", "bash"],
        vram_required=1.4,
        speed_tokens_per_sec=70.0,
        quality_score=85.0,
        description="1.7B general purpose model optimized for coding and technical tasks"
    ))
    
    # Linguistic/Simple Tasks Agent - DictaLM 1.7B
    registry.register_agent(AgentDefinition(
        id="linguistic_dictalm",
        name="DictaLM Linguistic Agent",
        model_name="dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0",
        model_size_gb=1.8,
        capabilities=["text_generation", "documentation", "summarization", "formatting", "explanation", "simple_tasks"],
        tools=["filesystem"],
        vram_required=1.8,
        speed_tokens_per_sec=50.0,
        quality_score=80.0,
        description="1.7B thinking model for linguistic tasks, documentation, and explanations"
    ))
    
    # Simple Tasks Agent - Qwen 1.5B (fast)
    registry.register_agent(AgentDefinition(
        id="simple_qwen",
        name="Qwen Simple Tasks",
        model_name="qwen2.5:1.5b",
        model_size_gb=1.0,
        capabilities=["simple_tasks", "quick_responses", "formatting"],
        tools=["filesystem"],
        vram_required=1.0,
        speed_tokens_per_sec=80.0,
        quality_score=70.0,
        description="1.5B model for simple, fast tasks"
    ))
    
    # DictaLM Coder - For code documentation
    registry.register_agent(AgentDefinition(
        id="coder_dictalm",
        name="DictaLM Coder",
        model_name="dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0",
        model_size_gb=1.8,
        capabilities=["code_documentation", "code_explanation", "readme_generation"],
        tools=["filesystem"],
        vram_required=1.8,
        speed_tokens_per_sec=50.0,
        quality_score=85.0,
        description="1.7B model for code documentation and explanation"
    ))
    
    # ===== REGISTER TOOLS =====
    
    # ===== REGISTER CAI SECURITY AGENTS =====
    try:
        from ai_autonom.agents.cai_security_agents import get_cai_security_agents
        
        cai_agents = get_cai_security_agents()
        for agent in cai_agents:
            registry.register_agent(agent)
            
        print(f"[AGENT_REGISTRY] Registered {len(cai_agents)} CAI security agents")
    except ImportError as e:
        print(f"[AGENT_REGISTRY] Could not load CAI agents: {e}")
    
    # ===== REGISTER KALI AGENTS WITH FULL TOOL ACCESS =====
    try:
        from ai_autonom.agents.kali_agents import ALL_AGENTS as KALI_AGENTS
        
        for agent_id, agent_def in KALI_AGENTS.items():
            registry.register_agent(AgentDefinition(
                id=f"kali_{agent_id}" if not agent_id.startswith("kali_") else agent_id,
                name=agent_def.get("name", agent_id),
                model_name=agent_def.get("model", "qwen3:1.7b"),
                model_size_gb=agent_def.get("vram_required", 1.8),
                capabilities=agent_def.get("capabilities", []) + ["kali_linux", "container_execution"],
                tools=agent_def.get("tools", []) + agent_def.get("kali_tools", []),
                vram_required=agent_def.get("vram_required", 1.8),
                speed_tokens_per_sec=50.0,
                quality_score=agent_def.get("quality_score", 85.0),
                description=f"[KALI] {agent_def.get('description', '')[:180]}"
            ))
        print(f"[AGENT_REGISTRY] Registered {len(KALI_AGENTS)} Kali Linux agents")
    except ImportError as e:
        print(f"[AGENT_REGISTRY] Could not load Kali agents: {e}")
    
    # Filesystem Tools
    registry.register_tool(ToolDefinition(
        id="filesystem_read",
        name="Filesystem Read",
        category="filesystem",
        description="Read files from disk",
        requires_sandbox=False
    ))
    
    registry.register_tool(ToolDefinition(
        id="filesystem_write",
        name="Filesystem Write",
        category="filesystem",
        description="Write files to disk",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="filesystem_search",
        name="Filesystem Search",
        category="filesystem",
        description="Search for files/patterns in filesystem",
        requires_sandbox=False
    ))
    
    # Code Execution Tools
    registry.register_tool(ToolDefinition(
        id="bash_exec",
        name="Bash Execution",
        category="code_execution",
        description="Execute bash commands in sandbox",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="python_interpreter",
        name="Python Interpreter",
        category="code_execution",
        description="Execute Python code in sandbox",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="python_repl",
        name="Python REPL",
        category="code_execution",
        description="Interactive Python shell",
        requires_sandbox=True
    ))
    
    # Web Tools
    registry.register_tool(ToolDefinition(
        id="web_search",
        name="Web Search (Tavily)",
        category="web",
        description="Search the web for information",
        requires_sandbox=False
    ))
    
    registry.register_tool(ToolDefinition(
        id="web_fetch",
        name="Web Fetch",
        category="web",
        description="Fetch content from URLs",
        requires_sandbox=False
    ))
    
    registry.register_tool(ToolDefinition(
        id="web_scrape",
        name="Web Scraping",
        category="web",
        description="Extract data from web pages",
        requires_sandbox=False
    ))
    
    # Development Tools
    registry.register_tool(ToolDefinition(
        id="git_operations",
        name="Git Operations",
        category="development",
        description="Git commands (clone, commit, push, pull)",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="compiler",
        name="Compiler",
        category="development",
        description="Compile code (C++, Go, Rust)",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="debugger",
        name="Debugger",
        category="development",
        description="Debug compiled code",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="docker",
        name="Docker",
        category="development",
        description="Docker container management",
        requires_sandbox=True
    ))
    
    # Database Tools
    registry.register_tool(ToolDefinition(
        id="sqlite",
        name="SQLite",
        category="database",
        description="SQLite database operations",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="chromadb",
        name="ChromaDB",
        category="database",
        description="Vector database for semantic search",
        requires_sandbox=False
    ))
    
    # Security/Cybersecurity Tools
    registry.register_tool(ToolDefinition(
        id="disassembler",
        name="Disassembler",
        category="security",
        description="Disassemble binary files",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="network_scan",
        name="Network Scanner",
        category="security",
        description="Scan networks (nmap-like)",
        requires_sandbox=True
    ))
    
    registry.register_tool(ToolDefinition(
        id="vulnerability_scan",
        name="Vulnerability Scanner",
        category="security",
        description="Scan for vulnerabilities",
        requires_sandbox=True
    ))
    
    # Analysis Tools
    registry.register_tool(ToolDefinition(
        id="static_analysis",
        name="Static Code Analysis",
        category="analysis",
        description="Analyze code without executing",
        requires_sandbox=False
    ))
    
    registry.register_tool(ToolDefinition(
        id="dynamic_analysis",
        name="Dynamic Analysis",
        category="analysis",
        description="Analyze code during execution",
        requires_sandbox=True
    ))
    
    # JSON/Data Tools
    registry.register_tool(ToolDefinition(
        id="json_parser",
        name="JSON Parser",
        category="data",
        description="Parse and manipulate JSON data",
        requires_sandbox=False
    ))
    
    return registry


if __name__ == "__main__":
    # Setup and test
    registry = setup_initial_registry()
    
    print("="*70)
    print("AGENT REGISTRY - INITIALIZED")
    print("="*70 + "\n")
    
    agents = registry.get_all_agents()
    print(f"Registered Agents: {len(agents)}\n")
    
    for agent in agents:
        print(f"• {agent.name}")
        print(f"  Model: {agent.model_name}")
        print(f"  VRAM: {agent.vram_required}GB")
        print(f"  Capabilities: {', '.join(agent.capabilities)}")
        print(f"  Quality: {agent.quality_score}/100")
        print()
    
    # Test agent selection
    print("="*70)
    print("TESTING AGENT SELECTION")
    print("="*70 + "\n")
    
    coding_agent = registry.find_agent_for_task(
        "Write Python code",
        required_capabilities=["code_generation"],
        max_vram=3.0
    )
    
    if coding_agent:
        print(f"✓ Selected: {coding_agent.name}")
        print(f"  Model: {coding_agent.model_name}")
        print(f"  VRAM: {coding_agent.vram_required}GB")
