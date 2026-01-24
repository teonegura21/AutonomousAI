#!/usr/bin/env python3
"""
Agent Registry - Dynamic Model & Tool Selection
Nemotron orchestrator queries this to assign the right agent for each task
Supports multiple LLM providers (Ollama, OpenAI, Azure, etc.)
"""

import sqlite3
from typing import List, Dict, Optional, Any
import re
from dataclasses import dataclass, field
from pathlib import Path
from ai_autonom.core.config import get_config

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

    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent from the registry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
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
    config = get_config()
    
    # Auto-sync Ollama models and dynamically select role models
    try:
        from ai_autonom.core.model_discovery import ModelDiscovery
        from ai_autonom.core.model_selector import DynamicModelSelector

        discovery = ModelDiscovery()
        auto_register = config.get('ollama_models.auto_register', True)
        auto_benchmark = config.get('ollama_models.auto_benchmark', False)
        sync = discovery.sync_ollama_models(auto_register=auto_register, auto_benchmark=auto_benchmark)
        available = {m for m in set(sync.get("available", [])) if not discovery.is_embedding_model(m)}
        if not available:
            raise RuntimeError("No Ollama models available for dynamic assignment")

        selector = DynamicModelSelector()

        def pick_model(task_type: str, config_key: str, fallback: str) -> str:
            preferred = config.get(config_key)
            if preferred and preferred in available:
                return preferred
            best = selector.select_best_model(task_type, {"max_vram": config.get("execution.vram_limit_gb", 20)})
            if isinstance(best, dict) and best.get("model_name") in available:
                return best["model_name"]
            if available:
                return sorted(available)[0]
            return fallback

        coder_model = pick_model("coding", "agents.coder.model", "qwen2.5-coder:7b")
        linguistic_model = pick_model("documentation", "agents.linguistic.model", coder_model)
        orchestrator_model = pick_model("reasoning", "orchestrator.model", coder_model)

        # Persist in-memory config so downstream uses dynamic choices
        config.set('agents.coder.model', coder_model)
        config.set('agents.linguistic.model', linguistic_model)
        config.set('orchestrator.model', orchestrator_model)
        config.set('providers.ollama.default_model', coder_model)

        # Register per-model dynamic agents
        capabilities = [
            c for c in discovery.get_all_capabilities()
            if not discovery.is_embedding_model(c.get("model_name"))
        ]
        _register_dynamic_model_agents(registry, capabilities, available)
    except Exception as e:
        print(f"[AGENT_REGISTRY] Dynamic model selection failed: {e}")
        coder_model = config.get('agents.coder.model', 'qwen2.5-coder:7b')
        linguistic_model = config.get('agents.linguistic.model', 'dicta-il/DictaLM-3.0-1.7B-Thinking:q8_0')
        orchestrator_model = config.get('orchestrator.model', 'huihui_ai/orchestrator-abliterated:latest')
    
    # ===== REGISTER AGENTS =====
    
    # Orchestrator - Nemotron 8B
    registry.register_agent(AgentDefinition(
        id="orchestrator_main",
        name="Main Orchestrator",
        model_name=orchestrator_model,
        model_size_gb=5.0,
        capabilities=["task_decomposition", "agent_assignment", "planning", "meta_orchestration"],
        tools=["json_output", "agent_registry_query"],
        vram_required=5.0,
        speed_tokens_per_sec=30.0,
        quality_score=90.0,
        description="8.2B parameter orchestrator - ONLY plans and assigns, never executes"
    ))
    
    # === CAI/SECURITY AGENTS PORT ===
    try:
        from ai_autonom.agents.prompt_loader import load_prompt
        
        # 1. Red Team Agent (The "Attacker")
        registry.register_agent(AgentDefinition(
            id="red_team_agent",
            name="Red Team Specialist",
            model_name=coder_model,
            model_size_gb=4.7,
            capabilities=["reconnaissance", "exploitation", "privilege_escalation", "network"],
            tools=["nmap_scan", "shodan_search", "netcat", "generic_linux_command"], # will be expanded by category
            vram_required=5.0,
            speed_tokens_per_sec=35.0,
            quality_score=95.0,
            description="Specialized in network penetration, lateral movement, and privilege escalation using methodology from CAI.",
            system_prompt=load_prompt("cai_red_team_agent")
        ))
        
        # 2. Web Pentester Agent (The "Web Expert")
        registry.register_agent(AgentDefinition(
            id="web_pentester_agent",
            name="Web Security Expert",
            model_name=coder_model,
            model_size_gb=4.7,
            capabilities=["web_hacking", "reconnaissance", "js_analysis"],
            tools=["web_request_framework", "js_surface_mapper", "web_spider", "curl_request"],
            vram_required=5.0,
            speed_tokens_per_sec=35.0,
            quality_score=95.0,
            description="Expert in Web Application Security, API testing, and JS analysis.",
            system_prompt=load_prompt("cai_web_pentester")
        ))
        
        # 3. Reporting Agent (The "Scribe")
        registry.register_agent(AgentDefinition(
            id="reporting_agent",
            name="Security Reporter",
            model_name=linguistic_model,
            model_size_gb=1.7,
            capabilities=["reporting", "summarization", "writing"],
            tools=["record_finding", "filesystem_write", "filesystem_read"],
            vram_required=2.0,
            speed_tokens_per_sec=50.0,
            quality_score=92.0,
            description="Generates professional security reports and organizes findings.",
            system_prompt=load_prompt("cai_reporting_agent")
        ))
        
        # 4. Research Agent (OpenManus-style research and synthesis)
        registry.register_agent(AgentDefinition(
            id="research_agent",
            name="Research Agent",
            model_name=coder_model,
            model_size_gb=4.7,
            capabilities=["research", "information_gathering", "summarization"],
            tools=["web_search", "web_fetch", "filesystem_read", "filesystem_write"],
            vram_required=5.0,
            speed_tokens_per_sec=35.0,
            quality_score=90.0,
            description="Finds, filters, and summarizes relevant information for downstream agents.",
            system_prompt=load_prompt("research_agent")
        ))
        
        # 5. Synthesizer Agent (final aggregation)
        registry.register_agent(AgentDefinition(
            id="synthesizer_agent",
            name="Synthesizer Agent",
            model_name=linguistic_model,
            model_size_gb=1.8,
            capabilities=["synthesis", "summarization", "reporting"],
            tools=["filesystem_read", "filesystem_write"],
            vram_required=2.0,
            speed_tokens_per_sec=50.0,
            quality_score=88.0,
            description="Aggregates artifacts and outputs into concise deliverables.",
            system_prompt=load_prompt("synthesizer_agent")
        ))
        
        print(f"[AGENT_REGISTRY] CAI Security Agents registered successfully.")
        
    except ImportError:
        print("[AGENT_REGISTRY] Could not load prompt_loader, skipping CAI agents.")
    except Exception as e:
        print(f"[AGENT_REGISTRY] Error registering CAI agents: {e}")

    # Technical/Coding Agent
    registry.register_agent(AgentDefinition(
        id="coder_qwen",
        name="Qwen3 Technical Coder",
        model_name=coder_model,
        model_size_gb=5.0,
        capabilities=["code_generation", "debugging", "testing", "python", "refactoring", "technical_tasks"],
        tools=["filesystem", "python_interpreter", "bash"],
        vram_required=5.0,
        speed_tokens_per_sec=70.0,
        quality_score=85.0,
        description=f"Coding agent using {coder_model}"
    ))
    
    # Linguistic/Simple Tasks Agent
    registry.register_agent(AgentDefinition(
        id="linguistic_dictalm",
        name="DictaLM Linguistic Agent",
        model_name=linguistic_model,
        model_size_gb=1.8,
        capabilities=["text_generation", "documentation", "summarization", "formatting", "explanation", "simple_tasks"],
        tools=["filesystem"],
        vram_required=1.8,
        speed_tokens_per_sec=50.0,
        quality_score=80.0,
        description=f"Linguistic agent using {linguistic_model}"
    ))
    
    # Simple Tasks Agent (Fallback to Coder)
    registry.register_agent(AgentDefinition(
        id="simple_qwen",
        name="Qwen Simple Tasks",
        model_name=coder_model,
        model_size_gb=1.0,
        capabilities=["simple_tasks", "quick_responses", "formatting"],
        tools=["filesystem"],
        vram_required=1.0,
        speed_tokens_per_sec=80.0,
        quality_score=70.0,
        description="Simple tasks agent"
    ))

    # Test Runner Agent (alias to coder for test execution)
    registry.register_agent(AgentDefinition(
        id="test_runner",
        name="Test Runner",
        model_name=coder_model,
        model_size_gb=1.0,
        capabilities=["testing", "test_execution", "quality_gates"],
        tools=["filesystem_read", "filesystem_write", "python_exec", "pytest_run", "bash_exec"],
        vram_required=2.0,
        speed_tokens_per_sec=60.0,
        quality_score=80.0,
        description="Runs and validates tests, reports results"
    ))
    
    # DictaLM Coder (Fallback to Linguistic)
    registry.register_agent(AgentDefinition(
        id="coder_dictalm",
        name="DictaLM Coder",
        model_name=linguistic_model,
        model_size_gb=1.8,
        capabilities=["code_documentation", "code_explanation", "readme_generation"],
        tools=["filesystem"],
        vram_required=1.8,
        speed_tokens_per_sec=50.0,
        quality_score=85.0,
        description="Code documentation agent"
    ))
    
    # ===== REGISTER TOOLS =====
    
    # ===== REGISTER CAI SECURITY AGENTS =====
    try:
        from ai_autonom.agents.cai_security_agents import get_cai_security_agents
        
        cai_agents = get_cai_security_agents()
        for agent in cai_agents:
            # Override model with configured one
            agent.model_name = coder_model # Default CAI agents to use the strong coder model
            registry.register_agent(agent)
            
        print(f"[AGENT_REGISTRY] Registered {len(cai_agents)} CAI security agents")
    except ImportError as e:
        print(f"[AGENT_REGISTRY] Could not load CAI agents: {e}")
    
    # ===== REGISTER KALI AGENTS WITH FULL TOOL ACCESS =====
    try:
        from ai_autonom.agents.kali_agents import ALL_AGENTS as KALI_AGENTS
        
        for agent_id, agent_def in KALI_AGENTS.items():
            # Determine appropriate model based on role
            category = agent_def.get("category", "")
            if category == "RAPORT":
                model_to_use = linguistic_model
            else:
                model_to_use = coder_model

            registry.register_agent(AgentDefinition(
                id=f"kali_{agent_id}" if not agent_id.startswith("kali_") else agent_id,
                name=agent_def.get("name", agent_id),
                model_name=model_to_use,
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


def _infer_model_role(cap: Dict[str, Any]) -> str:
    """Infer a general role based on capability scores."""
    scores = {
        "coding": cap.get("coding_score", 0.0),
        "reasoning": cap.get("reasoning_score", 0.0),
        "documentation": cap.get("documentation_score", 0.0),
    }
    # Speed-focused if clearly high
    speed = cap.get("speed_tokens_sec", 0.0)
    if speed >= 80:
        return "fast"
    return max(scores, key=scores.get)


def _sanitize_model_id(model_name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", model_name.lower()).strip("_")


def _register_dynamic_model_agents(
    registry: AgentRegistry,
    capabilities: List[Dict[str, Any]],
    available_models: Optional[set] = None
) -> None:
    """Register one generic agent per available model with an inferred role."""
    available_models = available_models or {c.get("model_name") for c in capabilities}
    allowed_ids = {f"ollama_{_sanitize_model_id(name)}" for name in available_models if name}

    # Clean up stale dynamic agents
    for agent in registry.get_all_agents():
        if agent.id.startswith("ollama_") and agent.id not in allowed_ids:
            registry.delete_agent(agent.id)

    role_capabilities = {
        "coding": ["code_generation", "debugging", "testing", "refactoring", "python", "technical_tasks"],
        "reasoning": ["planning", "analysis", "task_decomposition"],
        "documentation": ["documentation", "summarization", "text_generation", "formatting", "explanation"],
        "fast": ["simple_tasks", "quick_responses", "formatting"]
    }
    role_tools = {
        "coding": ["filesystem_read", "filesystem_write", "python_exec", "bash_exec", "pytest_run"],
        "reasoning": ["filesystem_read"],
        "documentation": ["filesystem_read", "filesystem_write"],
        "fast": ["filesystem_read"]
    }

    for cap in capabilities:
        model_name = cap.get("model_name")
        if not model_name or model_name not in available_models:
            continue

        role = _infer_model_role(cap)
        agent_id = f"ollama_{_sanitize_model_id(model_name)}"
        registry.register_agent(AgentDefinition(
            id=agent_id,
            name=f"Ollama {model_name}",
            model_name=model_name,
            model_size_gb=cap.get("vram_gb", 0.0),
            capabilities=role_capabilities.get(role, []),
            tools=role_tools.get(role, []),
            vram_required=cap.get("vram_gb", 0.0),
            speed_tokens_per_sec=cap.get("speed_tokens_sec", 0.0),
            quality_score=max(
                cap.get("coding_score", 0.0),
                cap.get("reasoning_score", 0.0),
                cap.get("documentation_score", 0.0)
            ),
            description=f"Auto role '{role}' for {model_name}",
            provider="ollama"
        ))


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
