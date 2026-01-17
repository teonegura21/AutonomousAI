#!/usr/bin/env python3
"""
Container Tool Router
Routes tool execution to appropriate Docker containers
Makes all tools available to agents through containerized execution
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Try to import docker
try:
    import docker
    from docker.errors import NotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("[WARNING] Docker SDK not installed. Run: pip install docker")


class ContainerType(Enum):
    """Available container types for tool execution"""
    KALI = "kali"              # Primary Kali Linux security environment
    SANDBOX = "sandbox"        # General Python execution
    SECURITY = "security"      # Security analysis tools
    WEB = "web"                # Web scraping & automation
    NODEJS = "nodejs"          # JavaScript/TypeScript
    HOST = "host"              # Run on host (fallback)


@dataclass
class ContainerConfig:
    """Configuration for a container"""
    name: str
    container_type: ContainerType
    container_name: str
    workdir: str
    tools: List[str]
    mem_limit: str = "2g"
    cpu_limit: float = 2.0
    network: str = "ai_autonom_network"
    read_only: bool = False


@dataclass
class ToolMapping:
    """Maps a tool to its container and execution details"""
    tool_id: str
    container_type: ContainerType
    command_template: str
    requires_file: bool = False
    timeout: int = 30
    description: str = ""


class ContainerToolRouter:
    """
    Routes tool execution to appropriate Docker containers.
    Provides unified interface for agents to execute tools.
    Falls back to local execution when Docker is unavailable.
    """
    
    # Flag to suppress repeated Docker warning messages
    _docker_warning_shown = False
    
    # Container configurations
    CONTAINERS: Dict[ContainerType, ContainerConfig] = {
        ContainerType.KALI: ContainerConfig(
            name="Kali Linux Security",
            container_type=ContainerType.KALI,
            container_name="agent_kali",
            workdir="/workspace",
            tools=["nmap", "metasploit", "sqlmap", "binwalk", "volatility", 
                   "wireshark", "nikto", "hydra", "john", "hashcat",
                   "gobuster", "dirb", "whatweb", "searchsploit"],
            mem_limit="4g",
            cpu_limit=4.0
        ),
        ContainerType.SANDBOX: ContainerConfig(
            name="Python Sandbox",
            container_type=ContainerType.SANDBOX,
            container_name="agent_sandbox",
            workdir="/workspace",
            tools=["python_exec", "bash_exec", "pytest_run", "filesystem_read", 
                   "filesystem_write", "filesystem_search", "pip_install"]
        ),
        ContainerType.SECURITY: ContainerConfig(
            name="Security Sandbox",
            container_type=ContainerType.SECURITY,
            container_name="agent_security",
            workdir="/analysis",
            tools=["bandit_scan", "safety_check", "nmap_scan", "file_analyze",
                   "hash_file", "ssl_check"],
            read_only=True
        ),
        ContainerType.WEB: ContainerConfig(
            name="Web Sandbox",
            container_type=ContainerType.WEB,
            container_name="agent_web",
            workdir="/web",
            tools=["web_fetch", "web_scrape", "selenium_browse", "playwright_browse",
                   "api_test", "download_file"]
        ),
        ContainerType.NODEJS: ContainerConfig(
            name="Node.js Sandbox",
            container_type=ContainerType.NODEJS,
            container_name="agent_nodejs",
            workdir="/workspace",
            tools=["npm_run", "node_exec", "typescript_exec", "eslint_check",
                   "jest_test", "prettier_format"]
        )
    }
    
    # Tool to container mapping
    TOOL_MAPPINGS: Dict[str, ToolMapping] = {
        # ==============================================
        # KALI LINUX TOOLS - Primary security execution
        # ==============================================
        "kali_nmap": ToolMapping(
            tool_id="kali_nmap",
            container_type=ContainerType.KALI,
            command_template="nmap {args} {target}",
            timeout=300,
            description="Network port scanning with nmap"
        ),
        "kali_sqlmap": ToolMapping(
            tool_id="kali_sqlmap",
            container_type=ContainerType.KALI,
            command_template="sqlmap {args}",
            timeout=600,
            description="SQL injection testing with sqlmap"
        ),
        "kali_nikto": ToolMapping(
            tool_id="kali_nikto",
            container_type=ContainerType.KALI,
            command_template="nikto -h {target} {args}",
            timeout=600,
            description="Web server vulnerability scanner"
        ),
        "kali_gobuster": ToolMapping(
            tool_id="kali_gobuster",
            container_type=ContainerType.KALI,
            command_template="gobuster {mode} -u {url} {args}",
            timeout=600,
            description="Directory/DNS brute-forcing"
        ),
        "kali_hydra": ToolMapping(
            tool_id="kali_hydra",
            container_type=ContainerType.KALI,
            command_template="hydra {args}",
            timeout=600,
            description="Network login cracker"
        ),
        "kali_binwalk": ToolMapping(
            tool_id="kali_binwalk",
            container_type=ContainerType.KALI,
            command_template="binwalk {args} {file}",
            timeout=120,
            description="Firmware analysis tool"
        ),
        "kali_searchsploit": ToolMapping(
            tool_id="kali_searchsploit",
            container_type=ContainerType.KALI,
            command_template="searchsploit {query}",
            timeout=60,
            description="Exploit database search"
        ),
        "kali_tshark": ToolMapping(
            tool_id="kali_tshark",
            container_type=ContainerType.KALI,
            command_template="tshark {args}",
            timeout=300,
            description="Network packet analyzer"
        ),
        "kali_exec": ToolMapping(
            tool_id="kali_exec",
            container_type=ContainerType.KALI,
            command_template="{command}",
            timeout=120,
            description="Execute any command in Kali"
        ),
        "kali_python": ToolMapping(
            tool_id="kali_python",
            container_type=ContainerType.KALI,
            command_template="python3 {file}",
            requires_file=True,
            timeout=60,
            description="Execute Python in Kali"
        ),
        
        # Python/General tools -> Sandbox
        "python_exec": ToolMapping(
            tool_id="python_exec",
            container_type=ContainerType.SANDBOX,
            command_template="python {file}",
            requires_file=True,
            timeout=60,
            description="Execute Python code"
        ),
        "bash_exec": ToolMapping(
            tool_id="bash_exec",
            container_type=ContainerType.SANDBOX,
            command_template="bash -c '{command}'",
            timeout=30,
            description="Execute bash command"
        ),
        "pytest_run": ToolMapping(
            tool_id="pytest_run",
            container_type=ContainerType.SANDBOX,
            command_template="pytest {file} -v",
            requires_file=True,
            timeout=120,
            description="Run pytest tests"
        ),
        "pip_install": ToolMapping(
            tool_id="pip_install",
            container_type=ContainerType.SANDBOX,
            command_template="pip install {package}",
            timeout=60,
            description="Install Python package"
        ),
        "filesystem_read": ToolMapping(
            tool_id="filesystem_read",
            container_type=ContainerType.SANDBOX,
            command_template="cat {file}",
            timeout=10,
            description="Read file contents"
        ),
        "filesystem_write": ToolMapping(
            tool_id="filesystem_write",
            container_type=ContainerType.SANDBOX,
            command_template="cat > {file}",
            requires_file=True,
            timeout=10,
            description="Write file contents"
        ),
        "filesystem_search": ToolMapping(
            tool_id="filesystem_search",
            container_type=ContainerType.SANDBOX,
            command_template="find {path} -name '{pattern}'",
            timeout=30,
            description="Search for files"
        ),
        
        # Security tools -> Security container
        "bandit_scan": ToolMapping(
            tool_id="bandit_scan",
            container_type=ContainerType.SECURITY,
            command_template="bandit -r {path} -f json",
            timeout=60,
            description="Run Bandit security scan"
        ),
        "safety_check": ToolMapping(
            tool_id="safety_check",
            container_type=ContainerType.SECURITY,
            command_template="safety check --json",
            timeout=30,
            description="Check for vulnerable dependencies"
        ),
        "nmap_scan": ToolMapping(
            tool_id="nmap_scan",
            container_type=ContainerType.SECURITY,
            command_template="nmap -sV {target}",
            timeout=120,
            description="Network port scan (defensive)"
        ),
        "file_analyze": ToolMapping(
            tool_id="file_analyze",
            container_type=ContainerType.SECURITY,
            command_template="file {file} && xxd {file} | head -50",
            timeout=30,
            description="Analyze file type and content"
        ),
        "hash_file": ToolMapping(
            tool_id="hash_file",
            container_type=ContainerType.SECURITY,
            command_template="sha256sum {file}",
            timeout=10,
            description="Calculate file hash"
        ),
        "ssl_check": ToolMapping(
            tool_id="ssl_check",
            container_type=ContainerType.SECURITY,
            command_template="openssl s_client -connect {host}:{port} -brief",
            timeout=30,
            description="Check SSL/TLS certificate"
        ),
        
        # Web tools -> Web container
        "web_fetch": ToolMapping(
            tool_id="web_fetch",
            container_type=ContainerType.WEB,
            command_template="python -c \"import httpx; print(httpx.get('{url}').text[:5000])\"",
            timeout=30,
            description="Fetch URL content"
        ),
        "web_scrape": ToolMapping(
            tool_id="web_scrape",
            container_type=ContainerType.WEB,
            command_template="python /tools/scraper.py '{url}' '{selector}'",
            timeout=60,
            description="Scrape web page content"
        ),
        "selenium_browse": ToolMapping(
            tool_id="selenium_browse",
            container_type=ContainerType.WEB,
            command_template="python /tools/selenium_runner.py '{url}'",
            timeout=120,
            description="Browse with Selenium"
        ),
        "playwright_browse": ToolMapping(
            tool_id="playwright_browse",
            container_type=ContainerType.WEB,
            command_template="python /tools/playwright_runner.py '{url}'",
            timeout=120,
            description="Browse with Playwright"
        ),
        "download_file": ToolMapping(
            tool_id="download_file",
            container_type=ContainerType.WEB,
            command_template="wget -O {output} '{url}'",
            timeout=120,
            description="Download file from URL"
        ),
        
        # Node.js tools -> Node container
        "npm_run": ToolMapping(
            tool_id="npm_run",
            container_type=ContainerType.NODEJS,
            command_template="npm run {script}",
            timeout=120,
            description="Run npm script"
        ),
        "node_exec": ToolMapping(
            tool_id="node_exec",
            container_type=ContainerType.NODEJS,
            command_template="node {file}",
            requires_file=True,
            timeout=60,
            description="Execute JavaScript file"
        ),
        "typescript_exec": ToolMapping(
            tool_id="typescript_exec",
            container_type=ContainerType.NODEJS,
            command_template="ts-node {file}",
            requires_file=True,
            timeout=60,
            description="Execute TypeScript file"
        ),
        "eslint_check": ToolMapping(
            tool_id="eslint_check",
            container_type=ContainerType.NODEJS,
            command_template="eslint {file} --format json",
            timeout=30,
            description="Run ESLint check"
        ),
        "jest_test": ToolMapping(
            tool_id="jest_test",
            container_type=ContainerType.NODEJS,
            command_template="jest {file} --json",
            requires_file=True,
            timeout=120,
            description="Run Jest tests"
        ),
        "prettier_format": ToolMapping(
            tool_id="prettier_format",
            container_type=ContainerType.NODEJS,
            command_template="prettier --write {file}",
            timeout=30,
            description="Format code with Prettier"
        )
    }
    
    def __init__(self):
        self.client = None
        self._containers: Dict[str, Any] = {}
        self._local_fallback = True  # Always use local fallback when Docker unavailable
        
        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
                # Test connection
                self.client.ping()
                self._discover_containers()
            except Exception as e:
                if not ContainerToolRouter._docker_warning_shown:
                    print(f"[ROUTER] Docker not available: {e}")
                    print("[ROUTER] Using local execution fallback. Start Docker Desktop for containerized execution.")
                    ContainerToolRouter._docker_warning_shown = True
                self.client = None
        else:
            if not ContainerToolRouter._docker_warning_shown:
                print("[ROUTER] Docker SDK not installed. Using local execution.")
                ContainerToolRouter._docker_warning_shown = True
    
    def _discover_containers(self):
        """Discover running AI Autonom containers"""
        if not self.client:
            return
        
        try:
            containers = self.client.containers.list(
                filters={"label": "ai.autonom.role"}
            )
            for container in containers:
                name = container.name
                self._containers[name] = container
                print(f"[ROUTER] Found container: {name}")
        except Exception as e:
            print(f"[ROUTER] Error discovering containers: {e}")
    
    def is_available(self) -> bool:
        """Check if Docker routing is available"""
        return DOCKER_AVAILABLE and self.client is not None
    
    def get_container_for_tool(self, tool_id: str) -> Optional[ContainerType]:
        """Get the container type for a tool"""
        mapping = self.TOOL_MAPPINGS.get(tool_id)
        return mapping.container_type if mapping else None
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tools with their container assignments"""
        tools = {}
        for tool_id, mapping in self.TOOL_MAPPINGS.items():
            tools[tool_id] = {
                "id": tool_id,
                "container": mapping.container_type.value,
                "description": mapping.description,
                "timeout": mapping.timeout,
                "requires_file": mapping.requires_file
            }
        return tools
    
    def get_container_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all containers"""
        status = {}
        
        for container_type, config in self.CONTAINERS.items():
            container_name = config.container_name
            container = self._containers.get(container_name)
            
            if container:
                try:
                    container.reload()
                    status[container_type.value] = {
                        "name": config.name,
                        "container_name": container_name,
                        "status": container.status,
                        "tools": config.tools,
                        "available": container.status == "running"
                    }
                except:
                    status[container_type.value] = {
                        "name": config.name,
                        "status": "error",
                        "available": False
                    }
            else:
                status[container_type.value] = {
                    "name": config.name,
                    "container_name": container_name,
                    "status": "not_found",
                    "tools": config.tools,
                    "available": False
                }
        
        return status
    
    def execute_tool(
        self,
        tool_id: str,
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Execute a tool in its designated container.
        
        Args:
            tool_id: The tool identifier
            params: Parameters for the tool
            timeout: Optional timeout override
        
        Returns:
            Tuple of (success, output_or_error)
        """
        mapping = self.TOOL_MAPPINGS.get(tool_id)
        if not mapping:
            return False, f"Unknown tool: {tool_id}"
        
        # Get container
        config = self.CONTAINERS.get(mapping.container_type)
        if not config:
            return False, f"No container configured for {mapping.container_type}"
        
        container = self._containers.get(config.container_name)
        
        # Build command
        try:
            command = mapping.command_template.format(**params)
        except KeyError as e:
            return False, f"Missing parameter: {e}"
        
        # Execute in container or fallback to local
        exec_timeout = timeout or mapping.timeout
        
        if container and container.status == "running":
            return self._execute_in_container(
                container, command, config.workdir, exec_timeout
            )
        else:
            print(f"[ROUTER] Container {config.container_name} not available, using fallback")
            return self._execute_local(command, exec_timeout)
    
    def _execute_in_container(
        self,
        container,
        command: str,
        workdir: str,
        timeout: int
    ) -> Tuple[bool, str]:
        """Execute command in Docker container"""
        try:
            exit_code, output = container.exec_run(
                f"bash -c '{command}'",
                workdir=workdir,
                demux=True
            )
            
            stdout = output[0].decode('utf-8', errors='replace') if output[0] else ""
            stderr = output[1].decode('utf-8', errors='replace') if output[1] else ""
            
            if exit_code == 0:
                return True, stdout
            else:
                return False, stderr or stdout
                
        except Exception as e:
            return False, f"Container execution error: {str(e)}"
    
    def _execute_local(self, command: str, timeout: int) -> Tuple[bool, str]:
        """Fallback local execution with enhanced safety"""
        import subprocess
        
        # Create outputs directory if it doesn't exist
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Basic safety check for dangerous commands
        dangerous_patterns = [
            r'\brm\s+-rf\s+[/~]',
            r'\bdd\s+if=.*of=/dev/',
            r'\bmkfs\.',
            r'\bshutdown\b',
            r'\breboot\b',
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"[SECURITY] Blocked dangerous command pattern"
        
        try:
            # Use shell on Windows, bash on Unix
            import platform
            if platform.system() == "Windows":
                # Use PowerShell for better compatibility
                result = subprocess.run(
                    ["powershell", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(outputs_dir)
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(outputs_dir)
                )
            
            output = result.stdout
            if result.returncode != 0 and result.stderr:
                output += f"\n[STDERR] {result.stderr}"
            
            return result.returncode == 0, output or "(no output)"
            
        except subprocess.TimeoutExpired:
            return False, f"[TIMEOUT] Command exceeded {timeout}s limit"
        except Exception as e:
            return False, f"[ERROR] {str(e)}"
    
    def write_file_to_container(
        self,
        container_type: ContainerType,
        filename: str,
        content: str
    ) -> Tuple[bool, str]:
        """Write a file to a container's workspace"""
        config = self.CONTAINERS.get(container_type)
        if not config:
            return False, f"Unknown container type: {container_type}"
        
        container = self._containers.get(config.container_name)
        if not container or container.status != "running":
            # Fallback to local
            path = Path("outputs") / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return True, f"Written to {path}"
        
        try:
            # Write file using heredoc
            cmd = f"cat > {config.workdir}/{filename} << 'ENDOFFILE'\n{content}\nENDOFFILE"
            exit_code, output = container.exec_run(
                f"bash -c \"{cmd}\"",
                workdir=config.workdir
            )
            
            if exit_code == 0:
                return True, f"Written to {config.workdir}/{filename}"
            else:
                return False, output.decode('utf-8', errors='replace')
                
        except Exception as e:
            return False, f"Write error: {str(e)}"
    
    def read_file_from_container(
        self,
        container_type: ContainerType,
        filename: str
    ) -> Tuple[bool, str]:
        """Read a file from a container's workspace"""
        config = self.CONTAINERS.get(container_type)
        if not config:
            return False, f"Unknown container type: {container_type}"
        
        container = self._containers.get(config.container_name)
        if not container or container.status != "running":
            # Fallback to local
            path = Path("outputs") / filename
            if path.exists():
                return True, path.read_text(encoding='utf-8')
            return False, f"File not found: {path}"
        
        try:
            exit_code, output = container.exec_run(
                f"cat {config.workdir}/{filename}",
                workdir=config.workdir
            )
            
            if exit_code == 0:
                return True, output.decode('utf-8', errors='replace')
            else:
                return False, f"File not found or error reading"
                
        except Exception as e:
            return False, f"Read error: {str(e)}"
    
    def start_containers(self) -> Dict[str, bool]:
        """Start all AI Autonom containers using docker-compose"""
        import subprocess
        
        results = {}
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        
        try:
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=docker_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("[ROUTER] All containers started")
                self._discover_containers()
                for name in self._containers:
                    results[name] = True
            else:
                print(f"[ROUTER] Failed to start containers: {result.stderr}")
                
        except Exception as e:
            print(f"[ROUTER] Error starting containers: {e}")
        
        return results
    
    def stop_containers(self) -> bool:
        """Stop all AI Autonom containers"""
        import subprocess
        
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        
        try:
            result = subprocess.run(
                ["docker-compose", "down"],
                cwd=docker_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("[ROUTER] All containers stopped")
                self._containers.clear()
                return True
                
        except Exception as e:
            print(f"[ROUTER] Error stopping containers: {e}")
        
        return False


# Global router instance
_router: Optional[ContainerToolRouter] = None


def get_router() -> ContainerToolRouter:
    """Get or create the global router instance"""
    global _router
    if _router is None:
        _router = ContainerToolRouter()
    return _router


if __name__ == "__main__":
    # Test the router
    print("\n" + "="*60)
    print("CONTAINER TOOL ROUTER TEST")
    print("="*60 + "\n")
    
    router = ContainerToolRouter()
    
    print(f"Docker available: {router.is_available()}")
    
    print("\nContainer Status:")
    status = router.get_container_status()
    for name, info in status.items():
        print(f"  {name}: {info.get('status', 'unknown')}")
    
    print("\nAvailable Tools:")
    tools = router.get_available_tools()
    for tool_id, info in tools.items():
        print(f"  {tool_id}: {info['container']} - {info['description']}")
    
    print("\n" + "="*60)
