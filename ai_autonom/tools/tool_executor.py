#!/usr/bin/env python3
"""
Tool Executor
Central execution engine for all tools - integrates registry, built-in tools, and Docker containers
Provides containerized execution for security and isolation
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import traceback

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .tool_registry import ToolRegistry, ToolDefinition
from .builtin_tools import BuiltinTools, create_builtin_tools

# Import CAI Security Tools
try:
    from .cai_security_tools import (
        CAI_SECURITY_TOOLS, get_tools_for_agent, get_tool,
        generic_linux_command, execute_code, nmap_scan, curl_request,
        wget_download, netcat, netstat_info, filesystem_read, filesystem_write,
        filesystem_search, hash_data, hash_file, base64_encode, base64_decode,
        think, thought_analysis, write_key_findings, read_key_findings,
        file_identify, strings_extract, hexdump, binwalk_analyze, readelf_info,
        objdump_disasm, volatility_analyze, pcap_analyze, log_analyze,
        timeline_create, http_request, google_search, shodan_search, shodan_host_info
    )
    from .compiler_tools import compile_cpp
    CAI_TOOLS_AVAILABLE = True
except ImportError as e:
    CAI_TOOLS_AVAILABLE = False
    print(f"[TOOL_EXECUTOR] CAI Security Tools not available: {e}")

# Import container router
try:
    from sandbox.container_router import ContainerToolRouter, ContainerType, get_router
    CONTAINER_ROUTER_AVAILABLE = True
except ImportError:
    CONTAINER_ROUTER_AVAILABLE = False
    print("[TOOL_EXECUTOR] Container router not available")

# Import Kali agent executor
try:
    from sandbox.kali_agent_executor import KaliAgentExecutor, get_kali_executor
    KALI_EXECUTOR_AVAILABLE = True
except ImportError:
    KALI_EXECUTOR_AVAILABLE = False
    print("[TOOL_EXECUTOR] Kali agent executor not available")


class ToolExecutor:
    """
    Central tool execution engine with Docker container support.
    Orchestrator and agents use this to execute tools.
    Tools can be executed locally or in isolated Docker containers.
    """
    
    def __init__(
        self,
        sandbox=None,
        workspace_dir: str = "outputs",
        enable_logging: bool = True,
        use_containers: bool = True
    ):
        self.registry = ToolRegistry()
        self.builtin_tools = BuiltinTools(sandbox, workspace_dir)
        self.sandbox = sandbox
        self.workspace_dir = workspace_dir
        self.enable_logging = enable_logging
        self.use_containers = use_containers
        
        # Initialize container router
        self.container_router: Optional[ContainerToolRouter] = None
        if use_containers and CONTAINER_ROUTER_AVAILABLE:
            self.container_router = get_router()
            if self.container_router.is_available():
                print("[TOOL_EXECUTOR] Container router initialized")
            else:
                print("[TOOL_EXECUTOR] Container router not available, using local execution")
        
        # Initialize Kali agent executor (automatic routing for security agents)
        self.kali_executor: Optional[KaliAgentExecutor] = None
        if use_containers and KALI_EXECUTOR_AVAILABLE:
            self.kali_executor = get_kali_executor()
            print("[TOOL_EXECUTOR] Kali agent executor initialized - security agents will use Kali container")
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        
        # Register all built-in tools
        self._register_builtin_tools()
        
        self.registry.register(ToolDefinition(
            id="compile_cpp",
            name="Compile C++",
            description="Compile C++ code to portable executable (supports Windows .exe)",
            category="development",
            function=compile_cpp,
            requires_sandbox=True,
            parameters={"source_file": "Path to .cpp", "output_name": "Output name (no ext)", "target_os": "windows/linux"},
            returns="Compilation output"
        ))
        
        # Register CAI security tools
        if CAI_TOOLS_AVAILABLE:
            self._register_cai_tools()
        
        # Register container-based tools
        if self.container_router:
            self._register_container_tools()
    
    def _register_cai_tools(self):
        """Register CAI security tools adapted from CAI Framework"""
        
        # Register all CAI tools from the security tools module
        for tool_id, tool_info in CAI_SECURITY_TOOLS.items():
            try:
                self.registry.register(ToolDefinition(
                    id=f"cai_{tool_id}",  # Prefix with cai_ to avoid conflicts
                    name=tool_info.get("description", tool_id),
                    description=tool_info.get("description", ""),
                    category=f"cai_{tool_info.get('category', 'misc')}",
                    function=tool_info["function"],
                    requires_sandbox=tool_info.get("category") in ["reconnaissance", "binary_analysis", "dfir"],
                    parameters=tool_info.get("parameters", {}),
                    returns="Tool output"
                ))
            except Exception as e:
                print(f"[TOOL_EXECUTOR] Failed to register CAI tool {tool_id}: {e}")
        
        print(f"[TOOL_EXECUTOR] Registered {len(CAI_SECURITY_TOOLS)} CAI security tools")
    
    def _register_builtin_tools(self):
        """Register all built-in tools in the registry"""
        
        # Filesystem tools
        self.registry.register(ToolDefinition(
            id="filesystem_read",
            name="Read File",
            description="Read contents of a file",
            category="filesystem",
            function=self.builtin_tools.filesystem_read,
            requires_sandbox=False,
            parameters={"path": "Path to file (relative or absolute)"},
            returns="File contents as string"
        ))
        
        self.registry.register(ToolDefinition(
            id="filesystem_write",
            name="Write File",
            description="Write content to a file",
            category="filesystem",
            function=self.builtin_tools.filesystem_write,
            requires_sandbox=True,
            parameters={"path": "Path to file", "content": "Content to write"},
            returns="Success message"
        ))
        
        self.registry.register(ToolDefinition(
            id="filesystem_append",
            name="Append to File",
            description="Append content to existing file",
            category="filesystem",
            function=self.builtin_tools.filesystem_append,
            requires_sandbox=True,
            parameters={"path": "Path to file", "content": "Content to append"},
            returns="Success message"
        ))
        
        self.registry.register(ToolDefinition(
            id="filesystem_search",
            name="Search Files",
            description="Search for files matching a pattern",
            category="filesystem",
            function=self.builtin_tools.filesystem_search,
            requires_sandbox=False,
            parameters={"directory": "Directory to search", "pattern": "Glob pattern (e.g., *.py)"},
            returns="List of matching file paths"
        ))
        
        self.registry.register(ToolDefinition(
            id="filesystem_list",
            name="List Directory",
            description="List files in a directory",
            category="filesystem",
            function=self.builtin_tools.filesystem_list,
            requires_sandbox=False,
            parameters={"directory": "Directory path (default: current)"},
            returns="Directory listing"
        ))
        
        self.registry.register(ToolDefinition(
            id="filesystem_delete",
            name="Delete File",
            description="Delete a file",
            category="filesystem",
            function=self.builtin_tools.filesystem_delete,
            requires_sandbox=True,
            parameters={"path": "Path to file to delete"},
            returns="Success message"
        ))
        
        # Code execution tools
        self.registry.register(ToolDefinition(
            id="python_exec",
            name="Execute Python",
            description="Execute Python code",
            category="code_execution",
            function=self.builtin_tools.python_exec,
            requires_sandbox=True,
            parameters={
                "code": "Python code to execute",
                "filename": "Optional filename (default: script.py)",
                "timeout": "Execution timeout in seconds (default: 30)"
            },
            returns="Execution output"
        ))
        
        self.registry.register(ToolDefinition(
            id="bash_exec",
            name="Execute Bash",
            description="Execute bash/shell command",
            category="code_execution",
            function=self.builtin_tools.bash_exec,
            requires_sandbox=True,
            parameters={
                "command": "Shell command to execute",
                "timeout": "Execution timeout in seconds (default: 30)"
            },
            returns="Command output"
        ))
        
        self.registry.register(ToolDefinition(
            id="pytest_run",
            name="Run Pytest",
            description="Run pytest on tests",
            category="code_execution",
            function=self.builtin_tools.pytest_run,
            requires_sandbox=True,
            parameters={
                "test_path": "Path to test file or directory",
                "timeout": "Test timeout in seconds (default: 60)"
            },
            returns="Test results"
        ))
        
        # Web tools
        self.registry.register(ToolDefinition(
            id="web_fetch",
            name="Fetch URL",
            description="Fetch content from a URL",
            category="web",
            function=self.builtin_tools.web_fetch,
            requires_sandbox=False,
            parameters={
                "url": "URL to fetch",
                "timeout": "Request timeout in seconds (default: 10)"
            },
            returns="Page content"
        ))
        
        self.registry.register(ToolDefinition(
            id="web_search",
            name="Web Search",
            description="Search the web",
            category="web",
            function=self.builtin_tools.web_search,
            requires_sandbox=False,
            parameters={"query": "Search query"},
            returns="Search results"
        ))
        
        # JSON tools
        self.registry.register(ToolDefinition(
            id="json_parse",
            name="Parse JSON",
            description="Parse JSON string to object",
            category="data",
            function=self.builtin_tools.json_parse,
            requires_sandbox=False,
            parameters={"content": "JSON string to parse"},
            returns="Parsed data"
        ))
        
        self.registry.register(ToolDefinition(
            id="json_format",
            name="Format JSON",
            description="Format data as JSON string",
            category="data",
            function=self.builtin_tools.json_format,
            requires_sandbox=False,
            parameters={"data": "Data to format"},
            returns="Formatted JSON string"
        ))
    
    def _register_container_tools(self):
        """Register tools that execute in Docker containers"""
        if not self.container_router:
            return
        
        # Security Analysis Tools (run in security container)
        self.registry.register(ToolDefinition(
            id="bandit_scan",
            name="Bandit Security Scan",
            description="Run Bandit security analysis on Python code",
            category="security",
            function=lambda path: self._execute_container_tool("bandit_scan", {"path": path}),
            requires_sandbox=True,
            parameters={"path": "Path to Python file or directory to scan"},
            returns="Security scan results in JSON format"
        ))
        
        self.registry.register(ToolDefinition(
            id="safety_check",
            name="Safety Dependency Check",
            description="Check for known security vulnerabilities in dependencies",
            category="security",
            function=lambda: self._execute_container_tool("safety_check", {}),
            requires_sandbox=True,
            parameters={},
            returns="Vulnerability report"
        ))
        
        self.registry.register(ToolDefinition(
            id="hash_file",
            name="Hash File",
            description="Calculate SHA256 hash of a file",
            category="security",
            function=lambda file: self._execute_container_tool("hash_file", {"file": file}),
            requires_sandbox=False,
            parameters={"file": "Path to file to hash"},
            returns="SHA256 hash"
        ))
        
        self.registry.register(ToolDefinition(
            id="ssl_check",
            name="SSL Certificate Check",
            description="Check SSL/TLS certificate of a host",
            category="security",
            function=lambda host, port="443": self._execute_container_tool("ssl_check", {"host": host, "port": port}),
            requires_sandbox=False,
            parameters={"host": "Hostname to check", "port": "Port (default: 443)"},
            returns="SSL certificate information"
        ))
        
        # Web Automation Tools (run in web container)
        self.registry.register(ToolDefinition(
            id="selenium_browse",
            name="Selenium Browser",
            description="Browse a URL with Selenium (headless Chrome)",
            category="web",
            function=lambda url: self._execute_container_tool("selenium_browse", {"url": url}),
            requires_sandbox=True,
            parameters={"url": "URL to browse"},
            returns="Page content and screenshot path"
        ))
        
        self.registry.register(ToolDefinition(
            id="playwright_browse",
            name="Playwright Browser",
            description="Browse a URL with Playwright",
            category="web",
            function=lambda url: self._execute_container_tool("playwright_browse", {"url": url}),
            requires_sandbox=True,
            parameters={"url": "URL to browse"},
            returns="Page content"
        ))
        
        self.registry.register(ToolDefinition(
            id="download_file",
            name="Download File",
            description="Download a file from URL",
            category="web",
            function=lambda url, output: self._execute_container_tool("download_file", {"url": url, "output": output}),
            requires_sandbox=True,
            parameters={"url": "URL to download from", "output": "Output filename"},
            returns="Download status and file path"
        ))
        
        # Node.js Tools (run in nodejs container)
        self.registry.register(ToolDefinition(
            id="npm_run",
            name="NPM Run Script",
            description="Run an npm script",
            category="nodejs",
            function=lambda script: self._execute_container_tool("npm_run", {"script": script}),
            requires_sandbox=True,
            parameters={"script": "NPM script name to run"},
            returns="Script output"
        ))
        
        self.registry.register(ToolDefinition(
            id="node_exec",
            name="Execute Node.js",
            description="Execute a JavaScript file with Node.js",
            category="nodejs",
            function=lambda file: self._execute_container_tool("node_exec", {"file": file}),
            requires_sandbox=True,
            parameters={"file": "JavaScript file to execute"},
            returns="Execution output"
        ))
        
        self.registry.register(ToolDefinition(
            id="typescript_exec",
            name="Execute TypeScript",
            description="Execute a TypeScript file with ts-node",
            category="nodejs",
            function=lambda file: self._execute_container_tool("typescript_exec", {"file": file}),
            requires_sandbox=True,
            parameters={"file": "TypeScript file to execute"},
            returns="Execution output"
        ))
        
        self.registry.register(ToolDefinition(
            id="eslint_check",
            name="ESLint Check",
            description="Run ESLint on JavaScript/TypeScript file",
            category="nodejs",
            function=lambda file: self._execute_container_tool("eslint_check", {"file": file}),
            requires_sandbox=False,
            parameters={"file": "File to lint"},
            returns="Lint results in JSON format"
        ))
        
        self.registry.register(ToolDefinition(
            id="jest_test",
            name="Jest Test",
            description="Run Jest tests",
            category="nodejs",
            function=lambda file: self._execute_container_tool("jest_test", {"file": file}),
            requires_sandbox=True,
            parameters={"file": "Test file to run"},
            returns="Test results"
        ))
        
        print(f"[TOOL_EXECUTOR] Registered container tools")
    
    def _execute_container_tool(self, tool_id: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute a tool through the container router"""
        if not self.container_router:
            return False, "Container router not available"
        
        return self.container_router.execute_tool(tool_id, params)
    
    def execute(
        self,
        tool_id: str,
        params: Dict[str, Any],
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Tuple[bool, Any]:
        """
        Execute a tool - automatically routes to Kali container for security agents.
        
        Args:
            tool_id: ID of the tool to execute
            params: Parameters for the tool
            agent_id: ID of the agent executing (for logging and routing)
            task_id: ID of the current task (for logging)
        
        Returns:
            Tuple of (success, result_or_error)
        """
        # === AUTOMATIC KALI ROUTING FOR SECURITY AGENTS ===
        if self.kali_executor and agent_id:
            # Check if this agent/tool should run in Kali
            if self.kali_executor.should_use_kali(agent_id, tool_id):
                print(f"[TOOL_EXECUTOR] Routing {agent_id}/{tool_id} to Kali container")
                
                # Build command from tool and params
                command = self._build_kali_command(tool_id, params)
                if command:
                    return self.kali_executor.execute_in_kali(
                        command=command,
                        agent_id=agent_id,
                        timeout=params.get('timeout', 300)
                    )
        
        # === STANDARD TOOL EXECUTION ===
        tool = self.registry.get_tool(tool_id)
        if not tool:
            return False, f"Tool not found: {tool_id}"
        
        start_time = datetime.now()
        success = False
        result = None
        error = None
        
        try:
            # Execute the tool
            result = tool.function(**params)
            
            # Handle tuple returns (success, result)
            if isinstance(result, tuple) and len(result) == 2:
                success, result = result
            else:
                success = True
            
        except Exception as e:
            success = False
            error = str(e)
            result = f"Error: {error}\n{traceback.format_exc()}"
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log execution
        if self.enable_logging:
            self.execution_history.append({
                "tool_id": tool_id,
                "agent_id": agent_id,
                "task_id": task_id,
                "params": {k: str(v)[:100] for k, v in params.items()},
                "success": success,
                "duration_sec": duration,
                "timestamp": start_time.isoformat(),
                "error": error
            })
        
        if self.enable_logging:
            status = "OK" if success else "FAIL"
            print(f"[TOOL] {tool_id} [{status}] ({duration:.2f}s)")
        
        return success, result
    
    def _build_kali_command(self, tool_id: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Build a command to execute in Kali from tool_id and params.
        Gives agents maximum freedom - maps ALL tools to actual commands.
        
        Args:
            tool_id: Tool identifier
            params: Tool parameters
            
        Returns:
            Command string or None
        """
        # === DIRECT COMMAND EXECUTION ===
        if 'command' in params:
            return params['command']
        
        # === PYTHON CODE EXECUTION ===
        if tool_id in ['python_exec', 'cai_execute_code', 'execute_code'] and 'code' in params:
            code = params['code']
            filename = params.get('filename', 'agent_script.py')
            # Write code to file and execute
            return f"cat > /workspace/{filename} << 'EOFCODE'\n{code}\nEOFCODE\npython3 /workspace/{filename}"
        
        # === RECONNAISSANCE & NETWORK TOOLS ===
        nmap_patterns = {
            'cai_nmap_scan': 'nmap {args} {target}',
            'nmap_scan': 'nmap {args} {target}',
        }
        if tool_id in nmap_patterns:
            target = params.get('target', '')
            args = params.get('args', '-sV -sC')
            return f"nmap {args} {target}"
        
        if tool_id in ['cai_masscan', 'masscan']:
            target = params.get('target', '')
            ports = params.get('ports', '1-65535')
            rate = params.get('rate', '1000')
            return f"masscan {target} -p{ports} --rate={rate}"
        
        # === WEB TOOLS ===
        if tool_id in ['cai_curl_request', 'curl_request']:
            url = params.get('url', '')
            args = params.get('args', '-I')
            return f"curl {args} {url}"
        
        if tool_id in ['cai_wget_download', 'wget_download']:
            url = params.get('url', '')
            args = params.get('args', '')
            output = params.get('output', '')
            cmd = f"wget {args} {url}"
            if output:
                cmd += f" -O {output}"
            return cmd
        
        if tool_id in ['cai_nikto', 'nikto']:
            target = params.get('target', '')
            args = params.get('args', '')
            return f"nikto -h {target} {args}"
        
        if tool_id in ['cai_gobuster', 'gobuster']:
            url = params.get('url', '')
            wordlist = params.get('wordlist', '/usr/share/wordlists/dirb/common.txt')
            mode = params.get('mode', 'dir')
            return f"gobuster {mode} -u {url} -w {wordlist}"
        
        if tool_id in ['cai_sqlmap', 'sqlmap']:
            target = params.get('target', '')
            args = params.get('args', '--batch')
            return f"sqlmap -u '{target}' {args}"
        
        # === NETWORK UTILITIES ===
        if tool_id in ['cai_netcat', 'netcat', 'nc']:
            host = params.get('host', '')
            port = params.get('port', '')
            args = params.get('args', '')
            if host and port:
                return f"nc {args} {host} {port}"
            return f"nc {args}"
        
        if tool_id in ['cai_netstat_info', 'netstat']:
            args = params.get('args', '-tuln')
            return f"netstat {args}"
        
        # === BINARY ANALYSIS TOOLS ===
        if tool_id in ['cai_strings_extract', 'strings_extract']:
            file_path = params.get('file', params.get('file_path', ''))
            args = params.get('args', '-a')
            return f"strings {args} {file_path}"
        
        if tool_id in ['cai_hexdump', 'hexdump']:
            file_path = params.get('file', params.get('file_path', ''))
            return f"hexdump -C {file_path}"
        
        if tool_id in ['cai_binwalk_analyze', 'binwalk']:
            file_path = params.get('file', params.get('file_path', ''))
            args = params.get('args', '-e')
            return f"binwalk {args} {file_path}"
        
        if tool_id in ['cai_readelf_info', 'readelf']:
            file_path = params.get('file', params.get('file_path', ''))
            args = params.get('args', '-a')
            return f"readelf {args} {file_path}"
        
        if tool_id in ['cai_objdump_disasm', 'objdump']:
            file_path = params.get('file', params.get('file_path', ''))
            args = params.get('args', '-d')
            return f"objdump {args} {file_path}"
        
        if tool_id in ['cai_file_identify', 'file_identify']:
            file_path = params.get('file', params.get('file_path', ''))
            return f"file {file_path}"
        
        # === REVERSE ENGINEERING ===
        if tool_id in ['cai_radare2', 'radare2', 'r2']:
            file_path = params.get('file', params.get('file_path', ''))
            args = params.get('args', '-A')
            commands = params.get('commands', 'pdf')
            return f"r2 {args} {file_path} -c '{commands}'"
        
        if tool_id in ['cai_gdb', 'gdb']:
            file_path = params.get('file', params.get('file_path', ''))
            commands = params.get('commands', '')
            if commands:
                return f"gdb -batch -ex '{commands}' {file_path}"
            return f"gdb {file_path}"
        
        # === MEMORY FORENSICS ===
        if tool_id in ['cai_volatility_analyze', 'volatility3', 'vol3']:
            memory_dump = params.get('memory_dump', params.get('file', ''))
            plugin = params.get('plugin', 'windows.pslist')
            args = params.get('args', '')
            return f"vol3 -f {memory_dump} {plugin} {args}"
        
        # === FORENSICS ===
        if tool_id in ['cai_pcap_analyze', 'pcap_analyze']:
            pcap_file = params.get('pcap_file', params.get('file', ''))
            return f"tcpdump -r {pcap_file} -n"
        
        if tool_id in ['cai_log_analyze', 'log_analyze']:
            log_file = params.get('log_file', params.get('file', ''))
            pattern = params.get('pattern', '')
            if pattern:
                return f"grep -i '{pattern}' {log_file}"
            return f"cat {log_file}"
        
        if tool_id in ['cai_timeline_create', 'timeline']:
            directory = params.get('directory', '/workspace')
            return f"find {directory} -type f -printf '%T@ %p\\n' | sort -n"
        
        # === CRYPTOGRAPHY ===
        if tool_id in ['cai_hash_file', 'hash_file']:
            file_path = params.get('file', params.get('file_path', ''))
            algorithm = params.get('algorithm', 'sha256')
            return f"{algorithm}sum {file_path}"
        
        if tool_id in ['cai_hash_data', 'hash_data']:
            data = params.get('data', '')
            algorithm = params.get('algorithm', 'sha256')
            return f"echo -n '{data}' | {algorithm}sum"
        
        if tool_id in ['cai_base64_encode', 'base64_encode']:
            data = params.get('data', '')
            return f"echo -n '{data}' | base64"
        
        if tool_id in ['cai_base64_decode', 'base64_decode']:
            data = params.get('data', '')
            return f"echo '{data}' | base64 -d"
        
        # === FILESYSTEM ===
        if tool_id in ['cai_filesystem_read', 'filesystem_read']:
            path = params.get('path', params.get('file', ''))
            return f"cat {path}"
        
        if tool_id in ['cai_filesystem_write', 'filesystem_write']:
            path = params.get('path', params.get('file', ''))
            content = params.get('content', '')
            return f"cat > {path} << 'EOFDATA'\n{content}\nEOFDATA"
        
        if tool_id in ['cai_filesystem_search', 'filesystem_search']:
            directory = params.get('directory', '/workspace')
            pattern = params.get('pattern', '*')
            return f"find {directory} -name '{pattern}'"
        
        # === SEARCH ENGINES ===
        if tool_id in ['cai_shodan_search', 'shodan_search']:
            query = params.get('query', '')
            # Note: Shodan requires API key, this is a placeholder
            return f"echo 'Shodan search for: {query}' && echo 'Note: Requires SHODAN_API_KEY'"
        
        if tool_id in ['cai_shodan_host_info', 'shodan_host_info']:
            host = params.get('host', '')
            return f"echo 'Shodan host info for: {host}' && echo 'Note: Requires SHODAN_API_KEY'"
        
        if tool_id in ['cai_google_search', 'google_search']:
            query = params.get('query', '')
            return f"echo 'Google search for: {query}' && echo 'Note: Use web scraping tools'"
        
        # === GENERIC LINUX COMMAND ===
        if tool_id in ['cai_generic_linux_command', 'generic_linux_command']:
            cmd = params.get('command', params.get('cmd', ''))
            return cmd
        
        # === FALLBACK: Try to extract command from tool_id ===
        # If tool starts with cai_, try to map to actual binary
        if tool_id.startswith('cai_'):
            base_tool = tool_id[4:]  # Remove 'cai_' prefix
            # Try to build simple command
            file_param = params.get('file', params.get('file_path', params.get('target', '')))
            if file_param:
                return f"{base_tool} {file_param}"
        
        # No mapping found
        return None
    
    def execute_sequence(
        self,
        tool_calls: List[Dict[str, Any]],
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        stop_on_error: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a sequence of tool calls
        
        Args:
            tool_calls: List of {"tool_id": str, "params": dict}
            stop_on_error: Whether to stop on first error
        
        Returns:
            List of execution results
        """
        results = []
        
        for call in tool_calls:
            tool_id = call.get("tool_id")
            params = call.get("params", {})
            
            success, result = self.execute(tool_id, params, agent_id, task_id)
            
            results.append({
                "tool_id": tool_id,
                "success": success,
                "result": result
            })
            
            if not success and stop_on_error:
                break
        
        return results
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "requires_sandbox": t.requires_sandbox,
                "parameters": t.parameters
            }
            for t in self.registry.list_all()
        ]
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompts"""
        return self.registry.get_tool_descriptions()
    
    def get_tools_for_agent(self, capabilities: List[str]) -> List[str]:
        """Get tool IDs appropriate for an agent's capabilities"""
        return self.registry.get_tool_list_for_agent(capabilities)
    
    def register_dynamic_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        function: callable,
        parameters: Dict[str, str],
        created_by: str = "dynamic"
    ) -> bool:
        """Register a dynamically created tool"""
        try:
            self.registry.register(ToolDefinition(
                id=tool_id,
                name=name,
                description=description,
                category="dynamic",
                function=function,
                requires_sandbox=True,
                parameters=parameters,
                created_by=created_by
            ))
            return True
        except Exception as e:
            print(f"[TOOL_EXECUTOR] Failed to register dynamic tool: {e}")
            return False
    
    def get_execution_history(
        self,
        tool_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered"""
        history = self.execution_history
        
        if tool_id:
            history = [h for h in history if h.get("tool_id") == tool_id]
        
        if task_id:
            history = [h for h in history if h.get("task_id") == task_id]
        
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get("success"))
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "registered_tools": len(self.registry.list_all()),
            "registry_stats": self.registry.get_stats()
        }
    
    def set_sandbox(self, sandbox) -> None:
        """Set or change sandbox"""
        self.sandbox = sandbox
        self.builtin_tools.set_sandbox(sandbox)
    
    def set_workspace(self, workspace_dir: str) -> None:
        """Change workspace directory"""
        self.workspace_dir = workspace_dir
        self.builtin_tools.set_workspace(workspace_dir)
    
    # ==========================================
    # Container Management Methods
    # ==========================================
    
    def get_container_status(self) -> Dict[str, Any]:
        """Get status of all Docker containers"""
        if not self.container_router:
            return {"available": False, "message": "Container router not initialized"}
        
        return {
            "available": self.container_router.is_available(),
            "containers": self.container_router.get_container_status()
        }
    
    def start_containers(self) -> Dict[str, bool]:
        """Start all Docker containers for tool execution"""
        if not self.container_router:
            print("[TOOL_EXECUTOR] Container router not available")
            return {}
        
        return self.container_router.start_containers()
    
    def stop_containers(self) -> bool:
        """Stop all Docker containers"""
        if not self.container_router:
            return False
        
        return self.container_router.stop_containers()
    
    def execute_in_container(
        self,
        container_type: str,
        command: str,
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        Execute a raw command in a specific container.
        
        Args:
            container_type: Type of container (sandbox, security, web, nodejs)
            command: Command to execute
            timeout: Execution timeout
        
        Returns:
            Tuple of (success, output)
        """
        if not self.container_router:
            return False, "Container router not available"
        
        # Map string to ContainerType
        from sandbox.container_router import ContainerType
        
        type_map = {
            "sandbox": ContainerType.SANDBOX,
            "security": ContainerType.SECURITY,
            "web": ContainerType.WEB,
            "nodejs": ContainerType.NODEJS
        }
        
        ct = type_map.get(container_type.lower())
        if not ct:
            return False, f"Unknown container type: {container_type}"
        
        # Get container config and execute
        config = self.container_router.CONTAINERS.get(ct)
        if not config:
            return False, f"No config for container: {container_type}"
        
        container = self.container_router._containers.get(config.container_name)
        if not container:
            return False, f"Container not running: {config.container_name}"
        
        return self.container_router._execute_in_container(
            container, command, config.workdir, timeout
        )
    
    def write_to_container(
        self,
        container_type: str,
        filename: str,
        content: str
    ) -> Tuple[bool, str]:
        """Write a file to a container's workspace"""
        if not self.container_router:
            return False, "Container router not available"
        
        from sandbox.container_router import ContainerType
        
        type_map = {
            "sandbox": ContainerType.SANDBOX,
            "security": ContainerType.SECURITY,
            "web": ContainerType.WEB,
            "nodejs": ContainerType.NODEJS
        }
        
        ct = type_map.get(container_type.lower())
        if not ct:
            return False, f"Unknown container type: {container_type}"
        
        return self.container_router.write_file_to_container(ct, filename, content)
    
    def read_from_container(
        self,
        container_type: str,
        filename: str
    ) -> Tuple[bool, str]:
        """Read a file from a container's workspace"""
        if not self.container_router:
            return False, "Container router not available"
        
        from sandbox.container_router import ContainerType
        
        type_map = {
            "sandbox": ContainerType.SANDBOX,
            "security": ContainerType.SECURITY,
            "web": ContainerType.WEB,
            "nodejs": ContainerType.NODEJS
        }
        
        ct = type_map.get(container_type.lower())
        if not ct:
            return False, f"Unknown container type: {container_type}"
        
        return self.container_router.read_file_from_container(ct, filename)
    
    def get_tools_by_container(self) -> Dict[str, List[str]]:
        """Get tools grouped by their container type"""
        if not self.container_router:
            return {}
        
        tools = self.container_router.get_available_tools()
        by_container = {}
        
        for tool_id, info in tools.items():
            container = info.get("container", "host")
            if container not in by_container:
                by_container[container] = []
            by_container[container].append(tool_id)
        
        return by_container


if __name__ == "__main__":
    # Test tool executor
    executor = ToolExecutor(workspace_dir="test_workspace")
    
    print("\n" + "="*60)
    print("TOOL EXECUTOR TEST")
    print("="*60 + "\n")
    
    # List available tools
    print("Available tools:")
    for tool in executor.get_available_tools():
        print(f"  - {tool['id']}: {tool['description']}")
    
    print("\n" + "-"*60)
    print("Executing tools...")
    print("-"*60 + "\n")
    
    # Test filesystem write
    success, result = executor.execute(
        "filesystem_write",
        {"path": "test.txt", "content": "Hello from executor!"},
        agent_id="test_agent",
        task_id="test_task"
    )
    print(f"Write: {success} - {result}")
    
    # Test filesystem read
    success, result = executor.execute(
        "filesystem_read",
        {"path": "test.txt"}
    )
    print(f"Read: {success} - {result}")
    
    # Test Python execution
    success, result = executor.execute(
        "python_exec",
        {"code": "print('Hello!')\nprint(sum(range(10)))"}
    )
    print(f"Python: {success}")
    print(f"Output: {result}")
    
    # Cleanup
    executor.execute("filesystem_delete", {"path": "test.txt"})
    
    # Stats
    print(f"\nStats: {executor.get_stats()}")
