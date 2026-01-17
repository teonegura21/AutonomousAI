"""
Kali Agent Executor - Automatic Tool Routing to Kali Container
================================================================
All Kali agents execute their tools inside the Kali container automatically.
Agents have FULL FREEDOM to run any command - this is their sandbox.
"""

import docker
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from monitoring.live_executor import get_live_monitor
    LIVE_MONITOR_AVAILABLE = True
except ImportError:
    LIVE_MONITOR_AVAILABLE = False
    print("[KALI_EXECUTOR] Live monitor not available - using basic output")

class KaliAgentExecutor:
    """
    Routes agent tool executions to Kali container automatically.
    Gives agents full freedom to execute any tool in their sandbox.
    """
    
    CONTAINER_NAME = "agent_kali"
    WORKSPACE_DIR = "/workspace"
    
    # Agents that should execute in Kali (all security agents)
    KALI_AGENTS = {
        "reverse_engineering_agent",
        "decompiling_analysis_agent",
        "dfir",
        "memory_analysis_agent",
        "bug_bounty_agent",
        "thought",
        "codeagent",
        "reporter",
        # Also include kali_ prefixed versions
        "kali_reverse_engineering_agent",
        "kali_decompiling_analysis_agent",
        "kali_dfir",
        "kali_memory_analysis_agent",
        "kali_bug_bounty_agent",
        "kali_thought",
        "kali_codeagent",
        "kali_reporter",
    }
    
    def __init__(self):
        self.docker_client = None
        self.container = None
        self._init_docker()
    
    def _init_docker(self):
        """Initialize Docker client and get Kali container"""
        try:
            self.docker_client = docker.from_env()
            self.container = self.docker_client.containers.get(self.CONTAINER_NAME)
            print(f"[KALI_EXECUTOR] Connected to {self.CONTAINER_NAME}")
        except Exception as e:
            print(f"[KALI_EXECUTOR] Docker not available: {e}")
            print("[KALI_EXECUTOR] Agents will run locally. Start Kali: cd docker && docker-compose up -d kali")
    
    def should_use_kali(self, agent_id: str, tool_id: str) -> bool:
        """
        Determine if this agent/tool should execute in Kali.
        
        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier
            
        Returns:
            True if should execute in Kali container
        """
        # Check if agent is a Kali agent
        if agent_id in self.KALI_AGENTS:
            return True
        
        # Check if tool is a security/Kali tool
        kali_tool_prefixes = ["cai_", "kali_", "nmap", "sqlmap", "metasploit", 
                              "binwalk", "volatility", "hydra", "john", "gobuster"]
        if any(tool_id.startswith(prefix) for prefix in kali_tool_prefixes):
            return True
        
        return False
    
    def execute_in_kali(
        self,
        command: str,
        agent_id: str = "unknown",
        timeout: int = 300,
        workdir: Optional[str] = None,
        stream: bool = True
    ) -> Tuple[bool, str]:
        """
        Execute a command in Kali container with REAL-TIME OUTPUT STREAMING.
        Agents have FULL FREEDOM - no restrictions except timeout.
        
        Args:
            command: Command to execute
            agent_id: Agent executing the command (for logging)
            timeout: Execution timeout in seconds
            workdir: Working directory (default: /workspace)
            stream: If True, print output in real-time
            
        Returns:
            Tuple of (success, output)
        """
        if not self.container:
            return False, "[ERROR] Kali container not available. Start with: docker-compose up -d kali"
        
        work_path = workdir or self.WORKSPACE_DIR
        
        try:
            # Log command execution
            self._log_command(agent_id, command)
            
            # === LIVE MONITOR: Show command header ===
            if LIVE_MONITOR_AVAILABLE and stream:
                monitor = get_live_monitor()
                monitor.show_command_header(agent_id, command, "agent_kali")
            else:
                print(f"\n[KALI] Executing: {command[:100]}...")
                print(f"[KALI] Stream: {'ENABLED' if stream else 'DISABLED'}\n")
            
            start_time = time.time()
            
            # Execute with streaming
            exec_id = self.docker_client.api.exec_create(
                container=self.container.id,
                cmd=["bash", "-c", command],
                workdir=work_path,
                environment={
                    "AGENT_ID": agent_id,
                    "EXECUTION_TIME": str(datetime.now())
                },
                stdout=True,
                stderr=True,
                tty=False
            )
            
            # Stream output in real-time
            output_buffer = []
            
            if not LIVE_MONITOR_AVAILABLE and stream:
                print("="*70)
                print("KALI CONTAINER OUTPUT (LIVE):")
                print("="*70)
            
            for chunk in self.docker_client.api.exec_start(
                exec_id=exec_id['Id'],
                stream=True,
                demux=False
            ):
                decoded = chunk.decode('utf-8', errors='replace')
                output_buffer.append(decoded)
                
                if stream:
                    # === LIVE MONITOR: Show output line by line ===
                    if LIVE_MONITOR_AVAILABLE:
                        monitor = get_live_monitor()
                        monitor.show_output_line(decoded, is_error=False)
                    else:
                        print(decoded, end='', flush=True)
            
            duration = time.time() - start_time
            
            # Get exit code
            inspect = self.docker_client.api.exec_inspect(exec_id['Id'])
            exit_code = inspect['ExitCode']
            
            full_output = ''.join(output_buffer)
            success = exit_code == 0
            
            # === LIVE MONITOR: Show completion footer ===
            if LIVE_MONITOR_AVAILABLE and stream:
                monitor = get_live_monitor()
                monitor.show_command_footer(success, duration, exit_code)
            elif stream:
                print("\n" + "="*70)
                print(f"{'SUCCESS' if success else 'FAILED'} | Duration: {duration:.2f}s | Exit: {exit_code}")
                print("="*70 + "\n")
            
            # Log result
            self._log_result(agent_id, command, success, full_output)
            
            return success, full_output or "(no output)"
            
        except Exception as e:
            error_msg = f"[KALI_EXECUTOR] Execution error: {str(e)}"
            self._log_result(agent_id, command, False, error_msg)
            
            if LIVE_MONITOR_AVAILABLE:
                monitor = get_live_monitor()
                monitor.show_error(error_msg)
            
            return False, error_msg
    
    def execute_python_in_kali(
        self,
        code: str,
        agent_id: str = "unknown",
        filename: Optional[str] = None,
        timeout: int = 120
    ) -> Tuple[bool, str]:
        """
        Execute Python code in Kali container.
        
        Args:
            code: Python code to execute
            agent_id: Agent identifier
            filename: Optional filename (auto-generated if not provided)
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, output)
        """
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agent_id}_{timestamp}.py"
        
        # Write code to container
        success, msg = self.write_file(filename, code)
        if not success:
            return False, f"Failed to write code: {msg}"
        
        # Execute
        command = f"python3 {self.WORKSPACE_DIR}/{filename}"
        return self.execute_in_kali(command, agent_id, timeout)
    
    def write_file(self, filename: str, content: str) -> Tuple[bool, str]:
        """
        Write a file to Kali container workspace.
        
        Args:
            filename: Name of file
            content: File contents
            
        Returns:
            Tuple of (success, message)
        """
        if not self.container:
            # Fallback to local
            local_path = Path("workspace") / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding='utf-8')
            return True, f"Written locally to {local_path}"
        
        try:
            # Use heredoc to write file (handles special characters)
            escaped_content = content.replace("'", "'\\''")
            command = f"cat > {self.WORKSPACE_DIR}/{filename} << 'EOFMARKER'\n{content}\nEOFMARKER"
            
            exit_code, output = self.container.exec_run(
                cmd=["bash", "-c", command],
                workdir=self.WORKSPACE_DIR
            )
            
            if exit_code == 0:
                return True, f"Written to {self.WORKSPACE_DIR}/{filename}"
            else:
                return False, f"Write failed: {output.decode('utf-8', errors='replace')}"
                
        except Exception as e:
            return False, f"Write error: {str(e)}"
    
    def read_file(self, filename: str) -> Tuple[bool, str]:
        """
        Read a file from Kali container workspace.
        
        Args:
            filename: Name of file to read
            
        Returns:
            Tuple of (success, content)
        """
        if not self.container:
            # Fallback to local
            local_path = Path("workspace") / filename
            if local_path.exists():
                return True, local_path.read_text(encoding='utf-8')
            return False, f"File not found: {local_path}"
        
        try:
            exit_code, output = self.container.exec_run(
                cmd=["cat", f"{self.WORKSPACE_DIR}/{filename}"],
                workdir=self.WORKSPACE_DIR
            )
            
            if exit_code == 0:
                return True, output.decode('utf-8', errors='replace')
            else:
                return False, "File not found or read error"
                
        except Exception as e:
            return False, f"Read error: {str(e)}"
    
    def list_files(self, directory: str = ".") -> Tuple[bool, str]:
        """
        List files in Kali container directory.
        
        Args:
            directory: Directory to list (relative to workspace)
            
        Returns:
            Tuple of (success, listing)
        """
        command = f"ls -lah {directory}"
        return self.execute_in_kali(command, "system", timeout=10)
    
    def get_container_status(self) -> Dict[str, Any]:
        """Get status of Kali container"""
        if not self.container:
            return {
                "available": False,
                "status": "not_connected",
                "message": "Container not available. Start with: docker-compose up -d kali"
            }
        
        try:
            self.container.reload()
            return {
                "available": True,
                "status": self.container.status,
                "name": self.container.name,
                "image": self.container.image.tags[0] if self.container.image.tags else "unknown",
                "ports": self.container.ports,
                "workspace": self.WORKSPACE_DIR
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def _log_command(self, agent_id: str, command: str):
        """Log command to container logs"""
        if not self.container:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{agent_id}] {command}\n"
            
            self.container.exec_run(
                cmd=["bash", "-c", f"echo '{log_entry}' >> /logs/commands.log"],
                workdir=self.WORKSPACE_DIR
            )
        except:
            pass  # Logging failure shouldn't break execution
    
    def _log_result(self, agent_id: str, command: str, success: bool, output: str):
        """Log execution result"""
        if not self.container:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = "SUCCESS" if success else "FAILED"
            log_entry = f"[{timestamp}] [{agent_id}] {result}: {command[:100]}\n"
            
            self.container.exec_run(
                cmd=["bash", "-c", f"echo '{log_entry}' >> /logs/output.log"],
                workdir=self.WORKSPACE_DIR
            )
        except:
            pass


# Global executor instance
_kali_executor: Optional[KaliAgentExecutor] = None


def get_kali_executor() -> KaliAgentExecutor:
    """Get or create global Kali executor instance"""
    global _kali_executor
    if _kali_executor is None:
        _kali_executor = KaliAgentExecutor()
    return _kali_executor


def execute_for_agent(
    agent_id: str,
    tool_id: str,
    command: str,
    **kwargs
) -> Tuple[bool, str]:
    """
    Execute a tool for an agent - automatically routes to Kali if needed.
    
    Args:
        agent_id: Agent identifier
        tool_id: Tool identifier
        command: Command to execute
        **kwargs: Additional parameters (timeout, workdir, etc.)
        
    Returns:
        Tuple of (success, output)
    """
    executor = get_kali_executor()
    
    if executor.should_use_kali(agent_id, tool_id):
        print(f"[KALI_EXECUTOR] Routing {agent_id}/{tool_id} to Kali container")
        return executor.execute_in_kali(
            command,
            agent_id=agent_id,
            timeout=kwargs.get('timeout', 300),
            workdir=kwargs.get('workdir')
        )
    else:
        # Not a Kali tool - use local execution
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=kwargs.get('timeout', 60)
            )
            return result.returncode == 0, result.stdout or result.stderr
        except Exception as e:
            return False, str(e)


if __name__ == "__main__":
    # Test the executor
    print("\n" + "="*70)
    print("KALI AGENT EXECUTOR TEST")
    print("="*70 + "\n")
    
    executor = KaliAgentExecutor()
    
    status = executor.get_container_status()
    print(f"Container Status: {json.dumps(status, indent=2)}\n")
    
    if status['available']:
        # Test command execution
        print("Testing command execution...")
        success, output = executor.execute_in_kali("whoami", "test_agent")
        print(f"Success: {success}")
        print(f"Output: {output}\n")
        
        # Test file operations
        print("Testing file operations...")
        success, msg = executor.write_file("test.txt", "Hello from Kali!")
        print(f"Write: {msg}")
        
        success, content = executor.read_file("test.txt")
        print(f"Read: {content}\n")
        
        # Test listing
        success, listing = executor.list_files()
        print(f"Files:\n{listing}")
    else:
        print("Kali container not available. Start with:")
        print("  cd docker && docker-compose up -d kali")
    
    print("\n" + "="*70)
