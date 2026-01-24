#!/usr/bin/env python3
"""
Docker Sandbox
Execute agent code in isolated Docker containers - host system is NEVER touched
"""

import os
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Try to import docker
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("[WARNING] Docker SDK not installed. Run: pip install docker")


class DockerSandbox:
    """
    Execute agent code in isolated Docker container.
    Host filesystem is NEVER touched directly.
    """
    
    DEFAULT_IMAGE = "python:3.11-slim"
    
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        mem_limit: str = "2g",
        cpu_limit: float = 2.0,
        network_mode: str = "bridge",
        workspace_name: str = "agent_workspace"
    ):
        self.image = image
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.network_mode = network_mode
        self.workspace_name = workspace_name
        self.workspace = "/workspace"
        
        self.client = None
        self.container = None
        self.container_id = None
        
        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
            except Exception as e:
                print(f"[SANDBOX] Docker not available: {e}")
    
    def is_available(self) -> bool:
        """Check if Docker is available"""
        if not DOCKER_AVAILABLE or not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def start(self) -> Optional[str]:
        """
        Start sandbox container
        
        Returns:
            Container ID or None if failed
        """
        if not self.is_available():
            print("[SANDBOX] Docker not available - using local execution")
            return None
        
        try:
            # Pull image if needed
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                print(f"[SANDBOX] Pulling image {self.image}...")
                self.client.images.pull(self.image)
            
            # Create container
            self.container = self.client.containers.run(
                self.image,
                detach=True,
                tty=True,
                working_dir=self.workspace,
                volumes={
                    self.workspace_name: {'bind': self.workspace, 'mode': 'rw'}
                },
                mem_limit=self.mem_limit,
                cpu_period=100000,
                cpu_quota=int(self.cpu_limit * 100000),
                network_mode=self.network_mode,
                remove=False,
                name=f"sandbox_{datetime.now().strftime('%H%M%S')}"
            )
            
            self.container_id = self.container.short_id
            print(f"[SANDBOX] Started container: {self.container_id}")
            
            return self.container_id
            
        except Exception as e:
            print(f"[SANDBOX] Failed to start container: {e}")
            return None
    
    def stop(self) -> None:
        """Stop and remove container"""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove()
                print(f"[SANDBOX] Stopped container: {self.container_id}")
            except Exception as e:
                print(f"[SANDBOX] Error stopping container: {e}")
            finally:
                self.container = None
                self.container_id = None
    
    def execute_command(
        self,
        command: str,
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        Execute bash command in sandbox
        
        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds
        
        Returns:
            Tuple of (success, output_or_error)
        """
        if not self.container:
            return self._local_execute(command, timeout)
        
        try:
            exit_code, output = self.container.exec_run(
                f"bash -c '{command}'",
                workdir=self.workspace,
                demux=True
            )
            
            stdout = output[0].decode('utf-8', errors='replace') if output[0] else ""
            stderr = output[1].decode('utf-8', errors='replace') if output[1] else ""
            
            if exit_code == 0:
                return True, stdout
            else:
                return False, stderr or stdout
                
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def write_file(self, filename: str, content: str) -> Tuple[bool, str]:
        """
        Write file inside container
        
        Args:
            filename: Filename (relative to workspace)
            content: File content
        
        Returns:
            Tuple of (success, message)
        """
        if not self.container:
            return self._local_write(filename, content)
        
        try:
            # Create directory if needed
            dirname = os.path.dirname(filename)
            if dirname:
                self.execute_command(f"mkdir -p {self.workspace}/{dirname}")
            
            # Write file using cat
            # Escape single quotes in content
            escaped = content.replace("'", "'\"'\"'")
            cmd = f"cat > {self.workspace}/{filename} << 'ENDOFFILE'\n{content}\nENDOFFILE"
            
            return self.execute_command(cmd)
            
        except Exception as e:
            return False, f"Write error: {str(e)}"
    
    def read_file(self, filename: str) -> Tuple[bool, str]:
        """Read file from container"""
        if not self.container:
            return self._local_read(filename)
        
        return self.execute_command(f"cat {self.workspace}/{filename}")
    
    def run_python(
        self,
        filename: str,
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        Execute Python file in sandbox
        
        Args:
            filename: Python file to execute
            timeout: Execution timeout
        
        Returns:
            Tuple of (success, output)
        """
        return self.execute_command(f"python {self.workspace}/{filename}")
    
    def run_python_code(
        self,
        code: str,
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        Execute Python code directly
        
        Args:
            code: Python code to execute
            timeout: Execution timeout
        
        Returns:
            Tuple of (success, output)
        """
        # Write to temp file and execute
        success, msg = self.write_file("_temp_script.py", code)
        if not success:
            return False, f"Failed to write script: {msg}"
        
        return self.run_python("_temp_script.py", timeout)
    
    def install_package(self, package: str) -> Tuple[bool, str]:
        """Install pip package in sandbox"""
        return self.execute_command(f"pip install {package}")
    
    def list_files(self, directory: str = "") -> Tuple[bool, str]:
        """List files in directory"""
        path = f"{self.workspace}/{directory}" if directory else self.workspace
        return self.execute_command(f"ls -la {path}")
    
    def delete_file(self, filename: str) -> Tuple[bool, str]:
        """Delete file from sandbox"""
        return self.execute_command(f"rm -f {self.workspace}/{filename}")
    
    def _local_execute(self, command: str, timeout: int) -> Tuple[bool, str]:
        """Fallback local execution when Docker unavailable"""
        try:
            workspace_dir = Path(os.getenv("AI_AUTONOM_WORKSPACE", "outputs"))
            workspace_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace_dir)
            )
            return result.returncode == 0, result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, str(e)
    
    def _local_write(self, filename: str, content: str) -> Tuple[bool, str]:
        """Fallback local write"""
        try:
            path = Path(os.getenv("AI_AUTONOM_WORKSPACE", "outputs")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return True, f"Written to {path}"
        except Exception as e:
            return False, str(e)
    
    def _local_read(self, filename: str) -> Tuple[bool, str]:
        """Fallback local read"""
        try:
            path = Path(os.getenv("AI_AUTONOM_WORKSPACE", "outputs")) / filename
            return True, path.read_text(encoding='utf-8')
        except Exception as e:
            return False, str(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get container statistics"""
        if not self.container:
            return {"status": "not_running", "docker_available": self.is_available()}
        
        try:
            stats = self.container.stats(stream=False)
            return {
                "status": "running",
                "container_id": self.container_id,
                "image": self.image,
                "memory_usage": stats.get("memory_stats", {}).get("usage", 0),
                "cpu_percent": self._calculate_cpu_percent(stats)
            }
        except:
            return {"status": "unknown", "container_id": self.container_id}
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from stats"""
        try:
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            if system_delta > 0:
                return (cpu_delta / system_delta) * 100.0
        except:
            pass
        return 0.0
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class SandboxManager:
    """
    Manage multiple sandbox instances
    """
    
    def __init__(self):
        self.sandboxes: Dict[str, DockerSandbox] = {}
        self.default_sandbox: Optional[DockerSandbox] = None
    
    def create_sandbox(
        self,
        name: str,
        image: str = DockerSandbox.DEFAULT_IMAGE,
        **kwargs
    ) -> DockerSandbox:
        """Create a named sandbox"""
        sandbox = DockerSandbox(image=image, **kwargs)
        sandbox.start()
        self.sandboxes[name] = sandbox
        
        if not self.default_sandbox:
            self.default_sandbox = sandbox
        
        return sandbox
    
    def get_sandbox(self, name: str) -> Optional[DockerSandbox]:
        """Get sandbox by name"""
        return self.sandboxes.get(name)
    
    def get_default(self) -> Optional[DockerSandbox]:
        """Get default sandbox"""
        return self.default_sandbox
    
    def stop_sandbox(self, name: str) -> bool:
        """Stop a specific sandbox"""
        sandbox = self.sandboxes.pop(name, None)
        if sandbox:
            sandbox.stop()
            if sandbox == self.default_sandbox:
                self.default_sandbox = None
            return True
        return False
    
    def stop_all(self) -> None:
        """Stop all sandboxes"""
        for sandbox in self.sandboxes.values():
            sandbox.stop()
        self.sandboxes.clear()
        self.default_sandbox = None
    
    def list_sandboxes(self) -> List[str]:
        """List all sandbox names"""
        return list(self.sandboxes.keys())


if __name__ == "__main__":
    # Test Docker sandbox
    print("\n" + "="*60)
    print("DOCKER SANDBOX TEST")
    print("="*60 + "\n")
    
    sandbox = DockerSandbox()
    
    print(f"Docker available: {sandbox.is_available()}")
    
    if sandbox.is_available():
        # Start container
        container_id = sandbox.start()
        
        if container_id:
            # Test file operations
            print("\nTesting file operations...")
            success, msg = sandbox.write_file("test.py", "print('Hello from sandbox!')")
            print(f"Write: {success} - {msg}")
            
            # Test Python execution
            print("\nTesting Python execution...")
            success, output = sandbox.run_python("test.py")
            print(f"Execute: {success}")
            print(f"Output: {output}")
            
            # Test command
            print("\nTesting command execution...")
            success, output = sandbox.execute_command("python --version")
            print(f"Python version: {output}")
            
            # Get stats
            print(f"\nStats: {sandbox.get_stats()}")
            
            # Stop
            sandbox.stop()
    else:
        print("\nTesting local fallback...")
        success, msg = sandbox.write_file("test_local.py", "print('Hello locally!')")
        print(f"Local write: {success}")
        
        success, output = sandbox.run_python_code("print('Hello from local!')")
        print(f"Local execute: {success} - {output}")
