#!/usr/bin/env python3
"""
Kali Container Visibility Module
Provides real-time visibility into the Kali Linux execution environment
"""

import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class KaliContainerMonitor:
    """
    Monitor and interact with the Kali Linux container
    Provides visibility into command execution and findings
    """
    
    CONTAINER_NAME = "agent_kali"
    WEB_TERMINAL_PORT = 7681
    
    def __init__(self):
        self.client = None
        self.container = None
        
        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
                self._connect_container()
            except Exception as e:
                print(f"[KALI_MONITOR] Docker connection failed: {e}")
    
    def _connect_container(self):
        """Connect to the Kali container"""
        if not self.client:
            return
        
        try:
            self.container = self.client.containers.get(self.CONTAINER_NAME)
        except docker.errors.NotFound:
            print(f"[KALI_MONITOR] Container '{self.CONTAINER_NAME}' not found")
        except Exception as e:
            print(f"[KALI_MONITOR] Error connecting to container: {e}")
    
    def is_running(self) -> bool:
        """Check if Kali container is running"""
        if not self.container:
            self._connect_container()
        
        if self.container:
            try:
                self.container.reload()
                return self.container.status == "running"
            except:
                pass
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get container status information"""
        if not self.is_running():
            return {
                "running": False,
                "status": "not_running",
                "message": "Kali container is not running. Start with: docker-compose up -d kali"
            }
        
        try:
            stats = self.container.stats(stream=False)
            
            return {
                "running": True,
                "status": self.container.status,
                "id": self.container.short_id,
                "name": self.container.name,
                "web_terminal_url": f"http://localhost:{self.WEB_TERMINAL_PORT}",
                "cpu_percent": self._calculate_cpu_percent(stats),
                "memory_mb": stats['memory_stats'].get('usage', 0) / (1024 * 1024),
                "memory_limit_mb": stats['memory_stats'].get('limit', 0) / (1024 * 1024)
            }
        except Exception as e:
            return {"running": True, "status": "error", "error": str(e)}
    
    def _calculate_cpu_percent(self, stats: dict) -> float:
        """Calculate CPU percentage from stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * 100
        except:
            pass
        return 0.0
    
    def open_web_terminal(self) -> bool:
        """Open web terminal in browser"""
        if not self.is_running():
            print("[KALI_MONITOR] Container not running")
            return False
        
        url = f"http://localhost:{self.WEB_TERMINAL_PORT}"
        print(f"[KALI_MONITOR] Opening web terminal: {url}")
        webbrowser.open(url)
        return True
    
    def get_logs(self, lines: int = 100, follow: bool = False) -> str:
        """Get container logs"""
        if not self.container:
            return "Container not available"
        
        try:
            if follow:
                # Return generator for streaming
                return self.container.logs(stream=True, follow=True, tail=lines)
            else:
                return self.container.logs(tail=lines).decode('utf-8', errors='replace')
        except Exception as e:
            return f"Error getting logs: {e}"
    
    def get_command_log(self, lines: int = 50) -> str:
        """Get the command execution log from the container"""
        if not self.is_running():
            return "Container not running"
        
        try:
            exit_code, output = self.container.exec_run(
                f"tail -n {lines} /logs/commands.log",
                demux=True
            )
            if output[0]:
                return output[0].decode('utf-8', errors='replace')
            return "No commands logged yet"
        except Exception as e:
            return f"Error: {e}"
    
    def get_output_log(self, lines: int = 100) -> str:
        """Get the output log from the container"""
        if not self.is_running():
            return "Container not running"
        
        try:
            exit_code, output = self.container.exec_run(
                f"tail -n {lines} /logs/output.log",
                demux=True
            )
            if output[0]:
                return output[0].decode('utf-8', errors='replace')
            return "No output logged yet"
        except Exception as e:
            return f"Error: {e}"
    
    def get_findings(self) -> str:
        """Get security findings from the container"""
        if not self.is_running():
            return "Container not running"
        
        try:
            exit_code, output = self.container.exec_run(
                "cat /findings/findings.txt",
                demux=True
            )
            if output[0]:
                return output[0].decode('utf-8', errors='replace')
            return "No findings recorded yet"
        except Exception as e:
            return f"Error: {e}"
    
    def execute_command(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute a command in the Kali container with visibility"""
        if not self.is_running():
            return {
                "success": False,
                "error": "Container not running",
                "output": ""
            }
        
        try:
            # Use the logged runner script
            exit_code, output = self.container.exec_run(
                f"/tools/run.sh {command}",
                demux=True,
                workdir="/workspace"
            )
            
            stdout = output[0].decode('utf-8', errors='replace') if output[0] else ""
            stderr = output[1].decode('utf-8', errors='replace') if output[1] else ""
            
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "output": stdout,
                "error": stderr,
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    def list_tools(self) -> List[str]:
        """List available tools in Kali container"""
        common_tools = [
            "nmap", "masscan", "nikto", "gobuster", "dirb",
            "sqlmap", "hydra", "john", "hashcat",
            "metasploit-framework (msfconsole)",
            "searchsploit", "binwalk", "radare2", "ghidra",
            "wireshark", "tshark", "tcpdump",
            "volatility3", "foremost", "exiftool",
            "python3", "perl", "ruby"
        ]
        return common_tools
    
    def start_container(self) -> bool:
        """Start the Kali container"""
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        
        try:
            result = subprocess.run(
                ["docker-compose", "up", "-d", "kali"],
                cwd=docker_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("[KALI_MONITOR] Container started successfully")
                self._connect_container()
                return True
            else:
                print(f"[KALI_MONITOR] Failed to start: {result.stderr}")
                return False
        except Exception as e:
            print(f"[KALI_MONITOR] Error starting container: {e}")
            return False
    
    def stop_container(self) -> bool:
        """Stop the Kali container"""
        if not self.container:
            return True
        
        try:
            self.container.stop(timeout=10)
            print("[KALI_MONITOR] Container stopped")
            return True
        except Exception as e:
            print(f"[KALI_MONITOR] Error stopping: {e}")
            return False


def print_status():
    """Print Kali container status"""
    monitor = KaliContainerMonitor()
    status = monitor.get_status()
    
    print("\n" + "=" * 60)
    print("KALI LINUX CONTAINER STATUS")
    print("=" * 60)
    
    if status.get("running"):
        print(f"\nStatus: RUNNING")
        print(f"Container ID: {status.get('id', 'N/A')}")
        print(f"CPU Usage: {status.get('cpu_percent', 0):.1f}%")
        print(f"Memory: {status.get('memory_mb', 0):.1f} MB / {status.get('memory_limit_mb', 0):.1f} MB")
        print(f"\nWeb Terminal: {status.get('web_terminal_url')}")
        print("\nTo view terminal in browser, run:")
        print("  python -c \"from ai_autonom.sandbox.kali_monitor import KaliContainerMonitor; KaliContainerMonitor().open_web_terminal()\"")
    else:
        print(f"\nStatus: NOT RUNNING")
        print("\nTo start Kali container:")
        print("  cd docker && docker-compose up -d kali")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_status()
