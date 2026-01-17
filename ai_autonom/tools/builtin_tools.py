#!/usr/bin/env python3
"""
Built-in Tools
Actual tool implementations that DO things - file operations, code execution, web access
"""

import os
import subprocess
import glob
import tempfile
import json
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class BuiltinTools:
    """
    Actual tool implementations that DO things.
    These are the real tools agents use to interact with the system.
    """
    
    def __init__(self, sandbox=None, workspace_dir: str = "outputs"):
        """
        Args:
            sandbox: Optional Docker sandbox for isolated execution
            workspace_dir: Directory for file operations
        """
        self.sandbox = sandbox
        self.workspace_dir = workspace_dir
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # FILESYSTEM TOOLS
    # =========================================================================
    
    def filesystem_read(self, path: str) -> Tuple[bool, str]:
        """
        Read file contents
        
        Args:
            path: Path to file (relative to workspace or absolute)
        
        Returns:
            Tuple of (success, content_or_error)
        """
        if self.sandbox:
            return self.sandbox.read_file(path)
        
        try:
            # Handle relative paths
            if not os.path.isabs(path):
                path = os.path.join(self.workspace_dir, path)
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
            
        except FileNotFoundError:
            return False, f"File not found: {path}"
        except PermissionError:
            return False, f"Permission denied: {path}"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def filesystem_write(self, path: str, content: str) -> Tuple[bool, str]:
        """
        Write content to file
        
        Args:
            path: Path to file
            content: Content to write
        
        Returns:
            Tuple of (success, message_or_error)
        """
        if self.sandbox:
            return self.sandbox.write_file(path, content)
        
        try:
            # Handle relative paths
            if not os.path.isabs(path):
                path = os.path.join(self.workspace_dir, path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"Successfully written to {path}"
            
        except PermissionError:
            return False, f"Permission denied: {path}"
        except Exception as e:
            return False, f"Error writing file: {str(e)}"
    
    def filesystem_append(self, path: str, content: str) -> Tuple[bool, str]:
        """Append content to file"""
        try:
            if not os.path.isabs(path):
                path = os.path.join(self.workspace_dir, path)
            
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"Successfully appended to {path}"
            
        except Exception as e:
            return False, f"Error appending: {str(e)}"
    
    def filesystem_search(self, directory: str, pattern: str) -> Tuple[bool, str]:
        """
        Search for files matching pattern
        
        Args:
            directory: Directory to search
            pattern: Glob pattern (e.g., "*.py", "**/*.txt")
        
        Returns:
            Tuple of (success, matched_files_or_error)
        """
        try:
            if not os.path.isabs(directory):
                directory = os.path.join(self.workspace_dir, directory)
            
            matches = glob.glob(os.path.join(directory, pattern), recursive=True)
            
            if matches:
                return True, "\n".join(matches)
            else:
                return True, "No files found matching pattern"
                
        except Exception as e:
            return False, f"Error searching: {str(e)}"
    
    def filesystem_list(self, directory: str = ".") -> Tuple[bool, str]:
        """List files in directory"""
        try:
            if not os.path.isabs(directory):
                directory = os.path.join(self.workspace_dir, directory)
            
            items = os.listdir(directory)
            result = []
            for item in sorted(items):
                full_path = os.path.join(directory, item)
                if os.path.isdir(full_path):
                    result.append(f"[DIR]  {item}/")
                else:
                    size = os.path.getsize(full_path)
                    result.append(f"[FILE] {item} ({size} bytes)")
            
            return True, "\n".join(result) if result else "Empty directory"
            
        except Exception as e:
            return False, f"Error listing directory: {str(e)}"
    
    def filesystem_delete(self, path: str) -> Tuple[bool, str]:
        """Delete a file"""
        try:
            if not os.path.isabs(path):
                path = os.path.join(self.workspace_dir, path)
            
            if os.path.isfile(path):
                os.remove(path)
                return True, f"Deleted: {path}"
            else:
                return False, f"Not a file or doesn't exist: {path}"
                
        except Exception as e:
            return False, f"Error deleting: {str(e)}"
    
    # =========================================================================
    # CODE EXECUTION TOOLS
    # =========================================================================
    
    def python_exec(
        self,
        code: str,
        filename: str = "script.py",
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        Execute Python code
        
        Args:
            code: Python code to execute
            filename: Optional filename for the script
            timeout: Execution timeout in seconds
        
        Returns:
            Tuple of (success, output_or_error)
        """
        if self.sandbox:
            self.sandbox.write_file(filename, code)
            return self.sandbox.run_python(filename)
        
        try:
            # Write to temp file
            script_path = os.path.join(self.workspace_dir, filename)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace_dir
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr}"
            
            return result.returncode == 0, output if output else "(No output)"
            
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {timeout}s"
        except FileNotFoundError:
            return False, "Python interpreter not found"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def bash_exec(self, command: str, timeout: int = 30) -> Tuple[bool, str]:
        """
        Execute bash/shell command
        
        Args:
            command: Shell command to execute
            timeout: Execution timeout
        
        Returns:
            Tuple of (success, output_or_error)
        """
        if self.sandbox:
            return self.sandbox.execute_command(command)
        
        # Security: Restrict dangerous commands when not sandboxed
        dangerous_patterns = [
            "rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){:|:&};:",
            "chmod -R 777 /", "curl | sh", "wget | sh"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command:
                return False, f"Blocked potentially dangerous command: {pattern}"
        
        try:
            # Use shell=True on Windows, shell=False with bash on Unix
            if os.name == 'nt':
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.workspace_dir
                )
            else:
                result = subprocess.run(
                    ['bash', '-c', command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.workspace_dir
                )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr}"
            
            return result.returncode == 0, output if output else "(No output)"
            
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def pytest_run(self, test_path: str, timeout: int = 60) -> Tuple[bool, str]:
        """
        Run pytest on file or directory
        
        Args:
            test_path: Path to test file or directory
            timeout: Test timeout
        
        Returns:
            Tuple of (success, test_output)
        """
        if self.sandbox:
            return self.sandbox.execute_command(f"pytest {test_path} -v")
        
        try:
            if not os.path.isabs(test_path):
                test_path = os.path.join(self.workspace_dir, test_path)
            
            result = subprocess.run(
                ['pytest', test_path, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace_dir
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr}"
            
            return result.returncode == 0, output
            
        except FileNotFoundError:
            return False, "pytest not found. Install with: pip install pytest"
        except subprocess.TimeoutExpired:
            return False, f"Tests timed out after {timeout}s"
        except Exception as e:
            return False, f"Test error: {str(e)}"
    
    # =========================================================================
    # WEB TOOLS
    # =========================================================================
    
    def web_fetch(self, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Fetch content from URL
        
        Args:
            url: URL to fetch
            timeout: Request timeout
        
        Returns:
            Tuple of (success, content_or_error)
        """
        if not REQUESTS_AVAILABLE:
            return False, "requests library not installed"
        
        try:
            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'AI-Autonom/1.0'
            })
            response.raise_for_status()
            
            # Limit response size
            content = response.text[:50000]
            if len(response.text) > 50000:
                content += "\n...(truncated)"
            
            return True, content
            
        except requests.exceptions.Timeout:
            return False, f"Request timed out after {timeout}s"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
    
    def web_search(self, query: str) -> Tuple[bool, str]:
        """
        Search the web (placeholder - requires API key)
        
        Args:
            query: Search query
        
        Returns:
            Tuple of (success, results_or_error)
        """
        # This would integrate with Tavily, SerpAPI, or similar
        return False, "Web search not configured. Set TAVILY_API_KEY or use alternative."
    
    # =========================================================================
    # JSON/DATA TOOLS
    # =========================================================================
    
    def json_parse(self, content: str) -> Tuple[bool, Any]:
        """Parse JSON string"""
        try:
            data = json.loads(content)
            return True, data
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {str(e)}"
    
    def json_format(self, data: Any) -> Tuple[bool, str]:
        """Format data as JSON"""
        try:
            formatted = json.dumps(data, indent=2)
            return True, formatted
        except Exception as e:
            return False, f"JSON format error: {str(e)}"
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_workspace_path(self, relative_path: str) -> str:
        """Get absolute path within workspace"""
        return os.path.join(self.workspace_dir, relative_path)
    
    def set_workspace(self, workspace_dir: str) -> None:
        """Change workspace directory"""
        self.workspace_dir = workspace_dir
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    
    def set_sandbox(self, sandbox) -> None:
        """Set or change sandbox"""
        self.sandbox = sandbox


def create_builtin_tools(sandbox=None, workspace_dir: str = "outputs") -> Dict[str, callable]:
    """
    Create dictionary of all built-in tools
    
    Returns:
        Dict mapping tool_id to callable function
    """
    tools = BuiltinTools(sandbox, workspace_dir)
    
    return {
        "filesystem_read": tools.filesystem_read,
        "filesystem_write": tools.filesystem_write,
        "filesystem_append": tools.filesystem_append,
        "filesystem_search": tools.filesystem_search,
        "filesystem_list": tools.filesystem_list,
        "filesystem_delete": tools.filesystem_delete,
        "python_exec": tools.python_exec,
        "bash_exec": tools.bash_exec,
        "pytest_run": tools.pytest_run,
        "web_fetch": tools.web_fetch,
        "web_search": tools.web_search,
        "json_parse": tools.json_parse,
        "json_format": tools.json_format,
    }


if __name__ == "__main__":
    # Test built-in tools
    tools = BuiltinTools(workspace_dir="test_workspace")
    
    print("\n" + "="*60)
    print("BUILTIN TOOLS TEST")
    print("="*60 + "\n")
    
    # Test filesystem
    print("Testing filesystem_write...")
    success, msg = tools.filesystem_write("test.txt", "Hello, World!")
    print(f"  Write: {success} - {msg}")
    
    print("\nTesting filesystem_read...")
    success, content = tools.filesystem_read("test.txt")
    print(f"  Read: {success} - {content}")
    
    print("\nTesting filesystem_list...")
    success, listing = tools.filesystem_list(".")
    print(f"  List: {success}")
    print(listing)
    
    # Test Python execution
    print("\nTesting python_exec...")
    code = "print('Hello from Python!')\nprint(2 + 2)"
    success, output = tools.python_exec(code, "test_script.py")
    print(f"  Exec: {success}")
    print(f"  Output: {output}")
    
    # Cleanup
    tools.filesystem_delete("test.txt")
    tools.filesystem_delete("test_script.py")
    
    print("\nTests completed!")
