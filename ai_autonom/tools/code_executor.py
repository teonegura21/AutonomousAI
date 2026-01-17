#!/usr/bin/env python3
"""
Code Executor
Extract and execute code from agent responses
"""

import re
import os
import subprocess
import tempfile
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class CodeExecutor:
    """
    Extract and execute code from agent responses.
    Handles code blocks in markdown format.
    """
    
    def __init__(
        self,
        workspace_dir: str = "outputs",
        sandbox=None,
        default_timeout: int = 30
    ):
        self.workspace_dir = workspace_dir
        self.sandbox = sandbox
        self.default_timeout = default_timeout
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
        
        # Execution history
        self.history: List[Dict[str, Any]] = []
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from markdown-formatted text
        
        Args:
            text: Text containing code blocks
        
        Returns:
            List of {"language": str, "code": str}
        """
        # Match code blocks: ```language\ncode\n```
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        blocks = []
        for lang, code in matches:
            # Clean up the code
            code = code.strip()
            if code:
                blocks.append({
                    "language": lang.lower() if lang else "text",
                    "code": code
                })
        
        # Also try to find inline code definitions (def, class, function)
        if not blocks:
            # Look for Python function/class definitions
            if "def " in text or "class " in text:
                # Extract the code portion
                lines = text.split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                        in_code = True
                    
                    if in_code:
                        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                            if not line.strip().startswith(('def ', 'class ', 'import ', 'from ', '@', '#')):
                                break
                        code_lines.append(line)
                
                if code_lines:
                    blocks.append({
                        "language": "python",
                        "code": '\n'.join(code_lines)
                    })
        
        return blocks
    
    def extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code specifically"""
        blocks = self.extract_code_blocks(text)
        
        # First, look for explicitly marked Python blocks
        for block in blocks:
            if block["language"] in ("python", "py", ""):
                return block["code"]
        
        # Return the first block if no Python-specific one found
        if blocks:
            return blocks[0]["code"]
        
        return None
    
    def execute_python(
        self,
        code: str,
        filename: str = "script.py",
        timeout: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Execute Python code
        
        Args:
            code: Python code to execute
            filename: Filename to save as
            timeout: Execution timeout
        
        Returns:
            Tuple of (success, output_or_error)
        """
        timeout = timeout or self.default_timeout
        
        if self.sandbox:
            self.sandbox.write_file(filename, code)
            return self.sandbox.run_python(filename)
        
        try:
            # Write to workspace
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
            
            success = result.returncode == 0
            
            # Log execution
            self._log_execution(code, output, success, filename)
            
            return success, output if output else "(No output)"
            
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {timeout}s"
        except FileNotFoundError:
            return False, "Python interpreter not found"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def execute_and_save(
        self,
        agent_response: str,
        filename: str = "output.py",
        execute: bool = True
    ) -> Dict[str, Any]:
        """
        Extract code from agent response, save to file, and optionally execute
        
        Args:
            agent_response: Full agent response text
            filename: Filename to save extracted code
            execute: Whether to execute after saving
        
        Returns:
            Dict with code, file_path, execution_result
        """
        result = {
            "code": None,
            "file_path": None,
            "executed": False,
            "success": False,
            "output": None
        }
        
        # Extract code
        code = self.extract_python_code(agent_response)
        if not code:
            result["output"] = "No code block found in response"
            return result
        
        result["code"] = code
        
        # Save to file
        file_path = os.path.join(self.workspace_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            result["file_path"] = file_path
        except Exception as e:
            result["output"] = f"Failed to save file: {str(e)}"
            return result
        
        # Execute if requested
        if execute:
            result["executed"] = True
            success, output = self.execute_python(code, filename)
            result["success"] = success
            result["output"] = output
        else:
            result["success"] = True
            result["output"] = f"Code saved to {file_path}"
        
        return result
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax without executing
        
        Returns:
            Tuple of (is_valid, error_message_if_invalid)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    def combine_code_blocks(self, responses: List[str]) -> str:
        """
        Combine code from multiple agent responses
        Used for synthesis
        """
        all_code = []
        seen_imports = set()
        main_code = []
        
        for response in responses:
            code = self.extract_python_code(response)
            if code:
                lines = code.split('\n')
                for line in lines:
                    stripped = line.strip()
                    
                    # Track imports to avoid duplicates
                    if stripped.startswith(('import ', 'from ')):
                        if stripped not in seen_imports:
                            seen_imports.add(stripped)
                            all_code.append(line)
                    else:
                        main_code.append(line)
        
        # Combine: imports first, then main code
        combined = '\n'.join(all_code) + '\n\n' + '\n'.join(main_code)
        return combined.strip()
    
    def create_main_wrapper(self, code: str) -> str:
        """
        Wrap code in if __name__ == '__main__' if not already present
        """
        if "if __name__" in code:
            return code
        
        # Check if there's a main() function
        if "def main(" in code:
            return code + '\n\nif __name__ == "__main__":\n    main()'
        
        return code
    
    def _log_execution(
        self,
        code: str,
        output: str,
        success: bool,
        filename: str
    ) -> None:
        """Log execution for history"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "code_length": len(code),
            "success": success,
            "output_preview": output[:200] if output else None
        })
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.history[-limit:]
    
    def clear_history(self) -> None:
        """Clear execution history"""
        self.history.clear()


class CodeValidator:
    """
    Validate code quality and safety
    """
    
    # Patterns that might indicate dangerous code
    DANGEROUS_PATTERNS = [
        (r'os\.system\([\'"]rm -rf', "Dangerous rm command"),
        (r'subprocess\..*shell=True', "Shell injection risk"),
        (r'eval\(', "Eval usage (potential security risk)"),
        (r'exec\(', "Exec usage (potential security risk)"),
        (r'__import__\(', "Dynamic import (potential security risk)"),
        (r'open\([\'"]\/etc\/', "System file access"),
    ]
    
    @classmethod
    def check_safety(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Check code for potentially dangerous patterns
        
        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []
        
        for pattern, description in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                warnings.append(description)
        
        return len(warnings) == 0, warnings
    
    @classmethod
    def check_quality(cls, code: str) -> Dict[str, Any]:
        """
        Basic code quality checks
        """
        lines = code.split('\n')
        
        return {
            "line_count": len(lines),
            "has_docstring": '"""' in code or "'''" in code,
            "has_type_hints": ':' in code and '->' in code,
            "function_count": len(re.findall(r'def \w+\(', code)),
            "class_count": len(re.findall(r'class \w+', code)),
            "import_count": len(re.findall(r'^(?:import|from)\s', code, re.MULTILINE)),
            "comment_lines": sum(1 for line in lines if line.strip().startswith('#'))
        }


if __name__ == "__main__":
    # Test code executor
    executor = CodeExecutor(workspace_dir="test_workspace")
    
    print("\n" + "="*60)
    print("CODE EXECUTOR TEST")
    print("="*60 + "\n")
    
    # Test agent response with code block
    agent_response = """
I'll create a simple function to calculate factorial:

```python
def factorial(n):
    '''Calculate factorial of n'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test
print(f"5! = {factorial(5)}")
print(f"10! = {factorial(10)}")
```

This uses recursion for a clean implementation.
"""
    
    print("Testing code extraction and execution...")
    
    # Extract blocks
    blocks = executor.extract_code_blocks(agent_response)
    print(f"Found {len(blocks)} code blocks")
    for b in blocks:
        print(f"  Language: {b['language']}, Lines: {len(b['code'].split(chr(10)))}")
    
    # Execute and save
    result = executor.execute_and_save(agent_response, "factorial.py")
    print(f"\nExecution result:")
    print(f"  Success: {result['success']}")
    print(f"  File: {result['file_path']}")
    print(f"  Output: {result['output']}")
    
    # Validation
    code = executor.extract_python_code(agent_response)
    is_safe, warnings = CodeValidator.check_safety(code)
    print(f"\nSafety check: {'PASS' if is_safe else 'WARNINGS'}")
    if warnings:
        for w in warnings:
            print(f"  - {w}")
    
    quality = CodeValidator.check_quality(code)
    print(f"\nQuality metrics: {quality}")
