
"""
Compiler Tools
Specialized tools for compiling C++ and Rust code into portable executables.
Supports Cross-Compilation to Windows .exe from Linux containers.
"""

from typing import Tuple
import os
from .cai_security_tools import _TOOLS # Reuse generic command runner

class CompilerTools:
    def __init__(self):
        pass

    def compile_cpp(self, source_file: str, output_name: str, target_os: str = "windows") -> Tuple[bool, str]:
        """
        Compile C++ source code to an executable.
        
        Args:
            source_file: Path to .cpp file
            output_name: Desired output filename (without extension)
            target_os: 'windows' or 'linux'
        """
        # Auto-detect path (if relative, it might be in src/)
        if not os.path.exists(source_file) and "src" not in source_file:
             # Try looking in the session src dir if passed via session context
             # For now, we rely on the agent getting the path right or the session manager logic
             pass

        if target_os.lower() == "windows":
            # Use MinGW for Windows .exe (Cross-compile)
            compiler = "x86_64-w64-mingw32-g++"
            extension = ".exe"
            flags = "-static -static-libgcc -static-libstdc++" # CRITICAL for portability
        else:
            # Native Linux
            compiler = "g++"
            extension = ""
            flags = "-static"

        output_file = f"{output_name}{extension}"
        
        # Ensure output goes to bin/ directory if logical
        output_path = output_file
        if "bin" not in output_path and "/" not in output_path:
             # This heuristic depends on where the agent runs it, 
             # but usually we want it in the same dir or specified bin dir
             pass

        command = f"{compiler} {source_file} -o {output_path} {flags}"
        
        return _TOOLS.generic_linux_command(command)

# Export for Registry
_COMPILER = CompilerTools()
compile_cpp = _COMPILER.compile_cpp
