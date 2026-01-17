"""
Live Execution Monitor - Real-Time Command Visualization
=========================================================
Shows agents executing commands in real-time with live output streaming.
Like watching a terminal - you see every command and its output as it happens.
"""

import sys
import time
from datetime import datetime
from typing import Optional
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows compatibility
init(autoreset=True)


class LiveExecutionMonitor:
    """
    Real-time command execution monitor with live output streaming.
    Displays commands and outputs as they execute - IDE-style visualization.
    """
    
    def __init__(self, enable_colors: bool = True):
        self.enable_colors = enable_colors
        self.command_count = 0
    
    def show_command_header(self, agent_id: str, command: str, container: str = "kali"):
        """
        Display command header before execution.
        Shows: agent, timestamp, command, container
        """
        self.command_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print("\n" + "█" * 80)
        print(f"{Fore.CYAN}[{self.command_count}] COMMAND EXECUTION{Style.RESET_ALL}")
        print("█" * 80)
        print(f"{Fore.GREEN}Agent:{Style.RESET_ALL}     {agent_id}")
        print(f"{Fore.GREEN}Time:{Style.RESET_ALL}      {timestamp}")
        print(f"{Fore.GREEN}Container:{Style.RESET_ALL} {container}")
        print(f"{Fore.GREEN}Command:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}{command}{Style.RESET_ALL}")
        print("─" * 80)
        print(f"{Fore.MAGENTA}▼ LIVE OUTPUT ▼{Style.RESET_ALL}")
        print("─" * 80)
    
    def show_output_line(self, line: str, is_error: bool = False):
        """
        Display a single line of output in real-time.
        """
        if is_error:
            print(f"{Fore.RED}{line}{Style.RESET_ALL}", end='', flush=True)
        else:
            print(line, end='', flush=True)
    
    def show_command_footer(self, success: bool, duration: float, exit_code: int = 0):
        """
        Display command completion status.
        Shows: success/failure, duration, exit code
        """
        print("─" * 80)
        
        if success:
            status_msg = f"{Fore.GREEN}✓ SUCCESS{Style.RESET_ALL}"
        else:
            status_msg = f"{Fore.RED}✗ FAILED{Style.RESET_ALL}"
        
        print(f"{status_msg} | Duration: {duration:.2f}s | Exit Code: {exit_code}")
        print("█" * 80 + "\n")
    
    def show_thinking(self, agent_id: str, message: str):
        """
        Display agent's thinking/planning phase.
        """
        print(f"\n{Fore.CYAN}💭 {agent_id}:{Style.RESET_ALL} {message}")
    
    def show_tool_selection(self, agent_id: str, tool: str, params: dict):
        """
        Display which tool the agent decided to use.
        """
        print(f"\n{Fore.YELLOW}🔧 {agent_id} selected tool:{Style.RESET_ALL} {tool}")
        if params:
            print(f"{Fore.YELLOW}   Parameters:{Style.RESET_ALL}")
            for key, value in params.items():
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:60] + "..."
                print(f"     {key}: {val_str}")
    
    def show_iteration_header(self, iteration: int, max_iterations: int):
        """
        Display iteration counter (for multi-step tasks).
        """
        print(f"\n{Back.BLUE}{Fore.WHITE} ITERATION {iteration}/{max_iterations} {Style.RESET_ALL}")
    
    def show_agent_response(self, agent_id: str, response: str, truncate: int = 200):
        """
        Display agent's text response (thinking/analysis).
        """
        print(f"\n{Fore.CYAN}🤖 {agent_id} says:{Style.RESET_ALL}")
        
        # Truncate if too long
        if len(response) > truncate:
            displayed = response[:truncate] + "..."
        else:
            displayed = response
        
        # Indent the response
        for line in displayed.split('\n'):
            print(f"   {line}")
    
    def show_task_start(self, task_id: str, agent_id: str, description: str):
        """
        Display task start banner.
        """
        print("\n" + "═" * 80)
        print(f"{Back.GREEN}{Fore.BLACK} TASK START {Style.RESET_ALL}")
        print("═" * 80)
        print(f"{Fore.GREEN}Task ID:{Style.RESET_ALL}     {task_id}")
        print(f"{Fore.GREEN}Agent:{Style.RESET_ALL}       {agent_id}")
        print(f"{Fore.GREEN}Description:{Style.RESET_ALL}")
        # Wrap description
        words = description.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 78:
                print(line)
                line = "  " + word
            else:
                line += " " + word if line != "  " else word
        if line != "  ":
            print(line)
        print("═" * 80 + "\n")
    
    def show_task_complete(self, task_id: str, success: bool, duration: float):
        """
        Display task completion banner.
        """
        print("\n" + "═" * 80)
        
        if success:
            status = f"{Back.GREEN}{Fore.BLACK} TASK COMPLETE ✓ {Style.RESET_ALL}"
        else:
            status = f"{Back.RED}{Fore.WHITE} TASK FAILED ✗ {Style.RESET_ALL}"
        
        print(status)
        print("═" * 80)
        print(f"{Fore.GREEN}Task ID:{Style.RESET_ALL}  {task_id}")
        print(f"{Fore.GREEN}Duration:{Style.RESET_ALL} {duration:.2f}s")
        print("═" * 80 + "\n")
    
    def show_error(self, error_msg: str):
        """
        Display error message prominently.
        """
        print(f"\n{Back.RED}{Fore.WHITE} ERROR {Style.RESET_ALL}")
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}\n")
    
    def show_warning(self, warning_msg: str):
        """
        Display warning message.
        """
        print(f"{Fore.YELLOW}⚠️  WARNING: {warning_msg}{Style.RESET_ALL}")
    
    def clear_screen(self):
        """
        Clear terminal screen (optional, for clean slate).
        """
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_progress_bar(self, current: int, total: int, prefix: str = "Progress"):
        """
        Display a simple progress bar.
        """
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        percent = (current / total) * 100
        
        print(f"\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})", end='', flush=True)
        
        if current >= total:
            print()  # New line when complete


# Global instance
_live_monitor = None

def get_live_monitor() -> LiveExecutionMonitor:
    """Get or create global live monitor instance."""
    global _live_monitor
    if _live_monitor is None:
        _live_monitor = LiveExecutionMonitor()
    return _live_monitor


# Convenience functions for easy import
def show_command(agent_id: str, command: str, container: str = "kali"):
    """Show command header."""
    get_live_monitor().show_command_header(agent_id, command, container)

def show_output(line: str, is_error: bool = False):
    """Show output line."""
    get_live_monitor().show_output_line(line, is_error)

def show_complete(success: bool, duration: float, exit_code: int = 0):
    """Show command completion."""
    get_live_monitor().show_command_footer(success, duration, exit_code)

def show_thinking(agent_id: str, message: str):
    """Show agent thinking."""
    get_live_monitor().show_thinking(agent_id, message)

def show_tool(agent_id: str, tool: str, params: dict):
    """Show tool selection."""
    get_live_monitor().show_tool_selection(agent_id, tool, params)

def show_iteration(iteration: int, max_iterations: int):
    """Show iteration header."""
    get_live_monitor().show_iteration_header(iteration, max_iterations)
