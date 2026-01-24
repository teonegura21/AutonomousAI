"""
Live Executor Monitor
Provides rich, real-time visualization of agent execution and tool output.
Inspired by CAI's CLI output logic.
"""

import sys
import threading
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from rich.style import Style

class LiveMonitor:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.console = Console()
        self.live: Optional[Live] = None
        self.current_buffer = ""
        self.current_title = ""
        self.current_agent = ""
        self.is_running = False
        
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def show_command_header(self, agent_id: str, command: str, container_name: str = "kali"):
        """Start a new live execution panel"""
        self.stop_live() # Stop any existing
        
        self.current_agent = agent_id
        self.current_buffer = ""
        self.current_title = f"[bold cyan]{agent_id}[/bold cyan] executing in [bold magenta]{container_name}[/bold magenta]"
        
        # Initial Panel
        panel = self._create_panel(command, "running")
        
        self.live = Live(panel, refresh_per_second=4, console=self.console)
        self.live.start()
        self.is_running = True

    def show_output_line(self, line: str, is_error: bool = False):
        """Update the live panel with new output"""
        if not self.live or not self.is_running:
            # Fallback if somehow live isn't started
            print(line, end="")
            return

        self.current_buffer += line
        # Update panel content
        panel = self._create_panel(self.current_buffer, "running")
        self.live.update(panel)

    def show_command_footer(self, success: bool, duration: float, exit_code: int):
        """Finalize the panel state"""
        if self.live:
            status = "completed" if success else "error"
            
            # Final update
            panel = self._create_panel(self.current_buffer, status, duration, exit_code)
            self.live.update(panel)
            self.live.stop()
            self.live = None
            self.is_running = False

    def show_error(self, message: str):
        """Show an error message"""
        if self.live:
            self.stop_live()
        
        self.console.print(f"[bold red]❌ ERROR: {message}[/bold red]")

    def stop_live(self):
        """Force stop live display"""
        if self.live:
            try:
                self.live.stop()
            except:
                pass
            self.live = None
            self.is_running = False

    def _create_panel(self, content: str, status: str, duration: float = None, exit_code: int = None) -> Panel:
        """Create the Rich Panel with current status"""
        
        border_style = "yellow"
        title_suffix = "[bold yellow]RUNNING[/bold yellow]"
        
        if status == "completed":
            border_style = "green"
            title_suffix = f"[bold green]COMPLETED[/bold green]"
        elif status == "error":
            border_style = "red"
            title_suffix = f"[bold red]FAILED[/bold red]"

        if duration is not None:
             title_suffix += f" ({duration:.2f}s)"
        if exit_code is not None:
             title_suffix += f" [Exit: {exit_code}]"

        title = f"{self.current_title} - {title_suffix}"

        # Truncate content if too long for preview, but keep tail for logs
        display_content = content
        if len(display_content) > 2000:
             display_content = "... (truncated) ...\n" + display_content[-2000:]
             
        return Panel(
            Text.from_ansi(display_content) if display_content else Text("Waiting for output...", style="dim italic"),
            title=title,
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
            title_align="left"
        )

def get_live_monitor() -> LiveMonitor:
    return LiveMonitor.get_instance()
