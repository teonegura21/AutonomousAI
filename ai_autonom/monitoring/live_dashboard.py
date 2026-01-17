
"""
AI Autonom Mission Control Dashboard
A real-time TUI (Text User Interface) for monitoring the Orchestrator, Agents, and Blackboard.
"""

import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.tree import Tree
from rich import box
from rich.syntax import Syntax

from ..memory.knowledge_base import KnowledgeBase

class MissionDashboard:
    def __init__(self):
        self.console = Console()
        self.kb = KnowledgeBase.get_instance()
        self.layout = Layout()
        
        # State
        self.active_goal = "Waiting for command..."
        self.orchestrator_status = "Idle"
        self.current_plan: List[Dict] = []
        self.active_agent: Optional[str] = None
        self.agent_status = "Idle"
        self.last_agent_thought = ""
        self.active_tool: Optional[str] = None
        self.logs: List[str] = []
        
        # Threading
        self.running = False
        self.update_queue = queue.Queue()
        
        self._setup_layout()

    def _setup_layout(self):
        """Define the grid layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=1),   # Orchestrator/Plan
            Layout(name="center", ratio=2), # Active Agent/Execution
            Layout(name="right", ratio=1)   # Blackboard
        )
        
        self.layout["left"].split(
            Layout(name="orchestrator", size=6),
            Layout(name="plan")
        )
        
        self.layout["right"].split(
            Layout(name="findings", ratio=1),
            Layout(name="assets", ratio=1)
        )

    def update_goal(self, goal: str):
        self.active_goal = goal
        self.update_queue.put("refresh")

    def set_orchestrator_status(self, status: str):
        self.orchestrator_status = status
        self.update_queue.put("refresh")

    def update_plan(self, tasks: List[Dict]):
        self.current_plan = tasks
        self.update_queue.put("refresh")

    def set_active_agent(self, agent_name: str, status: str):
        self.active_agent = agent_name
        self.agent_status = status
        self.update_queue.put("refresh")

    def update_agent_thought(self, text: str):
        self.last_agent_thought = text
        self.update_queue.put("refresh")

    def set_active_tool(self, tool_name: str):
        self.active_tool = tool_name
        self.update_queue.put("refresh")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 8:
            self.logs.pop(0)
        self.update_queue.put("refresh")

    def _generate_header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            f"[b]MISSION GOAL:[/b] {self.active_goal}",
            f"[b]TIME:[/b] {datetime.now().strftime('%H:%M:%S')}"
        )
        return Panel(grid, style="white on blue", box=box.HEAVY)

    def _generate_orchestrator(self) -> Panel:
        status_color = "green" if self.orchestrator_status == "Idle" else "yellow"
        return Panel(
            f"[b]Status:[/b] [{status_color}]{self.orchestrator_status}[/{status_color}]\n"
            f"[b]Model:[/b] Nemotron-8B",
            title="🧠 Orchestrator",
            border_style="blue"
        )

    def _generate_plan(self) -> Panel:
        tree = Tree("📋 Mission Plan")
        for i, task in enumerate(self.current_plan, 1):
            # Determine icon based on state (mocked for now, assumes linear)
            icon = "⚪"
            style = "white"
            
            # Simple heuristic for visualization
            if self.active_agent and task.get('assigned_agent') == self.active_agent:
                icon = "🔵"
                style = "bold yellow"
            
            label = Text(f"{icon} Task {i}: {task.get('description', '')[:30]}...", style=style)
            label.append(f"\n   👤 {task.get('assigned_agent')}", style="dim")
            tree.add(label)
            
        return Panel(tree, title="Execution Plan", border_style="blue")

    def _generate_agent_view(self) -> Panel:
        if not self.active_agent:
            return Panel("Waiting for assignment...", title="🕵️ Active Agent", border_style="white")
            
        content = f"[b]Agent:[/b] [cyan]{self.active_agent}[/cyan]\n"
        content += f"[b]Status:[/b] {self.agent_status}\n"
        
        if self.active_tool:
            content += f"\n[b]🔨 Executing Tool:[/b] [bold red]{self.active_tool}[/bold red]\n"
            
        content += "\n[b]💭 Live Thought Stream:[/b]\n"
        content += f"[italic white]{self.last_agent_thought}[/italic white]"
        
        return Panel(content, title="🕵️ Agent Live View", border_style="green")

    def _generate_findings(self) -> Panel:
        table = Table(show_header=True, expand=True, box=box.SIMPLE)
        table.add_column("Sev", width=4)
        table.add_column("Finding")
        
        # Pull from KB
        kb_data = self.kb.findings # Direct access for speed, assumes thread safety
        
        severity_colors = {
            "critical": "red",
            "high": "orange1",
            "medium": "yellow",
            "low": "blue",
            "info": "white"
        }
        
        for f in kb_data[-10:]:
            color = severity_colors.get(f.severity.lower(), "white")
            table.add_row(
                f"[{color}]{f.severity[:1].upper()}[/{color}]",
                f"{f.title}"
            )
            
        return Panel(table, title="🚩 Critical Findings", border_style="red")

    def _generate_assets(self) -> Panel:
        content = ""
        for asset in self.kb.assets[-10:]:
            content += f"🖥️ {asset['ip']} ({asset.get('hostname', '')})\n"
        for cred in self.kb.credentials[-5:]:
            content += f"🔑 {cred['username']} @ {cred['service']}\n"
            
        return Panel(content, title="📂 Assets & Creds", border_style="yellow")

    def _generate_footer(self) -> Panel:
        log_text = "\n".join(self.logs)
        return Panel(log_text, title="📜 System Logs", border_style="white")

    def start(self):
        self.running = True
        
        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            while self.running:
                # Update Layout
                self.layout["header"].update(self._generate_header())
                self.layout["left"]["orchestrator"].update(self._generate_orchestrator())
                self.layout["left"]["plan"].update(self._generate_plan())
                self.layout["center"].update(self._generate_agent_view())
                self.layout["right"]["findings"].update(self._generate_findings())
                self.layout["right"]["assets"].update(self._generate_assets())
                self.layout["footer"].update(self._generate_footer())
                
                # Check for updates or sleep
                try:
                    _ = self.update_queue.get(timeout=0.25)
                except queue.Empty:
                    pass

    def stop(self):
        self.running = False

# Global instance for easy import
_dashboard = None

def get_dashboard():
    global _dashboard
    if _dashboard is None:
        _dashboard = MissionDashboard()
    return _dashboard
