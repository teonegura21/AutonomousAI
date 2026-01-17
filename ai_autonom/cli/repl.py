
"""
AI Autonom REPL (Read-Eval-Print Loop)
A rich, interactive CLI experience inspired by CAI.
Features:
- Persistent Status Bar
- Autocompletion
- Syntax Highlighting
- Rich Output
"""

import sys
import os
import threading
from typing import List, Optional

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.completion import WordCompleter, NestedCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
from ai_autonom.memory.knowledge_base import KnowledgeBase

class AutonomREPL:
    def __init__(self):
        self.console = Console()
        self.kb = KnowledgeBase.get_instance()
        
        # Initialize Orchestrator
        self.orchestrator = NemotronOrchestrator(
            enable_checkpoints=True,
            enable_testing=False,
            enable_dashboard=False # Disable auto-popup, we control UI here
        )
        
        # Commands
        self.commands = {
            "/help": "Show help",
            "/status": "Show system status",
            "/agents": "List available agents",
            "/tools": "List available tools",
            "/clear": "Clear screen",
            "/memory": "Show Knowledge Base (The Blackboard)",
            "/quit": "Exit",
            "/exit": "Exit"
        }
        
        # Autocompleter
        self.completer = NestedCompleter.from_nested_dict({
            "/help": None,
            "/status": None,
            "/agents": None,
            "/tools": None,
            "/clear": None,
            "/memory": None,
            "/quit": None,
            "/exit": None,
            "scan": {"target": None},
            "analyze": None
        })
        
        # Style
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'bottom-toolbar': '#ffffff bg:#333333',
        })

    def get_bottom_toolbar(self):
        """Dynamic bottom toolbar"""
        agents_count = len(self.orchestrator.registry.get_all_agents())
        kb_findings = len(self.kb.findings)
        assets = len(self.kb.assets)
        return HTML(f' <b>AI Autonom</b> | Agents: {agents_count} | Assets: {assets} | Findings: {kb_findings} | <b>/help</b> for commands')

    def print_banner(self):
        self.console.print(Panel(
            "[bold cyan]AI AUTONOM[/bold cyan]\n"
            "[dim]Advanced Cyber-Operation Orchestrator[/dim]\n\n"
            "Type your objective or use [bold]/help[/bold] for commands.",
            box=box.DOUBLE,
            border_style="cyan"
        ))

    def handle_command(self, text: str):
        cmd = text.split()[0].lower()
        
        if cmd == "/help":
            table = Table(box=box.SIMPLE)
            table.add_column("Command", style="cyan")
            table.add_column("Description")
            for c, d in self.commands.items():
                table.add_row(c, d)
            self.console.print(table)
            
        elif cmd == "/status":
            status = self.orchestrator.get_status()
            self.console.print(Panel(str(status), title="System Status"))
            
        elif cmd == "/agents":
            agents = self.orchestrator.registry.get_all_agents()
            table = Table(title="Active Agents", box=box.ROUNDED)
            table.add_column("ID", style="green")
            table.add_column("Name")
            table.add_column("Capabilities", style="dim")
            for a in agents:
                table.add_row(a.id, a.name, ", ".join(a.capabilities[:3]))
            self.console.print(table)
            
        elif cmd == "/tools":
            tools = self.orchestrator.tool_executor.get_available_tools()
            self.console.print(f"[bold]Available Tools:[/bold] {len(tools)}")
            # Show summarized list
            categories = {}
            for t in tools:
                cat = t.get('category', 'misc')
                if cat not in categories: categories[cat] = []
                categories[cat].append(t['id'])
            
            for cat, t_list in categories.items():
                self.console.print(f"[yellow]{cat.upper()}[/yellow]: {', '.join(t_list)}")

        elif cmd == "/memory":
            summary = self.kb.get_summary()
            self.console.print(Panel(summary, title="🧠 Knowledge Base (Blackboard)", border_style="magenta"))
            
        elif cmd == "/clear":
            self.console.clear()
            self.print_banner()
            
        elif cmd in ["/quit", "/exit"]:
            sys.exit(0)
            
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")

    def run(self):
        self.print_banner()
        session = PromptSession()
        
        while True:
            try:
                # Use patch_stdout so background threads (dashboard logs) don't mess up the prompt
                with patch_stdout():
                    text = session.prompt(
                        'Autonom> ',
                        completer=self.completer,
                        style=self.style,
                        bottom_toolbar=self.get_bottom_toolbar
                    ).strip()
                
                if not text:
                    continue
                    
                if text.startswith("/"):
                    self.handle_command(text)
                else:
                    # Execute Goal
                    self.console.print(f"[bold green]Executing:[/bold green] {text}")
                    self.orchestrator.run(text)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    repl = AutonomREPL()
    repl.run()
