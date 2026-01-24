"""
AI Autonom REPL (Read-Eval-Print Loop)
Standard, robust CLI interface.
"""

import sys
import os
import threading
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Using prompt_toolkit only for input handling (robust)
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.patch_stdout import patch_stdout

from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
from ai_autonom.memory.knowledge_base import KnowledgeBase

class AutonomREPL:
    def __init__(self):
        self.kb = KnowledgeBase.get_instance()
        
        # Initialize Orchestrator
        self.orchestrator = NemotronOrchestrator(
            enable_checkpoints=True,
            enable_testing=False,
            enable_dashboard=True # Enable simple text UI
        )
        
        # Commands
        self.commands = {
            "/help": "Show help",
            "/status": "Show system status",
            "/agents": "List available agents",
            "/tools": "List available tools",
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
            "/memory": None,
            "/quit": None,
            "/exit": None
        })

    def print_banner(self):
        print("\n" + "="*50)
        print(" AI AUTONOM - COMMAND CENTER")
        print(" Advanced Cyber-Operation Orchestrator")
        print("="*50)
        print(" Type your objective or /help for commands.\n")

    def handle_command(self, text: str):
        cmd = text.split()[0].lower()
        
        if cmd == "/help":
            print("\nAVAILABLE COMMANDS:")
            print("-" * 40)
            print(f"{ 'COMMAND':<15} | {'DESCRIPTION'}")
            print("-" * 40)
            for c, d in self.commands.items():
                print(f"{c:<15} | {d}")
            print()
            
        elif cmd == "/status":
            status = self.orchestrator.get_status()
            print("\nSYSTEM STATUS:")
            print(str(status))
            print()
            
        elif cmd == "/agents":
            agents = self.orchestrator.registry.get_all_agents()
            print("\nACTIVE AGENTS:")
            print("-" * 60)
            print(f"{ 'ID':<20} | {'NAME':<20} | {'CAPABILITIES'}")
            print("-" * 60)
            for a in agents:
                caps = ", ".join(a.capabilities[:2])
                print(f"{a.id:<20} | {a.name:<20} | {caps}...")
            print()
            
        elif cmd == "/tools":
            tools = self.orchestrator.tool_executor.get_available_tools()
            print(f"\nAVAILABLE TOOLS: {len(tools)}")
            
        elif cmd == "/memory":
            summary = self.kb.get_summary()
            print("\n🧠 KNOWLEDGE BASE:")
            print("-" * 40)
            print(summary)
            print("-" * 40 + "\n")
            
        elif cmd in ["/quit", "/exit"]:
            print("Goodbye.")
            sys.exit(0)
            
        else:
            print(f"Unknown command: {cmd}")

    def run(self):
        self.print_banner()
        session = PromptSession()
        
        while True:
            try:
                # Use patch_stdout to handle background threads printing gracefully
                with patch_stdout():
                    text = session.prompt('Autonom> ', completer=self.completer).strip()
                
                if not text:
                    continue
                    
                if text.startswith("/"):
                    self.handle_command(text)
                else:
                    self.orchestrator.run(text)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    repl = AutonomREPL()
    repl.run()