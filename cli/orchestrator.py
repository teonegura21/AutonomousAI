#!/usr/bin/env python3
"""
AI Autonom - Multi-Agent Orchestration System CLI
Extended CLI with full feature support including rich formatting
"""

import argparse
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import rich for better CLI formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

def print_banner():
    """Print startup banner"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]AI AUTONOM[/bold cyan]\n"
            "[white]Multi-Agent Orchestration System[/white]\n\n"
            "[yellow]Architecture:[/yellow]\n"
            "  • Nemotron: Orchestrator (plans & assigns)\n"
            "  • Qwen3-1.7B: Coding tasks\n"
            "  • DictaLM-1.7B: Documentation & text",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*70)
        print("AI AUTONOM - Multi-Agent Orchestration System")
        print("="*70)
        print("\nArchitecture:")
        print("  - Nemotron: Orchestrator (plans & assigns)")
        print("  - Qwen3-1.7B: Coding tasks")
        print("  - DictaLM-1.7B: Documentation & text")
        print("="*70 + "\n")


def print_success(message):
    """Print success message"""
    if RICH_AVAILABLE:
        console.print(f"[green][OK][/green] {message}")
    else:
        print(f"[OK] {message}")


def print_error(message):
    """Print error message"""
    if RICH_AVAILABLE:
        console.print(f"[red][ERROR][/red] {message}")
    else:
        print(f"[ERROR] {message}")


def print_info(message):
    """Print info message"""
    if RICH_AVAILABLE:
        console.print(f"[blue]ℹ[/blue] {message}")
    else:
        print(f"ℹ {message}")


def cmd_run(args):
    """Run orchestrator with a goal"""
    from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
    
    print_banner()
    
    if RICH_AVAILABLE:
        with console.status("[bold green]Initializing orchestrator...") as status:
            orchestrator = NemotronOrchestrator(
                orchestrator_model=args.model,
                config_path=args.config,
                enable_checkpoints=args.checkpoints,
                enable_testing=args.testing,
                enable_dashboard=True
            )
    else:
        print("Initializing orchestrator...")
        orchestrator = NemotronOrchestrator(
            orchestrator_model=args.model,
            config_path=args.config,
            enable_checkpoints=args.checkpoints,
            enable_testing=args.testing,
            enable_dashboard=True
        )
    
    if args.goal:
        goal = args.goal
    else:
        print("\nEnter your goal:")
        goal = input("> ").strip()
    
    if not goal:
        print_error("No goal provided. Exiting.")
        return
    
    print_info(f"Goal: {goal}")
    
    result = orchestrator.run(goal)
    
    if result.get("success"):
        print_success("Orchestration completed successfully!")
        
        # Notify about full log and artifacts
        session_id = result.get("workflow_id", "unknown")
        session_dir = f".runtime/sessions/{session_id}" # Approximation, actual path is managed by SessionManager but typically mapped here or printed above
        
        print_info(f"Artifacts generated for this prompt:")
        print_info(f"  1. Full Execution Log (Thinking + Output): {session_dir}/full_orchestration_log.txt")
        print_info(f"  2. Nemotron Plan (JSON): {session_dir}/nemotron_plan.json")
        print_info(f"Agents executed entirely within Kali Linux container.")
    else:
        print_error(f"Orchestration failed: {result.get('error', 'Unknown error')}")


def cmd_web(args):
    """Launch the web UI server"""
    from multi_agent_framework.ui.web.backend import run

    host = args.host or "127.0.0.1"
    port = args.port or 8000
    print_info(f"Starting web UI at http://{host}:{port}")
    print_info("Start the React UI with: cd multi_agent_framework/ui/web/frontend && npm install && npm run dev")
    run(host=host, port=port)


def cmd_scan_models(args):
    """Scan for new Ollama models"""
    from ai_autonom.core.model_discovery import ModelDiscovery
    
    print("\nScanning for Ollama models...\n")
    
    discovery = ModelDiscovery()
    
    # Get all Ollama models
    models = discovery.scan_ollama_models()
    print(f"Found {len(models)} models in Ollama:\n")
    
    for m in models:
        print(f"  - {m['name']}")
    
    # Check for unregistered
    print("\nChecking for unregistered models...")
    new_models = discovery.discover_new_models()
    
    if new_models:
        print(f"\nFound {len(new_models)} new models:\n")
        for m in new_models:
            print(f"  - {m['name']}")
        
        if args.register:
            print("\nBenchmarking and registering new models...")
            for m in new_models:
                print(f"\nProcessing: {m['name']}")
                caps = discovery.auto_register_model(m['name'], skip_benchmark=args.skip_benchmark)
                print(f"  Coding: {caps.get('coding_score', 0):.1f}")
                print(f"  Speed: {caps.get('speed_tokens_sec', 0):.1f} t/s")
    else:
        print("\nAll models are already registered.")


def cmd_list_models(args):
    """List registered models with scores"""
    from ai_autonom.core.model_selector import DynamicModelSelector
    
    selector = DynamicModelSelector()
    rankings = selector.get_model_rankings(args.task_type or "balanced")
    
    if not rankings:
        print_error("No models registered. Run: python run_orchestrator.py --scan-models --register")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="Registered Models", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Model", style="green", width=40)
        table.add_column("Score", style="yellow", width=8)
        table.add_column("VRAM", style="blue", width=8)
        table.add_column("Speed", style="red", width=10)
        
        for r in rankings:
            table.add_row(
                str(r['rank']),
                r['model_name'],
                f"{r['composite_score']:.1f}",
                f"{r['vram_gb']:.1f}",
                f"{r['speed_tokens_sec']:.1f}"
            )
        
        console.print(table)
    else:
        print("\nRegistered Models:\n")
        print(f"{'Rank':<6}{'Model':<40}{'Score':<8}{'VRAM':<8}{'Speed':<10}")
        print("-" * 72)
        
        for r in rankings:
            print(f"{r['rank']:<6}{r['model_name']:<40}{r['composite_score']:<8.1f}{r['vram_gb']:<8.1f}{r['speed_tokens_sec']:<10.1f}")


def cmd_benchmark(args):
    """Benchmark a specific model"""
    from ai_autonom.core.model_discovery import ModelDiscovery
    
    print(f"\nBenchmarking: {args.model}\n")
    
    discovery = ModelDiscovery()
    caps = discovery.assess_capabilities(args.model)
    
    print("\nResults:")
    print(f"  Coding Score: {caps.get('coding_score', 0):.1f}/100")
    print(f"  Reasoning Score: {caps.get('reasoning_score', 0):.1f}/100")
    print(f"  Documentation Score: {caps.get('documentation_score', 0):.1f}/100")
    print(f"  Speed: {caps.get('speed_tokens_sec', 0):.1f} tokens/sec")
    print(f"  Est. VRAM: {caps.get('vram_gb', 0):.1f} GB")
    
    if args.save:
        discovery.register_model(caps)
        print("\nSaved to registry.")
def cmd_list_agents(args):
    """List registered agents"""
    from ai_autonom.core.agent_registry import AgentRegistry
    
    registry = AgentRegistry()
    agents = registry.get_all_agents()
    
    if not agents:
        print_error("No agents registered.")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="Registered Agents", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=16)
        table.add_column("Name", style="green", width=26)
        table.add_column("Model", style="yellow", width=24)
        table.add_column("Provider", style="blue", width=10)
        table.add_column("VRAM", style="magenta", width=8)
        table.add_column("Quality", style="white", width=8)
        
        for a in agents:
            table.add_row(
                a.id,
                a.name[:24] + "..." if len(a.name) > 26 else a.name,
                a.model_name[:22] + "..." if len(a.model_name) > 24 else a.model_name,
                getattr(a, "provider", "ollama"),
                f"{a.vram_required:.1f}",
                f"{a.quality_score:.1f}",
            )
        console.print(table)
    else:
        print("\nRegistered Agents:\n")
        print(f"{'ID':<18}{'Name':<28}{'Model':<26}{'Provider':<10}{'VRAM':<8}{'Quality':<8}")
        print("-" * 96)
        for a in agents:
            print(
                f"{a.id:<18}{a.name:<28}{a.model_name:<26}{getattr(a, 'provider', 'ollama'):<10}{a.vram_required:<8.1f}{a.quality_score:<8.1f}"
            )


def cmd_status(args):
    """Show system status"""
    from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
    
    print("\nSystem Status:\n")
    
    try:
        orchestrator = NemotronOrchestrator()
        status = orchestrator.get_status()
        
        print(f"Orchestrator Model: {status.get('orchestrator_model')}")
        print(f"Registered Agents: {status.get('registered_agents')}")
        print(f"Registered Tools: {status.get('registered_tools')}")
        
        watcher = status.get('model_watcher', {})
        print(f"\nModel Watcher:")
        print(f"  Running: {watcher.get('running', False)}")
        print(f"  Last Scan: {watcher.get('last_scan', 'Never')}")
        print(f"  Discovered: {watcher.get('total_discovered', 0)}")
        
        monitor = status.get('monitor', {})
        print(f"\nExecution Stats:")
        print(f"  Completed: {monitor.get('tasks_completed', 0)}")
        print(f"  Successful: {monitor.get('tasks_successful', 0)}")
        print(f"  Failed: {monitor.get('tasks_failed', 0)}")
        
    except Exception as e:
        print(f"Error getting status: {e}")


def cmd_list_tools(args):
    """List available tools"""
    from ai_autonom.tools.tool_executor import ToolExecutor
    
    executor = ToolExecutor()
    tools = executor.get_available_tools()
    
    if RICH_AVAILABLE:
        table = Table(title="Available Tools", show_header=True, header_style="bold magenta")
        table.add_column("Tool ID", style="cyan", width=25)
        table.add_column("Category", style="green", width=15)
        table.add_column("Description", style="white", width=40)
        table.add_column("Sandbox", style="yellow", width=10)
        
        for tool in sorted(tools, key=lambda t: (t['category'], t['id'])):
            sandbox = "Yes" if tool['requires_sandbox'] else "No"
            table.add_row(
                tool['id'],
                tool['category'],
                tool['description'][:37] + "..." if len(tool['description']) > 40 else tool['description'],
                sandbox
            )
        
        console.print(table)
    else:
        print("\nAvailable Tools:\n")
        
        current_category = None
        for tool in sorted(tools, key=lambda t: (t['category'], t['id'])):
            if tool['category'] != current_category:
                current_category = tool['category']
                print(f"\n[{current_category.upper()}]")
            
            sandbox = " [SANDBOX]" if tool['requires_sandbox'] else ""
            print(f"  {tool['id']}: {tool['description']}{sandbox}")


def cmd_query_memory(args):
    """Query vector memory"""
    from ai_autonom.memory.vector_store import VectorMemoryStore
    
    print(f"\nQuerying: {args.query}\n")
    
    store = VectorMemoryStore()
    answer = store.query_natural(args.query)
    
    print(answer)


def cmd_clear_memory(args):
    """Clear memory stores"""
    if not args.confirm:
        response = input("Are you sure you want to clear memory? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
    
    if args.type in ("all", "vector"):
        from ai_autonom.memory.vector_store import VectorMemoryStore
        store = VectorMemoryStore()
        store.clear_all()
        print("Vector memory cleared.")
    
    if args.type in ("all", "task"):
        from ai_autonom.memory.task_memory import TaskMemory
        memory = TaskMemory()
        # Clear would need workflow_id, so just notify
        print("Task memory: clear specific workflow with workflow ID")


def cmd_export_report(args):
    """Export execution report"""
    from ai_autonom.monitoring.telemetry import ExecutionMonitor
    
    monitor = ExecutionMonitor()
    filepath = args.output or f"reports/execution_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if monitor.export_report(filepath):
        print(f"Report exported to: {filepath}")
    else:
        print("Failed to export report.")


def cmd_kali_status(args):
    """Show Kali Linux container status and open terminal"""
    from ai_autonom.sandbox.kali_monitor import KaliContainerMonitor
    
    monitor = KaliContainerMonitor()
    status = monitor.get_status()
    
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich.panel import Panel
        
        if status.get("running"):
            console.print(Panel.fit(
                f"[green]KALI LINUX CONTAINER[/green]\n\n"
                f"Status: [green]RUNNING[/green]\n"
                f"Container ID: {status.get('id', 'N/A')}\n"
                f"CPU Usage: {status.get('cpu_percent', 0):.1f}%\n"
                f"Memory: {status.get('memory_mb', 0):.1f} MB\n\n"
                f"[cyan]Web Terminal:[/cyan] {status.get('web_terminal_url')}\n"
                f"[yellow]Open in browser for live visibility![/yellow]",
                border_style="green"
            ))
            
            if args.open_terminal:
                monitor.open_web_terminal()
        else:
            console.print(Panel.fit(
                f"[red]KALI LINUX CONTAINER[/red]\n\n"
                f"Status: [red]NOT RUNNING[/red]\n\n"
                f"To start:\n  cd docker && docker-compose up -d kali",
                border_style="red"
            ))
    else:
        print("\n" + "="*60)
        print("KALI LINUX CONTAINER STATUS")
        print("="*60)
        
        if status.get("running"):
            print(f"\nStatus: RUNNING")
            print(f"Container ID: {status.get('id', 'N/A')}")
            print(f"Web Terminal: {status.get('web_terminal_url')}")
            if args.open_terminal:
                monitor.open_web_terminal()
        else:
            print(f"\nStatus: NOT RUNNING")
            print("\nTo start: cd docker && docker-compose up -d kali")
        
        print("="*60)


def cmd_kali_logs(args):
    """Show Kali container logs"""
    from ai_autonom.sandbox.kali_monitor import KaliContainerMonitor
    
    monitor = KaliContainerMonitor()
    
    if args.log_type == "commands":
        print("\n=== Command Execution Log ===")
        print(monitor.get_command_log(lines=args.lines))
    elif args.log_type == "output":
        print("\n=== Output Log ===")
        print(monitor.get_output_log(lines=args.lines))
    elif args.log_type == "findings":
        print("\n=== Security Findings ===")
        print(monitor.get_findings())
    else:
        print("\n=== Container Logs ===")
        print(monitor.get_logs(lines=args.lines))


def cmd_kali_exec(args):
    """Execute command in Kali container"""
    from ai_autonom.sandbox.kali_monitor import KaliContainerMonitor
    
    monitor = KaliContainerMonitor()
    
    if not monitor.is_running():
        print_error("Kali container not running. Start with: cd docker && docker-compose up -d kali")
        return
    
    command = " ".join(args.command)
    print(f"\nExecuting in Kali: {command}")
    print("-" * 60)
    
    result = monitor.execute_command(command)
    
    if result.get("success"):
        print(result.get("output", ""))
    else:
        print_error(result.get("error", "Unknown error"))
        if result.get("output"):
            print(result["output"])


def cmd_interactive(args):
    """Interactive mode"""
    try:
        from ai_autonom.cli.repl import AutonomREPL
        repl = AutonomREPL()
        repl.run()
    except ImportError as e:
        print(f"Error loading REPL: {e}")
        # Fallback to old interactive mode if needed
        from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
        orchestrator = NemotronOrchestrator()
        print("Fallback interactive mode...")
        # ... (rest of old logic logic)
    print("Commands: /status, /models, /tools, /kali, /quit\n")
    
    while True:
        try:
            user_input = input("Goal> ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                cmd = user_input[1:].lower()
                if cmd == "quit" or cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "status":
                    status = orchestrator.get_status()
                    print(f"Agents: {status.get('registered_agents')}, Tools: {status.get('registered_tools')}")
                elif cmd == "models":
                    for a in orchestrator.registry.get_all_agents():
                        print(f"  - {a.name} ({a.model_name})")
                elif cmd == "tools":
                    tools = orchestrator.tool_executor.get_available_tools()
                    for t in tools[:10]:
                        print(f"  - {t['id']}")
                elif cmd == "kali":
                    cmd_kali_status(args)
                else:
                    print(f"Unknown command: {cmd}")
            else:
                orchestrator.run(user_input)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


def cmd_simple_tui(args):
    """Legacy simple TUI (debug only)"""
    import threading
    import time
    import json

    from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
    from ai_autonom.monitoring.simple_tui import get_simple_tui
    from ai_autonom.monitoring.simple_tui_prompt import get_simple_tui_command

    orchestrator = NemotronOrchestrator(
        orchestrator_model=args.model,
        config_path=args.config,
        enable_checkpoints=args.checkpoints,
        enable_testing=args.testing,
        enable_dashboard=False  # Disable rich dashboard
    )

    tui = get_simple_tui()
    tui.running = True

    tui.update_goal("Waiting for goal...")
    tui.set_orchestrator_status("Idle")
    tui.set_command_result("Type: goal <text> to start")

    run_thread = None

    while True:
        # Print simple status
        tui.print_status()

        # If a run is active, check status
        if run_thread and run_thread.is_alive():
            time.sleep(1)
            continue
        elif run_thread and not run_thread.is_alive():
            run_thread = None
            tui.set_orchestrator_status("Idle")
            tui.set_command_result("Task completed")
            continue

        # Get command with simple input
        cmd = get_simple_tui_command()

        if not cmd:
            continue

        cmd_str = cmd.strip()
        cmd_lower = cmd_str.lower()

        if cmd_lower.startswith("goal ") or cmd_lower.startswith("start "):
            goal = cmd_str.split(" ", 1)[1].strip()
            if not goal:
                tui.set_command_result("No goal provided")
                continue
            tui.update_goal(goal)
            tui.set_orchestrator_status("Running")
            run_thread = threading.Thread(target=orchestrator.run, args=(goal,), daemon=True)
            run_thread.start()
            tui.set_command_result("Goal started")
            continue

        if cmd_lower in ("quit", "exit"):
            if run_thread and run_thread.is_alive():
                tui.set_command_result("Stop workflow before quitting")
                continue
            break

        if cmd_lower == "status":
            status = orchestrator.get_status()
            summary = {
                "orchestrator_model": status.get("orchestrator_model"),
                "registered_agents": status.get("registered_agents"),
                "registered_tools": status.get("registered_tools"),
            }
            tui.set_command_result(str(summary))
            continue

        if cmd_lower == "agents":
            agents = orchestrator.registry.get_all_agents()
            agent_names = [a.id for a in agents]
            tui.set_command_result(f"Agents: {', '.join(agent_names[:10])}")
            continue

        if cmd_lower == "tools":
            tools = orchestrator.tool_executor.get_available_tools()
            tui.set_command_result(f"Tools: {len(tools)} available")
            continue

        tui.set_command_result("Unknown command")


def cmd_tui(args):
    """Legacy full-screen TUI (debug only)"""
    import threading
    import time
    import json

    from ai_autonom.orchestration.nemotron_orchestrator import NemotronOrchestrator
    from ai_autonom.monitoring.simple_tui import get_simple_tui
    from ai_autonom.monitoring.simple_tui_prompt import get_simple_tui_command

    orchestrator = NemotronOrchestrator(
        orchestrator_model=args.model,
        config_path=args.config,
        enable_checkpoints=args.checkpoints,
        enable_testing=args.testing,
        enable_dashboard=False  # Disable rich dashboard
    )

    tui = get_simple_tui()
    tui.running = True

    tui.update_goal("Waiting for goal...")
    tui.set_orchestrator_status("Idle")
    tui.set_command_result("Type: goal <text> to start")

    run_thread = None

    while True:
        # Print simple status
        tui.print_status()

        # If a run is active, check status
        if run_thread and run_thread.is_alive():
            time.sleep(1)
            continue
        elif run_thread and not run_thread.is_alive():
            run_thread = None
            tui.set_orchestrator_status("Idle")
            tui.set_command_result("Task completed")
            continue

        # Get command with simple input
        cmd = get_simple_tui_command()

        if not cmd:
            continue

        cmd_str = cmd.strip()
        cmd_lower = cmd_str.lower()

        if cmd_lower.startswith("goal ") or cmd_lower.startswith("start "):
            goal = cmd_str.split(" ", 1)[1].strip()
            if not goal:
                tui.set_command_result("No goal provided")
                continue
            tui.update_goal(goal)
            tui.set_orchestrator_status("Running")
            run_thread = threading.Thread(target=orchestrator.run, args=(goal,), daemon=True)
            run_thread.start()
            tui.set_command_result("Goal started")
            continue

        if cmd_lower in ("quit", "exit"):
            if run_thread and run_thread.is_alive():
                tui.set_command_result("Stop workflow before quitting")
                continue
            break

        if cmd_lower == "status":
            status = orchestrator.get_status()
            summary = {
                "orchestrator_model": status.get("orchestrator_model"),
                "registered_agents": status.get("registered_agents"),
                "registered_tools": status.get("registered_tools"),
            }
            tui.set_command_result(str(summary))
            continue

        if cmd_lower == "agents":
            agents = orchestrator.registry.get_all_agents()
            agent_names = [a.id for a in agents]
            tui.set_command_result(f"Agents: {', '.join(agent_names[:10])}")
            continue

        if cmd_lower == "tools":
            tools = orchestrator.tool_executor.get_available_tools()
            tui.set_command_result(f"Tools: {len(tools)} available")
            continue

        tui.set_command_result("Unknown command")


def main():
    parser = argparse.ArgumentParser(
        description="AI Autonom - Multi-Agent Orchestration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_orchestrator.py "Write a Python function to validate emails"
  python run_orchestrator.py --web                 # Launch web UI
  python run_orchestrator.py --simple-tui          # Debug-only legacy simple TUI
  python run_orchestrator.py --scan-models --register
  python run_orchestrator.py --list-models
  python run_orchestrator.py --benchmark qwen2.5-coder:7b
  python run_orchestrator.py --interactive
        """
    )
    
    # Mode selection
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--tui", action="store_true", help="Legacy full-screen TUI (debug only)")
    parser.add_argument("--simple-tui", action="store_true", help="Legacy simple TUI (debug only)")
    parser.add_argument("--web", action="store_true", help="Launch web UI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web UI host")
    parser.add_argument("--port", type=int, default=8000, help="Web UI port")
    
    # Model management
    parser.add_argument("--scan-models", action="store_true", help="Scan for new Ollama models")
    parser.add_argument("--list-models", action="store_true", help="List registered models")
    parser.add_argument("--benchmark", type=str, metavar="MODEL", help="Benchmark specific model")
    parser.add_argument("--register", action="store_true", help="Register discovered models")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmarking when registering")
    parser.add_argument("--task-type", type=str, choices=["coding", "reasoning", "documentation", "fast", "balanced"], help="Task type for ranking")
    parser.add_argument("--save", action="store_true", help="Save benchmark results")
    
    # Status and info
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    parser.add_argument("--list-agents", action="store_true", help="List registered agents")
    
    # Memory management
    parser.add_argument("--query", type=str, metavar="QUERY", help="Query vector memory")
    parser.add_argument("--clear-memory", action="store_true", help="Clear memory stores")
    parser.add_argument("--memory-type", type=str, choices=["all", "vector", "task"], default="all", dest="type", help="Memory type to clear")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation for destructive operations")
    
    # Reports
    parser.add_argument("--export-report", action="store_true", help="Export execution report")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    
    # Kali Linux container
    parser.add_argument("--kali", action="store_true", help="Show Kali container status")
    parser.add_argument("--kali-logs", action="store_true", help="Show Kali container logs")
    parser.add_argument("--kali-exec", nargs="*", metavar="CMD", dest="command", help="Execute command in Kali")
    parser.add_argument("--open-terminal", action="store_true", help="Open web terminal in browser")
    parser.add_argument("--log-type", type=str, choices=["all", "commands", "output", "findings"], default="all", help="Log type to show")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines to show")
    
    # Execution options
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file path")
    parser.add_argument("--checkpoints", action="store_true", help="Enable human checkpoints")
    parser.add_argument("--testing", action="store_true", help="Enable automatic testing")
    parser.add_argument("--model", type=str, help="Override orchestrator model")
    
    args = parser.parse_args()
    
    if not RICH_AVAILABLE:
        print("[WARNING] 'rich' library not found. TUI disabled. Install with: pip install rich")
    else:
        print("[INFO] 'rich' library loaded. TUI enabled during execution.")
    
    # Route to appropriate command
    try:
        if args.scan_models:
            cmd_scan_models(args)
        elif args.list_models:
            cmd_list_models(args)
        elif args.benchmark:
            args.model = args.benchmark
            cmd_benchmark(args)
        elif args.status:
            cmd_status(args)
        elif args.list_tools:
            cmd_list_tools(args)
        elif args.list_agents:
            cmd_list_agents(args)
        elif args.query:
            cmd_query_memory(args)
        elif args.clear_memory:
            cmd_clear_memory(args)
        elif args.export_report:
            cmd_export_report(args)
        elif args.kali:
            cmd_kali_status(args)
        elif args.kali_logs:
            cmd_kali_logs(args)
        elif args.command is not None:  # --kali-exec
            cmd_kali_exec(args)
        elif args.web:
            cmd_web(args)
        elif args.interactive:
            cmd_interactive(args)
        elif args.simple_tui:
            cmd_simple_tui(args)
        elif args.tui:
            cmd_tui(args)
        elif args.goal:
            cmd_run(args)
        else:
            # Default: launch web UI
            cmd_web(args)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
