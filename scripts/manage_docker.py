#!/usr/bin/env python3
"""
Docker Infrastructure Manager
Start, stop, and manage Docker containers for AI Autonom
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"


def run_docker_compose(command: list, capture=False):
    """Run docker-compose command"""
    cmd = ["docker-compose", "-f", str(DOCKER_DIR / "docker-compose.yml")] + command
    
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=DOCKER_DIR)
        return result.returncode == 0, result.stdout + result.stderr
    else:
        result = subprocess.run(cmd, cwd=DOCKER_DIR)
        return result.returncode == 0, ""


def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def start_containers(services=None, build=False):
    """Start Docker containers"""
    print("\n" + "="*60)
    print("STARTING AI AUTONOM CONTAINERS")
    print("="*60 + "\n")
    
    if not check_docker():
        print("ERROR: Docker is not running or not installed")
        print("Please start Docker Desktop and try again")
        return False
    
    cmd = ["up", "-d"]
    if build:
        cmd.append("--build")
    if services:
        cmd.extend(services)
    
    success, output = run_docker_compose(cmd)
    
    if success:
        print("\nContainers started successfully!")
        print("\nAvailable services:")
        print("  - sandbox:   Python execution environment")
        print("  - security:  Security analysis tools")
        print("  - web:       Web scraping & automation")
        print("  - nodejs:    JavaScript/TypeScript execution")
        print("  - chromadb:  Vector database (port 8000)")
        print("  - redis:     Cache & message queue (port 6379)")
        print("  - portainer: Container management UI (port 9000)")
        return True
    else:
        print(f"\nFailed to start containers")
        return False


def stop_containers(services=None):
    """Stop Docker containers"""
    print("\n" + "="*60)
    print("STOPPING AI AUTONOM CONTAINERS")
    print("="*60 + "\n")
    
    cmd = ["down"]
    if services:
        cmd = ["stop"] + services
    
    success, _ = run_docker_compose(cmd)
    
    if success:
        print("Containers stopped successfully!")
    return success


def status():
    """Show container status"""
    print("\n" + "="*60)
    print("AI AUTONOM CONTAINER STATUS")
    print("="*60 + "\n")
    
    success, output = run_docker_compose(["ps"], capture=True)
    print(output)
    
    return success


def logs(service=None, follow=False):
    """Show container logs"""
    cmd = ["logs"]
    if follow:
        cmd.append("-f")
    if service:
        cmd.append(service)
    
    run_docker_compose(cmd)


def build_containers(services=None):
    """Build Docker images"""
    print("\n" + "="*60)
    print("BUILDING AI AUTONOM CONTAINERS")
    print("="*60 + "\n")
    
    cmd = ["build"]
    if services:
        cmd.extend(services)
    
    success, _ = run_docker_compose(cmd)
    
    if success:
        print("\nBuild completed successfully!")
    return success


def exec_command(service: str, command: str):
    """Execute command in container"""
    cmd = ["exec", service] + command.split()
    run_docker_compose(cmd)


def shell(service: str):
    """Open shell in container"""
    cmd = ["exec", service, "bash"]
    subprocess.run(
        ["docker-compose", "-f", str(DOCKER_DIR / "docker-compose.yml")] + cmd,
        cwd=DOCKER_DIR
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI Autonom Docker Infrastructure Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_docker.py start              # Start all containers
  python manage_docker.py start --build      # Build and start
  python manage_docker.py start sandbox web  # Start specific containers
  python manage_docker.py stop               # Stop all containers
  python manage_docker.py status             # Show status
  python manage_docker.py logs sandbox       # Show sandbox logs
  python manage_docker.py logs -f            # Follow all logs
  python manage_docker.py shell sandbox      # Open shell in sandbox
  python manage_docker.py exec sandbox "python --version"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start containers")
    start_parser.add_argument("services", nargs="*", help="Specific services to start")
    start_parser.add_argument("--build", "-b", action="store_true", help="Build images first")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop containers")
    stop_parser.add_argument("services", nargs="*", help="Specific services to stop")
    
    # Status command
    subparsers.add_parser("status", help="Show container status")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show logs")
    logs_parser.add_argument("service", nargs="?", help="Service name")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow logs")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build images")
    build_parser.add_argument("services", nargs="*", help="Specific services to build")
    
    # Exec command
    exec_parser = subparsers.add_parser("exec", help="Execute command in container")
    exec_parser.add_argument("service", help="Service name")
    exec_parser.add_argument("cmd", help="Command to execute")
    
    # Shell command
    shell_parser = subparsers.add_parser("shell", help="Open shell in container")
    shell_parser.add_argument("service", help="Service name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "start":
        start_containers(args.services if args.services else None, args.build)
    elif args.command == "stop":
        stop_containers(args.services if args.services else None)
    elif args.command == "status":
        status()
    elif args.command == "logs":
        logs(args.service, args.follow)
    elif args.command == "build":
        build_containers(args.services if args.services else None)
    elif args.command == "exec":
        exec_command(args.service, args.cmd)
    elif args.command == "shell":
        shell(args.service)


if __name__ == "__main__":
    main()
