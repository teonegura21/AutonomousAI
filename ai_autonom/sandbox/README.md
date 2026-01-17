# Sandbox Module

## Purpose
The sandbox module provides isolated Docker-based execution environments for running untrusted code and security tools. It manages multiple specialized containers and ensures safe tool execution with resource limits and network isolation.

## Key Files

### `docker_executor.py`
**Purpose:** Execute commands and code in isolated Docker containers with security constraints.

**Key Classes:**
- `ContainerType`: Enum for container types
  - GENERAL: General-purpose sandbox
  - SECURITY: Security tools container
  - WEB: Web scraping and browser automation
  - CODE: Code execution and compilation
  - NODEJS: Node.js environment
  - PYTHON: Python environment

- `DockerExecutor`: Docker execution engine
  - `execute_in_container(container_type, command, timeout)`: Execute command in container
  - `execute_code(code, language, timeout)`: Execute code snippet
  - `start_container(container_type)`: Start container
  - `stop_container(container_id)`: Stop container
  - `list_running_containers()`: List active containers
  - `cleanup_containers()`: Stop and remove all containers
  - `get_container_logs(container_id)`: Retrieve container logs
  - `copy_file_to_container(container_id, src, dst)`: Copy file into container
  - `copy_file_from_container(container_id, src, dst)`: Extract file from container

**Container Configuration:**
```python
CONTAINER_CONFIGS = {
    ContainerType.GENERAL: {
        "image": "ubuntu:22.04",
        "memory_limit": "512m",
        "cpu_quota": 50000,  # 50% of one CPU
        "network_mode": "bridge",
        "capabilities": ["NET_RAW"]
    },
    ContainerType.SECURITY: {
        "image": "kalilinux/kali-rolling",
        "memory_limit": "1g",
        "cpu_quota": 100000,  # 100% of one CPU
        "network_mode": "bridge",
        "capabilities": ["NET_RAW", "NET_ADMIN"]
    },
    # ... other container types
}
```

**Dependencies:**
- External: `docker`, `logging`, `json`, `tempfile`
- Internal: `ai_autonom.core.config`

### `container_manager.py`
**Purpose:** Manage container lifecycle and resource allocation.

**Key Classes:**
- `ContainerStatus`: Enum for container states
  - CREATED: Container created but not started
  - RUNNING: Container actively running
  - STOPPED: Container stopped
  - PAUSED: Container paused
  - ERROR: Container in error state

- `ContainerInfo`: Container metadata
  - `container_id`: Unique container identifier
  - `type`: ContainerType
  - `status`: ContainerStatus
  - `created_at`: Creation timestamp
  - `resource_usage`: CPU/memory usage stats
  - `network_info`: Network configuration

- `ContainerManager`: High-level container management
  - `get_or_create_container(container_type)`: Get existing or create new container
  - `health_check(container_id)`: Check container health
  - `get_resource_usage(container_id)`: Get CPU/memory usage
  - `restart_container(container_id)`: Restart container
  - `pause_container(container_id)`: Pause container
  - `resume_container(container_id)`: Resume paused container
  - `get_all_containers()`: List all managed containers
  - `cleanup_idle_containers(idle_threshold)`: Remove idle containers

**Dependencies:**
- External: `docker`, `logging`, `datetime`, `psutil`
- Internal: `ai_autonom.sandbox.docker_executor`

## Internal Architecture

```
┌──────────────────────────────────────────┐
│       ContainerManager                    │
│  (Lifecycle management)                   │
│  - Container pooling                      │
│  - Health monitoring                      │
│  - Resource tracking                      │
│  - Idle cleanup                           │
└─────────────┬────────────────────────────┘
              │
              │ manages
              ▼
┌──────────────────────────────────────────┐
│       DockerExecutor                      │
│  (Command execution)                      │
│  - Container operations                   │
│  - Code execution                         │
│  - File transfer                          │
│  - Log retrieval                          │
└─────────────┬────────────────────────────┘
              │
              │ controls
              ▼
┌──────────────────────────────────────────┐
│     Docker Containers                     │
│  ┌────────────┐  ┌────────────┐          │
│  │  General   │  │  Security  │          │
│  │  Sandbox   │  │   Tools    │          │
│  └────────────┘  └────────────┘          │
│  ┌────────────┐  ┌────────────┐          │
│  │    Web     │  │   Python   │          │
│  │  Browser   │  │  Runtime   │          │
│  └────────────┘  └────────────┘          │
└──────────────────────────────────────────┘
```

## Usage Examples

### Docker Execution
```python
from ai_autonom.sandbox.docker_executor import DockerExecutor, ContainerType

# Initialize executor
executor = DockerExecutor()

# Execute command in security container
result = executor.execute_in_container(
    container_type=ContainerType.SECURITY,
    command="nmap -sV 192.168.1.1",
    timeout=60
)
print(result["stdout"])
print(result["exit_code"])

# Execute Python code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
result = executor.execute_code(
    code=code,
    language="python",
    timeout=5
)
print(result["stdout"])  # "55"

# Copy file to container
container_id = executor.start_container(ContainerType.GENERAL)
executor.copy_file_to_container(
    container_id=container_id,
    src="/local/path/script.py",
    dst="/tmp/script.py"
)

# Execute and get result
result = executor.execute_in_container(
    container_type=ContainerType.GENERAL,
    command="python /tmp/script.py",
    timeout=30
)

# Cleanup
executor.stop_container(container_id)
```

### Container Management
```python
from ai_autonom.sandbox.container_manager import ContainerManager, ContainerStatus

# Initialize manager
manager = ContainerManager()

# Get or create container (reuses existing if available)
container = manager.get_or_create_container(ContainerType.SECURITY)
print(container.container_id)
print(container.status)

# Health check
health = manager.health_check(container.container_id)
print(f"Healthy: {health}")

# Get resource usage
usage = manager.get_resource_usage(container.container_id)
print(f"CPU: {usage['cpu_percent']}%")
print(f"Memory: {usage['memory_usage_mb']} MB")

# Pause and resume
manager.pause_container(container.container_id)
# ... do something else ...
manager.resume_container(container.container_id)

# List all containers
containers = manager.get_all_containers()
for c in containers:
    print(f"{c.type}: {c.status}")

# Cleanup idle containers (stopped for >5 minutes)
cleaned = manager.cleanup_idle_containers(idle_threshold=300)
print(f"Removed {cleaned} idle containers")
```

### Multi-Container Workflow
```python
from ai_autonom.sandbox.docker_executor import DockerExecutor, ContainerType
from ai_autonom.sandbox.container_manager import ContainerManager

executor = DockerExecutor()
manager = ContainerManager()

# Security scan in security container
security_container = manager.get_or_create_container(ContainerType.SECURITY)
scan_result = executor.execute_in_container(
    container_type=ContainerType.SECURITY,
    command="nmap -p 80,443 example.com",
    timeout=60
)

# Process results in Python container
python_container = manager.get_or_create_container(ContainerType.PYTHON)
code = f"""
import json
scan_output = '''{scan_result["stdout"]}'''
# Process scan output
print(json.dumps({{"processed": True}}))
"""
process_result = executor.execute_code(
    code=code,
    language="python",
    timeout=10
)

# Cleanup all containers
executor.cleanup_containers()
```

## Dependencies

**External Dependencies:**
- `docker`: Docker SDK for Python
- `psutil`: Process and system resource monitoring
- `logging`: Event logging
- `json`: Data serialization
- `tempfile`: Temporary file handling
- `datetime`: Timestamp handling

**Internal Dependencies:**
- `ai_autonom.core.config`: Configuration for Docker settings

## Important Functionality

1. **Container Isolation**: Full process and filesystem isolation
2. **Resource Limits**: CPU and memory constraints per container
3. **Network Isolation**: Controlled network access per container type
4. **Timeout Enforcement**: Hard limits on execution time
5. **Multi-Language Support**: Execute code in Python, Node.js, Go, Rust, etc.
6. **Container Reuse**: Pool containers to avoid startup overhead
7. **Health Monitoring**: Automatic health checks and recovery
8. **Resource Tracking**: Real-time CPU/memory usage monitoring
9. **File Transfer**: Copy files in/out of containers
10. **Log Retrieval**: Access container logs for debugging
11. **Idle Cleanup**: Automatic removal of unused containers

## Container Types and Use Cases

| Container Type | Image | Use Case | Network | Capabilities |
|----------------|-------|----------|---------|--------------|
| GENERAL | ubuntu:22.04 | General commands | bridge | NET_RAW |
| SECURITY | kalilinux/kali-rolling | Security tools | bridge | NET_RAW, NET_ADMIN |
| WEB | selenium/standalone-chrome | Web scraping | bridge | - |
| CODE | python:3.11-slim | Code execution | none | - |
| NODEJS | node:18-alpine | Node.js code | bridge | - |
| PYTHON | python:3.11-slim | Python code | none | - |

## Security Considerations

1. **Resource Limits**: Prevent resource exhaustion
   ```python
   memory_limit="512m"  # 512MB RAM
   cpu_quota=50000      # 50% of one CPU core
   ```

2. **Network Isolation**: Limit network access
   ```python
   network_mode="none"  # No network access
   network_mode="bridge"  # Bridged network (controlled)
   ```

3. **Capability Restrictions**: Minimal Linux capabilities
   ```python
   capabilities=["NET_RAW"]  # Only raw socket access
   ```

4. **Filesystem Isolation**: Read-only root filesystem
   ```python
   read_only=True
   tmpfs={"/tmp": "rw,noexec,nosuid,size=100m"}
   ```

5. **Timeout Enforcement**: Hard execution time limits
   ```python
   timeout=30  # Kill after 30 seconds
   ```

6. **User Permissions**: Run as non-root user
   ```python
   user="nobody"
   ```

## Configuration

Set these in `.env`:
```
DOCKER_ENABLED=true
DOCKER_TIMEOUT=60
DOCKER_MEMORY_LIMIT=512m
DOCKER_CPU_QUOTA=50000
DOCKER_CLEANUP_INTERVAL=300
```

## Container Lifecycle

```
CREATED → RUNNING → STOPPED
    ↓         ↓         ↑
    └─────→ PAUSED ────┘
              ↓
           ERROR
```

## Resource Management

```python
# Example resource usage tracking
{
    "cpu_percent": 25.5,
    "memory_usage_mb": 128.3,
    "memory_limit_mb": 512.0,
    "memory_percent": 25.1,
    "network_rx_bytes": 1024,
    "network_tx_bytes": 2048,
    "block_io_read_bytes": 4096,
    "block_io_write_bytes": 8192
}
```

## Error Handling

```python
from ai_autonom.sandbox.docker_executor import DockerExecutor

executor = DockerExecutor()

try:
    result = executor.execute_code(code, "python", timeout=5)
except TimeoutError:
    # Code exceeded timeout
    print("Execution timed out")
except RuntimeError as e:
    # Container error
    print(f"Container error: {e}")
except docker.errors.DockerException as e:
    # Docker daemon error
    print(f"Docker error: {e}")
```

## Best Practices

1. **Reuse Containers**: Use `ContainerManager.get_or_create_container()` to avoid startup overhead
2. **Set Appropriate Timeouts**: Balance responsiveness vs. task completion
3. **Monitor Resources**: Track usage to prevent resource exhaustion
4. **Cleanup Idle Containers**: Periodically remove unused containers
5. **Use Specific Container Types**: Choose container type matching your workload
6. **Handle Errors Gracefully**: Always cleanup containers even on errors
7. **Test Locally First**: Verify commands work before deploying
