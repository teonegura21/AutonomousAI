# Monitoring Module

## Purpose
The monitoring module provides comprehensive telemetry, logging, and performance tracking for the multi-agent orchestration system. It enables observability, debugging, and performance optimization through structured logging and metrics collection.

## Key Files

### `telemetry.py`
**Purpose:** Centralized telemetry and metrics collection for system observability.

**Key Classes:**
- `MetricType`: Enum for metric categories
  - COUNTER: Monotonically increasing counter
  - GAUGE: Point-in-time value
  - HISTOGRAM: Distribution of values
  - TIMER: Duration measurements

- `Metric`: Metric definition
  - `name`: Metric identifier
  - `type`: MetricType
  - `value`: Current value
  - `timestamp`: Measurement timestamp
  - `labels`: Key-value labels for filtering
  - `unit`: Measurement unit

- `Telemetry`: Telemetry engine
  - `record_metric(name, value, metric_type, labels)`: Record metric
  - `start_timer(name, labels)`: Start timer
  - `stop_timer(timer_id)`: Stop timer and record duration
  - `increment_counter(name, labels, amount)`: Increment counter
  - `set_gauge(name, value, labels)`: Set gauge value
  - `record_histogram(name, value, labels)`: Record histogram value
  - `get_metrics(name, start_time, end_time)`: Query metrics
  - `get_metric_summary(name)`: Get aggregated statistics
  - `export_metrics(format)`: Export metrics (Prometheus, JSON, CSV)

**Key Metrics Tracked:**
- `agent_execution_duration`: Time taken by each agent
- `agent_execution_count`: Number of agent executions
- `pattern_execution_duration`: Pattern execution time
- `pattern_execution_count`: Pattern execution frequency
- `llm_api_calls`: LLM API call count
- `llm_api_latency`: LLM API response time
- `llm_tokens_used`: Token consumption
- `tool_execution_duration`: Tool execution time
- `tool_execution_count`: Tool usage frequency
- `error_count`: Error occurrences by type
- `checkpoint_triggered_count`: Checkpoint frequency
- `checkpoint_approval_rate`: Approval vs. rejection rate
- `task_memory_size`: Task context size in bytes
- `vector_memory_queries`: Vector search frequency
- `docker_container_count`: Active containers
- `docker_cpu_usage`: Container CPU usage
- `docker_memory_usage`: Container memory usage

**Dependencies:**
- External: `sqlite3`, `json`, `time`, `datetime`, `logging`
- Internal: None

### `logger.py`
**Purpose:** Structured logging with context enrichment and log aggregation.

**Key Classes:**
- `LogLevel`: Enum for log levels
  - DEBUG: Detailed debugging information
  - INFO: Informational messages
  - WARNING: Warning messages
  - ERROR: Error messages
  - CRITICAL: Critical errors

- `LogContext`: Context information for log entries
  - `task_id`: Current task identifier
  - `agent_name`: Current agent
  - `pattern`: Current pattern
  - `user_id`: User identifier
  - `session_id`: Session identifier
  - `trace_id`: Distributed trace ID

- `StructuredLogger`: Structured logging engine
  - `log(level, message, context, extra)`: Log with context
  - `debug(message, context, extra)`: Debug log
  - `info(message, context, extra)`: Info log
  - `warning(message, context, extra)`: Warning log
  - `error(message, context, extra)`: Error log
  - `critical(message, context, extra)`: Critical log
  - `set_context(context)`: Set log context for current scope
  - `clear_context()`: Clear log context
  - `query_logs(filters, start_time, end_time)`: Query logs
  - `export_logs(format, destination)`: Export logs

**Log Entry Structure:**
```python
{
    "timestamp": "2024-01-17T10:00:00.123Z",
    "level": "INFO",
    "message": "Agent execution completed",
    "context": {
        "task_id": "task_001",
        "agent_name": "web_pentester",
        "pattern": "CTF_SECURITY",
        "session_id": "session_123"
    },
    "extra": {
        "duration_ms": 1500,
        "findings_count": 5
    },
    "trace_id": "abc123def456",
    "hostname": "orchestrator-01",
    "pid": 12345
}
```

**Dependencies:**
- External: `logging`, `json`, `datetime`, `sqlite3`, `threading`
- Internal: None

## Internal Architecture

```
┌──────────────────────────────────────────┐
│         Telemetry                         │
│  (Metrics collection)                     │
│  - Counters, Gauges, Histograms          │
│  - Timers                                 │
│  - Aggregations                           │
│  - Export (Prometheus, JSON, CSV)        │
└─────────────┬────────────────────────────┘
              │
              │ provides metrics to
              ▼
┌──────────────────────────────────────────┐
│       StructuredLogger                    │
│  (Structured logging)                     │
│  - Context enrichment                     │
│  - Log aggregation                        │
│  - Query interface                        │
│  - Export                                 │
└─────────────┬────────────────────────────┘
              │
              │ writes to
              ▼
┌──────────────────────────────────────────┐
│     Persistent Storage                    │
│  - SQLite (metrics.db, logs.db)          │
│  - Log files (.runtime/logs/)            │
│  - Metric exports                         │
└──────────────────────────────────────────┘
```

## Usage Examples

### Telemetry
```python
from ai_autonom.monitoring.telemetry import Telemetry, MetricType

# Initialize telemetry
telemetry = Telemetry()

# Record counter
telemetry.increment_counter(
    name="agent_execution_count",
    labels={"agent": "web_pentester", "pattern": "CTF_SECURITY"},
    amount=1
)

# Start/stop timer
timer_id = telemetry.start_timer(
    name="agent_execution_duration",
    labels={"agent": "web_pentester"}
)
# ... agent execution ...
telemetry.stop_timer(timer_id)

# Set gauge
telemetry.set_gauge(
    name="docker_container_count",
    value=5,
    labels={"status": "running"}
)

# Record histogram
telemetry.record_histogram(
    name="llm_api_latency",
    value=250.5,  # milliseconds
    labels={"provider": "ollama", "model": "llama3:8b"}
)

# Query metrics
metrics = telemetry.get_metrics(
    name="agent_execution_duration",
    start_time="2024-01-17T00:00:00",
    end_time="2024-01-17T23:59:59"
)

# Get summary statistics
summary = telemetry.get_metric_summary("agent_execution_duration")
print(f"Avg: {summary['mean']}")
print(f"P50: {summary['p50']}")
print(f"P95: {summary['p95']}")
print(f"P99: {summary['p99']}")

# Export metrics (Prometheus format)
prometheus_output = telemetry.export_metrics("prometheus")
print(prometheus_output)
```

### Structured Logging
```python
from ai_autonom.monitoring.logger import StructuredLogger, LogContext, LogLevel

# Initialize logger
logger = StructuredLogger()

# Set context for current scope
context = LogContext(
    task_id="task_001",
    agent_name="web_pentester",
    pattern="CTF_SECURITY",
    session_id="session_123"
)
logger.set_context(context)

# Log with context
logger.info(
    message="Starting security assessment",
    extra={"target": "https://example.com"}
)

# Log execution details
logger.debug(
    message="Executing tool",
    extra={"tool": "nmap", "command": "nmap -sV 192.168.1.1"}
)

# Log errors
try:
    # ... some operation ...
    pass
except Exception as e:
    logger.error(
        message="Agent execution failed",
        extra={"error": str(e), "error_type": type(e).__name__}
    )

# Clear context
logger.clear_context()

# Query logs
logs = logger.query_logs(
    filters={"level": "ERROR", "agent_name": "web_pentester"},
    start_time="2024-01-17T00:00:00",
    end_time="2024-01-17T23:59:59"
)

# Export logs
logger.export_logs(format="json", destination=".runtime/logs/export.json")
```

### Combined Usage
```python
from ai_autonom.monitoring.telemetry import Telemetry
from ai_autonom.monitoring.logger import StructuredLogger, LogContext

telemetry = Telemetry()
logger = StructuredLogger()

# Set context
context = LogContext(task_id="task_001", agent_name="code_generator")
logger.set_context(context)

# Log start
logger.info("Starting code generation")

# Start timer
timer_id = telemetry.start_timer(
    "agent_execution_duration",
    labels={"agent": "code_generator"}
)

try:
    # ... agent execution ...
    
    # Increment counter
    telemetry.increment_counter(
        "agent_execution_count",
        labels={"agent": "code_generator", "status": "success"}
    )
    
    logger.info("Code generation completed", extra={"lines": 150})
    
except Exception as e:
    # Record error
    telemetry.increment_counter(
        "error_count",
        labels={"agent": "code_generator", "error_type": type(e).__name__}
    )
    
    logger.error("Code generation failed", extra={"error": str(e)})
    
finally:
    # Stop timer
    telemetry.stop_timer(timer_id)
    logger.clear_context()
```

## Dependencies

**External Dependencies:**
- `sqlite3`: Persistent storage for metrics and logs
- `json`: Data serialization
- `logging`: Python logging infrastructure
- `datetime`: Timestamp handling
- `threading`: Thread-safe operations

**Internal Dependencies:**
- None (Monitoring module has no internal dependencies to avoid circular imports)

## Important Functionality

1. **Metrics Collection**: Track system performance and behavior
2. **Structured Logging**: Context-enriched logs for debugging
3. **Time-Series Data**: Historical metrics for trend analysis
4. **Query Interface**: Flexible querying of metrics and logs
5. **Aggregations**: Statistical summaries (mean, median, percentiles)
6. **Export Formats**: Prometheus, JSON, CSV exports
7. **Context Propagation**: Automatic context inheritance across log entries
8. **Distributed Tracing**: Trace ID for following requests across components
9. **Alert Thresholds**: Define thresholds for metric-based alerting
10. **Performance Profiling**: Identify bottlenecks through timer metrics

## Key Metrics to Monitor

### Performance Metrics
- `agent_execution_duration`: Track slow agents
- `llm_api_latency`: Monitor LLM API performance
- `tool_execution_duration`: Identify slow tools
- `pattern_execution_duration`: Measure pattern efficiency

### Usage Metrics
- `agent_execution_count`: Agent popularity
- `pattern_execution_count`: Pattern usage frequency
- `tool_execution_count`: Tool utilization
- `llm_api_calls`: API usage for cost tracking
- `llm_tokens_used`: Token consumption

### Reliability Metrics
- `error_count`: Error frequency by type
- `checkpoint_approval_rate`: HITL approval patterns
- `agent_retry_count`: Agent reliability
- `docker_container_restarts`: Container stability

### Resource Metrics
- `docker_cpu_usage`: Container resource usage
- `docker_memory_usage`: Memory consumption
- `task_memory_size`: Context size growth
- `vector_memory_queries`: Search load

## Logging Best Practices

1. **Use Structured Logs**: Always include context and extra fields
   ```python
   logger.info("Event occurred", extra={"key": "value"})
   ```

2. **Set Context Early**: Establish context at workflow start
   ```python
   logger.set_context(LogContext(task_id="task_001"))
   ```

3. **Log Errors with Details**: Include error type and stack trace
   ```python
   logger.error("Failed", extra={"error": str(e), "trace": traceback.format_exc()})
   ```

4. **Use Appropriate Levels**: DEBUG for details, INFO for progress, ERROR for failures

5. **Clear Context**: Always clear context when done
   ```python
   try:
       # work
   finally:
       logger.clear_context()
   ```

## Metric Collection Patterns

### Timer Pattern
```python
timer_id = telemetry.start_timer("operation")
try:
    # operation
finally:
    telemetry.stop_timer(timer_id)
```

### Counter Pattern
```python
telemetry.increment_counter("event_count", labels={"type": "success"})
```

### Gauge Pattern
```python
telemetry.set_gauge("queue_size", current_size, labels={"queue": "main"})
```

### Histogram Pattern
```python
telemetry.record_histogram("response_size_bytes", size, labels={"endpoint": "/api"})
```

## Export Formats

### Prometheus
```
# HELP agent_execution_duration Agent execution time in seconds
# TYPE agent_execution_duration histogram
agent_execution_duration_bucket{agent="web_pentester",le="1.0"} 10
agent_execution_duration_bucket{agent="web_pentester",le="5.0"} 25
agent_execution_duration_bucket{agent="web_pentester",le="+Inf"} 30
agent_execution_duration_sum{agent="web_pentester"} 75.5
agent_execution_duration_count{agent="web_pentester"} 30
```

### JSON
```json
{
  "metrics": [
    {
      "name": "agent_execution_duration",
      "type": "histogram",
      "labels": {"agent": "web_pentester"},
      "values": [1.2, 2.3, 3.4],
      "statistics": {"mean": 2.3, "p50": 2.3, "p95": 3.4}
    }
  ]
}
```

## Storage Locations

- **Metrics Database**: `.runtime/data/metrics.db`
- **Logs Database**: `.runtime/data/logs.db`
- **Log Files**: `.runtime/logs/app.log`
- **Metric Exports**: `.runtime/logs/metrics_export.txt`
