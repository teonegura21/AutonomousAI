# Core Module

## Purpose
The core module provides the foundational infrastructure for the multi-agent orchestration system. It handles model management, configuration, LLM provider abstraction, and dynamic model capabilities.

## Key Files

### `config.py`
**Purpose:** Centralized configuration management for the entire system.

**Key Classes:**
- `Config`: Main configuration class managing all system settings
  - `LLM_PROVIDER`: Provider selection (ollama, openai, azure, compatible)
  - `LLM_BASE_URL`: Base URL for LLM API
  - `LLM_MODEL`: Default model name
  - `LLM_TEMPERATURE`: Temperature for generation
  - `MAX_RETRIES`: Maximum retry attempts
  - `TIMEOUT`: Request timeout in seconds
  - `DOCKER_ENABLED`: Enable/disable Docker sandboxing

**Dependencies:** 
- External: `os`, `dotenv`
- Internal: None

### `llm_provider.py`
**Purpose:** Abstract LLM provider interface supporting multiple backends (Ollama, OpenAI, Azure, compatible APIs).

**Key Classes:**
- `BaseLLMProvider` (ABC): Abstract base class for all providers
  - `generate(prompt, **kwargs)`: Generate text from prompt
  - `generate_structured(prompt, schema, **kwargs)`: Generate structured output
  - `embed(text)`: Generate embeddings
  - `chat(messages, **kwargs)`: Chat completion
  
- `OllamaProvider`: Local Ollama implementation
- `OpenAIProvider`: OpenAI API implementation
- `AzureProvider`: Azure OpenAI implementation
- `CompatibleProvider`: OpenAI-compatible APIs

**Key Functions:**
- `get_provider(provider_name: str) -> BaseLLMProvider`: Factory function for provider instantiation

**Dependencies:**
- External: `requests`, `openai`, `json`
- Internal: `ai_autonom.core.config`

### `model_manager.py`
**Purpose:** Dynamic model discovery, capability assessment, and intelligent selection.

**Key Classes:**
- `ModelCapability`: Enum defining model capabilities (CODE, CHAT, REASONING, CREATIVE, SECURITY, VISION, EMBEDDING)
- `ModelInfo`: Data class for model metadata
  - `name`: Model identifier
  - `provider`: Provider name
  - `capabilities`: List of capabilities
  - `parameters`: Model parameters (size, context_length, etc.)
  - `benchmark_score`: Performance score
  - `tags`: Metadata tags

- `ModelManager`: Manages model lifecycle and selection
  - `discover_models()`: Auto-discover available models from provider
  - `assess_capabilities(model_name)`: Assess model capabilities
  - `select_model(task_type, requirements)`: Intelligent model selection
  - `benchmark_model(model_name)`: Benchmark model performance
  - `get_all_models()`: Retrieve all registered models

- `ModelWatcher`: Monitors model availability and performance
  - `start_watching()`: Begin continuous monitoring
  - `stop_watching()`: Stop monitoring
  - `on_model_added(callback)`: Register callback for new models

**Dependencies:**
- External: `sqlite3`, `json`, `time`, `threading`
- Internal: `ai_autonom.core.llm_provider`, `ai_autonom.core.config`

## Internal Architecture

```
┌─────────────────────────────────────────┐
│            Config                        │
│  (Configuration Management)              │
└─────────────────┬───────────────────────┘
                  │
                  │ provides settings
                  ▼
┌─────────────────────────────────────────┐
│       LLM Provider Factory               │
│  (get_provider)                          │
└─────────────────┬───────────────────────┘
                  │
                  │ creates
                  ▼
┌─────────────────────────────────────────┐
│     BaseLLMProvider (Abstract)           │
│  ├── OllamaProvider                      │
│  ├── OpenAIProvider                      │
│  ├── AzureProvider                       │
│  └── CompatibleProvider                  │
└─────────────────┬───────────────────────┘
                  │
                  │ used by
                  ▼
┌─────────────────────────────────────────┐
│       ModelManager                       │
│  - Model discovery                       │
│  - Capability assessment                 │
│  - Intelligent selection                 │
│  - Benchmarking                          │
└─────────────────┬───────────────────────┘
                  │
                  │ monitors
                  ▼
┌─────────────────────────────────────────┐
│       ModelWatcher                       │
│  (Continuous monitoring)                 │
└─────────────────────────────────────────┘
```

## Usage Examples

### Configuration
```python
from ai_autonom.core.config import Config

# Access configuration
provider = Config.LLM_PROVIDER  # "ollama"
model = Config.LLM_MODEL        # "llama3:8b"
temperature = Config.LLM_TEMPERATURE  # 0.7
```

### LLM Provider
```python
from ai_autonom.core.llm_provider import get_provider

# Get provider instance
provider = get_provider("ollama")

# Generate text
response = provider.generate("What is AI?", temperature=0.7)

# Generate structured output
schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
result = provider.generate_structured("Explain AI", schema)

# Chat completion
messages = [{"role": "user", "content": "Hello!"}]
chat_response = provider.chat(messages)
```

### Model Management
```python
from ai_autonom.core.model_manager import ModelManager, ModelCapability

# Initialize manager
manager = ModelManager()

# Discover models
models = manager.discover_models()

# Select best model for task
model = manager.select_model(
    task_type="code_generation",
    requirements={"capabilities": [ModelCapability.CODE]}
)

# Benchmark model
score = manager.benchmark_model("llama3:8b")

# Start continuous monitoring
from ai_autonom.core.model_manager import ModelWatcher
watcher = ModelWatcher()
watcher.on_model_added(lambda m: print(f"New model: {m}"))
watcher.start_watching()
```

## Dependencies

**External Dependencies:**
- `requests`: HTTP client for API calls
- `openai`: OpenAI SDK for OpenAI/Azure providers
- `python-dotenv`: Environment variable management
- `sqlite3`: Model metadata persistence

**Internal Dependencies:**
- None (Foundation layer with no internal dependencies)

## Important Functionality

1. **Multi-Provider Support**: Seamless switching between Ollama, OpenAI, Azure, and compatible APIs
2. **Dynamic Model Discovery**: Auto-detection of available models from configured provider
3. **Capability Assessment**: Automatic capability inference from model metadata and tags
4. **Intelligent Selection**: Context-aware model selection based on task requirements
5. **Performance Benchmarking**: Objective scoring for model comparison
6. **Continuous Monitoring**: Real-time tracking of model availability and performance
7. **Structured Output**: JSON schema-constrained generation for reliable parsing
8. **Embedding Support**: Vector embedding generation for semantic search

## Configuration

Set these environment variables in `.env`:
```
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3:8b
LLM_TEMPERATURE=0.7
MAX_RETRIES=3
TIMEOUT=30
```
