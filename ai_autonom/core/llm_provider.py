#!/usr/bin/env python3
"""
LLM Provider - Unified abstraction for multiple LLM backends
Supports: Ollama (local), OpenAI, Azure OpenAI, and compatible APIs
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Generator, Union
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(Enum):
    """Supported LLM provider types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OPENAI_COMPATIBLE = "openai_compatible"  # For local OpenAI-compatible servers


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: ProviderType = ProviderType.OLLAMA
    model: str = "qwen3:1.7b"
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For custom endpoints
    organization: Optional[str] = None
    timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMMessage:
    """Chat message"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    provider: str
    tokens_input: int = 0
    tokens_output: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[Dict] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request and get response"""
        pass
    
    @abstractmethod
    def chat_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    def _messages_to_dict(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert messages to dict format"""
        return [{"role": m.role, "content": m.content} for m in messages]


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self._client = ollama
            if self.config.api_base:
                # Custom Ollama server
                os.environ['OLLAMA_HOST'] = self.config.api_base
        except ImportError:
            print("[OLLAMA] ollama package not installed")
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        if not self._client:
            return False
        try:
            self._client.list()
            return True
        except:
            return False
    
    def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request"""
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        
        model = model or self.config.model
        msg_dicts = self._messages_to_dict(messages)
        
        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        response = self._client.chat(
            model=model,
            messages=msg_dicts,
            options=options,
            format=kwargs.get("format")
        )
        
        return LLMResponse(
            content=response.get("message", {}).get("content", ""),
            model=model,
            provider="ollama",
            tokens_input=response.get("prompt_eval_count", 0),
            tokens_output=response.get("eval_count", 0),
            raw_response=response
        )
    
    def chat_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response"""
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        
        model = model or self.config.model
        msg_dicts = self._messages_to_dict(messages)
        
        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        response = self._client.chat(
            model=model,
            messages=msg_dicts,
            options=options,
            stream=True
        )
        
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = self.client.list()
            return [m['name'] for m in response.get('models', [])]
        except Exception as e:
            print(f"[OLLAMA] List models failed: {e}")
            return []

    def unload_model(self, model_name: str):
        """Force unload a model from VRAM (Ollama specific)"""
        try:
            print(f"[OLLAMA] Unloading model: {model_name}...")
            # keep_alive=0 tells Ollama to unload immediately
            self.client.chat(model=model_name, messages=[], keep_alive=0)
            print(f"[OLLAMA] Unloaded {model_name}")
        except Exception as e:
            print(f"[OLLAMA] Failed to unload {model_name}: {e}")

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (also works with compatible APIs)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print("[OPENAI] No API key provided")
                return
            
            client_kwargs = {"api_key": api_key}
            
            # Custom base URL for compatible APIs
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base
            
            if self.config.organization:
                client_kwargs["organization"] = self.config.organization
            
            self._client = OpenAI(**client_kwargs)
            
        except ImportError:
            print("[OPENAI] openai package not installed. Run: pip install openai")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if not self._client:
            return False
        try:
            # Try a simple models list call
            self._client.models.list()
            return True
        except Exception as e:
            print(f"[OPENAI] Not available: {e}")
            return False
    
    def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request"""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")
        
        model = model or self.config.model
        msg_dicts = self._messages_to_dict(messages)
        
        response = self._client.chat.completions.create(
            model=model,
            messages=msg_dicts,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_format={"type": kwargs.get("format", "text")} if kwargs.get("format") == "json" else None
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            provider="openai",
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
    
    def chat_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response"""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")
        
        model = model or self.config.model
        msg_dicts = self._messages_to_dict(messages)
        
        stream = self._client.chat.completions.create(
            model=model,
            messages=msg_dicts,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def list_models(self) -> List[str]:
        """List available models"""
        if not self._client:
            return []
        try:
            response = self._client.models.list()
            return [m.id for m in response.data]
        except:
            return []
    
    def embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embeddings"""
        if not self._client:
            return []
        try:
            response = self._client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except:
            return []


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Azure OpenAI client"""
        try:
            from openai import AzureOpenAI
            
            api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            api_base = self.config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = self.config.extra_params.get("api_version", "2024-02-01")
            
            if not api_key or not api_base:
                print("[AZURE] Missing API key or endpoint")
                return
            
            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )
            
        except ImportError:
            print("[AZURE] openai package not installed")
    
    def is_available(self) -> bool:
        """Check if Azure OpenAI is available"""
        return self._client is not None
    
    def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request"""
        if not self._client:
            raise RuntimeError("Azure OpenAI client not initialized")
        
        # For Azure, model is the deployment name
        deployment = model or self.config.model
        msg_dicts = self._messages_to_dict(messages)
        
        response = self._client.chat.completions.create(
            model=deployment,
            messages=msg_dicts,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            content=choice.message.content or "",
            model=deployment,
            provider="azure_openai",
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
    
    def chat_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response"""
        if not self._client:
            raise RuntimeError("Azure OpenAI client not initialized")
        
        deployment = model or self.config.model
        msg_dicts = self._messages_to_dict(messages)
        
        stream = self._client.chat.completions.create(
            model=deployment,
            messages=msg_dicts,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def list_models(self) -> List[str]:
        """List models (returns configured deployment for Azure)"""
        return [self.config.model] if self.config.model else []


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    _providers = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.AZURE_OPENAI: AzureOpenAIProvider,
        ProviderType.OPENAI_COMPATIBLE: OpenAIProvider,  # Same as OpenAI with custom base_url
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create provider based on config"""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {config.provider}")
        return provider_class(config)
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> BaseLLMProvider:
        """Create provider from dictionary config"""
        provider_type = ProviderType(config_dict.get("provider", "ollama"))
        config = LLMConfig(
            provider=provider_type,
            model=config_dict.get("model", "qwen3:1.7b"),
            api_key=config_dict.get("api_key"),
            api_base=config_dict.get("api_base"),
            organization=config_dict.get("organization"),
            timeout=config_dict.get("timeout", 120),
            max_tokens=config_dict.get("max_tokens", 4096),
            temperature=config_dict.get("temperature", 0.7),
            extra_params=config_dict.get("extra_params", {})
        )
        return cls.create(config)


class MultiProvider:
    """
    Multi-provider manager - fallback between providers
    Use local Ollama by default, fall back to OpenAI if unavailable
    """
    
    def __init__(self, providers: List[BaseLLMProvider]):
        self.providers = providers
        self._active_provider: Optional[BaseLLMProvider] = None
    
    def get_available_provider(self) -> Optional[BaseLLMProvider]:
        """Get first available provider"""
        for provider in self.providers:
            if provider.is_available():
                return provider
        return None
    
    def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        preferred_provider: Optional[ProviderType] = None,
        **kwargs
    ) -> LLMResponse:
        """Chat with fallback"""
        # Try preferred provider first
        if preferred_provider:
            for p in self.providers:
                if p.config.provider == preferred_provider and p.is_available():
                    return p.chat(messages, model, **kwargs)
        
        # Fall back to first available
        provider = self.get_available_provider()
        if not provider:
            raise RuntimeError("No LLM provider available")
        
        return provider.chat(messages, model, **kwargs)
    
    def chat_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        preferred_provider: Optional[ProviderType] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream with fallback"""
        if preferred_provider:
            for p in self.providers:
                if p.config.provider == preferred_provider and p.is_available():
                    yield from p.chat_stream(messages, model, **kwargs)
                    return
        
        provider = self.get_available_provider()
        if not provider:
            raise RuntimeError("No LLM provider available")
        
        yield from provider.chat_stream(messages, model, **kwargs)


# Convenience function
def get_provider(
    provider_type: str = "ollama",
    model: str = None,
    api_key: str = None,
    api_base: str = None
) -> BaseLLMProvider:
    """Get a configured LLM provider"""
    config = LLMConfig(
        provider=ProviderType(provider_type),
        model=model or ("gpt-4o-mini" if provider_type == "openai" else "qwen3:1.7b"),
        api_key=api_key,
        api_base=api_base
    )
    return LLMProviderFactory.create(config)


if __name__ == "__main__":
    # Test providers
    print("\n" + "="*60)
    print("LLM PROVIDER TEST")
    print("="*60 + "\n")
    
    # Test Ollama
    print("Testing Ollama provider...")
    ollama_config = LLMConfig(provider=ProviderType.OLLAMA, model="qwen3:1.7b")
    ollama_provider = LLMProviderFactory.create(ollama_config)
    print(f"  Available: {ollama_provider.is_available()}")
    if ollama_provider.is_available():
        models = ollama_provider.list_models()
        print(f"  Models: {len(models)}")
    
    # Test OpenAI (if key available)
    print("\nTesting OpenAI provider...")
    if os.getenv("OPENAI_API_KEY"):
        openai_config = LLMConfig(provider=ProviderType.OPENAI, model="gpt-4o-mini")
        openai_provider = LLMProviderFactory.create(openai_config)
        print(f"  Available: {openai_provider.is_available()}")
    else:
        print("  Skipped (no OPENAI_API_KEY)")
    
    print("\nProvider factory working correctly!")
