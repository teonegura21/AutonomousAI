#!/usr/bin/env python3
"""
Configuration Management
Central configuration for the entire system
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


class Config:
    """
    Singleton configuration manager
    Loads from YAML and provides dot-notation access
    """
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    _loaded: bool = False
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "config/settings.yaml") -> 'Config':
        """Load configuration from YAML file"""
        if self._loaded:
            return self
            
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            print(f"[CONFIG] Loaded from {config_path}")
        else:
            self._config = self._defaults()
            print(f"[CONFIG] Using defaults (no config file at {config_path})")
        
        self._loaded = True
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot notation
        Example: config.get('orchestrator.model') -> "huihui_ai/orchestrator-abliterated"
        """
        if not self._loaded:
            self.load()
            
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a config value at runtime"""
        if not self._loaded:
            self.load()
            
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section"""
        if not self._loaded:
            self.load()
        return self._config.get(section, {})
    
    def _defaults(self) -> Dict[str, Any]:
        """Default configuration when no file exists"""
        return {
            "orchestrator": {
                "model": "huihui_ai/orchestrator-abliterated:latest",
                "max_decomposition_depth": 3,
                "enable_human_checkpoints": False,
                "enable_testing": True,
                "timeout_sec": 300
            },
            "execution": {
                "vram_limit_gb": 20,
                "max_parallel_agents": 3,
                "default_timeout_sec": 300,
                "max_retries": 3
            },
            "memory": {
                "vector_db": {
                    "type": "chromadb",
                    "embedding_model": "nomic-embed-text",
                    "auto_pull_embeddings": False,
                    "persist_directory": "./.runtime/data/chromadb"
                },
                "structured_db": {
                    "type": "sqlite",
                    "path": "./.runtime/data/agent_registry.db"
                }
            },
            "sandbox": {
                "enabled": True,
                "type": "docker",
                "image": "python:3.11-slim"
            },
            "tools": {
                "builtin": ["filesystem_read", "filesystem_write", "python_exec", "bash_exec"],
                "enable_dynamic": True
            },
            "model_discovery": {
                "enabled": True,
                "scan_interval_sec": 60,
                "auto_benchmark": True
            },
            "logging": {
                "level": "INFO",
                "file": "./.runtime/logs/orchestrator.log"
            },
            "checkpoints": {
                "enabled": True,
                "db_path": "./.runtime/data/checkpoints.db"
            }
        }
    
    def reload(self, config_path: str = "config/settings.yaml") -> 'Config':
        """Force reload configuration"""
        self._loaded = False
        return self.load(config_path)
    
    def save(self, config_path: str = "config/settings.yaml") -> None:
        """Save current config to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        print(f"[CONFIG] Saved to {config_path}")
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        if not self._loaded:
            self.load()
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Config(loaded={self._loaded}, sections={list(self._config.keys())})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global config instance"""
    return config


if __name__ == "__main__":
    # Test configuration
    cfg = Config().load()
    
    print("\n" + "="*60)
    print("CONFIGURATION TEST")
    print("="*60 + "\n")
    
    print(f"Orchestrator model: {cfg.get('orchestrator.model')}")
    print(f"VRAM limit: {cfg.get('execution.vram_limit_gb')}GB")
    print(f"Sandbox enabled: {cfg.get('sandbox.enabled')}")
    print(f"Vector DB type: {cfg.get('memory.vector_db.type')}")
    print(f"Missing key: {cfg.get('nonexistent.key', 'default_value')}")
    
    print("\nFull orchestrator section:")
    print(cfg.get_section('orchestrator'))
