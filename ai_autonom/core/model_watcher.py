#!/usr/bin/env python3
"""
Model Watcher
Background service that watches for new Ollama models and auto-registers them
"""

import threading
import time
from datetime import datetime
from typing import List, Dict, Callable, Optional, Any


class ModelWatcher:
    """
    Watch for new Ollama models and auto-register them
    Runs in background thread, scanning at configurable intervals
    """
    
    def __init__(
        self,
        discovery,  # ModelDiscovery instance
        interval: int = 60,
        auto_benchmark: bool = True,
        on_new_model: Optional[Callable[[Dict], None]] = None
    ):
        """
        Args:
            discovery: ModelDiscovery instance for scanning and registering
            interval: Scan interval in seconds
            auto_benchmark: Whether to benchmark new models automatically
            on_new_model: Callback function when new model is discovered
        """
        self.discovery = discovery
        self.interval = interval
        self.auto_benchmark = auto_benchmark
        self.on_new_model = on_new_model
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_scan: Optional[str] = None
        self.discovered_count = 0
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start watching in background"""
        if self.running:
            print("[WATCHER] Already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        print(f"[WATCHER] Started - scanning every {self.interval}s")
    
    def stop(self) -> None:
        """Stop the watcher"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        print("[WATCHER] Stopped")
    
    def _watch_loop(self) -> None:
        """Main watch loop - runs in background thread"""
        while self.running:
            try:
                self._scan_and_register()
            except Exception as e:
                print(f"[WATCHER] Error in scan loop: {e}")
            
            # Sleep in small increments so we can stop quickly
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _scan_and_register(self) -> List[Dict]:
        """Scan for new models and register them"""
        with self._lock:
            self.last_scan = datetime.now().isoformat()
        
        new_models = self.discovery.discover_new_models()
        
        for model_info in new_models:
            model_name = model_info['name']
            print(f"[WATCHER] New model detected: {model_name}")
            
            # Register with or without benchmarking
            capabilities = self.discovery.auto_register_model(
                model_name,
                skip_benchmark=not self.auto_benchmark
            )
            
            with self._lock:
                self.discovered_count += 1
            
            # Call callback if provided
            if self.on_new_model:
                try:
                    self.on_new_model(capabilities)
                except Exception as e:
                    print(f"[WATCHER] Callback error: {e}")
        
        return new_models
    
    def scan_now(self) -> List[Dict]:
        """
        Manual scan - trigger immediate scan
        Returns list of newly discovered models
        """
        print("[WATCHER] Manual scan triggered")
        return self._scan_and_register()
    
    def get_status(self) -> Dict[str, Any]:
        """Get watcher status"""
        with self._lock:
            return {
                "running": self.running,
                "interval_sec": self.interval,
                "auto_benchmark": self.auto_benchmark,
                "last_scan": self.last_scan,
                "total_discovered": self.discovered_count
            }
    
    def set_interval(self, interval: int) -> None:
        """Change scan interval"""
        self.interval = max(10, interval)  # Minimum 10 seconds
        print(f"[WATCHER] Interval changed to {self.interval}s")
    
    def set_auto_benchmark(self, enabled: bool) -> None:
        """Enable/disable auto benchmarking"""
        self.auto_benchmark = enabled
        print(f"[WATCHER] Auto-benchmark: {'enabled' if enabled else 'disabled'}")


class ModelSyncManager:
    """
    Higher-level manager that coordinates model discovery and selection
    Integrates ModelDiscovery, ModelSelector, and ModelWatcher
    """
    
    def __init__(
        self,
        db_path: str = ".runtime/data/agent_registry.db",
        scan_interval: int = 60,
        auto_start: bool = True
    ):
        # Import here to avoid circular imports
        from .model_discovery import ModelDiscovery
        from .model_selector import DynamicModelSelector
        
        self.discovery = ModelDiscovery(db_path)
        self.selector = DynamicModelSelector(db_path)
        self.watcher = ModelWatcher(
            self.discovery,
            interval=scan_interval,
            auto_benchmark=True,
            on_new_model=self._on_new_model
        )
        
        if auto_start:
            self.start()
    
    def start(self) -> None:
        """Start the sync manager"""
        # Initial sync
        self.sync_now()
        # Start background watcher
        self.watcher.start()
    
    def stop(self) -> None:
        """Stop the sync manager"""
        self.watcher.stop()
    
    def sync_now(self) -> Dict[str, Any]:
        """Perform immediate sync"""
        new_models = self.watcher.scan_now()
        
        return {
            "new_models_count": len(new_models),
            "new_models": [m['name'] for m in new_models],
            "total_registered": len(self.discovery.get_all_capabilities())
        }
    
    def _on_new_model(self, capabilities: Dict) -> None:
        """Callback when new model is discovered"""
        model_name = capabilities.get('model_name', 'unknown')
        coding_score = capabilities.get('coding_score', 0)
        
        print(f"[SYNC] New model registered: {model_name}")
        print(f"[SYNC] Coding: {coding_score:.1f} | Speed: {capabilities.get('speed_tokens_sec', 0):.1f}t/s")
        
        # Check if this model should replace any existing assignments
        # This could trigger orchestrator reconfiguration
    
    def get_best_model(self, task_type: str, constraints: Dict = None) -> Optional[str]:
        """Get best model for a task type"""
        result = self.selector.select_best_model(task_type, constraints)
        if isinstance(result, dict):
            return result.get("model_name")
        return result
    
    def get_model_rankings(self, task_type: str = "balanced") -> List[Dict]:
        """Get ranked models for a task type"""
        return self.selector.get_model_rankings(task_type)
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status"""
        watcher_status = self.watcher.get_status()
        all_models = self.discovery.get_all_capabilities()
        
        return {
            "watcher": watcher_status,
            "registered_models": len(all_models),
            "model_names": [m['model_name'] for m in all_models]
        }


if __name__ == "__main__":
    # Test model watcher
    from model_discovery import ModelDiscovery
    
    print("\n" + "="*60)
    print("MODEL WATCHER TEST")
    print("="*60 + "\n")
    
    discovery = ModelDiscovery()
    
    def on_new(caps):
        print(f"Callback: New model {caps['model_name']} with coding score {caps['coding_score']}")
    
    watcher = ModelWatcher(
        discovery,
        interval=10,
        auto_benchmark=False,  # Skip benchmark for quick test
        on_new_model=on_new
    )
    
    # Manual scan
    print("Performing manual scan...")
    new_models = watcher.scan_now()
    print(f"Found {len(new_models)} new models")
    
    # Show status
    print(f"\nStatus: {watcher.get_status()}")
    
    # Start background watcher (would run indefinitely)
    # watcher.start()
    # time.sleep(30)
    # watcher.stop()
