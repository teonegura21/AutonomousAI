#!/usr/bin/env python3
"""
Model Auto-Discovery
Automatically detect new Ollama models and assess their capabilities
"""

import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# Lazy import for ollama
_ollama = None

def _get_ollama():
    """Lazy load ollama module"""
    global _ollama
    if _ollama is None:
        try:
            import ollama
            _ollama = ollama
        except ImportError:
            _ollama = False
    return _ollama if _ollama else None


class ModelDiscovery:
    """
    Auto-detect new Ollama models and update registry
    When user installs a new model (ollama pull xyz), system detects it automatically
    """
    
    def __init__(self, db_path: str = ".runtime/data/agent_registry.db"):
        self.db_path = db_path
        self._ensure_db()

    @staticmethod
    def is_embedding_model(model_name: str) -> bool:
        """Heuristic to detect embedding-only models by name."""
        if not model_name:
            return False
        name = model_name.lower()
        hints = ("embed", "embedding", "bge", "e5", "gte", "mxbai", "instructor", "arctic-embed")
        return any(hint in name for hint in hints)
    
    def _ensure_db(self):
        """Ensure database and tables exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create model_capabilities table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_capabilities (
                model_name TEXT PRIMARY KEY,
                coding_score REAL DEFAULT 0,
                reasoning_score REAL DEFAULT 0,
                documentation_score REAL DEFAULT 0,
                speed_tokens_sec REAL DEFAULT 0,
                vram_gb REAL DEFAULT 0,
                parameter_count TEXT,
                quantization TEXT,
                assessed_at TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Create model_comparisons table for tracking which model won
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_a TEXT,
                model_b TEXT,
                task_type TEXT,
                winner TEXT,
                compared_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def scan_ollama_models(self) -> List[Dict[str, Any]]:
        """Get all models currently available in Ollama"""
        ollama = _get_ollama()
        if not ollama:
            print("[MODEL_DISCOVERY] Ollama not installed")
            return []
        
        try:
            response = ollama.list()
            if isinstance(response, dict):
                models = response.get('models', [])
            else:
                models = getattr(response, "models", [])
            return [
                {
                    'name': (m.get('name') if isinstance(m, dict) else getattr(m, "name", None)) 
                            or (m.get('model') if isinstance(m, dict) else getattr(m, "model", "")),
                    'size': m.get('size', 0) if isinstance(m, dict) else getattr(m, "size", 0),
                    'modified_at': m.get('modified_at', '') if isinstance(m, dict) else getattr(m, "modified_at", ""),
                    'details': m.get('details', {}) if isinstance(m, dict) else getattr(m, "details", {})
                }
                for m in models
            ]
        except Exception as e:
            print(f"[MODEL_DISCOVERY] Error scanning Ollama: {e}")
            return []
    
    def get_registered_models(self) -> List[str]:
        """Get models already in our database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT model_name FROM model_capabilities WHERE is_active = 1")
            registered = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            registered = []
        
        conn.close()
        return registered
    
    def discover_new_models(self) -> List[Dict[str, Any]]:
        """Find models in Ollama not yet in our registry"""
        ollama_models = self.scan_ollama_models()
        registered = self.get_registered_models()
        
        new_models = []
        for model in ollama_models:
            model_name = model['name']
            if model_name not in registered:
                new_models.append(model)
        
        return new_models

    def sync_ollama_models(
        self,
        auto_register: bool = True,
        auto_benchmark: bool = False
    ) -> Dict[str, List[str]]:
        """
        Sync Ollama models with local registry.
        - Registers newly discovered models.
        - Deactivates models no longer present.
        """
        results = {
            "available": [],
            "registered": [],
            "deactivated": [],
            "failed": []
        }

        ollama_models = self.scan_ollama_models()
        available = {m.get("name") for m in ollama_models if m.get("name")}
        results["available"] = sorted(available)

        registered = set(self.get_registered_models())
        embedding_models = {m for m in available if self.is_embedding_model(m)}

        # Register new models
        for model_name in sorted(available - registered):
            try:
                if self.is_embedding_model(model_name):
                    continue
                if auto_register:
                    self.auto_register_model(model_name, skip_benchmark=not auto_benchmark)
                    results["registered"].append(model_name)
            except Exception as e:
                print(f"[MODEL_DISCOVERY] Sync register failed for {model_name}: {e}")
                results["failed"].append(model_name)

        # Deactivate embedding models from registry to avoid agent selection
        for model_name in sorted(embedding_models & registered):
            try:
                if self.deactivate_model(model_name):
                    results["deactivated"].append(model_name)
            except Exception as e:
                print(f"[MODEL_DISCOVERY] Sync deactivate failed for {model_name}: {e}")
                results["failed"].append(model_name)

        # Deactivate missing models
        for model_name in sorted(registered - available):
            try:
                if self.deactivate_model(model_name):
                    results["deactivated"].append(model_name)
            except Exception as e:
                print(f"[MODEL_DISCOVERY] Sync deactivate failed for {model_name}: {e}")
                results["failed"].append(model_name)

        return results

    def ensure_models(
        self,
        model_names: List[str],
        auto_register: bool = True,
        auto_benchmark: bool = True,
        pull_missing: bool = True
    ) -> Dict[str, List[str]]:
        """
        Ensure a list of Ollama models exists locally; pull missing ones and
        register capabilities if requested.
        """
        results = {
            "available": [],
            "pulled": [],
            "registered": [],
            "failed": []
        }

        if not model_names:
            return results

        ollama = _get_ollama()
        if not ollama:
            print("[MODEL_DISCOVERY] Ollama not installed; cannot ensure models")
            results["failed"].extend(model_names)
            return results

        existing = {m.get("name") for m in self.scan_ollama_models() if m.get("name")}

        for model_name in model_names:
            try:
                if model_name in existing:
                    results["available"].append(model_name)
                elif pull_missing:
                    print(f"[MODEL_DISCOVERY] Pulling Ollama model: {model_name}")
                    ollama.pull(model_name)
                    results["pulled"].append(model_name)
                else:
                    results["failed"].append(model_name)
                    continue

                if auto_register:
                    self.auto_register_model(model_name, skip_benchmark=not auto_benchmark)
                    results["registered"].append(model_name)

            except Exception as e:
                print(f"[MODEL_DISCOVERY] Failed to ensure {model_name}: {e}")
                results["failed"].append(model_name)

        return results
    
    def assess_capabilities(self, model_name: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Benchmark model to determine its capabilities
        Tests: coding ability, reasoning, speed
        """
        print(f"[BENCHMARK] Assessing {model_name}...")
        
        capabilities = {
            "model_name": model_name,
            "coding_score": 0.0,
            "reasoning_score": 0.0,
            "documentation_score": 0.0,
            "speed_tokens_sec": 0.0,
            "vram_gb": self._estimate_vram(model_name),
            "assessed_at": datetime.now().isoformat()
        }
        
        try:
            # Test coding ability
            capabilities["coding_score"] = self._test_coding(model_name, timeout)
            print(f"  Coding: {capabilities['coding_score']:.1f}/100")
            
            # Test reasoning
            capabilities["reasoning_score"] = self._test_reasoning(model_name, timeout)
            print(f"  Reasoning: {capabilities['reasoning_score']:.1f}/100")
            
            # Test documentation/text generation
            capabilities["documentation_score"] = self._test_documentation(model_name, timeout)
            print(f"  Documentation: {capabilities['documentation_score']:.1f}/100")
            
            # Measure speed
            capabilities["speed_tokens_sec"] = self._test_speed(model_name, timeout)
            print(f"  Speed: {capabilities['speed_tokens_sec']:.1f} tokens/sec")
            
        except Exception as e:
            print(f"[BENCHMARK] Error assessing {model_name}: {e}")
        
        return capabilities
    
    def _test_coding(self, model_name: str, timeout: int = 30) -> float:
        """Test code generation ability (0-100)"""
        ollama = _get_ollama()
        if not ollama:
            return 0.0
        
        test_prompt = "Write a Python function called 'is_prime' that checks if a number is prime. Include docstring."
        
        try:
            start = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                options={"num_predict": 500}
            )
            
            if time.time() - start > timeout:
                return 0.0
            
            result = response.get('message', {}).get('content', '')
            
            # Score based on code quality indicators
            score = 0.0
            
            # Check for Python code block
            if "```python" in result or "```py" in result:
                score += 15
            elif "def " in result:
                score += 10
            
            # Check for function definition
            if "def is_prime" in result or "def isPrime" in result:
                score += 20
            elif "def " in result:
                score += 10
            
            # Check for return statement
            if "return True" in result or "return False" in result:
                score += 15
            elif "return" in result:
                score += 10
            
            # Check for docstring
            if '"""' in result or "'''" in result:
                score += 10
            
            # Check for proper logic
            if "% 2" in result or "// 2" in result or "sqrt" in result:
                score += 15
            
            # Check for loop
            if "for " in result or "while " in result:
                score += 10
            
            # Check for edge case handling
            if "< 2" in result or "<= 1" in result or "== 1" in result:
                score += 15
            
            return min(score, 100.0)
            
        except Exception as e:
            print(f"[CODING_TEST] Error: {e}")
            return 0.0
    
    def _test_reasoning(self, model_name: str, timeout: int = 30) -> float:
        """Test reasoning ability (0-100)"""
        ollama = _get_ollama()
        if not ollama:
            return 0.0
        
        test_prompt = """Solve this step by step:
A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?
Explain your reasoning."""
        
        try:
            start = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                options={"num_predict": 300}
            )
            
            if time.time() - start > timeout:
                return 0.0
            
            result = response.get('message', {}).get('content', '').lower()
            
            score = 0.0
            
            # Check if correct answer is present
            if "9" in result:
                score += 40
            
            # Check for explanation
            if "all but" in result or "except" in result:
                score += 20
            
            # Check for step by step reasoning
            if "step" in result or "first" in result or "1." in result:
                score += 20
            
            # Check for clear conclusion
            if "therefore" in result or "so" in result or "answer" in result:
                score += 20
            
            return min(score, 100.0)
            
        except Exception as e:
            print(f"[REASONING_TEST] Error: {e}")
            return 0.0
    
    def _test_documentation(self, model_name: str, timeout: int = 30) -> float:
        """Test documentation/text generation ability (0-100)"""
        ollama = _get_ollama()
        if not ollama:
            return 0.0
        
        test_prompt = "Write a brief README section explaining how to install and use a Python package called 'datautils'."
        
        try:
            start = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                options={"num_predict": 400}
            )
            
            if time.time() - start > timeout:
                return 0.0
            
            result = response.get('message', {}).get('content', '')
            
            score = 0.0
            
            # Check for markdown headers
            if "#" in result:
                score += 15
            
            # Check for installation instructions
            if "pip install" in result or "install" in result.lower():
                score += 20
            
            # Check for code examples
            if "```" in result or "import " in result:
                score += 20
            
            # Check for usage section
            if "usage" in result.lower() or "example" in result.lower():
                score += 15
            
            # Check for proper formatting
            if "- " in result or "* " in result or "1." in result:
                score += 15
            
            # Check for reasonable length
            if len(result) > 200:
                score += 15
            
            return min(score, 100.0)
            
        except Exception as e:
            print(f"[DOC_TEST] Error: {e}")
            return 0.0
    
    def _test_speed(self, model_name: str, timeout: int = 30) -> float:
        """Measure tokens per second"""
        ollama = _get_ollama()
        if not ollama:
            return 0.0
        
        test_prompt = "Count from 1 to 50, one number per line."
        
        try:
            start = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                options={"num_predict": 200}
            )
            elapsed = time.time() - start
            
            if elapsed > timeout or elapsed == 0:
                return 0.0
            
            result = response.get('message', {}).get('content', '')
            tokens = len(result.split())
            
            return tokens / elapsed
            
        except Exception as e:
            print(f"[SPEED_TEST] Error: {e}")
            return 0.0
    
    def _estimate_vram(self, model_name: str) -> float:
        """Estimate VRAM requirement based on model name"""
        name_lower = model_name.lower()
        
        # Extract size from common patterns
        size_patterns = [
            ("70b", 40.0), ("65b", 38.0), ("34b", 20.0), ("33b", 20.0),
            ("32b", 19.0), ("27b", 16.0), ("22b", 13.0), ("14b", 8.0),
            ("13b", 8.0), ("8b", 5.0), ("7b", 4.5), ("6b", 4.0),
            ("4b", 2.5), ("3b", 2.0), ("2b", 1.5), ("1.7b", 1.4),
            ("1.5b", 1.2), ("1b", 0.8), ("0.5b", 0.5), ("500m", 0.5)
        ]
        
        for pattern, vram in size_patterns:
            if pattern in name_lower:
                return vram
        
        # Default estimate
        return 4.0
    
    def register_model(self, capabilities: Dict[str, Any]) -> bool:
        """Register model with assessed capabilities in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO model_capabilities 
                (model_name, coding_score, reasoning_score, documentation_score,
                 speed_tokens_sec, vram_gb, assessed_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                capabilities['model_name'],
                capabilities.get('coding_score', 0),
                capabilities.get('reasoning_score', 0),
                capabilities.get('documentation_score', 0),
                capabilities.get('speed_tokens_sec', 0),
                capabilities.get('vram_gb', 0),
                capabilities.get('assessed_at', datetime.now().isoformat())
            ))
            
            conn.commit()
            conn.close()
            print(f"[MODEL_DISCOVERY] Registered: {capabilities['model_name']}")
            return True
            
        except Exception as e:
            conn.close()
            print(f"[MODEL_DISCOVERY] Failed to register: {e}")
            return False
    
    def auto_register_model(self, model_name: str, skip_benchmark: bool = False) -> Dict[str, Any]:
        """Auto-register with capability assessment"""
        if skip_benchmark:
            capabilities = {
                "model_name": model_name,
                "coding_score": 50.0,  # Default
                "reasoning_score": 50.0,
                "documentation_score": 50.0,
                "speed_tokens_sec": 30.0,
                "vram_gb": self._estimate_vram(model_name),
                "assessed_at": datetime.now().isoformat()
            }
        else:
            capabilities = self.assess_capabilities(model_name)
        
        self.register_model(capabilities)
        return capabilities
    
    def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get capabilities for a specific model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM model_capabilities WHERE model_name = ?", (model_name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "model_name": row[0],
                "coding_score": row[1],
                "reasoning_score": row[2],
                "documentation_score": row[3],
                "speed_tokens_sec": row[4],
                "vram_gb": row[5],
                "assessed_at": row[8],
                "is_active": bool(row[9])
            }
        return None
    
    def get_all_capabilities(self) -> List[Dict[str, Any]]:
        """Get capabilities for all registered models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM model_capabilities WHERE is_active = 1")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "model_name": row[0],
                "coding_score": row[1],
                "reasoning_score": row[2],
                "documentation_score": row[3],
                "speed_tokens_sec": row[4],
                "vram_gb": row[5],
                "assessed_at": row[8] if len(row) > 8 else None,
                "is_active": bool(row[9]) if len(row) > 9 else True
            }
            for row in rows
        ]
    
    def deactivate_model(self, model_name: str) -> bool:
        """Mark a model as inactive (removed from Ollama)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE model_capabilities SET is_active = 0 WHERE model_name = ?",
            (model_name,)
        )
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        
        return affected > 0


if __name__ == "__main__":
    # Test model discovery
    discovery = ModelDiscovery()
    
    print("\n" + "="*60)
    print("MODEL DISCOVERY TEST")
    print("="*60 + "\n")
    
    # Scan for models
    print("Scanning Ollama for models...")
    models = discovery.scan_ollama_models()
    print(f"Found {len(models)} models in Ollama\n")
    
    for m in models[:5]:  # Show first 5
        print(f"  - {m['name']}")
    
    # Check for new models
    print("\nChecking for unregistered models...")
    new_models = discovery.discover_new_models()
    print(f"Found {len(new_models)} new models\n")
    
    # Benchmark first new model if any
    if new_models:
        first_new = new_models[0]['name']
        print(f"Benchmarking: {first_new}")
        caps = discovery.assess_capabilities(first_new)
        print(f"\nCapabilities: {caps}")
