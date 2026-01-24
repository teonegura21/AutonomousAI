
"""
Debug Logger
Dumps raw LLM interactions and Tool outputs to a file for deep inspection.
"""

import os
import json
from typing import Optional, Any
from datetime import datetime

class DebugLogger:
    _instance = None
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "debug_interactions.log")
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def log_interaction(self, source: str, content: str, type: str = "info"):
        timestamp = datetime.now().isoformat()
        
        separator = "=" * 80
        entry = f"\n{separator}\n[{timestamp}] [{source.upper()}] [{type.upper()}]\n{separator}\n{content}\n"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)

def log_debug(source: str, content: Any, type: str = "info"):
    """Helper to log strings or objects"""
    if not isinstance(content, str):
        try:
            content = json.dumps(content, indent=2)
        except:
            content = str(content)
            
    DebugLogger.get_instance().log_interaction(source, content, type)
