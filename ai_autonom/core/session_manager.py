
"""
Session Manager
Handles the "IDE-like" persistence of conversations.
Creates structured folders for every mission:
conversations/
  └── YYYYMMDD_HHMM_mission_slug/
      ├── src/       # Source code
      ├── bin/       # Compiled executables
      ├── docs/      # Auto-generated documentation
      └── memory/    # Session-specific knowledge base
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class SessionManager:
    def __init__(self, base_dir: str = "conversations"):
        self.base_dir = base_dir
        self.current_session_id = None
        self.current_session_dir = None
        
    def create_session(self, goal: str) -> str:
        """Create a new session folder based on the goal"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a slug from the goal (first 30 chars, safe chars only)
        slug = re.sub(r'[^a-zA-Z0-9]', '_', goal[:30]).strip('_')
        
        self.current_session_id = f"{timestamp}_{slug}"
        self.current_session_dir = os.path.join(self.base_dir, self.current_session_id)
        
        # Create structure
        os.makedirs(os.path.join(self.current_session_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.current_session_dir, "bin"), exist_ok=True)
        os.makedirs(os.path.join(self.current_session_dir, "docs"), exist_ok=True)
        os.makedirs(os.path.join(self.current_session_dir, "memory"), exist_ok=True)
        
        # Save meta
        self.save_metadata(goal)
        
        print(f"[SESSION] Created workspace: {self.current_session_dir}")
        return self.current_session_dir

    def save_metadata(self, goal: str):
        meta = {
            "id": self.current_session_id,
            "goal": goal,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        with open(os.path.join(self.current_session_dir, "session_info.json"), 'w') as f:
            json.dump(meta, f, indent=2)

    def get_path(self, category: str, filename: str) -> str:
        """Get path for a file in the current session (src, bin, docs)"""
        if not self.current_session_dir:
            raise RuntimeError("No active session")
            
        if category not in ["src", "bin", "docs", "memory"]:
            category = "src" # Default
            
        return os.path.join(self.current_session_dir, category, filename)

    def get_relative_path(self, category: str, filename: str) -> str:
        """Get path relative to the project root (for tools)"""
        abs_path = self.get_path(category, filename)
        return os.path.relpath(abs_path, os.getcwd())
