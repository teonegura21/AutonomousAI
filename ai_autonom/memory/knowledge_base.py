
"""
Operation Knowledge Base (The "Blackboard")
Acts as the central nervous system for the operation, storing structured data
that persists across all agents and tasks.
"""

import json
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

@dataclass
class Finding:
    id: str
    type: str  # vulnerability, credential, asset, artifact
    title: str
    details: str
    severity: str = "info"  # critical, high, medium, low, info
    source_agent: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class KnowledgeBase:
    _instance = None
    _lock = threading.RLock()

    def __init__(self, db_path: str = ".runtime/data/knowledge_base.json"):
        self.db_path = db_path
        self.assets: List[Dict] = []
        self.credentials: List[Dict] = []
        self.findings: List[Finding] = []
        self.artifacts: List[Dict] = []
        self.metadata: Dict[str, Any] = {"mission_start": datetime.now().isoformat()}
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_asset(self, ip: str, hostname: str = "", tags: List[str] = None):
        with self._lock:
            # Check deduplication
            for asset in self.assets:
                if asset['ip'] == ip:
                    if hostname: asset['hostname'] = hostname
                    if tags: asset['tags'].extend([t for t in tags if t not in asset['tags']])
                    return
            
            self.assets.append({
                "ip": ip,
                "hostname": hostname,
                "tags": tags or [],
                "discovered_at": datetime.now().isoformat()
            })
            self._save()

    def add_credential(self, username: str, password: str = "", hash_val: str = "", service: str = ""):
        with self._lock:
            self.credentials.append({
                "username": username,
                "password": password,
                "hash": hash_val,
                "service": service,
                "captured_at": datetime.now().isoformat()
            })
            self._save()

    def add_finding(self, title: str, type: str, details: str, severity: str, agent: str):
        with self._lock:
            import uuid
            finding = Finding(
                id=str(uuid.uuid4())[:8],
                title=title,
                type=type,
                details=details,
                severity=severity,
                source_agent=agent
            )
            self.findings.append(finding)
            self._save()

    def get_summary(self) -> str:
        """Returns a string representation of the KB for LLM context"""
        with self._lock:
            summary = "=== OPERATION KNOWLEDGE BASE ===\n"
            
            if self.assets:
                summary += "\n[ASSETS]\n"
                for a in self.assets:
                    summary += f"- {a['ip']} ({a['hostname']}) Tags: {', '.join(a['tags'])}\n"
            
            if self.credentials:
                summary += "\n[CREDENTIALS]\n"
                for c in self.credentials:
                    pwd = c['password'] if c['password'] else f"Hash: {c['hash'][:10]}..."
                    summary += f"- {c['username']} @ {c['service']} : {pwd}\n"
            
            if self.findings:
                summary += "\n[CRITICAL FINDINGS]\n"
                # Sort by severity
                severity_map = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
                sorted_findings = sorted(self.findings, key=lambda x: severity_map.get(x.severity.lower(), 5))
                
                for f in sorted_findings:
                    if severity_map.get(f.severity.lower(), 5) <= 2: # Show Medium+
                        summary += f"[{f.severity.upper()}] {f.title}\n"
            
            return summary

    def _save(self):
        try:
            data = {
                "assets": self.assets,
                "credentials": self.credentials,
                "findings": [asdict(f) for f in self.findings],
                "artifacts": self.artifacts,
                "metadata": self.metadata
            }
            import os
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[KB] Save failed: {e}")

    def load(self):
        import os
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.assets = data.get("assets", [])
                    self.credentials = data.get("credentials", [])
                    self.artifacts = data.get("artifacts", [])
                    self.metadata = data.get("metadata", {})
                    self.findings = [Finding(**f) for f in data.get("findings", [])]
            except Exception as e:
                print(f"[KB] Load failed: {e}")
