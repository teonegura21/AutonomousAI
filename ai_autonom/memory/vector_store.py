#!/usr/bin/env python3
"""
Vector Memory Store
ChromaDB integration with semantic search for storing and querying agent work
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[WARNING] ChromaDB not installed. Vector search disabled. Run: pip install chromadb")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from ai_autonom.core.config import get_config

PREFERRED_EMBEDDING_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "bge-small-en",
    "snowflake-arctic-embed",
    "all-minilm",
    "e5-small"
]


class VectorMemoryStore:
    """
    ChromaDB integration with semantic search.
    Stores ALL work: code, decisions, conversations.
    Agents can query: "What coordinate system did we choose?"
    """
    _embedding_pull_attempted = set()
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        auto_pull_embeddings: Optional[bool] = None
    ):
        cfg = get_config()
        if persist_dir is None:
            persist_dir = cfg.get("memory.vector_db.persist_directory", ".runtime/data/chromadb")
        if embedding_model is None:
            embedding_model = cfg.get("memory.vector_db.embedding_model", "nomic-embed-text")
        if auto_pull_embeddings is None:
            auto_pull_embeddings = cfg.get("memory.vector_db.auto_pull_embeddings", False)

        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.auto_pull_embeddings = auto_pull_embeddings
        self.client = None
        self.code_collection = None
        self.decision_collection = None
        self.conversation_collection = None
        self._force_default_embeddings = set()
        
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        self._ensure_embedding_model()

    def _list_ollama_models(self) -> List[str]:
        """List available Ollama models."""
        if not OLLAMA_AVAILABLE:
            return []
        try:
            response = ollama.list()
            if isinstance(response, dict):
                models = response.get("models", [])
            else:
                models = getattr(response, "models", [])
            names = []
            for m in models:
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
                if not name:
                    name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
                if name:
                    names.append(name)
            return names
        except Exception:
            return []

    def _select_embedding_model(self, available: List[str]) -> Optional[str]:
        """Select the best available embedding model."""
        available_set = {m for m in available if m}
        for name in PREFERRED_EMBEDDING_MODELS:
            if name in available_set:
                return name
            for candidate in available_set:
                if candidate.startswith(f"{name}:"):
                    return candidate
        return sorted(available_set)[0] if available_set else None

    def _pull_embedding_model(self, model_name: str) -> bool:
        if not model_name or not OLLAMA_AVAILABLE:
            return False
        attempted = type(self)._embedding_pull_attempted
        if model_name in attempted:
            return False
        attempted.add(model_name)
        try:
            print(f"[VECTOR_STORE] Pulling embedding model: {model_name}")
            ollama.pull(model_name)
            return True
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to pull embedding model {model_name}: {e}")
            return False

    def _ensure_embedding_model(self, force_refresh: bool = False) -> None:
        """Ensure the configured embedding model is available or select a fallback."""
        if not OLLAMA_AVAILABLE:
            return

        available = self._list_ollama_models()
        available_set = {m for m in available if m}

        requested = self.embedding_model
        if not requested or requested == "auto" or force_refresh:
            selected = self._select_embedding_model(available)
            if selected:
                self.embedding_model = selected
                return
            if self.auto_pull_embeddings:
                default_name = PREFERRED_EMBEDDING_MODELS[0]
                if self._pull_embedding_model(default_name):
                    self.embedding_model = default_name
            return

        if requested not in available_set:
            for candidate in available_set:
                if candidate.startswith(f"{requested}:"):
                    self.embedding_model = candidate
                    return
            if self.auto_pull_embeddings and self._pull_embedding_model(requested):
                self.embedding_model = requested
                return

            fallback = self._select_embedding_model(available)
            if fallback:
                self.embedding_model = fallback
    
    def _init_chromadb(self):
        """Initialize ChromaDB with persistent storage"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create collections
            self.code_collection = self.client.get_or_create_collection(
                name="code_artifacts",
                metadata={"description": "Code written by agents"}
            )
            
            self.decision_collection = self.client.get_or_create_collection(
                name="decisions",
                metadata={"description": "Design decisions and rationale"}
            )
            
            self.conversation_collection = self.client.get_or_create_collection(
                name="conversations",
                metadata={"description": "Agent conversations and outputs"}
            )
            
            print(f"[VECTOR_STORE] Initialized with {self.persist_dir}")
            
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to initialize: {e}")
            self.client = None

    def _add_with_optional_embedding(
        self,
        collection,
        doc_id: str,
        content: str,
        meta: Dict[str, Any]
    ) -> bool:
        if not collection:
            return False

        use_custom = collection.name not in self._force_default_embeddings
        embedding = self._get_embedding(content) if use_custom else None

        try:
            if embedding:
                collection.add(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[meta],
                    embeddings=[embedding]
                )
            else:
                collection.add(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[meta]
                )
            return True
        except Exception as e:
            error_text = str(e).lower()
            if "dimension" in error_text and collection.name not in self._force_default_embeddings:
                self._force_default_embeddings.add(collection.name)
                try:
                    collection.add(
                        ids=[doc_id],
                        documents=[content],
                        metadatas=[meta]
                    )
                    return True
                except Exception as retry_error:
                    print(f"[VECTOR_STORE] Failed to store content after retry: {retry_error}")
                    return False
            print(f"[VECTOR_STORE] Failed to store content: {e}")
            return False
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama"""
        if not OLLAMA_AVAILABLE:
            return None
        
        if not self.embedding_model or self.embedding_model == "auto":
            self._ensure_embedding_model()

        for _ in range(2):
            try:
                if not self.embedding_model:
                    return None
                response = ollama.embeddings(model=self.embedding_model, prompt=text[:8000])
                if isinstance(response, dict):
                    return response.get("embedding")
                return getattr(response, "embedding", None)
            except Exception as e:
                error_text = str(e).lower()
                if "not found" in error_text or "model" in error_text:
                    self._ensure_embedding_model(force_refresh=True)
                    continue
                print(f"[VECTOR_STORE] Embedding error: {e}")
                return None
        return None
    
    def store_code(
        self,
        task_id: str,
        code: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store code artifact with embeddings"""
        if not self.code_collection:
            return False
        
        try:
            doc_id = f"code_{task_id}_{datetime.now().timestamp()}"
            
            meta = {
                "task_id": task_id,
                "type": "code",
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            return self._add_with_optional_embedding(
                self.code_collection,
                doc_id,
                code,
                meta
            )
            
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to store code: {e}")
            return False
    
    def store_decision(
        self,
        task_id: str,
        decision: str,
        rationale: str
    ) -> bool:
        """Store design decision"""
        if not self.decision_collection:
            return False
        
        try:
            content = f"Decision: {decision}\nRationale: {rationale}"
            doc_id = f"decision_{task_id}_{datetime.now().timestamp()}"
            
            meta = {
                "task_id": task_id,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            }
            return self._add_with_optional_embedding(
                self.decision_collection,
                doc_id,
                content,
                meta
            )
            
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to store decision: {e}")
            return False
    
    def store_conversation(
        self,
        task_id: str,
        agent_id: str,
        content: str,
        role: str = "assistant"
    ) -> bool:
        """Store agent conversation/output"""
        if not self.conversation_collection:
            return False
        try:
            doc_id = f"conv_{task_id}_{agent_id}_{datetime.now().timestamp()}"
            
            meta = {
                "task_id": task_id,
                "agent_id": agent_id,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }
            return self._add_with_optional_embedding(
                self.conversation_collection,
                doc_id,
                content,
                meta
            )
            
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to store conversation: {e}")
            return False

    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: str = "conversations"
    ) -> bool:
        """Store arbitrary content in a named collection."""
        if not self.client:
            return False

        try:
            coll = self.client.get_or_create_collection(
                name=collection,
                metadata={"description": f"dynamic collection: {collection}"}
            )
            doc_id = f"{collection}_{datetime.now().timestamp()}"
            meta = {
                "collection": collection,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            return self._add_with_optional_embedding(coll, doc_id, content, meta)
        except Exception as e:
            print(f"[VECTOR_STORE] Failed to store content: {e}")
            return False

    def query(
        self,
        query: str,
        collection: str = "all",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Query a collection (or all collections) by meaning."""
        return self.semantic_search(query, collection=collection, n_results=n_results)
    
    def semantic_search(
        self,
        query: str,
        collection: str = "all",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search by meaning across collections
        
        Args:
            query: Natural language query
            collection: "code", "decisions", "conversations", or "all"
            n_results: Number of results to return
        
        Returns:
            List of matching documents with metadata and distance
        """
        if not self.client:
            return []
        
        results = []
        query_embedding = self._get_embedding(query)
        
        collections_map = {
            "code": self.code_collection,
            "decisions": self.decision_collection,
            "conversations": self.conversation_collection
        }
        
        if collection == "all":
            search_collections = [c for c in collections_map.values() if c]
        else:
            coll = collections_map.get(collection)
            search_collections = [coll] if coll else []
        
        for coll in search_collections:
            try:
                use_custom = coll.name not in self._force_default_embeddings
                if query_embedding and use_custom:
                    res = coll.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results
                    )
                else:
                    res = coll.query(
                        query_texts=[query],
                        n_results=n_results
                    )
                
                if res and res.get("documents"):
                    for i, doc in enumerate(res["documents"][0]):
                        results.append({
                            "content": doc,
                            "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
                            "distance": res["distances"][0][i] if res.get("distances") else None,
                            "collection": coll.name
                        })
                        
            except Exception as e:
                error_text = str(e).lower()
                if "dimension" in error_text and coll.name not in self._force_default_embeddings:
                    self._force_default_embeddings.add(coll.name)
                    try:
                        res = coll.query(query_texts=[query], n_results=n_results)
                        if res and res.get("documents"):
                            for i, doc in enumerate(res["documents"][0]):
                                results.append({
                                    "content": doc,
                                    "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
                                    "distance": res["distances"][0][i] if res.get("distances") else None,
                                    "collection": coll.name
                                })
                        continue
                    except Exception as retry_error:
                        print(f"[VECTOR_STORE] Search retry failed in {coll.name}: {retry_error}")
                        continue
                print(f"[VECTOR_STORE] Search error in {coll.name}: {e}")
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x.get("distance", float('inf')))
        return results[:n_results]
    
    def get_task_context(self, task_id: str) -> Dict[str, Any]:
        """Get all artifacts from a specific task"""
        if not self.client:
            return {"code": [], "decisions": [], "conversations": []}
        
        result = {
            "code": [],
            "decisions": [],
            "conversations": []
        }
        
        collections_map = {
            "code": self.code_collection,
            "decisions": self.decision_collection,
            "conversations": self.conversation_collection
        }
        
        for key, coll in collections_map.items():
            if coll:
                try:
                    res = coll.get(
                        where={"task_id": task_id}
                    )
                    if res and res.get("documents"):
                        result[key] = [
                            {
                                "content": doc,
                                "metadata": res["metadatas"][i] if res.get("metadatas") else {}
                            }
                            for i, doc in enumerate(res["documents"])
                        ]
                except Exception as e:
                    print(f"[VECTOR_STORE] Error getting {key}: {e}")
        
        return result
    
    def query_natural(self, question: str) -> str:
        """
        Answer natural language questions about stored knowledge
        
        Example: "What coordinate system did we decide to use?"
        """
        results = self.semantic_search(question, n_results=3)
        
        if not results:
            return "No relevant information found in memory."
        
        context_parts = []
        for r in results:
            collection = r.get("collection", "unknown")
            content = r.get("content", "")[:1000]  # Limit content
            distance = r.get("distance")
            
            relevance = "High" if distance and distance < 0.3 else "Medium" if distance and distance < 0.6 else "Low"
            
            context_parts.append(f"[{collection}] (Relevance: {relevance})\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        return f"Based on stored knowledge:\n\n{context}"
    
    def get_similar_code(self, code_snippet: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Find similar code in the store"""
        return self.semantic_search(code_snippet, collection="code", n_results=n_results)
    
    def get_relevant_decisions(self, topic: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Get decisions related to a topic"""
        return self.semantic_search(topic, collection="decisions", n_results=n_results)
    
    def clear_collection(self, collection: str) -> bool:
        """Clear a specific collection"""
        collections_map = {
            "code": self.code_collection,
            "decisions": self.decision_collection,
            "conversations": self.conversation_collection
        }
        
        coll = collections_map.get(collection)
        if coll:
            try:
                self.client.delete_collection(coll.name)
                # Recreate empty
                if collection == "code":
                    self.code_collection = self.client.create_collection("code_artifacts")
                elif collection == "decisions":
                    self.decision_collection = self.client.create_collection("decisions")
                elif collection == "conversations":
                    self.conversation_collection = self.client.create_collection("conversations")
                return True
            except Exception as e:
                print(f"[VECTOR_STORE] Failed to clear {collection}: {e}")
        return False
    
    def clear_all(self) -> bool:
        """Clear all collections"""
        success = True
        for coll in ["code", "decisions", "conversations"]:
            if not self.clear_collection(coll):
                success = False
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "available": self.client is not None,
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_model,
            "auto_pull_embeddings": self.auto_pull_embeddings,
            "default_embedding_collections": sorted(self._force_default_embeddings)
        }
        
        if self.client:
            for name, coll in [
                ("code", self.code_collection),
                ("decisions", self.decision_collection),
                ("conversations", self.conversation_collection)
            ]:
                if coll:
                    try:
                        stats[f"{name}_count"] = coll.count()
                    except:
                        stats[f"{name}_count"] = 0
        
        return stats


if __name__ == "__main__":
    # Test vector store
    store = VectorMemoryStore()
    
    print("\n" + "="*60)
    print("VECTOR MEMORY STORE TEST")
    print("="*60 + "\n")
    
    print(f"Stats: {store.get_stats()}\n")
    
    # Store some test data
    store.store_code(
        "task_1",
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        {"language": "python", "type": "function"}
    )
    
    store.store_decision(
        "task_1",
        "Use recursion for factorial",
        "Cleaner code, acceptable performance for small n"
    )
    
    store.store_conversation(
        "task_1",
        "coder_qwen",
        "I'll implement the factorial function using recursion for clarity."
    )
    
    print("Stored test data\n")
    
    # Search
    print("Searching for 'factorial implementation':")
    results = store.semantic_search("factorial implementation", n_results=2)
    for r in results:
        print(f"  - [{r['collection']}] {r['content'][:100]}...")
    
    # Natural language query
    print("\nQuerying: 'Why did we use recursion?'")
    answer = store.query_natural("Why did we use recursion?")
    print(answer[:500])
    
    print(f"\n\nFinal stats: {store.get_stats()}")
