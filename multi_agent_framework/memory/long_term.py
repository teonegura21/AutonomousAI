"""Long-term memory wrapper (vector database)."""

from __future__ import annotations

from ai_autonom.memory.vector_store import VectorMemoryStore


class LongTermMemory:
    def __init__(self, persist_directory: str = "./.runtime/data/chromadb") -> None:
        self._store = VectorMemoryStore(persist_directory=persist_directory)

    def add(self, collection: str, text: str, metadata=None) -> None:
        self._store.store(collection_name=collection, text=text, metadata=metadata)

    def query(self, collection: str, text: str, k: int = 5):
        return self._store.query(collection_name=collection, query=text, n_results=k)

    def stats(self):
        return self._store.stats()
